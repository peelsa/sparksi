# SPDX-License-Identifier: Apache-2.0

"""sparksi CLI — observe, lock, wanted. No load/swap/evict/install.

Also installed as ``dgx-mem`` (deprecated alias, one release)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from sparksi import protect
from sparksi import __version__
from sparksi.catalog import CatalogError, load_catalog
from sparksi import compat
from sparksi.config import CFG, CFG_ERRORS, check
from sparksi.lock import Lock, LockedError
from sparksi.apply import (
    build_plan,
    commit as apply_commit,
    format_plan,
    save_plan,
    union_spec,
    wanted_for_plan,
)
from sparksi.never import MUTATOR_VERBS, Refused, refuse_mutator
from sparksi.probe import snapshot
from sparksi.status import (
    assemble_inventory,
    assemble_status,
    format_dry_run,
    format_inventory,
    format_status,
    format_wanted,
    run_dry_run,
    _load_model_inventory,
)
from sparksi.tokens import format_ledger, report as token_report
from sparksi.wanted import apply_set, load_wanted, seat_view
from sparksi.cli_api import cmd_plan, cmd_runs, cmd_start, cmd_stop


def _dump(data: Any, as_json: bool, text: str) -> int:
    if as_json:
        print(json.dumps(data, indent=2, default=str))
    else:
        print(text)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cat = load_catalog()
    truth, err = _load_model_inventory()
    data = assemble_status(cat, truth, Lock(), truth_err=err)
    return _dump(data, args.json, format_status(data))


def cmd_inventory(args: argparse.Namespace) -> int:
    cat = load_catalog()
    truth, err = _load_model_inventory()
    data = assemble_inventory(cat, truth, truth_err=err)
    return _dump(data, args.json, format_inventory(data))


def cmd_dry_run(args: argparse.Namespace) -> int:
    cat = load_catalog()
    truth, err = _load_model_inventory()
    try:
        data = run_dry_run(cat, args.set, truth, truth_err=err)
    except CatalogError as e:
        print(str(e), file=sys.stderr)
        return 2
    rc = 0
    if data.get("verdict") == "NO":
        rc = 1
    elif data.get("verdict") == "UNCERTAIN":
        rc = 3
    _dump(data, args.json, format_dry_run(data))
    return rc


def cmd_lock(args: argparse.Namespace) -> int:
    st = Lock().acquire(reason=args.reason or "cli lock")
    data = st.to_dict()
    return _dump(data, args.json, f"lock: {data['on']}  holder={data['holder']}  {data['reason']}")


def cmd_unlock(args: argparse.Namespace) -> int:
    st, token = Lock().release(ttl_s=int(args.ttl))
    data = st.to_dict()
    data["unlock_token"] = token
    text = (
        f"lock: {data['on']}  holder={data['holder']}  expires={data['expires_at']}  "
        f"{data['reason']}\n  unlock token (API POST): {token}"
    )
    return _dump(data, args.json, text)


def cmd_tokens(args: argparse.Namespace) -> int:
    cat = load_catalog()
    data = token_report(cat, sample=False)
    return _dump(data, args.json, format_ledger(data))


def cmd_wanted(args: argparse.Namespace) -> int:
    cat = load_catalog()
    truth, err = _load_model_inventory()
    lock = Lock()
    snap = assemble_status(cat, truth, lock, truth_err=err)
    wanted = load_wanted() or {}
    data = {
        "wanted": wanted,
        "lock": snap["lock"],
        "seats": snap.get("seats") or seat_view(cat, snap["residents"], wanted),
        "residents": [r for r in snap["residents"] if r.get("resident")],
        "conflicts": snap.get("conflicts") or [],
    }
    return _dump(data, args.json, format_wanted(data))


def cmd_wanted_set(args: argparse.Namespace) -> int:
    cat = load_catalog()
    lock = Lock()
    try:
        lock.require_unlocked("wanted set")
    except LockedError as e:
        print(str(e), file=sys.stderr)
        return 2
    truth, err = _load_model_inventory()
    try:
        dry = run_dry_run(cat, args.spec, truth, truth_err=err)
    except CatalogError:
        # comma-separated ids: dry-run the first recipe-like token, or video/trio if mixed
        dry = {
            "set": args.spec,
            "verdict": "UNCERTAIN",
            "needed_gib": None,
            "reasons": [f"dry-run has no recipe for spec {args.spec!r}; wanted stored anyway"],
            "comparable": True,
        }
        # If spec resolves to a recipe via wanted.resolve later, re-run with name
        from sparksi.wanted import resolve_spec

        try:
            resolved = resolve_spec(cat, args.spec)
            try:
                dry = run_dry_run(cat, resolved["name"], truth, truth_err=err)
            except CatalogError:
                pass
        except CatalogError as e:
            print(str(e), file=sys.stderr)
            return 2
    payload = apply_set(cat, args.spec, dry)
    snap = assemble_status(cat, truth, lock, truth_err=err)
    data = {"wanted": payload, "dry_run": dry, "note": "desired state only; no load/evict"}
    text = format_wanted(
        {
            "wanted": payload,
            "lock": lock.read().to_dict(),
            "seats": seat_view(cat, snap["residents"], payload),
            "conflicts": dry.get("conflicts") or snap.get("conflicts") or [],
        }
    )
    text += f"\n  stored dry-run verdict: {dry.get('verdict')}"
    rc = 0
    if dry.get("verdict") == "NO":
        rc = 1
    elif dry.get("verdict") == "UNCERTAIN":
        rc = 3
    _dump(data, args.json, text)
    return rc


def _make_plan(catalog, spec: str | None, lock: Lock, truth, err):
    wanted = wanted_for_plan(catalog, spec)
    snap = snapshot(catalog, truth)
    union = union_spec(wanted, snap["residents"])
    try:
        dry = (
            run_dry_run(catalog, union, truth, truth_err=err)
            if union
            else {"verdict": "UNCERTAIN", "reasons": ["no wanted set"]}
        )
    except CatalogError as e:
        dry = {"verdict": "UNCERTAIN", "reasons": [str(e)]}
    plan = build_plan(
        catalog,
        wanted,
        snap["residents"],
        dry,
        lock_on=lock.read().locked,
    )
    save_plan(plan)
    return plan, wanted, snap


def cmd_apply(args: argparse.Namespace) -> int:
    cat = load_catalog()
    truth, err = _load_model_inventory()
    lock = Lock()
    spec = getattr(args, "set", None)
    plan, wanted, snap = _make_plan(cat, spec, lock, truth, err)
    if args.commit:
        # Re-probe immediately before mutate (the protected video service stays exempt).
        plan, wanted, snap = _make_plan(cat, spec, lock, truth, err)
        if any(
            r.get("resident") and (protect.is_video_id(r.get("id")) or protect.is_off_limits(r.get("port")))
            for r in snap["residents"]
        ):
            # Protected video service up: keep it in the exempt set by rebuilding; never stop it.
            pass
        plan = apply_commit(
            cat,
            plan,
            snap["residents"],
            lock,
            force_uncertain=bool(args.force_uncertain),
            wanted=wanted,
        )
    rc = 0
    v = plan.get("verdict")
    if v == "REFUSE":
        rc = 2
    elif v == "UNCERTAIN":
        rc = 3
    if args.commit and plan.get("apply_failed"):
        rc = 1
    _dump(plan, args.json, format_plan(plan))
    return rc


def cmd_serve(args: argparse.Namespace) -> int:
    from sparksi.never import assert_serve_port
    from sparksi.serve import parse_bind, serve

    state_dir = str(getattr(args, "state_dir", "") or "").strip()
    if state_dir:
        # This process only. A second instance passes its own directory so the
        # lock, job file, and recipe run registry stay off the live one's.
        os.environ["SPARKSI_STATE_DIR"] = state_dir
    cat = load_catalog()
    extra = list((cat.get("host") or {}).get("reserved_ports") or [])
    assert_serve_port(int(args.port), extra)
    serve(
        port=int(args.port),
        hosts=parse_bind(list(getattr(args, "bind", None) or [])),
        announce=not bool(getattr(args, "no_announce", False)),
    )
    return 0


def _wait_job(result: dict[str, Any]) -> int:
    if result.get("verdict") != "STARTED":
        print(json.dumps(result, indent=2, default=str))
        return 2
    import time

    from sparksi.jobs import current

    while True:
        job = current()
        if job and job.get("phase") in ("done", "error", "ready", "failed", "timed_out", "cancelled"):
            print(json.dumps(job, indent=2, default=str))
            return 0 if job.get("ok") else 1
        time.sleep(1)


def cmd_library(args: argparse.Namespace) -> int:
    from sparksi.library import fleet

    data = fleet(load_catalog())
    lines = []
    for row in data["library"]:
        on = row.get("on") or {}
        lines.append(
            f"{row.get('id')}  {row.get('gib')}GiB  {row.get('placement')}  "
            f"archive={row.get('archive_state')}  "
            + "  ".join(f"{CFG.id_for(n)}={on.get(n)}" for n in CFG.node_names)
            + f"  run={row.get('launcher')}"
        )
    return _dump(data, args.json, "\n".join(lines))


def cmd_stage(args: argparse.Namespace) -> int:
    from sparksi.jobs import start_stage

    return _wait_job(start_stage(load_catalog(), args.id, list(args.hosts), lock_on=False))


def cmd_drop(args: argparse.Namespace) -> int:
    from sparksi.jobs import start_drop

    return _wait_job(
        start_drop(load_catalog(), args.id, args.host, args.confirm, lock_on=False)
    )


def cmd_recipes_import(args: argparse.Namespace) -> int:
    """Read a live setup and print or save a pinned recipe. Never starts anything."""
    from sparksi.recipes.importer import import_all, import_model
    from sparksi.remote import get_runner

    catalog = load_catalog()
    runner = get_runner(catalog)
    if args.all:
        rows = import_all(catalog, runner, dry_run=bool(args.dry_run))
        if args.json:
            print(json.dumps(rows, indent=2, default=str))
        else:
            for row in rows:
                print(f"{row.get('id')}: {'ok' if row.get('ok') else 'refused'}")
                if row.get("port_note"):
                    print(f"  {row['port_note']}")
                for item in row.get("conflicts") or []:
                    print(f"  {item}")
                if row.get("recipe"):
                    print(json.dumps(row["recipe"], indent=2, default=str))
        return 0 if all(row.get("ok") or row.get("conflicts") for row in rows) else 1
    if not args.model_id:
        print("usage: sparksi recipes import <model-id> [--from SOURCE] [--dry-run]", file=sys.stderr)
        return 2
    result = import_model(catalog, runner, args.model_id, args.source or "", dry_run=bool(args.dry_run))
    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        if result.get("port_note"):
            print(result["port_note"])
        for item in result.get("conflicts") or []:
            print(item)
        if result.get("recipe"):
            print(json.dumps(result["recipe"], indent=2, default=str))
        if result.get("saved"):
            print(f"saved {args.model_id}")
    return 0 if result.get("ok") else 1


def cmd_recipes_bundle(args: argparse.Namespace) -> int:
    """Move an older pinned recipe into its own recipe folder. Reads the Sparks; never starts anything."""
    from sparksi.recipes.bundle import plan_bundle, write_bundle
    from sparksi.remote import get_runner

    catalog = load_catalog()
    bundle = plan_bundle(catalog, get_runner(catalog), args.model_id, args.recipe or "", keep=list(args.keep or []))
    view = bundle.view()
    if args.json:
        print(json.dumps(view, indent=2, default=str))
    else:
        print(f"{bundle.model_id} recipe {bundle.name}: {len(view['files'])} files")
        for item in view["files"]:
            print(f"  {item['path']}  {item['sha256'][:16]}  from {', '.join(item['from'])}")
        for item in view["kept"]:
            print(f"  kept as is: {item}")
        for item in view["notes"]:
            print(f"  note: {item}")
        for item in view["problems"]:
            print(item)
    if bundle.problems:
        return 1
    if args.dry_run:
        print("dry run: nothing was written")
        return 0
    ok, said = write_bundle(bundle)
    print(said)
    return 0 if ok else 1


def cmd_recipes_tidy(args: argparse.Namespace) -> int:
    """Move old recipe and config backups into backups/<date>/. Never deletes."""
    from sparksi.recipes.bundle import tidy_backups

    rows = tidy_backups(dry_run=bool(args.dry_run))
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            word = "moved" if row["done"] == "yes" else ("would move" if args.dry_run else "left")
            print(f"{word} {row['from']} -> {row['to']}" + (f" ({row['note']})" if row.get("note") else ""))
        if not rows:
            print("no backups to move")
    return 0


def cmd_run_model(args: argparse.Namespace) -> int:
    from sparksi.jobs import start_run

    lock = Lock()
    if lock.read().locked:
        print("lock is ON", file=sys.stderr)
        return 2
    return _wait_job(start_run(load_catalog(), args.id, lock_on=False))


def _unit_templates() -> Path:
    path = Path(__file__).resolve().parents[2] / "packaging" / "systemd"
    if not path.is_dir():
        raise FileNotFoundError(f"unit templates not found: {path}")
    return path


def _prefix(raw: str) -> str:
    text = (raw or "").strip().rstrip("/")
    if not text.startswith("/") or any(ch in text for ch in "\n\r"):
        raise ValueError("prefix must be an absolute path")
    return text


def _node_id(raw: str) -> str:
    text = (raw or "").strip()
    if not text or any(ch in text for ch in "\n\r=/ "):
        raise ValueError("node id must be a single token")
    return text


def _controller_target(raw: str) -> str:
    """SPARKSI_CONTROLLER value. A bare host gets the default announce port."""
    text = (raw or "").strip()
    if not text or any(ch in text for ch in "\n\r "):
        raise ValueError("controller must be host or host:port")
    host, sep, port = text.rpartition(":")
    if sep and host and port.isdigit():
        target = text
    else:
        host, port = text, "7501"
        target = f"{host}:{port}"
    if host.strip("[]").lower() in {"0.0.0.0", "::", "*"}:
        raise ValueError("refusing a controller address that is not a single host")
    return target


def _advertise_for(node_id: str) -> str:
    """Client address from that node's configured tailscale field.

    Empty means the unit starts with verification off. A loopback or wildcard
    value is refused so the unit cannot probe those.
    """
    text = _node_id(node_id)
    node = next((item for item in CFG.nodes if item.id == text), None)
    if node is None:
        raise ValueError(f"node id {text} is not in the config")
    address = str(node.tailscale or "").strip()
    if not address:
        return ""
    # Host characters only. A backslash or quote would continue or break the
    # Environment= line it is substituted into.
    if not all(ch.isalnum() or ch in ".:-" for ch in address):
        raise ValueError("advertised address must be a single host")
    if address.strip("[]").lower() in {"0.0.0.0", "::", "*", "127.0.0.1", "::1", "localhost"}:
        raise ValueError("refusing an advertised address that is loopback or wildcard")
    return address


def render_unit(kind: str, *, prefix: str, node_id: str, controller: str) -> str:
    templates = _unit_templates()
    if kind == "controller":
        text = (templates / "sparksi-controller.service.in").read_text(encoding="utf-8")
        text = text.replace("@PREFIX@", _prefix(prefix))
    elif kind == "announce":
        text = (templates / "sparksi-announce.service.in").read_text(encoding="utf-8")
        text = (
            text.replace("@NODE_ID@", _node_id(node_id))
            .replace("@CONTROLLER@", _controller_target(controller))
            .replace("@ADVERTISE@", _advertise_for(node_id))
            .replace("@PROTECT@", ",".join(sorted(protect.active())))
        )
    else:
        raise ValueError(f"unknown unit {kind}")
    leftover = [tok for tok in ("@PREFIX@", "@NODE_ID@", "@CONTROLLER@", "@ADVERTISE@", "@PROTECT@") if tok in text]
    if leftover:
        raise ValueError(f"unit template still has {', '.join(leftover)}")
    if not text.endswith("\n"):
        text += "\n"
    return text


def cmd_install_unit(args: argparse.Namespace) -> int:
    """Print a unit. This command has no install, enable, or start path."""
    if not args.print:
        print(
            "install-unit only prints a unit. Pass --print. It does not write, enable, or start anything.",
            file=sys.stderr,
        )
        return 2
    try:
        text = render_unit(
            args.unit,
            prefix=args.prefix,
            node_id=args.node_id,
            controller=args.controller,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    sys.stdout.write(text)
    return 0


def _config_target(args: argparse.Namespace) -> Path | None:
    """The file ``--file`` names; None means the one the usual search order finds."""
    return Path(args.file) if getattr(args, "file", "") else None


def _config_warnings(raw: dict[str, Any]) -> list[str]:
    """Things in a valid file that still work but should change."""
    out: list[str] = []
    paths = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    for new, old in compat.LEGACY_PATH_KEYS.items():
        if old in paths and new not in paths:
            out.append(f"[paths] {old} is deprecated; use [paths] {new}")
    section = raw.get("protect") if isinstance(raw.get("protect"), dict) else {}
    names = section.get("presets")
    if isinstance(names, str):
        names = [names]
    if names is None:
        names = []
    if not isinstance(names, list):
        out.append(f"[protect] presets must be a list of names, not {type(names).__name__}")
        names = []
    for name in names:
        if not isinstance(name, str):
            out.append(f"[protect] presets has an item that is not a name: {name!r}")
        elif name.strip() not in protect.PRESETS:
            out.append(f"[protect] presets names an unknown preset {name.strip()!r}")
    return out


def cmd_config_validate(args: argparse.Namespace) -> int:
    """Check a site file. Reads it and nothing else: no ssh, no controller, no write."""
    cfg, errors = check(_config_target(args))
    if cfg is None:
        for line in errors:
            print(f"error: {line}", file=sys.stderr)
        return 1
    if cfg.source is None:
        print(
            "no sparksi.toml found (looked at $SPARKSI_CONFIG, $XDG_CONFIG_HOME/sparksi/sparksi.toml "
            "and ./config/sparksi.toml); Sparksi would run on the built-in example",
            file=sys.stderr,
        )
        return 1
    for line in _config_warnings(cfg.file_raw):
        print(f"warning: {line}", file=sys.stderr)
    sparks = ", ".join(f"{node.id} ({role})" for role, node in cfg.role_nodes())
    count = len(cfg.nodes)
    print(f"ok: {cfg.source}: {count} Spark{'s' if count != 1 else ''}: {sparks}")
    return 0


def cmd_config_show(args: argparse.Namespace) -> int:
    """Print the settings in force as JSON. Secrets are redacted."""
    cfg, errors = check(_config_target(args))
    if cfg is None:
        for line in errors:
            print(f"error: {line}", file=sys.stderr)
        return 1
    try:
        text = json.dumps(cfg.effective(), indent=2, sort_keys=True, default=str)
    except RecursionError:
        print(f"error: {cfg.source}: the settings are nested too deeply to print", file=sys.stderr)
        return 1
    if cfg.source is None:
        print("no sparksi.toml found; showing the built-in example", file=sys.stderr)
    print(text)
    return 0


def cmd_config(args: argparse.Namespace) -> int:
    print("usage: sparksi config {validate,show} [--file PATH]", file=sys.stderr)
    return 2


def cmd_refuse(verb: str) -> int:
    print(refuse_mutator(verb), file=sys.stderr)
    try:
        Lock().require_unlocked(verb)
    except LockedError as e:
        print(str(e), file=sys.stderr)
    return 2


def build_parser() -> argparse.ArgumentParser:
    prog = "dgx-mem" if Path(sys.argv[0]).name == "dgx-mem" else "sparksi"
    p = argparse.ArgumentParser(prog=prog, description="DGX Spark unified-memory side view (Sparksi)")
    p.add_argument("--version", action="version", version=f"sparksi {__version__}")
    p.add_argument("--server", default="", help="controller base URL (default: loopback on the configured UI port)")
    p.add_argument(
        "--unlock",
        action="store_true",
        help="send the unlock secret from SPARKSI_UNLOCK, SPARKSI_UNLOCK_FILE, or the unlock file",
    )
    sub = p.add_subparsers(dest="cmd")

    st = sub.add_parser("status", help="live snapshot")
    st.add_argument("--json", action="store_true")
    st.set_defaults(func=cmd_status)

    inv = sub.add_parser("inventory", help="side catalog + live + model inventory file diff")
    inv.add_argument("--json", action="store_true")
    inv.set_defaults(func=cmd_inventory)

    dr = sub.add_parser("dry-run", help="tenant floor + recipe; fail closed")
    dr.add_argument("set")
    dr.add_argument("--json", action="store_true")
    dr.set_defaults(func=cmd_dry_run)

    lk = sub.add_parser("lock", help="acquire lease (default ON)")
    lk.add_argument("--json", action="store_true")
    lk.add_argument("--reason", default="cli lock")
    lk.set_defaults(func=cmd_lock)

    un = sub.add_parser("unlock", help="explicit unlock lease (TTL, then locked again)")
    un.add_argument("--json", action="store_true")
    un.add_argument("--ttl", type=int, default=15 * 60)
    un.set_defaults(func=cmd_unlock)

    wp = sub.add_parser("wanted", help="show or persist desired set (no load)")
    wp.add_argument("--json", action="store_true")
    wp.set_defaults(func=cmd_wanted)
    wsub = wp.add_subparsers(dest="wanted_op")
    ws = wsub.add_parser("set", help="persist wanted set; requires lock OFF")
    ws.add_argument("spec", help="recipe name or comma-separated catalog ids/seats")
    ws.add_argument("--json", action="store_true")
    ws.set_defaults(func=cmd_wanted_set)

    ap = sub.add_parser("apply", help="plan or commit wanted set (never stops the protected video service)")
    ap.add_argument("set", nargs="?", help="optional set to plan (does not write wanted)")
    ap.add_argument("--commit", action="store_true", help="execute if READY; lock must be OFF")
    ap.add_argument("--force-uncertain", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.set_defaults(func=cmd_apply)

    sw = sub.add_parser("swap", help="alias of apply; still never stops the protected video service")
    sw.add_argument("set", nargs="?", help="optional set to plan")
    sw.add_argument("--commit", action="store_true")
    sw.add_argument("--force-uncertain", action="store_true")
    sw.add_argument("--json", action="store_true")
    sw.set_defaults(func=cmd_apply)

    sv = sub.add_parser("serve", help="optional API/UI on :7500 (loopback+tailscale)")
    sv.add_argument("--port", type=int, default=7500, help="listen port (default 7500)")
    sv.add_argument(
        "--state-dir",
        default="",
        help="state directory for this process (lock, job, recipe run registry). Default is unchanged.",
    )
    sv.add_argument(
        "--bind",
        action="append",
        default=None,
        help="listen address. Repeat the flag or pass a comma list. Default is unchanged.",
    )
    sv.add_argument(
        "--no-announce",
        action="store_true",
        help="do not start the announce listener",
    )
    sv.set_defaults(func=cmd_serve)

    tk = sub.add_parser("tokens", help="per-model lifetime prompt/gen token counts")
    tk.add_argument("--json", action="store_true")
    tk.set_defaults(func=cmd_tokens)

    lib = sub.add_parser("library", help="archive releases and per-node copies")
    lib.add_argument("--json", action="store_true")
    lib.set_defaults(func=cmd_library)

    stg = sub.add_parser("stage", help="copy one archive onto a Spark (lock OFF)")
    stg.add_argument("id")
    stg.add_argument("--host", action="append", required=True, dest="hosts")
    stg.add_argument("--json", action="store_true")
    stg.set_defaults(func=cmd_stage)

    drp = sub.add_parser("drop-local", help="delete a node copy; the archive copy stays")
    drp.add_argument("id")
    drp.add_argument("--host", required=True)
    drp.add_argument("--confirm", required=True, help="must equal the local path")
    drp.add_argument("--json", action="store_true")
    drp.set_defaults(func=cmd_drop)

    def cmd_recipes(args: argparse.Namespace) -> int:
        print("usage: sparksi recipes {import,bundle,tidy-backups} ...", file=sys.stderr)
        return 2

    recipes = sub.add_parser("recipes", help="import, bundle or tidy recipes")
    recipes.set_defaults(func=cmd_recipes)
    recipes_sub = recipes.add_subparsers(dest="recipes_op")
    imp = recipes_sub.add_parser("import", help="read a container, compose file, unit, or script")
    imp.add_argument("model_id", nargs="?")
    imp.add_argument("--from", dest="source", default="", help="container:name, compose:file, unit:name, or script:path")
    imp.add_argument("--dry-run", action="store_true", help="print the recipe and do not save it")
    imp.add_argument("--all", action="store_true", help="survey every catalog row that has a launcher")
    imp.add_argument("--json", action="store_true")
    imp.set_defaults(func=cmd_recipes_import)
    bnd = recipes_sub.add_parser("bundle", help="move an older pinned recipe and the files it mounts into a recipe folder")
    bnd.add_argument("model_id")
    bnd.add_argument("--recipe", default="", help="recipe name (default: the model's default recipe)")
    bnd.add_argument("--keep", action="append", default=[], help="a mount source to leave as it is (repeatable)")
    bnd.add_argument("--dry-run", action="store_true", help="show what would be written and write nothing")
    bnd.add_argument("--json", action="store_true")
    bnd.set_defaults(func=cmd_recipes_bundle)
    tdy = recipes_sub.add_parser("tidy-backups", help="move recipes.bak-* folders and *.bak* files into backups/<date>/")
    tdy.add_argument("--dry-run", action="store_true", help="list the moves and move nothing")
    tdy.add_argument("--json", action="store_true")
    tdy.set_defaults(func=cmd_recipes_tidy)

    rn = sub.add_parser("run", help="legacy: start a catalog launcher on this machine (lock off)")
    rn.add_argument("id")
    rn.add_argument("--json", action="store_true")
    rn.set_defaults(func=cmd_run_model)

    start = sub.add_parser("start", help="start a recipe through the controller")
    start.add_argument("recipe_id")
    start.add_argument("--wait", action="store_true", help="poll until ready, failed, or timeout")
    start.add_argument("--timeout", type=int, default=180, help="seconds to wait when --wait is set")
    start.add_argument("--ctx", type=int, default=None, help="context length, when choosing one")
    start.add_argument("--json", action="store_true")
    start.set_defaults(func=cmd_start)

    stop = sub.add_parser("stop", help="stop a recipe or run through the controller")
    stop.add_argument("target", help="recipe id or run id")
    stop.add_argument("--wait", action="store_true", help="poll until the run is not running")
    stop.add_argument("--timeout", type=int, default=180, help="seconds to wait when --wait is set")
    stop.add_argument("--json", action="store_true")
    stop.set_defaults(func=cmd_stop)

    runs = sub.add_parser("runs", help="list live recipe runs")
    runs.add_argument("--json", action="store_true")
    runs.set_defaults(func=cmd_runs)

    plan = sub.add_parser("plan", help="dry plan only; does not start a run")
    plan.add_argument("recipe_id")
    plan.add_argument("--ctx", type=int, default=None)
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(func=cmd_plan)

    iu = sub.add_parser(
        "install-unit",
        help="print a user systemd unit; does not enable or start anything",
    )
    iu.add_argument("--print", action="store_true", help="write the unit to stdout")
    iu.add_argument("--unit", choices=("controller", "announce"), default="controller")
    iu.add_argument("--prefix", default="/home/USER/src/sparksi", help="controller WorkingDirectory and venv prefix")
    iu.add_argument("--node-id", default="a", help="announcer SPARKSI_NODE_ID (a config [[nodes]] id)")
    iu.add_argument("--controller", default="100.64.0.1:7501", help="announcer SPARKSI_CONTROLLER=host:port")
    iu.set_defaults(func=cmd_install_unit)

    cf = sub.add_parser("config", help="check or show sparksi.toml (reads only)")
    cf.set_defaults(func=cmd_config)
    cf_sub = cf.add_subparsers(dest="config_op")
    cv = cf_sub.add_parser("validate", help="check the file; exit 1 and name each problem when it is unusable")
    cv.add_argument("--file", default="", help="file to check (default: the one Sparksi would load)")
    cv.set_defaults(func=cmd_config_validate)
    cs = cf_sub.add_parser("show", help="print the settings in force as JSON, secrets redacted")
    cs.add_argument("--file", default="", help="file to read (default: the one Sparksi would load)")
    cs.set_defaults(func=cmd_config_show)

    for verb in MUTATOR_VERBS:
        m = sub.add_parser(verb, help="not implemented")
        m.set_defaults(func=lambda args, v=verb: cmd_refuse(v))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    if args.cmd != "config" and CFG_ERRORS:
        # The import kept the built-in example so it could finish. Do not act on it.
        print("\n".join(CFG_ERRORS), file=sys.stderr)
        return 1
    try:
        return int(args.func(args))
    except CatalogError as e:
        print(str(e), file=sys.stderr)
        return 2
    except Refused as e:
        print(str(e), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
