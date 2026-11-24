# SPDX-License-Identifier: Apache-2.0

"""Optional HTTP on :7500. Localhost + Tailscale only."""

from __future__ import annotations

import json
import socket
import threading
import tomllib
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from sparksi.apply import (
    build_plan,
    commit as apply_commit,
    save_plan,
    union_spec,
    wanted_for_plan,
)
from sparksi.catalog import CatalogError, load_catalog, load_yaml, model_inventory_path
from sparksi.compat import deprecation_notes, with_legacy_fields
from sparksi import protect
from sparksi.lock import Lock
from sparksi.never import Refused, assert_bind_allowed, assert_serve_port
from sparksi.probe import DOCKER_STATS_ARGV, docker_stats_gib, run_readonly, snapshot
from sparksi.refresh import PENDING, Refreshing
from sparksi.public import video_service_is_live, refuse_apply_while_video_live, strip_secrets
from sparksi.status import assemble_inventory, assemble_status, run_dry_run
from sparksi.activity import store as activity_store
from sparksi.dash import current as current_dash
from sparksi.history import LIVE_FRESH_SECS, sample_card, store as history_store
from sparksi.power import fleet_power
from sparksi.jobs import current as current_job
from sparksi.jobs import (
    load_in_progress,
    request_cancel,
    request_pause,
    restore_job,
    start_drop,
    start_run,
    start_stage,
    start_stop,
    start_switch,
)
from sparksi.log import logger
from sparksi.nodes import invalidate_cache, node_report, node_specs
from sparksi.launch import monitor_loaded, plan_switch
from sparksi.library import fleet, row_by_id, sync_report
from sparksi.models import list_models
from sparksi.recipes.service import (
    begin_run,
    halt_run,
    list_model_cards,
    plan_model,
    preview_recipe,
    service_action,
    run_one,
    runs_index,
    valid_id,
    write_recipe,
)
from sparksi.recipes.named import (
    BLANK,
    delete_named,
    get_named,
    list_named,
    pinned_default_refusal,
    rename_named,
    save_as,
    save_named,
    set_default,
)
from sparksi.recipes.store import using_recipe, valid_recipe_name
from sparksi.remote import SshRunner
from sparksi.stop import stop_id
from sparksi.ledger import day_view, poll_once
from sparksi.token_day import cached_lifetime
from sparksi.tokens import report as token_report
from sparksi.wanted import apply_set, load_wanted, seat_view
from sparksi.config import CFG, Config, config_path, env, reload as reload_config
from sparksi.hosts import REFUSAL, Allowlist, host_name
from sparksi import discover, monitor_hosts, netconfig, options

DEFAULT_PORT = 7500
DEFAULT_BIND = ["127.0.0.1"]


def _window_args(qs: dict[str, list[str]]) -> tuple[float | None, float | None]:
    """Optional ``start`` and ``end`` epoch seconds. Both or neither."""

    def one(name: str) -> float | None:
        raw = (qs.get(name) or [None])[0]
        if raw is None or raw == "":
            return None
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(name) from exc
        if number != number or number in (float("inf"), float("-inf")):
            raise ValueError(name)
        return number

    start = one("start")
    end = one("end")
    if (start is None) ^ (end is None):
        raise ValueError("window")
    return start, end


def _optional_stamp(qs: dict[str, list[str]], name: str) -> float | None:
    """One optional epoch query value. Absent stays absent. A bad value is refused."""
    raw = (qs.get(name) or [None])[0]
    if raw is None or raw == "":
        return None
    try:
        number = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(name)
    return number


INDEX_HTML = Path(__file__).resolve().parent / "static" / "index.html"
STATIC_DIR = INDEX_HTML.parent
_ASSET_TYPES = {".png": "image/png", ".woff2": "font/woff2"}
# Vendored dashboard files. Anything else, including index.html, stays off this map.
_ASSETS = {
    "peelsa.png",
    "favicon.png",
    "sparksi-wordmark-transparent.png",
    "sparksi-wordmark-transparent-2x.png",
    "fonts/inter-latin-400.woff2",
    "fonts/inter-latin-500.woff2",
    "fonts/inter-latin-600.woff2",
    "fonts/inter-latin-700.woff2",
}


def asset_path(url_path: str) -> Path | None:
    """A vendored image or font for the dashboard, or None.

    The page, its mark, and Inter are served from this process. No other
    path is accepted, so a request cannot walk out of the static directory.
    """
    rel = url_path.lstrip("/")
    if rel not in _ASSETS:
        return None
    path = (STATIC_DIR / rel).resolve()
    root = STATIC_DIR.resolve()
    if path != root and root not in path.parents:
        return None
    if not path.is_file():
        return None
    return path


class _State:
    lock: Lock
    catalog: dict[str, Any]
    runner: Any = None
    monitor_fn: Any = None
    # Stage feed or the optional monitor mirror. Unset by default, so the
    # collector and /api/runs keep their normal sources.
    observe_fn: Any = None
    runs_fn: Any = None
    readonly: bool = False
    mirror: Any = None


STATE = _State()
# One SSH runner for the process when tests have not injected one. A new
# object per request would still be correct, but the 15s fact cache is keyed
# apart from the runner, and holding one avoids rebuilding it on every poll.
_PROD_RUNNER: tuple[Any, Any] | None = None


def collector_card() -> dict[str, Any]:
    """Card the history loop records. ``observe_fn`` is the stage feed, not ``monitor_fn``."""
    fn = STATE.observe_fn
    if fn is not None:
        card = fn()
        return card if isinstance(card, dict) else {"hosts": [], "loaded": []}
    return current_dash()


def active_runner(catalog: dict[str, Any]):
    """The injected runner, or an ``SshRunner`` for this catalog.

    Production leaves ``STATE.runner`` unset. Tests that set it keep their fake.
    """
    global _PROD_RUNNER
    if STATE.runner is not None:
        return STATE.runner
    raw_hosts = catalog.get("hosts")
    hosts = raw_hosts if isinstance(raw_hosts, dict) else {}
    key = tuple(
        sorted(
            (str(name), str((spec or {}).get("ssh") or ""), str((spec or {}).get("local_when_hostname") or ""))
            for name, spec in hosts.items()
            if isinstance(spec, dict) or spec is None
        )
    )
    cached = _PROD_RUNNER
    if cached is not None and cached[0] == key and isinstance(cached[1], SshRunner):
        return cached[1]
    runner = SshRunner(catalog)
    _PROD_RUNNER = (key, runner)
    return runner


_PAGES = {"/", "/overview", "/tokens", "/activity", "/library", "/archive", "/models", "/network", "/options"}


def _live_card() -> Any:
    """The monitor card: the mirror when one is set, else the local announce feed."""
    return (STATE.monitor_fn or current_dash)()


def _monitor_view() -> tuple[list[Any], list[Any]]:
    try:
        card = _live_card()
    except Exception:  # noqa: BLE001 — runs still return without a serving guess
        return [], []
    if not isinstance(card, dict):
        return [], []
    loaded = card.get("loaded") if isinstance(card.get("loaded"), list) else []
    hosts = card.get("hosts") if isinstance(card.get("hosts"), list) else []
    return loaded, hosts


def _with_serving(body: dict[str, Any]) -> dict[str, Any]:
    """Add the serving state beside runs. A body that already has it is kept."""
    from sparksi.config import CFG
    from sparksi.serving_state import attach

    loaded, hosts = _monitor_view()
    extra = {role: node.id for role, node in CFG.role_nodes()[2:]}
    worker_id = CFG.worker.id if CFG.has_worker else ""
    return attach(body, loaded, hosts, head_id=CFG.head.id, worker_id=worker_id, extra_ids=extra or None)


def _model_suffix(path: str) -> tuple[str, str] | None:
    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[0] == "api" and parts[1] == "models" and parts[3] in ("recipe", "plan", "service"):
        if valid_id(parts[2]):
            return parts[2], parts[3]
    return None


def _external_container_refusal(cat: dict[str, Any], runner: Any, host: str, container: str) -> str | None:
    """Refuse a container action on an engine that serves a library row from outside Sparksi.

    Fails closed: when the library cannot be read, nothing is stopped.
    """
    from sparksi.library import external_refusal, fleet

    try:
        rows = fleet(cat, runner)["library"]
    except Exception:  # noqa: BLE001 — no library answer: refuse rather than guess
        return "Could not check whether this engine serves a library model; nothing was stopped."
    wanted = {host, CFG.id_for(host)}
    for row in rows:
        engine = row.get("external_engine") if row.get("external") else None
        if not isinstance(engine, dict):
            continue
        for side, names in (engine.get("containers") or {}).items():
            if {side, CFG.id_for(side)} & wanted and container in (names or []):
                return external_refusal(row)
    return None


def _named_route(path: str) -> tuple[str, str, str] | None:
    """/api/models/<id>/recipes[/<name>[/default|/rename]] -> (id, name, action)."""
    parts = path.strip("/").split("/")
    if len(parts) < 4 or parts[:4] != ["api", "models", parts[2], "recipes"]:
        return None
    model_id = parts[2]
    if not valid_id(model_id):
        return None
    if len(parts) == 4:
        return model_id, "", ""
    name = parts[4]
    if not valid_recipe_name(name):
        return None
    if len(parts) == 5:
        return model_id, name, ""
    if len(parts) == 6 and parts[5] in ("default", "rename"):
        return model_id, name, parts[5]
    return None


def _pick_recipe(source: dict[str, Any]) -> tuple[str | None, str | None]:
    """Pop recipe_name from a body or query. Returns (name, error)."""
    fresh = source.pop("recipe_new", None)
    raw = source.pop("recipe_name", None)
    if fresh and str(fresh).lower() not in ("0", "false"):
        return BLANK, None
    if not raw:
        return None, None
    if not valid_recipe_name(raw):
        return None, "bad recipe name"
    return str(raw), None


def _run_id(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[0] == "api" and parts[1] == "runs" and valid_id(parts[2]):
        return parts[2]
    return None


def _run_stop_id(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[0] == "api" and parts[1] == "runs" and parts[3] == "stop" and valid_id(parts[2]):
        return parts[2]
    return None


def _gpu_clocks_node(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) == 4 and parts[0] == "api" and parts[1] == "nodes" and parts[3] == "gpu-clocks" and valid_id(parts[2]):
        return parts[2]
    return None


def _gpu_clock_activity(rec: dict[str, Any]) -> str | None:
    """``decoding`` or ``running`` for a live loaded model, else None.

    An active decode is GEN/GENERATING/VIDEO, or a reported decode rate of at least
    1 tok/s — the same signals the dashboard already uses.
    """
    if rec.get("video_service") is True or rec.get("kind") == "video_service":
        return None
    raw = str(rec.get("state") or "").upper()
    tok = rec.get("tok_s")
    try:
        rate = float(tok) if tok not in (None, "") else 0.0
    except (TypeError, ValueError):
        rate = 0.0
    if raw in {"GEN", "GENERATING", "VIDEO"} or rate >= 1.0:
        return "decoding"
    return "running"


def _gpu_clock_busy(spec: dict[str, str]) -> str | None:
    """Why a clock change on this Spark needs a second OK, or None when nothing is running.

    A loaded model on the Spark (or spanning it, for a pair), or a Sparksi run that names it
    and has not stopped, counts. When the monitor card can tell the Spark is decoding,
    the message says so. When the card cannot be read, the answer is unknown, so the
    change also needs the OK.
    """
    names = {spec.get("id"), spec.get("name"), spec.get("key")}
    names.discard(None)
    names.discard("")
    loaded = monitor_loaded()
    if loaded is None:
        return "Sparksi cannot tell whether a model is running on this Spark. Confirm to change the GPU clock cap anyway."
    found: str | None = None
    for rec in loaded:
        if not isinstance(rec, dict):
            continue
        activity = _gpu_clock_activity(rec)
        if not activity:
            continue
        host = rec.get("host") or CFG.head.id
        span = rec.get("span") or []
        if host in names or any(item in names for item in span):
            if activity == "decoding":
                return "This Spark is decoding now. Confirm to change the GPU clock cap during an active decode."
            found = "A model is running on this Spark. Confirm to change the GPU clock cap while it runs."
    if found:
        return found
    try:
        from sparksi.recipes.runs import list_runs

        runs = list_runs()
    except Exception:  # noqa: BLE001
        return "Sparksi cannot tell whether a model is running on this Spark. Confirm to change the GPU clock cap anyway."
    for run in runs:
        if str(run.get("phase") or "") in ("failed", "timed_out", "stopped", "cancelled"):
            continue
        for node in run.get("nodes") or []:
            if str(node.get("phase") or "") in ("failed", "timed_out", "stopped", "cancelled"):
                continue
            if node.get("host") in names:
                return "A model is starting or running on this Spark. Confirm to change the GPU clock cap while it runs."
    return None


def _writes_allowed(handler: "Handler", body: dict[str, Any], token: str | None) -> bool:
    """Same rule as a library run: the lock is off, or the caller holds the unlock token."""
    if bool(body.get("from_ui")):
        if STATE.lock.read().locked:
            handler._json(403, {"error": "Unlock first."})
            return False
        return True
    ok, why = STATE.lock.allow_api_write(token)
    if not ok:
        handler._json(403, {"error": why})
        return False
    return True


def _known_sparks() -> dict[str, str]:
    """Configured Spark names and addresses (lower-case) -> node id, for Discover."""
    out: dict[str, str] = {}
    for node in CFG.nodes:
        for key in (node.id, node.name, node.ssh, node.tailscale):
            if key and key != "local":
                out[str(key).lower()] = node.id
    return out


# DNS rebinding defense: answer only to this machine's own names and addresses.
HOSTS = Allowlist(
    lambda: [str(a) for a in (CFG.controller.get("allowed_hosts") or [])]
    + [str(a) for a in (CFG.controller.get("bind") or [])]
    + [CFG.controller_name]
)


# One Network page config edit at a time.
_NET_WRITE = threading.Lock()


def _config_write_ok(handler: "Handler") -> bool:
    """Every state-changing write: same-origin JSON only. A caller with no Origin (CLI) passes.

    Recipe saves use it without the lock; model actions add the lock check on top."""
    media_type = (handler.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        handler._json(415, {"error": "Content-Type must be application/json"})
        return False
    site = (handler.headers.get("Sec-Fetch-Site") or "").strip().lower()
    if site == "cross-site":
        handler._json(403, {"error": "cross-site write refused (Sec-Fetch-Site: cross-site)"})
        return False
    origin = (handler.headers.get("Origin") or "").strip()
    if origin:
        host = (handler.headers.get("Host") or "").strip()
        if not host or not _same_origin(origin, host):
            handler._json(403, {"error": "cross-origin write refused (page origin does not match this server's address)"})
            return False
    return True


def _origin_key(scheme: str, hostport: str) -> tuple[str, int] | None:
    """(host, port) with case, a trailing dot and the default port folded away."""
    hostport = hostport.strip().lower()
    if not hostport:
        return None
    if hostport.startswith("["):
        end = hostport.find("]")
        if end < 0:
            return None
        name, rest = hostport[1:end], hostport[end + 1:]
        port_text = rest[1:] if rest.startswith(":") else ""
    elif hostport.count(":") == 1:
        name, port_text = hostport.split(":", 1)
    else:
        name, port_text = hostport, ""
    name = name.rstrip(".")
    if not name:
        return None
    if port_text:
        if not port_text.isdigit():
            return None
        port = int(port_text)
    else:
        port = 443 if scheme == "https" else 80
    return name, port


def _same_origin(origin: str, host: str) -> bool:
    """True when the browser's Origin names the same host and port the request was sent to."""
    parsed = urlparse(origin.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https") or not parsed.netloc:
        return False
    want = _origin_key(scheme, parsed.netloc)
    got = _origin_key(scheme, host)
    return want is not None and want == got


def is_page(path: str) -> bool:
    return path in _PAGES


# The old Setup page (and its steps) moved to Network: role assignment lives there now.
_SETUP_PAGES = ("/setup",)


def _redirect_for(path: str) -> str | None:
    """Where an old page address now lives, or None."""
    if path in _SETUP_PAGES or path.startswith("/setup/"):
        return "/network"
    if _legacy_page(path):
        return "/overview"
    # The Temps page became a collapsed section on Overview.
    if path == "/temps":
        return "/overview#temps"
    return None


def _legacy_page(path: str) -> bool:
    if path == "/legacy":
        return True
    if not path.startswith("/machine/"):
        return False
    rest = path[len("/machine/"):]
    return bool(rest) and "/" not in rest


_VIEW_FOR_PAGE = {"/": "overview", "/overview": "overview"}


def page_html(path: str = "/") -> str:
    """Dashboard page as stored, with the view for ``path`` already shown.

    The script picks the view from the URL too; marking it in the HTML means a
    reload shows the right page even before (or without) the script running.
    Node names come from ``GET /api/config``.
    """
    text = INDEX_HTML.read_text(encoding="utf-8")
    view = _VIEW_FOR_PAGE.get(path) or (path.strip("/") if path in _PAGES else "overview")
    if view == "overview":
        return text
    shown = f'<div id="view-{view}" class="view" hidden>'
    if shown not in text:
        return text
    text = text.replace('<div id="view-overview" class="view">', '<div id="view-overview" class="view" hidden>', 1)
    return text.replace(shown, f'<div id="view-{view}" class="view">', 1)


def note_activity(kind: str, message: str, machine: str = "") -> None:
    try:
        activity_store().append(kind, message, machine)
    except Exception:
        return


def on_controller() -> bool:
    """True on the machine that runs the announce listener and controller bind."""
    name = CFG.controller_name
    return env("SPARKSI_CONTROLLER_MODE") == name or socket.gethostname() == name


def parse_bind(values: list[str] | None) -> list[str] | None:
    """Addresses from repeated ``--bind`` flags or comma lists. Empty means the default."""
    if not values:
        return None
    found: list[str] = []
    for item in values:
        for part in str(item).split(","):
            text = part.strip()
            if text and text not in found:
                found.append(text)
    return found or None


def listen_targets(explicit: list[str] | None, catalog: dict[str, Any] | None = None) -> list[str]:
    """Bind list for this process. An explicit list replaces the catalog addresses."""
    chosen = parse_bind(explicit)
    if chosen:
        return chosen
    return allowed_hosts(catalog)


def start_announce(enabled: bool) -> None:
    """Start the announce listener only on the controller, and only when asked."""
    if not enabled or not on_controller():
        return
    from sparksi.announce_rx import start_listener

    start_listener()


def allowed_hosts(catalog: dict[str, Any] | None = None) -> list[str]:
    catalog = catalog or {}
    if on_controller():
        block = catalog.get("controller") or {
            "bind_allow": CFG.controller.get("bind"),
            "bind_deny": CFG.controller.get("bind_deny"),
        }
    else:
        block = catalog.get("host") or {}
    allow = list(block.get("bind_allow") or DEFAULT_BIND)
    deny = list(block.get("bind_deny") or [])
    assert_bind_allowed(allow, deny)
    return allow


def _read_token(handler: BaseHTTPRequestHandler, body: dict[str, Any] | None = None) -> str | None:
    h = (
        handler.headers.get("X-Sparksi-Token")
        or handler.headers.get("X-Dgx-Mem-Token")
        or handler.headers.get("X-Dgx-Mem-Unlock")
    )
    if h:
        return h.strip()
    auth = handler.headers.get("Authorization") or ""
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    if body and body.get("token"):
        return str(body["token"])
    return None


def _apply_plan(cat, lock: Lock, truth, truth_err, spec: str | None = None) -> tuple:
    wanted = wanted_for_plan(cat, spec)
    snap = snapshot(cat, truth)
    union = union_spec(wanted, snap["residents"])
    try:
        dry = (
            run_dry_run(cat, union, truth, truth_err=truth_err)
            if union
            else {"verdict": "UNCERTAIN", "reasons": ["no wanted set"]}
        )
    except CatalogError as e:
        dry = {"verdict": "UNCERTAIN", "reasons": [str(e)]}
    plan = build_plan(cat, wanted, snap["residents"], dry, lock_on=lock.read().locked)
    plan["video_service_live"] = video_service_is_live()
    with_legacy_fields(plan)
    plan["commit_allowed"] = (
        plan.get("verdict") == "READY" and not plan["video_service_live"] and not lock.read().locked
    )
    save_plan(plan)
    return plan, wanted, snap


def _wanted_payload(cat: dict[str, Any], lock: Lock, truth, truth_err) -> dict[str, Any]:
    snap = assemble_status(cat, truth, lock, truth_err=truth_err)
    wanted = load_wanted() or {}
    return with_legacy_fields({
        "wanted": wanted,
        "lock": snap["lock"],
        "seats": snap.get("seats") or seat_view(cat, snap["residents"], wanted),
        "residents": [r for r in snap["residents"] if r.get("resident")],
        "conflicts": snap.get("conflicts") or [],
        "video_service_live": snap.get("video_service_live"),
        "last_dry_run_stale": True,
    })


def post_apply(
    body: dict[str, Any],
    token: str | None,
    *,
    listen_fn=None,
    runner=None,
) -> tuple[int, dict[str, Any]]:
    """Commit gate. :8188 live ⇒ 4xx even if victims omit the protected video service."""
    blocked = refuse_apply_while_video_live(listen_fn)
    if blocked:
        return 409, blocked
    ok, why = STATE.lock.allow_api_write(token)
    if not ok:
        return 403, {"error": why, "executed": False}
    cat = load_catalog()
    try:
        truth = load_yaml(model_inventory_path())
        terr = None
    except Exception as e:
        truth, terr = None, str(e)
    spec = body.get("set") or body.get("spec")
    plan, wanted, snap = _apply_plan(cat, STATE.lock, truth, terr, spec)
    if video_service_is_live(listen_fn):
        return 409, refuse_apply_while_video_live(listen_fn)
    if any(v.get("protected") for v in plan.get("victims") or []):
        return 409, {**plan, "executed": False, "error": "REFUSE: would touch the protected video service"}
    force = bool(body.get("force_uncertain"))
    out = apply_commit(
        cat,
        plan,
        snap["residents"],
        STATE.lock,
        force_uncertain=force,
        wanted=wanted,
        runner=runner,
    )
    out["auth"] = why
    if out.get("verdict") == "REFUSE" and not out.get("already_applied"):
        return 409, out
    return 200, out


# Reads that reach the Sparks (SSH, probes) or sample for seconds (docker stats)
# answer from the last result and refresh in the background, so no poll waits.
READS = Refreshing(ttl=5.0, first_wait=0.6, max_age=60.0)  # older than a minute: not served
SLOW_READS = Refreshing(ttl=15.0, first_wait=0.3, max_age=120.0)


def _read(name: str, cat: dict[str, Any], runner: Any, fn, *, slow: bool = False) -> Any:
    hosts = json.dumps(cat.get("hosts") or {}, sort_keys=True, default=str)
    key = (name, hash(hosts), id(runner))
    return (SLOW_READS if slow else READS).get(key, fn)


def _docker_sample(wait: float | None = None):
    return SLOW_READS.get("docker-stats", lambda: run_readonly(list(DOCKER_STATS_ARGV)), wait=wait)


def _docker_fn(name_substr: str):
    raw = _docker_sample()
    if raw is PENDING:
        return None, [], "docker stats not sampled yet"
    return docker_stats_gib(name_substr, raw)


def _node_read(cat: dict[str, Any]) -> dict[str, Any]:
    from sparksi.heartbeat import HeartbeatReads

    rep = _read("nodes", cat, STATE.runner, lambda: node_report(cat, HeartbeatReads(active_runner(cat)), force=True))
    if rep is PENDING:
        return {"nodes": [], "stale": True, "pending": True}
    return rep


PROBES = Refreshing(ttl=5.0, max_age=60.0)


class CachedReads:
    """A runner whose read-only probes share one recent answer per (host, request).

    The background node reads and the recipe preview send the same probes, so a
    preview usually finds a fresh answer instead of waiting on SSH. Anything that
    is not a read goes straight to the real runner.
    """

    READ_OPS = frozenset({"probe", "list_shards"})

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def invoke(self, host: str, req: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        if not isinstance(req, dict) or req.get("op") not in self.READ_OPS:
            return self._runner.invoke(host, req, timeout=timeout)
        from sparksi import heartbeat

        beat = heartbeat.beat_answer(host, req)
        if beat is not None:
            return beat  # a fresh announcer beat answers it: no SSH
        key = (id(self._runner), str(host), json.dumps(req, sort_keys=True, default=str))
        got = PROBES.get(key, lambda: heartbeat.HeartbeatReads(self._runner).invoke(host, req, timeout=timeout), wait=float(timeout) + 1.0)
        if got is PENDING:
            return {"ok": False, "error": "no answer yet", "host": host}
        return got

    def capture(self, host: str, argv: list[str], timeout: int = 8) -> dict[str, Any]:
        """``capture`` is read-only argv (the runner refuses anything else); share recent answers."""
        key = (id(self._runner), "capture", str(host), json.dumps(list(argv), default=str))
        got = PROBES.get(key, lambda: self._runner.capture(host, argv, timeout), wait=float(timeout) + 1.0)
        if got is PENDING:
            return {"ok": False, "error": "no answer yet", "stdout": "", "stderr": "", "code": None}
        return got

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)


def _access_log_mode() -> str:
    import os

    from sparksi.config import CFG

    mode = os.environ.get("SPARKSI_ACCESS_LOG") or str(CFG.controller.get("access_log") or "slow")
    return mode if mode in ("off", "slow", "all") else "slow"


SLOW_REQUEST_S = 1.0


def _unmanaged_nodes() -> list[dict[str, Any]]:
    from sparksi.config import CFG

    return [{"id": n.id, "name": n.name or n.id, "role": role} for role, n in CFG.role_nodes()]


def _unmanaged_containers(runner: Any, host: str) -> dict[str, int] | None:
    """{container: port} for running containers Sparksi did not start on ``host`` (read-only probe)."""
    try:
        reply = runner.invoke(host, {"op": "probe", "ports": [], "unmanaged_ports": True}, timeout=20)
    except Exception:  # noqa: BLE001 - shown as "docker could not be asked"
        return None
    found = reply.get("unmanaged_containers") if isinstance(reply, dict) and reply.get("ok") else None
    if not isinstance(found, dict):
        return None
    out: dict[str, int] = {}
    for name, port in found.items():
        try:
            out[str(name)] = int(port)
        except (TypeError, ValueError):
            continue
    return out


def _unmanaged_items(runner: Any, *, cached: bool, cat: dict[str, Any]) -> tuple[list[dict[str, Any]], bool]:
    """Models serving on the Sparks with no active Sparksi run; (items, checked)."""
    from sparksi import unmanaged
    from sparksi.config import CFG
    from sparksi.heartbeat import HeartbeatReads
    from sparksi.recipes.runs import list_runs

    loaded, _hosts = _monitor_view()
    runs = list_runs()
    items = unmanaged.find(loaded, runs, _unmanaged_nodes())
    hosts = sorted({str(item["host"]) for item in items})
    checked = True
    by_host: dict[str, Any] = {}
    for host in hosts:
        if cached:
            beat_runner = HeartbeatReads(runner)  # the stop path below asks over SSH (cached=False)
            got = _read("unmanaged:" + host, cat, runner, lambda host=host: _unmanaged_containers(beat_runner, host))
            if got is PENDING:
                checked = False
                got = None
        else:
            got = _unmanaged_containers(runner, host)
        by_host[host] = got
    items = unmanaged.attach_containers(items, by_host)
    # A container named in any Sparksi run record is ours, whatever its name or phase.
    ours = unmanaged.sparksi_containers(runs)
    items = unmanaged.mark_stoppable(items, CFG.controller.get("stop_unmanaged") is True, ours)
    items = unmanaged.mark_emergency(items, ours, _outside_keep(cat), _outside_keep_ports(cat))
    return items, checked


def _outside_keep_ports(cat: dict[str, Any]) -> set[int]:
    """Ports of catalog rows the config protects (off-limits ports, never_touch rows)."""
    from sparksi.models import row_is_protected

    ports: set[int] = set()
    for row in (cat or {}).get("library") or []:
        if not isinstance(row, dict) or not row_is_protected(row, cat):
            continue
        try:
            ports.add(int(row.get("port") or 0))
        except (TypeError, ValueError):
            continue
    ports.discard(0)
    return ports


def _outside_keep(cat: dict[str, Any]) -> set[str]:
    """Lower-cased names the catalog says never to stop (never_touch.do_not_stop)."""
    touch = (cat or {}).get("never_touch") or {}
    return {str(item).strip().lower() for item in (touch.get("do_not_stop") or []) if str(item).strip()}


# One emergency stop at a time.
_OUTSIDE_STOP = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    # A client that connects and then says nothing frees its thread after this long.
    timeout = 60

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def send_response_only(self, code: int, message: str | None = None) -> None:
        self._replied = True
        self._status = code
        super().send_response_only(code, message)

    def _log_timing(self) -> None:
        t0 = getattr(self, "_t0", None)
        if t0 is None:
            return
        self._t0 = None
        took = time.perf_counter() - t0
        mode = _access_log_mode()
        if mode == "off" or (mode == "slow" and took < SLOW_REQUEST_S):
            return
        logger().info(
            "http %s %s %s %.3fs host=%s",
            getattr(self, "command", "-"),
            urlparse(getattr(self, "path", "") or "").path[:120],
            getattr(self, "_status", "-"),
            took,
            (self.headers.get("Host") if getattr(self, "headers", None) else "-") or "-",
        )

    def handle_one_request(self) -> None:
        """Every request gets an answer: an unexpected error becomes a 500, not a closed socket."""
        self._replied = False
        self._status = None
        self._t0 = None
        self._read_only_post = False
        try:
            super().handle_one_request()
            if getattr(self, "command", "") in ("POST", "PUT", "DELETE") and not self._read_only_post:
                READS.expire()  # a write may change what the next read shows; a plan or a scan does not
                SLOW_READS.expire()
        except (ConnectionError, TimeoutError, socket.timeout):
            self.close_connection = True
        except Exception:
            self.close_connection = True
            logger().exception("unhandled error on %s %s", getattr(self, "command", "?"), urlparse(getattr(self, "path", "") or "").path)
            if self._replied:
                return
            body = json.dumps({"error": "internal error"}).encode()
            try:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                if getattr(self, "command", "") != "HEAD":
                    self.wfile.write(body)
                self.wfile.flush()
            except OSError:
                pass
        finally:
            self._log_timing()

    def parse_request(self) -> bool:
        self._t0 = time.perf_counter()
        if not super().parse_request():
            return False
        host = (self.headers.get("Host") or "").strip()
        if HOSTS.allows(host):
            return True
        text = REFUSAL.format(name=host_name(host) or host[:80]).encode()
        self.send_response(421)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(text)))
        self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(text)
        self.close_connection = True
        return False

    def _json(self, code: int, payload: Any) -> None:
        if self.command in ("POST", "PUT", "DELETE") and not getattr(self, "_read_only_post", False):
            # Mark reads old before the reply leaves, so a read sent right after it
            # never gets a value from before the write (handle_one_request expires again).
            READS.expire()
            SLOW_READS.expire()
        body = json.dumps(strip_secrets(payload), indent=2, default=str).encode()
        if self.command in ("PUT", "POST", "DELETE") and "/recipe" in (self.path or ""):
            # One line per recipe write, so a failed save leaves a trace in the service log.
            error = payload.get("error") if isinstance(payload, dict) else ""
            logger().info(
                "recipe write %s %s -> %s host=%s origin=%s ua=%s%s",
                self.command,
                urlparse(self.path).path,
                code,
                (self.headers.get("Host") or "-")[:80],
                (self.headers.get("Origin") or "-")[:80],
                (self.headers.get("User-Agent") or "-")[:160],
                f" error={str(error)[:120]}" if error else "",
            )
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, code: int, text: str) -> None:
        body = text.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body_json(self) -> dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode())
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _asset(self, path: Path, *, head: bool = False) -> None:
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", _ASSET_TYPES[path.suffix])
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def do_HEAD(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        asset = asset_path(path)
        if asset is not None:
            return self._asset(asset, head=True)
        moved = _redirect_for(path)
        if moved:
            self.send_response(302)
            self.send_header("Location", moved)
            self.end_headers()
            return
        if is_page(path):
            body = page_html(path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        asset = asset_path(path)
        if asset is not None:
            return self._asset(asset)
        if path == "/api/config":
            body = CFG.public()
            body["protect"] = {
                "presets": sorted(protect.active()),
                "off_limits_ports": sorted(protect.off_limits_ports()),
            }
            return self._json(200, body)
        moved = _redirect_for(path)
        if moved:
            self.send_response(302)
            self.send_header("Location", moved)
            self.end_headers()
            return
        if is_page(path):
            return self._html(200, page_html(path))
        qs = parse_qs(parsed.query)
        if path == "/api/activity":
            return self._json(200, {"events": activity_store().events()})
        if path == "/api/fleet-power":
            try:
                card = _live_card()
            except Exception as exc:
                return self._json(200, {"estimate": True, "partial": True, "watts": None, "error": str(exc)})
            ids = list(CFG.node_ids)
            return self._json(200, fleet_power(card.get("hosts") or [], [i for i in ids if i]))
        if path == "/api/telemetry":
            role = (qs.get("role") or [None])[0] or None
            cursor = (qs.get("cursor") or [None])[0] or None
            raw_limit = (qs.get("limit") or ["30"])[0]
            try:
                payload = history_store().telemetry_index(role=role, cursor=cursor, limit=int(raw_limit))
            except (TypeError, ValueError, OverflowError):
                return self._json(400, {"error": "bad page"})
            return self._json(200, payload)
        if path == "/api/history":
            machine = (qs.get("machine") or [""])[0]
            window = (qs.get("range") or ["24h"])[0]
            if not machine:
                return self._json(400, {"error": "missing machine"})
            try:
                start, end = _window_args(qs)
                since = _optional_stamp(qs, "since")
                payload = history_store().query(machine, window, start=start, end=end, since=since)
            except (ValueError, OverflowError):
                return self._json(400, {"error": "bad range"})
            return self._json(200, payload)
        try:
            cat = load_catalog()
            try:
                truth = load_yaml(model_inventory_path())
            except Exception as e:
                truth = None
                truth_err = str(e)
            else:
                truth_err = None
            if path == "/api/nodes":
                if STATE.readonly:
                    return self._json(200, {"nodes": [], "stale": True})
                return self._json(200, _node_read(cat))
            if path == "/api/status":
                if STATE.readonly:
                    report = {"nodes": [], "stale": True}
                else:
                    _docker_sample(wait=0)  # start the ~2 s sample beside the node read
                    report = _node_read(cat)
                return self._json(200, assemble_status(cat, truth, STATE.lock, truth_err=truth_err, nodes=report, docker_fn=_docker_fn))
            if path == "/api/inventory":
                return self._json(200, assemble_inventory(cat, truth, truth_err=truth_err))
            if path == "/api/wanted":
                return self._json(200, _wanted_payload(cat, STATE.lock, truth, truth_err))
            if path == "/api/unmanaged":
                if STATE.readonly:
                    return self._json(200, {"items": [], "enabled": False, "checked": True})
                items, checked = _unmanaged_items(active_runner(cat), cached=True, cat=cat)
                return self._json(200, {"items": items, "enabled": CFG.controller.get("stop_unmanaged") is True, "checked": checked})
            if path == "/api/monitor":
                fn = STATE.monitor_fn or current_dash
                card = fn()
                if isinstance(card, dict):
                    # One fresh sample per machine. A stale tip stays in the hour, not here.
                    card = dict(card)
                    card["history"] = history_store().latest_samples(fresh_secs=LIVE_FRESH_SECS)
                    # This controller's own monitored hosts (never a mirrored controller's).
                    card["monitored"] = monitor_hosts.view()
                return self._json(200, card)
            if path == "/api/tokens":
                return self._json(200, {"models": token_report(cat, sample=False)})
            if path == "/api/token-day":
                return self._json(200, day_view())
            if path == "/api/token-life":
                return self._json(200, cached_lifetime())
            if path == "/api/models":
                if STATE.readonly:
                    return self._json(200, list_model_cards(cat, None))
                runner = active_runner(cat)
                cards = _read("models", cat, runner, lambda: list_model_cards(cat, CachedReads(runner)))
                if cards is PENDING:
                    cards = list_model_cards(cat, None)  # the catalog view until the nodes answer
                return self._json(200, cards)
            named = _named_route(path)
            if named:
                if STATE.readonly:
                    return self._json(403, {"error": "read-only stage"})
                if not named[1] and not named[2]:
                    return self._json(*list_named(named[0]))
                if named[1] and not named[2]:
                    return self._json(*get_named(named[0], named[1]))
                return self._json(404, {"error": "not found"})
            model_hit = _model_suffix(path)
            if model_hit and model_hit[1] == "recipe":
                if STATE.readonly:
                    return self._json(403, {"error": "read-only stage"})
                query = {}
                for key, values in qs.items():
                    if not values:
                        continue
                    if key in ("extra_flags", "extra_args", "extra_args_worker", "env", "env_worker"):
                        query[key] = [item for item in values if item] if key == "extra_flags" else values
                    else:
                        query[key] = values[0]
                name, bad = _pick_recipe(query)
                if bad:
                    return self._json(400, {"error": bad})
                with using_recipe(model_hit[0], name):
                    return self._json(*preview_recipe(cat, CachedReads(active_runner(cat)), model_hit[0], query))
            if path == "/api/runs":
                if STATE.readonly or STATE.runs_fn is not None:
                    mirrored = STATE.runs_fn() if STATE.runs_fn is not None else {"runs": [], "stale": True, "source": "unavailable"}
                    if not isinstance(mirrored, dict):
                        mirrored = {"runs": [], "stale": True, "source": "unavailable"}
                    if STATE.readonly:
                        return self._json(200, mirrored)
                    return self._json(200, _with_serving(mirrored))
                runner = active_runner(cat)
                rows = _read("runs", cat, runner, lambda: runs_index(CachedReads(runner), cat))
                if rows is PENDING:
                    rows = runs_index(None, cat)  # the recorded runs until the nodes answer
                return self._json(200, _with_serving(dict(rows)))
            run_hit = _run_id(path)
            if run_hit:
                if STATE.readonly:
                    return self._json(*run_one(run_hit, None))
                return self._json(*run_one(run_hit, active_runner(cat)))
            if path == "/api/fleet":
                if STATE.readonly:
                    return self._json(200, {"library": [], "machines": [], "hosts": [], "storage": [], "stale": True})
                view = _read("fleet", cat, STATE.runner, lambda: fleet(cat, STATE.runner), slow=True)
                if view is PENDING:
                    view = {"library": [], "machines": [], "hosts": [], "storage": [], "stale": True, "pending": True}
                return self._json(200, view)
            if path == "/api/network/layout":
                cfg_path = config_path()
                if cfg_path is None:
                    self._json(200, {"configured": False, "error": "no sparksi.toml found", "layout": None})
                    return None
                try:
                    with open(cfg_path, "rb") as fh:
                        raw = tomllib.load(fh)
                except (OSError, tomllib.TOMLDecodeError):
                    self._json(200, {"configured": True, "error": "could not read sparksi.toml", "layout": None})
                    return None
                restart_needed = (
                    raw.get("nodes") != CFG.raw.get("nodes")
                    or raw.get("endpoints", []) != CFG.raw.get("endpoints", [])
                )
                self._json(200, {
                    "configured": True,
                    "file": cfg_path.name,
                    "layout": netconfig.layout(raw, CFG.controller_name),
                    "restart_needed": restart_needed,
                })
                return None
            if path == "/api/options":
                return self._options_view()
            if path == "/api/discover":
                # The last scan only. A new scan is POST /api/discover.
                return self._json(200, discover.cached())
            if path == "/api/sync":
                # Read-only: compares each Spark with storage, copies nothing.
                if STATE.readonly:
                    return self._json(200, {"hosts": [], "rows": [], "stale": True})
                return self._json(200, sync_report(cat, STATE.runner))
            if path == "/api/job":
                return self._json(200, {"job": current_job()})
            if path == "/api/apply":
                spec = (qs.get("set") or [None])[0]
                plan, _, _ = _apply_plan(cat, STATE.lock, truth, truth_err, spec)
                return self._json(200, plan)
            if path == "/api/dry-run":
                sets = qs.get("set") or []
                if not sets:
                    return self._json(400, {"error": "missing set"})
                return self._json(200, run_dry_run(cat, sets[0], truth, truth_err=truth_err))
            return self._json(404, {"error": "not found"})
        except Exception as e:
            return self._json(500, {"error": str(e)})

    def _network_edit(self, cfg_path: Path, action: str, role: str, machine: dict[str, Any]) -> None:
        """Read sparksi.toml, apply one role change, check it, save it. Starts or stops nothing."""
        try:
            with open(cfg_path, "rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            self._json(409, {"error": "could not read sparksi.toml"})
            return None
        try:
            nodes, endpoints, note = netconfig.change(raw, action, role, machine)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return None
        new_raw = dict(raw)
        new_raw["nodes"] = nodes
        if endpoints:
            new_raw["endpoints"] = endpoints
        else:
            new_raw.pop("endpoints", None)
        try:
            Config(new_raw)
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
            return None
        try:
            saved = netconfig.save_tables(cfg_path, nodes, endpoints)
        except ValueError as exc:
            self._json(409, {"error": str(exc)})
            return None
        except OSError:
            logger().exception("could not write sparksi.toml")
            self._json(500, {"error": "could not write sparksi.toml"})
            return None
        logger().info("network config: %s %s %s -> %s", action, role, machine.get("name", ""), note)
        self._json(200, {
            "ok": True,
            "note": note,
            "changed": saved["changed"],
            "backup": saved["backup"],
            "restart_needed": True,
            "layout": netconfig.layout(new_raw, CFG.controller_name),
        })
        return None

    def _options_view(self) -> None:
        """The Options page's settings as the file has them, and which wait for a restart."""
        import os

        cfg_path = config_path()
        if cfg_path is None:
            return self._json(200, {"configured": False, "error": "no sparksi.toml found", "options": []})
        try:
            with open(cfg_path, "rb") as fh:
                raw = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError):
            return self._json(200, {"configured": True, "error": "could not read sparksi.toml", "options": []})
        return self._json(200, {
            "configured": True,
            "file": cfg_path.name,
            "options": options.current(raw, CFG.raw, dict(os.environ)),
        })

    def _options_post(self, body: dict[str, Any]) -> None:
        """Check and save one setting. Starts, stops or contacts nothing."""
        import os

        key = body.get("key")
        value = body.get("value")
        if not isinstance(key, str) or not isinstance(value, str):
            return self._json(400, {"error": "key and value are required"})
        cfg_path = config_path()
        if cfg_path is None:
            return self._json(409, {"error": "no sparksi.toml to edit"})
        if not _NET_WRITE.acquire(timeout=10):
            return self._json(409, {"error": "another config edit is in progress"})
        try:
            try:
                with open(cfg_path, "rb") as fh:
                    raw = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError):
                return self._json(409, {"error": "could not read sparksi.toml"})
            # Only the port this process really listens on (sparksi serve --port); the file's ui_port does not pick it.
            ports = (int(self.server.server_address[1]),)
            try:
                clean = options.check(key, value, raw, ports, tuple(HOSTS.names()))
                Config(options.with_value(raw, key, clean))
                saved = options.save(cfg_path, key, clean)
            except ValueError as exc:
                return self._json(400, {"error": str(exc)})
            except OSError:
                logger().exception("could not write sparksi.toml")
                return self._json(500, {"error": "could not write sparksi.toml"})
            with open(cfg_path, "rb") as fh:
                raw = tomllib.load(fh)
        finally:
            _NET_WRITE.release()
        logger().info("options: %s changed=%s", key, saved["changed"])
        return self._json(200, {
            "ok": True,
            "key": key,
            "changed": saved["changed"],
            "backup": saved["backup"],
            "options": options.current(raw, CFG.raw, dict(os.environ)),
        })

    def _unmanaged_stop(self, body: dict[str, Any], token: str) -> None:
        """Stop one model Sparksi did not start: lock, setting, a fresh match, then the node checks again."""
        from sparksi.config import CFG

        if load_in_progress():
            return self._json(409, {"error": "still loading; stop does not interrupt a load", "stopped": False})
        if bool(body.get("from_ui")):
            if STATE.lock.read().locked:
                return self._json(403, {"error": "Unlock first, then Stop.", "stopped": False})
        else:
            ok, why = STATE.lock.allow_api_write(token)
            if not ok:
                return self._json(403, {"error": why, "stopped": False})
        if CFG.controller.get("stop_unmanaged") is not True:
            return self._json(403, {"error": "Stopping models started outside Sparksi is off on this controller.", "stopped": False})
        host = str(body.get("host") or "")
        container = str(body.get("container") or "")
        try:
            port = int(body.get("port"))
        except (TypeError, ValueError):
            return self._json(400, {"error": "bad port", "stopped": False})
        if protect.is_off_limits(port) or not host or not container:
            return self._json(400, {"error": "nothing to stop", "stopped": False})
        cat = load_catalog()
        runner = active_runner(cat)
        from sparksi import unmanaged

        items, _checked = _unmanaged_items(runner, cached=False, cat=cat)
        item = unmanaged.same_item(items, host, port, container)
        if item is None or not item.get("stoppable"):
            return self._json(409, {"error": "That model changed or is not stoppable now; reload and check again.", "stopped": False})
        outside = _external_container_refusal(cat, runner, host, container)
        if outside:
            return self._json(409, {"error": outside, "stopped": False, "external": True})
        result = runner.invoke(host, {"op": "stop_unmanaged", "container": container, "port": port}, timeout=90)
        if not isinstance(result, dict):
            result = {"ok": False, "error": "no answer"}
        stopped = bool(result.get("stopped"))
        detail = str(result.get("error") or "")
        label = str(item.get("label") or container)
        note_activity("stop", f"stop outside model {label} port {port}" + ("" if stopped else " failed: " + detail[-200:]), machine=host)
        logger().info("stop unmanaged %s on %s port %s -> %s %s", container, host, port, stopped, detail[-200:])
        READS.expire()
        return self._json(200 if stopped else 500, {"stopped": stopped, "error": detail, "host": host, "port": port, "container": container})

    def _outside_stop(self, body: dict[str, Any], token: str) -> None:
        """Emergency stop of a server Sparksi did not start, container or bare process.

        Works with the lock on: the page asks for a confirmation naming the model, Spark
        and port, and the body must echo all three. API callers without the page still
        need the unlock token. The setting stop_unmanaged is not used.
        """
        if not bool(body.get("from_ui")):
            ok, why = STATE.lock.allow_api_write(token)
            if not ok:
                return self._json(403, {"error": why, "stopped": False})
        host = str(body.get("host") or "")
        container = str(body.get("container") or "") or None
        try:
            port = int(body.get("port"))
        except (TypeError, ValueError):
            return self._json(400, {"error": "bad port", "stopped": False})
        if not host or protect.is_off_limits(port):
            return self._json(400, {"error": "nothing to stop", "stopped": False})
        if not _OUTSIDE_STOP.acquire(blocking=False):
            return self._json(409, {"error": "Another emergency stop is running; wait for it to finish.", "stopped": False})
        try:
            return self._outside_stop_locked(body, host, container, port)
        finally:
            _OUTSIDE_STOP.release()

    def _outside_stop_locked(self, body: dict[str, Any], host: str, container: str | None, port: int) -> None:
        from sparksi import unmanaged

        cat = load_catalog()
        runner = active_runner(cat)
        items, _checked = _unmanaged_items(runner, cached=False, cat=cat)
        item = unmanaged.same_item(items, host, port, container)
        if item is None or not item.get("emergency"):
            return self._json(409, {"error": "That server changed or cannot be stopped from here; reload and check again.", "stopped": False})
        label = str(item.get("label") or "")
        spark = str(item.get("host_name") or host)
        confirm = body.get("confirm") if isinstance(body.get("confirm"), dict) else {}
        if confirm.get("model") != label or confirm.get("spark") != spark or str(confirm.get("port")) != str(port):
            return self._json(400, {"error": "Confirm the stop naming the model, Spark and port.", "stopped": False})
        req: dict[str, Any] = {"op": "stop_outside", "port": port, "keep": sorted(_outside_keep(cat))}
        if container:
            req["container"] = container
        result = runner.invoke(host, req, timeout=90)
        if not isinstance(result, dict):
            result = {"ok": False, "error": "no answer"}
        stopped = bool(result.get("stopped"))
        detail = str(result.get("error") or "")
        what = f"emergency stop {label} on {spark} port {port}"
        if stopped:
            freed = int(result.get("gpu_freed_mib") or 0)
            gpu = f", GPU freed {freed} MiB" if result.get("gpu_checked") else ", GPU not checked"
            note_activity("stop", f"{what}: stopped with {result.get('signal') or 'TERM'}, port closed{gpu}", machine=host)
        else:
            note_activity("stop", f"{what} failed: {detail[-200:]}", machine=host)
        logger().info("emergency stop %s on %s port %s -> %s %s", label, host, port, stopped, detail[-200:])
        READS.expire()
        payload = {
            "stopped": stopped, "error": detail, "host": host, "port": port, "container": container,
            "label": label, "spark": spark,
        }
        for key in ("signal", "port_closed", "gpu_checked", "gpu_freed_mib", "gpu_pids_left", "command", "pids"):
            if key in result:
                payload[key] = result[key]
        return self._json(200 if stopped else 502, payload)

    def _set_gpu_clocks(self, node_key: str, body: dict[str, Any]) -> None:
        """Apply or clear a per-Spark GPU max-clock cap with sudo -n only; no persistence.

        This action does not use the lock. Same-origin JSON is still required
        (``_config_write_ok``). A running or decoding model needs confirm_running.
        """
        from sparksi.gpu_clocks import apply_clock, begin_save, end_save, parse_request_mhz

        cat = load_catalog()
        spec = None
        for item in node_specs(cat):
            if node_key in (item["id"], item["name"], item["role"], item["key"]):
                spec = item
                break
        if spec is None:
            return self._json(404, {"error": "Unknown Spark.", "ok": False})
        max_mhz, why = parse_request_mhz(body)
        if why:
            return self._json(400, {"error": why, "ok": False, "id": spec["id"], "role": spec["role"]})
        if body.get("confirm_running") is not True:
            busy = _gpu_clock_busy(spec)
            if busy:
                return self._json(409, {
                    "ok": False,
                    "needs_confirm": True,
                    "error": busy,
                    "id": spec["id"],
                    "role": spec["role"],
                })
        if not begin_save(spec["key"]):
            return self._json(409, {
                "ok": False,
                "error": "A clock change is already running on this Spark. Wait for it to finish.",
                "id": spec["id"],
                "role": spec["role"],
            })
        try:
            runner = active_runner(cat)
            result = apply_clock(runner, spec["key"], max_mhz)
        finally:
            end_save(spec["key"])
        gpu = result.get("gpu") or {}
        payload = {
            "ok": bool(result.get("ok")),
            "id": spec["id"],
            "name": spec["name"],
            "role": spec["role"],
            "gpu": gpu,
        }
        if result.get("error"):
            payload["error"] = result["error"]
        if result.get("ok") or result.get("kind") == "failed":
            # A failure after some GPU took the change still moved the Spark: refresh reads.
            invalidate_cache()
            READS.expire()
            SLOW_READS.expire()
        if result.get("ok"):
            if max_mhz is None:
                note_activity("gpu-clock", f"GPU clock cap cleared on {spec['id']}")
            else:
                note_activity("gpu-clock", f"GPU max clock {max_mhz} MHz on {spec['id']}")
            return self._json(200, payload)
        # Permission and nvidia-smi failures are shown as written so the page can quote them.
        status = 400 if result.get("kind") == "invalid" else 502
        return self._json(status, payload)

    def do_POST(self) -> None:  # noqa: N802
        if STATE.readonly:
            return self._json(403, {"error": "read-only stage"})
        path = urlparse(self.path).path.rstrip("/")
        try:
            if not _config_write_ok(self):
                return None
            body = self._body_json()
            token = _read_token(self, body)
            node_key = _gpu_clocks_node(path)
            if node_key:
                return self._set_gpu_clocks(node_key, body)
            if path == "/api/lock":
                data = STATE.lock.acquire("api").to_dict()
                note_activity("lock", "Lock on")
                return self._json(200, data)
            if path == "/api/unlock":
                # Human on this page: no token. CLI/API still uses the key.
                # Lock file only — never probe models here (that made the UI lag).
                from_ui = bool(body.get("from_ui"))
                st = STATE.lock.read()
                if from_ui:
                    st2, _new_tok = STATE.lock.release()
                    data = st2.to_dict()
                    data["auth"] = "web-ui"
                    note_activity("unlock", "Lock off")
                    return self._json(200, data)
                if st.holder:
                    confirm = body.get("confirm_holder")
                    if confirm != st.holder:
                        return self._json(
                            409,
                            {
                                "error": (
                                    f"would displace holder {st.holder}; "
                                    "confirm_holder must match"
                                ),
                                "holder": st.holder,
                            },
                        )
                ok, why = STATE.lock.allow_api_write(token)
                if not ok:
                    return self._json(403, {"error": why, "hint": "CLI unlock writes unlock.token"})
                st2, _new_tok = STATE.lock.release()
                data = st2.to_dict()
                data["auth"] = why
                note_activity("unlock", "Lock off")
                return self._json(200, data)
            if path == "/api/apply":
                code, payload = post_apply(body, token)
                if payload.get("executed"):
                    note_activity("apply", "Apply committed")
                elif code >= 400:
                    note_activity("apply", "Apply refused")
                return self._json(code, payload)
            if path == "/api/unmanaged/stop":
                return self._unmanaged_stop(body, token)
            if path == "/api/outside/stop":
                return self._outside_stop(body, token)
            if path == "/api/stop":
                if load_in_progress():
                    return self._json(409, {"error": "still loading; stop does not interrupt a load", "stopped": False})
                from_ui = bool(body.get("from_ui"))
                if from_ui:
                    if STATE.lock.read().locked:
                        return self._json(403, {"error": "Unlock first, then Stop."})
                else:
                    ok, why = STATE.lock.allow_api_write(token)
                    if not ok:
                        return self._json(403, {"error": why})
                rid = body.get("id") or body.get("resident")
                if not rid:
                    return self._json(400, {"error": "missing id"})
                try:
                    result = stop_id(str(rid))
                except CatalogError as e:
                    return self._json(400, {"error": str(e), "stopped": False})
                code = 200 if result.get("stopped") else 500
                if result.get("stopped"):
                    note_activity("stop", "unload " + str(rid) + " phase=ready")
                    logger().info("unload %s phase=ready", rid)
                else:
                    detail = str(result.get("detail") or result.get("error") or "stop failed")
                    note_activity("stop", "unload " + str(rid) + " phase=failed error=" + detail[-400:])
                    logger().info("unload %s phase=failed error=%s", rid, detail[-400:])
                return self._json(code, result)
            if path == "/api/switch/plan":
                self._read_only_post = True
                cat = load_catalog()
                rid = str(body.get("id") or "")
                snap = fleet(cat, STATE.runner)
                target = row_by_id(snap["library"], rid)
                if not target:
                    return self._json(404, {"error": "unknown harness"})
                locked = STATE.lock.read().locked
                return self._json(200, plan_switch(
                    cat, target, snap["library"], lock_on=locked, host=str(body.get("host") or ""),
                    loaded=monitor_loaded(),
                ))
            if path == "/api/options":
                # Settings, not model actions: same-origin JSON (checked above), never the model lock.
                return self._options_post(body)
            if path == "/api/network/machine":
                if not _writes_allowed(self, body, token):
                    return None
                machine = body.get("machine")
                action = body.get("action")
                role = body.get("role")
                if not isinstance(machine, dict) or not isinstance(action, str) or not isinstance(role, str):
                    self._json(400, {"error": "machine, action and role are required"})
                    return None
                cfg_path = config_path()
                if cfg_path is None:
                    self._json(409, {"error": "no sparksi.toml to edit"})
                    return None
                if not _NET_WRITE.acquire(timeout=10):
                    self._json(409, {"error": "another config edit is in progress"})
                    return None
                try:
                    return self._network_edit(cfg_path, action, role, machine)
                finally:
                    _NET_WRITE.release()
            if path == "/api/discover":
                # Read-only probing from this machine; nothing runs on the Sparks. No lock needed.
                # The scan runs in the background; the page polls GET /api/discover for it.
                self._read_only_post = True
                return self._json(*discover.start(_known_sparks()))
            if path == "/api/job/pause":
                job = request_pause(bool(body.get("paused", True)))
                note_activity("pause", "Job paused")
                return self._json(200, {"job": job})
            if path == "/api/job/cancel":
                job = request_cancel()
                note_activity("cancel", "Job canceled")
                return self._json(200, {"job": job})
            if path in ("/api/library/stage", "/api/library/drop", "/api/library/run", "/api/library/stop", "/api/switch"):
                return self._library_action(body, token, "switch" if path == "/api/switch" else path.rsplit("/", 1)[-1])
            named = _named_route(path)
            if named:
                # Recipe edits are config, not model actions: same-origin JSON (checked above), no lock.
                cat = load_catalog()
                model_id, name, action = named
                if not name and not action:
                    return self._json(*save_as(model_id, body, active_runner(cat)))
                if name and action == "default":
                    return self._json(*set_default(model_id, name))
                if name and action == "rename":
                    return self._json(*rename_named(model_id, name, body))
                return self._json(404, {"error": "not found"})
            model_hit = _model_suffix(path)
            if model_hit and model_hit[1] == "service":
                if not _writes_allowed(self, body, token):
                    return None
                cat = load_catalog()
                return self._json(*service_action(cat, active_runner(cat), model_hit[0], str(body.get("action") or "")))
            if model_hit and model_hit[1] == "plan":
                self._read_only_post = True  # a plan or preview reads; it never changes what reads show
                # The Models page previews a pinned recipe through the plan. A refused
                # plan is a good preview answer (200 with verdict REFUSE), and a preview
                # may use recent node facts. Dry run and Load send no flag: fresh facts,
                # and 400 on REFUSE as before.
                preview = body.pop("preview", False) is True
                name, bad = _pick_recipe(body)
                if bad:
                    return self._json(400, {"error": bad})
                cat = load_catalog()
                runner = active_runner(cat)
                with using_recipe(model_hit[0], name):
                    code, payload = plan_model(cat, CachedReads(runner) if preview else runner, model_hit[0], body)
                if preview and code == 400 and isinstance(payload, dict) and payload.get("verdict") == "REFUSE":
                    code = 200
                return self._json(code, payload)
            if path == "/api/runs":
                if not _writes_allowed(self, body, token):
                    return None
                name, bad = _pick_recipe(body)
                if bad:
                    return self._json(400, {"error": bad})
                cat = load_catalog()
                with using_recipe(str(body.get("id") or body.get("model_id") or ""), name):
                    return self._json(*begin_run(cat, active_runner(cat), body))
            stop_id_hit = _run_stop_id(path)
            if stop_id_hit:
                if not _writes_allowed(self, body, token):
                    return None
                cat = load_catalog()
                return self._json(*halt_run(cat, active_runner(cat), stop_id_hit))
            if path == "/api/wanted":
                ok, why = STATE.lock.allow_api_write(token)
                if not ok:
                    return self._json(403, {"error": why})
                spec = body.get("set") or body.get("spec")
                if not spec:
                    return self._json(400, {"error": "missing set"})
                cat = load_catalog()
                try:
                    truth = load_yaml(model_inventory_path())
                    terr = None
                except Exception as e:
                    truth, terr = None, str(e)
                try:
                    dry = run_dry_run(cat, str(spec), truth, truth_err=terr)
                except Exception as e:
                    dry = {
                        "set": spec,
                        "verdict": "UNCERTAIN",
                        "reasons": [str(e)],
                        "needed_gib": None,
                    }
                payload = apply_set(cat, str(spec), dry)
                return self._json(
                    200,
                    {
                        "wanted": payload,
                        "dry_run": dry,
                        "auth": why,
                        "note": "desired state only; no load/evict/apply",
                    },
                )
            return self._json(404, {"error": "not found"})
        except Exception as e:
            logger().exception("POST %s failed", path)
            return self._json(500, {"error": str(e)})

    def do_PUT(self) -> None:  # noqa: N802
        if STATE.readonly:
            return self._json(403, {"error": "read-only stage"})
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._body_json()
            named = _named_route(path)
            if named and named[1] and not named[2]:
                if not _config_write_ok(self):
                    return None
                cat = load_catalog()
                return self._json(*save_named(named[0], named[1], body, active_runner(cat)))
            hit = _model_suffix(path)
            if not hit or hit[1] != "recipe":
                return self._json(404, {"error": "not found"})
            if not _config_write_ok(self):
                return None
            refused = pinned_default_refusal(hit[0])
            if refused is not None:
                return self._json(*refused)
            cat = load_catalog()
            return self._json(*write_recipe(hit[0], body, active_runner(cat)))
        except Exception as exc:
            logger().exception("PUT %s failed", path)
            return self._json(500, {"error": str(exc)})

    def do_DELETE(self) -> None:  # noqa: N802
        if STATE.readonly:
            return self._json(403, {"error": "read-only stage"})
        path = urlparse(self.path).path.rstrip("/")
        try:
            self._body_json()
            named = _named_route(path)
            if named and named[1] and not named[2]:
                if not _config_write_ok(self):
                    return None
                return self._json(*delete_named(named[0], named[1]))
            self.send_error(501, "Unsupported method ('DELETE')")
            return None
        except Exception as exc:
            logger().exception("DELETE %s failed", path)
            return self._json(500, {"error": str(exc)})

    def _library_action(self, body: dict[str, Any], token: str | None, kind: str) -> None:
        from_ui = bool(body.get("from_ui"))
        # Copy and delete are a transfer queue. Run and stop still need the GPU lock.
        if kind in ("run", "stop", "switch"):
            if from_ui:
                if STATE.lock.read().locked:
                    return self._json(403, {"error": "Unlock first."})
            else:
                ok, why = STATE.lock.allow_api_write(token)
                if not ok:
                    return self._json(403, {"error": why})
        cat = load_catalog()
        rid = str(body.get("id") or "")
        if not rid:
            return self._json(400, {"error": "missing id"})
        runner = STATE.runner
        if kind == "stage":
            hosts = body.get("hosts") or ([body["host"]] if body.get("host") else [])
            if not hosts:
                return self._json(400, {"error": "missing host"})
            refill = bool(body.get("refill"))
            result = start_stage(cat, rid, [str(h) for h in hosts], lock_on=False, runner=runner, refill=refill)
        elif kind == "drop":
            if not body.get("host") or not body.get("confirm"):
                return self._json(400, {"error": "missing host or confirm"})
            result = start_drop(
                cat, rid, str(body["host"]), str(body["confirm"]), lock_on=False, runner=runner
            )
        elif kind == "run":
            result = start_run(cat, rid, lock_on=False, runner=runner)
        elif kind == "switch":
            result = start_switch(cat, rid, lock_on=False, host=str(body.get("host") or ""), runner=runner)
        else:
            result = start_stop(cat, rid, lock_on=False, runner=runner)
        code = 200 if result.get("verdict") in ("STARTED", "READY") else 409
        if code == 200 and kind == "stage":
            note_activity("copy", "Copy")
        elif code == 200 and kind == "drop":
            note_activity("delete", "Delete")
        return self._json(code, result)


class DualServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    # The page polls about ten endpoints at once; the default backlog of 5 drops
    # connections under a burst.
    request_queue_size = 128


def bind_listeners(port: int, hosts: list[str], factory) -> tuple[list, list[str]]:
    """Bind each host. Loopback failure is fatal. Other failures stay pending."""
    servers: list = []
    pending: list[str] = []
    for host in hosts:
        try:
            servers.append(factory(host, port))
        except OSError as exc:
            if host == "127.0.0.1":
                raise
            print(f"sparksi serve: skip {host}:{port} ({exc})", flush=True)
            pending.append(host)
        else:
            print(f"sparksi serve {host}:{port}", flush=True)
    return servers, pending


# Listeners for the running process. A test shuts these down from another thread.
LISTENERS: list = []
# Set when serve() returns so the background polls stop with this process.
_STOP = threading.Event()


def _start_mirror(own_port: int) -> None:
    """Start the optional read-only monitor mirror. A bad setting is logged and ignored."""
    from sparksi import monitor_mirror

    raw = monitor_mirror.configured_upstream()
    if not raw or STATE.monitor_fn is not None:
        return
    try:
        base = monitor_mirror.upstream_url(raw)
        monitor_mirror.refuse_self(base, own_port, HOSTS.names())
    except ValueError as exc:
        logger().warning("monitor mirror off: %s", exc)
        return
    mirror = monitor_mirror.Mirror(base, current_dash)
    STATE.monitor_fn = mirror.card
    STATE.observe_fn = mirror.card
    STATE.mirror = mirror.start()
    logger().info("monitor mirror: GET %s%s every %.0f s", base, monitor_mirror.PATH, monitor_mirror.POLL_S)


def serve(
    port: int = DEFAULT_PORT,
    hosts: list[str] | None = None,
    *,
    announce: bool = True,
    readonly: bool = False,
) -> None:
    # Re-read the site file and push each node's mount roots into the in-process
    # checks. Import-time defaults are not the configured list.
    _STOP.clear()
    reload_config()
    allowed = HOSTS.refresh()
    logger().info("host allowlist: %d names and addresses", len(allowed))
    model_inventory_path()  # resolves now so a legacy env name is noted below
    for text in deprecation_notes():
        logger().warning("config: %s", text)
    protect.log_auto(logger())
    catalog = load_catalog()
    extra = list((catalog.get("host") or {}).get("reserved_ports") or [])
    assert_serve_port(int(port), extra)
    bind = listen_targets(hosts, catalog)
    assert_bind_allowed(bind, list((catalog.get("host") or {}).get("bind_deny") or []))
    STATE.lock = Lock()
    STATE.catalog = catalog
    STATE.readonly = bool(readonly)
    if not readonly:
        try:
            restore_job(STATE.runner, check=True, resume=True)
        except Exception:
            logger().exception("could not restore the in-flight job")

    def _factory(host: str, listen_port: int) -> DualServer:
        return DualServer((host, listen_port), Handler)

    servers, pending = bind_listeners(port, bind, _factory)
    if not servers:
        raise Refused("no bind address succeeded")
    LISTENERS[:] = servers

    def _token_loop() -> None:
        while not _STOP.is_set():
            try:
                poll_once()
            except Exception:
                pass
            _STOP.wait(15)

    def _warm_models() -> None:
        try:
            list_models(STATE.catalog)
        except Exception:
            pass

    def _rebind() -> None:
        left = list(pending)
        while left and not _STOP.is_set():
            _STOP.wait(2)
            still: list[str] = []
            for host in left:
                try:
                    srv = _factory(host, port)
                except OSError as exc:
                    print(f"sparksi serve: skip {host}:{port} ({exc})", flush=True)
                    still.append(host)
                    continue
                print(f"sparksi serve {host}:{port}", flush=True)
                threading.Thread(target=srv.serve_forever, name=f"sparksi-{host}", daemon=True).start()
                servers.append(srv)
            left = still

    if pending and not readonly:
        threading.Thread(target=_rebind, name="sparksi-rebind", daemon=True).start()

    start_announce(announce and not readonly)
    _start_mirror(int(port))
    if not readonly:
        monitor_hosts.start(list(CFG.raw.get("endpoints") or []), _STOP)
        threading.Thread(target=_token_loop, name="sparksi-tokens", daemon=True).start()
        threading.Thread(target=_warm_models, name="sparksi-warm-models", daemon=True).start()
    def _history_loop() -> None:
        import time

        while not _STOP.is_set():
            try:
                sample_card(collector_card(), history_store(), activity_store())
            except Exception:
                pass
            time.sleep(1)

    threading.Thread(target=_history_loop, name="sparksi-history", daemon=True).start()
    try:
        extras = servers[1:]
        for srv in extras:
            threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers[0].serve_forever()
    except KeyboardInterrupt:
        for srv in servers:
            srv.shutdown()
    finally:
        _STOP.set()
        if STATE.mirror is not None:
            STATE.mirror.stop.set()
