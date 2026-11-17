# SPDX-License-Identifier: Apache-2.0

"""Apply/swap kernel. Mutates only when safe. Never stop a service a protection preset covers."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from typing import Any, Callable

from sparksi import protect
from sparksi.catalog import CatalogError, core_ids, core_recipe_names, recipe_for, resident_by_id
from sparksi.lock import Lock, LockedError, state_dir
from sparksi.never import FORBIDDEN_INVOKE, Refused
from sparksi.wanted import load_wanted, resolve_spec, seat_view

# The core LLM stack is the catalog recipe marked `core: true` (see catalog.core_ids).

Runner = Callable[[list[str]], int]


class ApplyError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def plan_path():
    return state_dir() / "apply-plan.json"


def announce_path():
    return state_dir() / "oom-guard-announce.json"


def is_protected(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    rid = str(row.get("id") or "").lower()
    kind = str(row.get("kind") or "").lower()
    seat = str(row.get("seat") or row.get("role") or "").lower()
    port = row.get("port")
    if port is not None and int(port) in protect.off_limits_ports():
        return True
    if protect.is_protected_id(rid) or protect.is_video_kind(kind) or protect.is_video_seat(seat):
        return True
    return False


def live_ids(residents: list[dict[str, Any]]) -> set[str]:
    return {r["id"] for r in residents if r.get("resident") and r.get("id")}


def protected_live(residents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in residents if r.get("resident") and is_protected(r)]


def union_spec(wanted: dict[str, Any] | None, residents: list[dict[str, Any]]) -> str | None:
    """Selected picks plus live models (Apply does not drop live)."""
    members = set((wanted or {}).get("members") or [])
    for r in residents or []:
        if (
            r.get("resident")
            and (r.get("kind") in ("vllm", "llamacpp") or protect.is_video_kind(r.get("kind")))
            and r.get("id")
        ):
            members.add(r["id"])
    if members:
        return ",".join(sorted(members))
    return (wanted or {}).get("name")


def _row(catalog: dict[str, Any], rid: str, residents: list[dict[str, Any]]) -> dict[str, Any]:
    for r in residents:
        if r.get("id") == rid:
            return r
    return resident_by_id(catalog, rid) or {"id": rid}


def build_plan(
    catalog: dict[str, Any],
    wanted: dict[str, Any] | None,
    residents: list[dict[str, Any]],
    dry: dict[str, Any] | None,
    *,
    lock_on: bool = True,
) -> dict[str, Any]:
    live = live_ids(residents)
    members = list((wanted or {}).get("members") or [])
    if not wanted or not members:
        plan = _base(
            catalog, wanted, residents, dry, [], [], "REFUSE", lock_on,
            ["no wanted set"],
        )
        return plan

    wanted_set = set(members)
    prot = protected_live(residents)
    wanted_satisfied = wanted_set <= live

    # Additive: Apply starts missing picks. It does not drop live models
    # (use Stop for that). Unchecked residents stay.
    starts_ids = sorted(wanted_set - live)
    victims: list[dict[str, Any]] = []
    starts = [_row(catalog, i, residents) for i in starts_ids]
    keep_ids = sorted(live - wanted_set)

    video_up = bool(prot)

    if wanted_satisfied:
        extra = f"; keeping live {keep_ids}" if keep_ids else ""
        return _base(
            catalog, wanted, residents, dry, [], [], "NOOP", lock_on,
            [f"already applied: selected members are live{extra}"],
        )

    if video_up and any(s.get("id") in core_ids(catalog) for s in starts):
        return _base(
            catalog, wanted, residents, dry, victims, starts, "REFUSE", lock_on,
            ["REFUSE: will not start the LLM stack while the protected video service is resident"],
        )

    for s in starts:
        if is_protected(s) and not s.get("unit"):
            return _base(
                catalog, wanted, residents, dry, victims, starts, "REFUSE", lock_on,
                [
                    f"REFUSE: no safe starter for {s.get('id')} "
                    "(will not invoke video-mode.sh)"
                ],
            )

    dry_v = (dry or {}).get("verdict")
    if dry_v == "NO":
        return _base(
            catalog, wanted, residents, dry, victims, starts, "REFUSE", lock_on,
            ["dry-run verdict NO — will not mutate"] + list((dry or {}).get("reasons") or []),
        )
    if dry_v == "UNCERTAIN" or dry_v is None:
        return _base(
            catalog, wanted, residents, dry, victims, starts, "UNCERTAIN", lock_on,
            ["dry-run UNCERTAIN — refuse --commit unless --force-uncertain"]
            + list((dry or {}).get("reasons") or []),
        )

    if dry_v != "OK":
        return _base(
            catalog, wanted, residents, dry, victims, starts, "UNCERTAIN", lock_on,
            [f"dry-run verdict {dry_v!r} is not OK"],
        )

    keep_note = f"keeping live {keep_ids}; " if keep_ids else ""
    return _base(
        catalog, wanted, residents, dry, victims, starts, "READY", lock_on,
        [f"plan READY: {keep_note}will start {starts_ids}; Stop is how you unload"],
    )


def _base(
    catalog: dict[str, Any],
    wanted: dict[str, Any] | None,
    residents: list[dict[str, Any]],
    dry: dict[str, Any] | None,
    victims: list[dict[str, Any]],
    starts: list[dict[str, Any]],
    verdict: str,
    lock_on: bool,
    reasons: list[str],
) -> dict[str, Any]:
    live = [r for r in residents if r.get("resident")]
    rec = (dry or {}).get("recipe") or {}
    floor = (dry or {}).get("floor") or {}
    return {
        "ts": _now(),
        "verdict": verdict,
        "wanted": {
            "name": (wanted or {}).get("name"),
            "spec": (wanted or {}).get("spec"),
            "members": (wanted or {}).get("members") or [],
            "seats": (wanted or {}).get("seats") or {},
        },
        "live": [
            {
                "id": r.get("id"),
                "seat": r.get("seat") or r.get("role"),
                "port": r.get("port"),
                "identity": r.get("identity"),
            }
            for r in live
        ],
        "seats": seat_view(catalog, residents, wanted),
        "floor": {
            "total_gib": floor.get("total_gib"),
            "complete": floor.get("complete"),
            "parts": [
                {
                    "name": p.get("name"),
                    "gib": p.get("gib"),
                    "unit": p.get("unit"),
                    "instrument": p.get("instrument"),
                    "confidence": p.get("confidence"),
                }
                for p in (floor.get("parts") or [])
            ],
        },
        "recipe": {
            "estimate_gib": rec.get("estimate_gib"),
            "unit": rec.get("unit") or "declared_vram_gib",
            "source": rec.get("source"),
            "confidence": rec.get("confidence"),
        },
        "margin_gib": (dry or {}).get("margin_gib"),
        "needed_gib": (dry or {}).get("needed_gib"),
        "mem_total_gib": (dry or {}).get("mem_total_gib"),
        "dry_verdict": (dry or {}).get("verdict"),
        "compare_note": (dry or {}).get("compare_note"),
        "victims": [_brief(v) for v in victims],
        "starts": [_brief(s) for s in starts],
        "keep": [
            {"id": r.get("id"), "port": r.get("port")}
            for r in live
            if r.get("id") not in {s.get("id") for s in starts}
        ],
        "lock_on": lock_on,
        "reasons": reasons,
        "mutating": False,
    }


def resident_unit(row: dict[str, Any]) -> str | None:
    u = row.get("unit")
    if isinstance(u, str):
        return u
    if isinstance(u, dict):
        return u.get("unit")
    return None


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row.get("id"),
        "seat": row.get("seat") or row.get("role"),
        "port": row.get("port"),
        "unit": resident_unit(row),
        "protected": is_protected(row),
    }


def save_plan(plan: dict[str, Any]) -> dict[str, Any]:
    p = plan_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    tmp.replace(p)
    return plan


def write_oom_announce(residents: list[dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    exempt = []
    for r in protected_live(residents):
        pids = []
        for a in r.get("gpu_apps") or []:
            if a.get("pid"):
                pids.append(a["pid"])
        exempt.append({"id": r.get("id"), "port": r.get("port"), "pids": pids})
    payload = {
        "ts": _now(),
        "window": "apply",
        "exempt": exempt,
        "apply_starts": [s.get("id") for s in plan.get("starts") or []],
        "note": "dgx-oom-guard has no apply handshake; announce only. sparksi is not a killer.",
    }
    path = announce_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


SYSTEMCTL = "/bin/systemctl"


CORE_STACK_UNIT = "vllm-coresident.service"


def systemctl_cmd(action: str, unit: str, *, no_block: bool = False) -> list[str]:
    """Passwordless sudo — the node user needs NOPASSWD for these units.

    Bare `systemctl start` as a user hits polkit and asks for a password.
    Oneshot stack loads use --no-block so Apply returns while models load.
    `restart` is required for vllm-coresident: Type=oneshot RemainAfterExit=yes
    so `start` is a no-op while the unit is already active(exited).
    """
    if action not in ("start", "stop", "restart"):
        raise ApplyError(f"refusing systemctl {action}")
    cmd = ["sudo", "-n", SYSTEMCTL, action]
    if no_block:
        cmd.append("--no-block")
    cmd.append(unit)
    return cmd


def assert_safe_mutate_argv(argv: list[str]) -> None:
    if not argv:
        raise Refused("empty mutate argv")
    prog = argv[0].rsplit("/", 1)[-1]
    joined = " ".join(argv).lower()
    if prog in FORBIDDEN_INVOKE or any(x in argv[0] for x in FORBIDDEN_INVOKE):
        raise Refused(f"refusing invoke {argv[0]}")
    if prog == "sudo":
        ok_sudo = (
            len(argv) >= 5
            and argv[1] == "-n"
            and "systemctl" in argv[2]
            and argv[3] in ("start", "stop", "restart")
        )
        if ok_sudo and argv[4] == "--no-block":
            ok_sudo = len(argv) >= 6
        if not ok_sudo:
            raise Refused(f"refusing sudo argv {argv}")
    if "video-mode" in joined or "tmux" in joined:
        raise Refused(f"refusing argv {argv}")
    if any(x in joined for x in protect.argv_substrings()) and any(
        a in argv for a in ("stop", "kill", "rm", "restart")
    ):
        raise Refused(f"refusing stop/kill targeting the protected video service: {argv}")


def _start_steps(catalog: dict[str, Any], plan: dict[str, Any], wanted: dict[str, Any] | None) -> list[list[str]]:
    steps: list[list[str]] = []
    name = (wanted or {}).get("name")
    rec = None
    if name:
        try:
            _, rec = recipe_for(catalog, name)
        except CatalogError:
            rec = None
    apply_unit = (rec or {}).get("apply_unit") or CORE_STACK_UNIT
    start_ids = {s.get("id") for s in (plan.get("starts") or [])}
    core = core_ids(catalog)
    core_starts = start_ids & core
    wanted_members = set((wanted or {}).get("members") or [])
    wanted_name = (wanted or {}).get("name")
    full_trio = bool(core) and (core <= wanted_members or wanted_name in core_recipe_names(catalog))
    # Two or more missing trio leaves: sequential orchestrator (not parallel).
    # Restart, not start: oneshot RemainAfterExit is already active(exited)
    # after a previous boot, so `start` is a silent no-op and nothing loads.
    if core_starts and (full_trio or len(core_starts) >= 2):
        steps.append(systemctl_cmd("restart", apply_unit, no_block=True))
        return steps
    # A single trio leaf on an empty (or already-partial) box: start that
    # unit only. Do not boot the whole stack for a one-model pick.
    if plan.get("starts") and core_starts:
        for s in plan.get("starts") or []:
            if s.get("protected"):
                raise ApplyError(f"will not start protected {s.get('id')}")
            unit = s.get("unit")
            if not unit:
                raise ApplyError(f"no unit to start for {s.get('id')}")
            steps.append(systemctl_cmd("start", unit, no_block=True))
        return steps
    for s in plan.get("starts") or []:
        if s.get("protected"):
            raise ApplyError(f"will not start protected {s.get('id')}")
        unit = s.get("unit")
        if not unit:
            raise ApplyError(f"no unit to start for {s.get('id')}")
        steps.append(systemctl_cmd("start", unit))
    return steps


def _stop_steps(plan: dict[str, Any]) -> list[list[str]]:
    steps: list[list[str]] = []
    for v in plan.get("victims") or []:
        if v.get("protected") or is_protected(v):
            raise ApplyError(f"will not stop protected {v.get('id')}")
        unit = v.get("unit")
        if not unit:
            continue
        steps.append(systemctl_cmd("stop", unit))
    return steps


def default_runner(argv: list[str]) -> int:
    assert_safe_mutate_argv(argv)
    p = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=180)
    return p.returncode


def commit(
    catalog: dict[str, Any],
    plan: dict[str, Any],
    residents: list[dict[str, Any]],
    lock: Lock,
    *,
    force_uncertain: bool = False,
    runner: Runner | None = None,
    wanted: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a plan. Never stops the protected video service. runner injectable for tests."""
    calls: list[list[str]] = []

    def run(argv: list[str]) -> int:
        assert_safe_mutate_argv(argv)
        calls.append(list(argv))
        impl = runner if runner is not None else default_runner
        return impl(argv)

    try:
        lock.require_unlocked("apply --commit")
    except LockedError as e:
        plan = {**plan, "verdict": "REFUSE", "reasons": plan.get("reasons", []) + [str(e)], "mutating": False}
        save_plan(plan)
        return {**plan, "executed": False, "calls": calls}

    verdict = plan.get("verdict")
    if verdict == "NOOP":
        save_plan({**plan, "executed": False, "already_applied": True})
        return {**plan, "executed": False, "already_applied": True, "calls": []}

    if verdict == "REFUSE":
        return {**plan, "executed": False, "calls": []}

    if verdict == "UNCERTAIN" and not force_uncertain:
        plan = {
            **plan,
            "reasons": list(plan.get("reasons") or []) + ["--commit refused UNCERTAIN (pass --force-uncertain)"],
            "executed": False,
        }
        save_plan(plan)
        return {**plan, "calls": []}

    if verdict not in ("READY", "UNCERTAIN"):
        return {**plan, "executed": False, "calls": []}

    if any(v.get("protected") for v in plan.get("victims") or []):
        plan = {**plan, "verdict": "REFUSE", "executed": False, "reasons": ["protected video service in victims at commit"]}
        save_plan(plan)
        return {**plan, "calls": []}

    write_oom_announce(residents, plan)
    started: list[list[str]] = []
    err: str | None = None
    try:
        for argv in _stop_steps(plan):
            run(argv)
        for argv in _start_steps(catalog, plan, wanted or load_wanted()):
            rc = run(argv)
            started.append(argv)
            if rc != 0:
                raise ApplyError(f"start failed {argv} rc={rc}")
    except Exception as e:
        err = str(e)
        # rollback only what this apply started
        for argv in reversed(started):
            if "start" in argv:
                unit = argv[-1]
                try:
                    run(systemctl_cmd("stop", unit))
                except Exception:
                    pass
        try:
            lock.acquire(reason="apply_failed")
        except Exception:
            pass
        w = load_wanted() or {}
        w["apply_failed"] = {"ts": _now(), "error": err}
        from sparksi.wanted import save_wanted

        if w.get("members"):
            save_wanted(w)
        out = {
            **plan,
            "verdict": "REFUSE",
            "executed": False,
            "apply_failed": err,
            "calls": calls,
            "reasons": list(plan.get("reasons") or []) + [err],
        }
        save_plan(out)
        return out

    try:
        lock.acquire(reason="apply_done")
    except Exception:
        pass
    out = {**plan, "executed": True, "calls": calls, "mutating": True}
    save_plan(out)
    return out


def format_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"sparksi apply  verdict={plan.get('verdict')}",
        f"  wanted: {(plan.get('wanted') or {}).get('name')}  members={(plan.get('wanted') or {}).get('members')}",
        f"  lock: {'ON' if plan.get('lock_on') else 'OFF'}",
    ]
    for s in plan.get("seats") or []:
        live = "up" if s.get("resident") else "down"
        lines.append(f"    {s.get('seat')} → {s.get('id')} :{s.get('port')}  {live}")
    floor = plan.get("floor") or {}
    lines.append(f"  floor: {floor.get('total_gib')} GiB  complete={floor.get('complete')}")
    for p in floor.get("parts") or []:
        lines.append(
            f"    {p.get('name')}: {p.get('gib')} GiB  {p.get('unit')}/{p.get('confidence')}"
        )
    rec = plan.get("recipe") or {}
    lines.append(
        f"  recipe: {rec.get('estimate_gib')} GiB {rec.get('unit')}  "
        f"needed={plan.get('needed_gib')}  total={plan.get('mem_total_gib')}"
    )
    if plan.get("compare_note"):
        lines.append(f"  compare: {plan.get('compare_note')}")
    keep = plan.get("keep") or []
    if keep:
        lines.append("  keep: " + ", ".join(k.get("id") or "?" for k in keep))
    vics = plan.get("victims") or []
    lines.append("  victims: " + (", ".join(v.get("id") or "?" for v in vics) if vics else "(none — Apply does not drop live models)"))
    sts = plan.get("starts") or []
    lines.append("  starts: " + (", ".join(s.get("id") or "?" for s in sts) if sts else "(none)"))
    if plan.get("already_applied"):
        lines.append("  already applied")
    for r in plan.get("reasons") or []:
        lines.append(f"    - {r}")
    return "\n".join(lines)


def wanted_for_plan(catalog: dict[str, Any], spec: str | None) -> dict[str, Any] | None:
    if spec:
        return resolve_spec(catalog, spec)
    return load_wanted()
