# SPDX-License-Identifier: Apache-2.0

"""Start a rendered recipe on the Sparks and poll health.

The node agent runs the argv. This process records the run, polls health, and
stops the other side when one side of a pair dies. Phases match the load jobs:
starting, waiting_health, ready, failed, timed_out, stopped.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sparksi import protect
from sparksi.config import ALL_ROLES, CFG, env
from sparksi.placement import LEGACY, is_multi, roles_of
from sparksi.evict import evict_argv, eviction_block, module_sha256
from sparksi.jobs import _audit, _health_verdict, _pair_dead, _probe_node
from sparksi.recipes.serve_port import other_serve_ports
from sparksi.never import launcher_refusal
from sparksi.recipes.memory import MEMORY_MARGIN_GIB, consider_peak, record_node_peak
from sparksi.recipes.pinned import _cache_roots, flag_on, recipe_allows_roce, roots_for
from sparksi.recipes.pinned import _under as _path_under

_LOCK = threading.RLock()
_RUNS: dict[str, dict[str, Any]] = {}
_THREADS: set[str] = set()
_RUN_LOCKS: dict[str, threading.Lock] = {}
_WRITER = threading.local()
_VIEW = threading.local()
# After a test reset, do not resurrect runs that were already written to disk.
_SKIP_LOAD = False
_TERMINAL = frozenset({"ready", "failed", "timed_out", "stopped", "cancelled"})
_STOPPED = frozenset({"failed", "timed_out", "stopped", "cancelled"})
# The monitor must not move a run out of a user stop, even if its probe is stale.
_USER_STOP = frozenset({"stopping", "stopped"})
_FAILURES = frozenset({"failed", "timed_out"})
# A ready server that then refuses the port this many times in a row has died.
REFUSED_PROBES = 3


def reset_for_tests() -> None:
    global _SKIP_LOAD
    with _LOCK:
        _RUNS.clear()
        _THREADS.clear()
        _RUN_LOCKS.clear()
        _SKIP_LOAD = True


def runs_dir() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "recipe-runs"
    return Path.home() / ".local" / "state" / "sparksi" / "recipe-runs"


def _path(run_id: str) -> Path:
    return runs_dir() / f"{run_id}.json"


def _persist(run: dict[str, Any]) -> None:
    try:
        path = _path(str(run["id"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(run), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return


def _load_all() -> None:
    if _SKIP_LOAD:
        return
    folder = runs_dir()
    if not folder.is_dir():
        return
    for path in folder.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("id"):
            _RUNS.setdefault(str(data["id"]), data)


def get_run(run_id: str) -> dict[str, Any] | None:
    with _LOCK:
        if not _RUNS:
            _load_all()
        run = _RUNS.get(run_id)
        return dict(run) if run else None


def list_runs() -> list[dict[str, Any]]:
    with _LOCK:
        if not _RUNS:
            _load_all()
        rows = [dict(item) for item in _RUNS.values()]
    extra = getattr(_VIEW, "stack", None) or []
    if extra and extra[-1]:
        have = {str(item.get("id") or "") for item in rows}
        for item in extra[-1]:
            if str(item.get("id") or "") in have:
                continue
            rows.append(dict(item))
    rows.sort(key=lambda item: float(item.get("started") or 0), reverse=True)
    return rows


def push_run_view(rows: list[dict[str, Any]]) -> None:
    """Show unsaved runs to planners. Nothing is written."""
    stack = getattr(_VIEW, "stack", None)
    if stack is None:
        stack = []
        _VIEW.stack = stack
    stack.append([dict(item) for item in rows])


def pop_run_view() -> None:
    stack = getattr(_VIEW, "stack", None) or []
    if stack:
        stack.pop()


def _run_lock(run_id: str) -> threading.Lock:
    with _LOCK:
        lock = _RUN_LOCKS.get(run_id)
        if lock is None:
            lock = threading.Lock()
            _RUN_LOCKS[run_id] = lock
        return lock


def _user_stop(record: dict[str, Any] | None) -> bool:
    """True once a user stop or cancel has been stored for this run."""
    if not isinstance(record, dict):
        return False
    if record.get("stop_requested"):
        return True
    phase = str(record.get("phase") or "")
    return phase in _USER_STOP or phase == "cancelled"


def _stopped_copy(current: dict[str, Any]) -> dict[str, Any]:
    """A stop record. Every node is stopped, including a side whose container is already gone."""
    run = dict(current)
    nodes = []
    for node in current.get("nodes") or []:
        if isinstance(node, dict):
            item = dict(node)
            item["phase"] = "stopped"
            nodes.append(item)
        else:
            nodes.append(node)
    run["nodes"] = nodes
    run["phase"] = "stopped"
    run["ok"] = True
    run["error"] = None
    run["failed_side"] = None
    run["stop_requested"] = True
    if not run.get("stopped_at"):
        run["stopped_at"] = time.time()
    return run


def _health_failure(run: dict[str, Any], incoming: str) -> bool:
    """A pair/health/timeout failure, not a stop command that was refused."""
    if incoming == "timed_out" or run.get("failed_side"):
        return True
    return bool(getattr(_WRITER, "monitor", False))


def _save(run: dict[str, Any]) -> dict[str, Any]:
    """Persist ``run`` unless that would turn a user stop into a failure.

    The health monitor re-reads under the per-run lock and compares the version
    it started from. A stop bumps the version and sets ``stop_requested``. A
    probe that began earlier cannot write ``waiting_health``, ``failed``, or
    ``timed_out`` back over that stop, even if a nested health pass cleared the
    monitor flag before this save.
    """
    run_id = str(run.get("id") or "")
    monitor = bool(getattr(_WRITER, "monitor", False))
    with _run_lock(run_id):
        with _LOCK:
            current = _RUNS.get(run_id)
            incoming = str(run.get("phase") or "")
            if (
                isinstance(current, dict)
                and incoming in _FAILURES
                and _user_stop(current)
                and str(current.get("phase") or "") != "cancelled"
                and _health_failure(run, incoming)
            ):
                run = _stopped_copy(current)
                incoming = "stopped"
            drop = False
            if monitor and isinstance(current, dict) and incoming not in _USER_STOP and incoming != "cancelled":
                if _user_stop(current) or int(current.get("version") or 0) != int(run.get("version") or 0):
                    drop = True
            if drop and isinstance(current, dict):
                snapshot = dict(current)
            else:
                version = int((current or {}).get("version") or 0) + 1
                run["version"] = version
                if incoming in _USER_STOP or incoming == "cancelled" or run.get("stop_requested"):
                    run["stop_requested"] = True
                stored = dict(run)
                _RUNS[run_id] = stored
                snapshot = dict(stored)
                drop = False
        if not drop:
            _persist(snapshot)
        return snapshot


def _public(run: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for node in run.get("nodes") or []:
        nodes.append({
            "role": node.get("role"),
            "host": node.get("host"),
            "phase": node.get("phase"),
            "log_tail": node.get("log_tail") or "",
            "evict_warning": node.get("evict_warning") or "",
            "container": node.get("container") or "",
        })
    return {
        "id": run.get("id"),
        "model_id": run.get("model_id"),
        "engine": run.get("engine"),
        "placement": run.get("placement"),
        "port": run.get("port"),
        "phase": run.get("phase"),
        "ok": run.get("ok"),
        "error": run.get("error"),
        "failed_side": run.get("failed_side"),
        "stopped_sides": list(run.get("stopped_sides") or []),
        "health": run.get("health") or {},
        "nodes": nodes,
        "experimental": bool(run.get("experimental")),
        "experimental_reason": run.get("experimental_reason") or "",
        "started": run.get("started"),
        "started_by": run.get("started_by") or "",
        "health_note": run.get("health_note") or "",
    }


def public_run(run: dict[str, Any] | None) -> dict[str, Any] | None:
    if not run:
        return None
    return _public(run)


class _PairRunner:
    """``_pair_dead`` stops a side with ``run_argv``. Send that through the agent."""

    def __init__(self, runner: Any, by_host: dict[str, str]):
        self.runner = runner
        self.by_host = by_host

    def invoke(self, host: str, req: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
        return self.runner.invoke(host, req, timeout=timeout)

    def run_argv(self, host: str, argv: list[str], timeout: int = 120) -> dict[str, Any]:
        blocked = launcher_refusal(argv)
        if blocked:
            return {"ok": False, "error": blocked}
        if any(protect.is_off_limits(part) for part in argv):
            return {"ok": False, "error": "port 8188 is off limits"}
        return self.runner.invoke(
            host,
            {"op": "stop", "run_id": self.by_host.get(host) or "", "argv": list(argv)},
            timeout=timeout,
        )


def _lead_role(run: dict[str, Any]) -> str:
    """The role that serves the port: "head" for the two-Spark placements, else the first Spark."""
    placement = str(run.get("placement") or "")
    if placement in LEGACY or not placement:
        return "head"
    covered = roles_of(placement, ALL_ROLES)
    return covered[0] if covered else "head"


def _multi(run: dict[str, Any]) -> bool:
    return str(run.get("placement") or "") == "both" or (
        str(run.get("placement") or "") not in LEGACY and is_multi(str(run.get("placement") or ""), ALL_ROLES)
    )


def _step(run: dict[str, Any]) -> dict[str, Any]:
    lead = _lead_role(run)
    sides = []
    for node in run.get("nodes") or []:
        sides.append({
            "role": node.get("role"),
            "host": node.get("host"),
            "container": node.get("container") or None,
            "port": run.get("port") if node.get("role") == lead else None,
        })
    head = next((node for node in run.get("nodes") or [] if node.get("role") == lead), None)
    if head is None and run.get("nodes"):
        head = run["nodes"][0]
    stop = list((head or {}).get("stop_argv") or [])
    return {
        "host": (head or {}).get("host"),
        "port": run.get("port"),
        "http_paths": list(run.get("health_paths") or []),
        "placement": "pair" if _multi(run) else "single",
        "sides": sides,
        "stop_argv": stop,
    }


def refresh_run(run: dict[str, Any], runner: Any) -> dict[str, Any]:
    """Ask the node agent for the latest log. A dark Spark does not drop the run.

    Log lines are written onto the stored node records. The phase is left to the
    watch thread, so a status poll cannot put an older phase back.
    """
    if runner is None or run.get("phase") in _TERMINAL:
        return run
    _refresh_logs(run, runner)
    fresh = get_run(str(run.get("id") or ""))
    return fresh or run


LOG_EVERY_S = 60.0  # a serving run whose beat shows it running reads its log tail over SSH this often
_LOG_READ: dict[tuple[str, str], float] = {}


def _refresh_logs(run: dict[str, Any], runner: Any) -> None:
    from sparksi.heartbeat import container_seen

    serving = run.get("phase") == "ready"
    for node in run.get("nodes") or []:
        if serving and node.get("container"):
            key = (str(node.get("host") or ""), str(node["container"]))
            seen = container_seen(*key)
            if seen is not None and seen[0] and time.time() - _LOG_READ.get(key, 0.0) < LOG_EVERY_S:
                # A fresh beat shows the container running: no SSH status this tick.
                node["container_status"] = "running"
                if seen[1] is not None:
                    node["container_gib"] = float(seen[1])
                continue
            _LOG_READ[key] = time.time()
        try:
            status = runner.invoke(
                node.get("host"),
                {"op": "status", "run_id": node.get("run_id"), "container": node.get("container") or ""},
                timeout=8,
            )
        except Exception as exc:  # noqa: BLE001
            node["log_tail"] = str(exc)[-400:]
            continue
        if isinstance(status, dict):
            node["log_tail"] = str(status.get("log_tail") or "")[-4000:]
            node["agent_phase"] = status.get("phase")
            if status.get("container_status"):
                node["container_status"] = str(status.get("container_status"))
            if status.get("exit_code") is not None:
                node["exit_code"] = status.get("exit_code")
            if status.get("mem_gib") not in (None, ""):
                try:
                    node["container_gib"] = float(status["mem_gib"])
                except (TypeError, ValueError):
                    pass


def _used_gib(probe: dict[str, Any]) -> float | None:
    """Unified memory in use: MemTotal minus MemAvailable, in GiB."""
    try:
        total = float(probe.get("mem_total_gib"))
        avail = float(probe.get("mem_available_gib"))
    except (TypeError, ValueError):
        return None
    return max(0.0, total - avail)


def _sample_used(runner: Any, host: str) -> float | None:
    try:
        probe = runner.invoke(host, {"op": "probe", "ports": [], "paths": []}, timeout=8)
    except Exception:  # noqa: BLE001 — a missed sample does not fail the run
        return None
    if not isinstance(probe, dict):
        return None
    return _used_gib(probe)


def _host_is_solo(run: dict[str, Any], host: str) -> bool:
    """True when no other live run is using this node."""
    quiet = {"stopped", "failed", "timed_out", "cancelled"}
    for other in list_runs():
        if str(other.get("id") or "") == str(run.get("id") or ""):
            continue
        if str(other.get("phase") or "") in quiet:
            continue
        for node in other.get("nodes") or []:
            if str(node.get("host") or "") == host:
                return False
    return True


def _sample_estimate(run: dict[str, Any]) -> float | None:
    """Estimate implied by the reservation taken at plan time, without the margin."""
    raw = run.get("reservation_gib")
    if raw in (None, ""):
        return None
    try:
        return max(0.0, float(raw) - MEMORY_MARGIN_GIB)
    except (TypeError, ValueError):
        return None


def _note_memory(run: dict[str, Any], node: dict[str, Any], used: float | None = None) -> None:
    """Record unified memory for one node.

    Container accounting misses GPU allocations on a unified-memory host. When
    this load is alone, the sample is the larger of the container reading and
    the host used-memory delta from the pre-start baseline. A shared node does
    not take that delta. A container-only sample there cannot lower a stored
    peak, and it is not the first peak when it sits below the recipe estimate.
    A jump past 1.5 times the stored peak is ignored unless this load is alone.
    """
    host = str(node.get("host") or "")
    if host and not _host_is_solo(run, host):
        node["shared_host"] = True
    solo = not bool(node.get("shared_host"))
    if used is not None:
        try:
            node["host_used_gib"] = float(used)
        except (TypeError, ValueError):
            pass
    try:
        previous = float(node.get("measured_gib") or 0.0)
    except (TypeError, ValueError):
        previous = 0.0
    container: float | None = None
    if node.get("container_gib") not in (None, ""):
        try:
            container = float(node["container_gib"])
        except (TypeError, ValueError):
            container = None
    host_delta: float | None = None
    # An adopted run's baseline is the idle reading, not a sample taken just
    # before this process started. The delta then includes other processes.
    # That number must not become the stored peak.
    idle_baseline = str(node.get("baseline_kind") or "") == "idle"
    if solo and used is not None and node.get("mem_baseline_gib") not in (None, "") and not idle_baseline:
        try:
            host_delta = max(0.0, float(used) - float(node["mem_baseline_gib"]))
        except (TypeError, ValueError):
            host_delta = None
    sample: float | None = None
    if solo:
        parts = [value for value in (container, host_delta) if value is not None and value > 0]
        sample = max(parts) if parts else None
    else:
        if container is not None and container > 0:
            sample = container
        estimate = _sample_estimate(run)
        if sample is not None and previous <= 0 and estimate is not None and sample + 1e-9 < estimate:
            sample = None
    chosen = consider_peak(previous, sample, solo=solo) if sample is not None else None
    if chosen is None:
        return
    node["measured_gib"] = chosen
    role = str(node.get("role") or "head")
    by_role = dict(run.get("measured_by_role") or {})
    by_role[role] = chosen
    run["measured_by_role"] = by_role
    run["measured_gib"] = max(float(run.get("measured_gib") or 0.0), chosen)
    model_id = str(run.get("model_id") or "")
    if model_id and chosen > 0:
        record_node_peak(model_id, role, chosen, solo=solo)


def _single_exited(run: dict[str, Any]) -> dict[str, Any] | None:
    """A one-node container that has exited or disappeared.

    ``docker run --rm`` removes a stopped container, so a missing name is the
    same failure as an exited status.
    """
    if _multi(run):
        return None
    nodes = [node for node in (run.get("nodes") or []) if isinstance(node, dict)]
    if len(nodes) != 1:
        return None
    node = nodes[0]
    if not str(node.get("container") or ""):
        return None
    if str(node.get("container_status") or "") not in ("exited", "dead", "missing"):
        return None
    return node


def _settle_stop(run_id: str) -> dict[str, Any] | None:
    """Persist a user stop and skip the failure path when one is already requested."""
    current = get_run(run_id)
    if not _user_stop(current):
        return None
    if str(current.get("phase") or "") == "cancelled":
        return current
    nodes = [node for node in (current.get("nodes") or []) if isinstance(node, dict)]
    done = (
        str(current.get("phase") or "") == "stopped"
        and current.get("stop_requested")
        and not current.get("error")
        and not current.get("failed_side")
        and all(str(node.get("phase") or "") == "stopped" for node in nodes)
    )
    if done:
        return current
    return _save(_stopped_copy(current))


def _external(run: dict[str, Any]) -> bool:
    """True when the containers were already running and this process adopted them."""
    return str(run.get("started_by") or "") == "external"


def _side_stopped_note(role: str) -> str:
    if role == "worker":
        return "Worker side stopped"
    if role == "head":
        return "Head side stopped"
    if role.startswith("node") and role[4:].isdigit():
        return f"Spark {role[4:]} side stopped"
    return "One side stopped"


def _note_external(run: dict[str, Any], text: str) -> dict[str, Any]:
    """Record a health problem without stopping adopted containers."""
    run["health_note"] = text
    return _save(run)


def _fail_run(run: dict[str, Any], detail: str, *, side: str, host: str, stopped: list[str] | None = None) -> dict[str, Any]:
    """Record a failure unless a user stop landed while this probe was in flight."""
    settled = _settle_stop(str(run.get("id") or ""))
    if settled:
        return settled
    run["phase"] = "failed"
    run["ok"] = False
    run["error"] = detail
    run["failed_side"] = side
    if stopped is not None:
        run["stopped_sides"] = stopped
    saved = _save(run)
    if str(saved.get("phase") or "") == "failed":
        _audit("run", f"load {run.get('model_id') or ''}".rstrip(), phase="failed", error=detail, machine=host)
    return saved


_READY_SEEN: dict[str, float] = {}
_READY_LOCK = threading.Lock()


def ready_check_s() -> float:
    """Seconds between full health passes for runs already in phase "ready".

    Loading phases keep the watch pace (3 s). A serving run checked every 3 s was
    most of the SSH and ``docker stats`` work Sparksi put on the Sparks.
    """
    try:
        val = float(CFG.controller.get("ready_check_s", 15))
    except (TypeError, ValueError):
        return 15.0
    return max(0.0, min(120.0, val))


def ready_check_due(run: dict, *, mark: bool = True) -> bool:
    """True if a health pass should run now for this run.

    Runs in phase "ready" are throttled to one pass per ready_check_s seconds;
    every other phase is always due. With mark=False the check is read-only
    (a due result does not consume the slot).
    """
    if run.get("phase") != "ready":
        return True
    interval = ready_check_s()
    if interval <= 0:
        return True
    run_id = run.get("id")
    if not run_id:
        return True
    now = time.monotonic()
    with _READY_LOCK:
        last = _READY_SEEN.get(run_id)
        if last is None or (now - last) >= interval:
            if mark:
                _READY_SEEN[run_id] = now
            return True
        return False


def forget_ready_checks() -> None:
    """Drop all recorded ready-check timestamps (used by tests and restarts)."""
    with _READY_LOCK:
        _READY_SEEN.clear()


def tick(run: dict[str, Any], runner: Any, *, now: float | None = None) -> dict[str, Any]:
    """One health pass. A pair that loses a side stops the other and records which died."""
    if not ready_check_due(run):
        return run
    from sparksi.heartbeat import HeartbeatReads

    if runner is not None:
        # Good news may come from a fresh beat; any bad news (port down, container missing, not 200)
        # is read over SSH before it can fail the run or stop a side. Run/stop always go to the runner.
        expect: dict[str, set[str]] = {}
        for node in run.get("nodes") or []:
            if isinstance(node, dict) and node.get("host") and node.get("container"):
                expect.setdefault(str(node["host"]), set()).add(str(node["container"]))
        runner = HeartbeatReads(runner, confirm_bad=True, expect=expect)
    depth = int(getattr(_WRITER, "depth", 0) or 0) + 1
    _WRITER.depth = depth
    _WRITER.monitor = True
    try:
        return _tick(run, runner, now=now)
    finally:
        depth -= 1
        _WRITER.depth = depth
        _WRITER.monitor = depth > 0


def _listening_elsewhere(run: dict[str, Any], runner: Any, expected: int) -> str:
    """A serve port from the launch that is up while the expected port is not.

    Waiting out the full health timeout hides this: the process did answer,
    just not where Sparksi is looking.
    """
    host = ""
    lead = _lead_role(run)
    for node in run.get("nodes") or []:
        if isinstance(node, dict) and node.get("role") == lead:
            host = str(node.get("host") or "")
            break
    if not host:
        nodes = [node for node in (run.get("nodes") or []) if isinstance(node, dict)]
        host = str((nodes[0] if nodes else {}).get("host") or "")
    if not host or not expected:
        return ""
    for alt in other_serve_ports(run):
        if alt == expected or protect.is_off_limits(alt):
            continue
        probe = _probe_node(runner, host, alt, None)
        listening = probe.get("listening") if isinstance(probe, dict) else None
        if isinstance(listening, dict) and (listening.get(str(alt)) or listening.get(alt)):
            return f"the model is listening on {alt}, not {expected}"
    return ""


def _tick(run: dict[str, Any], runner: Any, *, now: float | None = None) -> dict[str, Any]:
    run_id = str(run.get("id") or "")
    settled = _settle_stop(run_id)
    if settled:
        return settled
    if run.get("phase") in _STOPPED or run.get("phase") in _USER_STOP:
        return run
    ready = run.get("phase") == "ready"
    if protect.is_off_limits(int(run.get("port") or 0)):
        run["phase"] = "failed"
        run["ok"] = False
        run["error"] = "port 8188 is off limits"
        return _save(run)
    clock = now if now is not None else time.time()
    if not ready:
        run["phase"] = "waiting_health"
    _refresh_logs(run, runner)
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        _note_memory(run, node, _sample_used(runner, str(node.get("host") or "")))
    settled = _settle_stop(run_id)
    if settled:
        return settled
    if str(run.get("started_by") or "") == "external":
        from sparksi.recipes.adopt import containers_gone

        if containers_gone(run, runner):
            return _save(_stopped_copy(get_run(run_id) or run))
    exited = _single_exited(run)
    if exited and str(run.get("started_by") or "") == "external":
        return _save(_stopped_copy(get_run(run_id) or run))
    if exited:
        status = str(exited.get("container_status") or "")
        code = exited.get("exit_code")
        tail = str(exited.get("log_tail") or "").strip()
        if status == "missing":
            detail = "container missing"
        elif code is not None:
            detail = f"container exited with code {code}"
        else:
            detail = "container exited"
        if tail:
            detail = detail + "\n" + tail
        return _fail_run(run, detail, side=str(exited.get("role") or "head"), host=str(exited.get("host") or ""))
    by_host = {str(node.get("host")): str(node.get("run_id") or "") for node in run.get("nodes") or []}
    wrapped = _PairRunner(runner, by_host)
    step = _step(run)
    seen = dict(run.get("seen") or {})
    external = _external(run)
    pair = (
        _pair_dead(wrapped, step, seen, stop_others=not external)
        if _multi(run)
        else None
    )
    run["seen"] = seen
    if pair and external:
        return _note_external(run, _side_stopped_note(str(pair.get("failed_side") or "")))
    if pair:
        return _fail_run(
            run,
            str(pair.get("error") or "a side failed"),
            side=str(pair.get("failed_side") or ""),
            host=str(pair.get("failed_side") or ""),
            stopped=list(pair.get("stopped_sides") or []),
        )
    host = str(step.get("host") or "")
    try:
        port = int(step["port"])
    except (TypeError, ValueError, KeyError):
        port = 0
    paths = [str(item) for item in (step.get("http_paths") or [])]
    verdict = _health_verdict(wrapped, host, port, paths) if port and not protect.is_off_limits(port) else "ready"
    if not ready and verdict == "refused":
        elsewhere = _listening_elsewhere(run, wrapped, port)
        if elsewhere:
            return _fail_run(run, elsewhere, side=_lead_role(run), host=host)
    run["health"] = {"host": host, "port": port, "paths": paths, "verdict": verdict}
    if verdict == "ready":
        run["phase"] = "ready"
        run["ok"] = True
        run["error"] = None
        run["health_note"] = ""
        run["refused_probes"] = 0
        if not ready:
            _audit("run", f"load {run.get('model_id') or ''}".rstrip(), phase="ready", machine=host)
        return _save(run)
    if ready and verdict == "refused":
        count = int(run.get("refused_probes") or 0) + 1
        run["refused_probes"] = count
        run["phase"] = "ready"
        if count >= REFUSED_PROBES:
            tail = ""
            for node in run.get("nodes") or []:
                if node.get("host") == host:
                    tail = str(node.get("log_tail") or "").strip()
                    break
            detail = "health refused"
            if tail:
                detail = detail + "\n" + tail
            return _fail_run(run, detail, side=_lead_role(run), host=host)
        return _save(run)
    if ready and verdict == "wait":
        run["phase"] = "ready"
        run["refused_probes"] = 0
        return _save(run)
    settled = _settle_stop(run_id)
    if settled:
        return settled
    if verdict == "fail":
        if external:
            return _note_external(run, "Not answering health checks")
        role = _lead_role(run)
        for node in run.get("nodes") or []:
            if node.get("host") == host and node.get("role"):
                role = str(node["role"])
        stopped = []
        if _multi(run):
            for node in run.get("nodes") or []:
                if node.get("role") == role:
                    continue
                wrapped.run_argv(str(node.get("host") or ""), list(node.get("stop_argv") or []))
                stopped.append(str(node.get("role") or ""))
        return _fail_run(run, f"health on {host} failed", side=role, host=host, stopped=stopped)
    started = float(run.get("started") or clock)
    timeout_s = float(run.get("timeout_s") or 1800)
    settled = _settle_stop(run_id)
    if settled:
        return settled
    if external and clock - started >= timeout_s:
        return _note_external(run, "Not answering health checks")
    if clock - started >= timeout_s:
        stopped = []
        for node in run.get("nodes") or []:
            wrapped.run_argv(str(node.get("host") or ""), list(node.get("stop_argv") or []))
            stopped.append(str(node.get("role") or ""))
        run["phase"] = "timed_out"
        run["ok"] = False
        run["error"] = f"port {port} did not answer"
        run["stopped_sides"] = stopped
        _audit("run", f"load {run.get('model_id') or ''}".rstrip(), phase="timed_out", error=run["error"], machine=host)
        return _save(run)
    return _save(run)


def _watch(run_id: str, runner: Any, pause: float) -> None:
    try:
        while True:
            run = get_run(run_id)
            if not run or run.get("phase") in _STOPPED or run.get("phase") in _USER_STOP:
                return
            tick(run, runner)
            run = get_run(run_id)
            if not run or run.get("phase") in _STOPPED or run.get("phase") in _USER_STOP:
                return
            time.sleep(pause)
    finally:
        with _LOCK:
            _THREADS.discard(run_id)


def _helper_drift(runner: Any, node: dict[str, Any]) -> str:
    """Warn when the node's evict module is not the controller's copy. Never blocks."""
    if not flag_on(node.get("evict_page_cache")):
        return ""
    role = str(node.get("role") or "head")
    try:
        local = module_sha256()
    except OSError:
        return ""
    try:
        remote = runner.invoke(node.get("host"), {"op": "evict_sha256"}, timeout=8)
    except Exception as exc:  # noqa: BLE001 — a missed check is a warning
        return f"page cache helper on {role} could not be checked: {exc}"[:400]
    if not isinstance(remote, dict) or not remote.get("sha256"):
        return f"page cache helper on {role} could not be read"
    if str(remote.get("sha256")) != local:
        return f"page cache helper on {role} does not match the controller"
    return ""


def _evict_page_cache(runner: Any, node: dict[str, Any]) -> str:
    """Best-effort page-cache drop. A problem is a warning, never a launch failure."""
    if not flag_on(node.get("evict_page_cache")):
        return ""
    role = str(node.get("role") or "head")
    path = str(node.get("evict_path") or "")
    if not path:
        return f"page cache not cleared on {role}: no specific model snapshot"
    blocked = eviction_block(path, extra_roots=_cache_roots())
    if blocked:
        return f"page cache not cleared on {role}: {blocked}"
    if not _path_under(path, roots_for(role)):
        return f"page cache not cleared on {role}: path is outside mount roots"
    argv = evict_argv(path)
    if argv[:3] != ["python3", "-m", "sparksi.evict"] or len(argv) != 4:
        return f"page cache not cleared on {role}: command refused"
    run_argv = getattr(runner, "run_argv", None)
    if run_argv is None:
        return f"page cache not cleared on {role}: no argv runner"
    try:
        result = run_argv(str(node.get("host") or ""), argv, 60)
    except Exception as exc:  # noqa: BLE001 — eviction must not block the launch
        return f"page cache not cleared on {role}: {exc}"[:400]
    if isinstance(result, dict) and (result.get("ok") or result.get("code") == 0):
        return ""
    detail = "failed"
    if isinstance(result, dict):
        detail = str(result.get("error") or result.get("stderr") or "failed")
    return f"page cache not cleared on {role}: {detail}"[:400]


def _sample_free(runner: Any, host: str) -> float | None:
    """MemAvailable in GiB. Missing when the node cannot be asked."""
    try:
        probe = runner.invoke(host, {"op": "probe", "ports": [], "paths": []}, timeout=8)
    except Exception:  # noqa: BLE001 — a missed sample does not invent free memory
        return None
    if not isinstance(probe, dict):
        return None
    try:
        number = float(probe.get("mem_available_gib"))
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _replace_sides(replace_run_id: str) -> dict[str, str]:
    """Role to the stopped run's node id, when a Restart names that run."""
    if not replace_run_id:
        return {}
    prior = get_run(replace_run_id)
    if not prior or not prior.get("stop_requested") or str(prior.get("phase") or "") != "stopped":
        return {}
    sides: dict[str, str] = {}
    for node in prior.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        role = str(node.get("role") or "")
        side = str(node.get("run_id") or "")
        if role and side:
            sides[role] = side
    return sides


def start_rendered(
    recipe: dict[str, Any],
    runner: Any,
    *,
    pause: float = 3.0,
    sync: bool = False,
    replace_run_id: str = "",
) -> dict[str, Any]:
    """Launch every node (worker first on a pair) and return the run id at once."""
    run_id = uuid.uuid4().hex[:12]
    nodes = []
    for node in recipe.get("nodes") or []:
        side_id = f"{run_id}-{node.get('role')}"
        nodes.append({
            "role": node.get("role"),
            "host": node.get("host"),
            "run_id": side_id,
            "argv": list(node.get("argv") or []),
            "env": dict(node.get("env") or {}),
            "container": node.get("container") or "",
            "stop_argv": list(node.get("stop_argv") or []),
            "phase": "starting",
            "log_tail": "",
            "evict_page_cache": bool(node.get("evict_page_cache")),
            "evict_path": str(node.get("evict_path") or ""),
            "evict_warning": "",
        })
    run = {
        "id": run_id,
        "model_id": recipe.get("id"),
        "engine": recipe.get("engine"),
        "placement": recipe.get("placement"),
        "port": recipe.get("port"),
        "phase": "starting",
        "ok": None,
        "error": None,
        "failed_side": None,
        "stopped_sides": [],
        "health": {},
        "nodes": nodes,
        "order": list(recipe.get("order") or [node.get("role") for node in nodes]),
        "health_paths": list(recipe.get("health_paths") or []),
        "timeout_s": int(recipe.get("timeout_s") or 1800),
        "experimental": bool(recipe.get("experimental")),
        "experimental_reason": recipe.get("experimental_reason") or "",
        "started": time.time(),
        "seen": {},
        "allow_roce": recipe_allows_roce(recipe),
        "reservation_gib": recipe.get("reservation_gib"),
        "measured_gib": None,
        "started_by": "sparksi",
    }
    _save(run)
    _audit("run", f"load {run.get('model_id') or ''}".rstrip(), phase="starting", machine=str((nodes[0] or {}).get("host") if nodes else ""))
    replaced_sides = _replace_sides(replace_run_id)
    order = {role: index for index, role in enumerate(run["order"])}
    ordered = sorted(nodes, key=lambda node: order.get(node.get("role"), 99))
    for node in ordered:
        if protect.is_off_limits(int(run.get("port") or 0)):
            run["phase"] = "failed"
            run["ok"] = False
            run["error"] = "port 8188 is off limits"
            return _save(run)
        baseline = _sample_used(runner, str(node.get("host") or ""))
        if baseline is not None:
            node["mem_baseline_gib"] = baseline
        warning = _evict_page_cache(runner, node)
        drift = _helper_drift(runner, node)
        if drift:
            run.setdefault("warnings", []).append(drift)
            if not warning:
                warning = drift
        if warning:
            node["evict_warning"] = warning
            node["log_tail"] = warning
            run.setdefault("warnings", []).append(warning)
            _audit(
                "run",
                f"load {run.get('model_id') or ''}".rstrip(),
                phase="warning",
                error=warning,
                machine=str(node.get("host") or ""),
            )
        blocked = launcher_refusal(node["argv"])
        if blocked:
            run["phase"] = "failed"
            run["ok"] = False
            run["error"] = blocked
            run["failed_side"] = node.get("role")
            return _save(run)
        request = {
            "op": "run",
            "run_id": node["run_id"],
            "argv": node["argv"],
            "env": node["env"],
            "container": node["container"],
            "allow_roce": bool(run.get("allow_roce")),
        }
        prior_side = replaced_sides.get(str(node.get("role") or ""))
        if prior_side:
            request["replace_run_id"] = prior_side
        result = runner.invoke(node["host"], request, timeout=120)
        if not isinstance(result, dict) or not result.get("ok"):
            run["phase"] = "failed"
            run["ok"] = False
            run["error"] = (result or {}).get("error") or f"{node.get('role')} did not start"
            run["failed_side"] = node.get("role")
            for earlier in ordered:
                if earlier is node:
                    break
                runner.invoke(
                    earlier["host"],
                    {"op": "stop", "run_id": earlier["run_id"], "argv": earlier["stop_argv"]},
                    timeout=60,
                )
            _audit("run", f"load {run.get('model_id') or ''}".rstrip(), phase="failed", error=run["error"], machine=str(node.get("host") or ""))
            return _save(run)
        node["phase"] = "started"
    run["phase"] = "waiting_health"
    _save(run)
    if sync:
        while run.get("phase") not in _TERMINAL:
            run = tick(run, runner)
            if run.get("phase") not in _TERMINAL:
                time.sleep(pause)
        return get_run(run_id) or run
    with _LOCK:
        if run_id not in _THREADS:
            _THREADS.add(run_id)
            threading.Thread(target=_watch, args=(run_id, runner, pause), name=f"sparksi-run-{run_id}", daemon=True).start()
    return get_run(run_id) or run


def stop_recorded(run: dict[str, Any], runner: Any) -> dict[str, Any]:
    """Stop every side. Refuses port 8188 before any argv is sent.

    ``stopping`` is written before the containers are removed, and it is terminal
    for the health monitor. A probe that is already in flight cannot put the run
    back to ``waiting_health`` or record the removed container as a failure.
    A side whose container disappears after this request is stopped.
    """
    current = get_run(str(run.get("id") or "")) or dict(run)
    if protect.is_off_limits(int(current.get("port") or 0)):
        current["phase"] = "failed"
        current["ok"] = False
        current["error"] = "port 8188 is off limits"
        return _save(current)
    if current.get("stop_requested") or str(current.get("phase") or "") in _USER_STOP:
        return current
    current["phase"] = "stopping"
    current["stop_requested"] = True
    current["ok"] = None
    free_at_stop: dict[str, float] = {}
    for node in current.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        role = str(node.get("role") or "")
        free = _sample_free(runner, str(node.get("host") or ""))
        if role and free is not None:
            free_at_stop[role] = free
            node["free_at_stop_gib"] = free
    current = _save(current)
    _audit("stop", f"unload {current.get('model_id') or ''}".rstrip(), phase="stopping")
    run = current
    errors = []
    for node in run.get("nodes") or []:
        argv = list(node.get("stop_argv") or [])
        blocked = launcher_refusal(argv) if argv else None
        if blocked:
            errors.append(blocked)
            continue
        if any("dgx-announce" in part or part.rsplit("/", 1)[-1] == "video-mode.sh" for part in argv):
            errors.append("refusing forbidden command")
            continue
        result = runner.invoke(
            node.get("host"),
            {"op": "stop", "run_id": node.get("run_id"), "argv": argv},
            timeout=120,
        )
        if not isinstance(result, dict) or not result.get("ok", True):
            errors.append(str((result or {}).get("error") or "stop failed"))
        node["phase"] = "stopped"
    if errors:
        run["phase"] = "failed"
        run["ok"] = False
        run["error"] = errors[0]
        _audit("stop", f"unload {run.get('model_id') or ''}".rstrip(), phase="failed", error=run["error"])
        return _save(run)
    run["phase"] = "stopped"
    run["ok"] = True
    run["error"] = None
    run["stopped_at"] = time.time()
    _audit("stop", f"unload {run.get('model_id') or ''}".rstrip(), phase="stopped")
    return _save(run)
