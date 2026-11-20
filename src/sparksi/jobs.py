# SPDX-License-Identifier: Apache-2.0

"""One copy, delete, or launch at a time. The HTTP handler returns the job id."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from sparksi import protect
from sparksi.activity import store as activity_store
from sparksi.config import CFG, env
from sparksi.health import DEFAULT_HEALTH_TIMEOUT_S
from sparksi.launch import monitor_loaded, plan_drop, plan_run, plan_stage, plan_stop, plan_switch
from sparksi.library import catalog_serving, external_refusal, fleet, row_by_id
from sparksi.log import logger
from sparksi.never import launcher_refusal
from sparksi.remote import SshRunner

# A load is finished when it reaches one of these. ``unknown`` is an in-flight
# job reloaded after a restart, still waiting for the next health check.
_TERMINAL = frozenset({"ready", "failed", "timed_out", "done", "error", "cancelled"})

_LOCK = threading.RLock()
_JOB: dict[str, Any] | None = None
_QUEUE: list[dict[str, Any]] = []
_PAUSE = False
_CANCEL = False
_WORKER = False


def reset_for_tests() -> None:
    global _JOB, _PAUSE, _CANCEL, _WORKER
    with _LOCK:
        _JOB = None
        _QUEUE.clear()
        _PAUSE = False
        _CANCEL = False
        _WORKER = False


def job_path() -> Path:
    """In-flight job. ``SPARKSI_STATE_DIR`` wins so tests stay off the home dir."""
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "job.json"
    return Path.home() / ".local" / "state" / "sparksi" / "job.json"


def _dumpable(job: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in job.items():
        if key in ("fn", "queue"):
            continue
        try:
            json.dumps(value)
        except TypeError:
            continue
        out[key] = value
    return out


def _persist() -> None:
    with _LOCK:
        payload = _dumpable(_JOB) if _JOB else None
        try:
            path = job_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / (path.name + ".tmp")
            tmp.write_text(json.dumps({"job": payload}), encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            return


def _read_job_file() -> dict[str, Any] | None:
    try:
        data = json.loads(job_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if not isinstance(data, dict):
        return None
    job = data.get("job")
    return job if isinstance(job, dict) else None


def current() -> dict[str, Any] | None:
    with _LOCK:
        if _JOB is None and not _QUEUE:
            return None
        job = dict(_JOB or {"kind": "stage", "phase": "queued", "ok": None})
        job["queue"] = [item.get("id_row") for item in _QUEUE]
        job["paused"] = _PAUSE
    started = job.get("started")
    if isinstance(started, (int, float)):
        job["elapsed_s"] = round(max(0.0, time.time() - float(started)), 1)
    return job


def _set(**fields: Any) -> None:
    with _LOCK:
        changed = _JOB is not None
        if changed:
            _JOB.update(fields)
    if changed:
        _persist()


def request_pause(paused: bool) -> dict[str, Any]:
    global _PAUSE
    with _LOCK:
        _PAUSE = bool(paused)
    return current() or {"paused": _PAUSE, "phase": "idle"}


def request_cancel() -> dict[str, Any]:
    global _CANCEL
    with _LOCK:
        _CANCEL = True
    return current() or {"phase": "cancelled"}


def _cancelled() -> bool:
    with _LOCK:
        return _CANCEL


def _paused() -> bool:
    with _LOCK:
        return _PAUSE


def _wait_if_paused() -> bool:
    while _paused():
        if _cancelled():
            return False
        _set(phase="paused")
        time.sleep(0.4)
    return not _cancelled()


def _begin(kind: str, payload: dict[str, Any], fn) -> dict[str, Any]:
    """Start fn now, or queue it behind the active transfer."""
    global _JOB, _WORKER, _CANCEL
    item = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind,
        "phase": "queued",
        "ok": None,
        "error": None,
        "started": time.time(),
        **payload,
    }
    with _LOCK:
        if _WORKER:
            _QUEUE.append({**item, "fn": fn})
            queued = dict(item)
            queued["queue"] = [row.get("id_row") for row in _QUEUE]
            return {"verdict": "QUEUED", "job": queued}
        _JOB = dict(item)
        _WORKER = True
        _CANCEL = False
    _persist()

    def run(first: Any = fn) -> None:
        global _WORKER, _JOB, _CANCEL
        active = first
        while True:
            with _LOCK:
                _CANCEL = False
            try:
                active()
            except Exception as e:  # noqa: BLE001
                _finish(False, str(e))
            with _LOCK:
                if not _QUEUE:
                    _WORKER = False
                    return
                nxt = _QUEUE.pop(0)
                active = nxt.pop("fn")
                _JOB = dict(nxt)
            _persist()

    _thread(run)
    return {"verdict": "STARTED", "job": dict(item)}


def _error_tail(error: str | None) -> str:
    text = " ".join((error or "").split())
    if len(text) > 400:
        return text[-400:]
    return text


def _audit(feed: str, message: str, *, phase: str, error: str | None = None, machine: str = "") -> None:
    """Activity feed and the controller journal. The error is one trimmed line."""
    tail = _error_tail(error)
    text = f"{message} phase={phase}"
    if tail:
        text += f" error={tail}"
    try:
        activity_store().append(feed, text, machine)
    except Exception:
        pass
    try:
        logger().info("%s", text)
    except Exception:
        return


def _audit_phase(phase: str) -> None:
    """One Activity/journal line for a load or unload phase. No-op for other jobs."""
    job = current() or {}
    kind = job.get("kind")
    if kind == "stop":
        feed, verb = "stop", "unload"
    elif kind in ("switch", "run"):
        feed, verb = ("switch" if kind == "switch" else "run"), "load"
    else:
        return
    _audit(
        feed,
        f"{verb} {job.get('id_row') or ''}".rstrip(),
        phase=phase,
        machine=str(job.get("host") or ""),
    )


def _audit_outcome(ok: bool, error: str | None, phase: str) -> None:
    job = current() or {}
    kind = job.get("kind")
    if kind not in ("switch", "stop", "run"):
        return
    if kind == "stop":
        feed, verb = "stop", "unload"
    elif kind == "run":
        feed, verb = "run", "load"
    else:
        feed, verb = "switch", "load"
    _audit(
        feed,
        f"{verb} {job.get('id_row') or ''}".rstrip(),
        phase=phase,
        error=None if ok else error,
        machine=str(job.get("host") or ""),
    )


def _finish(ok: bool, error: str | None = None, **extra: Any) -> None:
    phase = extra.pop("phase", None)
    if phase is None:
        phase = "ready" if ok else "failed"
    _set(phase=phase, ok=ok, error=error, **extra)
    _audit_outcome(ok, error, phase)


def _thread(fn) -> None:
    threading.Thread(target=fn, name="sparksi-job", daemon=True).start()


def _runner(catalog: dict[str, Any], runner: Any | None) -> Any:
    return runner or SshRunner(catalog)


def start_stage(catalog, row_id: str, hosts: list[str], *, lock_on: bool, runner=None, refill: bool = False) -> dict[str, Any]:
    snap = fleet(catalog, runner)
    row = row_by_id(snap["library"], row_id)
    if not row:
        return {"verdict": "REFUSE", "reasons": [f"unknown library id {row_id}"]}
    plans = [plan_stage(catalog, row, host, snap["probes"], lock_on=lock_on, refill=refill) for host in hosts]
    bad = [p for p in plans if p["verdict"] != "READY"]
    if bad:
        return {"verdict": "REFUSE", "reasons": [r for p in bad for r in p["reasons"]], "plans": plans}
    def work() -> None:
        exec_runner = _runner(catalog, runner)
        for plan in plans:
            if not _wait_if_paused():
                _finish(False, "cancelled", phase="cancelled")
                return
            req = plan.get("request") or {}
            if req.get("op") == "stage_flat":
                if not _copy_shards(exec_runner, plan):
                    return
                continue
            _set(phase=f"copy {plan['host']}", host=plan["host"])
            result = exec_runner.invoke(plan["host"], req, timeout=6 * 3600)
            if _cancelled():
                _finish(False, "cancelled", phase="cancelled")
                return
            if not result.get("ok"):
                _finish(False, result.get("error") or "copy failed", result=result)
                return
        _finish(True)

    started = _begin("stage", {"id_row": row_id, "hosts": hosts}, work)
    started["plans"] = plans
    return started


def start_drop(catalog, row_id: str, host: str, confirm: str, *, lock_on: bool, runner=None) -> dict[str, Any]:
    snap = fleet(catalog, runner)
    row = row_by_id(snap["library"], row_id)
    if not row:
        return {"verdict": "REFUSE", "reasons": [f"unknown library id {row_id}"]}
    plan = plan_drop(row, host, snap["probes"], lock_on=lock_on, confirm=confirm)
    if plan["verdict"] != "READY":
        return plan
    def work() -> None:
        _set(phase=f"delete {host}", host=host)
        result = _runner(catalog, runner).invoke(host, plan["request"], timeout=3600)
        if not result.get("ok"):
            _finish(False, result.get("error") or "delete failed", result=result)
            return
        _finish(True, result=result)

    started = _begin("drop", {"id_row": row_id, "hosts": [host]}, work)
    started["plan"] = plan
    return started


def relock_after_switch(path: Path | None = None) -> None:
    """A finished switch closes the GPU lock. A failed switch leaves it as it was.

    ``path`` is the lock file the switch took. A later change to the process
    environment must not move that close onto a different file.
    """
    from sparksi.lock import Lock

    Lock(path).acquire(reason="switch finished")


def lock_for_load() -> dict[str, Any]:
    """Load takes the lock as soon as it is accepted, before the model starts."""
    from sparksi.lock import Lock

    return Lock().acquire(reason="load started").to_dict()


def load_in_progress() -> bool:
    job = current()
    if not job or job.get("kind") != "switch":
        return False
    return job.get("phase") not in _TERMINAL


def _port_open(exec_runner: Any, host: str, port: int) -> bool:
    if protect.is_off_limits(int(port)):
        return False
    probe = exec_runner.invoke(host, {"op": "probe", "ports": [int(port)], "paths": []}, timeout=30)
    return bool((probe.get("listening") or {}).get(str(port)))


def _wait_port(
    exec_runner: Any,
    host: str,
    port: int,
    want_up: bool,
    attempts: int = 8,
    pause: float = 1.0,
    sleep: Any = None,
) -> bool:
    """Short wait used when a stop must free a port. Loads do not use this."""
    if protect.is_off_limits(int(port)):
        return True
    snooze = sleep or time.sleep
    for _ in range(attempts):
        if _port_open(exec_runner, host, port) == want_up:
            return True
        snooze(pause)
    return _port_open(exec_runner, host, port) == want_up


def _poll_interval(pause: float) -> float:
    if pause <= 0:
        return 0.0
    # The historical default pause was 1s. Health is polled every few seconds.
    if pause == 1.0:
        return 3.0
    return float(pause)


def _health_record(step: dict[str, Any], timeout_s: int) -> dict[str, Any]:
    return {
        "host": step.get("host"),
        "port": step.get("port"),
        "http_paths": list(step.get("http_paths") or []),
        "timeout_s": int(timeout_s),
        "placement": step.get("placement"),
        "sides": list(step.get("sides") or []),
        "stop_argv": list(step.get("stop_argv") or []),
    }


def _tailnet_for(host: str) -> str:
    """Tailnet address of ``host`` from local config. Empty when the node is unknown."""
    for node in CFG.nodes:
        if host and host in (node.name, node.id, node.ssh):
            return node.tailscale
    return ""


def _probe_node(exec_runner: Any, host: str, port: int | None, path: str | None) -> dict[str, Any]:
    """Agent probe on a node. Never asks for an off-limits port.

    The listen address is the node's tailnet address. The recipe binds that
    address, so a loopback connect does not see it. HTTP uses ``/health`` or
    ``/v1/models`` when the caller names one.
    """
    if protect.is_off_limits(port):
        return {"ok": True, "listening": {}, "http": {}, "containers": None}
    req: dict[str, Any] = {"op": "probe", "ports": [int(port)] if port else [], "paths": []}
    tailnet = _tailnet_for(host)
    if tailnet:
        req["listen_host"] = tailnet
    if path and port:
        req["http_paths"] = {str(int(port)): path}
    try:
        out = exec_runner.invoke(host, req, timeout=20)
    except Exception as exc:  # noqa: BLE001 — a dead node is a health miss, not a crash
        return {"ok": False, "error": str(exc), "listening": {}, "http": {}}
    return out if isinstance(out, dict) else {"ok": False, "error": "bad probe", "listening": {}, "http": {}}


def _listening(probe: dict[str, Any], port: int) -> bool:
    return bool((probe.get("listening") or {}).get(str(port)))


def _side_state(probe: dict[str, Any], side: dict[str, Any]) -> bool | None:
    """True when the side is up, False when it has exited, None when we cannot tell."""
    container = side.get("container")
    port = side.get("port")
    if container:
        containers = probe.get("containers") if "containers" in probe else None
        if containers is None:
            pass
        elif container in containers:
            return True
        else:
            return False
    if port in (None, ""):
        return None
    try:
        return _listening(probe, int(port))
    except (TypeError, ValueError):
        return None


def _stop_other_sides(exec_runner: Any, step: dict[str, Any], failed: dict[str, Any]) -> list[str]:
    """Stop every pair side except the one that died first."""
    stopped: list[str] = []
    argv = [str(part) for part in (step.get("stop_argv") or [])]
    if not argv:
        return stopped
    failed_role = failed.get("role")
    for side in step.get("sides") or []:
        if side.get("role") == failed_role:
            continue
        host = side.get("host")
        if not host:
            continue
        result = exec_runner.run_argv(host, argv, timeout=3600)
        # Only a stop that answered ok counts: the page keeps Unload live for the rest.
        if isinstance(result, dict) and result.get("ok") is not False:
            stopped.append(str(side.get("role") or host))
    return stopped


def _pair_dead(
    exec_runner: Any,
    step: dict[str, Any],
    seen: dict[str, bool],
    *,
    stop_others: bool = True,
) -> dict[str, Any] | None:
    if step.get("placement") != "pair":
        return None
    sides = list(step.get("sides") or [])
    if len(sides) < 2:
        return None
    for side in sides:
        try:
            port_i = int(side["port"]) if side.get("port") not in (None, "") else None
        except (TypeError, ValueError):
            port_i = None
        if protect.is_off_limits(port_i):
            port_i = None
        probe = _probe_node(exec_runner, str(side.get("host") or ""), port_i, None)
        state = _side_state(probe, {**side, "port": port_i})
        role = str(side.get("role") or side.get("host") or "")
        if state is True:
            seen[role] = True
        elif state is False and seen.get(role):
            stopped = _stop_other_sides(exec_runner, step, side) if stop_others else []
            return {
                "ok": False,
                "phase": "failed",
                "error": f"{role} failed first",
                "failed_side": role,
                "stopped_sides": stopped,
                "started": True,
            }
    return None


def _health_verdict(exec_runner: Any, host: str, port: int, paths: list[str]) -> str:
    """``ready``, ``wait``, or ``fail``. Connection refused is still loading."""
    if protect.is_off_limits(int(port)):
        return "ready"
    if not paths:
        probe = _probe_node(exec_runner, host, port, None)
        return "ready" if _listening(probe, port) else "wait"
    for path in paths:
        probe = _probe_node(exec_runner, host, port, path)
        if not _listening(probe, port):
            return "refused"
        http = (probe.get("http") or {}).get(str(port)) or {}
        status = http.get("status")
        if status == 200:
            return "ready"
        try:
            code = int(status) if status is not None else 0
        except (TypeError, ValueError):
            code = 0
        # llama-server answers 503 on /health until it has finished loading.
        if code == 503:
            return "wait"
        if code >= 500:
            return "fail"
    return "wait"


def _primary_role(step: dict[str, Any], host: str) -> str:
    for side in step.get("sides") or []:
        if side.get("host") == host and side.get("role"):
            return str(side["role"])
    return "head" if step.get("placement") == "pair" else ""


def _poll_health(
    exec_runner: Any,
    step: dict[str, Any],
    *,
    pause: float,
    timeout_s: float,
    sleep: Any,
    now: Any,
    attempts: int,
    begun: float | None = None,
) -> dict[str, Any]:
    try:
        port_i = int(step["port"])
    except (TypeError, ValueError, KeyError):
        return {"ok": True, "started": True}
    if protect.is_off_limits(port_i):
        return {"ok": True, "started": True, "skipped": str(port_i)}
    host = str(step.get("host") or "")
    paths = [str(item) for item in (step.get("http_paths") or []) if item]
    interval = _poll_interval(pause)
    start = begun if begun is not None else now()
    deadline = start + float(timeout_s)
    seen: dict[str, bool] = {}
    checks = 0
    _set(phase="waiting_health", host=host, elapsed_s=0, health=_health_record(step, int(timeout_s)))
    _audit_phase("waiting_health")
    while True:
        checks += 1
        _set(phase="waiting_health", host=host, elapsed_s=round(max(0.0, now() - start), 1))
        pair = _pair_dead(exec_runner, step, seen)
        if pair:
            return pair
        verdict = _health_verdict(exec_runner, host, port_i, paths)
        if verdict == "ready":
            return {"ok": True, "started": True}
        if verdict == "fail":
            role = _primary_role(step, host)
            failed = {"role": role, "host": host}
            stopped = _stop_other_sides(exec_runner, step, failed) if step.get("placement") == "pair" else []
            return {
                "ok": False,
                "phase": "failed",
                "error": f"health on {host} failed",
                "failed_side": role or None,
                "stopped_sides": stopped,
                "started": True,
            }
        if pause <= 0 and checks >= max(1, int(attempts)):
            return {"ok": False, "phase": "timed_out", "error": f"port {port_i} did not answer", "started": True}
        if now() >= deadline:
            return {"ok": False, "phase": "timed_out", "error": f"port {port_i} did not answer", "started": True}
        sleep(interval)


def perform_switch(
    exec_runner: Any,
    plan: dict[str, Any],
    copy_fn,
    attempts: int = 8,
    pause: float = 1.0,
    sleep: Any = None,
    now: Any = None,
) -> dict[str, Any]:
    """Run a switch plan. A stop that leaves the port up does not start the next harness.

    After start, health is polled until the port (and, for an HTTP engine,
    ``/health`` or ``/v1/models``) answers or the recipe timeout elapses.
    """
    snooze = sleep or time.sleep
    clock = now or time.time
    for step in plan.get("steps") or []:
        if step.get("op") in ("start", "stop"):
            blocked = launcher_refusal(list(step.get("argv") or []))
            if blocked:
                return {"ok": False, "error": blocked, "started": False, "phase": "failed"}
    started = False
    for step in plan.get("steps") or []:
        op = step.get("op")
        if op == "note":
            _set(detail=step.get("text") or "note")
            continue
        if op == "stop":
            _set(phase="stopping", host=step.get("host"), detail=step.get("text") or "stop")
            _audit_phase("stopping")
            result = exec_runner.run_argv(step["host"], list(step.get("argv") or []), timeout=3600)
            if not result.get("ok"):
                return {"ok": False, "error": result.get("error") or "stop failed", "started": False, "phase": "failed"}
            port = step.get("port")
            if (
                port
                and not protect.is_off_limits(int(port))
                and not _wait_port(exec_runner, step["host"], int(port), False, attempts, pause, snooze)
            ):
                return {"ok": False, "error": f"port {port} still up", "started": False, "phase": "failed"}
        elif op == "copy":
            _set(phase="copying", detail=step.get("text") or "copy")
            _audit_phase("copying")
            if not copy_fn(step):
                return {"ok": False, "error": "copy failed", "started": False, "phase": "failed"}
        elif op == "start":
            _set(phase="starting", host=step.get("host"), detail=step.get("text") or "start")
            _audit_phase("starting")
            # Some start scripts download a checkpoint and stay up until the
            # API answers; the catalog row sets start_timeout_s for those.
            argv = list(step.get("argv") or [])
            timeout = int(step.get("timeout_s") or 3600)
            result = exec_runner.run_argv(step["host"], argv, timeout=timeout)
            started = True
            if not result.get("ok"):
                detail = (result.get("error") or result.get("tail") or "start failed").strip()
                return {"ok": False, "error": detail, "started": True, "phase": "failed"}
            port = step.get("port")
            if port and not protect.is_off_limits(int(port)):
                budget = int(step.get("health_timeout_s") or DEFAULT_HEALTH_TIMEOUT_S)
                outcome = _poll_health(
                    exec_runner,
                    step,
                    pause=pause,
                    timeout_s=budget,
                    sleep=snooze,
                    now=clock,
                    attempts=attempts,
                )
                if not outcome.get("ok"):
                    return outcome
    return {"ok": True, "started": started}


def start_switch(catalog, row_id: str, *, lock_on: bool, host: str = "", runner=None) -> dict[str, Any]:
    snap = fleet(catalog, runner)
    target = row_by_id(snap["library"], row_id)
    if not target:
        return {"verdict": "REFUSE", "reasons": [f"unknown library id {row_id}"]}
    plan = plan_switch(
        catalog, target, snap["library"], lock_on=lock_on, host=host, loaded=monitor_loaded()
    )
    if plan["verdict"] != "READY":
        return plan
    from sparksi.lock import Lock, default_lock_path

    lock_path = default_lock_path()
    held = Lock(lock_path).acquire(reason="load started").to_dict()
    exec_runner = _runner(catalog, runner)

    def copy_fn(step: dict[str, Any]) -> bool:
        fresh = fleet(catalog, runner)
        row = row_by_id(fresh["library"], step["id"])
        if not row:
            return False
        for host in step.get("hosts") or []:
            staged = plan_stage(catalog, row, host, fresh["probes"], lock_on=False)
            if staged["verdict"] != "READY":
                _set(phase="copy refused", error="; ".join(staged["reasons"]))
                return False
            req = staged.get("request") or {}
            if req.get("op") == "stage_flat":
                if not _copy_shards(exec_runner, staged):
                    return False
            else:
                result = exec_runner.invoke(host, req, timeout=6 * 3600)
                if not result.get("ok"):
                    _set(error=result.get("error") or "copy failed")
                    return False
        return True

    def work() -> None:
        result = perform_switch(exec_runner, plan, copy_fn)
        extra: dict[str, Any] = {}
        if result.get("failed_side"):
            extra["failed_side"] = result["failed_side"]
        if result.get("stopped_sides"):
            extra["stopped_sides"] = result["stopped_sides"]
        if not result.get("ok"):
            _finish(False, result.get("error") or "switch failed", phase=result.get("phase") or "failed", **extra)
            return
        _finish(True, phase="ready")
        relock_after_switch(lock_path)

    started = _begin("switch", {"id_row": row_id, "hosts": [plan.get("id")]}, work)
    started["plan"] = plan
    started["lock"] = held
    return started


def start_run(catalog, row_id: str, *, lock_on: bool, runner=None) -> dict[str, Any]:
    snap = fleet(catalog, runner)
    row = row_by_id(snap["library"], row_id)
    if not row:
        return {"verdict": "REFUSE", "reasons": [f"unknown library id {row_id}"]}
    plan = plan_run(catalog, row, snap["probes"], lock_on=lock_on, rows=snap["library"])
    if plan["verdict"] != "READY":
        return plan
    return _start_argv(catalog, "run", plan, runner)


def start_stop(catalog, row_id: str, *, lock_on: bool, runner=None) -> dict[str, Any]:
    snap = fleet(catalog, runner)
    row = row_by_id(snap["library"], row_id)
    if not row:
        return {"verdict": "REFUSE", "reasons": [f"unknown library id {row_id}"]}
    if load_in_progress():
        return {"verdict": "REFUSE", "reasons": ["still loading; stop does not interrupt a load"]}
    outside = external_refusal(row)
    if outside:
        # An engine Sparksi did not start: no catalog stop command and no container action.
        return {"verdict": "REFUSE", "external": True, "stopped": False, "error": outside, "reasons": [outside]}
    if row.get("serving") is False:
        # Not running: no stop command runs, so stopped containers and other harnesses are left alone.
        name = row.get("title") or row_id
        return {
            "verdict": "REFUSE",
            "not_running": True,
            "stopped": False,
            "error": f"{name} is not running; nothing was stopped",
            "reasons": [f"{name} is not running; nothing was stopped"],
        }
    run_live = row.get("run_live")
    if isinstance(run_live, dict) and run_live.get("up") and run_live.get("id"):
        if catalog_serving(row, snap["probes"]) is not True:
            # Sparksi launched what is up: stop that run's own containers, not the catalog script.
            return _stop_run_of_row(catalog, row, str(run_live["id"]), lock_on=lock_on, runner=runner)
    plan = plan_stop(row, lock_on=lock_on)
    if plan["verdict"] != "READY":
        return plan
    return _start_argv(catalog, "stop", plan, runner)


def _stop_run_of_row(catalog, row: dict[str, Any], run_id: str, *, lock_on: bool, runner=None) -> dict[str, Any]:
    """Library-path stop of a row whose containers came from a Sparksi run: the run's own stop."""
    if lock_on:
        return {"verdict": "REFUSE", "reasons": ["lock is ON"]}
    if str(row.get("id") or "").lower() in protect.protected_ids():
        return {"verdict": "REFUSE", "reasons": ["refusing to stop the protected video service"]}
    from sparksi.recipes.service import halt_run

    _code, body = halt_run(catalog, _runner(catalog, runner), run_id)
    stopped = bool(body.get("stopped"))
    error = body.get("error") or ("" if stopped else "a stop of this run is already in progress")
    return {
        "verdict": "READY" if stopped else "REFUSE",
        "stopped": stopped,
        "run_id": run_id,
        "error": error or None,
        "reasons": [error] if error else [],
    }


def _copy_shards(exec_runner: Any, plan: dict[str, Any]) -> bool:
    """Copy one safetensors shard at a time. Finished shards are left in place."""
    req = plan["request"]
    host = plan["host"]
    listing = exec_runner.invoke(
        host,
        {
            "op": "list_shards",
            "src": req["src"],
            "dest": req["dest"],
            "shards": req.get("shards") or 120,
            "min_bytes": req.get("min_bytes") or 100_000_000,
            "exact": bool(req.get("exact")),
        },
        timeout=120,
    )
    if not listing.get("ok"):
        _finish(False, listing.get("error") or "could not list shards")
        return False
    exact = bool(req.get("exact"))
    pending = list(listing.get("pending") or [])
    total = int(listing.get("total") or 0)
    done = len(listing.get("done") or [])
    for offset, name in enumerate(pending):
        if not _wait_if_paused():
            _finish(False, "cancelled", phase="cancelled", shards_done=done + offset, shards_total=total)
            return False
        nth = done + offset + 1
        _set(phase=f"{nth}/{total} {name}", host=host, shards_done=nth - 1, shards_total=total, shard=name)
        ratio = req.get("min_free_ratio")
        result = exec_runner.invoke(
            host,
            {
                "op": "stage_shard",
                "src": req["src"],
                "dest": req["dest"],
                "name": name,
                "min_free_ratio": 0.20 if ratio is None else ratio,
                "refuse_if_listening": req.get("refuse_if_listening") or [],
                "exact": exact,
            },
            timeout=6 * 3600,
        )
        if not result.get("ok"):
            _finish(False, result.get("error") or f"{name} failed", shard=name)
            return False
    for extra in ("model.safetensors.index.json", "config.json"):
        result = exec_runner.invoke(
            host,
            {
                "op": "stage_shard",
                "src": req["src"],
                "dest": req["dest"],
                "name": extra,
                "min_free_ratio": 0.0,
                "exact": exact,
            },
            timeout=600,
        )
        if exact and not result.get("ok"):
            error = str(result.get("error") or "")
            if not error.startswith("missing shard"):
                _finish(False, f"{extra}: {error}", shard=extra)
                return False
    if exact:
        recheck = exec_runner.invoke(
            host,
            {
                "op": "list_shards",
                "src": req["src"],
                "dest": req["dest"],
                "shards": req.get("shards") or 120,
                "min_bytes": req.get("min_bytes") or 100_000_000,
                "exact": True,
            },
            timeout=120,
        )
        if not recheck.get("ok"):
            _finish(False, "could not re-check the copy: " + str(recheck.get("error") or ""))
            return False
        still_pending = list(recheck.get("pending") or [])
        files_differ = list(recheck.get("files_differ") or [])
        if still_pending or files_differ:
            names = still_pending + files_differ
            _finish(
                False,
                f"{len(names)} file(s) still differ from storage after the copy: {', '.join(names[:5])}",
                shards_done=total - len(still_pending),
                shards_total=total,
            )
            return False
    _set(phase=f"{total}/{total}", shards_done=total, shards_total=total)
    return True


def _start_argv(catalog, kind: str, plan: dict[str, Any], runner) -> dict[str, Any]:
    argv = list(plan.get("argv") or [])

    def work() -> None:
        phase = "stopping" if kind == "stop" else "starting"
        _set(phase=phase, host=plan.get("host"))
        feed = "stop" if kind == "stop" else "run"
        verb = "unload" if kind == "stop" else "load"
        _audit(feed, f"{verb} {plan.get('id') or ''}".rstrip(), phase=phase, machine=str(plan.get("host") or ""))
        result = _runner(catalog, runner).run_argv(plan["host"], argv, timeout=3600)
        if not result.get("ok"):
            _finish(False, result.get("error") or f"exit {result.get('code')}", phase="failed", result=result)
            return
        _finish(True, phase="ready", result=result)

    started = _begin(kind, {"id_row": plan.get("id"), "hosts": [plan.get("host")]}, work)
    started["plan"] = plan
    return started


def restore_job(runner: Any = None, *, check: bool = True, resume: bool = False) -> dict[str, Any] | None:
    """Reload ``job.json``. An in-flight job is ``unknown`` until a health check."""
    global _JOB
    raw = _read_job_file()
    if not raw:
        return None
    inflight = raw.get("phase") not in _TERMINAL
    if inflight and raw.get("phase") != "unknown":
        raw = dict(raw)
        raw["phase"] = "unknown"
        raw["ok"] = None
    with _LOCK:
        _JOB = dict(raw)
    _persist()
    if not inflight or not check:
        return current()
    _resolve_restored(runner, resume=resume)
    return current()


def _resolve_restored(runner: Any, *, resume: bool) -> None:
    job = current() or {}
    health = job.get("health") or {}
    host = health.get("host")
    port = health.get("port")
    if not host or port in (None, ""):
        _set(phase="unknown")
        return
    try:
        port_i = int(port)
    except (TypeError, ValueError):
        _set(phase="unknown")
        return
    if protect.is_off_limits(port_i):
        _set(phase="unknown")
        return
    exec_runner = runner or SshRunner(None)
    started = float(job.get("started") or time.time())
    timeout_s = float(health.get("timeout_s") or DEFAULT_HEALTH_TIMEOUT_S)
    if time.time() - started >= timeout_s:
        _finish(False, f"port {port_i} did not answer", phase="timed_out")
        return
    paths = [str(item) for item in (health.get("http_paths") or []) if item]
    step = {
        "host": host,
        "port": port_i,
        "http_paths": paths,
        "placement": health.get("placement"),
        "sides": list(health.get("sides") or []),
        "stop_argv": list(health.get("stop_argv") or []),
        "health_timeout_s": timeout_s,
    }
    verdict = _health_verdict(exec_runner, str(host), port_i, paths)
    if verdict == "ready":
        _finish(True, phase="ready")
        if job.get("kind") == "switch":
            from sparksi.lock import default_lock_path

            relock_after_switch(default_lock_path())
        return
    if verdict == "fail":
        _finish(False, f"health on {host} failed", phase="failed")
        return
    _set(phase="waiting_health")
    if resume:
        _resume_health(exec_runner, step, started, timeout_s)


def _resume_health(exec_runner: Any, step: dict[str, Any], started: float, timeout_s: float) -> None:
    global _WORKER
    with _LOCK:
        if _WORKER:
            return
        _WORKER = True

    from sparksi.lock import default_lock_path

    lock_path = default_lock_path()

    def run() -> None:
        global _WORKER
        try:
            result = _poll_health(
                exec_runner,
                step,
                pause=1.0,
                timeout_s=timeout_s,
                sleep=time.sleep,
                now=time.time,
                attempts=8,
                begun=started,
            )
            extra: dict[str, Any] = {}
            if result.get("failed_side"):
                extra["failed_side"] = result["failed_side"]
            if result.get("ok"):
                _finish(True, phase="ready")
                relock_after_switch(lock_path)
            else:
                _finish(False, result.get("error"), phase=result.get("phase") or "failed", **extra)
        except Exception as exc:  # noqa: BLE001 — a restored poll must record the failure
            _finish(False, str(exc), phase="failed")
        finally:
            with _LOCK:
                _WORKER = False

    _thread(run)
