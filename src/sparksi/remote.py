# SPDX-License-Identifier: Apache-2.0

"""Run a catalog op on a Spark. Local when this host is that Spark; SSH otherwise."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

from sparksi.agent import main as agent_main
from sparksi.catalog import load_catalog
from sparksi.never import launcher_refusal, nvidia_smi_is_clock_lock, observation_refusal
from sparksi.config import CFG
from sparksi.hostcheck import check_ssh

AGENT_PATH = Path(__file__).resolve().parent / "agent.py"
Invoke = Callable[[str, dict[str, Any], int], dict[str, Any]]


def resolve_target(host_id: str, catalog: dict[str, Any] | None = None) -> str | None:
    """SSH destination, or None when the op should run in this process."""
    catalog = catalog if catalog is not None else load_catalog()
    spec = (catalog.get("hosts") or {}).get(host_id) or {}
    local_name = spec.get("local_when_hostname")
    if local_name and socket.gethostname() == local_name:
        return None
    ssh = spec.get("ssh") or ""
    if ssh in ("", "local"):
        return None
    # Never hand ssh a destination that could be read as an option.
    return check_ssh(str(ssh))


_RECIPE_OPS = frozenset({"run", "stop", "status", "service", "stop_outside", "files_check", "files_stage", "files_read", "file_sha256"})

# Reads that may skip a machine that just failed to connect. Actions always try.
_READ_OPS = frozenset({"probe", "list_shards"})
DOWN_TTL_S = 30.0
_DOWN: dict[str, tuple[float, str]] = {}
_DOWN_LOCK = threading.Lock()


def _recent_failure(target: str) -> str | None:
    with _DOWN_LOCK:
        hit = _DOWN.get(target)
        if hit is None:
            return None
        if time.monotonic() - hit[0] >= DOWN_TTL_S:
            _DOWN.pop(target, None)
            return None
        return f"{hit[1]} (not retried for {int(DOWN_TTL_S - (time.monotonic() - hit[0])) + 1} s)"


def _note_reach(target: str, error: str | None) -> None:
    with _DOWN_LOCK:
        if error is None:
            _DOWN.pop(target, None)
        else:
            _DOWN[target] = (time.monotonic(), error)


def forget_failures() -> None:
    with _DOWN_LOCK:
        _DOWN.clear()



# One SSH connection per Spark, reused (OpenSSH ControlMaster). Each agent call
# otherwise pays a TCP connect, key exchange and login on the node.
MASTER_RETRY_S = 30.0
_MASTER_TRIED: dict[str, float] = {}
_MASTER_LOCK = threading.Lock()


def _mux_dir() -> Path | None:
    if CFG.controller.get("ssh_reuse") is False:
        return None
    import sparksi.lock
    try:
        d = sparksi.lock.state_dir() / "ssh"
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        st = d.stat()
        if st.st_uid != os.getuid():
            return None
        if st.st_mode & 0o077:
            return None
        if len(str(d)) > 45:  # + "/%C" (41) + ssh's temp suffix (17) must fit a 108-byte socket path
            return None
    except OSError:
        return None
    return d


def _mux_path(target: str) -> str | None:
    d = _mux_dir()
    if d is None:
        return None
    return str(d / "%C")


def _ssh_opts(target: str, connect_timeout: int) -> list[str]:
    argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}"]
    path = _mux_path(target)
    if path is not None:
        _ensure_master(target, path)
        argv += ["-o", "ControlMaster=no", "-o", f"ControlPath={path}"]
    argv += ["--", target]
    return argv


def _ensure_master(target: str, path: str) -> None:
    now = time.monotonic()
    with _MASTER_LOCK:
        if now - _MASTER_TRIED.get(target, float("-inf")) < MASTER_RETRY_S:
            return
        _MASTER_TRIED[target] = now
    t = threading.Thread(target=_start_master, args=(target, path), daemon=True)
    t.start()


def _start_master(target: str, path: str) -> None:
    argv = [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=8",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={path}",
        "-o", "ControlPersist=120",
        "-o", "ServerAliveInterval=3",
        "-o", "ServerAliveCountMax=2",
        "-f", "-N",
        "--", target,
    ]
    try:
        subprocess.run(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def _drop_master(target: str) -> None:
    """After a call timed out, retire that Spark's master: new calls do not ride it.

    ``-O stop`` (not ``exit``): calls already on the master finish normally,
    and the master leaves once they are done or the link is found dead
    (ServerAliveInterval x ServerAliveCountMax, 3 s x 2). The next call
    connects on its own and starts a fresh master.
    """
    path = _mux_path(target)
    if path is None:
        return
    try:
        subprocess.run(
            ["ssh", "-o", f"ControlPath={path}", "-O", "stop", "--", target],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass
    with _MASTER_LOCK:
        _MASTER_TRIED.pop(target, None)


def forget_masters() -> None:
    with _MASTER_LOCK:
        _MASTER_TRIED.clear()


def recipe_target_error(host_id: str, catalog: dict[str, Any] | None = None) -> str | None:
    """Error when a recipe op has no Spark to run on.

    ``None`` means the host is a real SSH destination, or this process is that
    node (``ssh: local`` or ``local_when_hostname``). A missing catalog host,
    or a host with no ssh destination, must not fall back onto the controller.
    """
    catalog = catalog if catalog is not None else load_catalog()
    hosts = catalog.get("hosts") or {}
    spec = hosts.get(host_id)
    if not isinstance(spec, dict):
        return f"could not resolve {host_id}"
    local_name = spec.get("local_when_hostname")
    if local_name and socket.gethostname() == local_name:
        return None
    ssh = spec.get("ssh") or ""
    if ssh:
        return None
    return f"could not resolve {host_id}"


def with_config(req: dict[str, Any]) -> dict[str, Any]:
    """Attach the agent's site values so agent.py itself carries none."""
    out = dict(req)
    out.setdefault("_config", CFG.agent_config())
    return out


class LocalRunner:
    """Test double and same-machine path. `invoke(host, req, timeout) -> dict`."""

    def __init__(self, fn: Invoke | None = None, argv_fn: Invoke | None = None):
        self.fn = fn or (lambda host, req, timeout: agent_main(with_config(req)))
        self.argv_fn = argv_fn

    def invoke(self, host: str, req: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        out = self.fn(host, req, timeout)
        if not isinstance(out, dict):
            return {"ok": False, "error": "runner returned a non-object"}
        return out

    def run_argv(self, host: str, argv: list[str], timeout: int = 3600) -> dict[str, Any]:
        if any(isinstance(a, str) and a.rsplit("/", 1)[-1] == "nvidia-smi" for a in argv or []):
            return {"ok": False, "error": "refusing nvidia-smi on the launch path"}
        if self.argv_fn is None:
            return {"ok": False, "error": "no argv runner"}
        out = self.argv_fn(host, {"argv": argv}, timeout)
        return out if isinstance(out, dict) else {"ok": False, "error": "argv runner returned a non-object"}

    def capture(self, host: str, argv: list[str], timeout: int = 8) -> dict[str, Any]:
        if self.argv_fn is None:
            return {"ok": False, "error": "no argv runner", "stdout": "", "stderr": "", "code": None}
        out = self.argv_fn(host, {"argv": argv, "capture": True}, timeout)
        if not isinstance(out, dict):
            return {"ok": False, "error": "argv runner returned a non-object", "stdout": "", "stderr": "", "code": None}
        out.setdefault("stdout", out.get("tail") or "")
        out.setdefault("stderr", "")
        return out

    def run_gpu_clock(self, host: str, gpu_index: int, max_mhz: int | None, timeout: int = 8) -> dict[str, Any]:
        """Set or clear a GPU clock cap. Builds the allowed sudo -n argv from integers only."""
        from sparksi.gpu_clocks import clock_lock_argv

        try:
            argv = clock_lock_argv(gpu_index, max_mhz)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": "", "code": None}
        if not nvidia_smi_is_clock_lock(argv):
            return {"ok": False, "error": "refusing non-clock-lock command", "stdout": "", "stderr": "", "code": None}
        if self.argv_fn is None:
            return {"ok": False, "error": "no argv runner", "stdout": "", "stderr": "", "code": None}
        out = self.argv_fn(host, {"gpu_index": gpu_index, "max_mhz": max_mhz, "gpu_clock": True}, timeout)
        if not isinstance(out, dict):
            return {"ok": False, "error": "argv runner returned a non-object", "stdout": "", "stderr": "", "code": None}
        out.setdefault("stdout", out.get("tail") or "")
        out.setdefault("stderr", "")
        return out

    def agent_call(self, host: str, op: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
        """``sparksi-agent run|stop|status`` as a JSON op. The agent execs argv; it does not shell out."""
        req = dict(payload or {})
        req["op"] = op
        return self.invoke(host, req, timeout=timeout)


class SshRunner:
    def __init__(self, catalog: dict[str, Any] | None = None):
        self.catalog = catalog

    def invoke(self, host: str, req: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        req = with_config(req)
        # A recipe targets a Spark. If that node is not in the catalog, do not
        # exec it here via the local agent — that would run on the controller.
        if req.get("op") in _RECIPE_OPS:
            why = recipe_target_error(host, self.catalog)
            if why:
                return {"ok": False, "error": why, "host": host}
        try:
            target = resolve_target(host, self.catalog)
        except ValueError as exc:
            return {"ok": False, "error": f"refused ssh destination for {host}: {exc}", "host": host}
        if target is None:
            return agent_main(req)
        read = req.get("op") in _READ_OPS
        if read:
            recent = _recent_failure(target)
            if recent:
                return {"ok": False, "error": recent, "host": host, "unreachable": True}
        script = AGENT_PATH.read_text(encoding="utf-8")
        script = script.split('if __name__ == "__main__":', 1)[0]
        # Embed the request. `python3 -` would otherwise treat stdin as the
        # program and agent.py's __main__ would try to parse it again.
        code = (
            script
            + "\nimport json,sys\n"
            + "sys.stdout.write(json.dumps(main(json.loads("
            + json.dumps(json.dumps(req))
            + "))))\n"
        )
        try:
            proc = subprocess.run(
                [*_ssh_opts(target, 8), "python3", "-"],
                input=code,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            if isinstance(e, subprocess.TimeoutExpired):
                _drop_master(target)
            _note_reach(target, str(e)[-200:])
            return {"ok": False, "error": str(e), "host": host}
        if proc.returncode != 0 and not proc.stdout.strip():
            err = (proc.stderr or "").strip()[-400:]
            if proc.returncode == 255:  # ssh itself failed: name, route or auth
                _note_reach(target, (err or "ssh exit 255")[-200:])
            return {"ok": False, "error": err or f"ssh exit {proc.returncode}", "host": host}
        _note_reach(target, None)
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return {"ok": False, "error": (proc.stdout or proc.stderr or "no json")[-400:], "host": host}
        if not isinstance(data, dict):
            return {"ok": False, "error": "remote returned a non-object", "host": host}
        return data

    def run_argv(self, host: str, argv: list[str], timeout: int = 3600) -> dict[str, Any]:
        if not argv or any(not isinstance(a, str) for a in argv):
            return {"ok": False, "error": "empty argv"}
        if any(a.rsplit("/", 1)[-1] == "nvidia-smi" for a in argv):
            return {"ok": False, "error": "refusing nvidia-smi on the launch path"}
        blocked = launcher_refusal(argv)
        if blocked:
            return {"ok": False, "error": blocked}
        try:
            target = resolve_target(host, self.catalog)
        except ValueError as exc:
            return {"ok": False, "error": f"refused ssh destination for {host}: {exc}"}
        cmd = list(argv) if target is None else [*_ssh_opts(target, 8), *argv]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            if target is not None and isinstance(e, subprocess.TimeoutExpired):
                _drop_master(target)
            return {"ok": False, "error": str(e)}
        if target is not None and proc.returncode != 255:
            _note_reach(target, None)  # ssh got through: reads stop skipping this Spark now
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-400:]
        return {"ok": proc.returncode == 0, "code": proc.returncode, "tail": tail}

    def capture(self, host: str, argv: list[str], timeout: int = 8) -> dict[str, Any]:
        """Read-only argv on a node. Stdout is not truncated; ssh failures set code 255."""
        refused = observation_refusal(argv)
        if refused:
            return {"ok": False, "error": refused, "stdout": "", "stderr": "", "code": None}
        blocked = launcher_refusal(argv)
        if blocked:
            return {"ok": False, "error": blocked, "stdout": "", "stderr": "", "code": None}
        try:
            target = resolve_target(host, self.catalog)
        except ValueError as exc:
            return {"ok": False, "error": f"refused ssh destination for {host}: {exc}", "stdout": "", "stderr": "", "code": None}
        cmd = list(argv) if target is None else [*_ssh_opts(target, 3), *argv]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if target is not None and isinstance(exc, subprocess.TimeoutExpired):
                _drop_master(target)
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": "", "code": 255}
        if target is not None and proc.returncode != 255:
            _note_reach(target, None)  # ssh got through: reads stop skipping this Spark now
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "error": None if proc.returncode == 0 else ((proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()[:400]),
        }

    def run_gpu_clock(self, host: str, gpu_index: int, max_mhz: int | None, timeout: int = 8) -> dict[str, Any]:
        """Set or clear a GPU clock cap over SSH.

        The command line is built here from a validated GPU index and MHz, never
        from a caller-supplied argv, so only ``sudo -n nvidia-smi -i <gpu> ...``
        can run.
        """
        from sparksi.gpu_clocks import clock_lock_argv

        try:
            argv = clock_lock_argv(gpu_index, max_mhz)
        except ValueError as exc:
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": "", "code": None}
        if not nvidia_smi_is_clock_lock(argv):
            return {"ok": False, "error": "refusing non-clock-lock command", "stdout": "", "stderr": "", "code": None}
        try:
            target = resolve_target(host, self.catalog)
        except ValueError as exc:
            return {"ok": False, "error": f"refused ssh destination for {host}: {exc}", "stdout": "", "stderr": "", "code": None}
        cmd = list(argv) if target is None else [*_ssh_opts(target, 3), *argv]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            if target is not None and isinstance(exc, subprocess.TimeoutExpired):
                _drop_master(target)
            return {"ok": False, "error": str(exc), "stdout": "", "stderr": "", "code": 255}
        if target is not None and proc.returncode != 255:
            _note_reach(target, None)
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        error = None if proc.returncode == 0 else (stderr or stdout or f"exit {proc.returncode}").strip()[:400]
        return {
            "ok": proc.returncode == 0,
            "code": proc.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
        }

    def agent_call(self, host: str, op: str, payload: dict[str, Any] | None = None, timeout: int = 60) -> dict[str, Any]:
        """Ship ``{"op": "run"|"stop"|"status"}`` to the node agent and return its JSON."""
        req = dict(payload or {})
        req["op"] = op
        return self.invoke(host, req, timeout=timeout)


def get_runner(catalog: dict[str, Any] | None = None) -> SshRunner:
    return SshRunner(catalog)
