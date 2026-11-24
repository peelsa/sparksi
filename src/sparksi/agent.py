# SPDX-License-Identifier: Apache-2.0

"""Stdlib-only remote ops. The controller ships this file to a Spark over stdin.

No third-party imports. Paths come from the controller's catalog lookup, then
this process checks them again before any write or delete.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import stat
import struct
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

# --- COMPAT: node copy of sparksi.compat (stdlib only; tests/test_compat.py keeps it equal) ---
AGENT_VIDEO_KEY = "video_service"
LEGACY_AGENT_VIDEO_KEY = "comfy"
# Protect-preset values (sparksi.protect.agent_values). The controller sends them with
# every request, empty when no preset is on; these stand-ins apply only to a bare run.
_PRESET_KEYS = ("off_limits_ports", "video_process_substrings")

# Site values. The controller sends its config with every request as
# req["_config"] (see sparksi.config.Config.agent_config). These defaults
# match config/sparksi.example.toml so a bare run is still safe.
_CFG: dict = {
    "archive_root": "/mnt/archive",
    "blocked_prefixes": [
        "/mnt/archive",
        "/mnt/archive-raw",  # raw mount the archive is bind-mounted from
        "/home/USER-agent",
        "/home/USER/services/video",
    ],
    "hub_root": "/home/USER/.cache/huggingface/hub",
    "off_limits_ports": [8188],
    "video_process_substrings": ["comfy"],
    "too_broad": ["/", "/home", "/home/USER", "/home/USER/.cache", "/home/USER/services"],
    "direct_prefixes": ["198.18.0.", "198.18.1."],
    # Stand-in until ``configure`` loads the site file. The shell-entrypoint
    # check does not use this list: an unconfigured process refuses instead.
    "mount_roots": [
        "/home/USER/.cache/huggingface",
        "/srv/example/cache",
        "/srv/example/builds",
        "/srv/example/worker",
    ],
    # Exact argv lists a catalog service may run. Matched verbatim.
    "service_commands": [],
    # Folders on this Spark that hold copied recipe folders. Empty refuses every copy.
    "recipe_roots": [],
    "stop_unmanaged": False,
}

# True after configure() is given mount_roots from the site file. The baked-in
# list above is not a configured root list.
_MOUNTS_CONFIGURED = False


def _clean_roots(values) -> list[str]:
    found: list[str] = []
    for item in values or []:
        text = os.path.normpath(str(item or ""))
        if text in ("", ".", "/"):
            continue
        if text not in found:
            found.append(text)
    return found


def configure(values: dict | None) -> None:
    """Apply controller-sent site values (only known keys).

    ``mount_roots`` replaces the stand-in list, including when the site file
    has none. An empty list stays empty so a shell entrypoint is refused
    instead of matching a placeholder path.
    """
    global _MOUNTS_CONFIGURED
    for key, val in (values or {}).items():
        if key in _PRESET_KEYS:
            if isinstance(val, list):
                _CFG[key] = [str(item) for item in val]
            continue
        if key == "mount_roots":
            if isinstance(val, list):
                _CFG["mount_roots"] = _clean_roots(val)
                _MOUNTS_CONFIGURED = True
            continue
        if key == "recipe_roots":
            if isinstance(val, list):
                _CFG["recipe_roots"] = _clean_roots(val)
            continue
        if key in _CFG and val not in (None, "", []):
            _CFG[key] = list(val) if isinstance(_CFG[key], list) else str(val)


def archive_root() -> str:
    return (
        os.environ.get("SPARKSI_ARCHIVE_ROOT")
        or str(_CFG["archive_root"])
    )


def blocked_prefixes() -> tuple:
    return tuple(str(p) for p in _CFG["blocked_prefixes"])


def hub_root() -> str:
    return str(_CFG["hub_root"])


MIN_PARTS = 5  # / home <user> <project> <leaf> at minimum


def main(req: dict) -> dict:
    try:
        if not isinstance(req, dict):
            return {"ok": False, "error": "request must be an object"}
        if isinstance(req.get("_config"), dict):
            configure(req["_config"])
        op = req.get("op")
        if op == "probe":
            return _probe(req)
        if op == "stage_tar":
            return _stage_tar(req)
        if op == "stage_flat":
            return _stage_flat(req)
        if op == "list_shards":
            return _list_shards(req)
        if op == "stage_shard":
            return _stage_shard(req)
        if op == "drop":
            return _drop(req)
        if op == "run":
            return _recipe_run(req)
        if op == "stop":
            return _recipe_stop(req)
        if op == "status":
            return _recipe_status(req)
        if op == "evict_sha256":
            return _evict_sha256()
        if op == "service":
            return _service(req)
        if op == "stop_unmanaged":
            return _stop_unmanaged(req)
        if op == "stop_outside":
            return _stop_outside(req)
        if op == "files_check":
            return _files_check(req)
        if op == "files_stage":
            return _files_stage(req)
        if op == "files_read":
            return _files_read(req)
        if op == "file_sha256":
            return _file_sha256(req)
        return {"ok": False, "error": f"unknown op {op!r}"}
    except Exception as e:  # noqa: BLE001 — remote JSON must carry the failure
        return {"ok": False, "error": str(e)}


def _probe(req: dict) -> dict:
    mem = _meminfo()
    disk = shutil.disk_usage("/")
    ports = [int(p) for p in (req.get("ports") or [])]
    address = _probe_address(str(req.get("listen_host") or ""))
    ips = _ips()
    # A server bound only to this Spark's own tailnet address is closed on
    # loopback. Try that one address too; never another host, never an
    # off-limits port.
    own = _own_tailnet(ips) if address == "127.0.0.1" else ""
    listening: dict[str, bool] = {}
    dial: dict[str, str] = {}
    for p in ports:
        if _port_open(p, address):
            listening[str(p)], dial[str(p)] = True, address
        elif own and not _off_limits(p) and _port_open(p, own):
            listening[str(p)], dial[str(p)] = True, own
        else:
            listening[str(p)] = False
    paths = {}
    for raw in req.get("paths") or []:
        paths[str(raw)] = _path_stat(str(raw))
    reachable = _archive_reachable(archive_root())
    http_paths = {str(k): str(v) for k, v in (req.get("http_paths") or {}).items()}
    http = {}
    for port, up in listening.items():
        if up:
            path = http_paths.get(port) or "/health"
            via = dial.get(port, address)
            identity = _http_identity(int(port), path, via)
            if identity.get("status") != 200 and path != "/v1/models" and port not in http_paths:
                identity = _http_identity(int(port), "/v1/models", via)
            http[port] = identity
    out: dict = {
        "ok": True,
        "hostname": socket.gethostname(),
        "mem_total_gib": mem.get("MemTotal"),
        "mem_available_gib": mem.get("MemAvailable"),
        "disk_total_gib": round(disk.total / 1024**3, 2),
        "disk_free_gib": round(disk.free / 1024**3, 2),
        "disk_used_gib": round(disk.used / 1024**3, 2),
        "disk_used_pct": _used_pct(disk.used, disk.free),
        "mount_archive": reachable,
        "archive_store": _archive_store(archive_root()) if reachable else None,
        "listening": listening,
        "http": http,
        "containers": _running_containers(),
        "ips": ips,
        "paths": paths,
        "gpu": _gpu(),
    }
    if req.get("container_ports"):
        out["container_ports"] = _sparksi_container_ports(out.get("containers"), address)
    if req.get("unmanaged_ports"):
        out["unmanaged_containers"] = _unmanaged_container_ports(out.get("containers"), address)
    if req.get("mounts"):
        out["mounts"] = _container_mounts(out.get("containers"))
    if req.get("scan_releases"):
        out["releases"] = _scan_releases() if reachable else []
    flat = req.get("scan_flat")
    if flat:
        if reachable:
            try:
                out["flat"] = _flat_status(str(flat), int(req.get("shards") or 120))
            except OSError as e:
                out["flat"] = {"ok": False, "full": 0, "index": False, "error": str(e)}
        else:
            out["flat"] = {"ok": False, "full": 0, "index": False, "error": "archive host down"}
    return out


def _cifs_addr(mounts: str, root: str) -> str | None:
    """Server address for a CIFS mount, read from mount text. Does not stat the share."""
    target = root.rstrip("/") or "/"
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] != target or parts[2] != "cifs":
            continue
        for opt in parts[3].split(","):
            if opt.startswith("addr="):
                return opt.split("=", 1)[1] or None
    return None


def _mount_entry(mounts: str, root: str) -> tuple[str, str, str] | None:
    """(source, fstype, options) of the last real (non-autofs) mount at root."""
    target = root.rstrip("/") or "/"
    found = None
    for line in mounts.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[1] != target or parts[2] == "autofs":
            continue
        found = (parts[0], parts[2], parts[3])
    return found


def _fstab_server(fstab: str, root: str) -> tuple[str, int] | None:
    """(server, port) for a cifs/nfs fstab entry at root, for automounts not yet triggered."""
    target = root.rstrip("/") or "/"
    for line in fstab.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0].startswith("#") or parts[1] != target:
            continue
        src, fstype = parts[0], parts[2]
        if fstype in ("cifs", "smb3") and src.startswith("//"):
            return src[2:].split("/", 1)[0], 445
        if fstype.startswith("nfs") and ":" in src:
            return src.split(":", 1)[0], 2049
    return None


def _port_reachable(addr: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        return sock.connect_ex((addr, port)) == 0
    except OSError:
        return False
    finally:
        sock.close()


def _archive_reachable(root: str) -> bool:
    """False when the archive is missing or its server does not answer within a second.

    stat on a soft network mount whose server is down returns EHOSTDOWN only after
    the kernel gives up. That used to fail the whole probe and leave the harness
    list blank until the browser quit waiting. So network mounts are checked by a
    1 s TCP connect (445 for CIFS, 2049 for NFS) and never stat-ed here. A local
    filesystem (for example an ext4 DAS bind-mounted at the archive root)
    counts as reachable when it is mounted.
    """
    try:
        mounts = Path("/proc/mounts").read_text()
    except OSError:
        return False
    entry = _mount_entry(mounts, root)
    if entry:
        _, fstype, _ = entry
        if fstype in ("cifs", "smb3"):
            addr = _cifs_addr(mounts, root)
            return bool(addr) and _port_reachable(addr, 445)
        if fstype.startswith("nfs"):
            addr = _opt(entry[2], "addr")
            return bool(addr) and _port_reachable(addr, 2049)
        return True  # local block device (ext4 etc.)
    try:
        fstab = Path("/etc/fstab").read_text()
    except OSError:
        return False
    server = _fstab_server(fstab, root)  # x-systemd.automount not triggered yet
    return bool(server) and _port_reachable(*server)


def _opt(options: str, key: str) -> str | None:
    for opt in options.split(","):
        if opt.startswith(key + "="):
            return opt.split("=", 1)[1] or None
    return None


def _used_pct(used: int, free: int) -> float | None:
    # Same formula as df: used / (used + available).
    denom = used + free
    return round(100.0 * used / denom, 1) if denom else None


def _archive_store(root: str) -> dict | None:
    """Size of the archive's backing filesystem, only when it is local to this host.

    Network mounts return None so a probe never blocks on a remote statvfs; the
    host that owns the disk reports it.
    """
    try:
        mounts = Path("/proc/mounts").read_text()
    except OSError:
        return None
    entry = _mount_entry(mounts, root)
    if not entry or entry[1] in ("cifs", "smb3") or entry[1].startswith("nfs"):
        return None
    try:
        usage = shutil.disk_usage(root)
    except OSError:
        return None
    return {
        "path": root,
        "device": entry[0],
        "fstype": entry[1],
        "total_gib": round(usage.total / 1024**3, 2),
        "used_gib": round(usage.used / 1024**3, 2),
        "free_gib": round(usage.free / 1024**3, 2),
        "used_pct": _used_pct(usage.used, usage.free),
        **_store_bytes(root),
    }


def _store_bytes(root: str) -> dict:
    """Byte counts for the GB summary: free is raw free space, avail is what non-root can use."""
    try:
        st = os.statvfs(root)
    except OSError:
        return {}
    size = st.f_frsize
    return {
        "total_bytes": st.f_blocks * size,
        "used_bytes": (st.f_blocks - st.f_bfree) * size,
        "free_bytes": st.f_bfree * size,
        "avail_bytes": st.f_bavail * size,
    }


def _meminfo() -> dict:
    out: dict[str, float] = {}
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith("MemTotal:") or line.startswith("MemAvailable:"):
            key, rest = line.split(":", 1)
            kb = int(rest.strip().split()[0])
            out[key] = round(kb / 1024 / 1024, 2)
    return out


def _probe_address(raw: str) -> str:
    """A tailnet address from local config, or loopback when the request has none.

    Readiness passes the node's tailnet address. Other probes stay on loopback.
    """
    text = str(raw or "").strip()
    parts = text.split(".")
    if len(parts) != 4:
        return "127.0.0.1"
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return "127.0.0.1"
    if any(part < 0 or part > 255 for part in nums):
        return "127.0.0.1"
    if nums[0] == 100 and 64 <= nums[1] <= 127:
        return text
    return "127.0.0.1"


def _http_identity(port: int, path: str = "/v1/models", host: str = "127.0.0.1") -> dict:
    """What answers on ``host:port``. A bound socket alone is not "up".

    ``models`` holds OpenAI ``data[].id`` plus llama.cpp ``models[].name``; ``status``
    is the HTTP code (None when nothing answered within 1.5 s). Readiness uses
    ``/health`` or ``/v1/models`` on the address the recipe binds.
    """
    import urllib.error
    import urllib.request

    if not path.startswith("/"):
        path = "/" + path
    address = _probe_address(host)
    out: dict = {"path": path, "status": None, "models": []}
    try:
        with urllib.request.urlopen(f"http://{address}:{int(port)}{path}", timeout=1.5) as resp:
            out["status"] = int(resp.status)
            body = resp.read(65536).decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        out["status"] = int(e.code)
        return out
    except (OSError, ValueError):
        return out
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, ValueError):
        return out
    if not isinstance(data, dict):
        return out
    names: list[str] = []
    for item in data.get("data") or []:
        if isinstance(item, dict) and item.get("id"):
            names.append(str(item["id"]))
    for item in data.get("models") or []:
        if isinstance(item, dict):
            for key in ("name", "model"):
                if item.get(key) and str(item[key]) not in names:
                    names.append(str(item[key]))
    out["models"] = names[:8]
    if path == "/system_stats":
        out[AGENT_VIDEO_KEY] = isinstance(data.get("system"), dict) or "devices" in data
        out[LEGACY_AGENT_VIDEO_KEY] = out[AGENT_VIDEO_KEY]  # read by an older controller
    return out


def _off_limits(port) -> bool:
    """True for a port an active protect preset reserves."""
    return str(port).strip() in {str(p) for p in _CFG["off_limits_ports"]}


def _video_process(low: str) -> bool:
    return any(s and s in low for s in _CFG["video_process_substrings"])


def _sparksi_container_ports(names: list[str] | None, address: str) -> dict[str, int]:
    """Listen port of each running ``sparksi-*`` container, from its env or command."""
    found: dict[str, int] = {}
    for name in names or []:
        text = str(name or "")
        if not text.startswith("sparksi-") or not _safe_container_name(text):
            continue
        port = _container_listen_port(text, address)
        if port and not _off_limits(port):
            found[text] = port
    return found


def _unmanaged_name_ok(name: str) -> bool:
    """A running container Sparksi did not start and may offer to stop (never its own, the video service or the announcer)."""
    low = name.lower()
    return (
        _safe_container_name(name)
        and not name.startswith("-")
        and not low.startswith("sparksi-")
        and not _video_process(low)
        and "announce" not in low
    )


def _unmanaged_container_ports(names: list[str] | None, address: str) -> dict[str, int]:
    """Listen port of each running container that Sparksi did not start (read-only docker inspect)."""
    found: dict[str, int] = {}
    for name in names or []:
        text = str(name or "")
        if not _unmanaged_name_ok(text):
            continue
        port = _container_listen_port(text, address)
        if port and not _off_limits(port):
            found[text] = port
    return found


def _stop_unmanaged(req: dict) -> dict:
    """Stop one running container that Sparksi did not start, after checking it again here.

    Off unless the controller sets ``stop_unmanaged``. The container must still be
    running, must not be one of Sparksi's own (``sparksi-*``), the video service or the
    announcer, and must still listen on the port the page showed; never an off-limits port.
    Only ``docker stop`` is sent; nothing is removed or started.
    """
    if _CFG.get("stop_unmanaged") is not True:
        return {"ok": False, "stopped": False, "error": "stopping models started outside Sparksi is off"}
    name = str(req.get("container") or "")
    try:
        port = int(req.get("port"))
    except (TypeError, ValueError):
        return {"ok": False, "stopped": False, "error": "bad port"}
    if _off_limits(port) or not 1 <= port <= 65535:
        return {"ok": False, "stopped": False, "error": "that port is off limits"}
    if not _unmanaged_name_ok(name):
        return {"ok": False, "stopped": False, "error": "that container cannot be stopped from here"}
    running = _running_containers()
    if running is None:
        return {"ok": False, "stopped": False, "error": "docker could not be asked"}
    if name not in running:
        return {"ok": False, "stopped": False, "error": "that container is not running"}
    address = _probe_address(str(req.get("listen_host") or ""))
    if _container_listen_port(name, address) != port:
        return {"ok": False, "stopped": False, "error": "that container no longer serves this port"}
    try:
        proc = subprocess.run(
            ["docker", "stop", "--time", "30", name],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "stopped": False, "error": str(exc)[:400]}
    if proc.returncode != 0:
        return {"ok": False, "stopped": False, "error": (proc.stderr or proc.stdout or "docker stop failed").strip()[-400:]}
    return {"ok": True, "stopped": True, "container": name, "port": port}


# --- Emergency stop for a server started outside Sparksi (op stop_outside) ---
# The controller only asks for a port its page showed under "Started outside Sparksi".
# Everything is checked again here; nothing is started, removed or restarted.

# A listener with one of these names is plumbing, never a model server.
_OUTSIDE_KEEP_COMM = frozenset({
    "init", "systemd", "sshd", "sudo", "su", "login", "tmux", "tmux: server", "screen",
    "dockerd", "containerd", "docker-proxy", "containerd-shim", "containerd-shim-runc-v2",
    "tailscaled", "rpcbind", "smbd", "nmbd", "cupsd", "systemd-resolved", "nfsd",
})
# Container and process names that are web front ends or Sparksi's own helpers.
_OUTSIDE_KEEP_SUBSTRINGS = ("openwebui", "open-webui", "announce", "sparksi-agent")
# comm is cut to 15 characters, so long plumbing names are matched by prefix.
_OUTSIDE_KEEP_PREFIXES = ("systemd", "containerd", "docker", "tmux", "sshd", "dbus", "polkit", "agetty")
_OUTSIDE_GRACE_DEFAULT = 15
_OUTSIDE_GRACE_MAX = 30
_OUTSIDE_KILL_WAIT = 10
_OUTSIDE_PROC = "/proc"


def _outside_fail(error: str, **extra) -> dict:
    out = {"ok": False, "stopped": False, "error": error}
    out.update(extra)
    return out


def _listen_inodes(port: int) -> set[str]:
    """Socket inodes listening on TCP ``port`` (IPv4 and IPv6), from /proc/net."""
    found: set[str] = set()
    for name in ("tcp", "tcp6"):
        try:
            with open(os.path.join(_OUTSIDE_PROC, "net", name), encoding="ascii", errors="replace") as fh:
                lines = fh.read().splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 10 or parts[3] != "0A":  # 0A is LISTEN
                continue
            try:
                local_port = int(parts[1].rsplit(":", 1)[1], 16)
            except (IndexError, ValueError):
                continue
            if local_port == port and parts[9] != "0":
                found.add(parts[9])
    return found


def _pid_dirs() -> list[int]:
    try:
        names = os.listdir(_OUTSIDE_PROC)
    except OSError:
        return []
    return [int(n) for n in names if n.isdigit()]


def _socket_pids(inodes: set[str]) -> tuple[set[int], bool]:
    """(pids holding one of ``inodes``, whether some process fds could not be read)."""
    wanted = {f"socket:[{i}]" for i in inodes}
    pids: set[int] = set()
    hidden = False
    for pid in _pid_dirs():
        fd_dir = os.path.join(_OUTSIDE_PROC, str(pid), "fd")
        try:
            fds = os.listdir(fd_dir)
        except PermissionError:
            hidden = True
            continue
        except OSError:
            continue
        for fd in fds:
            try:
                if os.readlink(os.path.join(fd_dir, fd)) in wanted:
                    pids.add(pid)
                    break
            except OSError:
                continue
    return pids, hidden


def _proc_info(pid: int) -> dict | None:
    """ppid, pgid, start time, state, uid, comm and argv for ``pid``; None when it is gone."""
    base = os.path.join(_OUTSIDE_PROC, str(pid))
    try:
        with open(os.path.join(base, "stat"), encoding="utf-8", errors="replace") as fh:
            raw = fh.read()
        with open(os.path.join(base, "status"), encoding="utf-8", errors="replace") as fh:
            status = fh.read()
    except OSError:
        return None
    try:
        comm = raw[raw.index("(") + 1:raw.rindex(")")]
        rest = raw[raw.rindex(")") + 2:].split()
        state, ppid, pgid, start = rest[0], int(rest[1]), int(rest[2]), rest[19]
    except (ValueError, IndexError):
        return None
    uid = -1
    for line in status.splitlines():
        if line.startswith("Uid:"):
            try:
                uid = int(line.split()[1])
            except (IndexError, ValueError):
                uid = -1
            break
    try:
        with open(os.path.join(base, "cmdline"), "rb") as fh:
            argv = [a.decode("utf-8", "replace") for a in fh.read().split(b"\0") if a]
    except OSError:
        argv = []
    try:
        with open(os.path.join(base, "cgroup"), encoding="utf-8", errors="replace") as fh:
            cgroup = fh.read()
    except OSError:
        cgroup = ""
    return {"pid": pid, "ppid": ppid, "pgid": pgid, "start": start, "state": state,
            "uid": uid, "comm": comm, "argv": argv, "cgroup": cgroup}


def _plumbing(info: dict) -> bool:
    comm = str(info.get("comm") or "")
    return int(info.get("pid") or 0) <= 1 or comm in _OUTSIDE_KEEP_COMM or comm.startswith(_OUTSIDE_KEEP_PREFIXES)


def _in_container(info: dict) -> bool:
    text = str(info.get("cgroup") or "")
    return any(mark in text for mark in ("docker", "containerd", "kubepods", "libpod"))


def _own_lineage() -> set[int]:
    """This process and every ancestor, so the stop can never reach its own session."""
    seen: set[int] = set()
    pid = os.getpid()
    while pid > 0 and pid not in seen:
        seen.add(pid)
        info = _proc_info(pid)
        if not info:
            break
        pid = int(info["ppid"])
    return seen


def _outside_targets(listeners: set[int]) -> list[dict]:
    """The listeners, every descendant, and the members of a process group a target leads."""
    infos = [i for i in (_proc_info(p) for p in _pid_dirs()) if i]
    by_parent: dict[int, list[dict]] = {}
    for info in infos:
        by_parent.setdefault(int(info["ppid"]), []).append(info)
    chosen: dict[int, dict] = {}
    queue = [i for i in infos if i["pid"] in listeners]
    while queue:
        info = queue.pop()
        if info["pid"] in chosen:
            continue
        chosen[info["pid"]] = info
        queue.extend(by_parent.get(info["pid"], []))
    leaders = {pid for pid, info in chosen.items() if info["pgid"] == pid}
    for info in infos:
        if info["pgid"] in leaders and info["pid"] not in chosen:
            chosen[info["pid"]] = info
    return sorted(chosen.values(), key=lambda i: i["pid"])


def _outside_kept(text: str, keep: list[str]) -> bool:
    """True when a name or argv matches a protected name: substrings for built-ins, whole parts for the catalog."""
    low = str(text or "").lower()
    if not low:
        return False
    if _video_process(low) or any(s in low for s in _OUTSIDE_KEEP_SUBSTRINGS):
        return True
    parts = set(re.split(r"[\s/=,:]+", low))
    return any(k and (k in parts or k == low) for k in keep)


def _gpu_apps() -> dict[int, int] | None:
    """{pid: MiB} of GPU compute processes, or None when nvidia-smi could not be asked."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    out: dict[int, int] = {}
    for line in (proc.stdout or "").splitlines():
        bits = [b.strip() for b in line.split(",")]
        try:
            out[int(bits[0])] = int(float(bits[1])) if len(bits) > 1 else 0
        except (IndexError, ValueError):
            continue
    return out


def _alive(info: dict) -> bool:
    now = _proc_info(int(info["pid"]))
    return bool(now) and now["start"] == info["start"] and now["state"] not in ("Z", "X")


def _shown_kill_command(sig: str, pids: list[int]) -> str:
    """The exact command for a person to run for another user's process: integers only.

    Sparksi never runs it: sudo is used only for the GPU clock cap.
    """
    return " ".join(["sudo", "kill", "-s", sig, "--", *[str(int(p)) for p in pids]])


def _signal_one(info: dict, signum: int) -> str:
    """Signal one target only if it is still the same process: "sent", "gone" or "denied".

    A pidfd pins the process before its start time is checked, so a reused PID is never hit.
    """
    pid = int(info["pid"])
    pidfd = -1
    opener = getattr(os, "pidfd_open", None)
    sender = getattr(signal, "pidfd_send_signal", None)
    if opener is not None and sender is not None:
        try:
            pidfd = opener(pid)
        except ProcessLookupError:
            return "gone"
        except OSError:
            pidfd = -1
    try:
        if not _alive(info):
            return "gone"
        if pidfd >= 0:
            sender(pidfd, signum)
        else:
            os.kill(pid, signum)
        return "sent"
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "denied"
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def _send(sig: str, targets: list[dict]) -> list[int]:
    """Signal each live target directly; return the PIDs that refused (never retried with sudo)."""
    signum = getattr(signal, "SIG" + sig)
    return [int(info["pid"]) for info in targets if _signal_one(info, signum) == "denied"]


def _wait_gone(targets: list[dict], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not any(_alive(t) for t in targets):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _wait_closed(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while True:
        if not _listen_inodes(port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _stop_outside(req: dict) -> dict:
    """Emergency stop of a server Sparksi did not start, found by the port it listens on.

    A container named by the controller gets ``docker stop`` after the same checks as the
    page's container Stop. Otherwise the process listening on the port, its descendants and
    any process group one of them leads get SIGTERM, then SIGKILL after a grace period.
    Sparksi never uses sudo here: when any of those processes belongs to another user, or is
    hidden from this user, nothing is signalled and the reply carries the exact command for a
    person to run instead. A port served by a running ``sparksi-*`` container is always refused.
    The reply says whether the port closed and whether the GPU let go of the stopped processes.
    """
    try:
        port = int(req.get("port"))
    except (TypeError, ValueError):
        return _outside_fail("bad port")
    if not 1024 <= port <= 65535 or _off_limits(port):
        return _outside_fail("that port is off limits")
    keep = [str(k).strip().lower() for k in (req.get("keep") or []) if str(k).strip()]
    try:
        grace = int(req.get("grace") or _OUTSIDE_GRACE_DEFAULT)
    except (TypeError, ValueError):
        grace = _OUTSIDE_GRACE_DEFAULT
    grace = max(1, min(grace, _OUTSIDE_GRACE_MAX))
    container = str(req.get("container") or "")
    if container:
        return _stop_outside_container(container, port, keep, req)
    address = _probe_address(str(req.get("listen_host") or ""))
    running = _running_containers()
    if running is not None and port in _sparksi_container_ports(running, address).values():
        return _outside_fail("that port is served by a Sparksi container; stop it from its run")
    inodes = _listen_inodes(port)
    if not inodes:
        return _outside_fail("nothing is listening on that port now", port_closed=True)
    pids, hidden = _socket_pids(inodes)
    if not pids:
        if hidden:
            # It may be another user's server or a container's: nothing is signalled and no
            # kill command is guessed; the shown command only says what holds the port.
            return _outside_fail(
                "the process on that port is hidden from Sparksi (another user's or inside a container); "
                "nothing was stopped. Check what holds the port",
                command=f"sudo ss -ltnp 'sport = :{port}'",
            )
        return _outside_fail("could not find the process listening on that port")
    mine = _own_lineage()
    targets = _outside_targets(pids)
    if any(t["pid"] in mine for t in targets):
        return _outside_fail("that port belongs to Sparksi's own session; nothing was stopped")
    listeners = [t for t in targets if t["pid"] in pids]
    if not listeners:
        return _outside_fail("could not find the process listening on that port")
    for info in listeners:
        if _plumbing(info):
            return _outside_fail(f"port {port} belongs to {info['comm']}, which Sparksi never stops")
        if _in_container(info):
            return _outside_fail("that port is served from a container Sparksi could not name; stop the container on that Spark")
    for info in targets:
        if _plumbing(info) or _in_container(info):
            return _outside_fail("a process in that server's group is outside it; nothing was stopped")
        if _outside_kept(info["comm"], keep) or _outside_kept(" ".join(info["argv"]), keep):
            return _outside_fail("that server is protected on this Spark; nothing was stopped")
    target_pids = [int(t["pid"]) for t in targets]
    me = os.getuid()
    others = [int(t["pid"]) for t in targets if t["uid"] != me]
    if others:
        # sudo is allowed only for the GPU clock cap: stop nothing and show the command.
        return _outside_fail(
            "that server belongs to another user; nothing was stopped",
            command=_shown_kill_command("TERM", target_pids), pids=target_pids,
        )
    gpu_before = _gpu_apps()
    used_before = sum((gpu_before or {}).get(p, 0) for p in target_pids)
    denied = _send("TERM", targets)
    if denied:
        return _outside_fail(
            "some of that server's processes refused the signal",
            command=_shown_kill_command("TERM", denied), pids=target_pids,
        )
    signal_used = "TERM"
    if not _wait_gone(targets, grace):
        signal_used = "KILL"
        denied = _send("KILL", targets)
        if denied:
            return _outside_fail(
                "the server did not stop and some processes refused SIGKILL",
                command=_shown_kill_command("KILL", denied), pids=target_pids,
            )
        _wait_gone(targets, _OUTSIDE_KILL_WAIT)
    left = [int(t["pid"]) for t in targets if _alive(t)]
    closed = _wait_closed(port, 3)
    gpu_after = _gpu_apps()
    gpu_left = sorted(p for p in target_pids if gpu_after is not None and p in gpu_after)
    stopped = not left and closed
    out = {
        "ok": stopped, "stopped": stopped, "port": port, "pids": target_pids, "signal": signal_used,
        "port_closed": closed, "left": left, "gpu_checked": gpu_after is not None,
        "gpu_pids_left": gpu_left, "gpu_freed_mib": used_before if gpu_after is not None and not gpu_left else 0,
    }
    if not stopped:
        out["error"] = "the server did not fully stop" if left else "the port is still open"
    return out


def _stop_outside_container(name: str, port: int, keep: list[str], req: dict) -> dict:
    if not _unmanaged_name_ok(name) or _outside_kept(name, keep):
        return _outside_fail("that container cannot be stopped from here")
    running = _running_containers()
    if running is None:
        return _outside_fail("docker could not be asked")
    if name not in running:
        return _outside_fail("that container is not running")
    address = _probe_address(str(req.get("listen_host") or ""))
    if _container_listen_port(name, address) != port:
        return _outside_fail("that container no longer serves this port")
    gpu_before = _gpu_apps()
    try:
        proc = subprocess.run(
            ["docker", "stop", "--time", "20", name],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _outside_fail(str(exc)[:400])
    if proc.returncode != 0:
        return _outside_fail((proc.stderr or proc.stdout or "docker stop failed").strip()[-400:])
    closed = _wait_closed(port, 10)
    gpu_after = _gpu_apps()
    freed = 0
    if gpu_before is not None and gpu_after is not None:
        freed = max(0, sum(gpu_before.values()) - sum(gpu_after.values()))
    return {
        "ok": closed, "stopped": closed, "container": name, "port": port, "signal": "docker stop",
        "port_closed": closed, "gpu_checked": gpu_after is not None, "gpu_freed_mib": freed,
        **({} if closed else {"error": "the container stopped but the port is still open"}),
    }


def _safe_container_name(name: str) -> bool:
    return bool(name) and all(ch.isalnum() or ch in "_.-" for ch in name)


def _container_listen_port(name: str, address: str) -> int:
    declared = _ports_from_inspect(_inspect_env_cmd(name))
    for port in declared:
        if port and not _off_limits(port) and _port_open(port, address):
            return port
    for port in declared:
        if port and not _off_limits(port):
            return port
    return 0


def _inspect_env_cmd(name: str) -> str:
    try:
        proc = subprocess.run(
            [
                "docker", "inspect", "--format",
                "{{range .Config.Env}}{{println .}}{{end}}\n---CMD---\n{{range .Config.Cmd}}{{println .}}{{end}}",
                name,
            ],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return proc.stdout or ""


def _ports_from_inspect(blob: str) -> list[int]:
    """``PORT``, ``VLLM_PORT``, ``SERVER_PORT``, then ``--port``, in that order."""
    env_ports: list[int] = []
    cmd_ports: list[int] = []
    command = False
    prev = ""
    for line in str(blob or "").splitlines():
        if line.strip() == "---CMD---":
            command = True
            prev = ""
            continue
        if not command:
            if "=" in line:
                key, value = line.split("=", 1)
                if key in ("PORT", "VLLM_PORT", "SERVER_PORT"):
                    number = _inspect_port(value)
                    if number:
                        env_ports.append(number)
            continue
        if line.startswith("--port="):
            number = _inspect_port(line.split("=", 1)[1])
            if number:
                cmd_ports.append(number)
        elif prev == "--port":
            number = _inspect_port(line)
            if number:
                cmd_ports.append(number)
        prev = line
    return env_ports + cmd_ports


def _inspect_port(value: str) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def _running_containers() -> list[str] | None:
    """Names of running docker containers, or None when docker could not be asked."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", "status=running", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=3, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return sorted(n for n in proc.stdout.split() if n)


def _container_mounts(names: list[str] | None) -> dict[str, list[str]] | None:
    """Host paths each running container mounts (``docker inspect``, read-only).

    None when docker could not be asked. The controller uses it to tell which
    model's weights an engine it did not start is serving.
    """
    if names is None:
        return None
    if not names:
        return {}
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{.Name}}{{range .Mounts}}\t{{.Source}}{{end}}", *names[:32]],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 and not proc.stdout.strip():
        return None
    out: dict[str, list[str]] = {}
    for line in proc.stdout.splitlines():
        bits = line.split("\t")
        name = bits[0].strip().lstrip("/")
        if name:
            out[name] = [b.strip() for b in bits[1:] if b.strip().startswith("/")][:32]
    return out


def _ips() -> dict:
    """LAN (default-route interface), Tailscale and ConnectX direct-link IPv4s, read live."""
    out: dict = {"lan": None, "tailscale": None, "direct": []}
    try:
        addrs = subprocess.run(["ip", "-4", "-o", "addr", "show"], capture_output=True, text=True, timeout=2).stdout
        route = subprocess.run(["ip", "-4", "route", "show", "default"], capture_output=True, text=True, timeout=2).stdout
    except (OSError, subprocess.SubprocessError):
        return out
    default_dev = None
    parts = route.split()
    if "dev" in parts:
        default_dev = parts[parts.index("dev") + 1]
    for line in addrs.splitlines():
        f = line.split()
        if len(f) < 4 or f[2] != "inet":
            continue
        dev, ip = f[1], f[3].split("/", 1)[0]
        if dev == "tailscale0" or ip.startswith("100."):
            out["tailscale"] = out["tailscale"] or ip
        elif dev == default_dev:
            out["lan"] = out["lan"] or ip
        elif any(ip.startswith(p) for p in _CFG["direct_prefixes"]):
            out["direct"].append(ip)
    return out


def _own_tailnet(ips: dict | None) -> str:
    """This Spark's own tailnet IPv4 from ``_ips()``, or "" when it has none."""
    text = str((ips or {}).get("tailscale") or "").strip()
    return text if text and _probe_address(text) == text else ""


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex((_probe_address(host), int(port))) == 0
    except OSError:
        return False
    finally:
        s.close()


def _path_stat(raw: str) -> dict:
    p = Path(raw)
    if not p.exists():
        return {"exists": False, "nonempty": False}
    nonempty = True
    if p.is_dir():
        try:
            nonempty = next(p.iterdir(), None) is not None
        except OSError:
            nonempty = False
    return {"exists": True, "nonempty": nonempty}


def _gpu() -> dict:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,clocks.sm",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    parts = [x.strip() for x in proc.stdout.strip().splitlines()[0].split(",")]
    def num(i: int):
        try:
            return float(parts[i])
        except (IndexError, ValueError):
            return None
    temp_c = num(1)
    return {
        "util_pct": num(0),
        "temp_c": temp_c,
        "temp_f": None if temp_c is None else round(temp_c * 9 / 5 + 32, 1),
        "sm_mhz": num(2),
    }


def _scan_releases() -> list[dict]:
    root = Path(archive_root()) / "models" / "releases"
    if not root.is_dir():
        return []
    rows = []
    for slug_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        metas = sorted(slug_dir.glob("*.meta.json"))
        if not metas:
            continue
        try:
            meta = json.loads(metas[0].read_text())
        except (OSError, json.JSONDecodeError):
            continue
        tar = metas[0].name[: -len(".meta.json")]
        tar_path = slug_dir / tar
        rows.append(
            {
                "slug": slug_dir.name,
                "title": meta.get("dc:title") or slug_dir.name,
                "bytes": int(meta.get("ea:bytes") or 0),
                "sha256": meta.get("ea:sha256") or "",
                "tar": str(tar_path),
                "source": meta.get("ea:source") or "",
                "tar_exists": tar_path.is_file(),
            }
        )
    return rows


def _flat_status(path: str, shards: int) -> dict:
    root = Path(path)
    if not root.is_dir():
        return {"ok": False, "full": 0, "index": False, "error": "flat folder missing"}
    names = list(root.glob(f"model-*-of-{shards:05d}.safetensors"))
    full = [p for p in names if p.is_file() and p.stat().st_size >= 100_000_000]
    index = (root / "model.safetensors.index.json").is_file()
    return {"ok": len(full) == shards and index, "full": len(full), "index": index, "names": len(names)}


def _weight_path(raw: str) -> Path:
    if not isinstance(raw, str) or not raw.startswith("/"):
        raise ValueError("path must be absolute")
    if ".." in Path(raw).parts:
        raise ValueError("path must not contain ..")
    p = Path(raw).resolve()
    text = str(p)
    if text in tuple(str(p) for p in _CFG["too_broad"]):
        raise ValueError("path is too broad")
    if len(p.parts) < MIN_PARTS:
        raise ValueError("path is too shallow")
    for prefix in blocked_prefixes():
        if text == prefix or text.startswith(prefix + "/"):
            raise ValueError(f"refusing path under {prefix}")
    if text == hub_root() or (text.startswith(hub_root() + "/") and "/models--" not in text):
        raise ValueError("refusing the hub root")
    return p


def _min_free_ratio(req: dict) -> float:
    """Floor for a write. Missing means 20%; an explicit 0 disables it.

    ``0.0`` is a real value. ``or 0.20`` would treat it as missing.
    """
    raw = req.get("min_free_ratio")
    if raw is None:
        return 0.20
    return float(raw)


def _free_ok(dest: Path, need: int, min_free_ratio: float) -> None:
    parent = dest if dest.exists() and dest.is_dir() else dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(parent)
    ratio = float(min_free_ratio)
    if usage.free - int(need) < ratio * usage.total:
        raise ValueError(
            f"refusing write: free {usage.free} minus {need} would fall through {ratio:.0%} of {usage.total}"
        )


def _refuse_listening(ports: list) -> None:
    for port in ports or []:
        if _port_open(int(port)):
            raise ValueError(f"refusing while port {port} is listening")


def _fadvise(fd: int) -> None:
    if hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY)
    try:
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
        _fadvise(fd)
    finally:
        os.close(fd)
    return h.hexdigest()


def _stage_tar(req: dict) -> dict:
    _refuse_listening(req.get("refuse_if_listening") or [])
    dest = _weight_path(str(req.get("dest") or ""))
    tar_path = Path(str(req.get("tar") or ""))
    if not str(tar_path).startswith(archive_root() + "/"):
        return {"ok": False, "error": "tar must live on the archive"}
    if not tar_path.is_file():
        return {"ok": False, "error": "tar missing"}
    if dest.exists() and _path_stat(str(dest))["nonempty"]:
        return {"ok": False, "error": "dest is not empty; drop the local copy first"}
    need = int(req.get("bytes") or tar_path.stat().st_size)
    _free_ok(dest, need, _min_free_ratio(req))
    digest = _sha256(tar_path)
    expect = str(req.get("sha256") or "")
    if not expect or digest != expect:
        return {"ok": False, "error": "sha256 does not match the sidecar", "sha256": digest}
    dest.mkdir(parents=True, exist_ok=True)
    _extract_release(tar_path, dest)
    for dirpath, _, files in os.walk(dest):
        for name in files:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                _fadvise(fd)
            finally:
                os.close(fd)
    return {"ok": True, "dest": str(dest), "sha256": digest}


def _extract_release(tar_path: Path, dest: Path) -> None:
    """Unpack a release tar into ``dest``.

    Release tars are often packed as ``<name>/...``. When every member sits in one
    top folder named like ``dest``, that folder is dropped, so the files land in
    ``dest`` and not in ``dest/<name>``.
    """
    with tarfile.open(tar_path, "r:") as tf:
        members = tf.getmembers()
        if _shared_top(members, dest.name):
            members = _strip_top(members, dest.name)
        tf.extractall(dest, members=members, filter="data")


def _shared_top(members: list[tarfile.TarInfo], name: str) -> bool:
    """True when every member is the folder ``name`` or sits under it."""
    if not members or not name:
        return False
    return all(m.name.rstrip("/") == name or m.name.startswith(name + "/") for m in members)


def _strip_top(members: list[tarfile.TarInfo], name: str) -> list[tarfile.TarInfo]:
    """The members without the top folder ``name``; hard links follow their targets."""
    top = name + "/"
    out = []
    for m in members:
        if m.name.rstrip("/") == name:
            continue  # the folder itself is ``dest``
        link = m.linkname
        if m.islnk() and link.startswith(top):
            link = link[len(top):]
        out.append(m.replace(name=m.name[len(top):], linkname=link, deep=False))
    return out


def _shard_names(root: Path, shards: int) -> list[str]:
    return sorted(p.name for p in root.glob(f"model-*-of-{shards:05d}.safetensors") if p.is_file())


def _safe_shard_name(name: str) -> str:
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("bad shard name")
    if not name.endswith(".safetensors") and not name.endswith(".json"):
        raise ValueError("bad shard name")
    return name


EXTRA_FILES = ("model.safetensors.index.json", "config.json")


def _list_shards(req: dict) -> dict:
    src = Path(str(req.get("src") or ""))
    dest = _weight_path(str(req.get("dest") or ""))
    if not str(src).startswith(archive_root() + "/"):
        return {"ok": False, "error": "flat source must live on the archive"}
    shards = int(req.get("shards") or 120)
    min_bytes = int(req.get("min_bytes") or 100_000_000)
    exact = bool(req.get("exact"))
    names = _shard_names(src, shards)
    done, pending = [], []
    for name in names:
        target = dest / name
        source = src / name
        if exact:
            # Sync check: a shard matches only when its size equals the stored one.
            same = target.is_file() and source.is_file() and target.stat().st_size == source.stat().st_size
        else:
            same = target.is_file() and target.stat().st_size >= min_bytes
        (done if same else pending).append(name)
    files_differ: list[str] = []
    if exact:
        for extra in EXTRA_FILES:
            target = dest / extra
            source = src / extra
            if source.is_file() and (not target.is_file() or target.stat().st_size != source.stat().st_size):
                files_differ.append(extra)
    return {
        "ok": True,
        "total": len(names),
        "done": done,
        "pending": pending,
        "files_differ": files_differ,
        "min_bytes": min_bytes,
        "exact": exact,
    }


def _shard_rsync_argv(source: Path, target: Path, exact: bool) -> tuple[list[str], str]:
    name = target.name
    if not exact or not target.is_file() or (
        target.stat().st_size < source.stat().st_size and name.endswith(".safetensors")
    ):
        return (["rsync", "-a", "--partial", "--append-verify", str(source), str(target)], "resume")
    if name.endswith(".json"):
        return (["rsync", "-a", "--checksum", str(source), str(target)], "replace")
    return (["rsync", "-a", str(source), str(target)], "replace")


def _stage_shard(req: dict) -> dict:
    _refuse_listening(req.get("refuse_if_listening") or [])
    src = Path(str(req.get("src") or ""))
    dest = _weight_path(str(req.get("dest") or ""))
    name = _safe_shard_name(str(req.get("name") or ""))
    if not str(src).startswith(archive_root() + "/"):
        return {"ok": False, "error": "flat source must live on the archive"}
    source = src / name
    if not source.is_file():
        return {"ok": False, "error": f"missing shard {name}"}
    dest.mkdir(parents=True, exist_ok=True)
    _free_ok(dest, source.stat().st_size, _min_free_ratio(req))
    argv, mode = _shard_rsync_argv(source, dest / name, bool(req.get("exact")))
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "rsync failed")[-400:]}
    try:
        fd = os.open(dest / name, os.O_RDONLY)
    except OSError:
        fd = None
    if fd is not None:
        try:
            _fadvise(fd)
        finally:
            os.close(fd)
    return {"ok": True, "name": name, "mode": mode, "bytes": (dest / name).stat().st_size}


def _stage_flat(req: dict) -> dict:
    _refuse_listening(req.get("refuse_if_listening") or [])
    src = Path(str(req.get("src") or ""))
    dest = _weight_path(str(req.get("dest") or ""))
    if not str(src).startswith(archive_root() + "/"):
        return {"ok": False, "error": "flat source must live on the archive"}
    shards = int(req.get("shards") or 120)
    before = _flat_status(str(src), shards)
    if not before["ok"]:
        return {"ok": False, "error": "archive flat folder is incomplete", "flat": before}
    need = int(req.get("bytes") or 0)
    _free_ok(dest, need, _min_free_ratio(req))
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["rsync", "-a", "--partial", "--exclude", ".cache", "--exclude", "*.partial", f"{src}/", f"{dest}/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"ok": False, "error": (proc.stderr or proc.stdout or "rsync failed")[-400:]}
    after = _flat_status(str(dest), shards)
    if not after["ok"]:
        return {"ok": False, "error": "local flat folder is incomplete", "flat": after}
    for dirpath, _, files in os.walk(dest):
        for name in files:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
            except OSError:
                continue
            try:
                _fadvise(fd)
            finally:
                os.close(fd)
    return {"ok": True, "dest": str(dest), "flat": after}


def _drop(req: dict) -> dict:
    _refuse_listening(req.get("refuse_if_listening") or [])
    if req.get("tar"):
        tar_path = Path(str(req["tar"]))
        if not str(tar_path).startswith(archive_root() + "/") or not tar_path.is_file():
            return {"ok": False, "error": "archive tar is missing"}
        digest = _sha256(tar_path)
        if digest != str(req.get("sha256") or ""):
            return {"ok": False, "error": "archive tar sha256 does not match"}
    if req.get("verify_flat"):
        flat = _flat_status(str(req["verify_flat"]), int(req.get("shards") or 120))
        if not flat.get("ok"):
            return {"ok": False, "error": "archive flat folder is incomplete", "flat": flat}
    dest = _weight_path(str(req.get("path") or ""))
    if str(req.get("confirm") or "") != str(dest):
        return {"ok": False, "error": "confirm must equal the catalog path"}
    if not dest.exists():
        return {"ok": True, "deleted": False, "path": str(dest), "note": "already absent"}
    if dest.is_symlink():
        return {"ok": False, "error": "refusing to delete a symlink"}
    shutil.rmtree(dest)
    return {"ok": True, "deleted": True, "path": str(dest)}


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_DOCKER_SUBCOMMANDS = frozenset({"run", "stop", "rm", "logs", "inspect"})


def _runs_dir() -> Path:
    """Node-local run state. ``SPARKSI_STATE_DIR`` keeps tests off the home directory."""
    state = os.environ.get("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "runs"
    return Path.home() / ".local" / "state" / "sparksi" / "runs"


def _run_file(run_id: str) -> Path:
    return _runs_dir() / f"{run_id}.json"


def _safe_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _RUN_ID.match(run_id):
        raise ValueError("bad run id")
    return run_id


def _executable_name(token: str) -> str:
    if token.startswith("/"):
        return token.rsplit("/", 1)[-1]
    return token


def _under_mount_root(token: str) -> bool:
    if not str(token).startswith("/"):
        return False
    path = os.path.normpath(token)
    for root in _mount_roots():
        if path == root or path.startswith(root + os.sep):
            return True
    return False


def _argv0_allowed(token: str) -> bool:
    """``docker``, ``llama-server`` on PATH, or an absolute binary under a mount root.

    ``trtllm-serve`` and ``mpirun`` are not host executables. They are allowed
    only as the command inside ``docker run``. An absolute ``llama-server``
    is still accepted when its basename is ``llama-server``.
    """
    if token == "docker":
        return True
    if token == "llama-server":
        return True
    if token.startswith("/") and _executable_name(token) == "llama-server":
        return True
    return _under_mount_root(token)


# ``docker run`` options that consume the next argv element when they have no ``=``.
_DOCKER_VALUE_FLAGS = frozenset({
    "-v", "--volume", "--mount", "--network", "--net", "--pid",
    "--cap-add", "--cap-drop", "--security-opt", "--device", "--entrypoint",
    "--gpus", "--name", "-e", "--env", "-p", "--publish", "--expose",
    "--ulimit", "--shm-size", "--ipc", "--user", "-u", "-w", "--workdir",
    "--label", "-l", "--hostname", "--add-host", "--group-add", "--runtime",
    "--memory", "-m", "--cpus", "--cpu-shares", "--restart",
    "--log-driver", "--log-opt", "--tmpfs", "--sysctl", "--pull", "--platform",
    "--cgroupns", "--userns", "--uts", "--memory-swap", "--health-cmd",
    "--health-interval", "--device-cgroup-rule",
})
# GPU device nodes may be passed with ``--device``. Generated recipes use
# ``--gpus`` for the GPU and ``--device /dev/infiniband`` for two-node TensorRT-LLM.
_GPU_DEVICE_PREFIXES = ("/dev/nvidia", "/dev/dri")
_FABRIC_DEVICE = "/dev/infiniband"


def _docker_refusal(argv: list, *, allow_roce: bool = False) -> str | None:
    if len(argv) < 2:
        return "refusing executable docker"
    sub = argv[1]
    if sub == "image":
        if len(argv) >= 3 and argv[2] == "inspect":
            return None
        name = argv[2] if len(argv) >= 3 else "image"
        return f"refusing executable {name}"
    if sub not in _DOCKER_SUBCOMMANDS:
        return f"refusing executable {sub}"
    if sub == "run":
        return _refuse_docker_run(argv, allow_roce=allow_roce)
    return None


def _mount_roots() -> list[str]:
    return _clean_roots(_CFG.get("mount_roots") or [])


def _under_roots(path: str, roots: list[str]) -> bool:
    text = os.path.normpath(str(path or ""))
    if not text.startswith("/"):
        return False
    for root in roots:
        if text == root or text.startswith(root + os.sep):
            return True
    return False


def _shell_roots(roots: list | None) -> list[str] | None:
    """Configured roots for a shell script. ``None`` means they were never loaded.

    A list passed by the pinned check is that node's mount roots. Otherwise
    the list is the one ``configure`` loaded from the site file. The baked-in
    placeholder paths are not used.
    """
    if roots is not None:
        return _clean_roots(roots)
    if not _MOUNTS_CONFIGURED:
        return None
    return _mount_roots()


def _split_docker_flag(token: str) -> tuple[str, str | None]:
    """Return ``(flag, value)``. ``value`` is None when this token has no ``=``."""
    if token.startswith("--"):
        name, eq, value = token.partition("=")
        return name, (value if eq else None)
    return token, None


def _docker_run_options(argv: list) -> list[tuple[str, str | None]]:
    """Options before the image. Tokens after the image are the container command."""
    options: list[tuple[str, str | None]] = []
    index = 2
    while index < len(argv):
        token = argv[index]
        if token == "--" or not token.startswith("-"):
            break
        flag, inline = _split_docker_flag(token)
        if inline is not None:
            options.append((flag, inline))
            index += 1
            continue
        short = flag[1:] if flag.startswith("-") and not flag.startswith("--") else ""
        if short and len(short) > 1 and ("-" + short[0]) in _DOCKER_VALUE_FLAGS:
            options.append(("-" + short[0], short[1:]))
            index += 1
            continue
        if flag in _DOCKER_VALUE_FLAGS:
            value = argv[index + 1] if index + 1 < len(argv) else ""
            options.append((flag, value))
            index += 2
            continue
        if short and len(short) > 1:
            for char in short:
                options.append(("-" + char, None))
            index += 1
            continue
        options.append((flag, None))
        index += 1
    return options


def _volume_source(value: str) -> str:
    text = str(value or "")
    if not text.startswith("/"):
        return text
    return text.split(":", 1)[0]


def _mount_source(value: str) -> str | None:
    bits: dict[str, str] = {}
    for item in str(value or "").split(","):
        if "=" not in item:
            continue
        key, raw = item.split("=", 1)
        bits[key.strip()] = raw.strip()
    source = bits.get("source") or bits.get("src") or ""
    kind = bits.get("type") or ""
    if kind == "bind" or source.startswith("/"):
        return source
    return None


def _bind_source_refused(source: str) -> str | None:
    raw = str(source or "").strip()
    if not raw.startswith("/"):
        return "refusing mount outside model roots"
    path = os.path.normpath(raw)
    if path in ("/", "/etc", "/root") or path.startswith("/etc/") or path.startswith("/root/"):
        return f"refusing mount {path}"
    if path == "/var/run/docker.sock" or path.endswith("/docker.sock"):
        return "refusing mount /var/run/docker.sock"
    for root in _mount_roots():
        if path == root or path.startswith(root + os.sep):
            return None
    return "refusing mount outside model roots"


def _device_source(value: str) -> str:
    return str(value or "").split(":", 1)[0]


_SHELLS = frozenset({"bash", "sh", "dash", "ash"})
_SERVE_ENTRYPOINTS = frozenset({
    "vllm", "python", "python3", "sglang",
    "trtllm-serve", "trtllm-llmapi-launch", "llama-server",
})
_APPROVED_ULIMITS = frozenset({"memlock", "stack"})
_ROCE_CAPS = frozenset({"IPC_LOCK", "CAP_IPC_LOCK"})


def shell_basename(token: str) -> str:
    """Basename when ``token`` is a shell, otherwise empty."""
    base = os.path.basename(str(token or "").strip())
    return base if base in _SHELLS else ""


def docker_option_refusal(
    flag: str,
    value: str | None,
    *,
    allow_roce: bool,
    argv: list | None = None,
    roots: list | None = None,
) -> str | None:
    """One policy for the launch check and the pinned check.

    A two-node RoCE placement (``allow_roce``) may add exactly ``--cap-add
    IPC_LOCK``, ``--security-opt label=disable``, and ``--device /dev/infiniband``.
    Approved ulimits are ``memlock`` and ``stack``; every other ulimit is refused.
    GPU device nodes stay allowed. A shell entrypoint is allowed only for one
    mounted script under ``roots`` (the node's mount roots). When ``roots`` is
    omitted, the list ``configure`` loaded is used, and an unconfigured
    process refuses. A serve binary may be the entrypoint so an image default
    of bash is not applied on top of it. Two-node TensorRT-LLM keeps its
    entrypoint exception.
    """
    text = str(value or "").strip()
    if flag == "--cap-add":
        if allow_roce and text.upper() in _ROCE_CAPS:
            return None
        return f"refusing --cap-add {text}".rstrip()
    if flag == "--security-opt":
        if allow_roce and text == "label=disable":
            return None
        return f"refusing --security-opt {text}".rstrip()
    if flag == "--device":
        source = os.path.normpath(_device_source(text) or "")
        if any(source == prefix or source.startswith(prefix) for prefix in _GPU_DEVICE_PREFIXES):
            return None
        fabric = source == _FABRIC_DEVICE or source.startswith(_FABRIC_DEVICE + os.sep)
        if fabric and allow_roce:
            return None
        return "refusing --device"
    if flag == "--ulimit":
        name = text.split("=", 1)[0].strip()
        if name in _APPROVED_ULIMITS:
            return None
        return f"refusing --ulimit {text}".rstrip() if text else "refusing --ulimit"
    if flag == "--entrypoint":
        base = shell_basename(text)
        if base:
            reason = _shell_script_allowed(argv or [], text, roots)
            return reason if reason else None
        if argv and _two_node_trtllm(argv):
            return None
        program = os.path.basename(text)
        if program in _SERVE_ENTRYPOINTS:
            return None
        return "refusing --entrypoint"
    return None


def _two_node_trtllm(argv: list) -> bool:
    """Head runs ``mpirun`` + ``trtllm``. The worker container carries the OpenMPI env."""
    has_trt = any("trtllm" in part or "tensorrt-llm" in part for part in argv)
    has_pair = any(part == "mpirun" or part.endswith("/mpirun") for part in argv) or any(
        "OMPI_ALLOW_RUN_AS_ROOT" in part for part in argv
    )
    return has_trt and has_pair


def _refuse_docker_run(argv: list, *, allow_roce: bool = False) -> str | None:
    """Refuse host escapes. Generated recipes only bind the cache and use ``--gpus``."""
    for flag, value in _docker_run_options(argv):
        if flag == "--privileged":
            return "refusing --privileged"
        if flag == "--pid" and str(value or "") == "host":
            return "refusing --pid=host"
        if flag in ("--cap-add", "--security-opt", "--device", "--ulimit", "--entrypoint"):
            refused = docker_option_refusal(flag, value, allow_roce=allow_roce, argv=argv)
            if refused:
                return refused
        if flag in ("--network", "--net") and str(value or "") != "host":
            return "refusing --network"
        if flag in ("-v", "--volume"):
            refused = _bind_source_refused(_volume_source(str(value or "")))
            if refused:
                return refused
        if flag == "--mount":
            source = _mount_source(str(value or ""))
            if source is not None:
                refused = _bind_source_refused(source)
                if refused:
                    return refused
    return None


def _refuse_recipe_argv(argv: list, *, allow_roce: bool = False) -> str | None:
    """Backstop for the detached runner. Argv is exec'd, never joined for a shell."""
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        return "argv must be a list of strings"
    joined = " ".join(argv)
    if "vllm-coresident-" in joined or any(part.rsplit("/", 1)[-1] == "tmux" for part in argv):
        return "refusing forbidden command"
    if any(part.rsplit("/", 1)[-1] in ("video-mode.sh", "dgx-announce", "dgx-announce.service") for part in argv):
        return "refusing forbidden command"
    if any("dgx-announce" in part for part in argv):
        return "refusing dgx-announce"
    for index, part in enumerate(argv):
        if part in ("--port", "-p") and index + 1 < len(argv) and _off_limits(argv[index + 1]):
            return f"port {argv[index + 1]} is off limits"
        if part.startswith("--port=") and _off_limits(part.split("=", 1)[1]):
            return f"port {part.split('=', 1)[1]} is off limits"
        if part in ("--host",) and index + 1 < len(argv) and argv[index + 1] in ("0.0.0.0", "::", "*"):
            return "refusing bind 0.0.0.0"
    program = argv[0]
    if not _argv0_allowed(program):
        return f"refusing executable {_executable_name(program)}"
    if program == "docker":
        return _docker_refusal(argv, allow_roce=allow_roce)
    return None


def _read_run(run_id: str) -> dict:
    try:
        data = json.loads(_run_file(run_id).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_run(state: dict) -> None:
    folder = _runs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    path = _run_file(str(state["run_id"]))
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    return Path(f"/proc/{pid}").exists()


def _log_tail(path: Path, lines: int = 80) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.splitlines()[-lines:])[-4000:]


def _parse_inspect_state(text: str) -> tuple[str, int | None]:
    parts = str(text or "").split()
    if len(parts) < 2:
        return "", None
    try:
        code = int(parts[-1])
    except ValueError:
        return parts[0], None
    return parts[0], code


def _parse_mem_gib(text: str) -> float | None:
    token = ""
    for part in str(text or "").replace("/", " ").split():
        token = part
        break
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([KMGT]i?B)?", token, re.IGNORECASE)
    if not match:
        return None
    try:
        number = float(match.group(1))
    except ValueError:
        return None
    unit = (match.group(2) or "B").lower()
    scale = 1.0
    if unit.startswith("k"):
        scale = 1.0 / 1024.0 / 1024.0
    elif unit.startswith("m"):
        scale = 1.0 / 1024.0
    elif unit.startswith("g"):
        scale = 1.0
    elif unit.startswith("t"):
        scale = 1024.0
    else:
        scale = 1.0 / 1024.0 / 1024.0 / 1024.0
    return number * scale


def _inspect_container(name: str) -> dict:
    """Status and exit code of one container. Empty when docker cannot be asked."""
    if not name or any(ch in name for ch in " \n\t;|&$`"):
        return {}
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Status}} {{.State.ExitCode}}", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        err = (proc.stderr or "").lower()
        if "no such" in err:
            return {"status": "missing"}
        return {}
    status, code = _parse_inspect_state(proc.stdout.strip())
    if not status:
        return {}
    return {"status": status, "exit_code": code}


def _cgroup_mem_gib(pid: int) -> float | None:
    """``memory.peak`` or ``memory.current`` for a container process, in GiB."""
    if not isinstance(pid, int) or pid <= 1:
        return None
    try:
        text = Path(f"/proc/{pid}/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) < 3:
            continue
        rel = parts[2].strip().lstrip("/")
        if not rel or ".." in rel.split("/"):
            continue
        base = Path("/sys/fs/cgroup") / rel
        for name in ("memory.peak", "memory.current"):
            try:
                raw = (base / name).read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if raw.isdigit():
                return int(raw) / (1024 ** 3)
    return None


def _container_pid(name: str) -> int | None:
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        pid = int((proc.stdout or "").strip())
    except ValueError:
        return None
    return pid if pid > 1 else None


def _container_mem_gib(name: str) -> float | None:
    if not name or any(ch in name for ch in " \n\t;|&$`"):
        return None
    try:
        proc = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        proc = None
    if proc is not None and proc.returncode == 0:
        parsed = _parse_mem_gib(proc.stdout.strip())
        if parsed is not None:
            return parsed
    return _cgroup_mem_gib(_container_pid(name) or 0)


def _docker_logs(name: str) -> str:
    if not name or any(ch in name for ch in " \n\t;|&$`"):
        return ""
    try:
        proc = subprocess.run(
            ["docker", "logs", "--tail", "80", name],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return ((proc.stdout or "") + (proc.stderr or ""))[-4000:]


def _service_allowed(argv: list) -> bool:
    want = [str(part) for part in argv]
    for item in _CFG.get("service_commands") or []:
        if isinstance(item, list) and [str(part) for part in item] == want:
            return True
    return False


def _service_refusal(argv: list) -> str | None:
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        return "argv must be a list of strings"
    if not _service_allowed(argv):
        return "not in the service allowlist"
    if any("dgx-announce" in part for part in argv):
        return "refusing dgx-announce"
    if any(part.rsplit("/", 1)[-1] in ("video-mode.sh", "dgx-announce", "dgx-announce.service") for part in argv):
        return "refusing forbidden command"
    for index, part in enumerate(argv):
        if part in ("--port", "-p") and index + 1 < len(argv) and _off_limits(argv[index + 1]):
            return f"port {argv[index + 1]} is off limits"
        if _off_limits(part) or (":" in part and _off_limits(part.rsplit(":", 1)[1])):
            return f"port {part.rsplit(':', 1)[-1]} is off limits"
    return None


def _service(req: dict) -> dict:
    """Run one allowlisted argv. The match is exact. Nothing else is executed."""
    argv = req.get("argv") if isinstance(req.get("argv"), list) else []
    refused = _service_refusal(argv)
    if refused:
        return {"ok": False, "error": refused}
    try:
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "pid": proc.pid}


def _replaced_run_is_stopped(replace_run_id: str) -> bool:
    """True when this node already recorded that run id as stopped.

    A Restart may name the run it is replacing. The record has to be stopped
    before a still-running container of the new name can be removed.
    """
    text = str(replace_run_id or "").strip()
    if not text:
        return False
    try:
        safe = _safe_run_id(text)
    except ValueError:
        return False
    state = _read_run(safe)
    return str(state.get("phase") or "") == "stopped"


def _remove_container(name: str, env, *, force: bool) -> dict:
    argv = ["docker", "rm", "-f", name] if force else ["docker", "rm", name]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "could not remove the old container")[-400:]
        return {"ok": False, "error": detail}
    return {}


def _reuse_or_run_argv(argv: list, run_id: str, log_path, env, replace_run_id: str = "") -> dict:
    """Make room for a new ``docker run`` without killing a live container.

    An exited or created container of the same name is removed so the new argv
    can start. A running container is removed only when ``replace_run_id`` is
    the run a user Restart is replacing and this node already has that run
    stopped. Anything else leaves the live container alone.
    """
    del run_id, log_path
    if not (len(argv) > 3 and argv[0] == "docker" and argv[1] == "run" and "-d" in argv[:8]):
        return {}
    try:
        name = argv[argv.index("--name") + 1]
    except (ValueError, IndexError):
        return {}
    state = _inspect_container(name)
    status = str(state.get("status") or "")
    if status in ("", "missing", "none"):
        return {}
    if status in ("exited", "created"):
        return _remove_container(name, env, force=False)
    if status == "running" and _replaced_run_is_stopped(replace_run_id):
        return _remove_container(name, env, force=True)
    if status == "running":
        return {"ok": False, "error": f"container {name} is already running"}
    return {}


def _recipe_run(req: dict) -> dict:
    try:
        run_id = _safe_run_id(str(req.get("run_id") or ""))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    argv = req.get("argv")
    allow_roce = req.get("allow_roce") is True
    refused = _refuse_recipe_argv(argv if isinstance(argv, list) else [], allow_roce=allow_roce)
    if refused:
        return {"ok": False, "error": refused}
    env = os.environ.copy()
    extra = req.get("env") if isinstance(req.get("env"), dict) else {}
    for key, value in extra.items():
        if isinstance(key, str) and isinstance(value, str) and key.replace("_", "").isalnum() and not key[:1].isdigit():
            env[key] = value
    folder = _runs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / f"{run_id}.log"
    container = str(req.get("container") or "")
    detached = argv[0] == "docker" and "-d" in argv[:8]
    if detached:
        reuse = _reuse_or_run_argv(
            argv, run_id, log_path, env, replace_run_id=str(req.get("replace_run_id") or ""),
        )
        if reuse:
            if reuse.get("ok"):
                state = {
                    "run_id": run_id,
                    "pid": None,
                    "code": reuse.get("code", 0),
                    "phase": reuse.get("phase") or "running",
                    "argv": list(reuse.get("argv") or argv),
                    "container": str(reuse.get("container") or container),
                    "log": str(log_path),
                }
                _write_run(state)
            return reuse
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}
        log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
        phase = "running" if proc.returncode == 0 else "exited"
        state = {
            "run_id": run_id,
            "pid": None,
            "code": proc.returncode,
            "phase": phase,
            "argv": argv,
            "container": container,
            "log": str(log_path),
        }
        _write_run(state)
        if proc.returncode != 0:
            return {"ok": False, "error": (proc.stderr or proc.stdout or "docker run failed")[-400:], "phase": phase, "run_id": run_id}
        return {"ok": True, "run_id": run_id, "phase": phase, "pid": None}
    try:
        log_handle = open(log_path, "ab")
    except OSError as exc:
        return {"ok": False, "error": str(exc)}
    try:
        proc = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        log_handle.close()
        return {"ok": False, "error": str(exc)}
    state = {
        "run_id": run_id,
        "pid": proc.pid,
        "code": None,
        "phase": "running",
        "argv": argv,
        "container": container,
        "log": str(log_path),
    }
    _write_run(state)
    return {"ok": True, "run_id": run_id, "phase": "running", "pid": proc.pid}


def _recipe_stop(req: dict) -> dict:
    run_id = str(req.get("run_id") or "")
    argv = req.get("argv") if isinstance(req.get("argv"), list) else []
    if argv:
        refused = _refuse_recipe_argv(argv)
        if refused:
            return {"ok": False, "error": refused}
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"ok": False, "error": str(exc)}
        if proc.returncode != 0:
            return {"ok": False, "error": (proc.stderr or proc.stdout or "stop failed")[-400:], "code": proc.returncode}
    state = {}
    if run_id:
        try:
            state = _read_run(_safe_run_id(run_id))
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
    pid = state.get("pid")
    if isinstance(pid, int) and pid > 1 and _pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
    if state:
        state["phase"] = "stopped"
        _write_run(state)
    return {"ok": True, "phase": "stopped", "run_id": run_id}


def _evict_sha256() -> dict:
    """Hash of the installed page-cache helper. Empty when it cannot be read.

    The helper is imported in a child interpreter so this shipped file stays
    on the standard library.
    """
    code = (
        "import hashlib, pathlib\n"
        "import sparksi.evict as helper\n"
        "print(hashlib.sha256(pathlib.Path(helper.__file__).read_bytes()).hexdigest())\n"
    )
    try:
        proc = subprocess.run(
            ["python3", "-c", code],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "error": str(exc)}
    digest = (proc.stdout or "").strip()
    if proc.returncode != 0 or len(digest) != 64:
        detail = (proc.stderr or "could not read helper").strip()[-200:]
        return {"ok": False, "error": detail}
    return {"ok": True, "sha256": digest}


# --- recipe folder copies -------------------------------------------------
# A copied recipe folder lives at <recipe_root>/<model>/<name>/<version>/ and is
# read-only once written. These ops never rewrite or remove an existing copy.

FILES_MAX_BYTES = 64 * 1024 * 1024  # equal to the controller's MAX_TOTAL_BYTES


def _inside_root(path: str, roots: list) -> str | None:
    """Normalised ``path`` when it is strictly inside one of ``roots``; None otherwise."""
    raw = str(path or "")
    if not raw.startswith("/") or ".." in raw.split("/"):
        return None
    text = os.path.normpath(raw)
    for root in roots:
        base = os.path.normpath(str(root))
        if base in ("", ".", "/"):
            continue
        if text.startswith(base + os.sep):
            return text
    return None


def _clean_rel(rel: str) -> str | None:
    text = str(rel or "")
    if not text or text.startswith("/") or "\\" in text:
        return None
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    if parts[0] not in ("files", "roles") or len(parts) < 2:
        return None
    return "/".join(parts)


def _file_digest(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest(req: dict) -> dict | None:
    """rel -> (sha256, executable) from the request; None when any entry is malformed."""
    found: dict = {}
    for item in req.get("manifest") or []:
        if not isinstance(item, dict):
            return None
        rel = _clean_rel(item.get("path"))
        sha = str(item.get("sha256") or "")
        if rel is None or len(sha) != 64 or any(ch not in "0123456789abcdef" for ch in sha) or rel in found:
            return None
        found[rel] = (sha, bool(item.get("executable")))
    return found


def _compare(target: str, wanted: dict) -> dict:
    """Hash every file in a copy and compare it with the manifest."""
    if not os.path.isdir(target) or os.path.islink(target):
        return {"ok": True, "state": "missing", "missing": sorted(wanted), "differs": [], "extra": []}
    seen: dict = {}
    for current, dirs, files in os.walk(target, followlinks=False):
        for name in files + [d for d in dirs if os.path.islink(os.path.join(current, d))]:
            full = os.path.join(current, name)
            rel = os.path.relpath(full, target).replace(os.sep, "/")
            if os.path.islink(full) or not os.path.isfile(full):
                seen[rel] = None
                continue
            seen[rel] = (_file_digest(full), bool(os.stat(full).st_mode & 0o111))
    missing = sorted(rel for rel in wanted if rel not in seen)
    differs = sorted(rel for rel in wanted if rel in seen and seen[rel] != wanted[rel])
    extra = sorted(rel for rel in seen if rel not in wanted)
    state = "match" if not (missing or differs or extra) else "differs"
    return {"ok": True, "state": state, "missing": missing, "differs": differs, "extra": extra}


def _files_check(req: dict) -> dict:
    """Read-only: does the copy at ``dir`` hold exactly the manifest?"""
    target = _inside_root(req.get("dir"), _CFG.get("recipe_roots") or [])
    if target is None:
        return {"ok": False, "error": "refusing a recipe folder outside the recipe roots"}
    wanted = _manifest(req)
    if wanted is None:
        return {"ok": False, "error": "bad recipe folder manifest"}
    out = _compare(target, wanted)
    out["dir"] = target
    return out


def _files_stage(req: dict) -> dict:
    """Write a recipe folder copy once, check it by hash, then make it read-only.

    An existing copy is only compared, never rewritten. The copy is built in a
    temporary sibling and renamed into place after every hash matches.
    """
    target = _inside_root(req.get("dir"), _CFG.get("recipe_roots") or [])
    if target is None:
        return {"ok": False, "error": "refusing a recipe folder outside the recipe roots"}
    wanted = _manifest(req)
    if wanted is None:
        return {"ok": False, "error": "bad recipe folder manifest"}
    if os.path.lexists(target):
        out = _compare(target, wanted)
        out["dir"] = target
        out["copied"] = False
        return out
    blobs: dict = {}
    total = 0
    for item in req.get("files") or []:
        rel = _clean_rel(item.get("path")) if isinstance(item, dict) else None
        if rel is None or rel not in wanted or rel in blobs:
            return {"ok": False, "error": "bad recipe folder file list"}
        try:
            data = base64.b64decode(str(item.get("data") or ""), validate=True)
        except (ValueError, TypeError):
            return {"ok": False, "error": f"bad data for {rel}"}
        total += len(data)
        if total > FILES_MAX_BYTES:
            return {"ok": False, "error": "recipe folder files are too large"}
        if hashlib.sha256(data).hexdigest() != wanted[rel][0]:
            return {"ok": False, "error": f"{rel} did not arrive intact"}
        blobs[rel] = data
    if set(blobs) != set(wanted):
        return {"ok": False, "error": "recipe folder file list does not match its manifest"}
    parent = os.path.dirname(target)
    # A linked folder on the way must not carry the copy outside its recipe root.
    root = next(
        os.path.normpath(str(r)) for r in _CFG.get("recipe_roots") or []
        if target.startswith(os.path.normpath(str(r)) + os.sep)
    )
    os.makedirs(root, mode=0o755, exist_ok=True)
    real_root = os.path.realpath(root)

    def within(path: str) -> bool:
        real = os.path.realpath(path)
        return real == real_root or real.startswith(real_root + os.sep)

    existing = parent
    while not os.path.lexists(existing):
        existing = os.path.dirname(existing)
    if not within(existing):
        return {"ok": False, "error": "refusing a recipe folder outside the recipe roots"}
    os.makedirs(parent, mode=0o755, exist_ok=True)
    if not within(parent):
        return {"ok": False, "error": "refusing a recipe folder outside the recipe roots"}
    tmp = tempfile.mkdtemp(prefix=".incoming-", dir=parent)
    try:
        for rel in sorted(blobs):
            full = os.path.join(tmp, *rel.split("/"))
            os.makedirs(os.path.dirname(full), mode=0o755, exist_ok=True)
            with open(full, "xb") as handle:
                handle.write(blobs[rel])
            os.chmod(full, 0o555 if wanted[rel][1] else 0o444)
        check = _compare(tmp, wanted)
        if check["state"] != "match":
            raise OSError("the written copy does not match its manifest")
        for current, dirs, _files in os.walk(tmp, topdown=False):
            for name in dirs:
                os.chmod(os.path.join(current, name), 0o555)
        os.chmod(tmp, 0o555)
        try:
            os.rename(tmp, target)
        except OSError:
            if not os.path.isdir(target):
                raise
            # Another start copied the same version first: keep that one.
            _drop_tree(tmp)
            out = _compare(target, wanted)
            out["dir"] = target
            out["copied"] = False
            return out
    except BaseException:
        _drop_tree(tmp)
        raise
    out = _compare(target, wanted)
    out["dir"] = target
    out["copied"] = True
    return out


def _drop_tree(path: str) -> None:
    """Remove a temporary copy this op made. Only ever called on its own mkdtemp folder."""
    for current, dirs, _files in os.walk(path):
        os.chmod(current, 0o755)
        for name in dirs:
            try:
                os.chmod(os.path.join(current, name), 0o755)
            except OSError:
                pass
    shutil.rmtree(path, ignore_errors=True)


def _single_recipe_input(path: str, limit: int) -> dict:
    """Only a regular file with no linked path component; never walk or follow a link.

    Open directories relative to held descriptors so swapping a parent for a
    link between stat and open cannot redirect the file read.
    """
    parts = path.strip("/").split("/")
    parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in parts[:-1]:
            info = os.stat(part, dir_fd=parent, follow_symlinks=False)
            if stat.S_ISLNK(info.st_mode):
                return {"skip": "link"}
            if not stat.S_ISDIR(info.st_mode):
                return {"skip": "missing"}
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = child
        leaf = parts[-1]
        info = os.stat(leaf, dir_fd=parent, follow_symlinks=False)
        if stat.S_ISLNK(info.st_mode):
            return {"skip": "link"}
        if stat.S_ISDIR(info.st_mode):
            return {"skip": "dir"}
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return {"skip": "large" if stat.S_ISREG(info.st_mode) else "special"}
        fd = os.open(leaf, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(fd, "rb") as handle:
            now = os.fstat(handle.fileno())
            if not stat.S_ISREG(now.st_mode) or (now.st_dev, now.st_ino) != (info.st_dev, info.st_ino):
                return {"error": f"{path} changed while it was read"}
            data = handle.read(limit + 1)
        if len(data) > limit:
            return {"skip": "large"}
        return {"size": len(data), "file": {
            "data": base64.b64encode(data).decode("ascii"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "executable": bool(now.st_mode & 0o111),
        }}
    except FileNotFoundError:
        return {"skip": "missing"}
    finally:
        os.close(parent)


def _files_read(req: dict) -> dict:
    """Read-only: files under the mount roots, base64, for copying into a recipe folder.

    A folder is read whole (regular files only; links are reported, not followed).
    With ``files_only`` a folder is not read but listed under ``skipped`` as
    "dir", and a file over ``file_max`` bytes is listed as "large": the bundle
    leaves those mounts where they are (weights and caches).
    """
    roots = _CFG.get("mount_roots") or []
    out: dict = {}
    links: list = []
    skipped: dict = {}
    files_only = bool(req.get("files_only"))
    file_max = int(req.get("file_max") or FILES_MAX_BYTES)
    total = 0
    for raw in req.get("paths") or []:
        path = _inside_root(raw, roots)
        if path is None:
            return {"ok": False, "error": f"refusing to read {raw}: outside the mount roots"}
        if any(path == p or path.startswith(p + os.sep) for p in blocked_prefixes()):
            return {"ok": False, "error": f"refusing to read {raw}"}
        if os.path.islink(path):
            links.append(path)
            continue
        if files_only:
            read = _single_recipe_input(path, file_max)
            if read.get("skip"):
                skipped[path] = read["skip"]
            elif read.get("error"):
                return {"ok": False, "error": read["error"]}
            else:
                total += read["size"]
                if total > FILES_MAX_BYTES:
                    return {"ok": False, "error": "files are too large to copy into a recipe folder"}
                out[path] = read["file"]
            continue
        if os.path.isfile(path):
            candidates = [path]
        elif os.path.isdir(path):
            candidates = []
            for current, dirs, files in os.walk(path, followlinks=False):
                dirs.sort()
                for name in sorted(files) + [d for d in dirs if os.path.islink(os.path.join(current, d))]:
                    candidates.append(os.path.join(current, name))
        else:
            out[path] = None
            continue
        for full in candidates:
            if os.path.islink(full) or not os.path.isfile(full):
                links.append(full)
                continue
            with open(full, "rb") as handle:
                data = handle.read(FILES_MAX_BYTES + 1)
            total += len(data)
            if total > FILES_MAX_BYTES:
                return {"ok": False, "error": "files are too large to copy into a recipe folder"}
            out[full] = {
                "data": base64.b64encode(data).decode("ascii"),
                "sha256": hashlib.sha256(data).hexdigest(),
                "executable": bool(os.stat(full).st_mode & 0o111),
            }
    return {"ok": True, "files": out, "links": links, "skipped": skipped}


def _file_sha256(req: dict) -> dict:
    """Read-only: SHA-256 of files under the mount roots (a pinned binary, for example).

    ``path`` hashes one file. With ``libs`` true it also lists the shared
    libraries that file loads from its own build tree (see ``_tree_libs``).
    ``paths`` hashes several files in one call and answers ``files``.
    """
    roots = _CFG.get("mount_roots") or []
    if req.get("paths") is not None:
        files: dict = {}
        paths = list(req.get("paths") or [])
        if len(paths) > LIBS_MAX + 1:  # a binary and its recorded libraries; never cut short
            return {"ok": False, "error": f"refusing more than {LIBS_MAX + 1} files in one check"}
        for raw in paths:
            path = _inside_root(raw, roots)
            if path is None:
                return {"ok": False, "error": f"refusing {raw}: outside the mount roots"}
            found = os.path.isfile(path)
            files[str(raw)] = {"exists": found, "sha256": _file_digest(path) if found else ""}
        return {"ok": True, "files": files}
    path = _inside_root(req.get("path"), roots)
    if path is None:
        return {"ok": False, "error": "refusing a file outside the mount roots"}
    if not os.path.isfile(path):
        return {"ok": True, "exists": False, "sha256": ""}
    out = {"ok": True, "exists": True, "sha256": _file_digest(path)}
    if req.get("libs"):
        out["libs"] = _tree_libs(path, str(req.get("ld_library_path") or ""), roots)
    return out


# --- shared libraries a native binary loads from its own build tree ---------
# Read from the ELF dynamic section (DT_NEEDED, DT_RPATH, DT_RUNPATH); nothing
# is run. Only libraries under the binary's tree (the folder above the one it
# sits in) and inside the mount roots count; system libraries are left out.

LIBS_MAX = 256
_DT_NEEDED, _DT_STRTAB, _DT_STRSZ, _DT_RPATH, _DT_RUNPATH = 1, 5, 10, 15, 29


def _elf_dynamic(path: str) -> tuple[list, list, list] | None:
    """(needed, rpath, runpath) of a 64-bit little-endian ELF file; None when it is not one."""
    try:
        with open(path, "rb") as handle:
            head = handle.read(64)
            if len(head) < 64 or head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
                return None
            (phoff,) = struct.unpack_from("<Q", head, 0x20)
            phentsize, phnum = struct.unpack_from("<HH", head, 0x36)
            if phentsize < 56 or phnum > 512:
                return None
            handle.seek(phoff)
            table = handle.read(phentsize * phnum)
            loads = []
            dynamic = None
            for index in range(phnum):
                p_type, _flags, p_offset, p_vaddr, _paddr, p_filesz, _memsz, _align = struct.unpack_from(
                    "<IIQQQQQQ", table, index * phentsize
                )
                if p_type == 1:
                    loads.append((p_vaddr, p_offset, p_filesz))
                elif p_type == 2:
                    dynamic = (p_offset, p_filesz)
            if dynamic is None:
                return [], [], []
            if dynamic[1] > 1 << 20:
                return None
            handle.seek(dynamic[0])
            raw = handle.read(dynamic[1])
            entries = []
            for offset in range(0, len(raw) - 15, 16):
                tag, value = struct.unpack_from("<qQ", raw, offset)
                if tag == 0:
                    break
                entries.append((tag, value))
            tags = dict(entries)
            strtab, strsz = tags.get(_DT_STRTAB), tags.get(_DT_STRSZ)
            if strtab is None or strsz is None or strsz > 1 << 22:
                return None
            start = next((off + strtab - vaddr for vaddr, off, size in loads if vaddr <= strtab < vaddr + size), None)
            if start is None:
                return None
            handle.seek(start)
            strings = handle.read(strsz)
    except (OSError, struct.error, ValueError, OverflowError):  # a truncated or malformed file
        return None

    def text(index: int) -> str:
        end = strings.find(b"\0", index)
        return strings[index:end if end >= 0 else len(strings)].decode("utf-8", "replace")

    needed: list = []
    rpath: list = []
    runpath: list = []
    for tag, value in entries:
        if value >= len(strings):
            continue
        if tag == _DT_NEEDED:
            needed.append(text(value))
        elif tag == _DT_RPATH:
            rpath.extend(text(value).split(":"))
        elif tag == _DT_RUNPATH:
            runpath.extend(text(value).split(":"))
    return needed, rpath, runpath


def _tree_libs(binary: str, ld_library_path: str, roots: list) -> dict:
    """Path -> SHA-256 of each shared library ``binary`` loads from its own build tree."""
    tree = os.path.dirname(os.path.dirname(os.path.realpath(binary)))
    if tree in ("", "/"):
        return {}
    env_dirs = [part for part in ld_library_path.split(":") if part]
    found: dict = {}
    # (file, RPATH folders of the files that loaded it): an object without RUNPATH also
    # searches its loaders' RPATH, as the dynamic loader does.
    pending: list = [(binary, [])]
    seen = {os.path.realpath(binary)}
    while pending and len(found) < LIBS_MAX:
        current, inherited = pending.pop(0)
        info = _elf_dynamic(current)
        if not info:
            continue
        needed, rpath, runpath = info
        origin = os.path.dirname(os.path.realpath(current))

        def expand(folders: list, origin: str = origin) -> list:
            return [d.replace("${ORIGIN}", origin).replace("$ORIGIN", origin) for d in folders if d]

        own = [] if runpath else expand(rpath)
        chain = own + inherited
        dirs = ([] if runpath else chain) + env_dirs + expand(runpath)
        for name in needed:
            if len(found) >= LIBS_MAX:
                break
            if not name or "/" in name:
                continue
            for folder in dirs:
                candidate = os.path.normpath(os.path.join(folder, name))
                if not os.path.isfile(candidate):
                    continue
                real = os.path.realpath(candidate)
                if real in seen:
                    break
                if not real.startswith(tree + os.sep) or _inside_root(candidate, roots) is None:
                    break  # a system library or one outside the tree: not recorded
                seen.add(real)
                found[candidate] = _file_digest(candidate)
                pending.append((candidate, chain))
                break
    return dict(sorted(found.items()))


def _image_command(argv: list) -> list[str]:
    """Tokens after the image. Those are the container command."""
    index = 2
    while index < len(argv):
        token = argv[index]
        if token == "--":
            return [str(part) for part in argv[index + 1:]]
        if not str(token).startswith("-"):
            return [str(part) for part in argv[index + 1:]]
        _flag, inline = _split_docker_flag(token)
        if inline is not None:
            index += 1
            continue
        flag = _flag
        if flag in _DOCKER_VALUE_FLAGS:
            index += 2
            continue
        index += 1
    return []


def _mounted_host(container: str, argv: list) -> str:
    """Host path of a container path through the longest matching -v mount."""
    target = os.path.normpath(str(container or ""))
    best = ""
    best_len = -1
    for flag, value in _docker_run_options(argv):
        if flag not in ("-v", "--volume") or not value or ":" not in str(value):
            continue
        host, _, rest = str(value).partition(":")
        dest = os.path.normpath(rest.split(":", 1)[0] or "")
        if not host.startswith("/") or not dest.startswith("/"):
            continue
        if target == dest or target.startswith(dest + os.sep):
            if len(dest) > best_len:
                best_len = len(dest)
                rel = target[len(dest):].lstrip("/")
                best = os.path.normpath(os.path.join(host, rel)) if rel else os.path.normpath(host)
    return best


# Same set the pinned check allows after a script. ``-c`` stays a shell command.
_GENERATED_FLAGS = {
    "--port", "-p", "--host", "--bind",
    "--max-model-len", "--context-length", "--max_seq_len", "--ctx-size",
    "--gpu-memory-utilization", "--mem-fraction-static", "--kv_cache_free_gpu_memory_fraction",
    "--kv-cache-memory-bytes", "--kv-cache-memory", "--kv-cache-dtype",
    "--max-num-seqs",
    "--tool-call-parser", "--reasoning-parser",
    "--speculative-config", "--spec-type", "--spec-draft-n-max", "--model-draft",
    "--speculative-algorithm", "--speculative-num-steps", "--speculative-draft-model-path",
}


def _shell_meta(value: str) -> bool:
    return any(ch in str(value) for ch in ";|&$`<>\n\r\\")


def _generated_tail(rest: list[str]) -> bool:
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "-c" or token.startswith("-c") or _shell_meta(token):
            return False
        flag = token.split("=", 1)[0]
        if flag not in _GENERATED_FLAGS:
            return False
        if "=" in token:
            if _shell_meta(token.split("=", 1)[1]):
                return False
            index += 1
            continue
        if index + 1 >= len(rest) or rest[index + 1].startswith("-") or _shell_meta(rest[index + 1]):
            return False
        index += 2
    return True


def _shell_script_allowed(argv: list, entry: str, roots: list | None = None) -> str | None:
    """Allow bash only for a mounted script plus Sparksi's own serve flags."""
    base = shell_basename(entry)
    if not base:
        return None
    command = _image_command(argv)
    script = command[0] if command else ""
    if not script.startswith("/") or not script.endswith((".sh", ".bash")) or not _generated_tail(command[1:]):
        return f"refusing shell entrypoint {base}"
    allowed = _shell_roots(roots)
    if not allowed:
        return f"refusing shell entrypoint {base}: mount roots are not configured"
    host = _mounted_host(script, argv)
    if not host or not _under_roots(host, allowed):
        return f"refusing shell entrypoint {base}: script is not under a mount root"
    return None


def _recipe_status(req: dict) -> dict:
    try:
        run_id = _safe_run_id(str(req.get("run_id") or ""))
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    state = _read_run(run_id)
    container = str(req.get("container") or (state or {}).get("container") or "")
    if not state and not container:
        return {"ok": False, "error": "unknown run", "run_id": run_id}
    if state:
        pid = state.get("pid")
        if isinstance(pid, int) and pid > 1 and not _pid_alive(pid) and state.get("phase") == "running":
            state["phase"] = "exited"
            _write_run(state)
    else:
        state = {}
    tail = _docker_logs(container) if container else _log_tail(Path(str(state.get("log") or "")))
    inspected = _inspect_container(container) if container else {}
    out = {
        "ok": True,
        "run_id": run_id,
        "phase": state.get("phase"),
        "pid": state.get("pid"),
        "code": state.get("code"),
        "log_tail": tail,
    }
    if inspected.get("status"):
        out["container_status"] = inspected["status"]
        out["exit_code"] = inspected.get("exit_code")
    mem = _container_mem_gib(container) if container else None
    if mem is not None:
        out["mem_gib"] = round(mem, 2)
    return out


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] in ("run", "stop", "status"):
        if len(sys.argv) >= 3:
            payload = json.loads(sys.argv[2])
        else:
            payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            payload = {}
        payload["op"] = sys.argv[1]
        sys.stdout.write(json.dumps(main(payload)))
    else:
        request = json.load(sys.stdin)
        sys.stdout.write(json.dumps(main(request)))
