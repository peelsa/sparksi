# SPDX-License-Identifier: Apache-2.0

"""Local Spark sample, sent as one UDP datagram. Stdlib only.

Run on a node:
    SPARKSI_NODE_ID=a SPARKSI_CONTROLLER=100.64.0.1:7501 python3 announce.py
The controller listens. This process does not open a port. No site values
live in this file; everything comes from the environment.

SPARKSI_ADVERTISE is that node's configured client address. A wildcard
socket is published only after a short GET of that address returns the same
served id. An empty variable publishes nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

MAX_BYTES = 1200

# --- COMPAT: node copy of sparksi.compat (stdlib only; tests/test_compat.py keeps it equal) ---
VIDEO_PACKET_KEY = "video"
LEGACY_VIDEO_PACKET_KEY = "comfy"
LEGACY_MONITOR_VIDEO_KEY = "comfy"
VIDEO_PROCESS_SUBSTRINGS = ("comfy",)
VIDEO_STATE_ENV = ("SPARKSI_VIDEO_STATE", "SPARKSI_COMFY_STATE")
VIDEO_SERVICE_ROOT_FALLBACKS = ("~/services/video-service", "~/services/comfyui")
VIDEO_SERVICE_PORT = 8188
# Node copy of the sparksi.protect video-service preset switch. Off unless
# SPARKSI_PROTECT names it (``sparksi install-unit --print --unit announce`` writes
# the controller's active presets there), or a legacy marker is present: the
# legacy node-id env name, SPARKSI_COMFY_STATE set, the legacy install directory
# (last VIDEO_SERVICE_ROOT_FALLBACKS entry), or the legacy unit name.
VIDEO_PRESET = "video-service"
LEGACY_NODE_ENV = ("DGX_ANNOUNCE_HOST",)
LEGACY_UNIT_NAMES = ("dgx-announce.service",)
INTERVAL = 3.0
ADVERTISE_TIMEOUT = 0.4


def _env(name: str, old: str = "", default: str = "") -> str:
    val = os.environ.get(name) or (os.environ.get(old) if old else "") or default
    return str(val).strip()


def dest() -> tuple:
    """Controller address from SPARKSI_CONTROLLER=host:port (port defaults to 7501)."""
    raw = _env("SPARKSI_CONTROLLER")
    if not raw:
        raise SystemExit("SPARKSI_CONTROLLER must be set to <controller-ip>[:port]")
    host, _, port = raw.rpartition(":") if raw.count(":") == 1 else (raw, "", "")
    return (host or raw, int(port or 7501))


def node_ids() -> tuple:
    """Accepted node ids, from SPARKSI_NODE_IDS=a,b (default: a,b)."""
    raw = _env("SPARKSI_NODE_IDS", default="a,b")
    return tuple(x.strip() for x in raw.split(",") if x.strip())


def _video_state_dir() -> Path:
    for name in VIDEO_STATE_ENV:
        val = _env(name)
        if val:
            return Path(val)
    roots = [Path(os.path.expanduser(r)) for r in VIDEO_SERVICE_ROOT_FALLBACKS]
    root = next((r for r in roots if r.exists()), roots[-1])
    return root / "state"


_VIDEO_NOTED = False


def video_on() -> bool:
    """True when this node reports and skips the protected video service."""
    global _VIDEO_NOTED
    names = {part.strip() for part in _env("SPARKSI_PROTECT").split(",")}
    if VIDEO_PRESET in names:
        return True
    reasons = _legacy_reasons()
    if reasons and not _VIDEO_NOTED:
        _VIDEO_NOTED = True
        print(
            f"announce: {VIDEO_PRESET} preset auto-enabled for a legacy install ({', '.join(reasons)}); "
            f"set SPARKSI_PROTECT={VIDEO_PRESET}",
            file=sys.stderr,
        )
    return bool(reasons)


def _own_unit_text() -> str:
    """This process's cgroup path, which names its systemd unit. Empty off Linux."""
    try:
        return Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _legacy_reasons() -> list[str]:
    """Legacy-install markers on this node, in a fixed order."""
    reasons = [name for name in LEGACY_NODE_ENV + VIDEO_STATE_ENV[-1:] if os.environ.get(name, "").strip()]
    if Path(os.path.expanduser(VIDEO_SERVICE_ROOT_FALLBACKS[-1])).exists():
        reasons.append(VIDEO_SERVICE_ROOT_FALLBACKS[-1])
    cgroup = _own_unit_text()
    reasons.extend(unit for unit in LEGACY_UNIT_NAMES if f"/{unit}" in cgroup)
    return reasons


def video_mode_file() -> Path:
    return _video_state_dir() / "mode"

# Short key -> the metric name discover_loaded and the rate math already use.
METRIC_SHORT = {
    "gt": "generation_tokens_total",
    "pt": "prompt_tokens_total",
    "r": "num_requests_running",
    "w": "num_requests_waiting",
    "kv": "kv_cache_usage_perc",
    "ph": "prefix_cache_hits_total",
    "pq": "prefix_cache_queries_total",
    "np": "num_preemptions_total",
    "ok": "request_success_total",
    "dd": "spec_decode_num_draft_tokens_total",
    "da": "spec_decode_num_accepted_tokens_total",
    # llama-server day counts, from its own log (it has no /metrics unless
    # started with --metrics). Spark-local day. "ps" keys the process so a
    # restart is a new row on the controller, not a reset.
    "dp": "llama_today_prompt_tokens",
    "dg": "llama_today_generation_tokens",
    "yp": "llama_yesterday_prompt_tokens",
    "yg": "llama_yesterday_generation_tokens",
    "dy": "llama_today_date",
    "ps": "llama_process_start",
    # llama-server live status (from /slots, /health and the log): decode
    # tok/s measured here between samples, slot context fill (0-1) while a
    # request runs, and whether /health says ok. "r" (running) is shared.
    "gs": "llama_gen_tps",
    "cx": "llama_ctx_used",
    "hl": "llama_health_ok",
}
_SHORT_BY_FULL = {full: short for short, full in METRIC_SHORT.items()}
# TensorFold uses its own names for the same counters the rate math already reads.
_METRIC_ALIASES = {
    "requests_running": "num_requests_running",
    "requests_waiting": "num_requests_waiting",
    "kv_cache_usage_ratio": "kv_cache_usage_perc",
    "mtp_drafted_total": "spec_decode_num_draft_tokens_total",
    "mtp_accepted_total": "spec_decode_num_accepted_tokens_total",
}


def _metrics_from_prom(text: str) -> dict[str, float]:
    """Short metric keys from a Prometheus page.

    TensorFold's finished-request generation counter stays flat while a reply
    is still decoding. ``tensorfold_health:completion_tokens_total`` moves
    with that reply, so it is the rate counter when it is present. The name is
    looked up before the value is parsed: most of a vLLM page is histogram
    buckets nobody reads.
    """
    metrics: dict[str, float] = {}
    live_gen = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        raw = parts[0].split("{", 1)[0]
        if raw == "tensorfold_health:completion_tokens_total":
            val = _num(parts[1])
            if val is not None:
                live_gen = val
            continue
        short = _SHORT_BY_FULL.get(_normalize_metric(raw))
        if not short:
            continue
        val = _num(parts[1])
        if val is None:
            continue
        metrics[short] = round(metrics.get(short, 0.0) + val, 4)
    if live_gen is not None:
        metrics["gt"] = round(live_gen, 4)
    return metrics


def _normalize_metric(base: str) -> str:
    """Drop an engine prefix and map TensorFold names onto the vLLM ones."""
    for prefix in ("vllm:", "tensorfold:"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    return _METRIC_ALIASES.get(base, base)


def _num(text: str):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _mem() -> list[float]:
    info = {}
    try:
        raw = Path("/proc/meminfo").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [0.0, 0.0]
    for line in raw.splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        parts = val.split()
        if parts and parts[0].isdigit():
            info[key] = int(parts[0])
    gib = 1024.0 * 1024.0
    return [round(info.get("MemTotal", 0) / gib, 1), round(info.get("MemAvailable", 0) / gib, 1)]


def _cpu_pct() -> float | None:
    def sample() -> tuple[int, int] | None:
        try:
            line = Path("/proc/stat").read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except (OSError, IndexError):
            return None
        vals = [int(x) for x in line.split()[1:] if x.isdigit()]
        if len(vals) < 4:
            return None
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        return sum(vals), idle

    a = sample()
    time.sleep(0.2)
    b = sample()
    if not a or not b:
        return None
    dt, di = b[0] - a[0], b[1] - a[1]
    if dt <= 0:
        return None
    return round(max(0.0, min(100.0, 100.0 * (dt - di) / dt)), 1)


def _cpu_mhz() -> float | None:
    freqs = []
    root = Path("/sys/devices/system/cpu")
    if not root.is_dir():
        return None
    for node in root.glob("cpu[0-9]*/cpufreq/cpuinfo_avg_freq"):
        try:
            raw = node.read_bytes().decode("ascii", "replace").strip()
            khz = int(raw or "0")
        except (OSError, ValueError, TypeError):
            continue
        if khz > 0:
            freqs.append(khz / 1000.0)
    if not freqs:
        return None
    return round(sum(freqs) / len(freqs), 0)


def _gpu() -> list:
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return [None, None, None, None]
    line = (out.splitlines() or [""])[0]
    parts = [p.strip() for p in line.split(",")]
    while len(parts) < 4:
        parts.append("")
    return [_num(parts[0]), _num(parts[1]), _num(parts[2]), _num(parts[3])]


def _zones() -> list:
    rows = []
    root = Path("/sys/class/thermal")
    if not root.is_dir():
        return rows
    for zone in sorted(root.glob("thermal_zone*")):
        try:
            label = (zone / "type").read_text().strip()[:12]
            temp = int((zone / "temp").read_text().strip()) / 1000.0
        except (OSError, ValueError):
            continue
        rows.append([label, round(temp, 1)])
        if len(rows) >= 8:
            break
    return rows


def _budget() -> dict:
    """Top comm names by RSS. Command lines are not read."""
    gib = 1024.0 ** 3
    info = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            parts = val.split()
            if parts and parts[0].isdigit():
                info[key] = int(parts[0])
    except OSError:
        info = {}

    def kib(key: str) -> int:
        return int(info.get(key, 0))

    cache = kib("Buffers") + kib("Cached") + kib("SReclaimable")
    slab = kib("SUnreclaim")
    small_kernel = kib("KernelStack") + kib("PageTables") + kib("Percpu") + kib("SecPageTables")
    gap = kib("MemTotal") - kib("MemFree") - kib("AnonPages") - kib("Buffers") - kib("Cached") - (kib("SReclaimable") + slab) - small_kernel
    rss: dict[str, int] = {}
    proc = Path("/proc")
    try:
        pids = [p for p in proc.iterdir() if p.name.isdigit()]
    except OSError:
        pids = []
    for pid in pids:
        try:
            comm = (pid / "comm").read_text().strip()
            pages = int((pid / "statm").read_text().split()[1])
        except (OSError, ValueError, IndexError):
            continue
        low = comm.lower()
        if low.startswith("vllm") or (video_on() and any(s in low for s in VIDEO_PROCESS_SUBSTRINGS)):
            continue
        safe = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in comm)[:16]
        rss[safe] = rss.get(safe, 0) + pages * 4096
    top = []
    rest = 0
    for name, nbytes in sorted(rss.items(), key=lambda kv: -kv[1]):
        if nbytes / gib < 0.05 or len(top) >= 5:
            rest += nbytes
            continue
        top.append([name, round(nbytes / gib, 2)])
    return {
        "cache": round(cache / 1024.0 / 1024.0, 1),
        "model": round(max(0.0, gap) / 1024.0 / 1024.0, 1),
        "slab": round(slab / 1024.0 / 1024.0, 2),
        "small": round(rest / gib, 2),
        "top": top,
    }


def _port_from_blob(blob: str) -> tuple[int, int]:
    port = tp = 0
    cmd = False
    prev = ""
    for line in blob.splitlines():
        if line == "---CMD---":
            cmd = True
            prev = line
            continue
        if not cmd:
            if line.startswith("PORT=") and line[5:].isdigit():
                port = int(line[5:])
            elif line.startswith("TP=") and line[3:].isdigit():
                tp = int(line[3:])
            elif line.startswith("TENSOR_PARALLEL_SIZE=") and line.split("=", 1)[1].isdigit():
                tp = int(line.split("=", 1)[1])
        else:
            if line.startswith("--port=") and line[7:].isdigit():
                port = int(line[7:])
            elif prev == "--port" and line.isdigit():
                port = int(line)
            elif line.startswith("--tp=") and line[5:].isdigit():
                tp = int(line[5:])
            elif prev == "--tp" and line.isdigit():
                tp = int(line)
        prev = line
    return port, tp


# Container ID -> its inspect blob (Env and Cmd). A container's ID is new each
# time it is created, so one inspect per container is enough while it runs.
_INSPECT_CACHE: dict[str, str] = {}
# (container ID, name) of every running container from the last docker ps; None when it failed.
_LAST_PS: list[tuple[str, str]] | None = None


def _inspect_blob(cid: str) -> str | None:
    """Env and Cmd of one container, inspected once per container ID."""
    blob = _INSPECT_CACHE.get(cid)
    if blob is not None:
        return blob
    try:
        proc = subprocess.run(
            ["docker", "inspect", "--format",
             "{{range .Config.Env}}{{println .}}{{end}}{{println \"---CMD---\"}}{{range .Config.Cmd}}{{println .}}{{end}}",
             cid],
            capture_output=True, text=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    _INSPECT_CACHE[cid] = proc.stdout
    return proc.stdout


def _engines() -> tuple[list, list]:
    """Running containers only. Port comes from PORT= or --port."""
    global _LAST_PS
    try:
        rows = subprocess.run(
            ["docker", "ps", "--no-trunc", "--filter", "status=running", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=2,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError):
        _LAST_PS = None
        return [], []
    ids: list[str] = []
    name_by_id: dict[str, str] = {}
    for row in rows:
        parts = row.split(None, 1)
        if len(parts) != 2:
            continue
        cid, name = parts
        ids.append(cid)
        name_by_id[cid] = name
    running = set(ids)
    _LAST_PS = [(cid, name_by_id[cid]) for cid in ids]
    for cid in list(_INSPECT_CACHE):
        if cid not in running:
            del _INSPECT_CACHE[cid]
    for cid in ids[:16]:
        _inspect_blob(cid)  # once per container ID; the probe beat reads declared ports from it
    dock, eng = [], []
    for cid in ids[:6]:
        name = name_by_id[cid]
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)[:40]
        dock.append([safe, "Up"])
        blob = _INSPECT_CACHE.get(cid)
        if blob is None:
            continue
        port, tp = _port_from_blob(blob)
        if port:
            eng.append([safe, port, tp])
    return dock, eng


# Model ports a plain process (llama-server, a bare vLLM) may hold without a
# container. Only ones that are actually listening are asked /v1/models.
MODEL_PORTS = (8004, 8007, 8009, 8011, 8013, 8014, 8015, 8016, 8017, 8020, 8888)


def _http_ports(engines: list) -> list:
    ports = []
    for _name, port, _tp in engines:
        if port and port not in ports:
            ports.append(port)
    for port in MODEL_PORTS:
        if port not in ports and _listening(port):
            ports.append(port)
    return ports


def _served_id(body: str) -> str:
    """First model id: OpenAI ``data[].id``, else llama.cpp ``models[].name``."""
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        return ""
    if not isinstance(data, dict):
        return ""
    for key, field in (("data", "id"), ("models", "name"), ("models", "model")):
        items = data.get(key) or []
        if items and isinstance(items, list) and isinstance(items[0], dict) and items[0].get(field):
            return str(items[0][field])[:40]
    return ""


def _listen_rows() -> list[str]:
    rows: list[str] = []
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.extend(text.splitlines()[1:])
    return rows


def _decode_addr(hexip: str) -> str:
    try:
        raw = bytes.fromhex(hexip)
    except ValueError:
        return ""
    if len(raw) == 4:
        return ".".join(str(b) for b in raw[::-1])
    if len(raw) != 16:
        return ""
    parts: list[str] = []
    for index in range(0, 16, 4):
        word = int.from_bytes(raw[index:index + 4], "little")
        parts.append(f"{word >> 16:x}")
        parts.append(f"{word & 0xFFFF:x}")
    return ":".join(parts)


_LOOPBACK_BIND = {"127.0.0.1", "::1", "0:0:0:0:0:0:0:1"}
_WILDCARD_BIND = {"0.0.0.0", "::", "0:0:0:0:0:0:0:0"}


def _bind_addr(port: int) -> str:
    """Address on the listening socket, unchanged. Empty when the port is absent."""
    hexport = f"{int(port):04X}"
    found = ""
    for line in _listen_rows():
        parts = line.split()
        if len(parts) < 4 or parts[3] != "0A" or ":" not in parts[1]:
            continue
        ip_hex, port_hex = parts[1].rsplit(":", 1)
        if port_hex.upper() != hexport:
            continue
        ip = _decode_addr(ip_hex)
        if not ip:
            continue
        if ip not in _LOOPBACK_BIND and ip not in _WILDCARD_BIND:
            return ip
        if not found:
            found = ip
    return found


def _dial_host(port: int) -> str:
    """Where this port answers. A wildcard or loopback socket stays on loopback.

    A server bound to one address does not answer on loopback. Dial that address.
    """
    bound = _bind_addr(port)
    if not bound or bound in _LOOPBACK_BIND or bound in _WILDCARD_BIND:
        return "127.0.0.1"
    return bound


def _http(engines: list) -> dict:
    out = {}
    for port in _http_ports(engines):
        host = _dial_host(port)
        dial = f"[{host}]" if ":" in host else host
        try:
            with urllib.request.urlopen(f"http://{dial}:{port}/v1/models", timeout=0.4) as resp:
                body = resp.read(4096).decode("utf-8", "replace")
        except (OSError, urllib.error.URLError, TimeoutError):
            continue
        served = _served_id(body)
        if not served:
            continue
        try:
            with urllib.request.urlopen(f"http://{dial}:{port}/metrics", timeout=0.4) as resp:
                text = resp.read(200000).decode("utf-8", "replace")
        except (OSError, urllib.error.URLError, TimeoutError):
            text = ""
        metrics = _metrics_from_prom(text)
        if "pt" not in metrics and "gt" not in metrics:
            day = _call(lambda: LLAMA.counts(port), None)
            if day:
                metrics.update(day)
        bound = _bind_addr(port)
        evidence = host if host not in _LOOPBACK_BIND else (bound or host)
        row = [served, metrics, evidence]
        if evidence in _WILDCARD_BIND:
            route = _verify_advertised(port, served)
            if route:
                row.append(route)
        out[str(port)] = row
    return out


def _advertised_address() -> str:
    """Explicit client address from SPARKSI_ADVERTISE. Empty when unset."""
    address = _env("SPARKSI_ADVERTISE")
    if not address or address in _LOOPBACK_BIND or address in _WILDCARD_BIND:
        return ""
    if not all(ch.isalnum() or ch in ".:-" for ch in address):
        return ""
    return address


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect is a different server. Refuse it instead of following."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "redirect refused", headers, fp)


def _advertised_opener() -> urllib.request.OpenerDirector:
    """Direct GET only. No proxy from the environment, and no redirect."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _open_advertised(url: str):
    return _advertised_opener().open(url, timeout=ADVERTISE_TIMEOUT)


def _verify_advertised(port: int, served: str) -> str:
    """GET the configured address. Return it only when the served id matches.

    One direct request, bounded by ADVERTISE_TIMEOUT and a 4096-byte read.
    Redirects and proxies are refused. Nothing is remembered for the next sample.
    Loopback and wildcard are not candidates.
    """
    address = _advertised_address()
    if not address or not served:
        return ""
    dial = f"[{address}]" if ":" in address else address
    url = f"http://{dial}:{int(port)}/v1/models"
    try:
        with _open_advertised(url) as resp:
            final = resp.geturl()
            if final != url:
                return ""
            body = resp.read(4096)
    except (OSError, urllib.error.URLError, TimeoutError):
        return ""
    if isinstance(body, bytes):
        body = body.decode("utf-8", "replace")
    if _served_id(body) != served:
        return ""
    return address


# --- llama-server token counts -------------------------------------------
# llama-server prints one summary per finished request:
#   "<min>.<sec>.<ms>.<us> I slot print_timing: id 0 | task N | prompt eval time = ... / P tokens"
#   "...                                                  |        eval time = ... / G tokens"
# The clock is time since the process started. The log is appended across
# restarts, so a clock that goes backwards starts a new process segment and
# only the last segment (the running process) is counted. "in" is prompt
# tokens actually evaluated (cache hits excluded), the same thing
# llamacpp:prompt_tokens_total counts with --metrics.

_TS_RE = None


def _log_line(line: str):
    """(seconds since start, kind, tokens) or None. kind is 'p', 'g' or None."""
    global _TS_RE
    if _TS_RE is None:
        import re
        _TS_RE = (
            re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) "),
            re.compile(r"\|\s*(prompt eval|eval) time =\s*[\d.]+ ms /\s*(\d+) tokens"),
        )
    m = _TS_RE[0].match(line)
    if not m:
        return None
    secs = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1000.0
    if "print_timing" not in line or "eval time" not in line:
        return secs, None, 0
    t = _TS_RE[1].search(line)
    if not t:
        return secs, None, 0
    return secs, ("p" if t.group(1) == "prompt eval" else "g"), int(t.group(2))


def _listen_inode(port: int) -> str | None:
    hexport = f"{port:04X}"
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) >= 10 and parts[3] == "0A" and parts[1].rsplit(":", 1)[-1].upper() == hexport:
                return parts[9]
    return None


def _pid_for_inode(inode: str) -> int | None:
    want = f"socket:[{inode}]"
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    if os.readlink(fd) == want:
                        return int(proc.name)
                except OSError:
                    continue
        except OSError:
            continue
    return None


def _proc_start(pid: int) -> float | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        ticks = int(stat.rsplit(")", 1)[1].split()[19])
        btime = None
        for line in Path("/proc/stat").read_text().splitlines():
            if line.startswith("btime "):
                btime = int(line.split()[1])
                break
        if btime is None:
            return None
        return btime + ticks / float(os.sysconf("SC_CLK_TCK"))
    except (OSError, ValueError, IndexError):
        return None


def _day_key(epoch: float) -> int:
    return int(time.strftime("%Y%m%d", time.localtime(epoch)))


def _yesterday_key(epoch: float) -> int:
    """The previous local calendar day. Subtracting 86400 lands on the wrong date across DST."""
    local = datetime.fromtimestamp(epoch).astimezone()
    start = local.replace(hour=0, minute=0, second=0, microsecond=0)
    prior = start - timedelta(seconds=1)
    return int(prior.strftime("%Y%m%d"))


class LlamaCounts:
    """Per-port llama-server day counts from its log, read incrementally."""

    def __init__(self) -> None:
        self._ports: dict[int, dict] = {}

    def _find(self, port: int) -> dict | None:
        row = self._ports.get(port)
        if row:
            try:
                comm_ok = Path(f"/proc/{row['pid']}/comm").read_text().strip() == "llama-server"
            except OSError:
                comm_ok = False
            if comm_ok and _proc_start(row["pid"]) == row["start"]:
                return row
            self._ports.pop(port, None)
        inode = _listen_inode(port)
        if not inode or inode == "0":
            return None
        pid = _pid_for_inode(inode)
        if not pid:
            return None
        try:
            if Path(f"/proc/{pid}/comm").read_text().strip() != "llama-server":
                return None
            log = os.readlink(f"/proc/{pid}/fd/1")
        except OSError:
            return None
        start = _proc_start(pid)
        if not start or not log.startswith("/") or not Path(log).is_file():
            return None
        row = {"pid": pid, "start": start, "log": log, "off": 0, "ino": None,
               "days": {}, "last": -1.0, "rest": "", "gen": 0}
        self._ports[port] = row
        return row

    def _read(self, row: dict) -> None:
        try:
            st = os.stat(row["log"])
        except OSError:
            return
        if row["ino"] != st.st_ino or st.st_size < row["off"]:
            row.update({"ino": st.st_ino, "off": 0, "days": {}, "last": -1.0, "rest": "", "gen": 0})
        if st.st_size == row["off"]:
            return
        with open(row["log"], "rb") as fh:
            fh.seek(row["off"])
            chunk = fh.read(min(st.st_size - row["off"], 64 << 20))
        row["off"] += len(chunk)
        text = row["rest"] + chunk.decode("utf-8", "replace")
        lines = text.split("\n")
        row["rest"] = lines.pop()
        for line in lines:
            got = _log_line(line)
            if not got:
                continue
            secs, kind, n = got
            if secs + 1.0 < row["last"]:
                row["days"] = {}  # an earlier process's segment ended here
                row["gen"] = 0
            row["last"] = secs
            if kind:
                day = _day_key(row["start"] + secs)
                pair = row["days"].setdefault(day, [0, 0])
                pair[0 if kind == "p" else 1] += n
                if kind == "g":
                    row["gen"] = row.get("gen", 0) + n

    def counts(self, port: int, now: float | None = None) -> dict | None:
        row = self._find(int(port))
        if not row:
            return None
        self._read(row)
        now = time.time() if now is None else now
        today = _day_key(now)
        yday = _yesterday_key(now)
        for key in [k for k in row["days"] if k not in (today, yday)]:
            row["days"].pop(key, None)
        tp, tg = row["days"].get(today, [0, 0])
        yp, yg = row["days"].get(yday, [0, 0])
        slots = _call(lambda: _slots_live(int(port)), None)
        live = (slots["p"], slots["g"]) if slots else (0, 0)
        out = {
            "dp": int(tp + live[0]),
            "dg": int(tg + live[1]),
            "yp": int(yp),
            "yg": int(yg),
            "dy": today,
            "ps": int(row["start"]),
            "hl": 1 if _call(lambda: _health_ok(int(port)), False) else 0,
        }
        if slots is not None:
            out["r"] = int(slots["running"])
            if slots["running"] and slots["ctx"]:
                out["cx"] = round(slots["used"] / float(slots["ctx"]), 4)
            # Decode rate from this process's generated-token count, kept
            # monotonic so a request finishing between the log read and the
            # /slots read cannot dip and then spike.
            cum = max(int(row.get("cum") or 0), int(row.get("gen", 0)) + int(live[1]))
            mono = time.monotonic()
            last = row.get("rate_at")
            if last is not None and mono - last[0] > 0.5:
                out["gs"] = round(max(0.0, (cum - last[1]) / (mono - last[0])), 1)
            elif last is not None:
                out["gs"] = row.get("rate", 0.0)
            if last is None or mono - last[0] > 0.5:
                row["rate_at"] = (mono, cum)
                row["rate"] = out.get("gs", 0.0)
            row["cum"] = cum
            if not slots["running"]:
                out["gs"] = 0.0
        return out


def _slots_live(port: int) -> dict:
    """Running requests and their tokens so far, from llama-server /slots.

    p/g: prompt tokens evaluated and tokens decoded by requests still running
    (so day counts move mid-reply). used/ctx: context filled by running
    requests over their slots' n_ctx. Raises when /slots does not answer.
    """
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=0.4) as resp:
        data = json.loads(resp.read(200000).decode("utf-8", "replace"))
    if not isinstance(data, list):
        raise ValueError("slots is not a list")
    p = g = running = used = ctx = 0
    for slot in data:
        if not isinstance(slot, dict) or not slot.get("is_processing"):
            continue
        running += 1
        p += int(slot.get("n_prompt_tokens_processed") or 0)
        dec = 0
        nxt = slot.get("next_token") or []
        if isinstance(nxt, list) and nxt and isinstance(nxt[0], dict):
            dec = int(nxt[0].get("n_decoded") or 0)
        g += dec
        used += int(slot.get("n_prompt_tokens") or 0) + dec
        ctx += int(slot.get("n_ctx") or 0)
    return {"p": p, "g": g, "running": running, "used": used, "ctx": ctx}


def _health_ok(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=0.4) as resp:
            return resp.status == 200
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


LLAMA = LlamaCounts()


def _listening(port: int) -> bool:
    hexport = f"{port:04X}"
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 4 or parts[3] != "0A":
                continue
            if parts[1].rsplit(":", 1)[-1].upper() == hexport:
                return True
    return False


def _video_state() -> list:
    mode = ""
    try:
        mode = video_mode_file().read_text().strip().lower()[:8]
    except OSError:
        mode = ""
    return [1 if _video_answers() else 0, mode]


def _video_answers(port: int = VIDEO_SERVICE_PORT) -> bool:
    """The video service is up only when /system_stats answers on loopback.

    A bound :8188 is not enough: ``tailscale serve`` holds 100.x:8188 to proxy
    it, so the socket exists even when the video service is stopped.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=0.8) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read(65536).decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError, TimeoutError, ValueError):
        return False
    return isinstance(data, dict) and (isinstance(data.get("system"), dict) or "devices" in data)


def _call(fn, default):
    try:
        return fn()
    except Exception:
        return default


# --- Probe beat: a second datagram with what the controller used to ask over SSH.
# Built from data this loop already has (docker ps, cached inspect, /proc) plus
# two local HTTP GETs per listening port. It answers nothing and takes no input.
def _inspect_port(text: str) -> int:
    value = str(text or "").strip()
    if not value.isdigit():
        return 0
    port = int(value)
    return port if 1 <= port <= 65535 else 0


def _declared_ports(blob: str) -> list[int]:
    """``PORT``, ``VLLM_PORT``, ``SERVER_PORT``, then ``--port``: the agent's own rule."""
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


def _disk_gib() -> list[float]:
    """[total, free, used] GiB of / ."""
    usage = shutil.disk_usage("/")
    return [round(usage.total / 1024**3, 2), round(usage.free / 1024**3, 2), round(usage.used / 1024**3, 2)]


def _listen_table() -> list[list]:
    """LISTEN sockets on ports 1024..65535: [port, code] rows, <=32."""
    out: list[list] = []
    seen: set[tuple[int, str]] = set()
    for row in _listen_rows():
        parts = row.split()
        if len(parts) < 4:
            continue
        local, st = parts[1], parts[3]
        if st != "0A":
            continue
        _, _, port_s = local.rpartition(":")
        try:
            port = int(port_s, 16)
        except ValueError:
            continue
        if not (1024 <= port <= 65535):
            continue
        hexip = local.split(":", 1)[0]
        ip = _decode_addr(hexip)
        if ip in _WILDCARD_BIND:
            code = "w"
        elif ip in _LOOPBACK_BIND:
            code = "l"
        else:
            code = ip
        key = (port, code)
        if key in seen:
            continue
        seen.add(key)
        out.append([port, code])
    out.sort(key=lambda e: (e[0], e[1]))
    return out[:32]


def _identity(port: int, path: str, host: str) -> list:
    """GET http://host:port/path; [status, model_ids]; never raises."""
    shown = host
    if ":" in host and not host.startswith("["):
        shown = f"[{host}]"
    url = f"http://{shown}:{port}{path}"
    try:
        with urllib.request.urlopen(url, timeout=0.8) as resp:
            status = getattr(resp, "status", None)
            raw = resp.read(65536)
    except urllib.error.HTTPError as e:
        return [e.code, []]
    except (OSError, ValueError):
        return [None, []]
    try:
        data = json.loads(raw.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return [status, []]
    names: list[str] = []
    if isinstance(data, dict):
        for item in data.get("data") or []:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                names.append(item["id"])
        for sub in ("models",):
            for item in data.get(sub) or []:
                if isinstance(item, dict):
                    for k in ("name", "model"):
                        v = item.get(k)
                        if isinstance(v, str):
                            names.append(v)
    out: list[str] = []
    for n in names:
        n = n[:64]
        if n not in out:
            out.append(n)
        if len(out) >= 4:
            break
    return [status, out]


def _container_mem_gib(cid: str) -> float | None:
    """Working-set GiB for a container from cgroup v2 files; None on any problem."""
    if not (isinstance(cid, str) and len(cid) == 64
            and all(c in "0123456789abcdef" for c in cid)):
        return None
    base = f"/sys/fs/cgroup/system.slice/docker-{cid}.scope"
    try:
        with open(f"{base}/memory.current", "rb") as f:
            total = int(f.read().strip())
        with open(f"{base}/memory.stat", "rb") as f:
            stat = f.read().decode("ascii", "replace")
    except (OSError, ValueError):
        return None
    inactive = 0
    for line in stat.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == "inactive_file":
            try:
                inactive = int(parts[1])
            except ValueError:
                pass
            break
    val = total - inactive
    if val < 0:
        val = 0
    return round(val / (1024 ** 3), 2)


_STATS_EVERY = 15.0  # container memory, read from cgroup files
_STATS: dict = {"t": None, "v": {}}


def _meminfo_gib() -> list[float]:
    total = avail = 0.0
    try:
        with open("/proc/meminfo", "rb") as f:
            txt = f.read().decode("ascii", "replace")
    except OSError:
        return [0.0, 0.0]
    for line in txt.splitlines():
        key, _, rest = line.partition(":")
        kb = rest.strip().split()
        if not kb:
            continue
        try:
            kib = int(kb[0])
        except ValueError:
            continue
        if key == "MemTotal":
            total = kib / 1024 / 1024
        elif key == "MemAvailable":
            avail = kib / 1024 / 1024
    return [round(total, 2), round(avail, 2)]


def probe_beat(host: str, gpu: list, now: float | None = None) -> dict:
    """One cheap probe datagram as a plain dict (pre-JSON)."""
    if now is None:
        now = time.monotonic()
    ls = _listen_table()
    beat: dict = {
        "ph": host,
        "v": 1,
        "hn": socket.gethostname()[:64],
        "mm": _meminfo_gib(),
        "dk": _disk_gib(),
        "g": list(gpu[:3]),
        "ls": ls,
    }
    if _LAST_PS is None:
        beat["ct"] = None
    else:
        ct = []
        for cid, name in _LAST_PS[:16]:
            ct.append([name[:64], _declared_ports(_INSPECT_CACHE.get(cid, ""))[:4]])
        beat["ct"] = ct
        beat["cn"] = len(_LAST_PS)  # running containers in all; more than ct holds means ct is partial
    hi: dict = {}
    # Only ports a model may serve on: declared by a container, or a known model port.
    wanted = set(MODEL_PORTS)
    for _name, ports in beat["ct"] or []:
        wanted.update(ports)
    dial_by_port: dict[int, str] = {}
    for port, code in ls:
        if (video_on() and port == VIDEO_SERVICE_PORT) or port not in wanted:
            continue
        if port in dial_by_port:
            continue
        if code in ("w", "l"):
            dial_by_port[port] = "127.0.0.1"
        else:
            dial_by_port[port] = code
    for port in sorted(dial_by_port)[:6]:
        dial = dial_by_port[port]
        h = _identity(port, "/health", dial)
        v = _identity(port, "/v1/models", dial)
        hi[str(port)] = [h[0], h[1], v[0], v[1]]
    beat["hi"] = hi
    if _STATS["t"] is None or (now - _STATS["t"]) >= _STATS_EVERY:
        cs: dict = {}
        if _LAST_PS:
            for cid, name in _LAST_PS:
                if not name.startswith("sparksi-"):
                    continue
                m = _container_mem_gib(cid)
                if m is not None:
                    cs[name] = m
                if len(cs) >= 8:
                    break
        _STATS["t"] = now
        _STATS["v"] = cs
    beat["cs"] = dict(_STATS["v"])
    return beat


def encode_probe(beat: dict) -> bytes | None:
    """Compact JSON; shed weight: models, then hi, else give up."""
    pkt = json.dumps(beat, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(pkt) <= MAX_BYTES:
        return pkt
    trimmed = {k: v for k, v in beat.items()}
    hi = trimmed.get("hi")
    if isinstance(hi, dict):
        trimmed["hi"] = {k: [v[0], [], v[2], []] if isinstance(v, list) and len(v) >= 3
                         else v for k, v in hi.items()}
        pkt = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(pkt) <= MAX_BYTES:
            return pkt
        trimmed["hi"] = {}
        pkt = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(pkt) <= MAX_BYTES:
            return pkt
    else:
        trimmed["hi"] = {}
        pkt = json.dumps(trimmed, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(pkt) <= MAX_BYTES:
            return pkt
    return None


def sample(host: str) -> dict:
    budget = _call(_budget, {"top": [], "cache": 0, "model": 0, "slab": 0, "small": 0})
    dock, eng = _call(_engines, ([], []))
    pkt = {
        "h": host,
        "mem": _call(_mem, [0.0, 0.0]),
        "cpu": [_call(_cpu_pct, None), _call(_cpu_mhz, None)],
        "gpu": _call(_gpu, [None, None, None, None]),
        "z": _call(_zones, []),
        "top": budget["top"],
        "cache": budget["cache"],
        "model": budget["model"],
        "slab": budget["slab"],
        "small": budget["small"],
        "dock": dock,
        "eng": eng,
        "http": _call(lambda: _http(eng), {}),
    }
    if video_on():
        # Both keys during the rename: an older controller reads only the legacy one.
        state = _call(_video_state, [0, ""])
        pkt[VIDEO_PACKET_KEY] = state
        pkt[LEGACY_VIDEO_PACKET_KEY] = state
    return pkt


def encode(pkt: dict) -> bytes:
    """JSON under 1200 bytes. Tailscale drops a larger UDP datagram."""
    def dump(obj: dict) -> bytes:
        return json.dumps(obj, separators=(",", ":")).encode("utf-8")

    raw = dump(pkt)
    if len(raw) <= MAX_BYTES:
        return raw
    shrunk = dict(pkt)
    shrunk["top"] = list(shrunk.get("top") or [])[:3]
    shrunk["z"] = list(shrunk.get("z") or [])[:4]
    raw = dump(shrunk)
    if len(raw) <= MAX_BYTES:
        return raw
    shrunk["top"] = []
    shrunk["z"] = []
    shrunk["dock"] = [[row[0], "Up"] for row in (shrunk.get("dock") or [])[:4]]
    raw = dump(shrunk)
    if len(raw) <= MAX_BYTES:
        return raw
    http = {}
    for port, pair in (shrunk.get("http") or {}).items():
        served = pair[0][:24] if pair else ""
        metrics = {k: pair[1][k] for k in ("r", "w", "kv") if pair and k in (pair[1] or {})}
        http[port] = [served, metrics]
    shrunk["http"] = http
    raw = dump(shrunk)
    if len(raw) <= MAX_BYTES:
        return raw
    return dump({"h": pkt.get("h"), "mem": pkt.get("mem"), "cpu": pkt.get("cpu"), "gpu": pkt.get("gpu")})


def to_peer(pkt: dict) -> dict:
    """Announced packet -> the dict the sparksi.monitor snapshot already accepts."""
    mem = pkt.get("mem") or [0.0, 0.0]
    cpu = pkt.get("cpu") or [None, None]
    gpu = list(pkt.get("gpu") or [None, None, None])
    while len(gpu) < 3:
        gpu.append(None)
    power_w = _num(gpu[3]) if len(gpu) > 3 else None
    zones = []
    for row in pkt.get("z") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and _num(row[1]) is not None:
            zones.append((str(row[0]), float(row[1])))
    consumers = []
    if _num(pkt.get("slab")) is not None:
        consumers.append({"name": "kernel slab", "gib": float(pkt["slab"]), "kind": "kernel"})
    for row in pkt.get("top") or []:
        if isinstance(row, (list, tuple)) and len(row) >= 2 and _num(row[1]) is not None:
            consumers.append({"name": str(row[0])[:16], "gib": float(row[1]), "kind": "process"})
    if _num(pkt.get("small")) is not None:
        consumers.append({"name": "smaller processes", "gib": float(pkt["small"]), "kind": "process"})
    containers = []
    for row in pkt.get("dock") or []:
        if isinstance(row, (list, tuple)) and row:
            containers.append({"name": str(row[0]), "status": str(row[1] if len(row) > 1 else "Up")})
    engines = []
    for row in pkt.get("eng") or []:
        if not isinstance(row, (list, tuple)) or not row:
            continue
        port = int(row[1]) if len(row) > 1 and str(row[1]).isdigit() else 0
        tp = int(row[2]) if len(row) > 2 and str(row[2]).isdigit() else 0
        engines.append({"name": str(row[0]), "status": "Up", "port": port or None, "tp": tp or None})
    http = {}
    for port, pair in (pkt.get("http") or {}).items():
        if not str(port).isdigit() or not isinstance(pair, (list, tuple)) or not pair:
            continue
        short = pair[1] if len(pair) > 1 and isinstance(pair[1], dict) else {}
        metrics = {}
        for key, val in short.items():
            full = METRIC_SHORT.get(str(key))
            num = _num(val)
            if full and num is not None:
                metrics[full] = num
        served = str(pair[0] or "") or None
        if served or metrics:
            record = {"served": served, "metrics": metrics or None}
            evidence = pair[2] if len(pair) > 2 and isinstance(pair[2], str) else ""
            if evidence in {"0.0.0.0", "::", "0:0:0:0:0:0:0:0"}:
                record["bind"] = "wildcard"
                route = pair[3] if len(pair) > 3 and isinstance(pair[3], str) else ""
                if route and route not in {"0.0.0.0", "::", "127.0.0.1", "::1"} and all(
                    ch.isalnum() or ch in ".:-" for ch in route
                ):
                    record["route"] = route
            elif evidence in {"127.0.0.1", "::1", "0:0:0:0:0:0:0:1"}:
                record["bind"] = "loopback"
            elif evidence and all(ch.isalnum() or ch in ".:-" for ch in evidence):
                record["bind"] = "specific"
                record["listen"] = evidence
            http[int(port)] = record
    video = (pkt.get(VIDEO_PACKET_KEY) if VIDEO_PACKET_KEY in pkt else pkt.get(LEGACY_VIDEO_PACKET_KEY)) or [0, ""]
    listen = bool(video[0]) if isinstance(video, (list, tuple)) and video else False
    mode = ""
    if isinstance(video, (list, tuple)) and len(video) > 1 and video[1]:
        mode = str(video[1])
    video_service = {"mode": mode or None, "rss_kb": 0, "listening": listen}
    return {
        "cpu_pct": _num(cpu[0]) if cpu else None,
        "cpu_mhz": _num(cpu[1]) if cpu and len(cpu) > 1 else None,
        "gpu": gpu[:3],
        "power_w": power_w,
        "mem": [float(mem[0] or 0), float(mem[1] or 0)] if len(mem) >= 2 else [0.0, 0.0],
        "zones": zones,
        "containers": containers,
        "http": http,
        "engines": engines,
        "video_service": video_service,
        LEGACY_MONITOR_VIDEO_KEY: dict(video_service),
        "truth": "",
        "budget": {
            "cache_gib": _num(pkt.get("cache")),
            "model_gib": _num(pkt.get("model")),
            "consumers": consumers,
        },
    }


def main() -> None:
    host = _env("SPARKSI_NODE_ID", "DGX_ANNOUNCE_HOST")
    ids = node_ids()
    if host not in ids:
        raise SystemExit(f"SPARKSI_NODE_ID must be one of {', '.join(ids)}")
    target = dest()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
        started = time.monotonic()
        try:
            pkt = sample(host)
            raw = encode(pkt)
            if len(raw) <= MAX_BYTES:
                sock.sendto(raw, target)
            beat = encode_probe(probe_beat(host, pkt.get("gpu") or []))
            if beat is not None:
                sock.sendto(beat, target)
        except Exception:
            pass
        delay = INTERVAL - (time.monotonic() - started)
        if delay > 0:
            time.sleep(delay)


if __name__ == "__main__":
    main()
