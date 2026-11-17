# SPDX-License-Identifier: Apache-2.0

"""Peer node collector (by default the worker node from config).

One ssh exec per poll pulls a dumb read-only dump: /proc, sysfs, nvidia-smi,
`docker ps`, and curl of ports a container declared with PORT= or --port.
/metrics is requested only after /v1/models returns a body. Gloo and other
undeclared listeners are never connected. NO logic runs on the peer — this side parses
and does all the math. Background thread, ~3 s cadence; the 1 s render loop
reads the cached value. The desktop window reads the controller's /api/monitor and
does not start a second poll while that answers.
"""

import base64
import shlex
import subprocess
import threading
import time

from sparksi import compat, protect
from sparksi.config import CFG
from sparksi.hostcheck import check_ssh

from .models import WANTED_METRICS, parse_metrics_text, served_from_models_json

HOST = CFG.worker.ssh if CFG.has_worker else CFG.head.ssh
INTERVAL = 3.0
STALE_SECS = 10.0

# bash -s over ssh. Read-only. Sections marked for parsing on this side.
_REMOTE_TEMPLATE = r"""
echo _CPU1_
head -n1 /proc/stat
sleep 0.3
echo _CPU2_
head -n1 /proc/stat
echo _MEM_
grep -E '^Mem(Total|Available):' /proc/meminfo
echo _FREQ_
cat /sys/devices/system/cpu/cpu*/cpufreq/cpuinfo_avg_freq
echo _GPU_
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,clocks.current.sm --format=csv,noheader,nounits
echo _ZONE_
for z in /sys/class/thermal/thermal_zone*; do
  printf '%s %s\n' "$(cat "$z/type")" "$(cat "$z/temp")"
done
echo _DOCKER_
timeout 2 docker ps --format '{{.Names}} {{.Status}}' 2>/dev/null || true
echo _ENGINES_
# Only ports the container itself declares (PORT= or --port). A gloo or
# TCPStore listener is not one of those, and curling it leaks accepts.
http_ports=""
while IFS= read -r name; do
  [ -n "$name" ] || continue
  blob=$(timeout 2 docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}{{println "---CMD---"}}{{range .Config.Cmd}}{{println .}}{{end}}' "$name" 2>/dev/null || true)
  port=$(printf '%s\n' "$blob" | sed -n 's/^PORT=//p' | head -n 1)
  if [ -z "$port" ]; then
    port=$(printf '%s\n' "$blob" | awk '
      $0 == "---CMD---" { cmd = 1; next }
      cmd && $0 ~ /^--port=/ { sub(/^--port=/, ""); print; exit }
      cmd && prev == "--port" { print; exit }
      { prev = $0 }
    ')
  fi
  tp=$(printf '%s\n' "$blob" | sed -n 's/^TP=//p' | head -n 1)
  if [ -z "$tp" ]; then
    tp=$(printf '%s\n' "$blob" | sed -n 's/^TENSOR_PARALLEL_SIZE=//p' | head -n 1)
  fi
  echo "$name ${port:-0} ${tp:-0}"
  case "$port" in
    ""|0|*[!0-9]*) ;;
    *) http_ports="$http_ports $port" ;;
  esac
done <<EOF
$(timeout 2 docker ps --format '{{.Names}}' 2>/dev/null || true)
EOF
echo _HTTP_
listen=$(ss -H -ltn 2>/dev/null || true)
# Declared ports only. Skip the direct node link (__DIRECT_RE__). Curl loopback when the
# socket is wildcard or localhost; otherwise the address it is bound to.
printf '%s\n' "$listen" | awk '
{
  addr = $4
  ip = addr
  port = addr
  if (addr ~ /^\[/) {
    ip = substr(addr, 2, index(addr, "]") - 2)
    sub(/^.*\]:/, "", port)
  } else {
    n = split(addr, p, ":")
    port = p[n]
    ip = p[1]
    sub(/%.*$/, "", ip)
  }
  if (port !~ /^[0-9]+$/) next
  if (ip ~ /__DIRECT_RE__/) next
  if (ip == "0.0.0.0" || ip == "*" || ip == "::" || ip == "127.0.0.1" || ip == "::1") {
    via[port] = "127.0.0.1"
  } else if (!(port in via) && (ip ~ /^100\./ || ip ~ /^fd7a:/)) {
    if (ip ~ /^fd7a:/) via[port] = "[" ip "]"
    else via[port] = ip
  }
}
END { for (port in via) print port, via[port] }
' | while read -r p host; do
  [ -n "$p" ] || continue
  case " $http_ports " in
    *" $p "*) ;;
    *) continue ;;
  esac
  mj=$(curl -sf --connect-timeout 0.3 -m 0.4 "http://${host}:${p}/v1/models" | tr '\n' ' ' || true)
  if [ -n "$mj" ]; then
    echo "$p MODEL $mj"
    curl -sf --connect-timeout 0.3 -m 0.4 "http://${host}:${p}/metrics" \
      | grep -E '__GREP__' | while IFS= read -r line; do
          echo "$p METRIC $line"
        done
  fi
done
__VIDEO_SECTION_START__
echo _VIDEO_
if [ -r __VIDEO_MODE__ ]; then
  echo "MODE $(tr -d '[:space:]' < __VIDEO_MODE__)"
else
  echo MODE
fi
rss=$(ps -eo rss=,args= | awk '/__VIDEO_RSS_RE__/ {s+=$1} END {print s+0}')
echo "RSS ${rss:-0}"
if printf '%s\n' "$listen" | grep -Eq ':__VIDEO_PORT__([^0-9]|$)'; then
  echo LISTEN 1
else
  echo LISTEN 0
fi
__VIDEO_SECTION_END__
echo _BUDGET_
python3 -c 'import base64; exec(base64.b64decode("__BUDGET_B64__"))'
echo _INVENTORY_
if [ -r __MODEL_INVENTORY__ ]; then
  cat __MODEL_INVENTORY__
fi
"""


# Budget probe run on the peer (sent base64 so the bash heredoc needs no quoting).
_BUDGET_PY = r'''import os
from collections import defaultdict
info = {}
for line in open("/proc/meminfo"):
    k, v = line.split(":", 1)
    info[k] = int(v.split()[0])
def kib(key):
    return info.get(key, 0)
GiB = 1024.0 * 1024.0
cache = kib("Buffers") + kib("Cached") + kib("SReclaimable")
slab = kib("SReclaimable") + kib("SUnreclaim")
small = kib("KernelStack") + kib("PageTables") + kib("Percpu") + kib("SecPageTables")
gap = kib("MemTotal") - kib("MemFree") - kib("AnonPages") - kib("Buffers") - kib("Cached") - slab - small
print("CACHE %.4f" % (cache / GiB))
print("MODEL %.4f" % (max(0.0, gap) / GiB))
print("SLAB %.4f" % (kib("SUnreclaim") / GiB))
rss = defaultdict(int)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    try:
        comm = open("/proc/%s/comm" % pid).read().strip()
        parts = open("/proc/%s/statm" % pid).read().split()
        res = int(parts[1]) * 4096
        cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\x00", b" ").decode("utf-8", "replace").lower()
    except OSError:
        continue
    low = comm.lower()
    if low.startswith("vllm") or "__VIDEO_SUB__" in low or "vllm" in cmd or "__VIDEO_SUB__" in cmd:
        continue
    rss[comm] += res
rest = 0
for name, nbytes in sorted(rss.items(), key=lambda kv: -kv[1]):
    gib = nbytes / (1024.0 ** 3)
    if gib < 0.05:
        rest += nbytes
        continue
    safe = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in name)
    print("PROC %s %.4f" % (safe, gib))
if rest:
    print("SMALL %.4f" % (rest / (1024.0 ** 3)))
'''


def remote_script():
    """Fill the metric grep. HTTP targets are container-declared ports only."""
    names = "|".join(sorted(WANTED_METRICS))
    grep = f"^(vllm:)?({names})([ {{]|$)"
    direct = [str(p) for p in CFG.paths.get("direct_prefixes") or []] or ["198.18.0."]
    direct_re = "^(" + "|".join(p.replace(".", "\\.") for p in direct) + ")"
    video_root = CFG.path("video_service_root") or _shell_home(compat.VIDEO_SERVICE_ROOT_FALLBACKS[-1])
    inventory = CFG.path("model_inventory") or _shell_home(compat.MODEL_INVENTORY_FALLBACKS[-1])
    template = _REMOTE_TEMPLATE
    start, end = "__VIDEO_SECTION_START__\n", "__VIDEO_SECTION_END__\n"
    head, rest = template.split(start, 1)
    section, tail = rest.split(end, 1)
    if not protect.video_on():
        section = "echo _VIDEO_\necho MODE\necho RSS 0\necho LISTEN 0\n"
    template = head + section + tail
    return (
        template.replace("__GREP__", grep)
        .replace("__DIRECT_RE__", direct_re)
        .replace("__VIDEO_MODE__", shlex.quote(video_root.rstrip("/") + "/state/mode"))
        .replace("__MODEL_INVENTORY__", shlex.quote(inventory))
        .replace("__VIDEO_RSS_RE__", protect.rss_awk())
        .replace("__VIDEO_PORT__", str(protect.video_port()))
        .replace("__BUDGET_B64__", _budget_b64())
    )


def _shell_home(path):
    """``~/x`` -> ``$HOME/x`` for the remote shell."""
    return "$HOME" + path[1:] if path.startswith("~") else path


def _budget_b64():
    subs = protect.process_substrings()
    if subs:
        src = _BUDGET_PY.replace("__VIDEO_SUB__", subs[0])
    else:
        src = _BUDGET_PY.replace(' or "__VIDEO_SUB__" in low', "").replace(' or "__VIDEO_SUB__" in cmd', "")
    return base64.b64encode(src.encode()).decode()


def _sections(out):
    secs, name, buf = {}, None, []
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("_") and s.endswith("_") and len(s) > 2:
            if name is not None:
                secs[name] = buf
            name, buf = s.strip("_"), []
        elif name is not None:
            buf.append(line)
    if name is not None:
        secs[name] = buf
    return secs


def _stat_delta(line):
    """cpu busy% between the two /proc/stat samples."""
    v = [int(x) for x in line.split()[1:]]
    return sum(v), v[3] + (v[4] if len(v) > 4 else 0)


def _parse_docker(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        name, _, status = line.partition(" ")
        rows.append({"name": name, "status": status})
    return rows


def _parse_http(lines):
    """`PORT MODEL json` / `PORT METRIC prom-line` → {port: {served, metrics}}."""
    blobs = {}
    for line in lines:
        parts = line.split(None, 2)
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        port = int(parts[0])
        kind = parts[1]
        rest = parts[2] if len(parts) > 2 else ""
        slot = blobs.setdefault(port, {"model": "", "metrics": []})
        if kind == "MODEL" and rest.strip():
            slot["model"] = rest
        elif kind == "METRIC" and rest.strip():
            slot["metrics"].append(rest)
    out = {}
    for port, slot in blobs.items():
        served = served_from_models_json(slot["model"])
        metrics = parse_metrics_text("\n".join(slot["metrics"]))
        if served is None and not metrics:
            continue
        out[port] = {"served": served, "metrics": metrics or None}
    return out


def _parse_engines(lines):
    """`name port tp` from `docker ps` + inspect. port/tp 0 means unset."""
    rows = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        port = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        tp = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 0
        rows.append({
            "name": parts[0],
            "status": "Up",
            "port": port or None,
            "tp": tp or None,
        })
    return rows


def _parse_video(lines):
    info = {"mode": None, "rss_kb": 0, "listening": False}
    for line in lines:
        parts = line.split(None, 1)
        if not parts:
            continue
        if parts[0] == "MODE" and len(parts) > 1 and parts[1].strip():
            info["mode"] = parts[1].strip().lower()
        elif parts[0] == "RSS" and len(parts) > 1 and parts[1].strip().isdigit():
            info["rss_kb"] = int(parts[1].strip())
        elif parts[0] == "LISTEN" and len(parts) > 1:
            info["listening"] = parts[1].strip() not in ("", "0")
    return info


def parse(out):
    """Dump text -> sensors plus containers, HTTP, engines, video service, truth."""
    s = _sections(out)
    data = {"cpu_pct": None, "cpu_mhz": None, "gpu": None,
            "mem": None, "zones": [], "containers": [], "http": {},
            "engines": [], "video_service": {}, "truth": ""}

    c1, c2 = s.get("CPU1", []), s.get("CPU2", [])
    if c1 and c2:
        t0, i0 = _stat_delta(c1[0])
        t1, i1 = _stat_delta(c2[0])
        dt, di = t1 - t0, i1 - i0
        if dt > 0:
            data["cpu_pct"] = max(0.0, min(100.0, 100.0 * (dt - di) / dt))

    freqs = []
    for line in s.get("FREQ", []):
        line = line.strip()
        if line.isdigit():
            khz = int(line)
            if khz > 0:
                freqs.append(khz / 1000.0)
    if freqs:
        data["cpu_mhz"] = sum(freqs) / len(freqs)

    mem = {}
    for line in s.get("MEM", []):
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
            mem[parts[0].rstrip(":")] = int(parts[1])
    if "MemTotal" in mem:
        data["mem"] = [mem["MemTotal"] / 1024.0 / 1024.0,
                       mem.get("MemAvailable", 0) / 1024.0 / 1024.0]

    g = s.get("GPU", [""])[0].split(",")
    if len(g) >= 3:
        vals = []
        for x in g[:3]:
            x = x.strip()
            try:
                vals.append(float(x))
            except ValueError:
                vals.append(None)
        data["gpu"] = vals

    for i, line in enumerate(s.get("ZONE", [])):
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            data["zones"].append((f"{parts[0]}{i}", int(parts[1]) / 1000.0))

    data["containers"] = _parse_docker(s.get("DOCKER", []))
    data["http"] = _parse_http(s.get("HTTP", []))
    data["engines"] = _parse_engines(s.get("ENGINES", []))
    data["video_service"] = _parse_video(s.get("VIDEO", []))
    data[compat.LEGACY_MONITOR_VIDEO_KEY] = dict(data["video_service"])
    data["truth"] = "\n".join(s.get("INVENTORY", []))
    data["budget"] = _parse_budget(s.get("BUDGET", []))
    return data


def _parse_budget(lines):
    out = {"cache_gib": None, "model_gib": None, "consumers": []}
    for line in lines:
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            gib = float(parts[-1])
        except ValueError:
            continue
        kind = parts[0]
        if kind == "CACHE":
            out["cache_gib"] = gib
        elif kind == "MODEL":
            out["model_gib"] = gib
        elif kind == "SLAB":
            out["consumers"].append({"name": "kernel slab", "gib": round(gib, 2), "kind": "kernel"})
        elif kind == "PROC" and len(parts) >= 3:
            out["consumers"].append({"name": parts[1], "gib": round(gib, 2), "kind": "process"})
        elif kind == "SMALL":
            out["consumers"].append({"name": "smaller processes", "gib": round(gib, 2), "kind": "process"})
    out["consumers"].sort(key=lambda row: -row["gib"])
    return out


class Peer:
    """Background ssh poller; snapshot() returns (data_or_None, age_secs)."""

    def __init__(self, host=HOST, interval=INTERVAL):
        self.host = check_ssh(host)  # never a destination ssh could read as an option
        self.interval = interval
        self._lock = threading.Lock()
        self._data = None
        self._ts = 0.0
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=2",
                     "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=1",
                     "--", self.host, "bash", "-s"],
                    input=remote_script(),
                    capture_output=True, text=True, timeout=25,
                ).stdout
                data = parse(out)
                if data["gpu"] is not None or data["mem"] is not None:
                    with self._lock:
                        self._data = data
                        self._ts = time.monotonic()
            except Exception:
                pass  # keep last value; staleness surfaces it
            self._stop.wait(self.interval)

    def snapshot(self):
        with self._lock:
            return self._data, (time.monotonic() - self._ts) if self._ts else None
