# SPDX-License-Identifier: Apache-2.0

"""Live observation only. Never start/stop/heal anything."""

from __future__ import annotations

import json
import os
import pwd
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from sparksi import compat, protect
from sparksi.never import Refused, subprocess_is_readonly

from sparksi.config import env
from sparksi.catalog import core_ids

SMI_TIMEOUT_S = float(env("SPARKSI_SMI_TIMEOUT", "5"))
HTTP_TIMEOUT_S = float(env("SPARKSI_HTTP_TIMEOUT", "1.0"))
CMD_TIMEOUT_S = float(env("SPARKSI_CMD_TIMEOUT", "5"))


def run_readonly(argv: list[str], timeout: float = CMD_TIMEOUT_S) -> tuple[int, str, str]:
    if not subprocess_is_readonly(argv):
        raise Refused(f"blocked non-readonly argv: {argv!r}")
    try:
        p = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError:
        return 127, "", "not found"


def meminfo_gib() -> dict[str, float | None]:
    raw: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                raw[k] = int(v.split()[0])
    except OSError:
        return {
            "mem_total_gib": None,
            "mem_avail_gib": None,
            "mem_free_gib": None,
            "swap_total_gib": None,
            "swap_free_gib": None,
        }

    def gib(kb: int | None) -> float | None:
        if kb is None:
            return None
        return round(kb / (1024 * 1024), 3)

    st = raw.get("SwapTotal", 0)
    sf = raw.get("SwapFree", 0)
    return {
        "mem_total_gib": gib(raw.get("MemTotal")),
        "mem_avail_gib": gib(raw.get("MemAvailable")),
        "mem_free_gib": gib(raw.get("MemFree")),
        "swap_total_gib": gib(st),
        "swap_free_gib": gib(sf),
    }


def port_listening(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.25)
    try:
        return s.connect_ex((host, int(port))) == 0
    except OSError:
        return False
    finally:
        s.close()


def http_get(url: str, timeout: float = HTTP_TIMEOUT_S) -> tuple[int | None, str | None]:
    req = Request(url, headers={"User-Agent": "sparksi/0.6"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except URLError:
        return None, None
    except TimeoutError:
        return None, None
    except Exception:
        return None, None


def _pid_cmdline(pid: int, limit: int = 400) -> str | None:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if not raw:
            return None
        return raw.replace(b"\x00", b" ").decode("utf-8", "replace").strip()[:limit]
    except OSError:
        return None


def _pid_comm(pid: int) -> str | None:
    try:
        return Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        return None


def _pid_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _pid_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _pid_uids(pid: int) -> tuple[int, int] | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("Uid:"):
                parts = line.split()
                return int(parts[1]), int(parts[2])
    except OSError:
        return None
    return None


def iter_pids() -> list[int]:
    out: list[int] = []
    try:
        for name in os.listdir("/proc"):
            if name.isdigit():
                out.append(int(name))
    except OSError:
        pass
    return out


def resolve_inference_identity(pid: int | None) -> dict[str, Any]:
    out: dict[str, Any] = {"served_model": None, "port": None}
    if not pid:
        return out
    cur = pid
    for _ in range(4):
        cmd = _pid_cmdline(cur, limit=800) or ""
        if any(x in cmd for x in ("vllm", "llama-server", "--served-model-name", "--port")):
            parts = cmd.split()
            for i, p in enumerate(parts):
                if p == "--served-model-name" and i + 1 < len(parts):
                    names = []
                    for q in parts[i + 1 :]:
                        if q.startswith("--"):
                            break
                        names.append(q)
                    if names:
                        out["served_model"] = names[0]
                if p == "--port" and i + 1 < len(parts):
                    try:
                        out["port"] = int(parts[i + 1])
                    except ValueError:
                        pass
            if out["served_model"] or out["port"]:
                return out
        parent = _pid_ppid(cur)
        if not parent or parent <= 1:
            break
        cur = parent
    return out


def compute_apps() -> tuple[list[dict[str, Any]] | None, str | None]:
    rc, out, err = run_readonly(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout=SMI_TIMEOUT_S,
    )
    if rc != 0:
        return None, err.strip() or f"nvidia-smi exit {rc}"
    apps: list[dict[str, Any]] = []
    for line in out.splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 2:
            continue
        try:
            pid = int(float(cols[0]))
        except ValueError:
            pid = None
        try:
            used = float(cols[2]) if len(cols) > 2 else None
        except ValueError:
            used = None
        ident = resolve_inference_identity(pid)
        apps.append(
            {
                "pid": pid,
                "name": cols[1],
                "used_gpu_mem_mib": used,
                "used_gpu_mem_gib": round(used / 1024, 3) if used is not None else None,
                "cmdline_head": _pid_cmdline(pid) if pid else None,
                "served_model": ident.get("served_model"),
                "port": ident.get("port"),
            }
        )
    return apps, None


def systemctl_query(unit: str) -> dict[str, Any]:
    active_rc, active_out, _ = run_readonly(["systemctl", "is-active", unit])
    en_rc, en_out, _ = run_readonly(["systemctl", "is-enabled", unit])
    show_rc, show_out, _ = run_readonly(
        ["systemctl", "show", unit, "-p", "ActiveState", "-p", "SubState", "-p", "UnitFileState"]
    )
    props: dict[str, str] = {}
    if show_rc == 0:
        for line in show_out.splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                props[k] = v
    return {
        "unit": unit,
        "is_active": (active_out.strip() if active_rc == 0 else active_out.strip() or "unknown"),
        "is_enabled": (en_out.strip() if en_rc == 0 else en_out.strip() or "unknown"),
        "active_state": props.get("ActiveState"),
        "sub_state": props.get("SubState"),
        "unit_file_state": props.get("UnitFileState"),
    }


def docker_ps() -> tuple[list[dict[str, str]] | None, str | None]:
    rc, out, err = run_readonly(
        ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}\t{{.Ports}}"]
    )
    if rc != 0:
        return None, err.strip() or f"docker ps exit {rc}"
    rows: list[dict[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if not parts or not parts[0]:
            continue
        rows.append(
            {
                "name": parts[0],
                "status": parts[1] if len(parts) > 1 else "",
                "ports": parts[2] if len(parts) > 2 else "",
            }
        )
    return rows, None


DOCKER_STATS_ARGV = ["docker", "stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}"]


def docker_stats_gib(
    name_substr: str, raw: tuple[int, str, str] | None = None
) -> tuple[float | None, list[dict[str, Any]], str | None]:
    """Memory of containers whose name contains ``name_substr``.

    ``docker stats --no-stream`` samples for about two seconds. A caller on a
    request path passes ``raw`` (the cached ``(rc, out, err)`` of that command).
    """
    rc, out, err = raw if raw is not None else run_readonly(list(DOCKER_STATS_ARGV))
    if rc != 0:
        return None, [], err.strip() or f"docker stats exit {rc}"
    matched: list[dict[str, Any]] = []
    total = 0.0
    found = False
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name, usage = parts[0], parts[1]
        if name_substr.lower() not in name.lower():
            continue
        used_s = usage.split("/", 1)[0].strip()
        gib = _parse_docker_mem(used_s)
        found = True
        matched.append({"name": name, "raw": usage, "gib": gib})
        if gib is not None:
            total += gib
    if not found:
        return 0.0, [], None
    if any(m["gib"] is None for m in matched):
        return None, matched, "unparsed docker stats"
    return round(total, 3), matched, None


def _parse_docker_mem(s: str) -> float | None:
    s = s.strip()
    if not s:
        return None
    units = [
        ("GiB", 1.0),
        ("GB", 1.0),
        ("MiB", 1 / 1024),
        ("MB", 1 / 1000),
        ("KiB", 1 / (1024 * 1024)),
        ("KB", 1 / 1_000_000),
        ("B", 1 / (1024**3)),
    ]
    for suf, mul in units:
        if s.endswith(suf):
            try:
                return float(s[: -len(suf)].strip()) * mul
            except ValueError:
                return None
    return None


def rss_match_gib(
    *,
    comms: list[str] | None = None,
    cmdline_substr: list[str] | None = None,
    user: str | None = None,
) -> tuple[float | None, list[dict[str, Any]], str | None]:
    uid = None
    if user:
        try:
            uid = pwd.getpwnam(user).pw_uid
        except KeyError:
            return None, [], f"unknown user {user}"
    comms_l = [c.lower() for c in (comms or [])]
    subs = [s.lower() for s in (cmdline_substr or [])]
    hits: list[dict[str, Any]] = []
    total_kib = 0
    for pid in iter_pids():
        if uid is not None:
            uids = _pid_uids(pid)
            if not uids or uids[0] != uid:
                continue
        comm = (_pid_comm(pid) or "").lower()
        cmd = (_pid_cmdline(pid) or "").lower()
        ok = False
        if comms_l and comm in comms_l:
            ok = True
        if subs and any(s in cmd or s in comm for s in subs):
            ok = True
        if not ok:
            continue
        rss = _pid_rss_kib(pid)
        if rss is None:
            continue
        total_kib += rss
        hits.append({"pid": pid, "comm": comm, "rss_kib": rss})
    return round(total_kib / (1024 * 1024), 3), hits, None


def probe_http_identity(host: str, port: int, path: str) -> dict[str, Any]:
    listening = port_listening(port, host)
    result: dict[str, Any] = {
        "host": host,
        "port": port,
        "path": path,
        "listening": listening,
        "http_ok": False,
        "identity": None,
        "error": None,
    }
    if not listening:
        return result
    url = f"http://{host}:{port}{path}"
    status, body = http_get(url)
    if status is None or body is None:
        result["error"] = "http_failed"
        return result
    result["http_status"] = status
    if status >= 400:
        result["error"] = f"http_{status}"
        return result
    result["http_ok"] = True
    ident = _parse_identity(path, body)
    result["identity"] = ident
    if ident == compat.VIDEO_HTTP_LABEL:
        result["legacy_identity"] = compat.LEGACY_VIDEO_HTTP_LABEL
    return result


def _parse_identity(path: str, body: str) -> str | None:
    if path.startswith("/v1/models"):
        try:
            data = json.loads(body)
            rows = data.get("data") or []
            if rows and isinstance(rows[0], dict):
                return rows[0].get("id")
        except json.JSONDecodeError:
            return None
    if path.startswith("/healthz"):
        return "healthz"
    if path.startswith("/system_stats"):
        return compat.VIDEO_HTTP_LABEL
    return body[:40].strip() or "ok"


def tenant_floor(
    catalog: dict[str, Any],
    *,
    rss_fn=None,
    docker_fn=None,
    listen_fn=None,
) -> dict[str, Any]:
    """Measure tenant floor.

    Always-tenants (e.g. os/docker): measure fail OR zero matches ⇒
    incomplete (not a confident 0.0). Optional tenants (the video service): absent is
    0.0 and complete.
    """
    rss_fn = rss_fn or rss_match_gib
    docker_fn = docker_fn or docker_stats_gib
    listen_fn = listen_fn or port_listening
    tenants = catalog.get("tenants") or {}
    parts: list[dict[str, Any]] = []
    complete = True
    reasons: list[str] = []
    total = 0.0

    for name, spec in tenants.items():
        always = bool(spec.get("always", True))
        measure = spec.get("measure")
        row: dict[str, Any] = {
            "name": name,
            "always": always,
            "measure": measure,
            "gib": None,
            "unit": None,
            "instrument": None,
            "present": False,
            "confidence": "uncertain",
            "detail": None,
        }

        if not always and measure == "rss" and spec.get("port"):
            host = spec.get("listen_host") or "127.0.0.1"
            port = int(spec.get("port"))
            if not listen_fn(port, host):
                row["present"] = False
                row["gib"] = 0.0
                row["unit"] = "rss_gib"
                row["instrument"] = "rss"
                row["confidence"] = "absent"
                row["detail"] = "not listening; excluded from floor (0.0 complete)"
                parts.append(row)
                continue
            row["present"] = True
            gib, hits, err = rss_fn(cmdline_substr=list(spec.get("cmdline_substr") or protect.tenant_cmdline()))
            if gib is None or err or not hits:
                complete = False
                reasons.append(f"{name} up but rss unclassified ({err or 'zero procs'})")
                row["confidence"] = "uncertain"
                row["unit"] = "rss_gib"
                row["instrument"] = "rss"
                row["detail"] = err or "rss failed"
            else:
                row["gib"] = gib
                row["unit"] = "rss_gib"
                row["instrument"] = "rss"
                row["confidence"] = "measured"
                row["detail"] = f"{len(hits)} procs RSS"
                total += gib
            parts.append(row)
            continue

        if measure == "catalog_min":
            min_g = spec.get("min_gib")
            if min_g is None:
                complete = False
                reasons.append(f"{name} catalog_min missing")
                row["detail"] = "no min_gib"
                row["unit"] = "catalog_min_gib"
                row["instrument"] = "catalog_min"
            else:
                row["gib"] = float(min_g)
                row["present"] = True
                row["confidence"] = "catalog_min"
                row["unit"] = "catalog_min_gib"
                row["instrument"] = "catalog_min"
                row["detail"] = "declared OS floor, not measured"
                total += float(min_g)
            parts.append(row)
            continue

        if measure == "rss":
            gib, hits, err = rss_fn(
                comms=list(spec.get("comm") or []),
                cmdline_substr=list(spec.get("cmdline_substr") or []),
                user=spec.get("user"),
            )
            row["unit"] = "rss_gib"
            row["instrument"] = "rss"
            if gib is None or err:
                complete = False
                reasons.append(f"{name} rss failed ({err})")
                row["detail"] = err
            elif always and not hits:
                complete = False
                reasons.append(f"{name} always-tenant has zero matching procs")
                row["gib"] = None
                row["present"] = False
                row["confidence"] = "uncertain"
                row["detail"] = "no matching procs (not 0.0)"
            else:
                row["gib"] = gib
                row["present"] = bool(hits)
                row["confidence"] = "measured"
                row["detail"] = f"{len(hits)} procs RSS"
                total += gib
            parts.append(row)
            continue

        if measure == "docker_stats":
            substr = (spec.get("name_substr") or ["buzz"])[0]
            gib, matched, err = docker_fn(substr)
            row["unit"] = "rss_gib"
            row["instrument"] = "docker_stats"
            if gib is None or err:
                complete = False
                reasons.append(f"{name} docker stats failed ({err})")
                row["detail"] = err
            elif always and not matched:
                complete = False
                reasons.append(f"{name} always-tenant has zero matching containers")
                row["gib"] = None
                row["present"] = False
                row["confidence"] = "uncertain"
                row["detail"] = "no matching containers (not 0.0)"
            else:
                row["gib"] = gib
                row["present"] = bool(matched)
                row["confidence"] = "measured"
                row["detail"] = f"{len(matched)} containers RSS"
                total += gib
            parts.append(row)
            continue

        complete = False
        reasons.append(f"{name} unknown measure {measure!r}")
        row["detail"] = f"unknown measure {measure}"
        parts.append(row)

    return {
        "complete": complete,
        "total_gib": round(total, 3) if complete else None,
        "partial_total_gib": round(total, 3),
        "parts": parts,
        "reasons": reasons,
        "units_note": "floor is RSS/catalog_min; recipe is declared VRAM — same UMA pool only",
        "note": "MemAvailable is not a floor and cannot produce OK",
    }


def live_residents(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    apps, apps_err = compute_apps()
    apps = apps or []
    out: list[dict[str, Any]] = []
    for row in catalog.get("residents") or []:
        host = row.get("listen_host") or "127.0.0.1"
        port = row.get("port")
        path = row.get("probe") or "/v1/models"
        http = (
            probe_http_identity(host, int(port), path)
            if port
            else {"listening": False, "http_ok": False, "identity": None}
        )
        gpu = [
            a
            for a in apps
            if (row.get("served_name") and a.get("served_model") == row.get("served_name"))
            or (port and a.get("port") == port)
        ]
        unit = None
        if row.get("unit"):
            unit = systemctl_query(row["unit"])
        listening = bool(http.get("listening"))
        http_ok = bool(http.get("http_ok"))
        gpu_hit = bool(gpu)
        # Resident only if live listen+identity OR matching compute app.
        resident = (listening and http_ok) or gpu_hit
        identity = http.get("identity")
        if resident and row.get("served_name") and http_ok and identity:
            # identity mismatch is still resident (something is there) but flagged
            pass
        seats = catalog.get("seats") or {}
        seat = None
        rid = row.get("id")
        for sname, sid in seats.items():
            if sid == rid:
                seat = sname
                break
        if seat is None and row.get("role") in ("brain", "vision", "counsel", "video"):
            seat = row.get("role")
        out.append(
            {
                "id": rid,
                "kind": row.get("kind"),
                "role": row.get("role"),
                "seat": seat,
                "served_name": row.get("served_name"),
                "port": port,
                "resident": resident,
                "listening": listening,
                "http_ok": http_ok,
                "identity": identity,
                "gpu_apps": gpu,
                "gpu_gib": round(sum(a["used_gpu_mem_gib"] or 0 for a in gpu), 3) if gpu else None,
                "unit": unit,
                "smi_error": apps_err,
            }
        )
    return out


def unit_table(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    names = list(catalog.get("units_status") or [])
    return [systemctl_query(u) for u in names]


def conflicts(
    catalog: dict[str, Any],
    residents: list[dict[str, Any]],
    units: list[dict[str, Any]],
    truth: dict[str, Any] | None,
) -> list[str]:
    found: list[str] = []
    by_id = {r["id"]: r for r in residents}
    trio_ids = core_ids(catalog)
    trio_up = any(by_id.get(i, {}).get("resident") for i in trio_ids)
    cores = next((u for u in units if u["unit"].startswith("vllm-coresident")), None)
    cores_active = bool(cores and cores.get("is_active") == "active")
    video = next((by_id[i] for i in sorted(protect.video_ids()) if i in by_id), {})
    if (trio_up or cores_active) and video.get("resident"):
        found.append("LLM stack and the video service both up (exclusive modes)")

    mode = None
    if truth:
        mode = (truth.get("stack") or {}).get("mode")
    trio_http = any(by_id.get(i, {}).get("http_ok") for i in trio_ids)
    if mode == "always_on" and not cores_active and not trio_http:
        found.append("model inventory file says always_on but vllm-coresident inactive and no LLM HTTP")

    for r in residents:
        unit = r.get("unit") or {}
        # http_ok is False only after a probe. None means this pass did not ask.
        if unit.get("is_active") == "active" and r.get("port") and r.get("http_ok") is False and not r.get("resident"):
            found.append(f"{r['id']} unit active but HTTP identity missing")
        elif unit.get("is_active") == "active" and r.get("listening") and r.get("http_ok") is False:
            found.append(f"{r['id']} unit active, port listening, HTTP identity missing")

    ports: dict[int, list[str]] = {}
    for r in residents:
        if r.get("resident") and r.get("port"):
            ports.setdefault(int(r["port"]), []).append(r["id"])
    for port, ids in ports.items():
        if len(ids) > 1:
            found.append(f"port {port} claimed by {', '.join(ids)}")

    return found


def snapshot(catalog: dict[str, Any], truth: dict[str, Any] | None = None, docker_fn=None) -> dict[str, Any]:
    mem = meminfo_gib()
    residents = live_residents(catalog)
    units = unit_table(catalog)
    floor = tenant_floor(catalog, docker_fn=docker_fn)
    return {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": catalog.get("host"),
        "mem": mem,
        "floor": floor,
        "residents": residents,
        "units": units,
        "conflicts": conflicts(catalog, residents, units, truth),
    }
