# SPDX-License-Identifier: Apache-2.0

"""Live per-node facts over the SSH runner.

Read-only commands only: ``free -b`` (or ``/proc/meminfo``), ``nvidia-smi``,
``ss -ltn``, ``docker ps``, plus ``systemctl show`` and ``test -e`` so a
catalog unit or weight path that is gone can be marked stale. Node names come
from sparksi.toml and the catalog host table. Nothing here is hard-coded to a
site. Results are cached for a few seconds so overview polling does not open
an SSH session per refresh. An off-limits port (8188 with the ``video-service``
preset) is only ever reported as a listening port; nothing connects to it.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from sparksi.config import CFG, role_name
from sparksi.remote import SshRunner

CACHE_TTL_S = 5.0
NODE_TIMEOUT_S = 4

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"mono": 0.0, "key": None, "data": None}


def invalidate_cache() -> None:
    """Drop the node-fact TTL so the next read talks to the Sparks again."""
    with _CACHE_LOCK:
        _CACHE["mono"] = 0.0
        _CACHE["key"] = None
        _CACHE["data"] = None


def reset_cache_for_tests() -> None:
    invalidate_cache()
    # Model-card facts live next to this cache. Import lazily: service imports nodes.
    from sparksi.recipes.service import reset_facts_cache

    reset_facts_cache()


def node_specs(catalog: dict[str, Any] | None = None) -> list[dict[str, str]]:
    """Head and worker identities the way the rest of the controller reads them.

    ``key`` is the catalog ``hosts`` entry the SSH runner already resolves.
    ``ssh`` is kept off the public payload; it only tells the runner where to
    go, and an empty catalog host falls back to the toml alias.
    """
    hosts = (catalog or {}).get("hosts") or {}
    specs: list[dict[str, str]] = []
    for index, node in enumerate(CFG.nodes):
        key = node.name if node.name in hosts or node.id not in hosts else node.id
        remote = hosts.get(key) or hosts.get(node.name) or hosts.get(node.id) or {}
        role = role_name(index)
        specs.append({
            "id": str(node.id),
            "name": str(node.name),
            "role": role,
            "key": str(key),
            "ssh": str(remote.get("ssh") or node.ssh or ""),
        })
    return specs


def parse_free_b(text: str) -> dict[str, float] | None:
    """``free -b`` Mem line: total, used, free, available, in GiB."""
    mem: dict[str, float] | None = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 or parts[0] not in ("Mem:", "Swap:"):
            continue

        def gib(index: int) -> float | None:
            try:
                return round(int(parts[index]) / 1024**3, 3)
            except (IndexError, ValueError):
                return None

        if parts[0] == "Mem:" and len(parts) >= 7:
            total, used, free, avail = gib(1), gib(2), gib(3), gib(6)
            if total is None or used is None or avail is None:
                continue
            mem = {
                "total_gib": total,
                "used_gib": used,
                "free_gib": free if free is not None else avail,
                "available_gib": avail,
            }
        elif parts[0] == "Swap:" and mem is not None and len(parts) >= 4:
            swap_total, swap_free = gib(1), gib(3)
            if swap_total is not None:
                mem["swap_total_gib"] = swap_total
            if swap_free is not None:
                mem["swap_free_gib"] = swap_free
    return mem


def parse_meminfo(text: str) -> dict[str, float] | None:
    """``/proc/meminfo`` fallback. Used memory is total minus available."""
    raw: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        bits = rest.split()
        if not bits:
            continue
        try:
            raw[key] = int(bits[0])
        except ValueError:
            continue
    if "MemTotal" not in raw or "MemAvailable" not in raw:
        return None

    def gib(kb: int) -> float:
        return round(kb / 1024**2, 3)

    total = gib(raw["MemTotal"])
    avail = gib(raw["MemAvailable"])
    out: dict[str, float] = {
        "total_gib": total,
        "available_gib": avail,
        "used_gib": round(max(0.0, total - avail), 3),
    }
    if "MemFree" in raw:
        out["free_gib"] = gib(raw["MemFree"])
    else:
        out["free_gib"] = avail
    if "SwapTotal" in raw:
        out["swap_total_gib"] = gib(raw["SwapTotal"])
    if "SwapFree" in raw:
        out["swap_free_gib"] = gib(raw["SwapFree"])
    return out


def parse_nvidia(text: str) -> dict[str, float] | None:
    """CSV from ``--query-gpu=utilization.gpu,temperature.gpu[,clocks...]``.

    Extra trailing fields are optional: current graphics MHz, then max MHz.
    """
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("utilization"):
            continue
        parts = [part.strip() for part in line.split(",")]

        def num(index: int) -> float | None:
            try:
                value = parts[index]
            except IndexError:
                return None
            if value in ("", "[N/A]", "N/A"):
                return None
            try:
                return float(value)
            except ValueError:
                return None

        util, temp = num(0), num(1)
        clock, clock_max = num(2), num(3)
        if util is None and temp is None and clock is None and clock_max is None:
            return None
        out: dict[str, float] = {}
        if util is not None:
            out["util_pct"] = util
        if temp is not None:
            out["temp_c"] = temp
        if clock is not None:
            out["clock_mhz"] = clock
        if clock_max is not None:
            out["clock_max_mhz"] = clock_max
        return out
    return None


def _listen_from_local(local: str) -> tuple[str, int] | None:
    """``address, port`` from an ``ss`` local column. The address is kept."""
    text = local.strip()
    if not text or text.endswith(":*"):
        return None
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        return None
    try:
        value = int(port)
    except ValueError:
        return None
    if value < 1 or value > 65535:
        return None
    return host, value


def parse_ss_ltn(text: str) -> list[dict[str, Any]]:
    """Listening TCP sockets from ``ss -ltn``.

    Each item is ``{"address", "port"}``. The same port may appear more than
    once when it is bound on more than one address. Off-limits ports are kept when present.
    """
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        if parts[0].lower() in ("state", "netid"):
            continue
        if "LISTEN" not in {part.upper() for part in parts[:3]}:
            continue
        # Local address is the column before the peer, with or without Netid.
        parsed = _listen_from_local(parts[-2])
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        found.append({"address": parsed[0], "port": parsed[1]})
    return found


def listen_ports(rows: list[dict[str, Any]]) -> list[int]:
    """Port numbers from ``parse_ss_ltn``, one each, in first-seen order."""
    found: list[int] = []
    for row in rows:
        try:
            port = int(row.get("port"))
        except (TypeError, ValueError, AttributeError):
            continue
        if port not in found:
            found.append(port)
    return found


def parse_docker_ps(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        bits = line.split("\t")
        name = bits[0].strip() if bits else ""
        if not name or name.upper() == "NAMES":
            continue
        image = bits[1].strip() if len(bits) > 1 else ""
        ports = bits[2].strip() if len(bits) > 2 else ""
        rows.append({"name": name, "image": image, "ports": ports})
    return rows


def parse_systemctl_show(text: str) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    current: dict[str, str] = {}

    def flush() -> None:
        name = current.get("Id")
        if not name:
            return
        out[name] = {
            "load_state": current.get("LoadState") or "unknown",
            "active_state": current.get("ActiveState") or "unknown",
        }

    for line in text.splitlines():
        if not line.strip():
            flush()
            current = {}
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key == "Id" and current.get("Id"):
            flush()
            current = {}
        current[key] = value
    flush()
    return out


def _watch_lists(catalog: dict[str, Any]) -> tuple[list[str], list[str]]:
    units: list[str] = []
    paths: list[str] = []
    for name in catalog.get("units_status") or []:
        if name:
            units.append(str(name))
    for row in catalog.get("residents") or []:
        if not isinstance(row, dict):
            continue
        unit = row.get("unit")
        if unit:
            units.append(str(unit))
        weights = row.get("weights")
        if weights and str(weights).startswith("/"):
            paths.append(str(weights))
    return _uniq(units), _uniq(paths)


def _uniq(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _capture(runner: Any, host: str, argv: list[str], timeout: int) -> dict[str, Any]:
    fn = getattr(runner, "capture", None)
    if fn is not None:
        out = fn(host, argv, timeout)
    else:
        out = runner.run_argv(host, argv, timeout=timeout)
    if not isinstance(out, dict):
        return {"ok": False, "error": "runner returned a non-object", "stdout": "", "stderr": "", "code": None}
    if "stdout" not in out:
        out = dict(out)
        out["stdout"] = out.get("tail") or ""
    out.setdefault("stdout", "")
    out.setdefault("stderr", "")
    out.setdefault("error", None)
    return out


def _remote_failed(result: dict[str, Any]) -> bool:
    if result.get("ok"):
        return False
    if (result.get("stdout") or "").strip():
        return False
    code = result.get("code")
    err = str(result.get("error") or result.get("stderr") or "").lower()
    if code in (255, None) and err:
        return True
    markers = ("timed out", "timeout", "unreachable", "could not resolve", "connection refused", "no route")
    return any(marker in err for marker in markers)


def _unreachable(spec: dict[str, str], error: str) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "name": spec["name"],
        "role": spec["role"],
        "reachable": False,
        "error": (error or "unreachable").strip()[:400],
        "mem": None,
        "gpu": None,
        "ports": [],
        "containers": [],
        "units": {},
        "paths": {},
    }


def _blank_reachable(spec: dict[str, str]) -> dict[str, Any]:
    card = _unreachable(spec, "")
    card["reachable"] = True
    card["error"] = None
    return card


def _fetch_one(catalog: dict[str, Any], runner: Any, spec: dict[str, str], timeout: int) -> dict[str, Any]:
    host = spec["key"]
    deadline = time.monotonic() + max(1, int(timeout))

    def call(argv: list[str]) -> dict[str, Any]:
        left = max(1, int(deadline - time.monotonic() + 0.999))
        return _capture(runner, host, argv, left)

    try:
        return _read_node(catalog, spec, call)
    except Exception as exc:  # noqa: BLE001 — one node must not sink the fleet
        return _unreachable(spec, str(exc))


def _read_node(catalog: dict[str, Any], spec: dict[str, str], call: Any) -> dict[str, Any]:
    free = call(["free", "-b"])
    mem = parse_free_b(free.get("stdout") or "") if (free.get("stdout") or "").strip() else None
    if mem is None:
        info = call(["cat", "/proc/meminfo"])
        if (info.get("stdout") or "").strip():
            mem = parse_meminfo(info.get("stdout") or "")
        elif _remote_failed(free) or _remote_failed(info):
            return _unreachable(spec, str(free.get("error") or info.get("error") or "unreachable"))
    if mem is None and _remote_failed(free):
        return _unreachable(spec, str(free.get("error") or "unreachable"))
    card = _blank_reachable(spec)
    card["mem"] = mem
    gpu_raw = call([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.max.graphics",
        "--format=csv,noheader,nounits",
    ])
    gpu = parse_nvidia(gpu_raw.get("stdout") or "") or {}
    clock_raw = call(["nvidia-smi", "-q", "--display=CLOCK"])
    if (clock_raw.get("stdout") or "").strip():
        from sparksi.gpu_clocks import parse_clock_report

        for key, value in parse_clock_report(clock_raw.get("stdout") or "").items():
            if value is not None:
                gpu[key] = value
    if gpu:
        from sparksi.gpu_clocks import with_memo

        gpu = with_memo(spec["key"], gpu)
    card["gpu"] = gpu or None
    ports_raw = call(["ss", "-ltn"])
    listeners = parse_ss_ltn(ports_raw.get("stdout") or "")
    card["listeners"] = listeners
    card["ports"] = listen_ports(listeners)
    docker_raw = call(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}\t{{.Ports}}"])
    if (docker_raw.get("stdout") or "").strip() or docker_raw.get("ok"):
        card["containers"] = parse_docker_ps(docker_raw.get("stdout") or "")
    units, paths = _watch_lists(catalog)
    if units:
        shown = call(["systemctl", "show", *units, "-p", "Id", "-p", "LoadState", "-p", "ActiveState"])
        card["units"] = parse_systemctl_show(shown.get("stdout") or "")
    path_map: dict[str, bool] = {}
    for path in paths:
        result = call(["test", "-e", path])
        if _remote_failed(result):
            continue
        code = result.get("code")
        if result.get("ok") or code == 0:
            path_map[path] = True
        elif code == 1:
            path_map[path] = False
    card["paths"] = path_map
    return card


def fetch_nodes(
    catalog: dict[str, Any],
    runner: Any,
    specs: list[dict[str, str]] | None = None,
    *,
    timeout: int = NODE_TIMEOUT_S,
) -> list[dict[str, Any]]:
    """Read every node at once. One slow Spark does not block the other."""
    specs = list(specs if specs is not None else node_specs(catalog))
    if not specs:
        return []
    workers = max(1, len(specs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one, catalog, runner, spec, timeout) for spec in specs]
        return [future.result() for future in futures]


def node_report(
    catalog: dict[str, Any],
    runner: Any = None,
    *,
    specs: list[dict[str, str]] | None = None,
    timeout: int = NODE_TIMEOUT_S,
    ttl: float = CACHE_TTL_S,
    now: Callable[[], float] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Cached ``GET /api/nodes`` body. ``cached`` is true on a TTL hit."""
    specs = list(specs if specs is not None else node_specs(catalog))
    clock = now or time.monotonic
    key = tuple((item["id"], item["name"], item["key"]) for item in specs)
    stamp = clock()
    with _CACHE_LOCK:
        cached = _CACHE["data"]
        if (
            not force
            and cached is not None
            and _CACHE["key"] == key
            and stamp - float(_CACHE["mono"]) < ttl
        ):
            data = dict(cached)
            data["cached"] = True
            data["nodes"] = list(cached.get("nodes") or [])
            return data
    live = fetch_nodes(catalog, runner or SshRunner(catalog), specs, timeout=timeout)
    data = {"nodes": live, "cached": False}
    with _CACHE_LOCK:
        _CACHE["mono"] = stamp
        _CACHE["key"] = key
        _CACHE["data"] = {"nodes": live, "cached": False}
    return data
