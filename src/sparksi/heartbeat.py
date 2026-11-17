# SPDX-License-Identifier: Apache-2.0

"""Probe beats: what a Spark announces about itself, used instead of an SSH probe while fresh.

Each Spark's announcer sends a second datagram per beat (``"ph"``) next to its
stats packet. It reaches this module only through the announce receiver's
source-address allowlist and size limit. :func:`parse` keeps the known fields
with bounded sizes. :func:`answer_probe` answers an agent ``probe`` request in
the agent's own shape, or returns None so the caller asks over SSH: for an
older announcer, a stale beat, or anything a beat does not carry (paths,
archive scans, other HTTP paths). Actions never use this module.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable

from sparksi import protect

MAX_AGE_S = 8.0
MAX_CT = 16  # the announcer lists at most this many containers
MAX_LISTEN = 32  # the announcer sends at most this many listening ports
LISTEN_MIN = 1024  # and none below this
BASE_TTL = 300.0
_NAME_OK = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
_CODE_OK = frozenset("0123456789abcdefABCDEF.:")
_HTTP_PATHS = ("/health", "/v1/models")
_SUPPORTED = frozenset({"op", "ports", "paths", "listen_host", "http_paths", "container_ports", "unmanaged_ports"})


def _num(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _num_or_none(value: object) -> bool:
    return value is None or _num(value)


def _port(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 65535


def _name(value: object) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 64 and all(ch in _NAME_OK for ch in value)


def _code(value: object) -> bool:
    return value in ("w", "l") or (
        isinstance(value, str) and 1 <= len(value) <= 45 and all(ch in _CODE_OK for ch in value)
    )


def _models(value: object) -> list[str] | None:
    if not isinstance(value, list) or len(value) > 4:
        return None
    if not all(isinstance(m, str) and 1 <= len(m) <= 64 for m in value):
        return None
    return list(value)


def _status(value: object) -> bool:
    return value is None or (isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599)


def parse(pkt: object) -> dict[str, Any] | None:
    """A cleaned beat, or None. Bad rows are dropped; a bad envelope drops the beat."""
    if not isinstance(pkt, dict) or pkt.get("v") != 1 or isinstance(pkt.get("v"), bool):
        return None
    host = pkt.get("ph")
    if not isinstance(host, str) or not 1 <= len(host) <= 64:
        return None
    out: dict[str, Any] = {"ph": host, "v": 1}
    hn = pkt.get("hn")
    out["hn"] = hn if isinstance(hn, str) and len(hn) <= 64 else ""
    mm = pkt.get("mm")
    if isinstance(mm, list) and len(mm) == 2 and all(_num(x) for x in mm):
        out["mm"] = [float(mm[0]), float(mm[1])]
    dk = pkt.get("dk")
    if isinstance(dk, list) and len(dk) == 3 and all(_num(x) for x in dk):
        out["dk"] = [float(x) for x in dk]
    g = pkt.get("g")
    out["g"] = (
        list(g) if isinstance(g, list) and len(g) == 3 and all(_num_or_none(x) for x in g) else [None, None, None]
    )
    ls = pkt.get("ls")
    if isinstance(ls, list):
        out["ls"] = [
            list(r) for r in ls[:MAX_LISTEN] if isinstance(r, list) and len(r) == 2 and _port(r[0]) and _code(r[1])
        ]
    ct = pkt.get("ct", "missing")
    if ct is None:
        out["ct"] = None
    elif isinstance(ct, list):
        rows = []
        for row in ct[:MAX_CT]:
            if isinstance(row, list) and len(row) == 2 and _name(row[0]) and isinstance(row[1], list):
                rows.append([row[0], [p for p in row[1][:4] if _port(p)]])
        out["ct"] = rows
        cn = pkt.get("cn")
        if isinstance(cn, int) and not isinstance(cn, bool) and 0 <= cn <= 100000:
            out["cf"] = cn <= len(rows)
        else:  # an older beat: a full list may have been cut
            out["cf"] = len(ct) < MAX_CT
    hi = pkt.get("hi")
    if isinstance(hi, dict):
        clean: dict[str, list] = {}
        for key, val in list(hi.items())[:6]:
            if not (isinstance(key, str) and key.isdigit() and _port(int(key))):
                continue
            if not (isinstance(val, list) and len(val) == 4 and _status(val[0]) and _status(val[2])):
                continue
            hm, vm = _models(val[1]), _models(val[3])
            if hm is None or vm is None:
                continue
            clean[str(int(key))] = [val[0], hm, val[2], vm]
        out["hi"] = clean
    cs = pkt.get("cs")
    if isinstance(cs, dict):
        out["cs"] = {k: float(v) for k, v in list(cs.items())[:8] if _name(k) and _num(v) and v >= 0}
    return out


class _Store:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self.clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._rows: dict[str, tuple[dict[str, Any], float]] = {}

    def note(self, host: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._rows[host] = (value, self.clock())

    def get(self, host: str, max_age: float) -> dict[str, Any] | None:
        with self._lock:
            row = self._rows.get(host)
        if row is None or self.clock() - row[1] > max_age:
            return None
        return row[0]

    def forget(self) -> None:
        with self._lock:
            self._rows.clear()


BEATS = _Store()  # host -> latest parsed beat
BASES = _Store()  # host -> latest SSH probe answer (archive mount, addresses: slow facts a beat does not carry)


def _address(listen_host: object) -> str:
    """The agent's rule: a tailnet IPv4 from the request, else loopback."""
    parts = str(listen_host or "").strip().split(".")
    if len(parts) == 4 and all(p.isdigit() for p in parts):
        nums = [int(p) for p in parts]
        if all(n <= 255 for n in nums) and nums[0] == 100 and 64 <= nums[1] <= 127:
            return ".".join(str(n) for n in nums)
    return "127.0.0.1"


def _listening(beat: dict[str, Any], port: int, address: str) -> bool:
    for row_port, code in beat.get("ls") or []:
        if row_port == port and (code == "w" or code == address or (code == "l" and address == "127.0.0.1")):
            return True
    return False


def _container_ports(beat: dict[str, Any], address: str, keep: Callable[[str], bool]) -> dict[str, int]:
    found: dict[str, int] = {}
    for name, declared in beat.get("ct") or []:
        if not keep(name):
            continue
        usable = [p for p in declared if not protect.is_off_limits(p)]
        live = [p for p in usable if _listening(beat, p, address)]
        if live or usable:
            found[name] = (live or usable)[0]
    return found


def _ours(name: str) -> bool:
    return name.startswith("sparksi-")


def _outside(name: str) -> bool:
    low = name.lower()
    return (
        not low.startswith("sparksi-")
        and not name.startswith("-")
        and not protect.is_video_process(low)
        and "announce" not in low
    )


def answer_probe(beat: dict[str, Any], req: dict[str, Any], base: dict[str, Any]) -> dict[str, Any] | None:
    """The agent's probe answer from a beat, or None when the beat cannot say."""
    if not isinstance(req, dict) or req.get("op") != "probe":
        return None
    if any(value for key, value in req.items() if key not in _SUPPORTED):
        return None
    if req.get("paths") or "mm" not in beat or "ls" not in beat:
        return None
    try:
        ports = [int(p) for p in (req.get("ports") or [])]
    except (TypeError, ValueError):
        return None
    if any(p < LISTEN_MIN or p > 65535 for p in ports):
        return None  # the beat lists 1024-65535 only
    if len(beat["ls"]) >= MAX_LISTEN and any(not _listening(beat, p, _address(req.get("listen_host"))) for p in ports):
        return None  # a full table may have dropped the port: ask over SSH
    http_paths = {str(k): str(v) for k, v in (req.get("http_paths") or {}).items()}
    address = _address(req.get("listen_host"))
    listening = {str(p): _listening(beat, p, address) for p in ports}
    http: dict[str, dict[str, Any]] = {}
    hi = beat.get("hi") or {}
    for port, up in listening.items():
        if not up:
            continue
        path = http_paths.get(port) or "/health"
        if path not in _HTTP_PATHS or port not in hi:
            return None
        row = hi[port]
        half = 0 if path == "/health" else 2
        identity = {"path": path, "status": row[half], "models": list(row[half + 1])}
        if identity["status"] != 200 and path != "/v1/models" and port not in http_paths:
            identity = {"path": "/v1/models", "status": row[2], "models": list(row[3])}
        http[port] = identity
    ct = beat.get("ct", None)
    if ct is not None and not beat.get("cf", False):
        ct = None  # a partial list: unknown, never "gone"
    if (req.get("container_ports") or req.get("unmanaged_ports")) and ct is None:
        return None
    total, avail = beat["mm"]
    disk = beat.get("dk")
    g = beat.get("g") or [None, None, None]
    out: dict[str, Any] = {
        "ok": True,
        "hostname": beat.get("hn") or base.get("hostname"),
        "mem_total_gib": total,
        "mem_available_gib": avail,
        "disk_total_gib": disk[0] if disk else base.get("disk_total_gib"),
        "disk_free_gib": disk[1] if disk else base.get("disk_free_gib"),
        "disk_used_gib": disk[2] if disk else base.get("disk_used_gib"),
        "disk_used_pct": (
            (round(100.0 * disk[2] / (disk[2] + disk[1]), 1) if disk[2] + disk[1] else None)
            if disk
            else base.get("disk_used_pct")
        ),
        "mount_archive": base.get("mount_archive"),
        "archive_store": base.get("archive_store"),
        "listening": listening,
        "http": http,
        "containers": None if ct is None else sorted(name for name, _ports in ct),
        "ips": base.get("ips"),
        "paths": {},
        "gpu": {
            "util_pct": g[0],
            "temp_c": g[1],
            "temp_f": None if g[1] is None else round(g[1] * 9 / 5 + 32, 1),
            "sm_mhz": g[2],
        },
        "via": "heartbeat",
    }
    if req.get("container_ports"):
        out["container_ports"] = _container_ports(beat, address, _ours)
    if req.get("unmanaged_ports"):
        out["unmanaged_containers"] = _container_ports(beat, address, _outside)
    return out


def node_key(host: object) -> str:
    """A node's id from its id, name or ssh alias (beats are filed by id)."""
    text = str(host or "")
    try:
        from sparksi.config import CFG

        for node in CFG.nodes:
            if text in (node.id, node.name, node.ssh):
                return node.id
    except Exception:  # noqa: BLE001 — no config: the text as given
        pass
    return text


def beat_answer(host: object, req: object, max_age: float = MAX_AGE_S) -> dict[str, Any] | None:
    """A probe answer from a fresh beat, or None (ask over SSH)."""
    if not isinstance(req, dict) or req.get("op") != "probe":
        return None
    key = node_key(host)
    beat = BEATS.get(key, max_age)
    base = BASES.get(key, BASE_TTL)
    if beat is None or base is None:
        return None
    return answer_probe(beat, req, base)


def note_reply(host: object, req: object, reply: object) -> None:
    """Keep an SSH probe answer: it carries the slow facts a beat does not (archive mount, addresses)."""
    if isinstance(req, dict) and req.get("op") == "probe" and isinstance(reply, dict):
        if reply.get("ok") is True and "via" not in reply:
            BASES.note(node_key(host), reply)


def healthy(answer: dict[str, Any], expect: set[str] | frozenset[str] = frozenset()) -> bool:
    """True when a beat answer says only good news: every asked port up, every identity 200, every expected container seen."""
    if not all((answer.get("listening") or {}).values()):
        return False
    if any(row.get("status") != 200 for row in (answer.get("http") or {}).values()):
        return False
    if expect:
        seen = answer.get("containers")
        if seen is None or not set(expect) <= set(seen):
            return False
    return True


class HeartbeatReads:
    """A runner whose routine probes come from a fresh beat; everything else goes to the real runner.

    ``confirm_bad=True`` is for health passes whose verdicts can stop or fail a run: a beat may only
    report good news there. Anything else (a port down, an identity not 200, an expected container
    missing) is read again over SSH, so a stop or a failure never rests on a beat alone.
    """

    def __init__(
        self,
        runner: Any,
        max_age: float = MAX_AGE_S,
        *,
        confirm_bad: bool = False,
        expect: dict[str, set[str]] | None = None,
    ) -> None:
        if runner is None:
            raise TypeError("HeartbeatReads needs a runner")
        self._runner = runner.unwrap() if isinstance(runner, HeartbeatReads) else runner
        self._max_age = max_age
        self._confirm_bad = confirm_bad
        self._expect = {node_key(k): set(v) for k, v in (expect or {}).items()}

    def unwrap(self) -> Any:
        return self._runner

    def invoke(self, host: str, req: dict[str, Any], timeout: int = 60) -> dict[str, Any]:
        answer = beat_answer(host, req, self._max_age)
        if answer is not None and (not self._confirm_bad or healthy(answer, self._expect.get(node_key(host), set()))):
            return answer
        reply = self._runner.invoke(host, req, timeout=timeout)
        note_reply(host, req, reply)
        return reply

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runner, name)


def container_seen(host: str, name: str, max_age: float = MAX_AGE_S) -> tuple[bool, float | None] | None:
    """(running, mem GiB) of a container from a fresh beat; None without one or when docker was not answered."""
    beat = BEATS.get(node_key(host), max_age)
    if beat is None or beat.get("ct", None) is None:
        return None
    running = any(row[0] == name for row in beat["ct"])
    if not running and not beat.get("cf", False):
        return None  # not in a partial list: unknown
    return running, (beat.get("cs") or {}).get(name)
