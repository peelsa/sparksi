# SPDX-License-Identifier: Apache-2.0

"""Read-only polling of the monitored inference hosts (``[[endpoints]]`` in sparksi.toml).

The Network page saves up to two monitored hosts beside the head and the worker.
This module polls each one in the background with safe GETs only (never POST,
never a prompt) and keeps the last answer for the Overview cards and
``GET /api/monitor`` (field ``monitored``).

Every request:

* goes to a URL rebuilt here as ``http://<checked host>:<checked port><path>``,
  with a path from the engine's fixed list below;
* resolves the host at poll time and refuses it unless every address it
  resolves to passes the same rules as ``hostcheck`` (private LAN or tailnet;
  never loopback, link-local/cloud metadata, multicast, reserved or "this
  network"), then connects to that checked address only, so a later lookup
  cannot point the request somewhere else;
* uses the mirror's opener (no proxy, no redirects), a short timeout and the
  mirror's reply size cap.
"""

from __future__ import annotations

import datetime
import functools
import http.client
import ipaddress
import json
import math
import re
import socket
import threading
import time
import unicodedata
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, Request

from sparksi import monitor_mirror
from sparksi.hostcheck import check_host, check_port, ip_ok

TIMEOUT_S = 2.0  # each socket step
TOTAL_S = 5.0  # the whole GET: lookup, connect, headers and body
MAX_BYTES = monitor_mirror.MAX_BYTES

# Safe GETs per engine. The first path must answer; the others are read when the engine has them.
PATHS: dict[str, tuple[str, ...]] = {
    "OpenAI-compatible": ("/v1/models",),
    "vLLM": ("/v1/models", "/health", "/metrics"),
    "SGLang": ("/v1/models", "/health", "/metrics"),
    "llama.cpp": ("/v1/models", "/health", "/metrics"),
    "Ollama": ("/api/tags", "/api/ps"),
    "": ("/v1/models",),
}
_ALLOWED_PATHS = frozenset(p for paths in PATHS.values() for p in paths)


class Refused(ValueError):
    """The endpoint or the address it resolves to is not one Sparksi may poll."""


def base_url(url: str) -> tuple[str, str, int]:
    """(``http://host:port``, host, port) for a saved endpoint URL, checked like the saved file."""
    parts = urlsplit(str(url or ""))
    if parts.scheme != "http" or parts.username or parts.password or parts.path not in ("", "/") or parts.query or parts.fragment:
        raise Refused("the endpoint must be http://host:port")
    try:
        host = check_host(parts.hostname or "")
        port = check_port(parts.port)
    except ValueError as exc:
        raise Refused(str(exc)) from None
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}", host, port


def _resolve_within(resolve: Callable[..., Any], host: str, port: int, seconds: float) -> Any:
    """getaddrinfo has no timeout of its own: run it beside the poll and stop waiting at the deadline."""
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["infos"] = resolve(host, port, 0, socket.SOCK_STREAM)
        except BaseException as exc:  # noqa: BLE001 - handed back to the poll below
            box["error"] = exc

    worker = threading.Thread(target=work, name="sparksi-monitored-lookup", daemon=True)
    worker.start()
    worker.join(max(0.0, seconds))
    if worker.is_alive():
        raise TimeoutError(f"could not resolve {host} in time")
    error = box.get("error")
    if error is not None:
        if isinstance(error, (OSError, UnicodeError)):
            raise URLError(f"could not resolve {host}") from error
        raise error
    return box.get("infos")


def resolve_checked(
    host: str,
    port: int,
    resolve: Callable[..., Any] = socket.getaddrinfo,
    seconds: float = TOTAL_S,
) -> str:
    """The address to connect to; Refused unless every address ``host`` resolves to is allowed now.

    A name that passed the save-time check can resolve anywhere later, so the
    check runs on what it resolves to at poll time. The lookup gives up after
    ``seconds``.
    """
    infos = _resolve_within(resolve, host, port, seconds)
    addresses: list[str] = []
    for info in infos or []:
        try:
            text = str(info[4][0])
        except (IndexError, TypeError):
            continue
        if text not in addresses:
            addresses.append(text)
    if not addresses:
        raise URLError(f"could not resolve {host}")
    for text in addresses:
        try:
            ipaddress.ip_address(text)
        except ValueError:
            raise Refused(f"{host} resolves to {text}, which is not an IP address") from None
        if not ip_ok(text):
            raise Refused(f"{host} resolves to {text}, which is not a private or tailnet address")
    return addresses[0]


class _Deadline:
    """One wall-clock limit for a whole GET: lookup, connect, headers and body.

    The socket timeout only bounds each read, so a host that sends one byte a
    second would never time out. At the deadline a timer shuts the socket down,
    which ends any read that is waiting, and every later step checks ``expired``.
    """

    def __init__(self, seconds: float) -> None:
        self.at = time.monotonic() + seconds
        self.expired = False
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._timer = threading.Timer(max(0.0, seconds), self.fire)
        self._timer.daemon = True
        self._timer.start()

    def remaining(self) -> float:
        return self.at - time.monotonic()

    def attach(self, sock: socket.socket) -> None:
        with self._lock:
            self._sock = sock
            late = self.expired
        if late:
            self._shut(sock)

    def fire(self) -> None:
        with self._lock:
            self.expired = True
            sock = self._sock
        if sock is not None:
            self._shut(sock)

    @staticmethod
    def _shut(sock: socket.socket) -> None:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    def check(self) -> None:
        if self.expired or self.remaining() <= 0:
            self.expired = True
            raise TimeoutError(f"no full answer within {TOTAL_S:g} s")

    def cancel(self) -> None:
        self._timer.cancel()


class _PinnedConnection(http.client.HTTPConnection):
    """Connects to the address checked a moment ago, never to a fresh lookup of the name."""

    def __init__(self, *args: Any, address: str, deadline: _Deadline | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._address = address
        self._deadline = deadline

    def connect(self) -> None:
        timeout = self.timeout
        if self._deadline is not None:
            self._deadline.check()
            timeout = max(0.05, min(float(timeout), self._deadline.remaining()))
        self.sock = socket.create_connection((self._address, self.port), timeout)
        if self._deadline is not None:
            self._deadline.attach(self.sock)
            self.sock.settimeout(self.timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


class _PinnedHandler(HTTPHandler):
    def __init__(self, address: str, deadline: _Deadline | None = None) -> None:
        super().__init__()
        self._address = address
        self._deadline = deadline

    def http_open(self, req):  # noqa: ANN001 - urllib handler signature
        return self.do_open(functools.partial(_PinnedConnection, address=self._address, deadline=self._deadline), req)


def get(
    url: str,
    path: str,
    *,
    timeout: float = TIMEOUT_S,
    total: float | None = None,
    resolve: Callable[..., Any] = socket.getaddrinfo,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    """One safe GET of ``path`` on a monitored endpoint; the reply body (at most MAX_BYTES).

    ``timeout`` bounds each socket step; ``total`` (default TOTAL_S) bounds the
    whole GET, lookup included. Raises Refused for a URL, path or resolved
    address that is not allowed, URLError/OSError when the host does not answer
    in time (HTTPError carries the status), ValueError for a reply that is too large.
    """
    if path not in _ALLOWED_PATHS:
        raise Refused("that path is not polled")
    base, host, port = base_url(url)
    limit = TOTAL_S if total is None else float(total)
    started = time.monotonic()
    address = resolve_checked(host, port, resolve, seconds=limit)
    deadline = _Deadline(limit - (time.monotonic() - started))
    request = Request(base + path, method="GET", headers={"Accept": "application/json, text/plain"})
    make = opener or monitor_mirror.no_redirect_opener
    chunks: list[bytes] = []
    size = 0
    try:
        with make(_PinnedHandler(address, deadline)).open(request, timeout=timeout) as response:
            deadline.check()
            if urlsplit(response.geturl()).path != path:
                raise URLError("unexpected path")
            while True:
                chunk = response.read1(65536)
                deadline.check()
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_BYTES:
                    raise ValueError("reply too large")
                chunks.append(chunk)
        deadline.check()  # a body cut short by the deadline is not an answer
    except (OSError, http.client.HTTPException) as exc:
        if deadline.expired:
            raise TimeoutError(f"no full answer within {limit:g} s") from exc
        if isinstance(exc, http.client.HTTPException):
            raise URLError(f"bad reply: {type(exc).__name__}") from exc
        raise
    finally:
        deadline.cancel()
    return b"".join(chunks)


MAX_MODELS = 50


def _clean(text, limit: int = 200) -> str:
    value = str(text or "")
    cleaned = "".join(
        ch
        for ch in value
        if unicodedata.category(ch) not in ("Cc", "Cf", "Zl", "Zp")
    )
    return cleaned.strip()[:limit]


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _gib(value) -> float | None:
    number = _num(value)
    if number is None:
        return None
    return round(number / 1024**3, 1)


def parse_openai_models(body: bytes) -> list[str]:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, TypeError):
        raise ValueError("reply is not valid JSON") from None
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str):
            continue
        name = _clean(identifier)
        if name:
            names.append(name)
    return _dedupe(names)[:MAX_MODELS]


def parse_ollama_tags(body: bytes) -> list[str]:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, TypeError):
        raise ValueError("reply is not valid JSON") from None
    items = data.get("models") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    names: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("name")
        if not isinstance(raw, str):
            raw = item.get("model")
        if not isinstance(raw, str):
            continue
        name = _clean(raw)
        if name:
            names.append(name)
    return _dedupe(names)[:MAX_MODELS]


def parse_ollama_ps(body: bytes) -> list[dict]:
    try:
        data = json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, TypeError):
        raise ValueError("reply is not valid JSON") from None
    items = data.get("models") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = item.get("name")
        if not isinstance(raw, str):
            raw = item.get("model")
        name = _clean(raw) if isinstance(raw, str) else ""
        out.append({"name": name, "size_gib": _gib(item.get("size")), "vram_gib": _gib(item.get("size_vram"))})
    return out


_METRIC_KEYS = {
    "vLLM": {
        "vllm:kv_cache_usage_perc": ("kv_cache_pct", True),  # current vLLM
        "vllm:gpu_cache_usage_perc": ("kv_cache_pct", True),  # older vLLM
        "vllm:num_requests_running": ("running", False),
    },
    "SGLang": {
        "sglang:token_usage": ("kv_cache_pct", True),
        "sglang:num_running_reqs": ("running", False),
    },
    "llama.cpp": {
        "llamacpp:kv_cache_usage_ratio": ("kv_cache_pct", True),
        "llamacpp:requests_processing": ("running", False),
    },
}


# name{labels} value [timestamp]: the name exactly, any labels, then the value.
_METRIC_LINE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?:[^"}]|"(?:[^"\\]|\\.)*")*\})?\s+(\S+)')


def parse_metrics(text_bytes: bytes, engine: str) -> dict:
    wanted = _METRIC_KEYS.get(engine)
    if not wanted:
        return {}
    try:
        text = text_bytes.decode("utf-8", errors="replace")
    except (UnicodeError, AttributeError):
        return {}
    out: dict = {}
    fractions: dict[str, list[float]] = {}
    counters: dict[str, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        hit = _METRIC_LINE.match(line)
        if hit is None or hit.group(1) not in wanted:
            continue
        name = hit.group(1)
        try:
            value = float(hit.group(2))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        key, is_fraction = wanted[name]
        if is_fraction:
            if value > 1:
                continue
            fractions.setdefault(key, []).append(value)
        else:
            counters.setdefault(key, []).append(value)
    for key, values in fractions.items():
        out[key] = round(max(values) * 100, 1)
    for key, values in counters.items():
        out[key] = int(sum(values))
    return out


def read_host(endpoint: dict, fetch=get) -> dict:
    url = str(endpoint.get("url") or "")
    engine = endpoint.get("engine")
    if engine not in PATHS:
        engine = ""
    paths = PATHS[engine]
    info: dict = {"served": [], "available": [], "loaded": [], "health": "", "metrics": {}}
    first = paths[0]
    body = fetch(url, first)
    if first == "/api/tags":
        info["available"] = parse_ollama_tags(body)
    else:
        info["served"] = parse_openai_models(body)
    for path in paths[1:]:
        try:
            extra = fetch(url, path)
        except Refused:
            raise
        except HTTPError as exc:
            if path == "/health" and exc.code == 503:
                info["health"] = "loading"
            continue
        except (OSError, ValueError):
            continue
        if path == "/health":
            info["health"] = "ok"
        elif path == "/api/ps":
            info["loaded"] = parse_ollama_ps(extra)
        elif path == "/metrics":
            info["metrics"] = parse_metrics(extra, engine)
    return info


# The same pattern as the node reads: refresh every 5 s (serve.READS ttl), an
# unreachable host is not retried for 30 s (remote.DOWN_TTL_S), and an answer
# older than a minute is not shown (serve.READS max_age).
POLL_S = 5.0
DOWN_S = 30.0
MAX_AGE_S = 60.0
LIMIT = 2  # head + worker + two monitored hosts (netconfig.MAX_MACHINES)


def _reason(exc: BaseException) -> str:
    """A short plain reason for an offline host (no Python repr)."""
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code}"
    if isinstance(exc, URLError):
        text = str(getattr(exc.reason, "strerror", None) or exc.reason or "")
    elif isinstance(exc, (TimeoutError, socket.timeout)):
        text = "timed out"
    elif isinstance(exc, json.JSONDecodeError):
        text = "the reply was not JSON"
    else:
        text = str(exc)
    return _clean(text or "no answer").lower() or "no answer"


class Poller:
    def __init__(self, endpoints: list, fetch=get, clock=time.monotonic, wall=time.time, spawn=None) -> None:
        self._fetch = fetch
        self._clock = clock
        self._wall = wall
        self._spawn = spawn or (
            lambda fn: threading.Thread(target=fn, name="sparksi-monitored-host", daemon=True).start()
        )
        self._lock = threading.Lock()
        self._endpoints: list[dict] = []
        self._states: dict[str, dict] = {}
        for item in endpoints:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("id") or "")
            if not eid or eid in self._states:
                continue
            kept = {
                "id": eid,
                "name": str(item.get("name") or ""),
                "url": str(item.get("url") or ""),
                "engine": str(item.get("engine") or ""),
            }
            self._endpoints.append(kept)
            self._states[eid] = {
                "in_flight": False,
                "next_at": 0.0,
                "ok_at": None,
                "seen_wall": None,
                "error": "",
                "refused": False,
                "info": None,
                "tried": False,
            }

    def due(self) -> list[str]:
        now = self._clock()
        with self._lock:
            return [
                ep["id"]
                for index, ep in enumerate(self._endpoints)
                if index < LIMIT
                and not self._states[ep["id"]]["in_flight"]
                and now >= self._states[ep["id"]]["next_at"]
            ]

    def poll(self, eid: str) -> None:
        endpoint = None
        for item in self._endpoints:
            if item["id"] == eid:
                endpoint = item
                break
        if endpoint is None:
            with self._lock:
                state = self._states.get(eid)
                if state:
                    state["in_flight"] = False
            return
        try:
            try:
                result = read_host(endpoint, self._fetch)
            except Refused as exc:
                with self._lock:
                    state = self._states[eid]
                    state["error"] = str(exc)
                    state["refused"] = True
                    state["next_at"] = self._clock() + DOWN_S
            except Exception as exc:  # noqa: BLE001 - worker must never die
                reason = _reason(exc)
                with self._lock:
                    state = self._states[eid]
                    state["error"] = reason
                    state["refused"] = False
                    state["next_at"] = self._clock() + DOWN_S
            else:
                with self._lock:
                    state = self._states[eid]
                    now = self._clock()
                    state["ok_at"] = now
                    state["seen_wall"] = self._wall()
                    state["info"] = result
                    state["error"] = ""
                    state["refused"] = False
                    state["next_at"] = now + POLL_S
        finally:
            with self._lock:
                state = self._states.get(eid)
                if state:
                    state["tried"] = True
                    state["in_flight"] = False

    def tick(self) -> None:
        for eid in self.due():
            with self._lock:
                state = self._states[eid]
                if state["in_flight"]:
                    continue
                state["in_flight"] = True
            self._spawn(lambda eid=eid: self.poll(eid))

    def view(self) -> list[dict]:
        now = self._clock()
        out: list[dict] = []
        with self._lock:
            for index, endpoint in enumerate(self._endpoints):
                state = self._states[endpoint["id"]]
                parts = urlsplit(endpoint["url"])
                host = parts.hostname or ""
                try:
                    port = parts.port or 80
                except ValueError:
                    port = 0
                address = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
                if index >= LIMIT:
                    status = "not polled"
                elif state["refused"]:
                    status = "refused"
                elif not state["tried"]:
                    status = "checking"
                elif state["error"] == "" and state["ok_at"] is not None and now - state["ok_at"] <= MAX_AGE_S:
                    status = "reachable"
                else:
                    status = "offline"
                info = state["info"] if status == "reachable" and isinstance(state["info"], dict) else {}
                seen = state["seen_wall"]
                if seen is not None:
                    stamp = datetime.datetime.fromtimestamp(seen, datetime.timezone.utc)
                    last_seen = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
                else:
                    last_seen = None
                age = int(now - state["ok_at"]) if state["ok_at"] is not None else None
                out.append(
                    {
                        "id": endpoint["id"],
                        "name": endpoint["name"],
                        "url": endpoint["url"],
                        "engine": endpoint["engine"],
                        "address": address,
                        "state": status,
                        "last_seen": last_seen,
                        "age_s": age,
                        "error": state["error"],
                        "served": info.get("served", []) if status == "reachable" else [],
                        "available": info.get("available", []) if status == "reachable" else [],
                        "loaded": info.get("loaded", []) if status == "reachable" else [],
                        "health": info.get("health", "") if status == "reachable" else "",
                        "metrics": info.get("metrics", {}) if status == "reachable" else {},
                    }
                )
        return out

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            self.tick()
            stop.wait(1.0)


_POLLER: Poller | None = None


def start(endpoints: list, stop: threading.Event) -> Poller | None:
    global _POLLER
    if not endpoints:
        return None
    poller = Poller(endpoints)
    _POLLER = poller
    threading.Thread(target=poller.run, args=(stop,), name="sparksi-monitored-hosts", daemon=True).start()
    return poller


def view() -> list:
    if _POLLER is None:
        return []
    return _POLLER.view()
