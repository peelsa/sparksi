# SPDX-License-Identifier: Apache-2.0

"""Optional read-only mirror of another controller's monitor feed.

``[controller] monitor_upstream = "http://127.0.0.1:7500"`` makes this process
poll ``GET /api/monitor`` on that controller once a second. The mirrored card
feeds Ready confirmation, the gauges and the history recorder. It is off by
default. The mirror only sends GET to that one path, never writes to the
upstream, ignores proxies, refuses redirects, caps the reply size and uses a
short timeout. Requests here never wait for it: they read the last good card,
and once that card is older than a few seconds they fall back to this
controller's own announce feed.
"""

from __future__ import annotations

import ipaddress
import json
import threading
import time
from typing import Any, Callable, Iterable
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

PATH = "/api/monitor"
TIMEOUT_S = 2.0
POLL_S = 1.0
FRESH_S = 5.0
MAX_BYTES = 4 * 1024 * 1024
_TAILNET_V4 = ipaddress.ip_network("100.64.0.0/10")
_TAILNET_V6 = ipaddress.ip_network("fd7a:115c:a1e0::/48")
_LOOPBACK_NAMES = {"localhost"}


class _RefuseRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        raise URLError("redirect refused")


def _opener():
    # An empty proxy map ignores http_proxy, so the request goes only to the upstream.
    return build_opener(_RefuseRedirect, ProxyHandler({}))


def no_redirect_opener(*handlers):
    """An opener that ignores proxy settings and refuses redirects (also used for metrics_url
    and the monitored hosts, which add a handler that connects only to a checked address)."""
    return build_opener(_RefuseRedirect, ProxyHandler({}), *handlers)


def _host_allowed(host: str) -> bool:
    host = host.lower().rstrip(".")
    if host in _LOOPBACK_NAMES or host.endswith(".ts.net"):
        return True
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    return addr.is_loopback or addr in _TAILNET_V4 or addr in _TAILNET_V6


def upstream_url(raw: str) -> str:
    """Normalise ``raw`` to ``http://host:port``; raise ValueError when not allowed.

    Plain http to a loopback or tailnet host only, with no credentials, query
    or path.
    """
    parsed = urlparse(str(raw or "").strip())
    host = parsed.hostname or ""
    if parsed.scheme != "http" or not host or parsed.username or parsed.password:
        raise ValueError("monitor_upstream must be http://host:port on loopback or the tailnet")
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
        raise ValueError("monitor_upstream must not have a path")
    if not _host_allowed(host):
        raise ValueError("monitor_upstream must be on loopback or the tailnet")
    port = parsed.port or 80
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}"


def fetch_card(base: str, timeout: float = TIMEOUT_S) -> dict:
    """One GET of the upstream monitor card."""
    request = Request(base + PATH, method="GET", headers={"Accept": "application/json"})
    with _opener().open(request, timeout=timeout) as response:
        if urlparse(response.geturl()).path != PATH:
            raise URLError("unexpected path")
        raw = response.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("monitor reply too large")
    body = json.loads(raw.decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError("monitor reply is not an object")
    return body


class Mirror:
    """Last good upstream card. Reads never block on the network."""

    def __init__(self, base: str, fallback: Callable[[], Any], fetch: Callable[[str], dict] = fetch_card) -> None:
        self.base = base
        self.fallback = fallback
        self.fetch = fetch
        self._card: dict | None = None
        self._at = 0.0
        self._lock = threading.Lock()
        self.stop = threading.Event()
        self.errors = 0

    def poll_once(self) -> bool:
        try:
            card = self.fetch(self.base)
        except (OSError, URLError, ValueError, json.JSONDecodeError):
            self.errors += 1
            return False
        card = dict(card)
        card.pop("history", None)  # this controller adds its own history
        card.pop("monitored", None)  # and its own monitored hosts
        with self._lock:
            self._card = card
            self._at = time.monotonic()
        return True

    def fresh(self, now: float | None = None) -> bool:
        moment = time.monotonic() if now is None else now
        with self._lock:
            return self._card is not None and moment - self._at <= FRESH_S

    def card(self) -> Any:
        with self._lock:
            card, at = self._card, self._at
        if card is not None and time.monotonic() - at <= FRESH_S:
            body = dict(card)
            body["mirrored"] = True
            return body
        return self.fallback()

    def run(self) -> None:
        while not self.stop.is_set():
            self.poll_once()
            self.stop.wait(POLL_S)

    def start(self) -> "Mirror":
        threading.Thread(target=self.run, name="sparksi-monitor-mirror", daemon=True).start()
        return self


def configured_upstream() -> str:
    """The configured upstream, or "" when the mirror is off."""
    from sparksi.config import CFG, env

    return str(env("SPARKSI_MONITOR_UPSTREAM") or CFG.controller.get("monitor_upstream") or "").strip()


def _host_key(text: str) -> str:
    """Lower case, no brackets, no trailing dots ("localhost." is localhost)."""
    return str(text or "").strip().strip("[]").lower().rstrip(".")


def _as_ip(text: str):
    """The address, with an IPv4-mapped IPv6 (::ffff:a.b.c.d) read as its IPv4 address."""
    try:
        addr = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return None
    mapped = getattr(addr, "ipv4_mapped", None)
    return mapped if mapped is not None else addr


def refuse_self(base: str, own_port: int, own_names: Iterable[str] = ()) -> None:
    """Refuse an upstream on this controller's own port that names this machine.

    Loopback (any spelling, trailing dots included) and every name or address this
    machine answers to (its host allowlist: hostname, tailnet IP, MagicDNS name)
    would make the mirror read itself and freeze the gauges and Ready.
    """
    parsed = urlparse(base)
    if parsed.port != int(own_port):
        return
    host = _host_key(parsed.hostname or "")
    names = {_host_key(n) for n in own_names if n} | _LOOPBACK_NAMES
    if host in names:
        raise ValueError("monitor_upstream points at this controller")
    addr = _as_ip(host)
    if addr is None:
        return
    own_ips = {ip for ip in (_as_ip(n) for n in names) if ip is not None}
    if addr.is_loopback or addr.is_unspecified or addr in own_ips:
        raise ValueError("monitor_upstream points at this controller")
