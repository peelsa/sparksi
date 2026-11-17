# SPDX-License-Identifier: Apache-2.0

"""Host allowlist for DNS-rebinding defense: which Host headers this controller answers to."""

from __future__ import annotations

import ipaddress
import json
import socket
import subprocess
import threading
import time


def host_name(host_header: str) -> str | None:
    """Normalize a Host header to a bare name: lower-case, no brackets, no port, no trailing dot.

    Mirrors serve's origin parsing: "[v6]:port", "name:port" and "name." all fold to "name".
    Returns None for empty or malformed input.
    """
    host = host_header.strip().lower()
    if not host:
        return None
    if host.startswith("["):
        end = host.find("]")
        if end < 0:
            return None
        name, rest = host[1:end], host[end + 1:]
        port_text = rest[1:] if rest.startswith(":") else ""
    elif host.count(":") == 1:
        name, port_text = host.split(":", 1)
    else:
        name, port_text = host, ""
    name = name.rstrip(".")
    if not name:
        return None
    if port_text and not port_text.isdigit():
        return None
    return name


def _addr_key(address: str) -> str:
    """An address as stored: lower-case, no brackets, no zone suffix, no trailing dot."""
    address = address.strip().lower().strip("[]").rstrip(".")
    if "%" in address:
        address = address.split("%", 1)[0]
    return address


def _names_from(name: str, out: set[str]) -> None:
    key = host_name(name)
    if not key:
        return
    out.add(key)
    try:
        ipaddress.ip_address(key)
        return  # an address has no short name
    except ValueError:
        pass
    first = key.split(".", 1)[0]
    if first:
        out.add(first)


def _getaddrinfo_names(host: str, out: set[str]) -> None:
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return
    for info in infos:
        try:
            address = info[4][0]
        except Exception:
            continue
        key = _addr_key(str(address))
        if key:
            out.add(key)


def _canonical_names(host: str, out: set[str]) -> None:
    try:
        infos = socket.getaddrinfo(host, None, 0, 0, 0, socket.AI_CANONNAME)
    except Exception:
        return
    for info in infos:
        try:
            canon = info[3]
        except Exception:
            continue
        if isinstance(canon, str) and canon.strip():
            _names_from(canon, out)


def _interface_addresses(out: set[str]) -> None:
    try:
        proc = subprocess.run(
            ["ip", "-j", "addr"],
            timeout=3,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return
    try:
        items = json.loads(proc.stdout)
    except Exception:
        return
    if not isinstance(items, list):
        return
    for item in items:
        if not isinstance(item, dict):
            continue
        for addr in item.get("addr_info") or []:
            if not isinstance(addr, dict):
                continue
            local = addr.get("local")
            if not isinstance(local, str):
                continue
            key = _addr_key(local)
            if key:
                out.add(key)


def _tailscale_names(out: set[str]) -> None:
    try:
        proc = subprocess.run(
            ["tailscale", "status", "--json"],
            timeout=3,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return
    try:
        status = json.loads(proc.stdout)
    except Exception:
        return
    if not isinstance(status, dict):
        return
    self_info = status.get("Self")
    if not isinstance(self_info, dict):
        return
    dns_name = self_info.get("DNSName")
    if isinstance(dns_name, str) and dns_name.strip():
        _names_from(dns_name, out)
    tailscale_ips = self_info.get("TailscaleIPs")
    if isinstance(tailscale_ips, list):
        for address in tailscale_ips:
            if isinstance(address, str):
                key = _addr_key(address)
                if key:
                    out.add(key)


def gather(extra: list[str]) -> set[str]:
    """Every name and address this machine answers to, plus the extra list. Never raises."""
    found: set[str] = set()
    found.update({"localhost", "127.0.0.1", "::1"})
    try:
        host = socket.gethostname()
    except Exception:
        host = ""
    if isinstance(host, str) and host.strip():
        _names_from(host, found)
        # The canonical name from a forward lookup. Not socket.getfqdn(): its reverse
        # lookup can wait for a DNS timeout (seconds) on a machine without PTR records.
        _canonical_names(host, found)
        _getaddrinfo_names(host, found)
    _interface_addresses(found)
    _tailscale_names(found)
    for item in extra or []:
        if isinstance(item, str) and item.strip():
            key = host_name(item)
            if key:
                found.add(key)
    found.discard("")
    return found


class Allowlist:
    """The set of acceptable Host names, rebuilt on a ttl, with a bounded refresh on miss.

    A request never waits for another request's rebuild: the names are swapped in
    under a short lock, and only one rebuild runs at a time. A stale list is rebuilt
    in the background while requests keep using the old one. Only a request whose
    Host is missing waits, for its own rebuild, at most every ``miss_refresh`` seconds.
    """

    def __init__(self, extra_fn, ttl: float = 300.0, miss_refresh: float = 30.0) -> None:
        self._extra_fn = extra_fn
        self._ttl = ttl
        self._miss_refresh = miss_refresh
        self._lock = threading.Lock()  # guards _names and _built_at only
        self._building = threading.Lock()  # one rebuild at a time
        self._names: set[str] = set()
        self._built_at = 0.0
        self._miss_at = 0.0
        self._pending = False

    def _rebuild(self) -> bool:
        """Rebuild unless another rebuild is running. True if this call rebuilt."""
        if not self._building.acquire(blocking=False):
            return False
        try:
            try:
                extra = list(self._extra_fn() or [])
            except Exception:
                extra = []
            try:
                names = gather(extra)
            except Exception:
                return False
            with self._lock:
                self._names = names
                self._built_at = time.monotonic()
            return True
        finally:
            self._building.release()

    def _rebuild_later(self) -> None:
        with self._lock:
            if self._pending:
                return
            self._pending = True

        def run() -> None:
            try:
                self._rebuild()
            finally:
                with self._lock:
                    self._pending = False

        threading.Thread(target=run, name="sparksi-hosts", daemon=True).start()

    def _snapshot(self) -> tuple[set[str], float]:
        with self._lock:
            return self._names, self._built_at

    def refresh(self) -> set[str]:
        """Rebuild now (at startup, so the first request does not wait)."""
        with self._building:
            pass
        self._rebuild()
        return set(self._snapshot()[0])

    def names(self) -> set[str]:
        """The current allowlist. Built now if empty; rebuilt in the background when old."""
        names, built = self._snapshot()
        if not names:
            self._rebuild()
            names, built = self._snapshot()
        elif time.monotonic() - built >= self._ttl:
            self._rebuild_later()
        return set(names)

    def allows(self, host_header: str) -> bool:
        """True if this Host header names this machine. Empty passes (not a browser)."""
        host_header = (host_header or "").strip()
        if not host_header:
            return True
        name = host_name(host_header)
        if not name:
            return False
        names = self.names()
        if name in names:
            return True
        now = time.monotonic()
        with self._lock:
            if now - max(self._built_at, self._miss_at) < self._miss_refresh:
                return False
            self._miss_at = now
        if not self._rebuild() and self._building.acquire(timeout=10):
            self._building.release()  # a rebuild was already running: use its result
        return name in self._snapshot()[0]


REFUSAL = (
    "This Sparksi controller does not answer to the host name {name!r}. "
    "Open it by one of its own names or addresses, or add the name to "
    "allowed_hosts under [controller] in sparksi.toml and restart.\n"
)
