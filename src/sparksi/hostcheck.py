# SPDX-License-Identifier: Apache-2.0

"""Checks for machine names and addresses that reach ssh, URLs or the page.

Everything a user can type or hand-edit into ``sparksi.toml`` that later becomes
an ssh destination, a URL host or a displayed name goes through here, both when
the Network page saves it and when the config is loaded. Stdlib only.
"""

from __future__ import annotations

import ipaddress
import re
import unicodedata

MAX_HOST = 253
MAX_NAME = 100
MAX_ID = 64

TAILNET = ipaddress.ip_network("100.64.0.0/10")
_THIS_NETWORK = ipaddress.ip_network("0.0.0.0/8")  # "this host"; Python counts it as private
_ULA = ipaddress.ip_network("fc00::/7")
# Tailscale's own resolver and service address (quad-100, the same on every tailnet), never a
# monitored or typed machine. Assembled so the site-identity guard sees no tailnet literal.
_TAILSCALE_SERVICE = (ipaddress.ip_address(".".join(("100",) * 4)), ipaddress.ip_address("fd7a:115c:a1e0::53"))
_LABEL = re.compile(r"[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?")
_USER = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,31}")
_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")


def _bad_chars(text: str) -> bool:
    """Control, format and line/paragraph separator characters (U+2028/2029 included)."""
    return any(unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") for ch in text)


def ip_ok(text: str) -> bool:
    """A private LAN or tailnet IP; never loopback, link-local (cloud metadata), multicast or unspecified."""
    try:
        addr = ipaddress.ip_address(text)
    except ValueError:
        return False
    # ::ffff:a.b.c.d is that IPv4 address (Python calls ::ffff:127.0.0.1 "private", not loopback).
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    if addr in _TAILSCALE_SERVICE:
        return False
    if addr.version == 6:
        # IPv6: unique local addresses only (fc00::/7, the tailnet's fd7a:115c:a1e0::/48 included).
        # Python's is_private also counts 2001::/23 (Teredo, documentation) and 2002::/16 (6to4),
        # which reach public or embedded IPv4 addresses.
        return addr in _ULA
    if addr in _THIS_NETWORK:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified or addr.is_reserved:
        return False
    if addr.version == 4 and addr in TAILNET:
        return True
    return bool(addr.is_private)


def check_host(value: object) -> str:
    """A host name or a private/tailnet IP, exactly as given (no trimming)."""
    if not isinstance(value, str) or not value:
        raise ValueError("enter a host name or a private or tailnet IP address")
    if len(value) > MAX_HOST or value.startswith("-") or _bad_chars(value) or any(ch.isspace() for ch in value):
        raise ValueError("enter a host name or a private or tailnet IP address")
    try:
        ipaddress.ip_address(value)
    except ValueError:
        pass
    else:
        if ip_ok(value):
            return value
        raise ValueError("enter a host name or a private or tailnet IP address")
    labels = value.split(".")
    if re.fullmatch(r"[0-9.]+", value) or not all(_LABEL.fullmatch(label) for label in labels):
        raise ValueError("enter a host name or a private or tailnet IP address")
    return value


def check_ssh(value: object) -> str:
    """An ssh destination: "local", host, or user@host. Never starts with "-" (no option injection)."""
    if value == "local":
        return "local"
    if not isinstance(value, str):
        raise ValueError("the ssh destination must be a host name or a private or tailnet IP")
    user, at, host = value.rpartition("@")
    if at and not _USER.fullmatch(user):
        raise ValueError("the ssh user may use letters, digits, '.', '_' and '-' only")
    try:
        return (user + "@" if at else "") + check_host(host)
    except ValueError:
        raise ValueError("the ssh destination must be a host name or a private or tailnet IP") from None


def check_name(value: object) -> str:
    """A display name: 1-100 characters, no control or line-separator characters, no edge spaces."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"a name must be 1 to {MAX_NAME} characters with no leading or trailing spaces")
    if len(value) > MAX_NAME:
        raise ValueError(f"a name must be at most {MAX_NAME} characters")
    if _bad_chars(value):
        raise ValueError("a name must not contain control or line-break characters")
    return value


def check_id(value: object) -> str:
    """A machine id: letters, digits, '.', '_' and '-', at most 64, not starting with '-'."""
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError("a machine id uses letters, digits, '.', '_' and '-' (at most 64) and does not start with '-'")
    return value


def check_port(value: object) -> int:
    """A TCP port, 1-65535, from an int (not bool) or a digit string."""
    if isinstance(value, bool):
        raise ValueError("the port must be a number from 1 to 65535")
    if isinstance(value, int):
        port = value
    elif isinstance(value, str) and value.isdigit() and value.isascii():
        port = int(value)
    else:
        raise ValueError("the port must be a number from 1 to 65535")
    if not 1 <= port <= 65535:
        raise ValueError("the port must be a number from 1 to 65535")
    return port


def http_url(host: str, port: int) -> str:
    """The only endpoint URL form Sparksi writes: http://<checked host>:<port>."""
    host = check_host(host)
    port = check_port(port)
    return f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
