# SPDX-License-Identifier: Apache-2.0

"""Read-only network discovery for the Sparksi controller.

Stdlib only.  Every external step fails soft: any exception during
discovery, probing or parsing yields an empty result rather than an
error.  Only GET requests are made, no credentials are sent and
redirects are never followed.

Example addresses in comments use documentation ranges only
(100.64.0.x, 192.0.2.x, example.ts.net).
"""

from __future__ import annotations

import concurrent.futures
import ipaddress
import json
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

# --------------------------------------------------------------------------
# Hosts
# --------------------------------------------------------------------------


def _run(argv: list[str], timeout: float) -> str:
    """Run a command, return stdout, or "" on any failure."""
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.stdout or ""
    except Exception:
        return ""


def _ipv4_first(ips: list[str]) -> list[str]:
    """Stable sort: IPv4 entries first, everything else after, order kept."""
    v4 = [ip for ip in ips if _is_ipv4(ip)]
    rest = [ip for ip in ips if not _is_ipv4(ip)]
    return v4 + rest


def _is_ipv4(text: str) -> bool:
    try:
        socket.inet_aton(text)
    except Exception:
        return False
    return text.count(".") == 3


def _endpoint_ip(endpoint: Any) -> str:
    """The private (LAN) IPv4 of an "ip:port" Tailscale endpoint, else ""."""
    if not isinstance(endpoint, str) or ":" not in endpoint:
        return ""
    ip = endpoint.rsplit(":", 1)[0]
    if not _is_ipv4(ip):
        return ""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ""
    lan = addr.is_private and not addr.is_loopback and not addr.is_link_local and addr not in _TAILNET
    return ip if lan else ""


_TAILNET = ipaddress.ip_network("100.64.0.0/10")


def _is_tailnet(ip: str) -> bool:
    try:
        return ipaddress.ip_address(ip) in _TAILNET
    except ValueError:
        return False


def tailscale_hosts(text: str | None = None) -> list[dict]:
    """Hosts from `tailscale status --json` (read-only)."""
    if text is None:
        text = _run(["tailscale", "status", "--json"], 4)
    hosts: list[dict] = []
    if not text:
        return hosts
    try:
        data = json.loads(text)
    except Exception:
        return hosts
    if not isinstance(data, dict):
        return hosts

    def _entry(raw: Any) -> dict | None:
        if not isinstance(raw, dict):
            return None
        name = raw.get("HostName") or ""
        dns = str(raw.get("DNSName") or "")
        if dns:
            dns = dns.rstrip(".")
            if not name:
                name = dns.split(".")[0]
        ips = _ipv4_first([ip for ip in (raw.get("TailscaleIPs") or []) if isinstance(ip, str)])
        # The peer's LAN addresses (from its endpoints) let a neighbor-table entry merge
        # with it, so the same machine is not probed twice. Probes use the tailnet address.
        for endpoint in [raw.get("CurAddr")] + list(raw.get("Addrs") or []):
            lan = _endpoint_ip(endpoint)
            if lan and lan not in ips:
                ips.append(lan)
        return {
            "name": name,
            "dns": dns,
            "ips": ips,
            "os": str(raw.get("OS") or ""),
            "sources": ["tailscale"],
            "self": False,
        }

    self_raw = data.get("Self")
    self_host = _entry(self_raw)
    if self_host is not None:
        self_host["self"] = True
        hosts.append(self_host)

    peers = data.get("Peer")
    if isinstance(peers, dict):
        for raw in peers.values():
            if not isinstance(raw, dict):
                continue
            if not raw.get("Online"):
                continue
            peer = _entry(raw)
            if peer is not None:
                hosts.append(peer)
    return hosts


def neighbor_hosts(text: str | None = None) -> list[dict]:
    """LAN neighbours from `ip -j neigh` (read-only)."""
    if text is None:
        text = _run(["ip", "-j", "neigh"], 3)
    hosts: list[dict] = []
    if not text:
        return hosts
    try:
        entries = json.loads(text)
    except Exception:
        return hosts
    if not isinstance(entries, list):
        return hosts
    skip_prefixes = ("docker", "br-", "veth", "virbr", "tailscale")
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        dst = entry.get("dst") or ""
        if not isinstance(dst, str) or not _is_ipv4(dst):
            continue
        state = entry.get("state")
        states = state if isinstance(state, list) else [state]
        if any(s in ("FAILED", "INCOMPLETE") for s in states):
            continue
        dev = str(entry.get("dev") or "")
        if dev.startswith(skip_prefixes):
            continue
        hosts.append(
            {
                "name": "",
                "dns": "",
                "ips": [dst],
                "os": "",
                "sources": ["lan"],
                "self": False,
            }
        )
    return hosts


def _lan_or_tailnet(ip: str) -> bool:
    """A private LAN IPv4 (not loopback, link-local, multicast or unspecified) or a tailnet address."""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    if addr.version != 4:
        return False
    if addr.is_loopback or addr.is_link_local or addr.is_multicast or addr.is_unspecified:
        return False
    if _is_tailnet(ip):
        return True
    return bool(addr.is_private)


def mdns_hosts(text: str | None = None) -> list[dict]:
    """SSH services from `avahi-browse -rtp _ssh._tcp` (read-only)."""
    if text is None:
        if shutil.which("avahi-browse") is None:
            return []
        text = _run(["avahi-browse", "-rtp", "_ssh._tcp"], 4)
    hosts: list[dict] = []
    if not text:
        return hosts
    for line in text.splitlines():
        if not line.startswith("="):
            continue
        parts = line.split(";")
        if len(parts) < 9:
            continue
        # =;iface;proto;name;type;domain;hostname;address;port;txt
        hostname = parts[6]
        address = parts[7]
        if not _is_ipv4(address):
            continue
        if not _lan_or_tailnet(address):
            continue
        name = hostname
        if name.endswith(".local"):
            name = name[: -len(".local")]
        name = name.rstrip(".")
        dns = hostname.rstrip(".")
        hosts.append(
            {
                "name": name,
                "dns": dns,
                "ips": [address],
                "os": "",
                "sources": ["mdns"],
                "self": False,
            }
        )
    return hosts


def merge(groups: list[list[dict]]) -> list[dict]:
    """Merge host records that share an IP or a non-empty lower-case name."""
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        while parent.get(a, a) != a:
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    flat: list[dict] = [dict(h) for group in groups for h in group]
    for i in range(len(flat)):
        parent.setdefault(i, i)

    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            a, b = flat[i], flat[j]
            shared_ip = bool(set(a.get("ips") or []) & set(b.get("ips") or []))
            na = str(a.get("name") or "").lower()
            nb = str(b.get("name") or "").lower()
            shared_name = bool(na and na == nb)
            if shared_ip or shared_name:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(len(flat)):
        clusters.setdefault(find(i), []).append(i)

    merged: list[dict] = []
    for members in clusters.values():
        base = dict(flat[members[0]])
        ips: list[str] = []
        for idx in members:
            for ip in flat[idx].get("ips") or []:
                if ip not in ips:
                    ips.append(ip)
        base["ips"] = _ipv4_first(ips)
        sources: set[str] = set()
        for idx in members:
            sources.update(flat[idx].get("sources") or [])
        base["sources"] = sorted(sources)
        for field in ("name", "dns", "os"):
            for idx in members:
                val = flat[idx].get(field)
                if val:
                    base[field] = val
                    break
        base["self"] = any(flat[idx].get("self") for idx in members)
        merged.append(base)

    def sort_key(h: dict) -> tuple:
        name = h.get("name") or ""
        first_ip = (h.get("ips") or [""])[0]
        return (not h.get("self"), name or first_ip)

    return sorted(merged, key=sort_key)


# --------------------------------------------------------------------------
# Probing
# --------------------------------------------------------------------------

PORTS = (8000, 8001, 8014, 8080, 11434, 1234, 5000, 7500)
PATHS = ("/v1/models", "/api/tags", "/health")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses to follow redirects."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)


def _open(ip: str, port: int, timeout: float = 0.4) -> bool:
    """True when a TCP connection to ip:port succeeds."""
    try:
        sock = socket.create_connection((ip, port), timeout=timeout)
        sock.close()
        return True
    except Exception:
        return False


def _url_for(ip: str, port: int, path: str) -> str:
    host = f"[{ip}]" if ":" in ip else ip
    return f"http://{host}:{port}{path}"


def _get(ip: str, port: int, path: str, timeout: float = 1.5) -> tuple[int, Any]:
    """GET a path; return (status, parsed json or None).  Never follows redirects."""
    url = _url_for(ip, port, path)
    try:
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "sparksi-discover",
            },
        )
        with _opener().open(request, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            raw = resp.read(262144)
        try:
            body = json.loads(raw.decode("utf-8", "replace"))
        except Exception:
            body = None
        return int(status), body
    except urllib.error.HTTPError as exc:
        try:
            exc.close()
        except Exception:
            pass
        return int(exc.code), None
    except Exception:
        return 0, None


def classify(port: int, path: str, status: int, body: Any) -> dict | None:
    """Identify the service answering on port/path.  None when unmatched."""
    if status in (401, 403):
        return {"port": port, "engine": "needs a key", "models": []}
    if path == "/health" and status == 200:
        return {"port": port, "engine": "HTTP service", "models": []}
    if status != 200 or not isinstance(body, dict):
        return None
    if path == "/v1/models":
        data = body.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            owned = ""
            if isinstance(first, dict):
                owned = str(first.get("owned_by") or "").lower()
            if owned == "vllm":
                engine = "vLLM"
            elif owned == "sglang":
                engine = "SGLang"
            elif owned == "llamacpp":
                engine = "llama.cpp"
            else:
                engine = "OpenAI-compatible"
            models = []
            for item in data:
                if isinstance(item, dict) and item.get("id"):
                    models.append(item["id"])
            return {"port": port, "engine": engine, "models": models}
        return None
    if path == "/api/tags":
        models_raw = body.get("models")
        if isinstance(models_raw, list) and models_raw:
            models = []
            for item in models_raw:
                if isinstance(item, dict) and item.get("name"):
                    models.append(item["name"])
            return {"port": port, "engine": "Ollama", "models": models}
        return None
    if path == "/api/config":
        if "controller" in body:
            return {"port": port, "engine": "Sparksi", "models": []}
        return None
    if path == "/health":
        return {"port": port, "engine": "HTTP service", "models": []}
    return None


def probe_port(ip: str, port: int) -> dict | None:
    """Probe one port; at most one GET per candidate path, stop on first match."""
    if not _open(ip, port):
        return None
    paths = ("/api/config",) if port == 7500 else PATHS
    for path in paths:
        status, body = _get(ip, port, path)
        result = classify(port, path, status, body)
        if result is not None:
            result["url"] = f"http://{ip}:{port}"
            return result
    return {"port": port, "engine": "open port", "models": [], "url": f"http://{ip}:{port}"}


def _probe_ip(host: dict) -> str:
    """The address to probe: a tailnet IPv4 when there is one, else the first IPv4."""
    v4 = [ip for ip in (host.get("ips") or []) if _is_ipv4(ip)]
    tailnet = [ip for ip in v4 if _is_tailnet(ip)]
    return (tailnet or v4 or [""])[0]


# A slow device must not hold a scan: probes still running after this are dropped.
SCAN_DEADLINE_S = 20.0


def probe(
    hosts: list[dict],
    ports: tuple[int, ...] = PORTS,
    workers: int = 24,
    deadline: float = SCAN_DEADLINE_S,
) -> int:
    """Probe every host in place, filling host["endpoints"]. One address per host.

    Returns the number of probes that did not finish within ``deadline`` seconds.
    """
    started = time.monotonic()
    jobs = [
        (index, _probe_ip(host), port)
        for index, host in enumerate(hosts)
        for port in ports
        if _probe_ip(host)
    ]
    found: dict[int, list[dict]] = {}
    timed_out = 0
    if jobs:
        pool = ThreadPoolExecutor(max_workers=max(1, workers))
        futures = [(index, pool.submit(probe_port, ip, port)) for index, ip, port in jobs]
        owner = {future: index for index, future in futures}
        pending = set(owner)
        try:
            while pending:
                remaining = deadline - (time.monotonic() - started)
                if remaining <= 0:
                    break
                done, pending = concurrent.futures.wait(
                    pending, timeout=remaining, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for future in done:
                    index = owner[future]
                    try:
                        res = future.result()
                    except Exception:
                        res = None
                    if res is not None:
                        found.setdefault(index, []).append(res)
        finally:
            not_done = [f for _, f in futures if not f.done()]
            timed_out = len(not_done)
            pool.shutdown(wait=False, cancel_futures=True)
    for index, host in enumerate(hosts):
        host["endpoints"] = sorted(found.get(index, []), key=lambda e: e.get("port", 0))
    return timed_out


# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------

SPARK_NAME = re.compile(r"spark|dgx|gx10|gb10", re.I)


def role_for(host: dict, known: dict[str, str]) -> tuple[str, str]:
    """Classify a host; known maps lower names/IPs of Sparks to node ids."""
    if host.get("self"):
        return ("controller", "this machine")
    ips = host.get("ips") or []
    names = [
        str(host.get("name") or "").lower(),
        str(host.get("dns") or "").lower(),
    ]
    for key in list(ips) + names:
        if key and key in known:
            return ("spark", "configured Spark " + known[key])
    haystack = " ".join(n for n in names if n)
    if SPARK_NAME.search(haystack):
        return ("spark", "name looks like a DGX Spark")
    for ep in host.get("endpoints") or []:
        if ep.get("engine") and ep.get("engine") != "open port":
            return ("inference", "serves an inference API")
    return ("machine", "no inference port answered")


# --------------------------------------------------------------------------
# Proposed config
# --------------------------------------------------------------------------


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "node"


def proposal(host: dict) -> str:
    """A TOML snippet proposing how to use this host (never written by us)."""
    role = host.get("role") or ""
    name = str(host.get("name") or "")
    if role == "spark":
        ts_ip = next((ip for ip in (host.get("ips") or []) if _is_tailnet(ip)), "")
        lines = [
            "[[nodes]]",
            "id = " + json.dumps(_slug(name)),
            "name = " + json.dumps(name),
            "ssh = " + json.dumps(name),
            "tailscale = " + json.dumps(ts_ip),
        ]
        return "\n".join(lines)
    if role == "inference":
        out = ["# A monitored inference host (Network page: Add to Sparksi)."]
        for ep in host.get("endpoints") or []:
            out.append("[[endpoints]]")
            out.append("name = " + json.dumps(str(host.get("name") or "")))
            out.append("url = " + json.dumps(str(ep.get("url") or "")))
            out.append("engine = " + json.dumps(str(ep.get("engine") or "")))
        return "\n".join(out)
    return ""


# --------------------------------------------------------------------------
# Scan and cache
# --------------------------------------------------------------------------

_CACHE: dict = {"scanned_at": None, "hosts": [], "took_s": None}
_LOCK = threading.Lock()
MIN_GAP_S = 15.0
_last: float = 0.0


def _do_scan(
    known: dict[str, str],
    *,
    tailscale_text: str | None = None,
    neigh_text: str | None = None,
    mdns_text: str | None = None,
    ports: tuple[int, ...] = PORTS,
) -> dict:
    """One discovery pass. The caller holds ``_LOCK``."""
    started = time.monotonic()
    groups = [
        tailscale_hosts(tailscale_text),
        neighbor_hosts(neigh_text),
        mdns_hosts(mdns_text),
    ]
    hosts = merge(groups)
    late = 0
    try:
        late = probe(hosts, ports=ports)
    except Exception:
        for host in hosts:
            host.setdefault("endpoints", [])
    for host in hosts:
        role, why = role_for(host, known)
        host["role"] = role
        host["why"] = why
        host["proposal"] = proposal(host)
    # Sparks and inference hosts first; silent LAN neighbors last.
    rank = {"controller": 0, "spark": 1, "inference": 2}
    hosts.sort(key=lambda h: (rank.get(h["role"], 3), not h.get("name"), str(h.get("name") or (h.get("ips") or [""])[0]).lower()))
    took = round(time.monotonic() - started, 1)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _CACHE.clear()
    _CACHE.update({"scanned_at": stamp, "hosts": hosts, "took_s": took})
    if late:
        _CACHE["late_probes"] = late  # ports still answering after SCAN_DEADLINE_S were skipped
    return dict(_CACHE)


def scan(known: dict[str, str], **kw: Any) -> dict:
    """Run a full discovery pass now, unless one is running or too recent."""
    global _last
    if not _LOCK.acquire(blocking=False):
        return dict(_CACHE, busy=True)
    try:
        now = time.monotonic()
        if _last and (now - _last) < MIN_GAP_S:
            return dict(_CACHE, too_soon=True)
        out = _do_scan(known, **kw)
        _last = now
        return out
    finally:
        _LOCK.release()


def start(known: dict[str, str], **kw: Any) -> tuple[int, dict]:
    """Start a scan in the background and answer at once.

    202 with ``running`` when a scan started or one is already running; 200
    with ``too_soon`` inside the cooldown. One scan at a time; the cooldown
    counts from when a scan starts, so a failing scan cannot be retried in a loop.
    """
    global _last
    if not _LOCK.acquire(blocking=False):
        return 202, dict(_CACHE, running=True, busy=True)
    now = time.monotonic()
    if _last and (now - _last) < MIN_GAP_S:
        _LOCK.release()
        return 200, dict(_CACHE, too_soon=True, running=False)
    _last = now

    def work() -> None:
        try:
            _do_scan(known, **kw)
        except Exception:
            _CACHE["error"] = "scan failed"
        finally:
            _LOCK.release()

    try:
        threading.Thread(target=work, name="sparksi-discover", daemon=True).start()
    except Exception:
        _LOCK.release()
        raise
    return 202, dict(_CACHE, running=True, started=True)


def cached() -> dict:
    """A copy of the most recent scan, and whether one is running now."""
    return dict(_CACHE, running=_LOCK.locked())


def reset_for_tests() -> None:
    """Clear cached scan state (test helper)."""
    global _last
    _CACHE.clear()
    _CACHE.update({"scanned_at": None, "hosts": [], "took_s": None})
    _last = 0.0
