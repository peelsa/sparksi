# SPDX-License-Identifier: Apache-2.0

"""DNS rebinding defense: the controller answers only to its own names and addresses."""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import threading
from types import SimpleNamespace

import pytest

import sparksi.hosts as hosts
import sparksi.serve as serve_mod
from sparksi.hosts import Allowlist, gather, host_name
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

IP_JSON = json.dumps([
    {"ifname": "lo", "addr_info": [{"local": "127.0.0.1"}, {"local": "::1"}]},
    {"ifname": "eth0", "addr_info": [{"local": "192.0.2.5"}, {"local": "fe80::1%eth0"}]},
    {"ifname": "tailscale0", "addr_info": [{"local": "100.64.0.9"}]},
])
TS_JSON = json.dumps({"Self": {"DNSName": "box.example.ts.net.", "TailscaleIPs": ["100.64.0.9", "fd7a:115c:a1e0::9"]}})


def _fake_machine(monkeypatch, ip_out=IP_JSON, ts_out=TS_JSON, fail=False):
    def run(argv, **kwargs):
        if fail:
            raise FileNotFoundError(argv[0])
        out = ip_out if argv[0] == "ip" else ts_out
        return subprocess.CompletedProcess(argv, 0, out, "")

    # Replace the modules as hosts.py sees them only; the test server keeps the real socket.
    monkeypatch.setattr(hosts, "subprocess", SimpleNamespace(run=run))
    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        canon = "box.lan.example" if flags & socket.AI_CANONNAME else ""
        return [(socket.AF_INET, 0, 0, canon, ("192.0.2.5", 0))]

    def getfqdn(name=""):
        raise AssertionError("getfqdn does a reverse lookup that can stall for seconds")

    monkeypatch.setattr(hosts, "socket", SimpleNamespace(
        gethostname=lambda: "box", getfqdn=getfqdn, getaddrinfo=getaddrinfo, AI_CANONNAME=socket.AI_CANONNAME,
    ))


def test_host_name_matches_the_origin_normalization() -> None:
    assert host_name("Box.Example.ts.net.:7601") == "box.example.ts.net"
    assert host_name("[FD7A:115C:A1E0::9]:7601") == "fd7a:115c:a1e0::9"
    assert host_name("127.0.0.1") == "127.0.0.1"
    assert host_name("box:abc") is None and host_name("[x") is None and host_name("") is None


def test_gather_finds_every_name_and_address(monkeypatch) -> None:
    _fake_machine(monkeypatch)
    names = gather(["Extra.Example.org."])
    for want in ("localhost", "127.0.0.1", "::1", "box", "box.lan.example", "192.0.2.5", "fe80::1",
                 "100.64.0.9", "fd7a:115c:a1e0::9", "box.example.ts.net", "extra.example.org"):
        assert want in names, (want, sorted(names))
    assert "100" not in names and "192" not in names  # an address has no short name


def test_gather_fails_soft_without_ip_or_tailscale(monkeypatch) -> None:
    _fake_machine(monkeypatch, fail=True)
    names = gather([])
    assert {"localhost", "127.0.0.1", "::1", "box"} <= names
    _fake_machine(monkeypatch, ip_out="not json", ts_out="{}")
    assert "box" in gather([])


def test_allowlist_refreshes_on_a_miss_at_most_every_interval(monkeypatch) -> None:
    builds = []
    found = {"localhost"}
    monkeypatch.setattr(hosts, "gather", lambda extra: builds.append(1) or set(found) | set(extra))
    clock = [1000.0]
    monkeypatch.setattr(hosts.time, "monotonic", lambda: clock[0])
    allow = Allowlist(lambda: ["cfg.example"], ttl=300, miss_refresh=30)
    assert allow.allows("") is True  # no Host: not a browser
    assert allow.allows("cfg.example:7601") and len(builds) == 1
    found.add("new.example")  # a tailnet name that appeared after the last build
    clock[0] += 10
    assert allow.allows("new.example") is False and len(builds) == 1  # too soon to rebuild
    clock[0] += 25
    assert allow.allows("new.example") is True and len(builds) == 2
    assert allow.allows("evil.example") is False and len(builds) == 2


def test_a_slow_rebuild_never_holds_up_known_hosts(monkeypatch) -> None:
    """A rebuild can take seconds (DNS). Requests for known names must not wait for it."""
    release = threading.Event()
    started = threading.Event()
    calls = []

    def slow_gather(extra):
        calls.append(1)
        if len(calls) > 1:
            started.set()
            release.wait(10)
        return {"localhost", "box.example"} | ({"late.example"} if len(calls) > 1 else set())

    monkeypatch.setattr(hosts, "gather", slow_gather)
    clock = [1000.0]
    monkeypatch.setattr(hosts.time, "monotonic", lambda: clock[0])
    allow = Allowlist(lambda: [], ttl=300, miss_refresh=30)
    allow.refresh()
    clock[0] += 301  # stale: the next request starts a background rebuild
    t0 = __import__("time").perf_counter()
    assert allow.allows("box.example") is True
    assert started.wait(5)
    late = []
    waiter = threading.Thread(target=lambda: late.append(allow.allows("late.example")))
    waiter.start()  # a name only the running rebuild will find waits for that rebuild
    for _ in range(20):  # the rebuild is still running
        assert allow.allows("box.example:7601") is True
    assert __import__("time").perf_counter() - t0 < 2
    assert late == []
    release.set()
    waiter.join(5)
    assert late == [True] and len(calls) == 2  # one rebuild, shared


def test_a_failing_rebuild_keeps_the_last_list(monkeypatch) -> None:
    state = {"fail": False}

    def gather_or_fail(extra):
        if state["fail"]:
            raise OSError("resolver down")
        return {"box.example"}

    monkeypatch.setattr(hosts, "gather", gather_or_fail)
    clock = [1000.0]
    monkeypatch.setattr(hosts.time, "monotonic", lambda: clock[0])
    allow = Allowlist(lambda: [], ttl=300, miss_refresh=30)
    allow.refresh()
    state["fail"] = True
    clock[0] += 60
    assert allow.allows("box.example") is True
    assert allow.allows("new.example") is False  # its rebuild failed; no exception escapes


@pytest.fixture()
def server(tmp_path, monkeypatch):
    _fake_machine(monkeypatch)
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(serve_mod, "HOSTS", Allowlist(lambda: []))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None


def _ask(port, method, path, host, headers=None, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host)
        for k, v in (headers or {}).items():
            conn.putheader(k, v)
        data = body.encode() if body is not None else b""
        conn.putheader("Content-Length", str(len(data)))
        conn.endheaders(data)
        resp = conn.getresponse()
        return resp.status, resp.getheader("Content-Type") or "", resp.read().decode()
    finally:
        conn.close()


@pytest.mark.parametrize("host", [
    "127.0.0.1:{port}", "localhost:{port}", "[::1]:{port}",
    "100.64.0.9:{port}",            # tailnet address
    "box.example.ts.net:{port}",    # MagicDNS name
    "Box.Example.ts.net.:{port}",   # same, as some browsers spell it
    "box:{port}",                   # short machine name
    "192.0.2.5:{port}",             # LAN interface address
])
def test_own_names_keep_working(server, host) -> None:
    code, _, _ = _ask(server, "GET", "/api/status", host.format(port=server))
    assert code == 200, host


@pytest.mark.parametrize("method", ["GET", "HEAD", "POST", "PUT", "DELETE"])
def test_foreign_host_is_refused_in_plain_words(server, method) -> None:
    code, ctype, text = _ask(server, method, "/api/status", f"rebind.example:{server}",
                             {"Content-Type": "application/json"}, "{}" if method != "GET" and method != "HEAD" else None)
    assert code == 421, (method, code)
    assert ctype.startswith("text/plain")
    if method != "HEAD":
        assert "allowed_hosts under [controller]" in text and "rebind.example" in text


def test_page_itself_is_refused_for_a_foreign_host(server) -> None:
    code, _, text = _ask(server, "GET", "/overview", "rebind.example")
    assert code == 421 and "<html" not in text.lower()


def test_cross_site_fetch_metadata_refuses_a_write(server) -> None:
    host = f"127.0.0.1:{server}"
    head = {"Content-Type": "application/json", "Sec-Fetch-Site": "cross-site"}
    code, _, text = _ask(server, "POST", "/api/lock", host, head, "{}")
    assert code == 403 and "Sec-Fetch-Site" in text
    for site in ("same-origin", "none"):
        code, _, _ = _ask(server, "POST", "/api/job/pause", host, dict(head, **{"Sec-Fetch-Site": site}), '{"paused": false}')
        assert code == 200, site
