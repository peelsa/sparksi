# SPDX-License-Identifier: Apache-2.0

"""Discover: read-only host gathering, port fingerprinting and roles, with canned inputs and local fake servers."""

from __future__ import annotations

import http.client
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import sparksi.discover as discover

TS = json.dumps({
    "Self": {"HostName": "ctl", "DNSName": "ctl.example.ts.net.", "TailscaleIPs": ["100.64.0.1", "fd7a:115c:a1e0::1"], "OS": "linux"},
    "Peer": {
        "k1": {"HostName": "spark-one", "DNSName": "spark-one.example.ts.net.", "TailscaleIPs": ["100.64.0.11"], "OS": "linux",
               "Online": True, "CurAddr": "192.0.2.11:41641", "Addrs": ["192.0.2.11:41641", "192.88.99.1:41641"]},
        "k2": {"HostName": "gpu-box", "DNSName": "gpu-box.example.ts.net.", "TailscaleIPs": ["100.64.0.20"], "OS": "linux", "Online": True},
        "k3": {"HostName": "laptop", "TailscaleIPs": ["100.64.0.30"], "Online": False},
    },
})
NEIGH = json.dumps([
    {"dst": "192.0.2.11", "dev": "eth0", "state": ["REACHABLE"]},
    {"dst": "192.0.2.40", "dev": "eth0", "state": ["STALE"]},
    {"dst": "192.0.2.41", "dev": "eth0", "state": ["FAILED"]},
    {"dst": "192.0.2.99", "dev": "docker0", "state": ["REACHABLE"]},
    {"dst": "fe80::1", "dev": "eth0", "state": ["REACHABLE"]},
])
MDNS = "\n".join([
    "+;eth0;IPv4;nas;_ssh._tcp;local",
    "=;eth0;IPv4;nas;_ssh._tcp;local;nas.local;192.0.2.40;22;",
    "=;eth0;IPv6;nas;_ssh._tcp;local;nas.local;fe80::2;22;",
])


@pytest.fixture(autouse=True)
def _fresh():
    discover.reset_for_tests()
    yield
    discover.reset_for_tests()


def test_tailscale_hosts_online_only_with_lan_endpoint() -> None:
    hosts = discover.tailscale_hosts(TS)
    names = [h["name"] for h in hosts]
    assert names == ["ctl", "spark-one", "gpu-box"]  # offline laptop skipped
    assert hosts[0]["self"] is True and hosts[0]["dns"] == "ctl.example.ts.net"
    spark = hosts[1]
    assert spark["ips"] == ["100.64.0.11", "192.0.2.11"]  # private LAN endpoint kept, public one dropped


def test_neighbors_and_mdns_filtering() -> None:
    assert [h["ips"] for h in discover.neighbor_hosts(NEIGH)] == [["192.0.2.11"], ["192.0.2.40"]]
    mdns = discover.mdns_hosts(MDNS)
    assert mdns == [{"name": "nas", "dns": "nas.local", "ips": ["192.0.2.40"], "os": "", "sources": ["mdns"], "self": False}]


def test_merge_joins_a_neighbor_with_its_tailnet_peer() -> None:
    hosts = discover.merge([discover.tailscale_hosts(TS), discover.neighbor_hosts(NEIGH), discover.mdns_hosts(MDNS)])
    by = {h["name"] or h["ips"][0]: h for h in hosts}
    assert set(by) == {"ctl", "spark-one", "gpu-box", "nas"}
    assert by["spark-one"]["sources"] == ["lan", "tailscale"]
    assert by["nas"]["sources"] == ["lan", "mdns"]
    assert hosts[0]["self"] is True


def test_probe_uses_one_address_per_host_preferring_the_tailnet(monkeypatch) -> None:
    asked = []
    monkeypatch.setattr(discover, "probe_port", lambda ip, port: asked.append((ip, port)) or None)
    hosts = discover.merge([discover.tailscale_hosts(TS), discover.neighbor_hosts(NEIGH)])
    discover.probe(hosts, ports=(8014,))
    assert sorted(asked) == sorted([("100.64.0.1", 8014), ("100.64.0.11", 8014), ("100.64.0.20", 8014), ("192.0.2.40", 8014)])


# --- fingerprinting against local fake servers ---

class _Fake(BaseHTTPRequestHandler):
    routes: dict = {}
    seen: list = []

    def log_message(self, *a):
        return

    def do_GET(self):  # noqa: N802
        type(self).seen.append(("GET", self.server.server_address[1], self.path, self.headers.get("Authorization")))
        status, body = type(self).routes.get((self.server.server_address[1], self.path), (404, {"error": "no"}))
        if status == 302:
            self.send_response(302)
            self.send_header("Location", body)
            self.end_headers()
            return
        raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        type(self).seen.append(("POST", self.server.server_address[1], self.path, None))
        self.send_response(405)
        self.end_headers()


@pytest.fixture()
def fakes():
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), _Fake) for _ in range(6)]
    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Fake.seen = []
    ports = [srv.server_address[1] for srv in servers]
    vllm, ollama, health, locked, redirect, sparksi = ports
    _Fake.routes = {
        (vllm, "/v1/models"): (200, {"data": [{"id": "example/model-a", "owned_by": "vllm"}]}),
        (ollama, "/api/tags"): (200, {"models": [{"name": "small:7b"}, {"name": "tiny:1b"}]}),
        (health, "/health"): (200, "OK"),
        (locked, "/v1/models"): (401, {"error": "key"}),
        (redirect, "/v1/models"): (302, f"http://127.0.0.1:{vllm}/v1/models"),
        (sparksi, "/api/config"): (200, {"controller": {"name": "ctl"}}),
    }
    try:
        yield ports
    finally:
        for srv in servers:
            srv.shutdown()


def test_fingerprints_and_one_get_for_a_vllm_port(fakes) -> None:
    vllm, ollama, health, locked, redirect, sparksi = fakes
    got = discover.probe_port("127.0.0.1", vllm)
    assert got == {"port": vllm, "engine": "vLLM", "models": ["example/model-a"], "url": f"http://127.0.0.1:{vllm}"}
    assert [s for s in _Fake.seen if s[1] == vllm] == [("GET", vllm, "/v1/models", None)]  # exactly one GET, no key
    assert discover.probe_port("127.0.0.1", ollama)["models"] == ["small:7b", "tiny:1b"]
    assert discover.probe_port("127.0.0.1", health)["engine"] == "HTTP service"
    assert discover.probe_port("127.0.0.1", locked)["engine"] == "needs a key"
    before = len([s for s in _Fake.seen if s[1] == vllm])
    assert discover.probe_port("127.0.0.1", redirect)["engine"] == "open port"
    assert len([s for s in _Fake.seen if s[1] == vllm]) == before  # the redirect was not followed
    assert all(s[0] == "GET" and s[3] is None for s in _Fake.seen)  # never a POST, never credentials


def test_sparksi_controller_port_is_asked_only_for_its_config(fakes, monkeypatch) -> None:
    sparksi = fakes[5]
    monkeypatch.setattr(discover, "PATHS", ("/v1/models",))
    # 7500 is special-cased; emulate it by probing the fake with the config path directly.
    status, body = discover._get("127.0.0.1", sparksi, "/api/config")
    assert discover.classify(7500, "/api/config", status, body)["engine"] == "Sparksi"
    assert discover.probe_port("127.0.0.1", 1) is None  # closed port: no GET at all


def test_roles_and_proposals() -> None:
    known = {"spark-one": "a", "100.64.0.11": "a"}
    hosts = discover.merge([discover.tailscale_hosts(TS), discover.neighbor_hosts(NEIGH), discover.mdns_hosts(MDNS)])
    by = {h["name"]: h for h in hosts}
    by["gpu-box"]["endpoints"] = [{"port": 11434, "engine": "Ollama", "models": ["small:7b"], "url": "http://100.64.0.20:11434"}]
    by["nas"]["endpoints"] = [{"port": 5000, "engine": "open port", "models": [], "url": "http://192.0.2.40:5000"}]
    roles = {}
    for host in hosts:
        host.setdefault("endpoints", [])
        host["role"], host["why"] = discover.role_for(host, known)
        roles[host["name"]] = host["role"]
    assert roles == {"ctl": "controller", "spark-one": "spark", "gpu-box": "inference", "nas": "machine"}
    assert by["spark-one"]["why"].startswith("configured Spark a")
    assert discover.role_for({"name": "dgx-rack", "ips": []}, {})[1] == "name looks like a DGX Spark"
    spark = discover.proposal(by["spark-one"])
    assert spark.splitlines() == ['[[nodes]]', 'id = "spark-one"', 'name = "spark-one"', 'ssh = "spark-one"', 'tailscale = "100.64.0.11"']
    other = discover.proposal(by["gpu-box"])
    assert other.startswith("# A monitored inference host") and 'url = "http://100.64.0.20:11434"' in other
    assert discover.proposal(by["nas"]) == ""


def test_scan_caches_and_refuses_a_storm() -> None:
    first = discover.scan({}, tailscale_text=TS, neigh_text=NEIGH, mdns_text=MDNS, ports=())
    assert first["scanned_at"] and len(first["hosts"]) == 4 and "too_soon" not in first
    assert [h["role"] for h in first["hosts"]] == ["controller", "spark", "machine", "machine"]
    assert [h["name"] for h in first["hosts"]][2:] == ["gpu-box", "nas"]  # named before unnamed, then by name
    again = discover.scan({}, tailscale_text=TS, neigh_text="[]", mdns_text="", ports=())
    assert again["too_soon"] is True and len(again["hosts"]) == 4
    assert discover.cached()["scanned_at"] == first["scanned_at"]


def test_discover_api_get_is_cached_and_post_is_guarded(monkeypatch, tmp_path) -> None:
    import sparksi.serve as serve_mod
    from sparksi.lock import Lock
    from sparksi.serve import STATE, DualServer, Handler

    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    discover.reset_for_tests()
    calls = []

    def fake_do_scan(known, **kw):
        calls.append(known)
        out = {
            "scanned_at": "2026-01-01T00:00:00Z",
            "hosts": [],
            "took_s": 0.1,
        }
        discover._CACHE.clear()
        discover._CACHE.update(out)
        return out

    monkeypatch.setattr(serve_mod.discover, "_do_scan", fake_do_scan)
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]

    def ask(method, ctype=None, body=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Content-Type": ctype} if ctype else {}
        conn.request(method, "/api/discover", body=body, headers=headers)
        resp = conn.getresponse()
        data = json.loads(resp.read().decode() or "{}")
        conn.close()
        return resp.status, data

    try:
        code, data = ask("GET")
        assert code == 200 and data["scanned_at"] is None and calls == []  # GET never scans
        assert ask("POST", "text/plain", "{}")[0] == 415
        t0 = time.monotonic()
        code, data = ask("POST", "application/json", "{}")
        elapsed = time.monotonic() - t0
        assert code == 202 and data["running"] is True and elapsed < 1.0
        assert _wait_until(lambda: ask("GET")[1]["running"] is False, timeout=5, interval=0.1)  # the page polls GET
        code, data = ask("GET")
        assert code == 200 and data["took_s"] == 0.1
        assert len(calls) == 1
        assert isinstance(calls[0], dict)
    finally:
        httpd.shutdown()
        STATE.runner = None


def _fake_scan_factory(calls, started, release, *, result=None, error=None):
    """Build a fake _do_scan that blocks on an Event until released."""

    def fake(known, **kw):
        calls.append((known, kw))
        try:
            started.set()
            release.wait(timeout=5)
        finally:
            release.set()
        if error is not None:
            raise error
        out = {
            "scanned_at": "2026-01-01T00:00:00Z",
            "hosts": list(result.get("hosts", [])) if result else [],
            "took_s": 0.1,
        }
        discover._CACHE.clear()
        discover._CACHE.update(out)
        return out

    return fake


def _wait_until(pred, timeout=2.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def test_start_returns_202_at_once_and_runs_one_scan_in_the_background(monkeypatch):
    discover.reset_for_tests()
    calls = []
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        discover, "_do_scan", _fake_scan_factory(calls, started, release)
    )
    try:
        t0 = time.monotonic()
        code, body = discover.start({"a": "1"}, depth=2)
        elapsed = time.monotonic() - t0
        assert (code, body["running"], body["started"]) == (202, True, True)
        assert elapsed < 0.2
        assert started.wait(timeout=2)
        assert discover.cached()["running"] is True

        code2, body2 = discover.start({"a": "1"})
        assert code2 == 202 and body2["busy"] is True and body2["running"] is True
        assert len(calls) == 1

        release.set()
        assert _wait_until(lambda: discover.cached()["running"] is False)
        cached = discover.cached()
        assert cached["scanned_at"] == "2026-01-01T00:00:00Z"
        assert cached["hosts"] == []
        assert cached["took_s"] == 0.1
        assert len(calls) == 1
    finally:
        release.set()


def test_start_keeps_the_cooldown(monkeypatch):
    discover.reset_for_tests()
    calls = []
    started = threading.Event()
    release = threading.Event()
    monkeypatch.setattr(
        discover, "_do_scan", _fake_scan_factory(calls, started, release)
    )
    try:
        code, body = discover.start({})
        assert code == 202
        assert started.wait(timeout=2)
        release.set()
        assert _wait_until(lambda: discover.cached()["running"] is False)

        code2, body2 = discover.start({})
        assert code2 == 200
        assert body2["too_soon"] is True
        assert body2["running"] is False
        assert len(calls) == 1
    finally:
        release.set()


def test_a_failing_scan_releases_the_lock_and_reports(monkeypatch):
    discover.reset_for_tests()
    calls = []
    started = threading.Event()
    release = threading.Event()
    err = RuntimeError("boom")
    monkeypatch.setattr(
        discover, "_do_scan", _fake_scan_factory(calls, started, release, error=err)
    )
    try:
        code, body = discover.start({})
        assert code == 202 and body["running"] is True
        assert started.wait(timeout=2)
        release.set()
        assert _wait_until(lambda: discover.cached()["running"] is False)
        cached = discover.cached()
        assert cached["running"] is False
        assert cached["error"] == "scan failed"

        code2, body2 = discover.start({})
        assert code2 == 200 and body2["too_soon"] is True
        assert len(calls) == 1
    finally:
        release.set()
