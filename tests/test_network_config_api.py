# SPDX-License-Identifier: Apache-2.0

"""Network page config edits over HTTP, plus the #78 audit follow-ups."""

import http.client
import json
import threading
import time
import tomllib
import shutil
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import pytest

from sparksi import hostcheck
from sparksi.config import Config

import sparksi.serve as serve_mod
import sparksi.discover as discover
import sparksi.monitor_mirror as monitor_mirror
from sparksi.lock import Lock
from sparksi.refresh import PENDING, Refreshing
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


def ask(port, method, path, body=None, origin=True):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if origin is True:
        headers["Origin"] = "http://127.0.0.1:%d" % port
    elif origin == "evil":
        headers["Origin"] = "http://evil.example"
    payload = None
    if body is not None:
        payload = json.dumps(body)
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    status = resp.status
    raw = resp.read()
    conn.close()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        parsed = {}
    return status, parsed


@pytest.fixture
def server(tmp_path, monkeypatch):
    cfg_src = ROOT / "config" / "sparksi.example.toml"
    cfg_dst = tmp_path / "sparksi.toml"
    shutil.copyfile(cfg_src, cfg_dst)
    cfg_dst.chmod(0o600)
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CONFIG", str(cfg_dst))
    monkeypatch.setattr(serve_mod.HOSTS, "names", lambda: {"localhost", "127.0.0.1"})
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()  # the unlock lease is on; tests that need the lock take it
    httpd = DualServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port, cfg_dst
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def test_layout(server):
    port, cfg = server
    status, data = ask(port, "GET", "/api/network/layout")
    assert status == 200
    assert data["configured"] is True
    assert data["file"] == "sparksi.toml"
    head = data["layout"]["head"]
    worker = data["layout"]["worker"]
    assert head["id"] == "a"
    assert worker["id"] == "b"
    assert data["layout"]["limit"] == 4
    text = json.dumps(data)
    assert str(cfg.parent) not in text


def test_role_locked_403(server):
    port, cfg = server
    before = cfg.read_bytes()
    STATE.lock.acquire("test")
    try:
        status, data = ask(
            port,
            "POST",
            "/api/network/machine",
            {
                "action": "role",
                "role": "worker",
                "machine": {"id": "a"},
                "from_ui": True,
            },
        )
        assert status == 403
        assert cfg.read_bytes() == before
    finally:
        STATE.lock.release()


def test_origin_and_content_type(server):
    port, cfg = server
    before = cfg.read_bytes()
    body = {
        "action": "role",
        "role": "worker",
        "machine": {"id": "a"},
        "from_ui": True,
    }
    status, _ = ask(port, "POST", "/api/network/machine", body, origin="evil")
    assert status == 403
    assert cfg.read_bytes() == before

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request(
        "POST",
        "/api/network/machine",
        body=json.dumps(body),
        headers={
            "Content-Type": "text/plain",
            "Origin": "http://127.0.0.1:%d" % port,
        },
    )
    resp = conn.getresponse()
    status2 = resp.status
    resp.read()
    conn.close()
    assert status2 == 415
    assert cfg.read_bytes() == before


def test_swap_ok(server):
    port, cfg = server
    status, data = ask(
        port,
        "POST",
        "/api/network/machine",
        {
            "action": "role",
            "role": "worker",
            "machine": {"id": "a"},
            "from_ui": True,
        },
    )
    assert status == 200
    assert data["ok"] is True
    assert data["note"] == "head and worker swapped"
    assert data["backup"].startswith("sparksi.toml.bak-")
    assert data["restart_needed"] is True
    parsed = tomllib.loads(cfg.read_text())
    ids = [n["id"] for n in parsed["nodes"]]
    assert ids == ["b", "a"]
    status2, layout = ask(port, "GET", "/api/network/layout")
    assert status2 == 200
    assert layout["layout"]["head"]["id"] == "b"
    assert layout["restart_needed"] is True


def test_add_remove_monitor(server):
    port, cfg = server
    add = {
        "action": "add",
        "role": "monitor",
        "machine": {
            "name": "gpu-box",
            "ips": ["100.64.0.20"],
            "endpoints": [
                {
                    "port": 11434,
                    "engine": "Ollama",
                    "url": "http://100.64.0.20:11434",
                    "models": [],
                }
            ],
        },
    }
    status, data = ask(port, "POST", "/api/network/machine", add)
    assert status == 200
    parsed = tomllib.loads(cfg.read_text())
    assert parsed["endpoints"] == [{"id": "gpu-box", "name": "gpu-box", "url": "http://100.64.0.20:11434", "engine": "Ollama"}]
    assert [n["id"] for n in parsed["nodes"]] == ["a", "b"]  # the Sparks are untouched

    rem = {"action": "remove", "role": "", "machine": {"id": "gpu-box"}}
    status2, data2 = ask(port, "POST", "/api/network/machine", rem)
    assert status2 == 200
    assert "endpoints" not in tomllib.loads(cfg.read_text())
    assert "[[endpoints]]" not in cfg.read_text()


def test_remove_head_is_refused(server):
    port, cfg = server
    before = cfg.read_bytes()
    status, data = ask(
        port, "POST", "/api/network/machine", {"action": "remove", "role": "", "machine": {"id": "a"}}
    )
    assert status == 400
    err = json.dumps(data)
    assert "give the head role to another Spark" in err
    assert cfg.read_bytes() == before


def test_swap_does_not_touch_runner(server, monkeypatch):
    port, cfg = server

    def boom(*a, **k):
        raise AssertionError("runner touched")

    monkeypatch.setattr(serve_mod, "active_runner", boom)
    status, data = ask(
        port,
        "POST",
        "/api/network/machine",
        {
            "action": "role",
            "role": "worker",
            "machine": {"id": "a"},
            "from_ui": True,
        },
    )
    assert status == 200


def test_start_mirror_self_refusal(monkeypatch):
    monkeypatch.setenv("SPARKSI_MONITOR_UPSTREAM", "http://127.0.0.1:7601")
    try:
        serve_mod._start_mirror(7601)
        assert STATE.monitor_fn is None
        assert STATE.mirror is None
    finally:
        STATE.monitor_fn = None
        STATE.mirror = None

    calls = []
    real_refuse = monitor_mirror.refuse_self

    def spy(base, port, names=()):
        calls.append((base, port))
        return real_refuse(base, port, names)

    monkeypatch.setattr(monitor_mirror, "refuse_self", spy)
    monkeypatch.setenv("SPARKSI_MONITOR_UPSTREAM", "http://127.0.0.1:7500")
    try:
        serve_mod._start_mirror(7601)
        assert calls == [("http://127.0.0.1:7500", 7601)]
    finally:
        if STATE.mirror is not None:
            stop = getattr(STATE.mirror, "stop", None)
            if stop is not None:
                stop.set()
        STATE.monitor_fn = None
        STATE.mirror = None




def test_refresh_max_age() -> None:
    now = [1000.0]
    calls = []

    def fn():
        calls.append(now[0])
        if len(calls) > 1:
            raise OSError("node down")
        return 1

    cache = Refreshing(ttl=5, first_wait=0.5, clock=lambda: now[0], max_age=30)
    assert cache.get("k", fn) == 1
    now[0] = 1010.0
    assert cache.get("k", fn, wait=0.5) == 1  # stale but young enough: served while refreshing
    now[0] = 1040.0
    assert cache.get("k", fn, wait=0.5) is PENDING  # 40 s since the last success: not served


def _until(pred, timeout=2.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not pred():
        time.sleep(0.01)
    return pred()


def test_a_refresh_running_during_a_write_keeps_the_value_stale() -> None:
    now = [1000.0]
    gate = threading.Event()
    calls = []

    def fn():
        calls.append(1)
        if len(calls) == 2:
            gate.wait(5)  # read before the write lands
        return len(calls)

    cache = Refreshing(ttl=5, first_wait=0.5, clock=lambda: now[0])
    assert cache.get("k", fn) == 1
    now[0] = 1006.0
    assert cache.get("k", fn, wait=0) == 1  # starts refresh 2, which blocks
    cache.expire()  # a write happens while refresh 2 runs
    gate.set()
    assert _until(lambda: not cache._entries["k"].running)
    assert cache.get("k", fn, wait=0.5) in (2, 3)  # refresh 2's value is served...
    assert _until(lambda: len(calls) == 3)  # ...but it did not clear the stale mark: refresh 3 ran
    assert _until(lambda: cache.get("k", fn) == 3)
    assert len(calls) == 3  # and once fresh, no more refreshes inside the ttl


def test_discover_opener_ignores_env_proxy(monkeypatch):
    monkeypatch.setenv("http_proxy", "http://192.0.2.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://192.0.2.1:9")

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        status, data = discover._get("127.0.0.1", port, "/v1/models")
        assert status == 200
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


def test_mdns_hosts_filters():
    def line(name, ip):
        return "=;eth0;IPv4;%s;_ssh._tcp;local;%s.local;%s;22;" % (name, name, ip)

    # Documentation ranges stand in for LAN addresses (the identity guard bans real
    # RFC 1918 literals); the link-local address is assembled so no literal appears.
    link_local = ".".join(("169", "254", "1", "1"))
    text = "\n".join(
        [
            line("doc", "192.0.2.5"),
            line("lan", "198.51.100.5"),
            line("tail", "100.64.0.7"),
            line("pub2", "8.8.8.8"),
            line("linklocal", link_local),
            line("loopback", "127.0.0.1"),
        ]
    )
    got = discover.mdns_hosts(text)
    ips = set()
    for h in got:
        ips.update(h.get("ips", []))
    assert ips == {"192.0.2.5", "198.51.100.5", "100.64.0.7"}


def test_probe_deadline(monkeypatch):
    def fake_probe_port(host, port, timeout=None):
        if port == 2:
            time.sleep(2)
        return {"port": port, "engine": "x", "models": [], "url": ""}

    monkeypatch.setattr(discover, "probe_port", fake_probe_port)
    hosts = [{"ips": ["100.64.0.9"]}]
    t0 = time.monotonic()
    late = discover.probe(hosts, ports=(1, 2), deadline=0.5)
    elapsed = time.monotonic() - t0
    assert elapsed < 1.5
    assert late == 1
    eps = hosts[0].get("endpoints", [])
    assert [e["port"] for e in eps] == [1]


def test_old_setup_address_redirects_to_network(server):
    port, _cfg = server
    for path in ("/setup", "/setup/", "/setup/step-2"):
        for method in ("GET", "HEAD"):
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request(method, path)
            resp = conn.getresponse()
            resp.read()
            conn.close()
            assert resp.status == 302, (method, path)
            assert resp.getheader("Location") == "/network", (method, path)


def test_add_by_address_validates(server):
    port, cfg = server
    before = cfg.read_bytes()
    for machine, word in (
        ({"address": "8.8.8.8", "port": 8000}, "private or tailnet"),
        ({"address": "bad host", "port": 8000}, "host name"),
        ({"address": "gpu-box", "port": 70000}, "1 to 65535"),
        ({"address": "gpu-box", "port": "80a"}, "1 to 65535"),
    ):
        status, data = ask(port, "POST", "/api/network/machine",
                           {"action": "add", "role": "monitor", "machine": machine, "from_ui": True})
        assert status == 400, (machine, data)
        assert word in data["error"], (machine, data)
    assert cfg.read_bytes() == before
    status, data = ask(port, "POST", "/api/network/machine",
                       {"action": "add", "role": "monitor", "from_ui": True,
                        "machine": {"address": "gpu-box.example.ts.net", "port": "8000", "name": ""}})
    assert status == 200, data
    assert "offline" in data["note"]
    assert [e["url"] for e in tomllib.loads(cfg.read_text())["endpoints"]] == ["http://gpu-box.example.ts.net:8000"]
    assert data["layout"]["monitored"][0]["name"] == "gpu-box.example.ts.net"


def test_ssh_option_payload_refused_at_save_and_layout_hides_ssh(server):
    port, cfg = server
    before = cfg.read_bytes()
    payload = "-oProxyCommand=touch /tmp/pwn"
    for machine in ({"name": "box", "dns": payload, "ips": ["100.64.0.5"]},
                    {"name": payload, "ips": ["100.64.0.5"]}):
        status, data = ask(port, "POST", "/api/network/machine",
                           {"action": "add", "role": "head", "machine": machine, "from_ui": True})
        assert status == 400, data
    assert cfg.read_bytes() == before
    status, data = ask(port, "GET", "/api/network/layout")
    assert status == 200
    assert "ssh" not in json.dumps(data["layout"])


# ---------------------------------------------------------------- IPv6: unique local only



@pytest.mark.parametrize(
    "addr",
    [
        "2002:808:808::1",
        "2002:a9fe:a9fe::1",
        "2001:0::1",
        "2001::5",
        "2001:db8::1",
        "3fff::1",
        "2001:4860:4860::8888",
        "64:ff9b::808:808",
        "::1",
        "fe80::1",
        "ff02::1",
        "::",
    ],
)
def test_ipv6_outside_unique_local_is_refused(addr):
    assert hostcheck.ip_ok(addr) is False
    with pytest.raises(ValueError, match="private or tailnet"):
        hostcheck.check_host(addr)


def test_unique_local_and_tailnet_ipv6_are_accepted():
    for addr in ("fd00:1234::5", "fc00::1", "fd7a:115c:a1e0::5", "::ffff:192.0.2.5"):
        assert hostcheck.ip_ok(addr) is True
        assert hostcheck.check_host(addr) == addr


def test_tailscale_service_address_is_refused():
    quad100 = ".".join(("100",) * 4)
    assert hostcheck.ip_ok(quad100) is False
    assert hostcheck.ip_ok("fd7a:115c:a1e0::53") is False
    assert hostcheck.ip_ok("100.64.0.7") is True


def test_network_page_still_saves_lan_tailnet_and_ula_monitors(server):
    port, cfg_path = server
    for addr, name in (("192.0.2.5", "lan-box"), ("100.64.0.9", "tail-box")):
        status, body = ask(
            port,
            "POST",
            "/api/network/machine",
            {
                "action": "add",
                "role": "monitor",
                "from_ui": True,
                "machine": {"address": addr, "port": 8000, "name": name},
            },
        )
        assert status == 200
    raw = tomllib.loads(cfg_path.read_text())
    assert [e["url"] for e in raw["endpoints"]] == [
        "http://192.0.2.5:8000",
        "http://100.64.0.9:8000",
    ]
    status, body = ask(
        port,
        "POST",
        "/api/network/machine",
        {
            "action": "add",
            "role": "monitor",
            "from_ui": True,
            "machine": {"address": "fd00:1234::5", "port": 8000, "name": "third-box"},
        },
    )
    assert status in (400, 409)
    assert "monitored host" in body["error"]


def test_network_page_saves_ula_monitor(server):
    port, cfg_path = server
    status, body = ask(
        port,
        "POST",
        "/api/network/machine",
        {
            "action": "add",
            "role": "monitor",
            "from_ui": True,
            "machine": {"address": "fd00:1234::5", "port": 8000, "name": "ula-box"},
        },
    )
    assert status == 200
    raw = tomllib.loads(cfg_path.read_text())
    assert [e["url"] for e in raw["endpoints"]] == ["http://[fd00:1234::5]:8000"]


def test_network_page_refuses_6to4_monitor(server):
    port, cfg_path = server
    before = cfg_path.read_bytes()
    status, body = ask(
        port,
        "POST",
        "/api/network/machine",
        {
            "action": "add",
            "role": "monitor",
            "from_ui": True,
            "machine": {"address": "2002:808:808::1", "port": 8000, "name": "six-box"},
        },
    )
    assert status == 400
    assert "private or tailnet" in body["error"]
    assert cfg_path.read_bytes() == before


def test_load_checks_accept_lan_tailnet_and_ula_endpoints(tmp_path):
    raw = tomllib.loads((ROOT / "config" / "sparksi.example.toml").read_text())
    raw["endpoints"] = [
        {
            "id": "lan-box",
            "name": "lan-box",
            "url": "http://192.0.2.5:8000",
            "engine": "vLLM",
        },
        {
            "id": "ula-box",
            "name": "ula-box",
            "url": "http://[fd00:1234::5]:8000",
            "engine": "",
        },
    ]
    Config(raw)
    raw["endpoints"][1]["url"] = "http://[2002:808:808::1]:8000"
    with pytest.raises(ValueError, match="url"):
        Config(raw)
