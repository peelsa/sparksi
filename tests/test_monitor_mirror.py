# SPDX-License-Identifier: Apache-2.0

"""The optional monitor mirror: GET only, short timeout, never blocks a request."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen

import pytest

import sparksi.monitor_mirror as mm
import sparksi.serve as serve_mod
from sparksi.history import History, sample_card
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

CARD = {
    "hosts": [
        {"id": "head", "fresh": True, "age": 1, "gpu_pct": 41, "mem_pct": 88, "mem_used_gib": 105.0, "power_w": 90},
        {"id": "worker", "fresh": True, "age": 1, "gpu_pct": 39, "mem_pct": 86, "mem_used_gib": 103.0, "power_w": 85},
    ],
    "loaded": [{"host": "head", "port": 8014, "served": "example-served", "state": "UP", "span": ["worker"], "tok_s": 12.5}],
    "history": {"head": {"gpu_pct": 1}},
}


class _Upstream(BaseHTTPRequestHandler):
    seen: list[str] = []
    mode = "ok"

    def log_message(self, *args) -> None:  # noqa: ARG002
        pass

    def _any(self) -> None:
        type(self).seen.append(f"{self.command} {self.path}")
        if type(self).mode == "slow":
            time.sleep(5)
        if type(self).mode == "redirect":
            self.send_response(302)
            self.send_header("Location", "/api/runs")
            self.end_headers()
            return
        body = json.dumps(CARD).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = do_POST = do_PUT = do_DELETE = _any


@pytest.fixture()
def upstream():
    _Upstream.seen = []
    _Upstream.mode = "ok"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


@pytest.fixture(autouse=True)
def _no_mirror():
    yield
    if STATE.mirror is not None:
        STATE.mirror.stop.set()
    STATE.mirror = None
    STATE.monitor_fn = None
    STATE.observe_fn = None


@pytest.mark.parametrize("raw", [
    "http://127.0.0.1:7500", "http://localhost:7500/", "http://100.64.0.1:7500",
    "http://controller.example.ts.net:7500", "http://[::1]:7500",
])
def test_loopback_and_tailnet_upstreams_are_allowed(raw) -> None:
    assert mm.upstream_url(raw).startswith("http://")


@pytest.mark.parametrize("raw", [
    "https://127.0.0.1:7500", "http://192.0.2.10:7500", "http://example.com:7500",
    "http://user:pw@127.0.0.1:7500", "http://127.0.0.1:7500/api/runs", "http://127.0.0.1:7500/?x=1",
    "ftp://127.0.0.1", "",
])
def test_other_upstreams_are_refused(raw) -> None:
    with pytest.raises(ValueError):
        mm.upstream_url(raw)


def test_an_upstream_on_our_own_loopback_port_is_refused() -> None:
    with pytest.raises(ValueError):
        mm.refuse_self("http://127.0.0.1:7601", 7601)
    mm.refuse_self("http://127.0.0.1:7500", 7601)
    mm.refuse_self("http://100.64.0.1:7601", 7601)  # another machine on the tailnet


def test_the_mirror_only_sends_get_to_the_monitor_path(upstream) -> None:
    mirror = mm.Mirror(upstream, lambda: {"hosts": [], "loaded": []})
    assert mirror.poll_once() and mirror.poll_once()
    assert _Upstream.seen == ["GET /api/monitor", "GET /api/monitor"]
    card = mirror.card()
    assert card["mirrored"] is True and card["loaded"][0]["port"] == 8014
    assert "history" not in card  # this controller adds its own


def test_a_redirect_is_refused(upstream) -> None:
    _Upstream.mode = "redirect"
    mirror = mm.Mirror(upstream, lambda: {"local": True})
    assert mirror.poll_once() is False
    assert _Upstream.seen == ["GET /api/monitor"]  # the redirect target was not fetched
    assert mirror.card() == {"local": True}


def test_a_hung_upstream_times_out_and_requests_never_wait(upstream, monkeypatch) -> None:
    _Upstream.mode = "slow"
    monkeypatch.setattr(mm, "TIMEOUT_S", 0.3)
    mirror = mm.Mirror(upstream, lambda: {"local": True}, fetch=lambda base: mm.fetch_card(base, timeout=mm.TIMEOUT_S))
    t0 = time.perf_counter()
    assert mirror.poll_once() is False
    assert time.perf_counter() - t0 < 2
    mirror.start()
    t0 = time.perf_counter()
    for _ in range(20):
        assert mirror.card() == {"local": True}
    assert time.perf_counter() - t0 < 0.1
    mirror.stop.set()


def test_an_unreachable_upstream_falls_back_to_the_local_feed() -> None:
    mirror = mm.Mirror("http://127.0.0.1:9", lambda: {"local": True})
    t0 = time.perf_counter()
    assert mirror.poll_once() is False
    assert time.perf_counter() - t0 < mm.TIMEOUT_S + 1
    assert mirror.card() == {"local": True}


def test_an_old_card_falls_back(upstream) -> None:
    mirror = mm.Mirror(upstream, lambda: {"local": True})
    assert mirror.poll_once()
    mirror._at -= mm.FRESH_S + 1
    assert mirror.card() == {"local": True}


def test_off_by_default_and_a_bad_setting_is_ignored(monkeypatch) -> None:
    monkeypatch.delenv("SPARKSI_MONITOR_UPSTREAM", raising=False)
    serve_mod._start_mirror(7601)
    assert STATE.monitor_fn is None and STATE.mirror is None
    monkeypatch.setenv("SPARKSI_MONITOR_UPSTREAM", "http://192.0.2.10:7500")
    serve_mod._start_mirror(7601)
    assert STATE.monitor_fn is None and STATE.mirror is None


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(serve_mod.HOSTS, "names", lambda: {"localhost", "127.0.0.1"})
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()


def test_the_mirror_feeds_monitor_gauges_and_history(upstream, server, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_MONITOR_UPSTREAM", upstream)
    serve_mod._start_mirror(server)
    for _ in range(100):
        if STATE.mirror.fresh():
            break
        time.sleep(0.02)
    with urlopen(f"http://127.0.0.1:{server}/api/monitor", timeout=5) as resp:
        card = json.loads(resp.read())
    assert card["mirrored"] is True
    assert [h["id"] for h in card["hosts"]] == ["head", "worker"]  # gauges read these
    assert card["loaded"][0]["state"] == "UP"  # Ready confirmation reads this
    assert serve_mod.collector_card()["hosts"][0]["gpu_pct"] == 41  # the history recorder's source
    history = History(tmp_path / "history.json")
    sample_card(serve_mod.collector_card(), history, None)
    assert history.latest_samples(fresh_secs=60)
    assert all(line.startswith("GET /api/monitor") for line in _Upstream.seen)
