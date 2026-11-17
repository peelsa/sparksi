# SPDX-License-Identifier: Apache-2.0

"""Spark announce packets stay small, and a silent host is not called down."""

import json

from sparksi.announce import (
    _SHORT_BY_FULL,
    _metrics_from_prom,
    _normalize_metric,
    _port_from_blob,
    encode,
    to_peer,
)
from sparksi.announce_rx import Inbox
from sparksi.dash import monitor_card

HEAD_IP = "100.64.0.11"
WORKER_IP = "100.64.0.12"


def _pkt(host: str) -> dict:
    return {
        "h": host,
        "mem": [121.7, 110.0],
        "cpu": [1.2, 3200],
        "gpu": [0, 37, 200],
        "z": [["acpitz0", 39.7], ["acpitz1", 38.2]],
        "top": [["gnome-shell", 0.27], ["dockerd", 0.06]],
        "cache": 3.1,
        "model": 0.4,
        "slab": 0.2,
        "small": 1.1,
        "dock": [[f"pair-exl3-{'head' if host == 'a' else 'worker'}", "Up"]],
        "eng": [["pair-exl3-head", 8014, 2]] if host == "a" else [],
        "http": {"8014": ["Pair-EXL3", {"r": 0, "w": 0, "kv": 0.02, "gt": 30, "pt": 59}]} if host == "a" else {},
        "video": [0, ""],
    }


def test_live_completion_counter_is_the_rate_source():
    got = _metrics_from_prom("\n".join([
        "tensorfold:generation_tokens_total 10",
        "tensorfold_health:completion_tokens_total 40",
        "tensorfold:prompt_tokens_total 7",
        "tensorfold:requests_running 1",
    ]))
    assert got["gt"] == 40
    assert got["pt"] == 7
    assert got["r"] == 1


def test_tensorfold_metric_names_use_the_vllm_shorts():
    assert _SHORT_BY_FULL[_normalize_metric("tensorfold:generation_tokens_total")] == "gt"
    assert _SHORT_BY_FULL[_normalize_metric("tensorfold:prompt_tokens_total")] == "pt"
    assert _SHORT_BY_FULL[_normalize_metric("tensorfold:requests_running")] == "r"
    assert _SHORT_BY_FULL[_normalize_metric("tensorfold:requests_waiting")] == "w"
    assert _SHORT_BY_FULL[_normalize_metric("tensorfold:kv_cache_usage_ratio")] == "kv"
    assert _SHORT_BY_FULL[_normalize_metric("vllm:generation_tokens_total")] == "gt"


def test_tensorfold_command_declares_port_and_tp():
    blob = "\n".join([
        "HF_HUB_OFFLINE=1",
        "---CMD---",
        "tensorfold",
        "serve",
        "model",
        "--tp",
        "2",
        "--rank",
        "0",
        "--port",
        "8014",
    ])
    assert _port_from_blob(blob) == (8014, 2)


def test_encode_stays_under_1200_bytes():
    fat = _pkt("a")
    fat["top"] = [[f"proc{i}", 1.2] for i in range(8)]
    fat["z"] = [[f"zone{i}", 40.0] for i in range(12)]
    fat["dock"] = [[f"container-name-{i}", "Up 3 hours"] for i in range(8)]
    raw = encode(fat)
    assert len(raw) <= 1200
    assert json.loads(raw)["h"] == "a"


def test_rejected_sources_and_oversize():
    box = Inbox()
    raw = encode(_pkt("a"))
    assert box.accept(HEAD_IP, raw)
    assert not box.accept("192.0.2.52", raw)
    assert not box.accept(HEAD_IP, encode(_pkt("b")))
    assert not box.accept(HEAD_IP, b"not-json")
    assert not box.accept(HEAD_IP, b"x" * 1201)


def test_pair_is_one_span_and_one_silent_host_is_no_announce():
    clock = [0.0]
    box = Inbox(clock=lambda: clock[0])
    assert box.accept(HEAD_IP, encode(_pkt("a")))
    clock[0] = 10.0
    assert box.accept(WORKER_IP, encode(_pkt("b")))
    clock[0] = 11.0
    card = monitor_card(box.peer("a"), box.peer("b"))
    by_id = {h["id"]: h for h in card["hosts"]}
    assert by_id["a"]["detail"] == "no announce"
    assert by_id["a"]["unreachable"] is False
    assert by_id["b"]["fresh"] is True
    assert card["loaded"] == []

    clock[0] = 12.0
    assert box.accept(HEAD_IP, encode(_pkt("a")))
    card = monitor_card(box.peer("a"), box.peer("b"))
    assert card["loaded"]
    assert card["loaded"][0]["span"] == ["a", "b"]
    assert card["loaded"][0]["worker_up"] is True


def test_both_silent_is_unreachable():
    clock = [0.0]
    box = Inbox(clock=lambda: clock[0])
    box.accept(HEAD_IP, encode(_pkt("a")))
    box.accept(WORKER_IP, encode(_pkt("b")))
    clock[0] = 11.0
    card = monitor_card(box.peer("a"), box.peer("b"))
    assert {h["detail"] for h in card["hosts"]} == {"unreachable"}
    assert all(h["unreachable"] is True for h in card["hosts"])
    assert card["loaded"] == []


def test_to_peer_expands_metric_names():
    peer = to_peer(_pkt("a"))
    assert peer["http"][8014]["metrics"]["kv_cache_usage_perc"] == 0.02
    assert peer["engines"][0]["port"] == 8014



def test_served_id_reads_llama_cpp_and_openai():
    from sparksi.announce import _served_id

    assert _served_id('{"models":[{"name":"single-gguf-model"}],"data":[{"id":"single-gguf-model"}]}') == "single-gguf-model"
    assert _served_id('{"models":[{"name":"only-llama"}]}') == "only-llama"
    assert _served_id('{"error":"x"}') == ""


def test_specific_bind_is_dialed_instead_of_loopback(monkeypatch):
    from sparksi import announce

    # 192.0.2.10 is documentation space. Port 8014 is 1F4E. The address word is little-endian.
    row = "0: 0A0200C0:1F4E 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 0 10 0"
    monkeypatch.setattr(announce, "_listen_rows", lambda: [row])
    assert announce._dial_host(8014) == "192.0.2.10"
    monkeypatch.setattr(announce, "_listen_rows", lambda: ["0: 00000000:1F4E 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 0 10 0"])
    assert announce._dial_host(8014) == "127.0.0.1"
    seen = []

    class Response:
        status = 200

        def read(self, _n):
            return b'{"data":[{"id":"example-model"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(url, timeout=0):
        seen.append(url)
        if str(url).endswith("/metrics"):
            raise OSError("no metrics")
        return Response()

    monkeypatch.setattr(announce, "_http_ports", lambda _engines: [8014])
    monkeypatch.setattr(announce, "_dial_host", lambda _port: "192.0.2.10")
    monkeypatch.setattr(announce.urllib.request, "urlopen", opener)
    found = announce._http([])
    assert seen[0].startswith("http://192.0.2.10:8014/")
    assert "127.0.0.1" not in seen[0]
    assert found["8014"][0] == "example-model"
    assert found["8014"][2] == "192.0.2.10"
    peer = announce.to_peer({"h": "head", "mem": [1, 1], "cpu": [0, 0], "gpu": [0, 0, 0], "http": found})
    assert peer["http"][8014]["listen"] == "192.0.2.10"


def test_wildcard_bind_stays_distinct_from_loopback(monkeypatch):
    from sparksi import announce

    monkeypatch.setattr(
        announce,
        "_listen_rows",
        lambda: ["0: 00000000:1F4E 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 0 10 0"],
    )
    monkeypatch.setattr(announce, "_http_ports", lambda _engines: [8014])

    class Response:
        def read(self, _n):
            return b'{"data":[{"id":"example-model"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(url, timeout=0):
        if not str(url).startswith("http://127.0.0.1:8014/"):
            raise OSError(str(url))
        if str(url).endswith("/metrics"):
            raise OSError("no metrics")
        return Response()

    monkeypatch.setattr(announce.urllib.request, "urlopen", opener)
    found = announce._http([])
    assert found["8014"][2] == "0.0.0.0"
    peer = announce.to_peer({"h": "a", "mem": [1, 1], "cpu": [0, 0], "gpu": [0, 0, 0], "http": found})
    assert peer["http"][8014]["bind"] == "wildcard"
    assert "listen" not in peer["http"][8014]


def test_advertised_probe_refuses_redirects_proxies_and_long_bodies(monkeypatch) -> None:
    import threading
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from sparksi import announce

    other_hits = {"n": 0}

    class Other(BaseHTTPRequestHandler):
        def do_GET(self):
            other_hits["n"] += 1
            body = b'{"data":[{"id":"current-served"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt, *_args):
            return

    class Redirect(BaseHTTPRequestHandler):
        location = ""

        def do_GET(self):
            self.send_response(302)
            self.send_header("Location", type(self).location)
            self.end_headers()

        def log_message(self, _fmt, *_args):
            return

    class Fat(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b" " * 4096 + b'{"data":[{"id":"current-served"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt, *_args):
            return

    class Proxy(BaseHTTPRequestHandler):
        hits = 0

        def do_GET(self):
            type(self).hits += 1
            body = b'{"data":[{"id":"current-served"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _fmt, *_args):
            return

    def serve(handler):
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server

    other = serve(Other)
    Redirect.location = f"http://127.0.0.1:{other.server_port}/v1/models"
    redirected = serve(Redirect)
    fat = serve(Fat)
    proxy = serve(Proxy)
    try:
        opener = announce._advertised_opener()
        assert not any(getattr(handler, "proxies", None) for handler in opener.handlers)
        assert any(isinstance(handler, announce._NoRedirect) for handler in opener.handlers)
        assert not any(type(handler) is urllib.request.HTTPRedirectHandler for handler in opener.handlers)
        try:
            opener.open(
                f"http://127.0.0.1:{redirected.server_port}/v1/models",
                timeout=announce.ADVERTISE_TIMEOUT,
            )
            raise AssertionError("redirect was followed")
        except urllib.error.HTTPError as exc:
            assert exc.code == 302
        assert other_hits["n"] == 0

        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.setenv("https_proxy", f"http://127.0.0.1:{proxy.server_port}")
        monkeypatch.setenv("SPARKSI_ADVERTISE", "192.0.2.55")
        assert announce._verify_advertised(8014, "current-served") == ""
        assert Proxy.hits == 0

        monkeypatch.setenv("SPARKSI_ADVERTISE", "127.0.0.1")
        assert announce._advertised_address() == ""

        # The address function refuses loopback, so this stand-in is only the
        # read cap. The request still uses the real opener.
        original = announce._advertised_address
        announce._advertised_address = lambda: "127.0.0.1"
        try:
            assert announce._verify_advertised(fat.server_port, "current-served") == ""
        finally:
            announce._advertised_address = original
    finally:
        for server in (other, redirected, fat, proxy):
            server.shutdown()
            server.server_close()


def test_plain_process_port_is_probed_without_a_container(monkeypatch):
    from sparksi import announce

    monkeypatch.setattr(announce, "_listening", lambda port: port == 8014)
    assert announce._http_ports([]) == [8014]
    assert announce._http_ports([["pair-exl3-head", 8014, 2]]) == [8014]


def test_video_service_is_down_when_8188_is_only_a_bound_socket(monkeypatch):
    from sparksi import announce

    monkeypatch.setattr(announce, "_listening", lambda port: True)

    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(announce.urllib.request, "urlopen", boom)
    assert announce._video_state()[0] == 0
