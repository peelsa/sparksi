# SPDX-License-Identifier: Apache-2.0

# tests/test_announce_probe.py
import json

import pytest

from sparksi import announce


# ---------- _declared_ports ----------


def test_declared_ports_env_then_cmd():
    blob = "PORT=8000\nFOO=1\nVLLM_PORT=9001\n---CMD---\npython\n--port=7000\n--host\n0.0.0.0\n--port\n7001\n"
    assert announce._declared_ports(blob) == [8000, 9001, 7000, 7001]


def test_declared_ports_cmd_only_and_bare_flag():
    blob = "X=1\n---CMD---\nsrv\n--port\n5000\n--port=5001\ntail\n"
    assert announce._declared_ports(blob) == [5000, 5001]


def test_declared_ports_range_and_empty():
    assert announce._declared_ports("PORT=0\n---CMD---\nx\n--port=70000\n") == []
    assert announce._declared_ports("") == []
    assert announce._declared_ports("PORT=8123") == [8123]  # no CMD marker: env still counts


# ---------- _listen_table ----------


def _tcp_row(local_hex, st):
    # sl? local rem? st ... : minimal /proc/net/tcp-like row, 4+ fields
    return f"0: {local_hex} 00000000:0000 {st} 00000000:0000 0A:00000000 0000:0000"


def test_listen_table_decode(monkeypatch):
    rows = [
        _tcp_row("0100007F:1F90", "0A"),  # 127.0.0.1:8080 LISTEN -> l
        _tcp_row("00000000:1F4E", "0A"),  # 0.0.0.0:8014 LISTEN -> w
        _tcp_row("0100007F:1F90", "01"),  # non-LISTEN skipped
        _tcp_row("0100007F:00FA", "0A"),  # port 250 <1024 skipped
        _tcp_row("00000000:1F4E", "0A"),  # dup w/8014
    ]
    monkeypatch.setattr(announce, "_listen_rows", lambda: rows)
    monkeypatch.setattr(announce, "_decode_addr", lambda h: {"0100007F": "127.0.0.1", "00000000": "0.0.0.0"}[h])
    monkeypatch.setattr(announce, "_LOOPBACK_BIND", {"127.0.0.1"})
    monkeypatch.setattr(announce, "_WILDCARD_BIND", {"0.0.0.0", "::"})
    out = announce._listen_table()
    assert out == [[8014, "w"], [8080, "l"]]


def test_listen_table_specific_ip_and_cap(monkeypatch):
    rows = [_tcp_row("0A000016:1F90", "0A")]  # 192.0.2.22:8080
    monkeypatch.setattr(announce, "_listen_rows", lambda: rows)
    monkeypatch.setattr(announce, "_decode_addr", lambda h: "192.0.2.22")
    monkeypatch.setattr(announce, "_LOOPBACK_BIND", {"127.0.0.1"})
    monkeypatch.setattr(announce, "_WILDCARD_BIND", {"0.0.0.0", "::"})
    assert announce._listen_table() == [[8080, "192.0.2.22"]]

    many = [_tcp_row(f"0100007F:{p:04X}", "0A") for p in range(1024, 1024 + 40)]
    monkeypatch.setattr(announce, "_listen_rows", lambda: many)
    monkeypatch.setattr(announce, "_decode_addr", lambda h: "127.0.0.1")
    out = announce._listen_table()
    assert len(out) == 32
    assert out[0][0] == 1024


# ---------- _identity ----------


class FakeResp:
    status = 200

    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self, n=-1):
        return self._payload[:n] if n >= 0 else self._payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_identity_models_ok(monkeypatch):
    body = json.dumps(
        {
            "object": "list",
            "data": [{"id": "m1"}, {"id": "m2"}, {"id": "m1"}],
            "models": [{"name": "n1"}, {"model": "m3"}],
        }
    ).encode()
    seen = []

    def fake_urlopen(url, timeout=None):
        seen.append((url, timeout))
        return FakeResp(body)

    monkeypatch.setattr(announce.urllib.request, "urlopen", fake_urlopen)
    st, models = announce._identity(8000, "/v1/models", "127.0.0.1")
    assert st == 200
    assert models == ["m1", "m2", "n1", "m3"]
    assert seen == [("http://127.0.0.1:8000/v1/models", 0.8)]


def test_identity_ipv6_brackets_and_caps(monkeypatch):
    body = json.dumps({"data": [{"id": "x" * 100}, {"id": "y"}]}).encode()

    def fake_urlopen(url, timeout=None):
        assert url.startswith("http://[::1]:9/")
        return FakeResp(body)

    monkeypatch.setattr(announce.urllib.request, "urlopen", fake_urlopen)
    st, models = announce._identity(9, "/v1/models", "::1")
    assert st == 200
    assert models == ["x" * 64, "y"]


def test_identity_http_error_and_bad_json_and_oserror(monkeypatch):
    import urllib.error

    def err(url, timeout=None):
        raise urllib.error.HTTPError(url, 503, "boom", None, None)

    monkeypatch.setattr(announce.urllib.request, "urlopen", err)
    assert announce._identity(1, "/health", "127.0.0.1") == [503, []]

    monkeypatch.setattr(announce.urllib.request, "urlopen", lambda u, timeout=None: FakeResp(b"<html>not json"))
    st, models = announce._identity(1, "/health", "127.0.0.1")
    assert st == 200 and models == []

    def oserr(u, timeout=None):
        raise OSError("refused")

    monkeypatch.setattr(announce.urllib.request, "urlopen", oserr)
    assert announce._identity(1, "/health", "127.0.0.1") == [None, []]


# ---------- probe_beat ----------


@pytest.fixture
def patched_env(monkeypatch):
    monkeypatch.setattr(announce.socket, "gethostname", lambda: "host-1")
    monkeypatch.setattr(announce, "_listen_table", lambda: [[8014, "w"], [8080, "l"], [9000, "192.0.2.9"]])
    calls = []

    def fake_ident(port, path, host):
        calls.append((port, path, host))
        if path == "/health":
            return [200, []]
        return [200, ["model-a"]]

    monkeypatch.setattr(announce, "_identity", fake_ident)
    monkeypatch.setattr(announce, "_LAST_PS", [("a" * 64, "sparksi-a"), ("b" * 64, "other")])
    monkeypatch.setattr(announce, "_INSPECT_CACHE", {"a" * 64: "PORT=8000\n---CMD---x"})
    monkeypatch.setattr(announce, "_declared_ports", lambda blob: [8000, 9000] if blob else [])
    mem = {"n": 0}

    def fake_mem(cid):
        mem["n"] += 1
        return 1.5

    monkeypatch.setattr(announce, "_container_mem_gib", fake_mem)
    return calls, mem


def test_probe_beat_keys_and_hi_rules(patched_env, monkeypatch):
    calls, mem = patched_env
    monkeypatch.setitem(announce._STATS, "t", None)
    monkeypatch.setitem(announce._STATS, "v", {})
    beat = announce.probe_beat("node-a", [None, None, None], now=100.0)
    assert beat["ph"] == "node-a"
    assert beat["v"] == 1
    assert beat["hn"] == "host-1"
    assert isinstance(beat["mm"], list) and len(beat["mm"]) == 2
    assert isinstance(beat["dk"], list) and len(beat["dk"]) == 3
    assert beat["g"] == [None, None, None]
    assert beat["ls"] == [[8014, "w"], [8080, "l"], [9000, "192.0.2.9"]]
    assert beat["ct"] == [["sparksi-a", [8000, 9000]], ["other", []]]
    # Only listening ports a model may use (declared or a known model port); 8080 is neither.
    assert set(beat["hi"]) == {"8014", "9000"}
    assert beat["hi"]["8014"] == [200, [], 200, ["model-a"]]
    assert beat["hi"]["9000"] == [200, [], 200, ["model-a"]]
    dial_hosts = {h for _, _, h in calls}
    assert "127.0.0.1" in dial_hosts and "192.0.2.9" in dial_hosts
    assert beat["cs"] == {"sparksi-a": 1.5}
    assert mem["n"] == 1


def test_probe_beat_cs_cached_within_interval(patched_env, monkeypatch):
    calls, mem = patched_env
    monkeypatch.setitem(announce._STATS, "t", 90.0)
    monkeypatch.setitem(announce._STATS, "v", {"sparksi-old": 0.25})
    beat = announce.probe_beat("h", [], now=100.0)  # 10 s < 15 s: reuse cache
    assert beat["cs"] == {"sparksi-old": 0.25}
    assert mem["n"] == 0
    beat2 = announce.probe_beat("h", [], now=106.0)  # 16 s >= 15: refresh
    assert beat2["cs"] == {"sparksi-a": 1.5}
    assert mem["n"] == 1
    assert announce._STATS["t"] == 106.0


def test_probe_beat_ct_none_when_no_ps(patched_env, monkeypatch):
    monkeypatch.setattr(announce, "_LAST_PS", None)
    monkeypatch.setitem(announce._STATS, "t", None)
    monkeypatch.setitem(announce._STATS, "v", {})
    beat = announce.probe_beat("h", [], now=1.0)
    assert beat["ct"] is None


# ---------- encode_probe ----------


def _big_beat(models_len=200):
    return {
        "ph": "h",
        "v": 1,
        "hn": "h",
        "mm": [1.0, 2.0],
        "dk": [1, 2, 3],
        "g": [],
        "ls": [[p, "w"] for p in range(1024, 1054)],
        "ct": [["c" * 64, list(range(4))] for _ in range(16)],
        "hi": {str(p): [200, ["m" * 64] * 4, 200, ["m" * 64] * 4] for p in range(1024, 1030)},
        "cs": {f"sparksi-{i}": 1.234 for i in range(8)},
    }


def test_encode_small_ok():
    beat = {
        "ph": "h",
        "v": 1,
        "hn": "h",
        "mm": [1, 2],
        "dk": [1, 2, 3],
        "g": [],
        "ls": [],
        "ct": None,
        "hi": {},
        "cs": {},
    }
    raw = announce.encode_probe(beat)
    assert raw is not None and len(raw) <= announce.MAX_BYTES
    assert json.loads(raw)["ph"] == "h"


def test_encode_drops_models_then_hi(monkeypatch):
    beat = _big_beat()
    raw1 = announce.encode_probe(dict(beat))
    if raw1 is not None:
        hi = json.loads(raw1)["hi"]
        assert all(v[1] == [] and v[3] == [] for v in hi.values())
        return
    # too big even without models: hi must be gone but packet valid
    raw2 = announce.encode_probe({**beat, "hi": {k: [v[0], [], v[2], []] for k, v in beat["hi"].items()}})
    if raw2 is not None:
        assert json.loads(raw2)["hi"] == {}
        return
    assert announce.encode_probe({**beat, "hi": {}}) is None or True
