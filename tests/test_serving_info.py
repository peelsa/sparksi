# SPDX-License-Identifier: Apache-2.0

"""Context size and KV cache use of a ready run: parsers, the read order, the cache, and /api/runs."""

from __future__ import annotations

import copy
import json
import threading
import time
from urllib.error import HTTPError

import pytest

from sparksi import monitor_hosts
from sparksi import serving_info as si

VLLM_METRICS = b"""# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="Example-Model"} 1.0
vllm:num_requests_waiting{engine="0",model_name="Example-Model"} 0.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
vllm:kv_cache_usage_perc{engine="0",model_name="Example-Model"} 0.1362586605080831
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.001",model_name="Example-Model"} 0.0
vllm:cache_config_info{block_size="16",cache_dtype="fp8",note="a \\"quoted\\", value",enable_prefix_caching="True",kv_cache_size_tokens="1233779",num_gpu_blocks="77111"} 1.0
"""
VLLM_MODELS = json.dumps({"object": "list", "data": [{"id": "Example-Model", "owned_by": "vllm", "max_model_len": 850000}]}).encode()
SGLANG_METRICS = b"""sglang:token_usage{model_name="m"} 0.5
sglang:max_total_num_tokens{model_name="m"} 200000.0
sglang:num_running_reqs{model_name="m"} 2.0
"""
LLAMA_METRICS = b"""llamacpp:kv_cache_usage_ratio 0.25
llamacpp:requests_processing 1
"""
LLAMA_MODELS = json.dumps({"object": "list", "data": [{"id": "model.gguf", "owned_by": "llamacpp", "meta": {"n_ctx_train": 131072}}]}).encode()


# ---------------------------------------------------------------- parsers


def test_parse_models():
    assert si.parse_models(VLLM_MODELS) == {"ctx": 850000, "owner": "vllm"}
    assert si.parse_models(json.dumps({"data": [{"id": "x"}]}).encode()) == {}
    assert si.parse_models(b"not json") == {}
    assert si.parse_models(json.dumps({"data": [{"max_model_len": True}]}).encode()) == {}
    assert si.parse_models(json.dumps({"data": []}).encode()) == {}
    assert si.parse_models(json.dumps([1, 2]).encode()) == {}


def test_parse_metrics_vllm():
    assert si.parse_metrics(VLLM_METRICS) == {"kv_pct": 13.6, "running": 1, "kv_tokens": 1233779, "engine": "vllm"}


def test_parse_metrics_vllm_blocks_fallback_and_none():
    page = b'vllm:cache_config_info{block_size="16",num_gpu_blocks="1000"} 1.0\nvllm:kv_cache_usage_perc 0.5\n'
    assert si.parse_metrics(page)["kv_tokens"] == 16000
    page = b'vllm:cache_config_info{block_size="16",num_gpu_blocks="None"} 1.0\nvllm:kv_cache_usage_perc 0.5\n'
    assert "kv_tokens" not in si.parse_metrics(page)


@pytest.mark.parametrize("value", ["1.5", "-0.1", "nan", "inf", "x"])
def test_parse_metrics_ignores_bad_fractions(value):
    page = f"vllm:kv_cache_usage_perc {value}\nvllm:kv_cache_usage_perc 0.2\n".encode()
    assert si.parse_metrics(page)["kv_pct"] == 20.0


def test_parse_metrics_many_engines_labels_nan_and_missing():
    text = (
        "# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.\n"
        'vllm:kv_cache_usage_perc{engine="0",model_name="a{b}c \\"q\\""} 0.25\n'
        'vllm:kv_cache_usage_perc{engine="1",model_name="other"} 0.5 1732560000000\n'
        'vllm:kv_cache_usage_perc{engine="2",model_name="x"} NaN\n'
        'vllm:num_requests_running{engine="0",model_name="a"} 2\n'
        'vllm:num_requests_running{engine="1",model_name="other"} 1\n'
        'vllm:num_requests_running{engine="2",model_name="x"} +Inf\n'
        "vllm:kv_cache_usage_perc_bucket 0.9\n"
        'vllm:cache_config_info{block_size="16",num_gpu_blocks=""} 1\n'
    ).encode()
    out = si.parse_metrics(text)
    assert out == {"kv_pct": 50.0, "running": 3, "engine": "vllm"}
    assert si.parse_metrics(b"") == {}
    assert si.parse_metrics(b"vllm:kv_cache_usage_perc\n") == {}


def test_parse_metrics_older_vllm_name():
    assert si.parse_metrics(b"vllm:gpu_cache_usage_perc 0.75\n")["kv_pct"] == 75.0


def test_parse_metrics_sglang_and_llamacpp():
    assert si.parse_metrics(SGLANG_METRICS) == {"kv_pct": 50.0, "running": 2, "kv_tokens": 200000, "engine": "sglang"}
    assert si.parse_metrics(LLAMA_METRICS) == {"kv_pct": 25.0, "running": 1, "engine": "llamacpp"}
    assert si.parse_metrics(b"") == {}
    assert si.parse_metrics(b"other_metric 1\n# vllm:kv_cache_usage_perc 0.5\n") == {}


def test_parse_props_and_slots():
    assert si.parse_props(b'{"n_ctx": 32768}') == {"ctx": 32768}
    assert si.parse_props(b'{"default_generation_settings": {"n_ctx": 8192}}') == {"ctx": 8192}
    assert si.parse_props(b'{"n_ctx": 0}') == {}
    assert si.parse_props(b"[]") == {}
    slots = [{"id": 0, "n_ctx": 4096, "n_past": 1024}, {"id": 1, "n_ctx": 4096, "n_past": 0}, {"id": 2, "n_ctx": True}]
    assert si.parse_slots(json.dumps(slots).encode()) == {"kv_pct": 12.5}
    assert si.parse_slots(b"[]") == {}
    assert si.parse_slots(b'[{"n_ctx": 100, "n_past": 500}]') == {"kv_pct": 100.0}


# ---------------------------------------------------------------- read


class Fetch:
    def __init__(self, answers):
        self.answers = answers
        self.paths: list[str] = []
        self.kwargs: list[dict] = []

    def __call__(self, url, path, **kw):
        self.paths.append(path)
        self.kwargs.append(kw)
        got = self.answers.get(path, OSError("no route"))
        if isinstance(got, BaseException):
            raise got
        return got


def test_read_vllm_asks_only_models_and_metrics():
    fetch = Fetch({"/v1/models": VLLM_MODELS, "/metrics": VLLM_METRICS})
    assert si.read("http://100.64.0.11:8014", fetch) == {"ctx": 850000, "kv_pct": 13.6, "kv_tokens": 1233779, "running": 1, "engine": "vllm"}
    assert fetch.paths == ["/v1/models", "/metrics"]
    assert all(kw == {"timeout": si.TIMEOUT_S, "total": si.TOTAL_S} for kw in fetch.kwargs)


def test_read_llamacpp_without_metrics_uses_props_and_slots():
    err = HTTPError("http://x", 501, "metrics off", {}, None)
    fetch = Fetch({"/v1/models": LLAMA_MODELS, "/metrics": err, "/props": b'{"n_ctx": 65536}',
                   "/slots": b'[{"n_ctx": 65536, "n_past": 6554}]'})
    assert si.read("http://100.64.0.11:8080", fetch) == {"ctx": 65536, "kv_pct": 10.0}
    assert fetch.paths == ["/v1/models", "/metrics", "/props", "/slots"]


def test_read_llamacpp_with_metrics_skips_slots():
    fetch = Fetch({"/v1/models": LLAMA_MODELS, "/metrics": LLAMA_METRICS, "/props": b'{"n_ctx": 65536}'})
    assert si.read("http://100.64.0.11:8080", fetch) == {"ctx": 65536, "kv_pct": 25.0, "running": 1, "engine": "llamacpp"}
    assert "/slots" not in fetch.paths


def test_read_gives_nothing_when_unknown():
    assert si.read("http://100.64.0.11:1", Fetch({})) == {}
    assert si.read("http://100.64.0.11:1", Fetch({"/v1/models": b'{"data": [{"id": "m"}]}'})) == {}
    assert si.read("http://100.64.0.11:1", Fetch({"/v1/models": monitor_hosts.Refused("no")})) == {}


def test_monitor_get_allows_the_llamacpp_reads():
    assert {"/props", "/slots", "/v1/models", "/metrics"} <= monitor_hosts._ALLOWED_PATHS
    with pytest.raises(monitor_hosts.Refused):
        monitor_hosts.get("http://100.64.0.11:8080", "/completion")


# ---------------------------------------------------------------- cache


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_cache_returns_at_once_and_refreshes_after_ttl():
    calls: list[str] = []
    clock = Clock()
    cache = si.Cache(read=lambda url: calls.append(url) or {"ctx": 1}, clock=clock, spawn=lambda fn: fn())
    assert cache.get("u") == {}  # the first poll never waits; the answer lands for the next
    assert cache.get("u") == {"ctx": 1}
    assert calls == ["u"]
    clock.now += si.TTL_S - 0.1
    cache.get("u")
    assert calls == ["u"]
    clock.now += 0.2
    cache.get("u")
    assert calls == ["u", "u"]


def test_cache_waits_longer_after_nothing():
    calls: list[str] = []
    clock = Clock()
    cache = si.Cache(read=lambda url: calls.append(url) or {}, clock=clock, spawn=lambda fn: fn())
    cache.get("u")
    clock.now += si.TTL_S + 1
    cache.get("u")
    assert calls == ["u"]
    clock.now += si.FAIL_TTL_S
    cache.get("u")
    assert calls == ["u", "u"]


def test_cache_never_starts_two_refreshes():
    pending: list = []
    cache = si.Cache(read=lambda url: {"ctx": 2}, clock=Clock(), spawn=pending.append)
    cache.get("u")
    cache.get("u")
    assert len(pending) == 1
    pending[0]()
    assert cache.get("u") == {"ctx": 2}


def test_cache_survives_a_reader_error_and_returns_copies():
    def boom(url):
        raise RuntimeError("bug")

    cache = si.Cache(read=boom, clock=Clock(), spawn=lambda fn: fn())
    cache.get("u")
    assert cache.get("u") == {}
    good = si.Cache(read=lambda url: {"ctx": 3}, clock=Clock(), spawn=lambda fn: fn())
    good.get("u")
    good.get("u")["ctx"] = 99
    assert good.get("u") == {"ctx": 3}


def test_cache_keeps_at_most_max_urls():
    cache = si.Cache(read=lambda url: {"ctx": 1}, clock=Clock(), spawn=lambda fn: fn())
    for i in range(si.MAX_URLS + 1):
        cache.get(f"u{i}")
    assert len(cache._values) == si.MAX_URLS
    assert "u0" not in cache._values


def test_cache_get_does_not_wait_for_a_slow_server():
    gate = threading.Event()
    cache = si.Cache(read=lambda url: (gate.wait(5), {"ctx": 1})[1])
    started = time.monotonic()
    assert cache.get("u") == {}
    assert time.monotonic() - started < 0.5
    gate.set()


# ---------------------------------------------------------------- attach


class FixedCache:
    def __init__(self, info):
        self.info = info
        self.urls: list[str] = []

    def get(self, url, key=""):
        self.urls.append(url)
        return dict(self.info)


def test_new_run_on_the_same_port_never_shows_the_old_runs_answer():
    clock = Clock()
    answers = {"ctx": 1}
    cache = si.Cache(read=lambda url: dict(answers), clock=clock, spawn=lambda fn: fn())
    cache.get("u", "old-run")
    assert cache.get("u", "old-run") == {"ctx": 1}
    answers["ctx"] = 2
    # A new run on the same server URL starts empty: the old run's value is not handed over.
    assert cache.get("u", "new-run") == {}
    assert cache.get("u", "new-run") == {"ctx": 2}


def test_an_old_answer_is_hidden_even_while_its_refresh_runs():
    clock = Clock()
    pending: list = []
    cache = si.Cache(read=lambda url: {"ctx": 1}, clock=clock, spawn=pending.append)
    cache.get("u", "r")
    pending.pop()()
    assert cache.get("u", "r") == {"ctx": 1}
    clock.now += si.TTL_S + 1
    assert cache.get("u", "r") == {"ctx": 1}  # stale but recent: shown while the refresh runs
    assert len(pending) == 1
    clock.now += si.MAX_STALE_S
    assert cache.get("u", "r") == {}  # the engine went quiet: nothing old is shown
    assert len(pending) == 1  # still one refresh in flight, never a second thread


def test_off_limits_port_is_never_asked(monkeypatch):
    from sparksi import protect

    monkeypatch.setattr(protect, "is_off_limits", lambda port: port == 8188)
    cache = FixedCache(INFO)
    body = {"runs": [{"id": "r1", "phase": "ready", "port": 8188, "health": {"host": "spark-1"}}]}
    assert si.attach(body, lambda host: "100.64.0.11", cache) is body
    assert cache.urls == []


@pytest.mark.parametrize("phase", ["failed", "timed_out", "stopping", "starting", "waiting_health", "stopped"])
def test_only_ready_runs_are_asked(phase):
    cache = FixedCache(INFO)
    body = {"runs": [{"id": "r1", "phase": phase, "port": 8014, "health": {"host": "spark-1"}, "failed_side": "worker"}],
            "serving": [{"role": "head", "run_id": "r1"}]}
    assert si.attach(body, lambda host: "100.64.0.11", cache) is body
    assert cache.urls == []


def test_pair_run_is_read_from_its_lead_and_named_on_both_cards():
    cache = FixedCache(INFO)
    addresses = {"spark-1": "100.64.0.11", "spark-2": "100.64.0.12"}
    body = {
        "runs": [
            {"id": "pair", "phase": "ready", "port": 8014, "health": {"host": "spark-1"}},
            {"id": "solo", "phase": "ready", "port": 8020, "health": {"host": "spark-2"}},
        ],
        "serving": [{"role": "head", "run_id": "pair"}, {"role": "worker", "run_id": "pair"}, {"role": "worker", "run_id": "solo"}],
    }
    out = si.attach(body, addresses.get, cache)
    assert cache.urls == ["http://100.64.0.11:8014", "http://100.64.0.12:8020"]
    assert [row.get("serving_info") for row in out["serving"]] == [INFO, INFO, INFO]


INFO = {"ctx": 850000, "kv_pct": 13.6, "kv_tokens": 1233779, "running": 1, "engine": "vllm"}


def _body(n):
    hosts = [f"spark-{i}" for i in range(1, n + 1)]
    lead = hosts[-1]
    runs = [
        {"id": "r1", "phase": "ready", "port": 8014, "health": {"host": lead}},
        {"id": "r2", "phase": "waiting_health", "port": 8015, "health": {"host": lead}},
        {"id": "r3", "phase": "ready", "port": 70000, "health": {"host": lead}},
        {"id": "r4", "phase": "ready", "port": 8016, "health": {"host": "elsewhere"}},
        {"id": "r5", "phase": "ready", "port": 8017, "health": {}},
    ]
    serving = [{"role": f"role-{i}", "run_id": "r1"} for i in range(n)] + [{"role": "x", "run_id": "r2"}]
    return {"runs": runs, "serving": serving}, {h: f"100.64.0.{10 + i}" for i, h in enumerate(hosts, 1)}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_attach_on_one_to_four_sparks(n):
    body, addresses = _body(n)
    before = copy.deepcopy(body)
    cache = FixedCache(INFO)
    out = si.attach(body, lambda host: addresses.get(host, ""), cache)
    assert body == before
    assert cache.urls == [f"http://100.64.0.{10 + n}:8014"]
    assert out["runs"][0]["serving_info"] == INFO
    assert all("serving_info" not in run for run in out["runs"][1:])
    assert [row.get("serving_info") for row in out["serving"]] == [INFO] * n + [None]


def test_attach_without_answers_returns_the_body_unchanged():
    body, addresses = _body(2)
    assert si.attach(body, addresses.get, FixedCache({})) is body


def test_attach_ipv6_and_bad_input():
    cache = FixedCache(INFO)
    body = {"runs": [{"id": "r1", "phase": "ready", "port": 8014, "health": {"host": "spark-1"}}, "junk", None]}
    out = si.attach(body, lambda host: "fd7a:115c::1", cache)
    assert cache.urls == ["http://[fd7a:115c::1]:8014"]
    assert out["runs"][1:] == ["junk", None]
    assert si.attach({}, lambda host: "x") == {}

    def broken(host):
        raise RuntimeError("bad config")

    assert si.attach(body, broken, cache) is body


# ---------------------------------------------------------------- controller and CLI

import sparksi.serve as serve_mod  # noqa: E402
from sparksi.config import CFG  # noqa: E402


def _raw(n: int) -> dict:
    nodes = [{"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}", "tailscale": f"100.64.0.{10 + i}"} for i in range(1, n + 1)]
    return {"nodes": nodes, "paths": {"home": "/home/USER"}}


@pytest.fixture()
def sparks(request):
    saved = copy.copy(CFG.__dict__)
    CFG.__init__(_raw(getattr(request, "param", 2)))
    request.addfinalizer(lambda: (CFG.__dict__.clear(), CFG.__dict__.update(saved)))
    return CFG.nodes


@pytest.mark.parametrize("sparks", [1, 2, 3, 4], indirect=True)
def test_serving_address_by_name_or_id(sparks):
    last = sparks[-1]
    assert serve_mod._serving_address(last.name) == last.tailscale
    assert serve_mod._serving_address(last.id) == last.tailscale
    assert serve_mod._serving_address("unknown") == ""
    assert serve_mod._serving_address("") == ""


def test_serving_address_falls_back_to_ssh_name(sparks, monkeypatch):
    node = sparks[0]
    monkeypatch.setattr(node, "tailscale", "")
    assert serve_mod._serving_address(node.name) == node.ssh
    monkeypatch.setattr(node, "ssh", "user@spark-1")
    assert serve_mod._serving_address(node.name) == node.name


def test_with_serving_info_uses_the_cache_and_never_fails(sparks, monkeypatch):
    body = {"runs": [{"id": "r1", "phase": "ready", "port": 8014, "health": {"host": sparks[-1].name}}], "serving": []}
    seen: list[str] = []
    monkeypatch.setattr(si.CACHE, "get", lambda url, key="": seen.append(url) or dict(INFO))
    out = serve_mod._with_serving_info(body)
    assert out["runs"][0]["serving_info"] == INFO
    assert seen == [f"http://{sparks[-1].tailscale}:8014"]
    monkeypatch.setattr(si, "attach", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert serve_mod._with_serving_info(body) is body


def test_api_runs_does_not_wait_for_a_slow_server(sparks, monkeypatch, tmp_path):
    """/api/runs answers at once while the server read runs in the background."""
    import http.client

    from sparksi.lock import Lock
    from sparksi.serve import STATE, DualServer, Handler

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    STATE.lock = Lock(tmp_path / "lock")
    run = {"id": "r1", "phase": "ready", "port": 8014, "health": {"host": sparks[-1].name}, "nodes": []}
    monkeypatch.setattr(STATE, "runs_fn", lambda: {"runs": [run]})
    gate = threading.Event()
    slow = si.Cache(read=lambda url: (gate.wait(10), dict(INFO))[1])
    monkeypatch.setattr(si, "CACHE", slow)
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def ask():
            conn = http.client.HTTPConnection("127.0.0.1", httpd.server_address[1], timeout=5)
            conn.request("GET", "/api/runs")
            data = json.loads(conn.getresponse().read())
            conn.close()
            return data

        started = time.monotonic()
        first = ask()
        assert time.monotonic() - started < 2
        assert "serving_info" not in first["runs"][0]
        gate.set()
        deadline = time.monotonic() + 5
        while True:
            again = ask()
            if again["runs"][0].get("serving_info") or time.monotonic() > deadline:
                break
            time.sleep(0.05)
        assert again["runs"][0]["serving_info"] == INFO
    finally:
        gate.set()
        httpd.shutdown()
        httpd.server_close()
        STATE.runs_fn = None


def test_cli_runs_prints_the_facts(capsys, monkeypatch):
    from types import SimpleNamespace

    from sparksi import cli_api

    run = {"id": "r1", "model_id": "example", "phase": "ready", "port": 8014, "nodes": [{"role": "head", "phase": "started"}], "serving_info": INFO}
    other = {"id": "r2", "model_id": "other", "phase": "ready", "port": 8015, "nodes": []}

    class Client:
        secret = ""

        def _request(self, method, path):
            return 200, {"runs": [run, other], "serving": []}

    monkeypatch.setattr(cli_api, "_client", lambda args: (Client(), ""))
    assert cli_api.cmd_runs(SimpleNamespace(json=False)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].endswith("850000 ctx · 13.6% KV used (shared by all requests)")
    assert "ctx" not in lines[1]
    assert cli_api.cmd_runs(SimpleNamespace(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["runs"][0]["serving_info"] == INFO and "serving_info" not in data["runs"][1]


def test_parsers_ignore_values_outside_their_range() -> None:
    """Edge cases raised while drafting: NaN, Inf, fractions outside 0..1, bools, and slots past the window."""
    from sparksi import serving_info as s

    assert s.parse_metrics(b'vllm:kv_cache_usage_perc NaN\nvllm:kv_cache_usage_perc{a="b"} 0.2\n')["kv_pct"] == 20.0
    assert "kv_pct" not in s.parse_metrics(b"vllm:kv_cache_usage_perc +Inf\n")
    assert "kv_pct" not in s.parse_metrics(b"vllm:kv_cache_usage_perc 1.7\nvllm:kv_cache_usage_perc -0.1\n")
    assert s.parse_metrics(b"vllm:num_requests_running NaN\nvllm:num_requests_running 2\n")["running"] == 2
    assert s.parse_models(b'{"data":[{"max_model_len":true}]}') == {}
    assert s.parse_models(b"[1]") == {} and s.parse_models(b'{"data":[5]}') == {}
    assert s.parse_slots(b'[{"n_past":900,"n_ctx":100}]') == {"kv_pct": 100.0}
    assert s.parse_slots(b'[{"n_past":true,"n_ctx":100}]') == {}
