# SPDX-License-Identifier: Apache-2.0

import json
import threading
import urllib.error
import urllib.parse

import pytest

from sparksi import monitor_hosts as mh


def _b(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


# ---------------------------------------------------------------- parsers


class TestParseOpenaiModels:
    def test_normal_vllm_reply(self):
        body = _b(
            {"object": "list", "data": [{"id": "org/model-a", "object": "model"}, {"id": "model-b"}]}
        )
        assert mh.parse_openai_models(body) == ["org/model-a", "model-b"]

    def test_duplicates_removed(self):
        body = _b({"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"}]})
        assert mh.parse_openai_models(body) == ["model-a", "model-b"]

    def test_non_dict_items_and_non_str_ids_skipped(self):
        body = _b({"data": ["nope", 42, None, {"id": 7}, {"id": "model-a"}, {"other": "x"}]})
        assert mh.parse_openai_models(body) == ["model-a"]

    def test_control_chars_and_rtl_override_removed(self):
        body = _b({"data": [{"id": "bad\u202ename\n"}]})
        assert mh.parse_openai_models(body) == ["badname"]

    def test_capped_at_max_models(self):
        ids = [{"id": f"model-{i:02d}"} for i in range(60)]
        out = mh.parse_openai_models(_b({"data": ids}))
        assert len(out) == mh.MAX_MODELS == 50
        assert out[0] == "model-00"
        assert out[-1] == "model-49"

    def test_data_not_a_list(self):
        assert mh.parse_openai_models(_b({"data": "x"})) == []

    def test_not_json_raises(self):
        with pytest.raises(ValueError):
            mh.parse_openai_models(b"not json")

    def test_empty_data(self):
        assert mh.parse_openai_models(_b({"data": []})) == []

    def test_top_level_list(self):
        assert mh.parse_openai_models(_b([{"id": "model-a"}])) == []


class TestParseOllamaTags:
    def test_name_and_model_keys(self):
        body = _b({"models": [{"name": "llama3:8b", "model": "llama3:8b"}, {"model": "qwen:4b"}]})
        assert mh.parse_ollama_tags(body) == ["llama3:8b", "qwen:4b"]

    def test_not_json_raises(self):
        with pytest.raises(ValueError):
            mh.parse_ollama_tags(b"nope")

    def test_missing_models_key(self):
        assert mh.parse_ollama_tags(_b({"other": []})) == []


class TestParseOllamaPs:
    def test_sizes_converted_to_gib(self):
        body = _b(
            {"models": [{"name": "llama3:8b", "size": 5583457484, "size_vram": 5583457484}]}
        )
        assert mh.parse_ollama_ps(body) == [
            {"name": "llama3:8b", "size_gib": 5.2, "vram_gib": 5.2}
        ]

    def test_bad_sizes_become_none(self):
        body = b'{"models":[{"name":"m","size":true,"size_vram":-5}]}'
        assert mh.parse_ollama_ps(body) == [{"name": "m", "size_gib": None, "vram_gib": None}]

    def test_nan_size_is_none(self):
        body = b'{"models":[{"name":"m","size":NaN}]}'
        (entry,) = mh.parse_ollama_ps(body)
        assert entry["size_gib"] is None

    def test_model_key_fallback_and_missing_name(self):
        body = _b({"models": [{"model": "qwen:4b"}, {}]})
        assert mh.parse_ollama_ps(body) == [
            {"name": "qwen:4b", "size_gib": None, "vram_gib": None},
            {"name": "", "size_gib": None, "vram_gib": None},
        ]

    def test_not_json_raises(self):
        with pytest.raises(ValueError):
            mh.parse_ollama_ps(b"zzz")


class TestParseMetrics:
    def test_vllm(self):
        text = b"""
# HELP vllm:gpu_cache_usage_perc gpu cache
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model="model-a"} 0.25
vllm:gpu_cache_usage_perc{model="model-b"} 0.5
vllm:num_requests_running 1
vllm:num_requests_running 2
vllm:gpu_cache_usage_perc_total 0.9
"""
        out = mh.parse_metrics(text, "vLLM")
        assert out == {"kv_cache_pct": 50.0, "running": 3}

    def test_vllm_fraction_above_one_and_nan_ignored(self):
        text = b"vllm:gpu_cache_usage_perc 1.5\nvllm:gpu_cache_usage_perc NaN\nvllm:num_requests_running 2"
        assert mh.parse_metrics(text, "vLLM") == {"running": 2}

    def test_sglang(self):
        text = b"sglang:token_usage 0.125\nsglang:num_running_reqs 4"
        assert mh.parse_metrics(text, "SGLang") == {"kv_cache_pct": 12.5, "running": 4}

    def test_llamacpp(self):
        text = b"llamacpp:kv_cache_usage_ratio 0.3\nllamacpp:requests_processing 1"
        out = mh.parse_metrics(text, "llama.cpp")
        assert out == {"kv_cache_pct": 30.0, "running": 1}

    def test_unknown_engine(self):
        assert mh.parse_metrics(b"vllm:num_requests_running 1", "weird") == {}

    def test_no_matching_lines(self):
        assert mh.parse_metrics(b"other:metric 1\n", "vLLM") == {}

    def test_malformed_lines_ignored(self):
        text = b'vllm:num_requests_running {oops\nvllm:num_requests_running abc\nvllm:num_requests_running -1\n'
        assert mh.parse_metrics(text, "vLLM") == {}


# ---------------------------------------------------------------- read_host


class FakeFetch:
    def __init__(self, replies=None, error=None):
        self.replies = dict(replies or {})
        self.error = error or {}
        self.calls: list[str] = []

    def __call__(self, url, path):
        self.calls.append(path)
        if path in self.error:
            raise self.error[path]
        return self.replies[path]


def _http_error(url, code):
    return urllib.error.HTTPError(url, code, "x", {}, None)


class TestReadHost:
    def _endpoint(self, engine):
        return {"url": "http://192.0.2.5:8000", "engine": engine}

    def test_vllm_call_order_and_result(self):
        fetch = FakeFetch(
            {
                "/v1/models": _b({"data": [{"id": "org/model-a"}]}),
                "/health": b"OK",
                "/metrics": b"vllm:num_requests_running 2\n",
            }
        )
        info = mh.read_host(self._endpoint("vLLM"), fetch)
        assert fetch.calls == ["/v1/models", "/health", "/metrics"]
        assert info["served"] == ["org/model-a"]
        assert info["health"] == "ok"
        assert info["metrics"] == {"running": 2}
        assert info["available"] == []
        assert info["loaded"] == []

    def test_vllm_health_503_loading_metrics_404(self):
        fetch = FakeFetch(
            {"/v1/models": _b({"data": [{"id": "model-a"}]})},
            {"/health": _http_error("http://192.0.2.5:8000", 503), "/metrics": _http_error("http://192.0.2.5:8000", 404)},
        )
        info = mh.read_host(self._endpoint("vLLM"), fetch)
        assert info["health"] == "loading"
        assert info["metrics"] == {}
        assert info["served"] == ["model-a"]

    def test_llamacpp_metrics_501_still_served(self):
        fetch = FakeFetch(
            {"/v1/models": _b({"data": [{"id": "model-a"}]}), "/health": b'{"status":"ok"}'},
            {"/metrics": _http_error("http://192.0.2.5:8000", 501)},
        )
        info = mh.read_host(self._endpoint("llama.cpp"), fetch)
        assert info["served"] == ["model-a"]
        assert info["metrics"] == {}

    def test_ollama_paths_and_result(self):
        fetch = FakeFetch(
            {
                "/api/tags": _b({"models": [{"name": "llama3:8b"}]}),
                "/api/ps": _b({"models": [{"name": "llama3:8b", "size": 5583457484, "size_vram": 5583457484}]}),
            }
        )
        info = mh.read_host(self._endpoint("Ollama"), fetch)
        assert fetch.calls == ["/api/tags", "/api/ps"]
        assert info["available"] == ["llama3:8b"]
        assert info["loaded"] == [{"name": "llama3:8b", "size_gib": 5.2, "vram_gib": 5.2}]
        assert info["served"] == []

    @pytest.mark.parametrize("engine", ["OpenAI-compatible", "", "weird"])
    def test_models_only_engines(self, engine):
        fetch = FakeFetch({"/v1/models": _b({"data": [{"id": "model-a"}]})})
        info = mh.read_host(self._endpoint(engine), fetch)
        assert fetch.calls == ["/v1/models"]
        assert info["served"] == ["model-a"]

    def test_first_path_urlerror_propagates(self):
        fetch = FakeFetch(error={"/v1/models": urllib.error.URLError("down")})
        with pytest.raises(urllib.error.URLError):
            mh.read_host(self._endpoint("vLLM"), fetch)

    def test_refused_from_later_path_propagates(self):
        fetch = FakeFetch(
            {"/v1/models": _b({"data": []}), "/health": b"OK"},
            {"/metrics": mh.Refused("no metrics for box-a")},
        )
        with pytest.raises(mh.Refused):
            mh.read_host(self._endpoint("vLLM"), fetch)

    def test_all_paths_are_reads(self):
        allowed = {"/v1/models", "/health", "/metrics", "/api/tags", "/api/ps"}
        for engine, paths in mh.PATHS.items():
            fetch = FakeFetch(
                {
                    "/v1/models": _b({"data": []}),
                    "/health": b"OK",
                    "/metrics": b"",
                    "/api/tags": _b({"models": []}),
                    "/api/ps": _b({"models": []}),
                }
            )
            mh.read_host({"url": "http://192.0.2.9:8000", "engine": engine}, fetch)
            assert set(fetch.calls) <= allowed
            assert fetch.calls == list(paths)


# ---------------------------------------------------------------- Poller


class FakeClock:
    def __init__(self, start=100.0):
        self.now = [start]

    def __call__(self):
        return self.now[0]


def make_poller(endpoints, fetch, now=None):
    clock = FakeClock(100.0 if now is None else now)
    wall = lambda: 1767312000.0 + clock.now[0]  # noqa: E731
    spawned: list = []
    poller = mh.Poller(endpoints, fetch=fetch, clock=clock, wall=wall, spawn=spawned.append)
    return poller, clock, spawned


def run_spawned(spawned):
    while spawned:
        fn = spawned.pop(0)
        fn()


class TestPoller:
    def _endpoint(self, **over):
        base = {"id": "box-a", "name": "Box A", "url": "http://192.0.2.5:8000", "engine": "vLLM"}
        base.update(over)
        return base

    def _ok_fetch(self):
        return FakeFetch(
            {
                "/v1/models": _b({"data": [{"id": "org/model-a"}]}),
                "/health": b"OK",
                "/metrics": b"vllm:num_requests_running 1\n",
            }
        )

    def test_initial_view_checking_and_address(self):
        poller, _clock, _spawned = make_poller([self._endpoint()], FakeFetch())
        view = poller.view()
        assert view[0]["state"] == "checking"
        assert view[0]["address"] == "192.0.2.5:8000"
        assert view[0]["name"] == "Box A"
        assert view[0]["last_seen"] is None
        assert view[0]["age_s"] is None

    def test_tick_spawns_once_and_in_flight_blocks_second(self):
        poller, _clock, spawned = make_poller([self._endpoint()], FakeFetch())
        poller.tick()
        assert len(spawned) == 1
        poller.tick()
        assert len(spawned) == 1

    def test_success_then_poll_interval(self):
        poller, clock, spawned = make_poller([self._endpoint()], self._ok_fetch())
        poller.tick()
        run_spawned(spawned)
        view = poller.view()
        assert view[0]["state"] == "reachable"
        assert view[0]["served"] == ["org/model-a"]
        assert view[0]["metrics"] == {"running": 1}
        assert view[0]["health"] == "ok"
        assert view[0]["last_seen"].endswith("Z")
        assert view[0]["age_s"] == 0
        clock.now[0] += 4
        poller.tick()
        assert spawned == []
        clock.now[0] += 1
        poller.tick()
        assert len(spawned) == 1

    def test_failure_backoff_and_no_stale_data(self):
        poller, clock, spawned = make_poller([self._endpoint()], self._ok_fetch())
        poller.tick()
        run_spawned(spawned)
        ok_view = poller.view()[0]
        last_seen = ok_view["last_seen"]

        failing = FakeFetch(error={"/v1/models": urllib.error.URLError("connection refused")})
        poller._fetch = failing
        clock.now[0] += 5
        poller.tick()
        run_spawned(spawned)
        view = poller.view()[0]
        assert view["state"] == "offline"
        assert "connection refused" in view["error"]
        assert view["served"] == []
        assert view["last_seen"] == last_seen

        clock.now[0] += 29
        poller.tick()
        assert spawned == []
        clock.now[0] += 1
        poller.tick()
        assert len(spawned) == 1

    def test_stale_result_becomes_offline(self):
        poller, clock, spawned = make_poller([self._endpoint()], self._ok_fetch())
        poller.tick()
        run_spawned(spawned)
        clock.now[0] += 61
        view = poller.view()[0]
        assert view["state"] == "offline"
        assert view["served"] == []

    def test_refused_state(self):
        exc = mh.Refused("box-a resolves to 127.0.0.1, which is not a private or tailnet address")
        poller, clock, spawned = make_poller([self._endpoint()], FakeFetch(error={"/v1/models": exc}))
        poller.tick()
        run_spawned(spawned)
        view = poller.view()[0]
        assert view["state"] == "refused"
        assert "127.0.0.1" in view["error"]
        clock.now[0] += mh.DOWN_S - 0.5
        poller.tick()
        assert spawned == []
        clock.now[0] += 0.5
        poller.tick()
        assert len(spawned) == 1

    def test_third_endpoint_not_polled(self):
        endpoints = [
            self._endpoint(),
            self._endpoint(id="box-b", url="http://192.0.2.6:8000"),
            self._endpoint(id="box-c", url="http://192.0.2.7:8000"),
        ]
        poller, _clock, spawned = make_poller(endpoints, self._ok_fetch())
        view = poller.view()
        assert view[2]["state"] == "not polled"
        assert poller.due() == ["box-a", "box-b"]
        poller.tick()
        assert len(spawned) == 2
        assert poller.due() == []
        run_spawned(spawned)
        assert poller.view()[2]["state"] == "not polled"

    def test_non_dict_and_duplicate_ids_skipped(self):
        endpoints = [
            "not-a-dict",
            self._endpoint(),
            self._endpoint(id="box-a", url="http://192.0.2.6:8000"),
            {"id": "", "url": "http://192.0.2.7"},
        ]
        poller, _clock, _spawned = make_poller(endpoints, FakeFetch())
        assert [ep["id"] for ep in poller._endpoints] == ["box-a"]
        assert len(poller._states) == 1

    def test_poll_unknown_id_is_safe(self):
        poller, _clock, _spawned = make_poller([self._endpoint()], FakeFetch())
        poller.poll("no-such-host")

    def test_view_address_ipv6_style_and_bad_port(self):
        poller, _clock, _spawned = make_poller(
            [self._endpoint(url="http://192.0.2.5:8000/notbase", engine="weird")], FakeFetch()
        )
        view = poller.view()
        assert view[0]["address"] == "192.0.2.5:8000"


# ---------------------------------------------------------------- module level


class TestModuleLevel:
    def test_view_empty_without_poller(self, monkeypatch):
        monkeypatch.setattr(mh, "_POLLER", None)
        assert mh.view() == []

    def test_start_with_no_endpoints_returns_none(self):
        assert mh.start([], threading.Event()) is None


def test_a_mirrored_card_never_carries_the_upstream_monitored_hosts():
    from sparksi import monitor_mirror

    upstream = {"hosts": [], "loaded": [], "monitored": [{"id": "elsewhere"}], "history": []}
    mirror = monitor_mirror.Mirror("http://127.0.0.1:7500", fallback=dict, fetch=lambda base: upstream)
    assert mirror.poll_once() is True
    assert "monitored" not in mirror.card()


def test_current_vllm_kv_cache_metric_name():
    text = b'vllm:kv_cache_usage_perc{engine="0",model_name="model-a"} 0.375\nvllm:num_requests_running{engine="0",model_name="model-a"} 0.0\n'
    assert mh.parse_metrics(text, "vLLM") == {"kv_cache_pct": 37.5, "running": 0}


def test_tick_never_starts_a_second_poll_for_a_host_in_flight():
    """Even when due() is stale (it ran before another tick claimed the host), tick() checks again."""
    spawned: list = []
    poller = mh.Poller(
        [{"id": "box-a", "name": "Box A", "url": "http://192.0.2.5:8000", "engine": "vLLM"}],
        fetch=lambda url, path: b'{"data":[]}',
        clock=lambda: 100.0,
        spawn=spawned.append,
    )
    poller.due = lambda: ["box-a"]  # a stale answer: the host is already in flight below
    poller.tick()
    poller.tick()
    poller.tick()
    assert len(spawned) == 1
