# SPDX-License-Identifier: Apache-2.0

"""Announcer: same output, less work per 3 s sample."""

import pytest

from sparksi import announce


def _old_metrics(text: str) -> dict[str, float]:
    """The parser before this change, on the module's own name table."""
    _num = announce._num
    _normalize_metric = announce._normalize_metric
    _SHORT_BY_FULL = announce._SHORT_BY_FULL
    metrics: dict[str, float] = {}
    live_gen = None
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        raw = parts[0].split("{", 1)[0]
        val = _num(parts[1])
        if val is None:
            continue
        if raw == "tensorfold_health:completion_tokens_total":
            live_gen = val
            continue
        short = _SHORT_BY_FULL.get(_normalize_metric(raw))
        if not short:
            continue
        metrics[short] = round(metrics.get(short, 0.0) + val, 4)
    if live_gen is not None:
        metrics["gt"] = round(live_gen, 4)
    return metrics


PROM_TEXT = """# HELP vllm:prompt_tokens_total help
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 12.5
vllm:prompt_tokens_total{model="m"} 3.5
vllm:generation_tokens_total 7
vllm:num_requests_running 2
vllm:e2e_request_latency_seconds_bucket{le="0.1"} 1
vllm:e2e_request_latency_seconds_bucket{le="1"} 4
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 9
tensorfold:prompt_tokens_total 5
tensorfold_health:completion_tokens_total 42
vllm:num_requests_waiting not_a_float\nvllm:num_requests_waiting 3
trailing_no_value
"""


class _Proc:
    def __init__(self, stdout: str, returncode: int = 0):
        self.stdout = stdout
        self.returncode = returncode


@pytest.fixture(autouse=True)
def _clear_cache():
    announce._INSPECT_CACHE.clear()
    yield
    announce._INSPECT_CACHE.clear()


def test_metrics_matches_old():
    got = announce._metrics_from_prom(PROM_TEXT)
    assert got == _old_metrics(PROM_TEXT)
    assert got["pt"] == 21.0 and got["gt"] == 42.0 and got["r"] == 2.0 and got["w"] == 3.0


def test_engines_inspects_each_id_once(monkeypatch):
    calls = []

    def fake_run(cmd, capture_output, text, timeout):
        calls.append(list(cmd))
        if cmd[0] == "docker" and cmd[1] == "ps":
            return _Proc("abc123 web\ndef456 api\n")
        return _Proc("PORT=8000\n---CMD---\n")

    monkeypatch.setattr(announce.subprocess, "run", fake_run)
    dock1, eng1 = announce._engines()
    dock2, eng2 = announce._engines()
    inspects = [c for c in calls if c[:2] == ["docker", "inspect"]]
    assert len(inspects) == 2
    assert {c[-1] for c in inspects} == {"abc123", "def456"}
    assert dock1 == [["web", "Up"], ["api", "Up"]]
    assert eng1 == [
        ["web", 8000, None] if False else ["web", 8000, eng1[0][2]],
        ["api", 8000, eng2[0][2] if False else eng1[1][2]],
    ]
    assert all(row[1] == 8000 for row in eng1 + eng2)


def test_engines_drops_gone_id(monkeypatch):
    state = {"ids": "abc123 web\n"}

    def fake_run(cmd, capture_output, text, timeout):
        if cmd[1] == "ps":
            return _Proc(state["ids"])
        return _Proc("PORT=8000\n---CMD---\n")

    monkeypatch.setattr(announce.subprocess, "run", fake_run)
    announce._engines()
    assert "abc123" in announce._INSPECT_CACHE
    state["ids"] = "def456 api\n"
    announce._engines()
    assert "abc123" not in announce._INSPECT_CACHE
    assert "def456" in announce._INSPECT_CACHE


def test_engines_failed_inspect_not_cached(monkeypatch):
    flips = {"n": 0}

    def fake_run(cmd, capture_output, text, timeout):
        if cmd[1] == "ps":
            return _Proc("abc123 web\n")
        flips["n"] += 1
        if flips["n"] == 1:
            return _Proc("", returncode=1)
        return _Proc("PORT=8000\n---CMD---\n")

    monkeypatch.setattr(announce.subprocess, "run", fake_run)
    dock, eng = announce._engines()
    assert eng == []
    assert "abc123" not in announce._INSPECT_CACHE
    dock, eng = announce._engines()
    assert eng and eng[0][1] == 8000
    assert "abc123" in announce._INSPECT_CACHE


def test_engines_docker_ps_oserror(monkeypatch):
    def fake_run(cmd, capture_output, text, timeout):
        raise OSError("no docker")

    monkeypatch.setattr(announce.subprocess, "run", fake_run)
    assert announce._engines() == ([], [])
