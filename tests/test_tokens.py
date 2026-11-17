# SPDX-License-Identifier: Apache-2.0

from sparksi.public import strip_secrets
from sparksi.tokens import (
    _parse_metrics,
    fmt_count,
    format_ledger,
    observe,
    report,
    sample_vllm,
    snapshot,
)


def test_fmt_count() -> None:
    assert fmt_count(12) == "12"
    assert fmt_count(1500) == "1.5k"
    assert fmt_count(1_200_000) == "1.20M"


def test_parse_metrics_vllm_prefix_and_labels() -> None:
    blob = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
vllm:prompt_tokens_total{engine="0",model_name="brain-a"} 59337.0
vllm:generation_tokens_total{engine="0",model_name="brain-a"} 442.0
vllm:request_success_total{engine="0",finished_reason="stop",model_name="brain-a"} 4.0
vllm:kv_cache_usage_perc{engine="0"} 0.01
"""
    got = _parse_metrics(blob)
    assert got["prompt_tokens_total"] == 59337.0
    assert got["generation_tokens_total"] == 442.0
    assert got["request_success_total"] == 4.0
    assert "kv_cache_usage_perc" not in got


def test_observe_credits_delta_and_survives_reset(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    a = observe("counsel-a", prompt=100, gen=10, requests=1, port=8011)
    assert a["engine_gen"] == 10
    assert a["life_gen"] == 0
    assert a["life_prompt"] == 0
    b = observe("counsel-a", prompt=150, gen=25, requests=2, port=8011)
    assert b["life_gen"] == 15
    assert b["life_prompt"] == 50
    # engine restart: the new counter is what has been counted since the reset
    c = observe("counsel-a", prompt=5, gen=3, requests=1, port=8011)
    assert c["engine_gen"] == 3
    assert c["life_gen"] == 18
    assert c["life_prompt"] == 55
    snap = snapshot()
    assert snap["counsel-a"]["life_gen"] == 18


def test_observe_isolates_models(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    observe("counsel-a", prompt=0, gen=0, requests=1)
    observe("counsel-a", prompt=10, gen=2, requests=1)
    observe("brain-a", prompt=0, gen=0, requests=1)
    observe("brain-a", prompt=99, gen=40, requests=4)
    snap = snapshot()
    assert snap["counsel-a"]["life_gen"] == 2
    assert snap["brain-a"]["life_gen"] == 40
    text = format_ledger(snap)
    assert "counsel-a" in text and "brain-a" in text
    assert "2" in text and "40" in text


def test_strip_secrets_keeps_usage_counts() -> None:
    raw = {
        "models": {
            "counsel-a": {
                "life_prompt": 153300,
                "life_gen": 4045,
                "tok_label": "4.0k out · 153.3k in",
            }
        },
        "token": "SECRET",
        "unlock_token": "NOPE",
    }
    out = strip_secrets(raw)
    blob = str(out)
    assert "SECRET" not in blob
    assert "NOPE" not in blob
    assert out["models"]["counsel-a"]["life_gen"] == 4045
    assert out["models"]["counsel-a"]["tok_label"].startswith("4.0k")


def test_report_seeds_seated_zeros(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    rows = report(sample=True)
    for cid in ("brain-a", "vision-a", "counsel-a"):
        assert cid in rows
        assert rows[cid]["life_gen"] == 0
        assert rows[cid]["life_prompt"] == 0
        assert rows[cid]["tok_label"] == "0 out · 0 in"
        assert rows[cid]["seated"] is True
        assert rows[cid]["port"] in (8007, 8004, 8011)


def test_sample_vllm_skips_closed_ports(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    sample_vllm()
    assert snapshot() == {}
