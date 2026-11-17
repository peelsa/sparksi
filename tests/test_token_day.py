# SPDX-License-Identifier: Apache-2.0

from datetime import datetime
from zoneinfo import ZoneInfo

from sparksi.token_day import build_report, format_report, lifetime_report


def _query(expr, at):
    if 'spark="spark-b"' in expr:
        return []
    if "157680000" in expr:
        return [({"model_name": "kept"}, 10.0), ({"model_name": "fresh"}, 1.0)]
    if "count_over_time" in expr:
        return [({"model_name": "kept"}, 4.0), ({"model_name": "fresh"}, 1.0)]
    if "prompt_tokens_total" in expr:
        return [({"model_name": "kept"}, 3.0), ({"model_name": "fresh"}, 9.0)]
    if "generation_tokens_total" in expr:
        return [({"model_name": "kept"}, 1.0), ({"model_name": "fresh"}, 8.0)]
    return []


def test_queries_cover_vllm_and_tensorfold_counters():
    seen = []

    def query(expr, at):
        seen.append(expr)
        return []

    now = datetime(2026, 10, 2, 12, 0, tzinfo=ZoneInfo("America/Chicago"))
    body = build_report(now, query)
    assert body["ok"] is True
    prompt = [e for e in seen if "prompt_tokens_total" in e]
    assert prompt
    assert all('__name__=~"vllm:prompt_tokens_total|tensorfold:prompt_tokens_total"' in e for e in prompt)
    assert any("tensorfold:generation_tokens_total" in e for e in seen)


def test_not_scraped_and_fresh_stay_out_of_the_total():
    now = datetime(2026, 9, 27, 1, 0, tzinfo=ZoneInfo("America/Chicago"))
    body = build_report(now, _query)
    assert body["ok"] is True
    today = body["days"][0]
    assert today["date"] == "2026-09-27"
    worker = [row for row in today["lines"] if row["spark"] == "spark-b"]
    assert worker == [{"spark": "spark-b", "scraped": False}]
    fresh = [row for row in today["lines"] if row.get("model") == "fresh"][0]
    assert fresh["fresh"] is True
    assert today["total"] == {"prompt": "3", "generation": "1"}
    text = format_report(body)
    assert "spark-b      not scraped" in text
    assert "collection just started" in text
    assert "prompt=3  generation=1" in text


def test_lifetime_sums_head_series_and_skips_an_unscraped_spark():
    def query(expr, at):
        assert "role=\"head\"" in expr
        assert "increase(" in expr
        if 'spark="spark-b"' in expr:
            return []
        if "prompt_tokens_total" in expr:
            return [({"model_name": "a"}, 59.0), ({"model_name": "b"}, 0.0)]
        return [({"model_name": "a"}, 30.0)]

    body = lifetime_report(query=query)
    assert body["sparks"]["spark-a"] == {"scraped": True, "prompt": "59", "generation": "30"}
    assert body["sparks"]["spark-b"] == {"scraped": False}


def test_llama_log_rows_join_the_day_and_the_total():
    def query(expr, at):
        if 'spark="spark-b"' in expr:
            if "count_over_time" in expr and "157680000" not in expr:
                return [({"model_name": "qwen"}, 50.0)]
            if "157680000" in expr:
                return [({"model_name": "qwen"}, 500.0)]
            if "prompt_tokens_total" in expr:
                return [({"model_name": "qwen"}, 10.0)]
            return [({"model_name": "qwen"}, 20.0)]
        return []

    def extra(date):
        assert date == "2026-10-19"
        return [("spark-a", "gguf", 1000, 30)]

    now = datetime(2026, 10, 19, 12, 0, tzinfo=ZoneInfo("America/Chicago"))
    body = build_report(now, query, lambda d: extra(d) if d == "2026-10-19" else [])
    today = body["days"][0]
    head = [r for r in today["lines"] if r["spark"] == "spark-a"]
    assert head == [{
        "spark": "spark-a", "scraped": True, "model": "gguf",
        "prompt": "1000", "generation": "30", "fresh": False, "source": "llama-log",
    }]
    assert today["total"]["prompt"] == fmt_total(1000 + 10 * 12)
    yday = body["days"][1]
    assert {"spark": "spark-a", "scraped": False} in yday["lines"]
    assert "from llama-server log" in format_report(body)


def fmt_total(n):
    return str(n)
