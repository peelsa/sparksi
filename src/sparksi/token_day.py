# SPDX-License-Identifier: Apache-2.0

"""Optional VictoriaMetrics query helpers.

The dashboard's today and lifetime totals come from the counter-delta ledger
in ``sparksi.ledger``. These query helpers stay for tests of the old
hour-window math. ``cached_report`` and ``cached_lifetime`` read the ledger
and do not query or write.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from typing import Any, Callable

from sparksi.config import CFG

BASE = str(CFG.controller.get("metrics_url") or "http://127.0.0.1:8428")
SPARKS = tuple(CFG.node_names)
# Suffixes shared by vLLM (`vllm:`) and TensorFold (`tensorfold:`).
COUNTERS = (
    ("prompt", "prompt_tokens_total"),
    ("generation", "generation_tokens_total"),
)
LIFE_WINDOW = 157680000
Query = Callable[[str, datetime], list[tuple[dict, float]]]
Extra = Callable[[str], list[tuple[str, str, float, float]]]


def local_now() -> datetime:
    return datetime.now().astimezone()


def midnight(day: datetime) -> datetime:
    return day.replace(hour=0, minute=0, second=0, microsecond=0)


def query_vector(expr: str, at: datetime) -> list[tuple[dict, float]]:
    params = urllib.parse.urlencode({"query": expr, "time": f"{at.timestamp():.3f}"})
    req = urllib.request.Request(f"{BASE}/api/v1/query?{params}")
    from sparksi import monitor_mirror

    # No proxy and no redirects: the metrics address is only ever that one host.
    with monitor_mirror.no_redirect_opener().open(req, timeout=10) as resp:
        body = json.load(resp)
    if body.get("status") != "success":
        raise RuntimeError(body.get("error") or "query failed")
    out = []
    for row in body["data"]["result"]:
        out.append((row.get("metric") or {}, float(row["value"][1])))
    return out


def hour_windows(start: datetime, end: datetime):
    t = start
    while t < end:
        nxt = t + timedelta(hours=1)
        if nxt > end:
            nxt = end
        dur = (nxt - t).total_seconds()
        if dur >= 1:
            yield nxt, int(dur)
        if nxt == t:
            break
        t = nxt


def by_model(metric: str, spark: str, at: datetime, seconds: int, fn: str, query: Query) -> dict[str, float]:
    expr = (
        f"sum by (model_name) ({fn}("
        f'{{__name__=~"vllm:{metric}|tensorfold:{metric}",'
        f'spark="{spark}",role="head"}}[{seconds}s]))'
    )
    totals: dict[str, float] = {}
    for labels, value in query(expr, at):
        name = labels.get("model_name") or "(unnamed)"
        totals[name] = totals.get(name, 0.0) + value
    return totals


def fmt(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.3f}"


def _day(label: str, day: datetime, now: datetime, query: Query, extra: Extra | None = None) -> dict[str, Any]:
    start = midnight(day)
    end = min(midnight(day) + timedelta(days=1), now)
    span = max(1, int((end - start).total_seconds())) if end > start else 1
    at = end if end > start else now
    lines: list[dict[str, Any]] = []
    totals = {name: 0.0 for name, _ in COUNTERS}
    side: dict[str, list[tuple[str, float, float]]] = {}
    if extra is not None:
        for spark, model, p, g in extra(f"{start:%Y-%m-%d}"):
            side.setdefault(spark, []).append((model, float(p), float(g)))
    for spark in SPARKS:
        present: dict[str, float] = {}
        for _, metric in COUNTERS:
            present.update(by_model(metric, spark, at, span, "count_over_time", query))
        logged = side.get(spark) or []
        for model, p, g in logged:
            lines.append({
                "spark": spark,
                "scraped": True,
                "model": model,
                "prompt": fmt(p),
                "generation": fmt(g),
                "fresh": False,
                "source": "llama-log",
            })
            totals["prompt"] += p
            totals["generation"] += g
        if not present:
            if not logged:
                lines.append({"spark": spark, "scraped": False})
            continue
        lifetime = by_model(COUNTERS[1][1], spark, now, 157680000, "count_over_time", query)
        bucket = {model: {key: 0.0 for key, _ in COUNTERS} for model in present}
        if end > start:
            for window_end, seconds in hour_windows(start, end):
                for key, metric in COUNTERS:
                    found = by_model(metric, spark, window_end, seconds, "increase", query)
                    for model, value in found.items():
                        bucket.setdefault(model, {k: 0.0 for k, _ in COUNTERS})
                        bucket[model][key] += value
        for model in sorted(bucket):
            fresh = lifetime.get(model, present.get(model, 0)) < 2
            lines.append({
                "spark": spark,
                "scraped": True,
                "model": model,
                "prompt": fmt(bucket[model]["prompt"]),
                "generation": fmt(bucket[model]["generation"]),
                "fresh": fresh,
            })
            if fresh:
                continue
            for key, _ in COUNTERS:
                totals[key] += bucket[model][key]
    return {
        "date": f"{start:%Y-%m-%d}",
        "label": label,
        "tz": start.tzname() or "",
        "lines": lines,
        "total": {key: fmt(totals[key]) for key, _ in COUNTERS},
    }


def build_report(
    now: datetime | None = None, query: Query | None = None, extra: Extra | None = None
) -> dict[str, Any]:
    now = now or local_now()
    if extra is None and query is None:
        from sparksi.llama_day import rows_for as extra
    query = query or query_vector
    try:
        from sparksi import monitor_mirror

        with monitor_mirror.no_redirect_opener().open(f"{BASE}/health", timeout=5) as resp:
            resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if query is query_vector:
            return {"ok": False, "error": f"victoria-metrics is not answering on {BASE}: {exc}"}
    try:
        from sparksi.ledger import previous_calendar_day

        days = [
            _day("today", now, now, query, extra),
            _day("yesterday", previous_calendar_day(now), now, query, extra),
        ]
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "days": days}


def lifetime_report(now: datetime | None = None, query: Query | None = None) -> dict[str, Any]:
    """Head-series prompt and generation over the long window, per Spark."""
    now = now or local_now()
    query = query or query_vector
    sparks: dict[str, Any] = {}
    try:
        for spark in SPARKS:
            found = False
            totals = {key: 0.0 for key, _ in COUNTERS}
            for key, metric in COUNTERS:
                rows = by_model(metric, spark, now, LIFE_WINDOW, "increase", query)
                if rows:
                    found = True
                for value in rows.values():
                    totals[key] += value
            if not found:
                sparks[spark] = {"scraped": False}
            else:
                sparks[spark] = {
                    "scraped": True,
                    "prompt": fmt(totals["prompt"]),
                    "generation": fmt(totals["generation"]),
                }
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError, KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "sparks": sparks}


def cached_lifetime(ttl: float = 30.0) -> dict[str, Any]:
    """Lifetime totals from the counter-delta ledger. This does not query or write."""
    del ttl
    from sparksi.ledger import lifetimes

    models = {
        mid: {"prompt": str(row.get("prompt") or 0), "generation": str(row.get("generation") or 0)}
        for mid, row in lifetimes().items()
    }
    return {"ok": True, "models": models}


def cached_report(ttl: float = 30.0) -> dict[str, Any]:
    """Today and yesterday from the counter-delta ledger. This does not query or write."""
    del ttl
    from sparksi.ledger import day_view

    return day_view()


def format_report(body: dict[str, Any]) -> str:
    if not body.get("ok"):
        return str(body.get("error") or "token ledger unavailable")
    chunks = []
    for day in body.get("days") or []:
        lines = [f"{day.get('date')}  ({day.get('label')}, {day.get('tz')})"]
        for row in day.get("lines") or []:
            spark = f"{row.get('spark') or '':<12}"
            if row.get("scraped") is False:
                lines.append(f"  {spark} not scraped")
                continue
            bits = f"prompt={row.get('prompt')}  generation={row.get('generation')}"
            note = "  (collection just started)" if row.get("fresh") else ""
            if row.get("source") == "llama-log":
                note += "  (from llama-server log)"
            model = f"{row.get('model') or '':<24}"
            lines.append(f"  {spark} {model} {bits}{note}")
        total = day.get("total") or {}
        lines.append(f"  {'total':<12} prompt={total.get('prompt')}  generation={total.get('generation')}")
        chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def main() -> int:
    body = cached_report()
    if not body.get("ok"):
        import sys
        print(body.get("error") or "token ledger unavailable", file=sys.stderr)
        return 1
    print(format_report(body), end="")
    return 0
