# SPDX-License-Identifier: Apache-2.0

"""Day token counts for engines with no Prometheus counters (llama-server).

A node's announcer reads llama-server's own log and sends today's and
yesterday's prompt/generation counts for the running process, keyed by the
process start time. Here each (date, spark, model, process) keeps the largest
value seen, so a restart adds a new process row instead of resetting the day,
and a controller restart loses nothing the Spark can still recount.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sparksi.config import CFG, env, state_subdir

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {"data": None, "dirty": False, "saved": 0.0}
KEEP_DAYS = 14
HOSTS = {n.id: n.name for n in CFG.nodes}


def ledger_path() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "llama-day.json"
    return state_subdir(Path.home() / ".local" / "state") / "llama-day.json"


def _load() -> dict[str, Any]:
    if _STATE["data"] is not None:
        return _STATE["data"]
    data: dict[str, Any] = {"days": {}}
    try:
        raw = json.loads(ledger_path().read_text())
        if isinstance(raw, dict) and isinstance(raw.get("days"), dict):
            data = raw
    except (OSError, ValueError):
        pass
    _STATE["data"] = data
    return data


def _save(force: bool = False) -> None:
    if not _STATE["dirty"]:
        return
    now = time.monotonic()
    if not force and now - _STATE["saved"] < 10.0:
        return
    path = ledger_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_STATE["data"], indent=1, sort_keys=True) + "\n")
        tmp.replace(path)
        _STATE["dirty"] = False
        _STATE["saved"] = now
    except OSError:
        pass


def _date(n: int) -> str | None:
    s = str(int(n))
    if len(s) != 8:
        return None
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"


def _bump(days: dict, date: str, key: str, ps: str, prompt: int, gen: int) -> bool:
    row = days.setdefault(date, {}).setdefault(key, {})
    old = row.get(ps) or [0, 0]
    new = [max(int(old[0]), int(prompt)), max(int(old[1]), int(gen))]
    if new != old:
        row[ps] = new
        return True
    return False


def note(host: str, http: dict) -> None:
    """Fold one announced packet's llama day counts into the ledger."""
    spark = HOSTS.get(host)
    if not spark or not isinstance(http, dict):
        return
    with _LOCK:
        data = _load()
        days = data.setdefault("days", {})
        changed = False
        for _port, rec in http.items():
            if not isinstance(rec, dict):
                continue
            m = rec.get("metrics") or {}
            served = rec.get("served")
            if not served or "llama_today_date" not in m or "llama_process_start" not in m:
                continue
            today = _date(m["llama_today_date"])
            if not today:
                continue
            key = f"{spark}|{served}"
            ps = str(int(m["llama_process_start"]))
            changed |= _bump(days, today, key, ps,
                             m.get("llama_today_prompt_tokens") or 0,
                             m.get("llama_today_generation_tokens") or 0)
            yday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
            if m.get("llama_yesterday_prompt_tokens") or m.get("llama_yesterday_generation_tokens"):
                changed |= _bump(days, yday, key, ps,
                                 m.get("llama_yesterday_prompt_tokens") or 0,
                                 m.get("llama_yesterday_generation_tokens") or 0)
        if changed:
            for old in sorted(days)[:-KEEP_DAYS]:
                days.pop(old, None)
            _STATE["dirty"] = True
        _save()


def rows_for(date: str) -> list[tuple[str, str, int, int]]:
    """(spark, model, prompt, generation) for a local date, summed over processes."""
    with _LOCK:
        data = _load()
        _save()
        day = dict((data.get("days") or {}).get(date) or {})
    out = []
    for key, procs in sorted(day.items()):
        spark, _, model = key.partition("|")
        p = sum(int(v[0]) for v in procs.values())
        g = sum(int(v[1]) for v in procs.values())
        out.append((spark, model, p, g))
    return out


def reset_for_tests() -> None:
    with _LOCK:
        _STATE.update({"data": None, "dirty": False, "saved": 0.0})
