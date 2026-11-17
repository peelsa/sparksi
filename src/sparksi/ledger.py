# SPDX-License-Identifier: Apache-2.0

"""One counter-delta token ledger for every running model.

Each poll reads Prometheus ``/metrics`` on the Spark and port stored in
the run record (a split run is polled on the head and the worker). The
first reading of a counter is a baseline and is not added. Later readings
add the increase. A drop is a reset: the new reading is what the engine
has counted since it restarted, which is the same rule Prometheus
``rate()`` uses. Reads never write. The poll loop is the only writer.

The counter-delta method follows SparkDash, Apache-2.0 from version 2.0.0.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from sparksi import protect
from sparksi.config import CFG, env, state_subdir
from sparksi.placement import roles_of

POLL_SECONDS = 15.0
# A rate older than this is not "live". Three missed polls.
_RATE_FRESH_S = POLL_SECONDS * 3
_QUIET = frozenset({"stopped", "failed", "timed_out", "cancelled"})
_ACTIVE = frozenset({"starting", "waiting_health", "ready"})
_PROMPT = frozenset({
    "prompt_tokens_total",
    "prompt_tokens",
    "tokens_prompt_total",
})
_GEN = frozenset({
    "generation_tokens_total",
    "generated_tokens_total",
    "tokens_predicted_total",
})
_PREFIXES = ("vllm:", "tensorfold:", "sglang:", "llama:", "llamacpp:")

_LOCK = threading.Lock()
Fetch = Callable[[str, int], str]


def ledger_path() -> Path:
    """Where the ledger file lives. A read must not create the directory."""
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "token-ledger.json"
    return state_subdir(Path.home() / ".local" / "state") / "token-ledger.json"


def local_now() -> datetime:
    """Controller local time. The zone database supplies the DST offset."""
    return datetime.now().astimezone()


def _aware(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.astimezone()
    return moment.astimezone(moment.tzinfo)


def calendar_day(moment: datetime) -> datetime:
    """Midnight at the start of this local calendar day."""
    local = _aware(moment)
    return local.replace(hour=0, minute=0, second=0, microsecond=0)


def next_calendar_day(moment: datetime) -> datetime:
    """Midnight at the start of the next local calendar day.

    Adding 24 hours is wrong on a 23-hour or 25-hour civil day. Walk forward
    until the local date changes, then take that midnight.
    """
    start = calendar_day(moment)
    cursor = start + timedelta(hours=1)
    while calendar_day(cursor) == start:
        cursor += timedelta(hours=1)
        if cursor - start > timedelta(hours=30):
            break
    return calendar_day(cursor)


def previous_calendar_day(moment: datetime) -> datetime:
    """Midnight of the previous local calendar day, across a DST change."""
    start = calendar_day(moment)
    return calendar_day(start - timedelta(seconds=1))


def day_key(moment: datetime | None = None) -> str:
    return f"{calendar_day(moment or local_now()):%Y-%m-%d}"


def _empty() -> dict[str, Any]:
    return {"series": {}, "days": {}, "lifetime": {}, "rates": {}}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return _empty()
    if not isinstance(data, dict):
        return _empty()
    if "series" not in data and isinstance(data.get("models"), dict):
        # A ledger written by the previous sampler. Keep its lifetime totals.
        # Do not replay the engine counters: that would credit them again.
        life: dict[str, dict[str, int]] = {}
        for mid, row in data["models"].items():
            if not isinstance(row, dict):
                continue
            life[str(mid)] = {
                "prompt": max(0, int(row.get("life_prompt") or 0)),
                "generation": max(0, int(row.get("life_gen") or 0)),
            }
        return {"series": {}, "days": {}, "lifetime": life, "rates": {}}
    out = _empty()
    if isinstance(data.get("series"), dict):
        out["series"] = data["series"]
    if isinstance(data.get("days"), dict):
        out["days"] = data["days"]
    if isinstance(data.get("lifetime"), dict):
        out["lifetime"] = data["lifetime"]
    if isinstance(data.get("rates"), dict):
        out["rates"] = data["rates"]
    return out


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_metrics(text: str) -> dict[str, float]:
    """Sum prompt and generation counters. Labels and engine prefixes are ignored."""
    totals = {"prompt": 0.0, "generation": 0.0}
    saw = {"prompt": False, "generation": False}
    for line in (text or "").splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        name = parts[0].split("{", 1)[0]
        for prefix in _PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        kind = ""
        if name in _PROMPT:
            kind = "prompt"
        elif name in _GEN:
            kind = "generation"
        if not kind:
            continue
        try:
            totals[kind] += float(parts[1])
        except ValueError:
            continue
        saw[kind] = True
    if not saw["prompt"] and not saw["generation"]:
        return {}
    return totals


def _credit(previous: float | None, current: float) -> float:
    """Delta to add. The first sight of a counter adds nothing.

    A lower reading is a reset. The value after the reset is the increase
    since the engine restarted, which is what a rate over a reset counts.
    """
    if previous is None:
        return 0.0
    if current + 1e-9 >= previous:
        return current - previous
    return current


def fold(
    model_id: str,
    *,
    role: str,
    port: int,
    prompt: float,
    generation: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Fold one sample into the ledger. Returns the model's totals."""
    moment = now or local_now()
    mid = str(model_id or "")
    side = role if role in CFG.roles else "head"
    try:
        number = int(port)
    except (TypeError, ValueError):
        number = 0
    prompt_n = max(0.0, float(prompt))
    gen_n = max(0.0, float(generation))
    path = ledger_path()
    with _LOCK:
        data = _load(path)
        added_gen = 0.0
        for kind, current in (("prompt", prompt_n), ("generation", gen_n)):
            key = f"{mid}|{side}|{number}|{kind}"
            row = data["series"].get(key) if isinstance(data["series"].get(key), dict) else {}
            previous = row.get("value") if row.get("seen") else None
            try:
                previous_n = float(previous) if previous is not None else None
            except (TypeError, ValueError):
                previous_n = None
            added = _credit(previous_n, current)
            data["series"][key] = {"value": current, "seen": True}
            if kind == "generation":
                added_gen = added
            life = data["lifetime"].setdefault(mid, {"prompt": 0, "generation": 0})
            life[kind] = int(life.get(kind) or 0) + int(round(added))
            bucket = data["days"].setdefault(day_key(moment), {})
            model = bucket.setdefault(mid, {"prompt": 0, "generation": 0})
            model[kind] = int(model.get(kind) or 0) + int(round(added))
        prior = data["rates"].get(mid) if isinstance(data["rates"].get(mid), dict) else {}
        try:
            last_at = float(prior.get("at") or 0)
            span = float(prior.get("span") or 0)
            prior_rate = float(prior.get("tok_s") or 0)
        except (TypeError, ValueError):
            last_at = span = prior_rate = 0.0
        stamp = moment.timestamp()
        # Two Sparks of one split are folded in the same poll. The second
        # sample shares the timestamp, so its tokens add to the same rate.
        if last_at and stamp > last_at + 0.5:
            span = stamp - last_at
            rate = added_gen / span
        elif last_at and span > 0:
            rate = prior_rate + added_gen / span
        else:
            rate = 0.0
        data["rates"][mid] = {"tok_s": rate, "at": stamp, "span": span}
        _write(path, data)
        life = data["lifetime"].get(mid) or {}
        return {
            "id": mid,
            "port": number or None,
            "engine_prompt": int(prompt_n),
            "engine_gen": int(gen_n),
            "life_prompt": int(life.get("prompt") or 0),
            "life_gen": int(life.get("generation") or 0),
            "updated_at": moment.isoformat(timespec="seconds"),
        }


def read() -> dict[str, Any]:
    """The stored ledger. Does not sample and does not write."""
    with _LOCK:
        return _load(ledger_path())


def lifetimes() -> dict[str, dict[str, int]]:
    found: dict[str, dict[str, int]] = {}
    for mid, row in (read().get("lifetime") or {}).items():
        if not isinstance(row, dict):
            continue
        found[str(mid)] = {
            "prompt": int(row.get("prompt") or 0),
            "generation": int(row.get("generation") or 0),
        }
    return found


def decode_tok_s(now: datetime | None = None) -> float | None:
    """Sum of fresh generation rates, or None when nothing is generating."""
    moment = now or local_now()
    stamp = moment.timestamp()
    total = 0.0
    live = False
    for row in (read().get("rates") or {}).values():
        if not isinstance(row, dict):
            continue
        try:
            at = float(row.get("at") or 0)
            rate = float(row.get("tok_s") or 0)
        except (TypeError, ValueError):
            continue
        if stamp - at > _RATE_FRESH_S:
            continue
        if rate > 0.05:
            live = True
            total += rate
    if not live:
        return None
    return total


def day_view(now: datetime | None = None) -> dict[str, Any]:
    """Today and yesterday from the ledger. Does not sample and does not write."""
    moment = now or local_now()
    data = read()
    days_stored = data.get("days") if isinstance(data.get("days"), dict) else {}
    today = day_key(moment)
    yesterday = day_key(previous_calendar_day(moment))
    days = []
    for label, key in (("today", today), ("yesterday", yesterday)):
        bucket = days_stored.get(key) if isinstance(days_stored.get(key), dict) else {}
        lines = []
        totals = {"prompt": 0, "generation": 0}
        for mid in sorted(bucket):
            row = bucket[mid] if isinstance(bucket[mid], dict) else {}
            prompt = int(row.get("prompt") or 0)
            generation = int(row.get("generation") or 0)
            totals["prompt"] += prompt
            totals["generation"] += generation
            lines.append({
                "model": mid,
                "scraped": True,
                "fresh": False,
                "prompt": str(prompt),
                "generation": str(generation),
                "source": "ledger",
            })
        days.append({
            "date": key,
            "label": label,
            "window": label,
            "tz": moment.tzname() or "",
            "lines": lines,
            "total": {name: str(value) for name, value in totals.items()},
        })
    rate = decode_tok_s(moment)
    return {
        "ok": True,
        "window": "today",
        "tz": moment.tzname() or "",
        "decode_tok_s": rate,
        "days": days,
    }


def targets_from_runs(runs: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """One target per Spark a live run is actually on.

    The port is the run's serve port. A split run contributes the head and
    the worker, so a worker that is not its own monitor row is still counted.
    """
    if runs is None:
        from sparksi.recipes.runs import list_runs

        runs = list_runs()
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    for run in runs:
        if not isinstance(run, dict):
            continue
        phase = str(run.get("phase") or "")
        if phase in _QUIET or phase not in _ACTIVE:
            continue
        try:
            port = int(run.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if port <= 0 or protect.is_off_limits(port):
            continue
        model_id = str(run.get("model_id") or "")
        if not model_id:
            continue
        nodes = [node for node in (run.get("nodes") or []) if isinstance(node, dict)]
        if not nodes:
            placement = str(run.get("placement") or "head")
            roles = roles_of(placement) or ("head",)
            nodes = []
            for role in roles:
                try:
                    host = CFG.node_by_role(role).name
                except Exception:  # noqa: BLE001
                    host = role
                nodes.append({"role": role, "host": host})
        for node in nodes:
            role = str(node.get("role") or "head")
            if role not in CFG.roles:
                role = "head"
            host = str(node.get("host") or "")
            try:
                node_obj = CFG.node_by_role(role)
                address = str(node_obj.tailscale or host or node_obj.name)
            except Exception:  # noqa: BLE001
                address = host
            key = (model_id, role, port)
            if not address or key in seen:
                continue
            seen.add(key)
            found.append({
                "model_id": model_id,
                "role": role,
                "host": host,
                "address": address,
                "port": port,
            })
    return found


def _fetch(address: str, port: int) -> str:
    url = f"http://{address}:{int(port)}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=2.0) as resp:
            return resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        return ""


def poll_once(
    runs: list[dict[str, Any]] | None = None,
    fetch: Fetch | None = None,
    now: datetime | None = None,
) -> int:
    """Sample every live run target. Returns how many readings were folded."""
    getter = fetch or _fetch
    moment = now or local_now()
    folded = 0
    for target in targets_from_runs(runs):
        text = getter(str(target["address"]), int(target["port"]))
        metrics = parse_metrics(text)
        if not metrics:
            continue
        fold(
            str(target["model_id"]),
            role=str(target["role"]),
            port=int(target["port"]),
            prompt=float(metrics.get("prompt") or 0),
            generation=float(metrics.get("generation") or 0),
            now=moment,
        )
        folded += 1
    return folded
