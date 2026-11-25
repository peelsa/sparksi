# SPDX-License-Identifier: Apache-2.0

"""Context size and KV cache use of a running model server, read with safe GETs.

The controller asks the Spark that leads a ready run for ``/v1/models`` and
``/metrics`` (and, for llama.cpp, ``/props`` and ``/slots``) through
``monitor_hosts.get``: a private or tailnet address only, pinned, no redirects,
a short timeout and a size cap. Answers are cached for a few seconds and
refreshed in the background, so a page poll never waits for a server. Anything
missing or unreadable is simply left out.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable

from sparksi import monitor_hosts

TIMEOUT_S = 1.5  # each socket step
TOTAL_S = 2.5  # one whole GET
TTL_S = 10.0  # a good answer is reused this long
FAIL_TTL_S = 30.0  # a server that gave nothing is asked again after this long
MAX_URLS = 16

# Fractions 0..1 of the KV cache in use, by engine.
_KV_FRACTION = {
    "vllm:kv_cache_usage_perc": "vllm",
    "vllm:gpu_cache_usage_perc": "vllm",  # older vLLM
    "sglang:token_usage": "sglang",
    "llamacpp:kv_cache_usage_ratio": "llamacpp",
}
_RUNNING = {
    "vllm:num_requests_running": "vllm",
    "sglang:num_running_reqs": "sglang",
    "llamacpp:requests_processing": "llamacpp",
}
_KV_SIZE_INFO = "vllm:cache_config_info"
_KV_SIZE_SGLANG = "sglang:max_total_num_tokens"
_WANTED = frozenset(_KV_FRACTION) | frozenset(_RUNNING) | {_KV_SIZE_INFO, _KV_SIZE_SGLANG}

# name{labels} value: labels may hold quoted strings with escapes.
_LINE = re.compile(r'([A-Za-z_:][A-Za-z0-9_:]*)(\{(?:[^"}]|"(?:[^"\\]|\\.)*")*\})?\s+(\S+)')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')


def _json(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8", errors="replace"))
    except (ValueError, TypeError, AttributeError):
        return None


def _count(value: Any) -> int | None:
    """A positive whole number, or None (bools, floats with a fraction, text and zero are not)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and value > 0:
        return value
    return None


def parse_models(body: bytes) -> dict:
    """``ctx`` (max_model_len) and ``owner`` (owned_by) of the first model in /v1/models."""
    data = _json(body)
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        return {}
    first = items[0]
    out: dict = {}
    ctx = _count(first.get("max_model_len"))
    if ctx:
        out["ctx"] = ctx
    owner = first.get("owned_by")
    if isinstance(owner, str) and owner.strip():
        out["owner"] = owner.strip().lower()[:40]
    return out


def _labels(text: str) -> dict[str, str]:
    return {key: value for key, value in _LABEL.findall(text or "")}


def parse_metrics(body: bytes) -> dict:
    """``kv_pct``, ``running``, ``kv_tokens`` and ``engine`` from a Prometheus page."""
    try:
        text = body.decode("utf-8", errors="replace")
    except (UnicodeError, AttributeError):
        return {}
    fractions: list[float] = []
    running: list[float] = []
    kv_tokens: int | None = None
    engine = ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(None, 1)[0]
        if name not in _WANTED:
            continue
        hit = _LINE.match(line)
        if hit is None or hit.group(1) != name:
            continue
        try:
            value = float(hit.group(3))
        except ValueError:
            continue
        if not math.isfinite(value) or value < 0:
            continue
        if name in _KV_FRACTION:
            if value > 1:
                continue
            fractions.append(value)
            engine = engine or _KV_FRACTION[name]
        elif name in _RUNNING:
            running.append(value)
            engine = engine or _RUNNING[name]
        elif name == _KV_SIZE_INFO and kv_tokens is None:
            labels = _labels(hit.group(2) or "")
            size = _count(_int_text(labels.get("kv_cache_size_tokens")))
            if size is None:
                blocks = _count(_int_text(labels.get("num_gpu_blocks")))
                block = _count(_int_text(labels.get("block_size")))
                size = blocks * block if blocks and block else None
            kv_tokens = size
            engine = engine or "vllm"
        elif name == _KV_SIZE_SGLANG and kv_tokens is None:
            kv_tokens = _count(value)
            engine = engine or "sglang"
    out: dict = {}
    if fractions:
        out["kv_pct"] = round(max(fractions) * 100, 1)
    if running:
        out["running"] = int(sum(running))
    if kv_tokens:
        out["kv_tokens"] = kv_tokens
    if engine:
        out["engine"] = engine
    return out


def _int_text(text: Any) -> int | None:
    try:
        return int(str(text))
    except (TypeError, ValueError):
        return None


def parse_props(body: bytes) -> dict:
    """``ctx`` from llama.cpp /props: ``n_ctx``, else ``default_generation_settings.n_ctx``."""
    data = _json(body)
    if not isinstance(data, dict):
        return {}
    ctx = _count(data.get("n_ctx"))
    settings = data.get("default_generation_settings")
    if ctx is None and isinstance(settings, dict):
        ctx = _count(settings.get("n_ctx"))
    return {"ctx": ctx} if ctx else {}


def parse_slots(body: bytes) -> dict:
    """``kv_pct`` from llama.cpp /slots: tokens held over slot context, all slots together."""
    data = _json(body)
    if not isinstance(data, list):
        return {}
    used = 0
    total = 0
    for slot in data:
        if not isinstance(slot, dict):
            continue
        n_past = slot.get("n_past")
        n_ctx = slot.get("n_ctx")
        if isinstance(n_past, bool) or isinstance(n_ctx, bool):
            continue
        if not isinstance(n_past, int) or not isinstance(n_ctx, int) or n_past < 0 or n_ctx <= 0:
            continue
        used += n_past
        total += n_ctx
    if total <= 0:
        return {}
    return {"kv_pct": round(min(100.0, used * 100.0 / total), 1)}


Fetch = Callable[..., bytes]


def _get(fetch: Fetch, url: str, path: str) -> bytes | None:
    try:
        return fetch(url, path, timeout=TIMEOUT_S, total=TOTAL_S)
    except Exception:  # noqa: BLE001 - refused, down, slow, too large: all mean "no data"
        return None


def read(url: str, fetch: Fetch | None = None) -> dict:
    """``ctx``, ``kv_pct``, ``kv_tokens``, ``running`` and ``engine`` for the server at ``url``; {} when unknown."""
    fetch = fetch or monitor_hosts.get
    body = _get(fetch, url, "/v1/models")
    if body is None:
        return {}
    found = parse_models(body)
    owner = str(found.pop("owner", ""))
    metrics_body = _get(fetch, url, "/metrics")
    metrics = parse_metrics(metrics_body) if metrics_body is not None else {}
    found.update(metrics)
    llama = "llama" in owner or metrics.get("engine") == "llamacpp"
    if "ctx" not in found and (llama or not metrics):
        props = _get(fetch, url, "/props")
        if props is not None:
            found.update(parse_props(props))
    if "kv_pct" not in found and llama:
        slots = _get(fetch, url, "/slots")
        if slots is not None:
            found.update(parse_slots(slots))
    if "ctx" not in found and "kv_pct" not in found:
        return {}
    return {key: found[key] for key in ("ctx", "kv_pct", "kv_tokens", "running", "engine") if key in found}


def _spawn(fn: Callable[[], None]) -> None:
    threading.Thread(target=fn, daemon=True, name="sparksi-serving-info").start()


class Cache:
    """The last answer per server URL; a stale one is refreshed in the background, never waited for."""

    def __init__(
        self,
        read: Callable[[str], dict] = read,
        clock: Callable[[], float] = time.monotonic,
        spawn: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self._read = read
        self._clock = clock
        self._spawn = spawn or _spawn
        self._lock = threading.Lock()
        self._values: OrderedDict[str, tuple[dict, float]] = OrderedDict()
        self._running: set[str] = set()

    def get(self, url: str) -> dict:
        with self._lock:
            held = self._values.get(url)
            value = dict(held[0]) if held else {}
            if held is not None:
                ttl = TTL_S if held[0] else FAIL_TTL_S
                fresh = (self._clock() - held[1]) < ttl
            else:
                fresh = False
            start = not fresh and url not in self._running
            if start:
                self._running.add(url)
        if start:
            try:
                self._spawn(lambda: self._refresh(url))
            except Exception:  # noqa: BLE001 - no thread: try again on the next poll
                with self._lock:
                    self._running.discard(url)
        return value

    def _refresh(self, url: str) -> None:
        try:
            try:
                value = self._read(url)
            except Exception:  # noqa: BLE001 - a reader bug must not stop later refreshes
                value = {}
            with self._lock:
                self._values[url] = (dict(value) if isinstance(value, dict) else {}, self._clock())
                self._values.move_to_end(url)
                while len(self._values) > MAX_URLS:
                    self._values.popitem(last=False)
        finally:
            with self._lock:
                self._running.discard(url)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._running.clear()


CACHE = Cache()


def attach(body: dict, address_for: Callable[[str], str], cache: Cache | None = None) -> dict:
    """Add ``serving_info`` to each ready run and its ``serving`` rows when its server has reported it."""
    use = cache or CACHE
    runs = body.get("runs") if isinstance(body.get("runs"), list) else []
    serving = body.get("serving") if isinstance(body.get("serving"), list) else None
    found: dict[str, dict] = {}
    new_runs: list = []
    for run in runs:
        info = _run_info(run, address_for, use)
        if info:
            run = {**run, "serving_info": info}
            found[str(run.get("id") or "")] = info
        new_runs.append(run)
    if not found:
        return body
    out = dict(body)
    out["runs"] = new_runs
    if serving is not None:
        rows = []
        for row in serving:
            key = str(row.get("run_id") or "") if isinstance(row, dict) else ""
            if key and key in found:
                row = {**row, "serving_info": found[key]}
            rows.append(row)
        out["serving"] = rows
    return out


def _run_info(run: Any, address_for: Callable[[str], str], cache: Cache) -> dict:
    try:
        if not isinstance(run, dict) or run.get("phase") != "ready" or not run.get("id"):
            return {}
        port = run.get("port")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            return {}
        health = run.get("health") if isinstance(run.get("health"), dict) else {}
        host = health.get("host")
        if not isinstance(host, str) or not host:
            return {}
        address = str(address_for(host) or "")
        if not address:
            return {}
        shown = f"[{address}]" if ":" in address and not address.startswith("[") else address
        return cache.get(f"http://{shown}:{port}")
    except Exception:  # noqa: BLE001 - one odd run never breaks the list
        return {}
