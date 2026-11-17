# SPDX-License-Identifier: Apache-2.0

"""Reservation and the peak memory measured for a recipe.

vLLM reserves ``gpu_memory_utilization`` times the node's memory. Other
engines keep the weight estimate, raised to the peak measured after a
successful run when that peak is higher.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from sparksi.config import env

# Extra GiB held back so a start cannot leave a node with no free unified memory.
MEMORY_MARGIN_GIB = 5.0

_LOCK = threading.RLock()
_PEAKS: dict[str, dict[str, float]] = {}
# True when that number was set by a solo load. Legacy files have no flag.
_SOLO: dict[str, dict[str, bool]] = {}
_LOADED = False
# Live free memory may sit this far under a measured solo peak and still admit it.
SOLO_FREE_TOLERANCE_GIB = 1.0


def reset_for_tests() -> None:
    global _LOADED
    with _LOCK:
        _PEAKS.clear()
        _SOLO.clear()
        _LOADED = True
        if env("SPARKSI_STATE_DIR"):
            try:
                path = _path()
                if path.is_file():
                    path.unlink()
            except OSError:
                pass


def _path() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "recipe-memory.json"
    return Path.home() / ".local" / "state" / "sparksi" / "recipe-memory.json"


def _load() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    path = _path()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if isinstance(value, dict):
            node_peaks: dict[str, float] = {}
            marked = value.get("solo") if isinstance(value.get("solo"), dict) else None
            flags: dict[str, bool] = {}
            for role, gib in value.items():
                if role == "solo":
                    continue
                try:
                    node_peaks[str(role)] = float(gib)
                except (TypeError, ValueError):
                    continue
                # A file written before solo flags existed recorded the host
                # delta of a solo load. Those numbers stay eligible.
                flags[str(role)] = True if marked is None else bool(marked.get(role))
            if node_peaks:
                _PEAKS[str(key)] = node_peaks
                _SOLO[str(key)] = flags
        # A bare number is the old per-process reading. It is not unified memory.


def node_peak_gib(model_id: str, role: str) -> float | None:
    """Peak unified-memory delta recorded for one recipe on one node."""
    with _LOCK:
        _load()
        found = _PEAKS.get(str(model_id or "")) or {}
        value = found.get(str(role or ""))
        return None if value is None else float(value)


def node_peak_is_solo(model_id: str, role: str) -> bool:
    """True when the stored peak was set by a solo load."""
    with _LOCK:
        _load()
        key = str(model_id or "")
        side = str(role or "")
        if side not in (_PEAKS.get(key) or {}):
            return False
        return bool((_SOLO.get(key) or {}).get(side))


def _persist_peaks() -> None:
    blob: dict[str, dict] = {}
    for key, roles in _PEAKS.items():
        item = {role: gib for role, gib in roles.items()}
        flags = _SOLO.get(key) or {}
        if flags:
            item["solo"] = {role: bool(flags.get(role)) for role in roles}
        blob[key] = item
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob), encoding="utf-8")
    os.replace(tmp, path)


def record_node_peak(model_id: str, role: str, gib: float, *, solo: bool = False) -> float:
    """Remember the higher used-memory delta for this recipe on this node.

    ``solo`` is kept only when this sample sets the stored number. A later
    shared sample that does not raise it leaves a solo peak solo.
    """
    with _LOCK:
        _load()
        key = str(model_id or "")
        side = str(role or "head")
        try:
            number = float(gib)
        except (TypeError, ValueError):
            number = 0.0
        current = dict(_PEAKS.get(key) or {})
        old = float(current.get(side) or 0.0)
        stored = max(old, number)
        flags = dict(_SOLO.get(key) or {})
        if number > old + 1e-9 or old <= 0:
            flags[side] = bool(solo)
        if key:
            current[side] = stored
            _PEAKS[key] = current
            _SOLO[key] = flags
        try:
            _persist_peaks()
        except OSError:
            pass
        return stored


def peak_gib(model_id: str) -> float | None:
    """Largest per-node peak for a recipe, for callers that want one number."""
    with _LOCK:
        _load()
        found = _PEAKS.get(str(model_id or "")) or {}
        if not found:
            return None
        return max(found.values())


def consider_peak(previous: float | None, sample: float, *, solo: bool) -> float | None:
    """Peak to store, or None when this sample must be ignored.

    A jump past 1.5x the previous peak is kept only for a solo load. Anything
    else would bill this model for memory another model was using.
    """
    try:
        value = float(sample)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    try:
        prior = float(previous) if previous not in (None, "") else 0.0
    except (TypeError, ValueError):
        prior = 0.0
    if prior <= 0:
        return value
    if value <= prior * 1.5 + 1e-9:
        return max(prior, value)
    if solo:
        return value
    return None


def record_peak(model_id: str, gib: float) -> float:
    """Record ``gib`` as the head node's peak. Prefer ``record_node_peak``."""
    return record_node_peak(model_id, "head", gib)


def spark_count(placement: str) -> int:
    """How many Sparks share the weights: 2 for "both", N for "a+b+...", else 1."""
    text = str(placement or "")
    if text == "both":
        return 2
    if "+" in text:
        return len([part for part in text.split("+") if part.strip()]) or 1
    return 1


def estimate_gib(
    *,
    engine: str,
    gpu_mem_util: float | None,
    uma_gib: float,
    weights_gib: float,
    reserve_gib: float,
    placement: str,
) -> float:
    """Recipe estimate for one node, before the planning margin and any measured peak."""
    if str(engine) == "vllm" and gpu_mem_util is not None:
        return float(gpu_mem_util) * float(uma_gib)
    share = float(weights_gib) / float(spark_count(placement))
    return share + float(reserve_gib)


def role_need_gib(
    *,
    engine: str,
    gpu_mem_util: float | None,
    uma_gib: float,
    weights_gib: float,
    reserve_gib: float,
    measured_gib: float | None,
    placement: str,
) -> float:
    """GiB to hold against one node's budget, including the planning margin."""
    estimate = estimate_gib(
        engine=engine,
        gpu_mem_util=gpu_mem_util,
        uma_gib=uma_gib,
        weights_gib=weights_gib,
        reserve_gib=reserve_gib,
        placement=placement,
    )
    if measured_gib is not None:
        estimate = max(estimate, float(measured_gib))
    return estimate + MEMORY_MARGIN_GIB
