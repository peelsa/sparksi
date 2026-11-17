# SPDX-License-Identifier: Apache-2.0

"""Hot history of announce readings, plus the durable telemetry archive.

The controller writes one sample per machine every second from the
monitor card it already holds. A second sample in that same second
is ignored. A field that was missing is left out of that sample.
The in-memory ring and ``history.json`` keep one hour. The overview
chart draws the last 20 minutes of a chosen window. Older samples
stay in the telemetry files and are not deleted. This module does
not open SSH or run nvidia-smi.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

from sparksi.config import env, state_subdir
from sparksi.lock import state_dir
from sparksi.telemetry import FIELDS as ARCHIVE_FIELDS
from sparksi.telemetry import ROLES, Archive

FIELDS = ("gpu_pct", "gpu_temp_c", "cpu_pct", "mem_pct", "mem_used_gib", "tok_s", "power_w")
RANGES = {"1h": 3600, "6h": 6 * 3600, "24h": 24 * 3600}
KEEP_SECS = 3600
# Monitor piggyback only. The hot ring still keeps the full hour for /api/history.
LIVE_FRESH_SECS = 1200
BUCKET_SECS = 1
MAX_POINTS = 1200
MAX_QUERY_SPAN = 24 * 3600
# Archive rows stay with the interval they were read from. Two windows, and
# only when the point count itself is bounded. Evicting a window drops its rows.
CACHE_WINDOWS = 2
CACHE_POINTS = 4096
# Interactive queries stay inside a practical clock. The files themselves are not trimmed.
STAMP_MAX = 4_102_444_800

_LOCK = threading.Lock()
_STORES: dict[str, "History"] = {}


def history_path() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "history.json"
    durable = state_subdir(Path.home() / ".local" / "state")
    try:
        durable.mkdir(parents=True, exist_ok=True)
        return durable / "history.json"
    except OSError:
        folder = state_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "history.json"


def store() -> "History":
    path = str(history_path())
    with _LOCK:
        found = _STORES.get(path)
        if found is None:
            found = History(Path(path))
            _STORES[path] = found
        return found


def reset_for_tests() -> None:
    with _LOCK:
        _STORES.clear()


def machine_role(machine: str) -> str | None:
    """Map a config id to its role (head, worker, node3, node4). Unknown ids stay unarchived."""
    if machine in ROLES:
        return machine
    try:
        from sparksi.config import CFG
    except Exception:
        return None
    try:
        for role, node in CFG.role_nodes():
            if machine and machine == node.id:
                return role
    except Exception:
        return None
    return None


class History:
    def __init__(self, path: Path, now=None) -> None:
        self.path = Path(path)
        self._now = now or time.time
        self._guard = threading.Lock()
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self._roles: dict[str, str] = {}
        self._archive_cache: dict[str, list[tuple[int, int, dict[int, dict[str, Any]]]]] = {}
        self.archive = Archive(self.path.parent / "telemetry")
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        machines = data.get("machines") if isinstance(data, dict) else None
        if not isinstance(machines, dict):
            return
        for mid, rows in machines.items():
            if isinstance(rows, list):
                self._rows[str(mid)] = [row for row in rows if isinstance(row, dict) and "t" in row]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"machines": self._rows}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)

    def observe(
        self,
        machine: str,
        fields: dict[str, Any],
        now: float | None = None,
        role: str | None = None,
    ) -> bool:
        """Record one second. Return False when that second already has a sample."""
        stamp = self._now() if now is None else now
        minute = int(stamp) // BUCKET_SECS * BUCKET_SECS
        sample: dict[str, Any] = {"t": minute}
        for key in FIELDS:
            if key in fields and fields[key] is not None:
                sample[key] = fields[key]
        chosen = role if role in ROLES else None
        with self._guard:
            key = str(machine)
            if chosen:
                self._roles[key] = chosen
            rows = self._rows.setdefault(key, [])
            if rows and int(rows[-1].get("t") or 0) == minute:
                return False
            rows.append(sample)
            cutoff = int(stamp) - KEEP_SECS
            self._rows[key] = [row for row in rows if int(row.get("t") or 0) >= cutoff]
            if chosen:
                # Durable line first, so a later hot-file error cannot drop it.
                self.archive.append(chosen, sample)
                self._invalidate_archive(key, int(sample["t"]))
            self._save()
            return True

    def role_for(self, machine: str) -> str | None:
        saved = self._roles.get(str(machine))
        if saved in ROLES:
            return saved
        return machine_role(str(machine))

    def latest_samples(self, *, fresh_secs: int | None = None) -> dict[str, dict[str, Any]]:
        """Newest hot sample per machine. One row, not the hour.

        ``fresh_secs`` omits a tip older than that many seconds. The stored
        hour is unchanged; a quiet machine is simply left off this payload.
        """
        now_i = int(self._now())
        with self._guard:
            found: dict[str, dict[str, Any]] = {}
            for machine, rows in self._rows.items():
                if not rows:
                    continue
                sample = _public_sample(rows[-1])
                if fresh_secs is not None and now_i - int(sample["t"]) > fresh_secs:
                    continue
                found[str(machine)] = sample
            return found

    def query(
        self,
        machine: str,
        range_name: str,
        now: float | None = None,
        start: float | None = None,
        end: float | None = None,
        since: float | None = None,
    ) -> dict[str, Any]:
        if start is None and end is None and range_name not in RANGES:
            raise ValueError(range_name)
        if (start is None) ^ (end is None):
            raise ValueError(range_name)
        since_i = None if since is None else _finite_stamp(since)
        stamp = _finite_stamp(self._now() if now is None else now)
        if start is None or end is None:
            end_i = stamp
            start_i = end_i - RANGES[range_name]
        else:
            start_i = _finite_stamp(start)
            end_i = _finite_stamp(end)
            if end_i < start_i or end_i - start_i > MAX_QUERY_SPAN:
                raise ValueError(range_name)
        # `since` drops samples the caller already has. It does not widen the window.
        read_start = start_i
        if since_i is not None and since_i + 1 > read_start:
            read_start = since_i + 1
        role = self.role_for(str(machine))
        span = end_i - read_start
        if read_start > end_i:
            samples = []
            down = False
        elif span <= KEEP_SECS:
            samples = self._raw_window(str(machine), role, read_start, end_i)
            down = False
        else:
            samples = self._bucket_window(str(machine), role, read_start, end_i)
            down = len(samples) > 0 and span > MAX_POINTS
        if since_i is not None:
            samples = [row for row in samples if int(row.get("t") or 0) > since_i]
        payload = {
            "machine": str(machine),
            "range": range_name if range_name in RANGES else "custom",
            "now": stamp,
            "start": start_i,
            "end": end_i,
            "downsampled": down,
            "samples": samples,
        }
        if since_i is not None:
            payload["since"] = since_i
        return payload

    def telemetry_index(
        self,
        role: str | None = None,
        cursor: str | None = None,
        limit: int = 30,
    ) -> dict[str, Any]:
        return self.archive.discover(role=role, cursor=cursor, limit=limit)

    def _hot_rows(self, machine: str, start: int, end: int) -> dict[int, dict[str, Any]]:
        with self._guard:
            rows = list(self._rows.get(machine) or [])
        found: dict[int, dict[str, Any]] = {}
        for row in rows:
            stamp = int(row.get("t") or 0)
            if start <= stamp <= end:
                found[stamp] = dict(row)
        return found

    def _invalidate_archive(self, machine: str, stamp: int) -> None:
        """A new sample inside a cached window has to be read, not assumed absent."""
        rows = self._archive_cache.get(machine)
        if not rows:
            return
        self._archive_cache[machine] = [item for item in rows if stamp < item[0] or stamp > item[1]]

    def _cached_archive(self, machine: str, start: int, end: int) -> dict[int, dict[str, Any]] | None:
        """Rows for an interval this process actually stored. A miss is None."""
        for left, right, rows in self._archive_cache.get(machine, []):
            if left <= start and end <= right:
                return {stamp: dict(row) for stamp, row in rows.items() if start <= stamp <= end}
        return None

    def _store_archive(self, machine: str, start: int, end: int, rows: dict[int, dict[str, Any]]) -> None:
        if len(rows) > CACHE_POINTS:
            return
        kept = [
            item
            for item in self._archive_cache.get(machine, [])
            if item[1] < start or item[0] > end
        ]
        kept.append((start, end, rows))
        self._archive_cache[machine] = kept[-CACHE_WINDOWS:]

    def cache_size(self) -> tuple[int, int]:
        """(windows, samples) held for archive repeats."""
        windows = 0
        samples = 0
        for rows in self._archive_cache.values():
            windows += len(rows)
            samples += sum(len(item[2]) for item in rows)
        return windows, samples

    def _merged(self, machine: str, role: str | None, start: int, end: int) -> dict[int, dict[str, Any]]:
        """One sample per timestamp. The hot sample wins when both stores have it.

        An archive read is skipped only when this process still holds the rows
        for that whole interval. Dropping a cached window drops the right to
        skip the read.
        """
        hot = self._hot_rows(machine, start, end)
        if role is None:
            return hot
        cached = self._cached_archive(machine, start, end)
        if cached is not None:
            cached.update(hot)
            return cached
        archived: dict[int, dict[str, Any]] = {}

        def visit(row: dict[str, Any]) -> None:
            archived[int(row["t"])] = _public_sample(row)

        if self.archive.reduce(role, start, end, visit):
            self._store_archive(machine, start, end, archived)
        archived.update(hot)
        return archived

    def _raw_window(self, machine: str, role: str | None, start: int, end: int) -> list[dict[str, Any]]:
        by_t = self._merged(machine, role, start, end)
        return [by_t[stamp] for stamp in sorted(by_t)]

    def _bucket_window(self, machine: str, role: str | None, start: int, end: int) -> list[dict[str, Any]]:
        by_t = self._merged(machine, role, start, end)
        span = max(1, end - start)
        bucket = max(1, math.ceil(span / MAX_POINTS))
        slots: dict[int, dict[str, Any]] = {}
        for stamp in sorted(by_t):
            row = by_t[stamp]
            key = stamp // bucket * bucket
            slot = slots.get(key)
            if slot is None:
                slot = {"last": stamp, "sum": {}, "n": {}}
                slots[key] = slot
            if stamp >= slot["last"]:
                slot["last"] = stamp
            for field in ARCHIVE_FIELDS:
                if row.get(field) is None:
                    continue
                slot["sum"][field] = slot["sum"].get(field, 0.0) + float(row[field])
                slot["n"][field] = slot["n"].get(field, 0) + 1
        samples: list[dict[str, Any]] = []
        for key in sorted(slots):
            slot = slots[key]
            sample: dict[str, Any] = {"t": int(slot["last"])}
            for field, total in slot["sum"].items():
                count = slot["n"][field]
                if not count:
                    continue
                value = total / count
                sample[field] = int(value) if float(value).is_integer() else value
            samples.append(sample)
        return samples


def _finite_stamp(value: float) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("stamp")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("stamp")
    try:
        stamp = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("stamp") from exc
    if stamp < 0 or stamp > STAMP_MAX:
        raise ValueError("stamp")
    return stamp


def _public_sample(row: dict[str, Any]) -> dict[str, Any]:
    sample = {"t": int(row["t"])}
    for key in FIELDS:
        if row.get(key) is not None:
            sample[key] = row[key]
    return sample


def sample_card(card: dict[str, Any], history: History, activity: Any, now: float | None = None) -> None:
    """Fold one monitor card into history and announce/temperature events."""
    loaded = card.get("loaded") or []
    for host in card.get("hosts") or []:
        mid = str(host.get("id") or "")
        if not mid:
            continue
        fresh = bool(host.get("fresh")) and not host.get("unreachable")
        if activity is not None:
            activity.note_announce(mid, fresh)
        if not fresh:
            continue
        fields: dict[str, Any] = {}
        for key in ("gpu_pct", "gpu_temp_c", "cpu_pct", "mem_pct", "mem_used_gib", "power_w"):
            if host.get(key) is not None:
                fields[key] = host.get(key)
        tok = 0.0
        saw_tok = False
        for rec in loaded:
            owners = [str(rec.get("host") or "")]
            owners.extend(str(item) for item in (rec.get("span") or []))
            if mid not in owners or rec.get("tok_s") is None:
                continue
            tok += float(rec["tok_s"])
            saw_tok = True
        if saw_tok:
            fields["tok_s"] = tok
        history.observe(mid, fields, now=now, role=machine_role(mid))
        if activity is not None and host.get("gpu_temp_c") is not None:
            activity.note_temp(mid, float(host["gpu_temp_c"]))
