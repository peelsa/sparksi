# SPDX-License-Identifier: Apache-2.0

"""Bounded activity log for the dashboard.

Events are the actions the controller already performs, plus announce loss
and a GPU temperature crossing. A reading that sits on a threshold does not
append again. The file lives in the state dir and is not served with secrets.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from sparksi.config import env, state_subdir
from sparksi.lock import state_dir

CAP = 500
HOT_C = 80.0
COOL_C = 75.0

_LOCK = threading.Lock()
_STORES: dict[str, "Activity"] = {}


def activity_path() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "activity.json"
    durable = state_subdir(Path.home() / ".local" / "state")
    try:
        durable.mkdir(parents=True, exist_ok=True)
        return durable / "activity.json"
    except OSError:
        folder = state_dir()
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "activity.json"


def store() -> "Activity":
    path = str(activity_path())
    with _LOCK:
        found = _STORES.get(path)
        if found is None:
            found = Activity(Path(path))
            _STORES[path] = found
        return found


def reset_for_tests() -> None:
    with _LOCK:
        _STORES.clear()


class Activity:
    def __init__(self, path: Path, now=None) -> None:
        self.path = Path(path)
        self._now = now or time.time
        self._guard = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._hot: dict[str, str] = {}
        self._fresh: dict[str, bool] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            return
        if not isinstance(data, dict):
            return
        events = data.get("events")
        if isinstance(events, list):
            self._events = [row for row in events if isinstance(row, dict)][-CAP:]
        hot = data.get("hot")
        if isinstance(hot, dict):
            self._hot = {str(k): str(v) for k, v in hot.items()}
        fresh = data.get("fresh")
        if isinstance(fresh, dict):
            self._fresh = {str(k): bool(v) for k, v in fresh.items()}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"events": self._events[-CAP:], "hot": self._hot, "fresh": self._fresh}
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, self.path)

    def append(self, kind: str, message: str, machine: str = "") -> None:
        row = {"t": int(self._now()), "kind": str(kind), "message": str(message)}
        if machine:
            row["machine"] = str(machine)
        with self._guard:
            self._events.append(row)
            del self._events[:-CAP]
            self._save()

    def events(self) -> list[dict[str, Any]]:
        with self._guard:
            return list(reversed(self._events))

    def note_temp(self, machine: str, temp_c: float | None) -> None:
        if temp_c is None:
            return
        with self._guard:
            state = self._hot.get(machine)
            if temp_c >= HOT_C and state != "hot":
                self._hot[machine] = "hot"
                kind, message = "gpu-hot", f"{machine} GPU at {temp_c:.0f} C"
            elif state == "hot" and temp_c < COOL_C:
                self._hot[machine] = "cool"
                kind, message = "gpu-cool", f"{machine} GPU under 75 C"
            else:
                return
            self._events.append({"t": int(self._now()), "kind": kind, "message": message, "machine": machine})
            del self._events[:-CAP]
            self._save()

    def note_announce(self, machine: str, fresh: bool) -> None:
        with self._guard:
            # The first packet only records that we have seen the machine.
            # A stale start must not become "announce restored" once it answers.
            if machine not in self._fresh:
                if not fresh:
                    return
                self._fresh[machine] = True
                self._save()
                return
            prev = self._fresh[machine]
            if prev is fresh:
                return
            self._fresh[machine] = bool(fresh)
            kind = "announce-lost" if prev and not fresh else "announce-restored"
            word = "announce lost" if kind == "announce-lost" else "announce restored"
            self._events.append({
                "t": int(self._now()),
                "kind": kind,
                "message": f"{machine} {word}",
                "machine": machine,
            })
            del self._events[:-CAP]
            self._save()
