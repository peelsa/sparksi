# SPDX-License-Identifier: Apache-2.0

"""Serve the last known answer now; refresh it from the machines in the background.

A read that has to reach a Spark (SSH, a probe, ``docker stats``) can take seconds,
or wait out a timeout when a machine is down. Page polls must not wait for that.
``Refreshing.get`` returns the cached value at once and, when it is older than its
ttl, starts one background refresh. Only the very first read of a key waits, and
only up to ``first_wait`` seconds; after that the caller gets ``PENDING`` and
answers with what it has. With ``max_age``, a value whose last successful refresh
is older than that is not served (the read waits like a first read). ``expire``
marks values old after a write; a refresh that was already running when the write
came keeps them marked old. A read after a write waits for a value refreshed after
it, so a page polled right after an action does not show the state from before it.
That wait only runs inside the first ``first_wait`` of the refresh, then the old
value is served at once: a slow refresh costs one short wait per write, not one
per read.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

PENDING = object()


class _Entry:
    __slots__ = ("value", "at", "ok_at", "running", "done", "gen", "dirty", "started")

    def __init__(self) -> None:
        self.value: Any = PENDING
        self.at = 0.0
        self.ok_at = 0.0
        self.running = False
        self.done = threading.Event()
        self.gen = 0
        self.dirty = False  # a write came after the value was read
        self.started = 0.0  # when the refresh now running began


class Refreshing:
    def __init__(
        self,
        ttl: float,
        first_wait: float = 0.8,
        clock: Callable[[], float] = time.monotonic,
        max_age: float | None = None,
    ) -> None:
        self.ttl = ttl
        self.first_wait = first_wait
        self._clock = clock
        self.max_age = max_age
        self._lock = threading.Lock()
        self._entries: dict[Any, _Entry] = {}

    def _start(self, key: Any, entry: _Entry, fn: Callable[[], Any]) -> None:
        entry.running = True
        entry.started = self._clock()
        entry.done.clear()
        gen = entry.gen

        def run() -> None:
            try:
                value = fn()
            except Exception:  # noqa: BLE001 — keep the last value; the next read retries
                value = PENDING
            with self._lock:
                if value is not PENDING:
                    entry.value = value
                if value is not PENDING:
                    entry.ok_at = self._clock()
                # A write after this refresh started leaves the value marked old,
                # so the next read refreshes again.
                entry.at = self._clock() if gen == entry.gen else 0.0
                if gen == entry.gen:
                    entry.dirty = False
                entry.running = False
                entry.done.set()

        threading.Thread(target=run, name=f"sparksi-refresh-{key}", daemon=True).start()

    def get(self, key: Any, fn: Callable[[], Any], wait: float | None = None) -> Any:
        """The cached value for ``key`` (refreshing it when old), or ``PENDING``.

        ``wait`` overrides ``first_wait`` (0 starts the refresh without waiting).
        ``max_age``: a value older than that is treated as absent, so the read
        waits for a fresh refresh and returns ``PENDING`` if none arrives.
        """
        with self._lock:
            entry = self._entries.setdefault(key, _Entry())
            now = self._clock()
            # A failed refresh also stamps ``at``, so a machine that is down is
            # retried once per ttl, not on every read.
            stale = now - entry.at >= self.ttl
            too_old = self._too_old(entry, now)
            if (stale or too_old or entry.dirty) and not entry.running:
                self._start(key, entry, fn)
            value, done = entry.value, entry.done
            if too_old:
                value = PENDING
            after_write = entry.dirty and value is not PENDING
            # Wait only inside the first ``first_wait`` of the refresh now running, so
            # a slow refresh costs one short wait per write, not one per read.
            budget = self.first_wait if wait is None else wait
            left = budget - (now - entry.started) if entry.running else 0.0
        if after_write:
            # The old value is from before a write: give the refresh a moment to answer.
            if left > 0 and done.wait(left):
                with self._lock:
                    if entry.value is not PENDING:
                        value = entry.value
            return value
        if value is PENDING:
            done.wait(self.first_wait if wait is None else wait)
            with self._lock:
                value = entry.value
                if value is not PENDING and self._too_old(entry, self._clock()):
                    value = PENDING
        return value

    def _too_old(self, entry: _Entry, now: float) -> bool:
        """A value from a success longer than ``max_age`` ago is not served."""
        return self.max_age is not None and entry.value is not PENDING and now - entry.ok_at > self.max_age

    def age(self, key: Any) -> float | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None or entry.value is PENDING:
                return None
            return max(0.0, self._clock() - entry.at)

    def expire(self) -> None:
        """Mark every value old (after a write), keeping it to serve until refreshed."""
        with self._lock:
            for entry in self._entries.values():
                entry.gen += 1
                entry.at = 0.0
                entry.dirty = True

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
