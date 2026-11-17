# SPDX-License-Identifier: Apache-2.0

"""The first read after a write waits briefly for a value read after the write."""

from __future__ import annotations

import threading
import time

from sparksi.refresh import PENDING, Refreshing


class _Source:
    def __init__(self) -> None:
        self.value = "before"
        self.delay = 0.0
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return self.value


def _primed(first_wait: float = 0.5) -> tuple[Refreshing, _Source]:
    cache = Refreshing(ttl=60.0, first_wait=first_wait)
    src = _Source()
    assert cache.get("runs", src) == "before"
    return cache, src


def test_read_after_a_write_gets_the_new_value():
    cache, src = _primed()
    src.value = "after"
    cache.expire()
    assert cache.get("runs", src) == "after"


def test_slow_refresh_after_a_write_falls_back_to_the_old_value():
    cache, src = _primed(first_wait=0.1)
    src.value = "after"
    src.delay = 0.6
    cache.expire()
    t0 = time.monotonic()
    got = cache.get("runs", src)
    took = time.monotonic() - t0
    assert got == "before"
    assert got is not PENDING
    assert took < 0.45
    deadline = time.monotonic() + 3
    while cache.get("runs", src, wait=0) != "after" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert cache.get("runs", src, wait=0) == "after"


def test_read_without_a_write_does_not_wait():
    cache, src = _primed()
    src.delay = 0.6
    t0 = time.monotonic()
    assert cache.get("runs", src) == "before"
    assert time.monotonic() - t0 < 0.1
    assert src.calls == 1


def test_a_write_during_a_refresh_keeps_the_entry_marked():
    cache, src = _primed()
    gate = threading.Event()
    src.value = "mid"

    def slow() -> str:
        gate.wait(2)
        return src()

    cache.expire()
    first = cache.get("runs", slow, wait=0.05)  # refresh started, not done yet
    assert first == "before"
    src.value = "after"
    cache.expire()  # a second write while that refresh runs
    gate.set()
    deadline = time.monotonic() + 3
    got = None
    while time.monotonic() < deadline:
        got = cache.get("runs", src)
        if got == "after":
            break
        time.sleep(0.05)
    assert got == "after"


def _wait_for(cache: Refreshing, key: str, src: _Source, want: str, limit: float = 4.0) -> str:
    deadline = time.monotonic() + limit
    got = cache.get(key, src, wait=0)
    while got != want and time.monotonic() < deadline:
        time.sleep(0.05)
        got = cache.get(key, src, wait=0)
    return got


def test_a_slow_refresh_after_a_write_costs_one_wait_not_one_per_read():
    cache, src = _primed(first_wait=0.2)
    src.delay = 1.5  # a node that is slow to answer
    src.value = "after"
    cache.expire()
    t0 = time.monotonic()
    assert cache.get("runs", src) == "before"
    first = time.monotonic() - t0
    assert 0.15 <= first < 0.5
    for _ in range(5):
        t = time.monotonic()
        assert cache.get("runs", src) == "before"
        assert time.monotonic() - t < 0.05
    assert src.calls == 2  # one refresh for the write, not one per read
    assert _wait_for(cache, "runs", src, "after") == "after"


def test_several_values_each_wait_at_most_once_per_write():
    cache = Refreshing(ttl=60.0, first_wait=0.2)
    sources = {key: _Source() for key in ("runs", "nodes", "fleet")}
    for key, src in sources.items():
        assert cache.get(key, src) == "before"
        src.delay = 1.5
        src.value = "after"
    cache.expire()
    t0 = time.monotonic()
    for key, src in sources.items():
        assert cache.get(key, src) == "before"
    first_poll = time.monotonic() - t0
    assert first_poll < 0.9  # three short waits at most
    t1 = time.monotonic()
    for _ in range(3):
        for key, src in sources.items():
            assert cache.get(key, src) == "before"
    assert time.monotonic() - t1 < 0.15  # later polls do not wait at all
