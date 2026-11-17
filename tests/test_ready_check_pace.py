# SPDX-License-Identifier: Apache-2.0

"""A serving run is health-checked at the ready pace; loading runs keep the 3 s pace."""

import pytest

from sparksi.recipes import runs
from sparksi.config import CFG


class _Clock:
    """Controllable monotonic clock backed by a list holder."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = [start]

    def __call__(self) -> float:
        return self.now[0]

    def advance(self, delta: float) -> None:
        self.now[0] += delta


@pytest.fixture(autouse=True)
def _ready_env(monkeypatch):
    runs.forget_ready_checks()
    monkeypatch.setitem(CFG.controller, "ready_check_s", 15)
    yield
    runs.forget_ready_checks()


@pytest.fixture
def clock(monkeypatch):
    c = _Clock(100.0)
    monkeypatch.setattr(runs.time, "monotonic", c)
    return c


@pytest.fixture
def tick_calls(monkeypatch):
    calls = {"n": 0}

    def fake_tick(run, runner, *, now=None):
        calls["n"] += 1
        return run

    monkeypatch.setattr(runs, "_tick", fake_tick)
    return calls


def test_ready_run_paced(clock, tick_calls):
    run = {"id": "r1", "phase": "ready"}
    runs.tick(run, None)  # t=100 -> runs
    assert tick_calls["n"] == 1

    clock.now[0] = 110.0
    runs.tick(run, None)  # t=110, only 10s elapsed -> skipped
    assert tick_calls["n"] == 1

    clock.now[0] = 115.1
    runs.tick(run, None)  # t=115.1, >=15s -> runs
    assert tick_calls["n"] == 2


def test_loading_phases_never_skipped(clock, tick_calls):
    for phase in ("starting", "waiting_health", "copying"):
        run = {"id": "p1", "phase": phase}
        for _ in range(3):
            runs.tick(run, None)
    assert tick_calls["n"] == 9


def test_ready_runs_paced_separately(clock, tick_calls):
    r1 = {"id": "r1", "phase": "ready"}
    r2 = {"id": "r2", "phase": "ready"}
    runs.tick(r1, None)
    runs.tick(r2, None)
    assert tick_calls["n"] == 2

    clock.now[0] = 110.0
    runs.tick(r1, None)  # skipped (10s < 15s)
    runs.tick(r2, None)  # skipped
    assert tick_calls["n"] == 2

    clock.now[0] = 115.1
    runs.tick(r1, None)
    runs.tick(r2, None)
    assert tick_calls["n"] == 4


def test_ready_check_s_values(monkeypatch):
    monkeypatch.setitem(CFG.controller, "ready_check_s", 0)
    assert runs.ready_check_s() == 0.0

    monkeypatch.setitem(CFG.controller, "ready_check_s", "abc")
    assert runs.ready_check_s() == 15.0

    monkeypatch.setitem(CFG.controller, "ready_check_s", 999)
    assert runs.ready_check_s() == 120.0


def test_ready_check_due_mark_false_does_not_consume(clock, tick_calls):
    run = {"id": "r1", "phase": "ready"}
    assert runs.ready_check_due(run, mark=False) is True
    assert runs.ready_check_due(run, mark=False) is True  # still due, slot not consumed
    runs.tick(run, None)
    assert tick_calls["n"] == 1


def test_zero_checks_every_tick(clock, tick_calls, monkeypatch):
    monkeypatch.setitem(CFG.controller, "ready_check_s", 0)
    run = {"id": "r1", "phase": "ready"}
    for _ in range(3):
        runs.tick(run, None)
    assert tick_calls["n"] == 3
