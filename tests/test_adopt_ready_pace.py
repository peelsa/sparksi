# SPDX-License-Identifier: Apache-2.0

"""Adopted serving runs are not re-checked on every status poll (ready pace)."""

import pytest

from sparksi.recipes import adopt, runs
from sparksi.config import CFG


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 100.0}
    monkeypatch.setattr(runs.time, "monotonic", lambda: now["t"])
    return now


@pytest.fixture
def run_record():
    return {"id": "x1", "phase": "ready", "started_by": "external"}


@pytest.fixture
def harness(monkeypatch, clock, run_record):
    monkeypatch.setitem(CFG.controller, "ready_check_s", 15)
    runs.forget_ready_checks()
    yield
    runs.forget_ready_checks()


def test_ready_run_checked_on_interval(monkeypatch, harness, clock, run_record):
    gone_calls = []
    tick_calls = []

    monkeypatch.setattr(adopt, "list_runs", lambda: [dict(run_record)])

    def containers_gone(r, runner):
        gone_calls.append(r["id"])
        return False

    def tick(r, runner):
        tick_calls.append(r["id"])
        runs.ready_check_due(r)
        return True

    monkeypatch.setattr(adopt, "containers_gone", containers_gone)
    monkeypatch.setattr(adopt, "tick", tick)

    adopt._live_external(None, watch=True)
    assert len(gone_calls) == 1
    assert len(tick_calls) == 1

    clock["t"] = 105.0
    adopt._live_external(None, watch=True)
    assert len(gone_calls) == 1
    assert len(tick_calls) == 1

    clock["t"] = 115.5
    adopt._live_external(None, watch=True)
    assert len(gone_calls) == 2
    assert len(tick_calls) == 2


def test_waiting_health_checked_every_call(monkeypatch, harness, clock, run_record):
    gone_calls = []
    tick_calls = []

    waiting = dict(run_record, phase="waiting_health")
    monkeypatch.setattr(adopt, "list_runs", lambda: [waiting])

    monkeypatch.setattr(adopt, "containers_gone", lambda r, runner: gone_calls.append(r["id"]) or False)
    monkeypatch.setattr(adopt, "tick", lambda r, runner: tick_calls.append(r["id"]) or True)

    for _ in range(3):
        adopt._live_external(None, watch=True)

    assert len(gone_calls) == 3
    assert len(tick_calls) == 3
