# SPDX-License-Identifier: Apache-2.0

"""Fake-peer checks for the shared dashboard snapshot."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sparksi.monitor.__main__ import frame_from_snapshot, snapshot
from sparksi.monitor.peer import STALE_SECS


class Fake:
    def __init__(self, data, age):
        self._data = data
        self._age = age

    def snapshot(self):
        return self._data, self._age


def _hw():
    return {
        "mem": (119.0, 40.0),
        "cpu_pct": 12.0,
        "cpu_mhz": 1800.0,
        "gpu": [3.0, 50.0, 900.0],
        "zones": [("soc0", 45.0)],
        "http": {},
        "containers": [],
        "engines": [],
        "comfy": {},
        "truth": "",
    }


def test_both_fresh():
    snap = snapshot(Fake(_hw(), 1.0), Fake(_hw(), 1.2), {})
    assert [h["id"] for h in snap["hosts"]] == ["a", "b"]
    assert all(h["fresh"] for h in snap["hosts"])
    assert snap["hosts"][0]["sys_nominal_gib"] == 128.0
    assert snap["hosts"][0]["mem_total_gib"] == 119.0
    text = "\n".join(frame_from_snapshot(snap))
    assert "a" in text and "b" in text
    assert "peer unreachable" not in text


def test_worker_stale():
    snap = snapshot(Fake(_hw(), 1.0), Fake(_hw(), STALE_SECS + 5), {})
    worker = snap["hosts"][1]
    assert worker["fresh"] is False
    assert worker["unreachable"] is True
    text = "\n".join(frame_from_snapshot(snap))
    assert "peer unreachable" in text


def test_tensorfold_same_name_pair_keeps_container():
    head = _hw()
    head["http"] = {
        8015: {
            "served": "Pair-EXL3",
            "metrics": {"num_requests_running": 0},
        }
    }
    head["engines"] = [{"name": "pair-tf", "port": 8015, "tp": 2}]
    worker = _hw()
    worker["containers"] = [{"name": "pair-tf", "status": "Up"}]
    snap = snapshot(Fake(head, 0.4), Fake(worker, 0.4), {})
    assert len(snap["loaded"]) == 1
    row = snap["loaded"][0]
    assert row["port"] == 8015
    assert row["served"] == "Pair-EXL3"
    assert row["container"] == "pair-tf"
    assert row["span"] == ["a", "b"]
    assert row["worker_up"] is True
    assert row["worker_name"] == "pair-tf"


def test_pair_worker_down_is_split():
    head = _hw()
    head["http"] = {
        8014: {
            "served": "glm",
            "metrics": {"num_requests_running": 0, "generation_tokens_total": 0},
        }
    }
    head["engines"] = [{"name": "pair-exl3-head", "port": 8014, "tp": 2}]
    worker = _hw()
    worker["containers"] = [{"name": "pair-exl3-worker", "status": "Exited"}]
    snap = snapshot(Fake(head, 0.5), Fake(worker, 0.5), {})
    assert len(snap["loaded"]) == 1
    row = snap["loaded"][0]
    assert row["span"] == ["a", "b"]
    assert row["worker_up"] is False
    assert row["state"] == "SPLIT"
    text = "\n".join(frame_from_snapshot(snap))
    assert "SPLIT" in text
    assert "a+b" in text
