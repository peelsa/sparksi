# SPDX-License-Identifier: Apache-2.0

"""Serving display: a live split pair is not a stopped run, and not an empty page."""

import json
import subprocess

import pytest

from sparksi.config import CFG
from sparksi.monitor.__main__ import _loaded_row
from sparksi.monitor.models import discover_loaded
from sparksi.serve import INDEX_HTML
from sparksi.serving_state import _reject, attach, reconcile

STOPPED_ID = "8618b9772c00"


def _split_pair() -> dict:
    """The monitor row a headless engine pair actually produces."""
    raw = discover_loaded(
        {"fresh": True, "http": {}, "engines": [{"name": "example-engine-head", "port": 8014}]},
        {
            "fresh": True,
            "http": {},
            "containers": [{"name": "example-engine-worker", "status": "Up 2 hours"}],
        },
    )
    assert len(raw) == 1
    row = _loaded_row(raw[0], 0, {})
    assert row["served"] is None
    assert row["head_up"] is False
    assert row["worker_up"] is True
    assert row["state"] == "SPLIT"
    assert row["container_source"] == "engine"
    assert row["bind"] == ""
    assert row["route"] == ""
    assert row["listen"] == ""
    assert row["span"] == [CFG.head.id, CFG.worker.id]
    return row


def _stopped() -> dict:
    return {
        "id": STOPPED_ID,
        "model_id": "retired-model",
        "title": "Retired model",
        "phase": "stopped",
        "port": 8014,
        "placement": "both",
        "started": 1,
        "nodes": [
            {"role": "head", "phase": "stopped", "container": "example-engine-head"},
            {"role": "worker", "phase": "stopped", "container": "example-engine-worker"},
        ],
    }


def _hosts() -> list[dict]:
    return [{"id": CFG.head.id, "fresh": True}, {"id": CFG.worker.id, "fresh": True}]


def test_split_pair_is_not_labeled_by_the_stopped_run() -> None:
    rows = reconcile([_split_pair()], [_stopped()], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    by_role = {row["role"]: row for row in rows}
    assert set(by_role) == {"head", "worker"}
    for row in rows:
        assert row["state"] == "split"
        assert row["run_id"] == ""
        assert row["port"] == 8014
        assert row["title"] == "example-engine"
        assert "retired" not in row["title"].lower()
        assert STOPPED_ID not in row["title"]
        assert "Ready" not in row["text"]
        assert "earlier instance" in row["reason"]
    assert "not confirmed" in by_role["head"]["text"]
    assert "worker up" in by_role["worker"]["text"]
    assert "Nothing running" not in by_role["head"]["text"]
    assert "Nothing running" not in by_role["worker"]["text"]


def test_stopped_run_alone_stays_idle() -> None:
    rows = reconcile([], [_stopped()], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    assert [row["state"] for row in rows] == ["idle", "idle"]
    assert all(row["run_id"] == "" for row in rows)


def test_stopped_run_cannot_be_attached() -> None:
    with pytest.raises(ValueError, match="active run"):
        _reject({"state": "ready", "run_id": STOPPED_ID}, [_stopped()])
    with pytest.raises(ValueError, match="cannot carry a run id"):
        _reject({"state": "split", "run_id": STOPPED_ID}, [_stopped()])
    with pytest.raises(ValueError, match="cannot carry a run id"):
        _reject({"state": "answering", "run_id": "old-fail"}, [_stopped()])


def test_ready_run_stays_ready_when_the_role_is_confirmed() -> None:
    live = {
        "name": "demo",
        "served": "demo",
        "host": CFG.head.id,
        "port": 8014,
        "span": [CFG.head.id, CFG.worker.id],
        "head_up": True,
        "worker_up": True,
        "worker_host": CFG.worker.id,
        "worker_name": "demo-worker",
        "container": "demo-head",
        "container_source": "engine",
        "state": "UP",
    }
    run = {
        "id": "live-1",
        "model_id": "demo",
        "phase": "ready",
        "port": 8014,
        "started": 5,
        "nodes": [
            {"role": "head", "phase": "started", "container": "demo-head"},
            {"role": "worker", "phase": "started", "container": "demo-worker"},
        ],
    }
    rows = reconcile([live], [run, _stopped()], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    assert [row["state"] for row in rows] == ["ready", "ready"]
    assert {row["run_id"] for row in rows} == {"live-1"}
    assert rows[0]["text"] == "Ready · port 8014"


def test_stale_report_is_not_treated_as_up() -> None:
    hosts = [{"id": CFG.head.id, "fresh": False}, {"id": CFG.worker.id, "fresh": True}]
    rows = reconcile([_split_pair()], [], hosts=hosts, head_id=CFG.head.id, worker_id=CFG.worker.id)
    assert rows[0]["state"] == "stale"
    assert rows[0]["run_id"] == ""
    assert rows[1]["state"] == "stale"
    missing = reconcile([_split_pair()], [], hosts=None, head_id=CFG.head.id, worker_id=CFG.worker.id)
    unnamed = reconcile([_split_pair()], [], hosts=[{"id": CFG.head.id}], head_id=CFG.head.id, worker_id=CFG.worker.id)
    assert [row["state"] for row in missing] == ["stale", "stale"]
    assert [row["state"] for row in unnamed] == ["stale", "stale"]
    assert all(row["run_id"] == "" for row in missing + unnamed)


def _row_from_producer(head: dict, worker: dict) -> dict:
    raw = discover_loaded(head, worker)
    assert len(raw) == 1, raw
    return _loaded_row(raw[0], 0, {})


def _healthy_pair() -> dict:
    row = _row_from_producer(
        {
            "fresh": True,
            "http": {8014: {"served": "example-served", "metrics": {"num_requests_running": 0}}},
            "engines": [{"name": "example-engine-head", "port": 8014}],
            "truth": "",
        },
        {
            "fresh": True,
            "http": {},
            "containers": [{"name": "example-engine-worker", "status": "Up 2 hours"}],
        },
    )
    assert row["head_up"] is True
    assert row["worker_up"] is True
    assert row["served"] == "example-served"
    assert row["container_source"] == "engine"
    assert row["port"] == 8014
    return row


def _closed(phase: str, run_id: str, worker_phase: str) -> dict:
    return {
        "id": run_id,
        "model_id": "example-served",
        "title": "Old label",
        "phase": phase,
        "port": 8014,
        "started": 1,
        "nodes": [
            {"role": "head", "phase": phase, "container": "example-engine-head"},
            {"role": "worker", "phase": worker_phase, "container": "example-engine-worker"},
        ],
    }


def test_closed_runs_do_not_label_a_healthy_pair() -> None:
    live = _healthy_pair()
    failed = _closed("failed", "old-fail", "stopped")
    partial = _closed("stopped", "partial-stop", "started")
    timed = _closed("timed_out", "old-timeout", "started")
    for run in (failed, partial, timed):
        rows = reconcile([live], [run], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
        assert [item["state"] for item in rows] == ["answering", "answering"]
        assert all(item["run_id"] == "" for item in rows)
        assert all(item["title"] == "example-served" for item in rows)
        assert all("Ready" not in item["text"] for item in rows)
        assert run["id"] not in json.dumps(rows)


def _truth_span() -> dict:
    truth = "models:\n- port: 8014\n  served_name: example-served\n  container: example-engine-worker\n  display: Example TP2\n  role: svc\n"
    row = _row_from_producer(
        {"fresh": True, "http": {}, "engines": [], "truth": truth},
        {"fresh": True, "http": {}, "containers": [{"name": "example-engine-worker", "status": "Up 1 hour"}]},
    )
    assert row["head_up"] is False
    assert row["worker_up"] is True
    assert row["served"] == "example-served"
    assert row["container_source"] == "truth"
    assert row["span"] == [CFG.head.id, CFG.worker.id]
    return row


def test_truth_hint_does_not_make_a_split_pair_ready() -> None:
    ready = {
        "id": "r1",
        "model_id": "example-served",
        "title": "Ready label",
        "phase": "ready",
        "port": 8014,
        "started": 9,
        "nodes": [
            {"role": "head", "phase": "ready", "container": "example-engine-worker"},
            {"role": "worker", "phase": "started", "container": "example-engine-worker"},
        ],
    }
    rows = reconcile([_truth_span()], [ready], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    assert [item["state"] for item in rows] == ["split", "split"]
    assert all(item["run_id"] == "" for item in rows)
    assert all("Ready" not in item["text"] for item in rows)
    assert all(item["title"] != "Ready label" for item in rows)
    assert "r1" not in json.dumps(rows)


def test_runs_payload_adds_serving_from_the_monitor(monkeypatch) -> None:
    row = _split_pair()
    monkeypatch.setattr(
        "sparksi.serve.current_dash",
        lambda: {"ok": True, "loaded": [row], "hosts": _hosts()},
    )
    from sparksi.serve import _with_serving

    body = _with_serving({"runs": [_stopped()]})
    assert body["runs"][0]["id"] == STOPPED_ID
    assert body["runs"][0]["phase"] == "stopped"
    assert [item["state"] for item in body["serving"]] == ["split", "split"]
    assert all(item["run_id"] == "" for item in body["serving"])
    assert "retired" not in json.dumps(body["serving"]).lower()


def test_attach_keeps_an_existing_serving_list() -> None:
    given = [{"role": "head", "state": "split", "run_id": ""}]
    body = attach({"runs": [_stopped()], "serving": given}, [], [], head_id="a", worker_id="b")
    assert body["serving"] is given
    assert body["runs"][0]["id"] == STOPPED_ID


def _worker_only() -> dict:
    row = _row_from_producer(
        {"fresh": True, "http": {}, "engines": [], "truth": ""},
        {
            "fresh": True,
            "http": {8020: {"served": "example-worker-served", "metrics": {"num_requests_running": 0}}},
            "containers": [],
        },
    )
    assert row["host"] == CFG.worker.id
    assert row["served"] == "example-worker-served"
    assert row["head_up"] is True
    assert row["worker_up"] is None
    assert not row.get("span")
    assert row["port"] == 8020
    assert row.get("state") != "DOWN"
    return row


def test_worker_only_http_row_answers_on_the_worker() -> None:
    live = _worker_only()
    alone = reconcile([live], [], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    by_role = {row["role"]: row for row in alone}
    assert by_role["head"]["state"] == "idle"
    assert by_role["worker"]["state"] == "answering"
    assert by_role["worker"]["title"] == "example-worker-served"
    assert by_role["worker"]["run_id"] == ""
    assert "identity not reported" not in by_role["worker"]["text"]
    ready = {
        "id": "worker-live",
        "model_id": "example-worker-served",
        "phase": "ready",
        "port": 8020,
        "placement": "worker",
        "started": 4,
        "nodes": [{"role": "worker", "phase": "started"}],
    }
    owned = reconcile([live], [ready], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    owned_by = {row["role"]: row for row in owned}
    assert owned_by["head"]["state"] == "idle"
    assert owned_by["head"]["run_id"] == ""
    assert owned_by["worker"]["state"] == "ready"
    assert owned_by["worker"]["run_id"] == "worker-live"
    assert owned_by["worker"]["text"] == "Ready · port 8020"


def test_page_renders_split_rows_instead_of_an_empty_state() -> None:
    rows = reconcile([_split_pair()], [_stopped()], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* serving-state-start */", 1)[1].split("/* serving-state-end */", 1)[0]
    proc = subprocess.run(
        ["node"],
        input=script
        + "servingRows = "
        + json.dumps(rows)
        + ";\nprocess.stdout.write(JSON.stringify({head: observedServing('head'), worker: observedServing('worker')}));",
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    painted = json.loads(proc.stdout)
    assert "not confirmed" in painted["head"][0]["status"]
    assert "worker up" in painted["worker"][0]["status"]
    assert painted["head"][0]["title"] == "example-engine"
    assert "Nothing running" not in json.dumps(painted)
    assert "retired" not in json.dumps(painted).lower()
