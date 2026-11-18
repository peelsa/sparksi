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

STOPPED_ID = "0a0a0a0a0a0a"


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


# A Sparksi run names its own containers. Two catalog rows (or a recipe copy) can serve
# one id, so the served name alone cannot say which run owns the live process.
SHARED = "shared-served"


def _shared_pair(head: str = "run-beta-head", worker: str = "run-beta-worker") -> dict:
    row = _row_from_producer(
        {
            "fresh": True,
            "http": {8014: {"served": SHARED, "metrics": {"num_requests_running": 0}}},
            "engines": [{"name": head, "port": 8014}],
            "truth": "",
        },
        {"fresh": True, "http": {}, "containers": [{"name": worker, "status": "Up 2 hours"}]},
    )
    assert row["served"] == SHARED
    assert row["container_source"] == "engine"
    assert row["container"] == head
    assert row["worker_name"] == worker
    return row


def _beta_run(phase: str = "ready", run_id: str = "beta-run", **over) -> dict:
    run = {
        "id": run_id,
        "model_id": "beta",
        "phase": phase,
        "port": 8014,
        "placement": "both",
        "started": 20,
        "nodes": [
            {"role": "worker", "phase": "started", "container": "run-beta-worker"},
            {"role": "head", "phase": "started", "container": "run-beta-head"},
        ],
    }
    run.update(over)
    return run


def _alpha_failed() -> dict:
    """An older run of another row that serves the same id. It failed with its sides still named."""
    return {
        "id": "alpha-old",
        "model_id": "alpha",
        "phase": "failed",
        "port": 8014,
        "placement": "both",
        "started": 10,
        "stopped_sides": ["worker"],
        "nodes": [
            {"role": "worker", "phase": "started", "container": "run-alpha-worker"},
            {"role": "head", "phase": "started", "container": "run-alpha-head"},
        ],
    }


def _pair_rows(loaded: list, runs: list) -> list:
    return reconcile(loaded, runs, hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)


def test_run_whose_containers_answer_owns_a_shared_served_name() -> None:
    rows = _pair_rows([_shared_pair()], [_alpha_failed(), _beta_run(), _stopped()])
    assert [row["state"] for row in rows] == ["ready", "ready"]
    assert [row["run_id"] for row in rows] == ["beta-run", "beta-run"]
    assert all(row["title"] == "beta" for row in rows)
    assert all(row["text"] == "Ready · port 8014" for row in rows)
    titled = _pair_rows([_shared_pair()], [_beta_run(title="Beta title")])
    assert [row["title"] for row in titled] == ["Beta title", "Beta title"]


def test_container_owner_needs_an_active_run_and_the_live_names() -> None:
    for phase in ("failed", "timed_out", "stopped", "cancelled"):
        rows = _pair_rows([_shared_pair()], [_beta_run(phase=phase)])
        assert [row["state"] for row in rows] == ["answering", "answering"], phase
        assert all(row["run_id"] == "" for row in rows), phase
        assert all(row["title"] == SHARED for row in rows), phase
    # A replaced container is a different instance, even with the same served name and port.
    different = dict(_shared_pair())
    different.update(container="someone-else-head", worker_name="someone-else-worker")
    replaced = _pair_rows([different], [_beta_run()])
    assert all(row["run_id"] == "" for row in replaced)
    # Another port is another process.
    other_port = _pair_rows([_shared_pair()], [_beta_run(port=8015)])
    assert all(row["run_id"] == "" for row in other_port)
    # Loading phases still own their process and say so.
    loading = _pair_rows([_shared_pair()], [_beta_run(phase="waiting_health")])
    assert [row["state"] for row in loading] == ["loading", "loading"]
    assert {row["run_id"] for row in loading} == {"beta-run"}
    # A side whose stop answered no longer owns the name it recorded.
    half = _pair_rows([_shared_pair()], [_beta_run(stopped_sides=["worker"])])
    assert [(row["role"], row["run_id"]) for row in half] == [("head", "beta-run"), ("worker", "")]
    # The newest active owner wins when an older record names the same containers.
    older = _beta_run(run_id="beta-older", started=5)
    newest = _pair_rows([_shared_pair()], [older, _beta_run()])
    assert {row["run_id"] for row in newest} == {"beta-run"}


def test_static_or_unmanaged_container_names_do_not_prove_ownership() -> None:
    truth = dict(_shared_pair())
    truth["container_source"] = "truth"
    rows = _pair_rows([truth], [_beta_run()])
    assert all(row["run_id"] == "" for row in rows)
    hand = dict(_shared_pair())
    hand.update(container="hand-started", worker_name="hand-started-too")
    unmanaged = _pair_rows([hand], [_beta_run()])
    assert [row["state"] for row in unmanaged] == ["answering", "answering"]
    assert all(row["run_id"] == "" for row in unmanaged)


def test_half_up_pair_stays_split_with_a_container_owner() -> None:
    row = _split_pair()
    run = _beta_run(
        nodes=[
            {"role": "head", "phase": "started", "container": "example-engine-head"},
            {"role": "worker", "phase": "started", "container": "example-engine-worker"},
        ]
    )
    rows = _pair_rows([row], [run])
    assert [item["state"] for item in rows] == ["split", "split"]
    assert all(item["run_id"] == "" for item in rows)


def test_container_owner_on_one_to_four_sparks() -> None:
    def single(host: str, container: str, port: int) -> dict:
        return {
            "name": SHARED,
            "served": SHARED,
            "host": host,
            "port": port,
            "head_up": True,
            "container": container,
            "container_source": "engine",
            "state": "UP",
        }

    def run(role: str, container: str, port: int) -> dict:
        return {
            "id": f"beta-{role}",
            "model_id": "beta",
            "phase": "ready",
            "port": port,
            "placement": role,
            "started": 3,
            "nodes": [{"role": role, "phase": "started", "container": container}],
        }

    hosts = [{"id": name, "fresh": True} for name in ("n1", "n2", "n3", "n4")]
    one = reconcile([single("n1", "run-beta-head", 8030)], [run("head", "run-beta-head", 8030)],
                    hosts=hosts, head_id="n1", worker_id="")
    assert [(row["role"], row["run_id"]) for row in one] == [("head", "beta-head")]
    extra = {"node3": "n3", "node4": "n4"}
    loaded = [single("n3", "run-beta-node3", 8031), single("n4", "run-beta-node4", 8032)]
    runs = [run("node3", "run-beta-node3", 8031), run("node4", "run-beta-node4", 8032)]
    for count, ids in ((3, {"node3": "n3"}), (4, extra)):
        rows = reconcile(loaded, runs, hosts=hosts, head_id="n1", worker_id="n2", extra_ids=ids)
        by_role = {row["role"]: row for row in rows}
        assert len(rows) == count
        assert by_role["head"]["state"] == "idle" and by_role["worker"]["state"] == "idle"
        for role in ids:
            assert by_role[role]["state"] == "ready", (count, role)
            assert by_role[role]["run_id"] == f"beta-{role}"
    # A single-host container on node3 is compared with that host's own container name.
    swapped = reconcile([single("n3", "run-beta-node4", 8031)], [run("node3", "run-beta-node3", 8031)],
                        hosts=hosts, head_id="n1", worker_id="n2", extra_ids={"node3": "n3"})
    assert {row["role"]: row["run_id"] for row in swapped}["node3"] == ""


def test_shared_served_id_follows_run_containers_before_catalog_prefix() -> None:
    from tests.test_serving_reconcile import _accept, _js

    row = _shared_pair()
    runs = [_alpha_failed(), _beta_run()]
    got = _js(
        """
        libraryRows = [
          { id: "alpha", title: "Alpha", identity: { served: ["shared-served"], served_prefix: "shared", container: "alpha-harness" } },
          { id: "beta", title: "Beta recipe", identity: { served: ["shared-served"], container: "beta-harness" } }
        ];
        modelCards = libraryRows;
        modelRuns = %s;
        %s
        var hit = lastMonitor.loaded[0];
        var result = {
          id: libraryIdFor(hit),
          head: serveEntries("head").map(function (r) { return r.id; }),
          worker: serveEntries("worker").map(function (r) { return r.id; }),
          shared: sharedRun(modelRuns) && sharedRun(modelRuns).id,
          alpha: roleStanding(modelRuns[0], "head"),
          beta: roleStanding(modelRuns[1], "head"),
          alphaLabel: liveLabel(modelCards[0]).text,
          betaLabel: liveLabel(modelCards[1]).text
        };
        modelRuns[1].phase = "failed";
        result.failed = roleStanding(modelRuns[1], "head");
        modelRuns[1].phase = "timed_out";
        result.timed = roleStanding(modelRuns[1], "head");
        modelRuns[1].phase = "stopped";
        modelRuns[1].nodes.forEach(function (n) { n.phase = "stopped"; });
        result.stopped = libraryIdFor(hit);
        result.stoppedAlpha = roleStanding(modelRuns[0], "head");
        modelRuns[1].phase = "ready";
        modelRuns[1].nodes.forEach(function (n) { n.phase = "started"; });
        hit.container = "unmanaged-head";
        hit.worker_name = "unmanaged-worker";
        result.unmanaged = libraryIdFor(hit);
        result.unmanagedAlpha = roleStanding(modelRuns[0], "head");
        result.unmanagedBeta = roleStanding(modelRuns[1], "head");
        var single = {
          name: "shared-served", served: "shared-served", host: WORKER_ID, port: 8020,
          head_up: true, container: "run-beta-worker", container_source: "engine", state: "UP"
        };
        modelRuns = [{
          id: "beta-one", model_id: "beta", phase: "ready", placement: "worker", port: 8020, started: 30,
          nodes: [{ role: "worker", phase: "started", container: "run-beta-worker" }]
        }];
        result.single = libraryIdFor(single);
        single.host = HEAD_ID;
        result.singleWrongHost = libraryIdFor(single);
        process.stdout.write(JSON.stringify(result));
        """ % (json.dumps(runs), _accept({"loaded": [row], "hosts": _hosts()}))
    )
    assert got["id"] == "beta"
    assert got["head"] == ["beta-run"] and got["worker"] == ["beta-run"]
    assert got["shared"] == "beta-run"
    assert got["alpha"] == "" and got["beta"] == "confirmed"
    assert "failed" in got["alphaLabel"].lower()
    assert got["betaLabel"] == "Ready · port 8014"
    assert got["failed"] == "survivor" and got["timed"] == "survivor"
    # Without an owning run the catalog rules still decide (here the prefix), but a run
    # whose recorded containers are not the live ones never labels them as a survivor.
    assert got["stopped"] == "alpha" and got["stoppedAlpha"] == ""
    assert got["unmanaged"] == "alpha"
    assert got["unmanagedAlpha"] == got["unmanagedBeta"] == ""
    # One host, no span: only the node on that host's role is compared.
    assert got["single"] == "beta"
    assert got["singleWrongHost"] == "alpha"
