# SPDX-License-Identifier: Apache-2.0

"""Stale runs and run-container identity: liveness follows the real container lists.

A failed or timed_out run whose containers are gone is not running. A library row counts the
containers its newest run recorded, so a Sparksi-launched pair shows serving and a library-path
stop stops that run. An old run never claims, or stops, a container name a newer run reuses.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time

from sparksi import library
from sparksi.library import annotate
from sparksi.recipes.runs import _save, reset_for_tests as reset_runs
from sparksi.serve import INDEX_HTML

A, B, C, D = "spark-a", "spark-b", "spark-c", "spark-d"


def _probe(containers, *, models=()) -> dict:
    answering = bool(models)
    return {
        "listening": {"8014": answering},
        "http": {"8014": {"status": 200, "models": list(models)}} if answering else {},
        "paths": {},
        "containers": containers,
    }


def _row(row_id: str, *, placement: str = "pair", container: str = "", worker: str = "", served: str = "") -> dict:
    ident = {"served": [served or row_id.upper()]}
    if container:
        ident["container"] = container
    if worker:
        ident["worker_container"] = worker
    return {"id": row_id, "title": row_id, "placement": placement, "port": 8014, "listen_host": A, "identity": ident}


def _node(role: str, host: str, container: str, phase: str = "started") -> dict:
    return {"role": role, "host": host, "container": container, "phase": phase,
            "run_id": f"{container}-{role}", "stop_argv": ["docker", "rm", "-f", container]}


def _run(run_id: str, model_id: str, nodes: list, *, phase: str = "failed", started: float = 100.0, **extra) -> dict:
    run = {"id": run_id, "model_id": model_id, "phase": phase, "started": started, "placement": "both",
           "port": 8014, "stopped_sides": [], "nodes": nodes, "version": 1}
    run.update(extra)
    return run


def _stale(phase: str = "failed", **extra) -> dict:
    # The 2026-11-01 record: the worker died first, the head stop answered ok, the worker was never marked.
    return _run("old1", "stale-pair", [_node("worker", B, "sparksi-stale-pair-worker"),
                                        _node("head", A, "sparksi-stale-pair-head")],
                phase=phase, failed_side="worker", stopped_sides=["head"], **extra)


def _one(rows, probes, runs) -> dict:
    return annotate(rows, probes, runs)[0]


# --- reproduction ---------------------------------------------------------------------------

def test_repro_stale_pair_card_unload_is_greyed_out_when_its_containers_are_gone() -> None:
    """The 2026-11-01 failed run kept Unload enabled although nothing of it was running."""
    rows = [{"id": "stale-pair", "serving": False, "live": False,
             "run_live": {"id": "old1", "live": False, "up": False, "sides": {"worker": "gone", "head": "stopped"}}}]
    script = (
        "var libraryRows = " + json.dumps(rows) + "; var RUN = " + json.dumps(_stale()) + ";\n"
        "function runFor() { return RUN; }\n"
        "process.stdout.write(JSON.stringify(modelServing({ id: 'stale-pair' })));"
    )
    assert _page(script) is False


def test_repro_pair_row_serving_from_its_run_containers() -> None:
    """The row said not serving: the catalog expects pair-tf, the run uses sparksi-pair-tf-*."""
    probes = {A: _probe(["sparksi-pair-tf-head"], models=["Pair-EXL3"]), B: _probe(["sparksi-pair-tf-worker"])}
    assert annotate([_tf_row()], probes, [_tf_run()])[0]["serving"] is True


def test_a_stale_failed_run_whose_containers_are_gone_is_not_running() -> None:
    item = _one([_row("stale-pair")], {A: _probe([]), B: _probe([])}, [_stale()])
    assert item["run_live"] == {"id": "old1", "live": False, "up": False,
                                "sides": {"worker": "gone", "head": "stopped"}}
    assert item["serving"] is False


def test_stale_timed_out_and_cancelled_runs_are_not_running() -> None:
    probes = {A: _probe([]), B: _probe([])}
    assert _one([_row("stale-pair")], probes, [_stale("timed_out")])["run_live"]["live"] is False
    assert _one([_row("stale-pair")], probes, [_stale("cancelled")])["run_live"]["live"] is False


def test_a_half_up_failed_run_stays_live() -> None:
    item = _one([_row("stale-pair")], {A: _probe([]), B: _probe(["sparksi-stale-pair-worker"])}, [_stale()])
    assert item["run_live"]["live"] is True and item["run_live"]["up"] is True
    assert item["run_live"]["sides"]["worker"] == "up"
    assert item["serving"] is True


def test_an_unreachable_machine_counts_as_maybe_up() -> None:
    item = _one([_row("stale-pair")], {A: _probe([])}, [_stale()])
    assert item["run_live"]["sides"]["worker"] == "unknown"
    assert item["run_live"]["live"] is True
    unlisted = _one([_row("stale-pair")], {A: _probe([]), B: _probe(None)}, [_stale()])
    assert unlisted["run_live"]["sides"]["worker"] == "unknown" and unlisted["run_live"]["live"] is True


def test_an_active_run_is_live_even_before_its_container_shows() -> None:
    for phase in ("starting", "waiting_health", "ready", "stopping"):
        run = _run("new1", "stale-pair", [_node("head", A, "sparksi-stale-pair-head")], phase=phase)
        assert _one([_row("stale-pair")], {A: _probe([])}, [run])["run_live"]["live"] is True, phase


def test_only_the_newest_run_of_the_row_is_reconciled() -> None:
    newer = _run("new2", "stale-pair", [_node("head", A, "sparksi-stale-pair-head", "stopped")],
                 phase="stopped", started=200.0)
    item = _one([_row("stale-pair")], {A: _probe([]), B: _probe(["sparksi-stale-pair-worker"])}, [_stale(), newer])
    assert item["run_live"]["id"] == "new2" and item["run_live"]["live"] is False


# --- Pair runs: the run's own containers are its identity -----------------------------------

def _tf_row() -> dict:
    return _row("pair-tf", container="pair-tf", worker="pair-tf", served="Pair-EXL3")


def _tf_run(phase: str = "ready") -> dict:
    return _run("tf1", "pair-tf", [_node("worker", B, "sparksi-pair-tf-worker"),
                                      _node("head", A, "sparksi-pair-tf-head")], phase=phase)


def test_a_sparksi_launched_pair_shows_serving() -> None:
    probes = {A: _probe(["sparksi-pair-tf-head", "webui"], models=["Pair-EXL3"]),
              B: _probe(["sparksi-pair-tf-worker"])}
    item = _one([_tf_row()], probes, [_tf_run()])
    assert item["serving"] is True
    assert item["run_live"]["up"] is True and item["run_live"]["sides"] == {"worker": "up", "head": "up"}


def test_legacy_catalog_identity_names_still_resolve() -> None:
    # The harness containers, no Sparksi run at all.
    probes = {A: _probe(["pair-tf"], models=["Pair-EXL3"]), B: _probe(["pair-tf"])}
    item = _one([_tf_row()], probes, [])
    assert item["serving"] is True and "run_live" not in item
    # Another harness answering with the same served id on the port is still not this row.
    other = {A: _probe(["someone-else"], models=["Pair-EXL3"]), B: _probe([])}
    assert _one([_tf_row()], other, [])["serving"] is False


def test_a_stopped_run_does_not_make_the_row_serving() -> None:
    probes = {A: _probe([]), B: _probe([])}
    assert _one([_tf_row()], probes, [_tf_run("stopped")])["serving"] is False


def test_four_sparks_worker_on_the_fourth() -> None:
    run = _run("p4", "stale-pair", [_node("head", A, "sparksi-stale-pair-head"),
                                    _node("worker", D, "sparksi-stale-pair-worker")], phase="failed")
    probes = {A: _probe([]), B: _probe(["sparksi-stale-pair-worker"]), C: _probe([]), D: _probe([])}
    # The same name on another Spark is not this side.
    assert _one([_row("stale-pair")], probes, [run])["run_live"]["live"] is False
    probes[D] = _probe(["sparksi-stale-pair-worker"])
    assert _one([_row("stale-pair")], probes, [run])["run_live"]["live"] is True


def test_single_spark_head_and_worker_only_runs() -> None:
    head = _run("h1", "solo", [_node("head", A, "sparksi-solo")], phase="failed")
    assert _one([_row("solo", placement="single")], {A: _probe([])}, [head])["run_live"]["live"] is False
    assert _one([_row("solo", placement="single")], {A: _probe(["sparksi-solo"])}, [head])["serving"] is True
    worker = _run("w1", "solo", [_node("worker", B, "sparksi-solo")], phase="timed_out")
    assert _one([_row("solo", placement="single")], {A: _probe([]), B: _probe(["sparksi-solo"])}, [worker])["run_live"]["live"] is True


def test_an_adopted_run_is_reconciled_like_any_other() -> None:
    run = _run("ad1", "stale-pair", [_node("head", A, "sparksi-stale-pair")], phase="failed", started_by="external")
    assert _one([_row("stale-pair")], {A: _probe([])}, [run])["run_live"]["live"] is False
    assert _one([_row("stale-pair")], {A: _probe(["sparksi-stale-pair"])}, [run])["run_live"]["live"] is True


def test_a_node_without_a_recorded_container_counts_as_maybe_up() -> None:
    run = _run("o1", "stale-pair", [{"role": "head", "host": A, "phase": "started"}], phase="failed")
    assert library.run_sides(run, {A: _probe([])}, [run]) == {"head": "unknown"}


# --- reused container names ------------------------------------------------------------------

def test_an_old_run_does_not_claim_a_name_a_newer_run_reuses() -> None:
    newer = _run("new3", "stale-pair", [_node("worker", B, "sparksi-stale-pair-worker"),
                                        _node("head", A, "sparksi-stale-pair-head")], phase="ready", started=300.0)
    probes = {A: _probe(["sparksi-stale-pair-head"]), B: _probe(["sparksi-stale-pair-worker"])}
    assert library.run_sides(_stale(), probes, [_stale(), newer]) == {"worker": "superseded", "head": "stopped"}
    item = _one([_row("stale-pair")], probes, [_stale(), newer])
    assert item["run_live"]["id"] == "new3" and item["run_live"]["live"] is True


def test_a_stop_of_an_old_run_never_touches_a_reused_name(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.runs import get_run, stop_recorded

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    old = _save(_run("old9", "other", [_node("head", A, "sparksi-shared"), _node("worker", B, "sparksi-old-w")],
                     phase="failed", stopped_sides=[]))
    _save(_run("new9", "shared", [_node("head", A, "sparksi-shared")], phase="ready", started=500.0))
    sent: list[tuple[str, str]] = []

    class Stub:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "stop":
                sent.append((host, req["argv"][-1]))
                return {"ok": True}
            return {"ok": True, "listening": {}, "http": {}, "containers": []}

    done = stop_recorded(old, Stub())
    assert sent == [(B, "sparksi-old-w")]
    assert done["phase"] == "stopped"
    head = next(n for n in get_run("old9")["nodes"] if n["role"] == "head")
    assert head["phase"] == "stopped" and head["superseded_by"] == "new9"
    assert get_run("new9")["phase"] == "ready"
    reset_runs()


# --- routes: library-path stop ---------------------------------------------------------------

def _library_stop(monkeypatch, tmp_path, row: dict, probes: dict, runs: list):
    from sparksi import jobs

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    for run in runs:
        _save(run)
    item = annotate([row], probes, runs)[0]
    item["launch"] = {"host": A, "stop": ["/opt/harness/stop.sh"]}
    monkeypatch.setattr("sparksi.jobs.fleet", lambda *a, **k: {"library": [item], "probes": probes})
    monkeypatch.setattr("sparksi.jobs.load_in_progress", lambda: False)
    harness: list = []
    monkeypatch.setattr("sparksi.jobs._start_argv",
                        lambda catalog, kind, plan, runner: harness.append(plan["argv"]) or {"verdict": "STARTED"})
    sent: list[tuple[str, str]] = []

    class Stub:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "stop":
                sent.append((host, req["argv"][-1]))
                return {"ok": True}
            return {"ok": True, "listening": {}, "http": {}, "containers": []}

    out = jobs.start_stop({"models": []}, row["id"], lock_on=False, runner=Stub())
    reset_runs()
    return out, harness, sent


def test_library_stop_of_a_sparksi_pair_stops_its_run_containers(monkeypatch, tmp_path) -> None:
    probes = {A: _probe(["sparksi-pair-tf-head"], models=["Pair-EXL3"]),
              B: _probe(["sparksi-pair-tf-worker"])}
    out, harness, sent = _library_stop(monkeypatch, tmp_path, _tf_row(), probes, [_tf_run()])
    assert harness == []
    assert sorted(sent) == [(A, "sparksi-pair-tf-head"), (B, "sparksi-pair-tf-worker")]
    assert out["verdict"] == "READY" and out["stopped"] is True and out["run_id"] == "tf1"


def test_library_stop_of_harness_containers_still_runs_the_catalog_script(monkeypatch, tmp_path) -> None:
    probes = {A: _probe(["pair-tf"], models=["Pair-EXL3"]), B: _probe(["pair-tf"])}
    out, harness, sent = _library_stop(monkeypatch, tmp_path, _tf_row(), probes, [_tf_run("stopped")])
    assert harness == [["/opt/harness/stop.sh"]] and sent == []


def test_library_stop_of_a_stale_run_is_409_not_running(monkeypatch, tmp_path) -> None:
    out, harness, sent = _library_stop(monkeypatch, tmp_path, _row("stale-pair"), {A: _probe([]), B: _probe([])}, [_stale()])
    assert out["not_running"] is True and out["verdict"] == "REFUSE"
    assert harness == [] and sent == []


def test_library_stop_while_the_run_is_already_stopping_sends_nothing(monkeypatch, tmp_path) -> None:
    probes = {A: _probe(["sparksi-pair-tf-head"]), B: _probe(["sparksi-pair-tf-worker"])}
    out, harness, sent = _library_stop(monkeypatch, tmp_path, _tf_row(), probes, [_tf_run("stopping")])
    assert harness == [] and sent == []
    assert out["stopped"] is False


def test_library_stop_of_a_half_up_failed_run_stops_only_the_live_side(monkeypatch, tmp_path) -> None:
    probes = {A: _probe([]), B: _probe(["sparksi-stale-pair-worker"])}
    out, harness, sent = _library_stop(monkeypatch, tmp_path, _row("stale-pair"), probes, [_stale()])
    assert harness == [] and sent == [(B, "sparksi-stale-pair-worker")]
    assert out["stopped"] is True


# --- the page copy of the rule ---------------------------------------------------------------

def _page(script: str) -> object:
    node = shutil.which("node")
    assert node, "node is required for the page logic tests"
    html = INDEX_HTML.read_text(encoding="utf-8")
    live = html.split("/* run-live-start */", 1)[1].split("/* run-live-end */", 1)[0]
    serving = "function modelServing" + html.split("function modelServing", 1)[1].split("function paintModelActions", 1)[0]
    out = subprocess.run([node, "-e", live + serving + script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_the_page_greys_out_a_stale_run_and_keeps_a_half_up_one() -> None:
    stale = _stale()
    rows = [
        {"id": "stale-pair", "serving": False, "run_live": {"id": "old1", "live": False, "up": False}},
        {"id": "half", "serving": True, "run_live": {"id": "old1h", "live": True, "up": True}},
        {"id": "unknown", "serving": None},
        {"id": "newer", "serving": False, "run_live": {"id": "someone-newer", "live": False, "up": False}},
        {"id": "tf", "serving": True, "run_live": {"id": "tf1", "live": True, "up": True}},
    ]
    runs = {
        "stale-pair": stale,
        "half": dict(stale, id="old1h", model_id="half"),
        "unknown": dict(stale, id="old1u", model_id="unknown"),
        # The row reconciled a different run: trust this run's record.
        "newer": dict(stale, id="old1n", model_id="newer"),
        "tf": _tf_run(),
    }
    script = (
        "var libraryRows = " + json.dumps(rows) + "; var RUNS = " + json.dumps(runs) + ";\n"
        "function runFor(id) { return RUNS[id] || null; }\n"
        "process.stdout.write(JSON.stringify(['stale-pair','half','unknown','newer','tf'].map(function (id) {"
        " return [modelServing({ id: id }), runIsLive(RUNS[id])]; })));"
    )
    assert _page(script) == [[False, False], [True, True], [True, True], [True, True], [True, True]]


def test_a_stale_run_with_its_record_closed_stays_greyed_without_a_library_row() -> None:
    script = (
        "var libraryRows = []; var RUN = " + json.dumps(_stale("stopped")) + ";\n"
        "function runFor() { return RUN; }\n"
        "process.stdout.write(JSON.stringify([modelServing({ id: 'stale-pair' }), runIsLive(RUN), runIsLive(null)]));"
    )
    assert _page(script) == [False, False, False]


def test_age_does_not_decide_liveness() -> None:
    # Reconciliation reads container lists only; no clock or age threshold decides liveness.
    old = _stale(started=time.time() - 30 * 86400)
    assert library.run_sides(old, {A: _probe([]), B: _probe(["sparksi-stale-pair-worker"])}, [old])["worker"] == "up"


def test_library_run_path_stop_refuses_while_locked(monkeypatch, tmp_path) -> None:
    from sparksi import jobs

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    _save(_tf_run())
    out = jobs._stop_run_of_row({"models": []}, _tf_row(), "tf1", lock_on=True, runner=object())
    assert out["verdict"] == "REFUSE" and out["reasons"] == ["lock is ON"]
    reset_runs()


def test_the_run_stop_route_on_a_stale_run_only_closes_its_record(monkeypatch, tmp_path) -> None:
    """/api/runs/<id>/stop on the 2026-11-01 record: the gone worker gets its own exact-name stop, the head none."""
    from sparksi.recipes.runs import get_run
    from sparksi.recipes.service import halt_run

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    _save(_stale())
    sent: list[tuple[str, str]] = []

    class Stub:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "stop":
                sent.append((host, req["argv"][-1]))
                return {"ok": True}
            return {"ok": True, "listening": {}, "http": {}, "containers": []}

    code, body = halt_run({"models": []}, Stub(), "old1")
    assert code == 200 and body["stopped"] is True
    assert sent == [(B, "sparksi-stale-pair-worker")]
    assert get_run("old1")["phase"] == "stopped"
    reset_runs()


def test_a_second_start_is_refused_while_a_run_is_active(monkeypatch, tmp_path) -> None:
    """Why only stop_recorded needs the reused-name guard.

    The health-fail, timeout and pair-side stops act only on the run being watched, in
    starting, waiting_health or ready (the watcher exits on stopping, whose stop is
    stop_recorded with the guard). No second run of the same model, or on the same port
    (container names derive from both), can start in those phases, so no newer run can
    hold the watched run's names while those paths run.
    """
    from sparksi.config import CFG
    from sparksi.recipes.service import begin_run

    host = CFG.head.name

    def row(model_id: str) -> dict:
        return {"id": model_id, "title": model_id.title(), "format": "fp8", "gib": 8, "engines": ["vllm"],
                "local": {host: "/srv/example/cache/weights"}, "port": 8014}

    catalog = {"library": [row("demo"), row("other")],
               "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
               "exclusive_groups": {}, "never_touch": {}}
    started: list = []

    def no_start(*args, **kwargs):
        started.append(args)
        raise AssertionError("a second start must be refused before anything is launched")

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    monkeypatch.setattr("sparksi.recipes.service.start_rendered", no_start)
    for phase, word in (("starting", "loading"), ("waiting_health", "loading"), ("ready", "running")):
        _save({"id": "busy1", "model_id": "demo", "engine": "vllm", "placement": "head", "port": 8014,
               "phase": phase, "started": time.time(),
               "nodes": [{"role": "head", "host": host, "container": "sparksi-demo-8014", "phase": "started"}]})
        for model in ("demo", "other"):
            code, body = begin_run(catalog, None, {"id": model, "engine": "vllm", "placement": "head", "port": 8014})
            assert code == 400, (phase, model, body)
            why = " ".join(body.get("conflicts") or [])
            if model == "demo":
                assert f"Demo is already {word}" in why, (phase, why)
            else:
                assert "port 8014 is in use by Demo" in why, (phase, why)
    assert started == []
    reset_runs()
