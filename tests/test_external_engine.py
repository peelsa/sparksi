# SPDX-License-Identifier: Apache-2.0

"""A library row served by an engine Sparksi did not start: shown as serving, never stopped.

Display only. Stop, Unload, Restart, switch eviction, run and the unmanaged stop all
refuse it, and no runner call is made.
"""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import threading

import pytest

from sparksi import jobs, launch, library, serve
from sparksi.config import CFG
from sparksi.library import annotate, external_engine, external_refusal
from sparksi.lock import Lock
from sparksi.serve import INDEX_HTML, STATE, DualServer, Handler

HEAD = CFG.head.name
WORKER = CFG.worker.name
PORT = 8014
SERVED = "example-served"
WEIGHTS = "/home/USER/.cache/huggingface/hub/models--example--pair-4bpw"
OTHER_WEIGHTS = "/home/USER/.cache/huggingface/hub/models--example--pair-tr3"
ENGINE = "example-engine"


def _row(row_id: str = "pair-a", weights: str = WEIGHTS, **extra) -> dict:
    row = {
        "id": row_id,
        "title": f"Example {row_id}",
        "placement": "pair",
        "port": PORT,
        "listen_host": HEAD,
        "identity": {"served": [SERVED], "container": f"{row_id}-head", "worker_container": f"{row_id}-worker"},
        "local": {HEAD: weights, WORKER: weights},
        "launch": {"start": ["/srv/example/start.sh"], "stop": ["/srv/example/start.sh", "stop"]},
        "exclusive": "big-pair",
    }
    row.update(extra)
    return row


def _probe(containers=(ENGINE,), mounts=None, answers=True, served=SERVED) -> dict:
    mounts = {ENGINE: [WEIGHTS + "/snapshots/abc", "/home/USER/.cache/example/state"]} if mounts is None else mounts
    return {
        "ok": True,
        "listening": {str(PORT): answers},
        "http": {str(PORT): {"path": "/v1/models", "status": 200, "models": [served]}} if answers else {},
        "containers": list(containers) if containers is not None else None,
        "mounts": mounts,
        "paths": {WEIGHTS: {"exists": True, "nonempty": True}, OTHER_WEIGHTS: {"exists": True, "nonempty": True}},
    }


def _probes(**head) -> dict:
    return {HEAD: _probe(**head), WORKER: _probe(mounts={ENGINE: [WEIGHTS]}, answers=False)}


def _annotated(rows=None, probes=None, runs=None) -> list[dict]:
    return annotate(rows or [_row()], probes or _probes(), runs or [])


# ---------- detection ----------


def test_outside_pair_serves_its_row_as_external() -> None:
    (row,) = _annotated()
    assert row["serving"] is True
    assert row["external"] is True
    assert row["live"] is False  # display only: never the key stop and switch plans read
    assert row["external_note"] == "Started outside a Sparksi run"
    assert row["external_engine"]["hosts"] == [HEAD, WORKER]
    assert row["external_engine"]["containers"] == {HEAD: [ENGINE], WORKER: [ENGINE]}
    assert row["external_engine"]["port"] == PORT


def test_single_engine_serves_its_row_as_external() -> None:
    probes = {HEAD: _probe(), WORKER: _probe(containers=(), mounts={}, answers=False)}
    (row,) = _annotated([_row(placement="single")], probes)
    assert row["external"] is True
    assert row["external_engine"]["hosts"] == [HEAD]


def test_weights_tell_two_rows_with_one_served_name_apart() -> None:
    rows = _annotated([_row("pair-a"), _row("pair-b", OTHER_WEIGHTS)])
    assert [(r["id"], r["external"], r["serving"]) for r in rows] == [("pair-a", True, True), ("pair-b", False, False)]


def test_pinned_container_running_keeps_the_catalog_rule() -> None:
    (row,) = _annotated(probes=_probes(containers=(ENGINE, "pair-a-head")))
    assert row["external"] is False
    assert row["serving"] is True
    assert row["live"] is True


@pytest.mark.parametrize(
    "head",
    [
        {"answers": False},
        {"served": "something-else"},
        {"mounts": {}},
        {"mounts_unknown": True},
        {"mounts": {ENGINE: ["/home/USER/.cache/huggingface"]}},  # a parent folder holds every model
        {"mounts": {"sparksi-example-head": [WEIGHTS]}, "containers": ("sparksi-example-head",)},
        {"containers": None},
    ],
)
def test_not_external_without_every_sign(head) -> None:
    unknown = head.pop("mounts_unknown", False)
    probes = _probes(**head)
    if unknown:
        probes[HEAD]["mounts"] = None  # an older agent, or docker could not be asked
    (row,) = _annotated(probes=probes)
    assert row["external"] is False
    assert "external_engine" not in row


def test_container_of_a_sparksi_run_is_never_external() -> None:
    run = {"id": "r1", "model_id": "other-model", "phase": "ready", "started": 1,
           "nodes": [{"role": "head", "host": HEAD, "container": ENGINE, "phase": "started"}]}
    (row,) = _annotated(runs=[run])
    assert row["external"] is False


def test_sparksi_owned_live_run_keeps_its_behavior() -> None:
    run = {"id": "r2", "model_id": "pair-a", "phase": "ready", "started": 1, "placement": "both",
           "nodes": [{"role": "head", "host": HEAD, "container": "sparksi-pair-a-head", "phase": "started"}]}
    probes = _probes(containers=(ENGINE, "sparksi-pair-a-head"))
    (row,) = _annotated(probes=probes, runs=[run])
    assert row["run_live"]["up"] is True
    assert row["external"] is False
    assert external_refusal(row) is None


def test_off_limits_port_is_never_external() -> None:
    assert external_engine(_row(port=8188), {HEAD: _probe()}) is None


# ---------- planning guards ----------


def _rows() -> list[dict]:
    return _annotated([_row("pair-a"), _row("pair-b", OTHER_WEIGHTS), _row(
        "single-b", OTHER_WEIGHTS, placement="single", port=8015,
        identity={"served": ["single-b"]},
        seats={HEAD: {"start": ["/srv/example/single.sh"], "stop": ["/srv/example/single.sh", "stop"], "port": 8015},
               WORKER: {"start": ["/srv/example/single.sh"], "stop": ["/srv/example/single.sh", "stop"], "port": 8015}},
    )])


def test_stop_plan_refuses_and_carries_no_command() -> None:
    row = _rows()[0]
    for lock_on in (True, False):
        plan = launch.plan_stop(row, lock_on=lock_on)
        assert plan["verdict"] == "REFUSE"
        assert plan["argv"] is None
        assert "started outside a Sparksi run" in plan["reasons"][0]


def test_switch_to_the_external_row_is_refused() -> None:
    rows = _rows()
    plan = launch.plan_switch({}, rows[0], rows, lock_on=False, host="pair")
    assert plan["verdict"] == "REFUSE"
    assert plan["steps"] == []
    assert "Stop it where it was started" in plan["reasons"][0]


@pytest.mark.parametrize("host", ["pair", HEAD, WORKER])
def test_switch_that_would_evict_the_external_engine_is_refused(host) -> None:
    rows = _rows()
    target = rows[1] if host == "pair" else rows[2]
    for lock_on in (True, False):
        plan = launch.plan_switch({}, target, rows, lock_on=lock_on, host="" if host == "pair" else host)
        assert plan["verdict"] == "REFUSE", plan
        assert plan["steps"] == []
        assert "started outside a Sparksi run" in plan["reasons"][0]


def test_no_stop_step_is_ever_built_for_an_external_row() -> None:
    row = dict(_rows()[0], live=True)  # even if something marked it live
    assert launch._stop_step(row, HEAD) is None


def test_run_and_drop_plans_refuse() -> None:
    rows = _rows()
    probes = _probes()
    plan = launch.plan_run({}, rows[0], probes, lock_on=False, rows=rows)
    assert plan["verdict"] == "REFUSE" and plan["argv"] is None
    other = launch.plan_run({}, rows[1], probes, lock_on=False, rows=rows)
    assert other["verdict"] == "REFUSE" and other["argv"] is None
    drop = launch.plan_drop(rows[0], HEAD, probes, lock_on=False, confirm=WEIGHTS)
    assert "weights are in use by an engine started outside a Sparksi run" in drop["reasons"]
    assert drop["request"] is None


def test_one_spark_site_refuses_the_same_way(monkeypatch) -> None:
    monkeypatch.setattr(launch, "_worker", lambda: "")
    rows = _rows()
    plan = launch.plan_switch({}, rows[2], rows, lock_on=False, host=HEAD)
    assert plan["verdict"] == "REFUSE"


# ---------- jobs and the API routes ----------


class Recorder:
    def __init__(self) -> None:
        self.calls: list = []

    def invoke(self, host, req, timeout=20):
        self.calls.append((host, req))
        return {"ok": False, "error": "a guarded test must not reach the runner"}

    def run_argv(self, host, argv, timeout=20):
        self.calls.append((host, argv))
        return {"ok": False}


def _fake_fleet(monkeypatch) -> None:
    snap = {"library": _rows(), "probes": _probes(), "hosts": []}
    for mod in (jobs, serve, library):
        monkeypatch.setattr(mod, "fleet", lambda *_a, **_k: snap)
    monkeypatch.setattr(jobs, "monitor_loaded", lambda: None)
    monkeypatch.setattr(serve, "monitor_loaded", lambda: None)
    monkeypatch.setattr(jobs, "load_in_progress", lambda: False)
    monkeypatch.setattr(serve, "load_in_progress", lambda: False)


def test_start_stop_refuses_without_touching_the_runner(monkeypatch) -> None:
    _fake_fleet(monkeypatch)
    rec = Recorder()
    for _ in range(2):  # a retry or double click gets the same answer
        out = jobs.start_stop({}, "pair-a", lock_on=False, runner=rec)
        assert out["verdict"] == "REFUSE" and out["external"] is True and out["stopped"] is False
        assert "Stop it where it was started" in out["error"]
    assert rec.calls == []


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
    rec = Recorder()
    STATE.runner = rec
    _fake_fleet(monkeypatch)
    monkeypatch.setattr(serve, "load_catalog", lambda: {})
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], rec
    finally:
        httpd.shutdown()
        STATE.runner = None


def _post(port: int, path: str, body: dict) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request("POST", path, body=json.dumps(body), headers={
            "Host": f"127.0.0.1:{port}", "Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}",
        })
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode() or "{}")
    finally:
        conn.close()


def test_api_stop_and_switch_refuse_a_double_click(server) -> None:
    port, rec = server
    asks = [
        ("/api/library/stop", {"id": "pair-a", "from_ui": True}),
        ("/api/switch", {"id": "pair-a", "from_ui": True}),
        ("/api/switch", {"id": "pair-b", "from_ui": True}),
        ("/api/switch", {"id": "single-b", "host": HEAD, "from_ui": True}),
        ("/api/library/run", {"id": "pair-b", "from_ui": True}),
    ]
    for path, body in asks:
        for _ in range(2):
            code, out = _post(port, path, body)
            assert code == 409, (path, code, out)
            assert "started outside a Sparksi run" in " ".join(out.get("reasons") or [out.get("error", "")])
    code, plan = _post(port, "/api/switch/plan", {"id": "pair-b", "from_ui": True})
    assert code == 200 and plan["verdict"] == "REFUSE" and plan["steps"] == []
    assert rec.calls == []


def test_api_stop_refuses_while_locked_too(server) -> None:
    port, rec = server
    STATE.lock.acquire("test")
    code, _out = _post(port, "/api/library/stop", {"id": "pair-a", "from_ui": True})
    assert code == 403
    assert rec.calls == []


def test_unmanaged_stop_refuses_an_engine_that_serves_a_row(monkeypatch) -> None:
    _fake_fleet(monkeypatch)
    rec = Recorder()
    why = serve._external_container_refusal({}, rec, HEAD, ENGINE)
    assert why and "started outside a Sparksi run" in why
    assert serve._external_container_refusal({}, rec, WORKER, ENGINE)
    assert serve._external_container_refusal({}, rec, HEAD, "unrelated") is None

    def broken(*_a, **_k):
        raise OSError("no answer")

    monkeypatch.setattr(library, "fleet", broken)
    assert "nothing was stopped" in serve._external_container_refusal({}, rec, HEAD, "unrelated")
    assert rec.calls == []


# ---------- page ----------


def _node_js(script: str) -> object:
    node = shutil.which("node")
    assert node, "node is required for the page logic tests"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_page_greys_unload_and_restart_and_blocks_stop() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    live = html.split("/* run-live-start */", 1)[1].split("/* run-live-end */", 1)[0]
    block = html.split("/* external-start */", 1)[1].split("/* external-end */", 1)[0]
    paint = "function paintModelActions" + html.split("function paintModelActions", 1)[1].split("/* overview-status-start */", 1)[0]
    stop = "function stopLoaded" + html.split("function stopLoaded", 1)[1].split("function portLiveRows", 1)[0]
    script = live + block + paint + stop + """
var libraryRows = [{id: "pair-a", external: true, serving: true}, {id: "pair-b", serving: true}];
var buttons = {"model-unload": {attrs: {}}, "model-restart": {attrs: {}}};
Object.keys(buttons).forEach(function (k) { buttons[k].setAttribute = function (n, v) { this.attrs[n] = v; }; });
var notes = {}, posts = [], picked = "pair-a";
function el(id) { return buttons[id] || null; }
function selectedCard() { return {id: picked}; }
function runFor() { return null; }
function modelServing(card) { return true; }
function setText(id, text) { notes[id] = text; }
function lockOn() { return false; }
function libPost(path, body) { posts.push([path, body.id]); }
paintModelActions();
var a = {disabled: buttons["model-unload"].disabled, title: buttons["model-unload"].title, restart: buttons["model-restart"].disabled};
stopLoaded("pair-a", "A", true, true, "model-load-note");
stopLoaded("pair-a", "A", true, true, "model-load-note");
picked = "pair-b"; paintModelActions();
stopLoaded("pair-b", "B", true, true, "model-load-note");
process.stdout.write(JSON.stringify({a: a, b: buttons["model-unload"].disabled, note: notes["model-load-note"] || "", posts: posts}));
"""
    out = _node_js(script)
    assert out["a"]["disabled"] is True and out["a"]["restart"] is True
    assert out["a"]["title"] == "Started outside a Sparksi run. Stop it where it was started."
    assert out["b"] is False
    assert out["posts"] == [["/api/library/stop", "pair-b"]]


# ---------- review: unknown worker, other Sparks ----------


def test_pair_holds_the_worker_even_when_its_probe_failed() -> None:
    probes = {HEAD: _probe(), WORKER: {"ok": False, "error": "unreachable"}}
    rows = annotate([_row("pair-a"), _row(
        "single-b", OTHER_WEIGHTS, placement="single", port=8015, identity={"served": ["single-b"]},
        seats={WORKER: {"start": ["/srv/example/single.sh"], "port": 8015}},
    )], probes, [])
    assert rows[0]["external"] is True
    assert rows[0]["external_engine"]["hosts"] == [HEAD, WORKER]
    plan = launch.plan_switch({}, rows[1], rows, lock_on=False, host=WORKER)
    assert plan["verdict"] == "REFUSE" and "started outside a Sparksi run" in plan["reasons"][0]


def test_a_spark_the_engine_does_not_hold_is_not_blocked() -> None:
    third = "spark-c"
    rows = _rows()
    target = dict(rows[2], seats={third: {"start": ["/srv/example/single.sh"], "port": 8015}}, on={third: "local"})
    plan = launch.plan_switch({}, target, rows, lock_on=False, host=third)
    assert not any("outside" in reason for reason in plan["reasons"]), plan
    assert plan["verdict"] == "READY"
    assert [step["op"] for step in plan["steps"]] == ["start"]
