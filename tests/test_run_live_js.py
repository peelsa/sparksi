# SPDX-License-Identifier: Apache-2.0

"""Unload stays live while any side of a failed or timed-out run may still be up."""

import json
import shutil
import subprocess

from sparksi.serve import INDEX_HTML


def _node_js(script: str) -> object:
    # No skip: CI installs node for the test job, and a missing node must fail there.
    node = shutil.which("node")
    assert node, "node is required for the page logic tests"
    out = subprocess.run([node, "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def _live_code() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.split("/* run-live-start */", 1)[1].split("/* run-live-end */", 1)[0]


def _live(runs: list) -> list:
    return _node_js(_live_code() + "\nprocess.stdout.write(JSON.stringify(" + json.dumps(runs) + ".map(runHasLiveSide)));")


def _node(role: str, phase: str) -> dict:
    return {"role": role, "phase": phase}


def test_run_has_live_side() -> None:
    runs = [
        None,
        {"phase": "ready", "nodes": [_node("head", "started")]},
        {"phase": "starting", "nodes": []},
        {"phase": "stopped", "nodes": [_node("head", "started")]},
        {"phase": "cancelled", "nodes": [_node("head", "started")]},
        # Health refused or served elsewhere: nothing was stopped, the head is still up.
        {"phase": "failed", "nodes": [_node("head", "started")], "stopped_sides": []},
        # Launcher refused the worker of a pair: the head came up and is still running.
        {"phase": "failed", "failed_side": "worker",
         "nodes": [_node("head", "started"), _node("worker", "failed")], "stopped_sides": []},
        # A pair whose dead side was noted and whose other side was stopped.
        {"phase": "failed", "failed_side": "worker",
         "nodes": [_node("head", "started"), _node("worker", "stopped")], "stopped_sides": ["head"]},
        # A pair where only one stop answered ok.
        {"phase": "failed", "nodes": [_node("head", "started"), _node("worker", "started")],
         "stopped_sides": ["worker"]},
        {"phase": "timed_out", "nodes": [_node("head", "started"), _node("worker", "started")],
         "stopped_sides": ["head", "worker"]},
        {"phase": "timed_out", "nodes": [_node("head", "started")], "stopped_sides": []},
        # Stop errors leave nodes stopped one by one.
        {"phase": "failed", "nodes": [_node("head", "stopped"), _node("worker", "stopped")]},
    ]
    assert _live(runs) == [False, True, True, False, False, True, True, False, True, False, True, False]


def test_unknown_serving_keeps_unload_enabled() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    serving = "function modelServing" + html.split("function modelServing", 1)[1].split("function paintModelActions", 1)[0]
    rows = [
        {"id": "up", "serving": True, "live": False},
        {"id": "down", "serving": False, "live": True},
        {"id": "unknown", "serving": None, "live": False},
        {"id": "older", "live": True},
        {"id": "older-off", "live": False},
    ]
    script = (
        _live_code() + serving
        + "var libraryRows = " + json.dumps(rows) + ";\nfunction runFor() { return null; }\n"
        + "process.stdout.write(JSON.stringify(libraryRows.map(function (r) { return modelServing({ id: r.id }); })"
        + ".concat([modelServing({ id: 'missing' })])));"
    )
    assert _node_js(script) == [True, False, True, True, False, False]
