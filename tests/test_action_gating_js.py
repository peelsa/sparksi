# SPDX-License-Identifier: Apache-2.0

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _gating() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.split("/* gating-start */", 1)[1].split("/* gating-end */", 1)[0]


def _run(body: str):
    script = _gating() + "\n" + body + "\nprocess.stdout.write(JSON.stringify(out));\n"
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_load_words_quiet_but_reachable():
    out = _run("out = loadWordsFor({monitor: true, monitorError: false, locked: false, report: 'missing', reachable: true, seat: true, seated: true});")
    assert out == "Load"


def test_load_words_stale_but_reachable():
    out = _run("out = loadWordsFor({monitor: true, monitorError: false, locked: false, report: 'stale', reachable: true, seat: true, seated: true});")
    assert out == "Load"


def test_load_words_quiet_unreachable_missing():
    out = _run("out = loadWordsFor({monitor: true, monitorError: false, locked: false, report: 'missing', reachable: false, seat: true, seated: true});")
    assert out == "No report yet"


def test_load_words_quiet_unreachable_stale():
    out = _run("out = loadWordsFor({monitor: true, monitorError: false, locked: false, report: 'stale', reachable: false, seat: true, seated: true});")
    assert out == "Load (stale)"


def test_load_words_locked_even_when_fresh():
    out = _run("out = loadWordsFor({monitor: true, monitorError: false, locked: true, report: 'fresh', reachable: false, seat: true, seated: true});")
    assert out == "Unlock first"


def test_gates_quiet_reachable_unlocked():
    out = _run(
        "out = [actionGates({usable: true, locked: false, hasRow: true, present: 'absent', stopName: 'x'}),"
        " actionGates({usable: true, locked: false, hasRow: true, present: 'live', stopName: 'x'})];"
    )
    assert out[0]["stop"] is False
    assert out[0]["copy"] is False
    assert out[1]["drop"] is False


def test_gates_locked_but_usable():
    out = _run(
        "out = [actionGates({usable: true, locked: true, hasRow: true, present: 'absent', stopName: 'x'}),"
        " actionGates({usable: true, locked: true, hasRow: true, present: 'live', stopName: 'x'})];"
    )
    assert out[0]["stop"] is True
    assert out[0]["copy"] is False
    assert out[1]["drop"] is False


def test_gates_not_usable():
    out = _run("out = actionGates({usable: false, locked: false, hasRow: true, present: 'absent', stopName: 'x'});")
    assert out == {"stop": True, "copy": True, "drop": True}


def test_pair_head_fresh_worker_missing_but_reachable():
    out = _run(
        "out = pairWordsFor({monitor: true, monitorError: false, locked: false,"
        " head: {report: 'fresh', reachable: false, seat: true},"
        " worker: {report: 'missing', reachable: true, seat: true}});"
    )
    assert out == "Load"


def test_pair_head_fresh_worker_missing_unreachable():
    out = _run(
        "out = pairWordsFor({monitor: true, monitorError: false, locked: false,"
        " head: {report: 'fresh', reachable: false, seat: true},"
        " worker: {report: 'missing', reachable: false, seat: true}});"
    )
    assert out == "Waiting on worker"
