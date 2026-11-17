# SPDX-License-Identifier: Apache-2.0

"""The page's groupMachines matches the Python grouper and drops addresses."""

import json
import subprocess

from sparksi.groups import group_machines
from sparksi.serve import INDEX_HTML
from tests.test_groups import SHARED_PORT

NODES = [
    {"role": "head", "id": "a", "name": "head"},
    {"role": "worker", "id": "b", "name": "worker"},
]
MODELS = [
    {"id": "pair-exl3", "title": "Pair EXL3", "port": 8014, "placement": "pair"},
    {"id": "single-a", "title": "Single A", "port": 8015, "placement": "single", "seats": {"head": {"port": 8015}}},
    {"id": "comfy", "title": "ComfyUI video", "port": 8188, "placement": "single", "seats": {"head": {"port": 8188}}},
]
LIBRARY = [
    {"id": "pair-exl3", "gib": 42, "on": {"head": "live", "worker": "live"}, "local": {"head": "planted-path"}},
    {"id": "single-a", "gib": 18, "on": {"head": "local", "worker": "absent"}},
    {"id": "comfy", "gib": 40, "on": {"head": "live", "worker": "absent"}},
]


def _live(head_fresh=True, worker=None, loaded=None):
    worker = {"id": "b", "fresh": True, "age": 1} if worker is None else worker
    hosts = [{
        "id": "a",
        "fresh": head_fresh,
        "age": 1,
        "mem_used_gib": 80.4,
        "mem_total_gib": 119,
        "cpu_pct": 22,
        "gpu_pct": 41,
        "gpu_temp_f": 98,
        "disk_used_gib": 800,
        "disk_total_gib": 3700,
        "mount_archive": True,
        "ips": {"lan": "192.0.2.11", "tailscale": "100.64.0.11"},
    }]
    if worker is not False:
        hosts.append(worker)
    return {"hosts": hosts, "loaded": loaded or []}


def _cases():
    return [
        _live(loaded=[
            {"name": "Pair EXL3", "host": "a", "port": 8014, "state": "idle", "span": ["a", "b"]},
            {"name": "Single A", "host": "a", "port": 8015, "state": "UP"},
            {"name": "ComfyUI video", "host": "a", "port": 8188, "state": "idle"},
        ]),
        _live(worker={"id": "b", "fresh": False, "age": 12, "mem_used_gib": 54, "cpu_pct": 11}),
        _live(worker=False),
        _live(loaded=[{"name": "ComfyUI video", "host": "a", "port": 8188, "state": "DOWN"}]),
        _live(loaded=[{"name": "Pair EXL3", "host": "a", "port": 8014, "state": "GEN", "tok_s": 42, "span": ["a", "b"]}]),
        _live(),
    ]


def test_page_group_machines_matches_python():
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* group-machines-start */", 1)[1].split("/* group-machines-end */", 1)[0]
    payload = []
    expect = []
    for live in _cases():
        payload.append({
            "nodes": NODES,
            "models": MODELS,
            "live": live,
            "library": LIBRARY,
            "controller": {"name": "controller"},
            "extras": [{"name": "extra"}],
        })
        expect.append(json.loads(json.dumps(group_machines(
            NODES, MODELS, live=live, library=LIBRARY,
            controller={"name": "controller"}, extras=[{"name": "extra"}],
        ))))
    js = script + "\nvar cases = " + json.dumps(payload) + ";\n"
    js += "var out = cases.map(function (c) { return groupMachines(c.nodes, c.models, c.live, c.library, c.controller, c.extras); });\n"
    js += "process.stdout.write(JSON.stringify(out));\n"
    proc = subprocess.run(["node"], input=js, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got == expect
    blob = json.dumps(got)
    assert "192.0.2.11" not in blob
    assert "100.64.0.11" not in blob
    assert "planted-path" not in blob


def test_page_group_machines_matches_served_name_rules():
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* group-machines-start */", 1)[1].split("/* group-machines-end */", 1)[0]
    cases = [
        (SHARED_PORT, _live(loaded=[{
            "name": "GLM A…",
            "served": "glm-a-id",
            "host": "a",
            "port": 8014,
            "state": "GEN",
            "tok_s": 20,
            "span": ["a", "b"],
        }])),
        (SHARED_PORT, _live(loaded=[{
            "name": "Qwen…",
            "served": "Qwen3-extra",
            "host": "a",
            "port": 8014,
            "state": "GEN",
            "tok_s": 20,
        }])),
        (SHARED_PORT, _live(loaded=[{
            "name": "Eval…",
            "served": "eval-candidate",
            "host": "b",
            "port": 8020,
            "state": "BUSY",
        }])),
        ([
            {"id": "plain", "title": "Plain", "port": 8015, "seats": {"head": {"port": 8015}}},
            {"id": "named", "title": "Named", "port": 8015, "seats": {"head": {"port": 8015}}, "identities": {"served": ["named-id"]}},
        ], _live(loaded=[{
            "name": "Plain…",
            "served": "something-else",
            "host": "a",
            "port": 8015,
            "state": "GEN",
            "tok_s": 7,
        }])),
    ]
    payload = []
    expect = []
    for models, live in cases:
        payload.append({
            "nodes": NODES,
            "models": models,
            "live": live,
            "library": LIBRARY,
            "controller": {"name": "controller"},
            "extras": [{"name": "extra"}],
        })
        expect.append(json.loads(json.dumps(group_machines(
            NODES, models, live=live, library=LIBRARY,
            controller={"name": "controller"}, extras=[{"name": "extra"}],
        ))))
    js = script + "\nvar cases = " + json.dumps(payload) + ";\n"
    js += "var out = cases.map(function (c) { return groupMachines(c.nodes, c.models, c.live, c.library, c.controller, c.extras); });\n"
    js += "process.stdout.write(JSON.stringify(out));\n"
    proc = subprocess.run(["node"], input=js, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == expect


def test_pair_sits_on_the_sparks_it_names():
    nodes = [
        {"role": "head", "id": "a", "name": "spark-a"},
        {"role": "worker", "id": "b", "name": "spark-b"},
        {"role": "node3", "id": "c", "name": "spark-c"},
    ]
    models = [
        {"id": "p23", "title": "P", "port": 8020, "placement": "pair",
         "nodes": [{"role": "worker", "host": "spark-b"},
                   {"role": "node3", "host": "spark-c"}]},
        {"id": "old", "title": "Old", "port": 8014, "placement": "pair"},
    ]
    live = {"hosts": [{"id": x, "fresh": True, "age": 1} for x in "abc"],
            "loaded": []}
    py = group_machines(nodes, models, live=live, library=[], controller=None, extras=[])

    def on(out):
        return {m["role"]: sorted(x["id"] for x in m["models"]) for m in out["machines"]}

    assert on(py) == {"head": ["old"], "worker": ["old", "p23"], "node3": ["p23"]}
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* group-machines-start */", 1)[1].split("/* group-machines-end */", 1)[0]
    js = script + "\nprocess.stdout.write(JSON.stringify(groupMachines(%s, %s, %s, [], null, [])));\n" % (
        json.dumps(nodes), json.dumps(models), json.dumps(live))
    proc = subprocess.run(["node"], input=js, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == json.loads(json.dumps(py))
