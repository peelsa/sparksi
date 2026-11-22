# SPDX-License-Identifier: Apache-2.0

"""A second Load while a folder recipe is still being copied is refused, not started twice."""

from __future__ import annotations

import os
import threading

import pytest

from sparksi import agent
from sparksi.agent import main as agent_main
from sparksi.config import CFG
from sparksi.recipes import service
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.recipes.service import BEGIN_BUSY, begin_run
from sparksi.recipes.store import RECIPE_FILE, folder_path, save_recipe, set_default_name, using_recipe

MODEL = "demo"
NAME = "folder-one"


def _catalog() -> dict:
    host = CFG.head.name
    return {
        "library": [{"id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "engines": ["vllm"],
                     "local": {host: "/srv/example/cache/weights"}, "port": 8014}],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {},
        "never_touch": {},
    }


class _Sparks:
    """Probes say 'present'. A copy waits on ``gate`` so a second click lands mid-copy."""

    def __init__(self):
        self.gate = threading.Event()
        self.copying = threading.Event()
        self.gate.set()

    def invoke(self, host, req, timeout=20):
        op = req.get("op")
        if op == "probe":
            return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}
        if op == "files_stage":
            self.copying.set()
            assert self.gate.wait(10)
            return agent_main(dict(req))
        if op == "files_check":
            return agent_main(dict(req))
        raise AssertionError(f"nothing may run: {op}")

    def capture(self, host, argv, timeout=8):
        if list(argv[:3]) == ["docker", "image", "inspect"]:
            return {"ok": True, "code": 0, "stdout": "[]", "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "not here"}

    def run_argv(self, host, argv, timeout=3600):
        raise AssertionError("nothing may run")

    def agent_call(self, host, op, payload=None, timeout=60):
        raise AssertionError("nothing may run")


@pytest.fixture()
def site(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    roots = []
    for node in CFG.nodes:
        root = tmp_path / "sparks" / node.id
        roots.append(str(root))
        monkeypatch.setattr(node, "recipe_root", str(root))
    monkeypatch.setitem(agent._CFG, "recipe_roots", roots)
    started: list[dict] = []
    monkeypatch.setattr(service, "start_rendered", lambda recipe, runner, **kw: started.append(recipe) or {"id": "r1", "nodes": []})
    reset_runs()
    root = folder_path(MODEL, NAME)
    (root / "files").mkdir(parents=True)
    (root / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")
    (root / "files" / "patch.py").write_text("x = 1\n", encoding="utf-8")
    set_default_name(MODEL, NAME)
    with using_recipe(MODEL, NAME):
        save_recipe(MODEL, {
            "kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014,
            "imported_from": "container:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
            "roles": {"head": {
                "runtime": "docker", "image": "example.invalid/vllm:1", "env": {},
                "mounts": ["/srv/example/cache:/root/.cache/huggingface", "@files/patch.py:/opt/app/patch.py:ro"],
                "network": "host", "gpus": "all", "container": "sparksi-example",
                "args": ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"], "devices": [],
            }},
        })
    yield started
    for current, _dirs, _files in os.walk(tmp_path):
        os.chmod(current, 0o755)
    reset_runs()


def test_a_second_click_during_the_copy_is_refused(site):
    started = site
    sparks = _Sparks()
    sparks.gate.clear()
    first: dict = {}
    worker = threading.Thread(target=lambda: first.setdefault("out", begin_run(_catalog(), sparks, {"id": MODEL})))
    worker.start()
    try:
        assert sparks.copying.wait(10)
        code, out = begin_run(_catalog(), sparks, {"id": MODEL})
    finally:
        sparks.gate.set()
        worker.join(10)
    assert code == 409 and out["conflicts"] == [BEGIN_BUSY]
    assert first["out"][0] == 200, first["out"]
    assert len(started) == 1


def test_the_next_load_after_a_refused_copy_is_allowed(site):
    started = site
    sparks = _Sparks()
    real = sparks.invoke
    sparks.invoke = lambda host, req, timeout=20: {"ok": False, "error": "disk full"} if req.get("op") == "files_stage" else real(host, req, timeout)
    code, _out = begin_run(_catalog(), sparks, {"id": MODEL})
    assert code == 409 and started == []
    sparks.invoke = real
    code, out = begin_run(_catalog(), sparks, {"id": MODEL})
    assert code == 200, out
    assert len(started) == 1
