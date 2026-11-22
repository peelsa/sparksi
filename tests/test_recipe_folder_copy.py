# SPDX-License-Identifier: Apache-2.0

"""Before every start a folder recipe is copied to each Spark it uses and checked by hash."""

from __future__ import annotations

import os

import pytest

from sparksi import agent
from sparksi.agent import main as agent_main
from sparksi.config import CFG
from sparksi.recipes import folder as rf
from sparksi.recipes import service
from sparksi.recipes.runs import _save, get_run, public_run, reset_for_tests as reset_runs
from sparksi.recipes.service import begin_run, plan_model
from sparksi.recipes.store import RECIPE_FILE, folder_path, save_recipe, set_default_name, using_recipe

MODEL = "demo"
NAME = "folder-one"
IMAGE = "example.invalid/vllm:1"


def _catalog() -> dict:
    hosts = {CFG.head.name: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}}
    local = {CFG.head.name: "/srv/example/cache/weights"}
    if CFG.has_worker:
        hosts[CFG.worker.name] = {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}
        local[CFG.worker.name] = "/srv/example/cache/weights"
    return {
        "library": [{"id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "engines": ["vllm"], "local": local, "port": 8014}],
        "hosts": hosts,
        "exclusive_groups": {},
        "never_touch": {},
    }


def _role() -> dict:
    return {
        "runtime": "docker",
        "image": IMAGE,
        "env": {},
        "mounts": ["/srv/example/cache:/root/.cache/huggingface", "@files/patch.py:/opt/app/patch.py:ro", "@role/start.sh:/start.sh:ro"],
        "network": "host",
        "gpus": "all",
        "container": "sparksi-example",
        "args": ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"],
        "devices": [],
    }


def _recipe(placement: str) -> dict:
    roles = {"head": _role()}
    if placement == "both":
        roles["worker"] = _role()
    return {
        "kind": "pinned", "engine": "vllm", "placement": placement, "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
        "roles": roles,
    }


class _Sparks:
    """Probes answer 'present'; recipe folder ops run the real agent code on per-Spark folders."""

    def __init__(self, fail_host: str = ""):
        self.calls: list[tuple[str, str]] = []
        self.fail_host = fail_host

    def invoke(self, host, req, timeout=20):
        op = req.get("op")
        self.calls.append((host, op))
        if op == "probe":
            return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}
        if op in ("files_stage", "files_check"):
            if host == self.fail_host and op == "files_stage":
                return {"ok": False, "error": "disk full"}
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
    roots = {}
    for node in CFG.nodes:
        roots[node.name] = tmp_path / "sparks" / node.id
        monkeypatch.setattr(node, "recipe_root", str(roots[node.name]))
    monkeypatch.setitem(agent._CFG, "recipe_roots", [str(p) for p in roots.values()])
    started: list[dict] = []
    monkeypatch.setattr(service, "start_rendered", lambda recipe, runner, **kw: started.append(recipe) or {"id": "r1", "nodes": []})
    reset_runs()

    def make(placement: str = "head") -> None:
        root = folder_path(MODEL, NAME)
        root.mkdir(parents=True)
        (root / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")
        for rel, text in {"files/patch.py": "x = 1\n", "roles/head/start.sh": "#!/bin/sh\n", "roles/worker/start.sh": "#!/bin/sh\n# rank 1\n"}.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        set_default_name(MODEL, NAME)
        with using_recipe(MODEL, NAME):
            save_recipe(MODEL, _recipe(placement))

    yield make, roots, started
    for current, _dirs, _files in os.walk(tmp_path):
        os.chmod(current, 0o755)
    reset_runs()


def test_a_start_copies_the_folder_to_each_spark_first(site):
    make, roots, started = site
    make("both")
    sparks = _Sparks()
    code, planned = plan_model(_catalog(), sparks, MODEL, {})
    assert code == 200, planned["conflicts"]
    version = planned["files"]["version"]
    assert {row["state"] for row in planned["files"]["sparks"].values()} == {"missing"}
    code, out = begin_run(_catalog(), sparks, {"id": MODEL})
    assert code == 200, out
    assert len(started) == 1 and started[0]["files"]["version"] == version
    for node in CFG.nodes[:2]:
        copy = roots[node.name] / MODEL / NAME / version
        assert (copy / "files" / "patch.py").read_text() == "x = 1\n"
    assert (roots[CFG.worker.name] / MODEL / NAME / version / "roles" / "worker" / "start.sh").read_text().endswith("# rank 1\n")
    stage_calls = [call for call in sparks.calls if call[1] == "files_stage"]
    assert sorted(host for host, _op in stage_calls) == sorted([CFG.head.name, CFG.worker.name])
    _code, again = plan_model(_catalog(), sparks, MODEL, {})
    assert {row["state"] for row in again["files"]["sparks"].values()} == {"match"}


def test_a_copy_that_differs_refuses_the_start(site):
    make, roots, started = site
    make("head")
    sparks = _Sparks()
    code, out = begin_run(_catalog(), sparks, {"id": MODEL})
    assert code == 200 and len(started) == 1
    version = started[0]["files"]["version"]
    copy = roots[CFG.head.name] / MODEL / NAME / version / "files"
    os.chmod(copy, 0o755)
    os.chmod(copy / "patch.py", 0o644)
    (copy / "patch.py").write_text("edited on the Spark\n")
    code, planned = plan_model(_catalog(), sparks, MODEL, {})
    assert code == 400
    assert f"{CFG.head.name}'s copy of this recipe's files does not match the recipe folder" in planned["conflicts"]
    assert planned["files"]["sparks"]["head"]["differs"] == ["files/patch.py"]
    started.clear()
    why, _states = rf.copy_to_sparks(sparks, planned["draft"]["files"])
    assert why == f"{CFG.head.name}'s copy of this recipe's files does not match the recipe folder, so nothing was started."
    assert (copy / "patch.py").read_text() == "edited on the Spark\n"  # never rewritten
    assert started == []


def test_a_failed_copy_refuses_the_start(site):
    make, _roots, started = site
    make("both")
    code, out = begin_run(_catalog(), _Sparks(fail_host=CFG.worker.name), {"id": MODEL})
    assert code == 409
    assert out["conflicts"] == [f"Could not copy this recipe's files to {CFG.worker.name} (disk full), so nothing was started."]
    assert started == []


def test_an_edit_after_the_check_refuses_the_start(site):
    make, _roots, started = site
    make("head")
    sparks = _Sparks()
    _code, planned = plan_model(_catalog(), sparks, MODEL, {})
    (folder_path(MODEL, NAME) / "files" / "patch.py").write_text("x = 2\n")
    why, states = rf.copy_to_sparks(sparks, planned["files"])
    assert why == "The recipe folder changed after it was checked, so nothing was started. Load it again."
    assert states == {}
    assert not any(op == "files_stage" for _host, op in sparks.calls)


def test_an_edit_makes_a_new_copy_and_leaves_the_old_one(site):
    make, roots, started = site
    make("head")
    sparks = _Sparks()
    begin_run(_catalog(), sparks, {"id": MODEL})
    first = started[-1]["files"]["version"]
    (folder_path(MODEL, NAME) / "files" / "patch.py").write_text("x = 2\n")
    code, out = begin_run(_catalog(), sparks, {"id": MODEL})
    assert code == 200, out
    second = started[-1]["files"]["version"]
    assert second != first
    base = roots[CFG.head.name] / MODEL / NAME
    assert (base / first / "files" / "patch.py").read_text() == "x = 1\n"
    assert (base / second / "files" / "patch.py").read_text() == "x = 2\n"


def test_the_run_record_keeps_the_version_it_mounts(site):
    _save({"id": "r9", "model_id": MODEL, "phase": "ready", "nodes": [], "files": {"version": "abc", "folder": "/x", "nodes": {}}})
    assert public_run(get_run("r9"))["files_version"] == "abc"
    _save({"id": "r8", "model_id": MODEL, "phase": "ready", "nodes": []})
    assert public_run(get_run("r8"))["files_version"] == ""
