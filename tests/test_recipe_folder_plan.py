# SPDX-License-Identifier: Apache-2.0

"""A plan or save of a folder recipe checks the Spark paths its @files/ and @role/ sources become."""

from __future__ import annotations

import pytest

from sparksi.config import CFG
from sparksi.recipes import named
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.recipes.service import plan_model
from sparksi.recipes.store import RECIPE_FILE, folder_path, save_recipe, set_default_name, using_recipe

MODEL = "demo"
NAME = "folder-one"
IMAGE = "example.invalid/vllm:1"


def _catalog() -> dict:
    host = CFG.head.name
    return {
        "library": [{
            "id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "engines": ["vllm"],
            "local": {host: "/srv/example/cache/weights"}, "port": 8014,
        }],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {},
        "never_touch": {},
    }


def _recipe(mounts: list[str], args: list[str] | None = None) -> dict:
    return {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": {"head": {
            "runtime": "docker",
            "image": IMAGE,
            "env": {},
            "mounts": ["/srv/example/cache:/root/.cache/huggingface", *mounts],
            "network": "host",
            "gpus": "all",
            "container": "sparksi-example",
            "args": args or ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"],
            "devices": [],
        }},
    }


@pytest.fixture()
def recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()

    def make(recipe: dict, files: dict[str, str]) -> None:
        root = folder_path(MODEL, NAME)
        root.mkdir(parents=True)
        (root / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        set_default_name(MODEL, NAME)
        with using_recipe(MODEL, NAME):
            save_recipe(MODEL, recipe)

    yield make
    reset_runs()


class _Facts:
    """Answers read-only probes: every asked path exists and every image is present. Runs nothing."""

    def invoke(self, host, req, timeout=20):
        if req.get("op") != "probe":
            raise AssertionError(f"nothing may run: {req.get('op')}")
        return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}

    def capture(self, host, argv, timeout=8):
        if list(argv[:3]) == ["docker", "image", "inspect"]:
            return {"ok": True, "code": 0, "stdout": "[]", "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "not here"}

    def run_argv(self, host, argv, timeout=3600):
        raise AssertionError("nothing may run")

    def agent_call(self, host, op, payload=None, timeout=60):
        raise AssertionError("nothing may run")


def _node_base(version: str) -> str:
    return f"{CFG.recipe_root('head')}/{MODEL}/{NAME}/{version}"


def test_plan_mounts_the_copied_folder_on_the_spark(recipes):
    recipes(_recipe(["@files/patch.py:/opt/app/patch.py:ro"]), {"files/patch.py": "x = 1\n"})
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {})
    files = planned["files"]
    assert files and files["count"] == 1 and files["manifest"][0]["path"] == "files/patch.py"
    base = _node_base(files["version"])
    assert files["nodes"] == {"head": base}
    argv = planned["draft"]["nodes"][0]["argv"]
    assert f"{base}/files/patch.py:/opt/app/patch.py:ro" in argv
    assert not any(str(item).startswith("@files/") for item in argv)
    assert not any("outside model roots" in item for item in planned["conflicts"]), planned["conflicts"]
    assert code == 200, planned["conflicts"]
    assert planned["draft"]["files"]["version"] == files["version"]


def test_a_missing_folder_file_refuses_the_plan(recipes):
    recipes(_recipe(["@files/gone.py:/opt/app/gone.py:ro"]), {"files/patch.py": "x = 1\n"})
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {})
    assert code == 400
    assert "recipe file files/gone.py is missing from the recipe folder" in planned["conflicts"]


def test_a_mounted_start_script_is_read_from_the_folder(recipes):
    script = "#!/bin/sh\nvllm serve /models/example --host 0.0.0.0 --port 8014\n"
    recipes(_recipe(["@role/start.sh:/start.sh:ro"], args=["bash", "/start.sh"]), {"roles/head/start.sh": script})
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {})
    base = _node_base(planned["files"]["version"])
    assert code == 400
    assert f"refusing bind 0.0.0.0 in {base}/roles/head/start.sh:2" in planned["conflicts"], planned["conflicts"]
    assert not any(item.startswith("could not read mounted script") for item in planned["conflicts"])


def test_an_older_recipe_is_planned_as_before(recipes, monkeypatch):
    save_recipe(MODEL, _recipe([]))
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {})
    assert planned["files"] is None
    assert "files" not in (planned["draft"] or {})
    assert code == 200, planned["conflicts"]


def test_save_checks_folder_files_but_keeps_the_written_sources(recipes):
    recipes(_recipe(["@files/patch.py:/opt/app/patch.py:ro"]), {"files/patch.py": "x = 1\n"})
    code, out = named.save_as(MODEL, {"name": "copy-two", "recipe_name": NAME, "port": 8015})
    assert code == 200, out
    text = (folder_path(MODEL, "copy-two") / RECIPE_FILE).read_text(encoding="utf-8")
    assert "@files/patch.py:/opt/app/patch.py:ro" in text
    code, out = named.save_as(MODEL, {
        "name": "copy-three", "recipe_name": NAME,
        "roles": {"head": {**_recipe(["@files/nope.py:/opt/nope.py"])["roles"]["head"]}},
    })
    assert code == 400
    assert "recipe file files/nope.py is missing from the recipe folder" in out["conflicts"]
    assert not folder_path(MODEL, "copy-three").exists()
