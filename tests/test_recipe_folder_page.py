# SPDX-License-Identifier: Apache-2.0

"""The recipe page shows a folder recipe's version, and the catalog launch command steps aside."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparksi.config import CFG
from sparksi.recipes import named
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.recipes.service import FOLDER_STARTS, list_model_cards, plan_model, service_action
from sparksi.recipes.store import RECIPE_FILE, folder_path, save_recipe, set_default_name, using_recipe

MODEL = "demo"
NAME = "folder-one"
IMAGE = "example.invalid/vllm:1"
START = ["/srv/example/bin/start-demo"]
STATIC = Path(__file__).resolve().parents[1] / "src" / "sparksi" / "static" / "index.html"


def _catalog(kind: str = "") -> dict:
    host = CFG.head.name
    row = {
        "id": MODEL, "title": "Example", "format": "fp8", "gib": 8,
        "local": {host: "/srv/example/cache/weights"}, "port": 8014,
        "launch": {"start": START, "stop": ["/srv/example/bin/stop-demo"], "host": host},
    }
    if kind:
        row["kind"] = kind
    return {
        "library": [row],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {},
        "never_touch": {},
    }


def _recipe() -> dict:
    return {
        "kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
        "roles": {"head": {
            "runtime": "docker", "image": IMAGE, "env": {},
            "mounts": ["/srv/example/cache:/root/.cache/huggingface", "@files/patch.py:/opt/app/patch.py:ro"],
            "network": "host", "gpus": "all", "container": "sparksi-example",
            "args": ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"], "devices": [],
        }},
    }


class _Facts:
    """Read-only answers. Every path exists, every image is there, no copy exists yet. Runs nothing."""

    def invoke(self, host, req, timeout=20):
        op = req.get("op")
        if op == "probe":
            return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}
        if op == "files_check":
            return {"ok": True, "state": "missing", "missing": [], "differs": [], "extra": []}
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
def recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()

    def make() -> None:
        root = folder_path(MODEL, NAME)
        (root / "files").mkdir(parents=True)
        (root / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")
        (root / "files" / "patch.py").write_text("x = 1\n", encoding="utf-8")
        set_default_name(MODEL, NAME)
        with using_recipe(MODEL, NAME):
            save_recipe(MODEL, _recipe())

    yield make
    reset_runs()


def test_the_recipe_list_and_the_plan_show_the_same_version(recipes):
    recipes()
    code, listed = named.list_named(MODEL)
    row = next(item for item in listed["recipes"] if item["name"] == NAME)
    assert code == 200 and row["folder"] is True and len(row["version"]) == 16
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {})
    assert code == 200, planned["conflicts"]
    assert planned["files"]["version"] == row["version"]
    assert planned["files"]["sparks"]["head"]["state"] == "missing"


def test_an_older_recipe_has_no_version(recipes):
    save_recipe(MODEL, _recipe())
    code, listed = named.list_named(MODEL)
    assert all(item["version"] == "" and item["folder"] is False for item in listed["recipes"])


def test_the_catalog_launch_engine_steps_aside_for_a_folder_recipe(recipes):
    before = next(m for m in list_model_cards(_catalog(), _Facts())["models"] if m["id"] == MODEL)
    assert "custom" in before["engines"] and not before.get("folder_recipe")
    recipes()
    after = next(m for m in list_model_cards(_catalog(), _Facts())["models"] if m["id"] == MODEL)
    assert after["folder_recipe"] is True and "custom" not in after["engines"]
    code, planned = plan_model(_catalog(), _Facts(), MODEL, {"engine": "custom", "kind": "generated"})
    assert code == 400 and FOLDER_STARTS in planned["conflicts"]


def test_a_catalog_service_start_is_refused_once_a_folder_recipe_exists(recipes):
    class _Never(_Facts):
        def invoke(self, host, req, timeout=20):
            raise AssertionError("nothing may run")

    recipes()
    services = list_model_cards(_catalog("service"), _Facts())["services"]
    assert services and services[0]["folder_recipe"] is True
    code, out = service_action(_catalog("service"), _Never(), MODEL, "start")
    assert code == 409 and out["error"] == FOLDER_STARTS


def test_the_page_paints_the_folder_with_text_only():
    page = STATIC.read_text(encoding="utf-8")
    assert 'id="model-recipe-folder"' in page
    start = page.index("var FOLDER_STATES")
    body = page[start:page.index("function applyPreview", start)]
    assert "innerHTML" not in body and "textContent" in body
    assert "Load is refused" in body and "Load copies it first" in body
    assert "Starts from its recipe folder" in page
