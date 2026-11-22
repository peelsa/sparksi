# SPDX-License-Identifier: Apache-2.0

"""Bundle an older pinned recipe into a recipe folder, and move old backups aside."""

from __future__ import annotations

import json

import pytest

from sparksi import agent
from sparksi.config import CFG
from sparksi.recipes.bundle import plan_bundle, tidy_backups, write_bundle
from sparksi.recipes.folder import resolve
from sparksi.recipes.store import (
    backups_dir,
    folder_path,
    is_folder_recipe,
    load_recipe,
    named_recipe_path,
    recipes_dir,
    save_recipe,
    using_recipe,
)

MODEL = "demo"
IMAGE = "example.invalid/vllm:1"
BUILD = "sha256:" + "e" * 64


class _Sparks:
    """Reads go to the real agent over a local folder; docker image inspect gives one build. Runs nothing."""

    def __init__(self):
        self.ops = []

    def invoke(self, host, req, timeout=20):
        self.ops.append(req.get("op"))
        if req.get("op") in ("files_read", "file_sha256"):
            return agent.main(req)
        raise AssertionError(f"nothing may run: {req.get('op')}")

    def capture(self, host, argv, timeout=8):
        assert list(argv[:3]) == ["docker", "image", "inspect"]
        return {"ok": True, "code": 0, "stdout": json.dumps([{"Id": BUILD}]), "stderr": ""}


@pytest.fixture()
def site(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "config" / "recipes"))
    overlay = tmp_path / "spark" / "overlay"
    (overlay / "pkg").mkdir(parents=True)
    (overlay / "patch.py").write_text("x = 1\n", encoding="utf-8")
    (overlay / "pkg" / "mod.py").write_text("y = 2\n", encoding="utf-8")
    script = overlay / "inner.sh"
    script.write_text("#!/bin/sh\nexec vllm serve /models/example\n", encoding="utf-8")
    script.chmod(0o755)
    monkeypatch.setitem(agent._CFG, "mount_roots", [str(tmp_path / "spark")])
    for node in CFG.nodes:
        monkeypatch.setattr(node, "recipe_root", str(tmp_path / "copies" / node.id), raising=False)
    return overlay


def _role(overlay, extra=()) -> dict:
    return {
        "runtime": "docker", "image": IMAGE, "env": {},
        "mounts": [
            f"{CFG.hf_cache()}:/root/.cache/huggingface",
            f"{overlay}/patch.py:/opt/app/patch.py:ro",
            f"{overlay}/pkg:/opt/app/pkg:ro",
            f"{overlay}/inner.sh:/start.sh:ro",
            *extra,
        ],
        "network": "host", "gpus": "all", "container": "sparksi-example",
        "args": ["bash", "/start.sh"], "devices": [],
    }


def _recipe(overlay, placement="head") -> dict:
    roles = {"head": _role(overlay)}
    if placement == "both":
        roles["worker"] = _role(overlay)
    return {
        "kind": "pinned", "engine": "vllm", "placement": placement, "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [], "roles": roles,
    }


def _catalog(reserve=None) -> dict:
    row = {"id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "local": {CFG.head.name: "/srv/example/cache/weights"}}
    if reserve is not None:
        row["kv_reserve_gib"] = reserve
    return {"library": [row]}


def test_a_dry_run_reads_only_and_maps_every_file(site):
    save_recipe(MODEL, _recipe(site))
    before = named_recipe_path(MODEL, "default").read_bytes()
    runner = _Sparks()
    plan = plan_bundle(_catalog(15), runner, MODEL)
    view = plan.view()
    assert view["ok"], view["problems"]
    assert [f["path"] for f in view["files"]] == ["files/opt/app/patch.py", "files/opt/app/pkg/mod.py", "roles/head/start.sh"]
    head = plan.recipe["roles"]["head"]
    assert head["mounts"][0] == f"{CFG.hf_cache()}:/root/.cache/huggingface"
    assert head["mounts"][1:] == ["@files/opt/app/patch.py:/opt/app/patch.py:ro", "@files/opt/app/pkg:/opt/app/pkg:ro", "@role/start.sh:/start.sh:ro"]
    assert head["image_id"] == BUILD
    assert plan.recipe["memory"] == {"kv_reserve_gib": 15.0}
    assert set(runner.ops) <= {"files_read", "file_sha256"}
    assert not folder_path(MODEL, "default").exists()
    assert named_recipe_path(MODEL, "default").read_bytes() == before


def test_a_write_makes_the_folder_and_moves_the_older_file(site):
    save_recipe(MODEL, _recipe(site, "both"))
    old = named_recipe_path(MODEL, "default")
    plan = plan_bundle(_catalog(), _Sparks(), MODEL)
    assert [f["path"] for f in plan.view()["files"]].count("files/opt/app/patch.py") == 1
    ok, said = write_bundle(plan, stamp="20261122")
    assert ok, said
    assert is_folder_recipe(MODEL, "default")
    assert not old.exists()
    assert (backups_dir("20261122") / "bundled" / MODEL / "default.toml").is_file()
    root = folder_path(MODEL, "default")
    assert (root / "roles" / "head" / "start.sh").stat().st_mode & 0o111
    assert (root / "roles" / "worker" / "start.sh").is_file()
    with using_recipe(MODEL, "default"):
        saved = load_recipe(MODEL)
    resolved = resolve(saved, MODEL, "default", root)
    assert not resolved.conflicts, resolved.conflicts
    assert len(resolved.version) == 16


def test_a_file_that_differs_between_sparks_stays_per_role(site, tmp_path):
    other = tmp_path / "spark" / "other"
    other.mkdir()
    (other / "patch.py").write_text("x = 2\n", encoding="utf-8")
    recipe = _recipe(site, "both")
    recipe["roles"]["worker"]["mounts"][1] = f"{other}/patch.py:/opt/app/patch.py:ro"
    save_recipe(MODEL, recipe)
    plan = plan_bundle(_catalog(), _Sparks(), MODEL)
    paths = [f["path"] for f in plan.view()["files"]]
    assert "files/opt/app/patch.py" in paths and "roles/worker/opt/app/patch.py" in paths
    assert plan.recipe["roles"]["worker"]["mounts"][1] == "@role/opt/app/patch.py:/opt/app/patch.py:ro"
    assert any("differs between Sparks" in note for note in plan.notes)


def test_a_missing_source_refuses_and_writes_nothing(site):
    save_recipe(MODEL, _recipe(site))
    recipe = load_recipe(MODEL)
    recipe["roles"]["head"]["mounts"].append(f"{site}/gone.py:/opt/app/gone.py:ro")
    save_recipe(MODEL, recipe)
    plan = plan_bundle(_catalog(), _Sparks(), MODEL)
    assert plan.problems and "gone.py" in plan.problems[0]
    ok, said = write_bundle(plan)
    assert not ok and not folder_path(MODEL, "default").exists()


def test_a_folder_recipe_is_not_bundled_again(site):
    save_recipe(MODEL, _recipe(site))
    assert write_bundle(plan_bundle(_catalog(), _Sparks(), MODEL))[0]
    again = plan_bundle(_catalog(), _Sparks(), MODEL)
    assert again.problems == [f"{MODEL} recipe default is already in a recipe folder"]


def test_tidy_backups_only_moves(site):
    config = recipes_dir().parent
    (config / "recipes.bak-1").mkdir(parents=True)
    (config / "recipes.bak-1" / "a.toml").write_text("x\n", encoding="utf-8")
    (config / "sparksi.toml.bak-2").write_text("y\n", encoding="utf-8")
    recipes_dir().mkdir(parents=True, exist_ok=True)
    (recipes_dir() / f"{MODEL}.toml.bak-3").write_text("z\n", encoding="utf-8")
    (config / "sparksi.toml").write_text("keep\n", encoding="utf-8")
    dry = tidy_backups(stamp="20261122", dry_run=True)
    assert len(dry) == 3 and all(row["done"] == "no" for row in dry)
    assert (config / "sparksi.toml.bak-2").exists()
    done = tidy_backups(stamp="20261122", dry_run=False)
    assert all(row["done"] == "yes" for row in done)
    dest = backups_dir("20261122")
    assert (dest / "recipes.bak-1" / "a.toml").read_text(encoding="utf-8") == "x\n"
    assert (dest / "sparksi.toml.bak-2").is_file()
    assert (dest / "recipes" / f"{MODEL}.toml.bak-3").is_file()
    assert (config / "sparksi.toml").read_text(encoding="utf-8") == "keep\n"
    (config / "sparksi.toml.bak-2").write_text("again\n", encoding="utf-8")
    later = tidy_backups(stamp="20261122", dry_run=False)
    assert later == [{"from": str(config / "sparksi.toml.bak-2"), "to": str(dest / "sparksi.toml.bak-2"), "done": "no", "note": "already there; left in place"}]
    assert (config / "sparksi.toml.bak-2").exists()


def test_the_cli_dry_run_writes_nothing(site, monkeypatch, capsys):
    from sparksi import cli, remote

    save_recipe(MODEL, _recipe(site))
    monkeypatch.setattr(cli, "load_catalog", _catalog)
    monkeypatch.setattr(remote, "get_runner", lambda catalog: _Sparks())
    assert cli.main(["recipes", "bundle", MODEL, "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "roles/head/start.sh" in out and "dry run: nothing was written" in out
    assert not folder_path(MODEL, "default").exists()
    assert cli.main(["recipes", "tidy-backups", "--dry-run"]) == 0
