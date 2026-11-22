# SPDX-License-Identifier: Apache-2.0

"""Recipe folders: one folder holds a recipe's settings and the files it mounts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sparksi.recipes import named
from sparksi.recipes.store import (
    RECIPE_FILE,
    default_recipe_name,
    folder_path,
    is_folder_recipe,
    load_recipe,
    recipe_names,
    recipe_path,
    recipes_dir,
    save_recipe,
    set_default_name,
    using_recipe,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL = "pair-fast"
PLAIN = {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 16384, "gpu_mem_util": 0.55}


@pytest.fixture(autouse=True)
def _recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    yield tmp_path


def _folder(name: str, ctx: int = 32768, files: dict[str, bytes] | None = None) -> Path:
    root = folder_path(MODEL, name)
    root.mkdir(parents=True)
    (root / RECIPE_FILE).write_text(
        f'engine = "vllm"\nplacement = "head"\nport = 8012\nctx = {ctx}\ngpu_mem_util = 0.55\n', encoding="utf-8"
    )
    for rel, data in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def test_a_folder_recipe_is_listed_and_loaded():
    _folder("big", files={"files/patch.py": b"x = 1\n", "roles/head/start.sh": b"#!/bin/sh\n"})
    assert is_folder_recipe(MODEL, "big")
    assert recipe_names(MODEL) == ["big"]
    with using_recipe(MODEL, "big"):
        assert recipe_path(MODEL) == folder_path(MODEL, "big") / RECIPE_FILE
        assert load_recipe(MODEL)["ctx"] == 32768


def test_a_folder_default_is_read_without_a_selection():
    _folder("big")
    set_default_name(MODEL, "big")
    assert recipe_path(MODEL) == folder_path(MODEL, "big") / RECIPE_FILE
    assert load_recipe(MODEL)["ctx"] == 32768
    assert recipe_names(MODEL) == ["big"]


def test_a_folder_wins_over_an_older_file_of_the_same_name():
    with using_recipe(MODEL, "big"):
        save_recipe(MODEL, {**PLAIN, "ctx": 1024})
    _folder("big", ctx=65536)
    with using_recipe(MODEL, "big"):
        assert load_recipe(MODEL)["ctx"] == 65536
    assert recipe_names(MODEL).count("big") == 1


def test_older_layout_is_unchanged():
    save_recipe(MODEL, PLAIN)
    with using_recipe(MODEL, "small"):
        save_recipe(MODEL, {**PLAIN, "ctx": 4096})
    assert recipe_path(MODEL) == recipes_dir() / f"{MODEL}.toml"
    assert recipe_names(MODEL) == ["default", "small"]
    assert not is_folder_recipe(MODEL, "default")


def test_save_writes_inside_the_folder():
    root = _folder("big", files={"files/patch.py": b"x = 1\n"})
    set_default_name(MODEL, "big")
    code, out = named.save_named(MODEL, "big", {**PLAIN, "ctx": 8192})
    assert code == 200, out
    assert "ctx = 8192" in (root / RECIPE_FILE).read_text(encoding="utf-8")
    assert (root / "files" / "patch.py").read_bytes() == b"x = 1\n"
    assert not (recipes_dir() / f"{MODEL}.toml").exists()


def test_save_as_from_a_folder_copies_its_files():
    _folder("big", files={"files/patch.py": b"x = 1\n", "roles/worker/start.sh": b"#!/bin/sh\n"})
    set_default_name(MODEL, "big")
    code, out = named.save_as(MODEL, {"name": "big-copy", "recipe_name": "big", **PLAIN, "ctx": 2048})
    assert code == 200, out
    copy = folder_path(MODEL, "big-copy")
    assert (copy / "files" / "patch.py").read_bytes() == b"x = 1\n"
    assert (copy / "roles" / "worker" / "start.sh").is_file()
    with using_recipe(MODEL, "big-copy"):
        assert load_recipe(MODEL)["ctx"] == 2048
    with using_recipe(MODEL, "big"):
        assert load_recipe(MODEL)["ctx"] == 32768
    assert not list(copy.parent.glob(".*.tmp"))


def test_save_as_never_overwrites_a_folder():
    _folder("big")
    _folder("other", ctx=999)
    set_default_name(MODEL, "big")
    code, out = named.save_as(MODEL, {"name": "other", "recipe_name": "big", "overwrite": True, **PLAIN})
    assert code == 409
    with using_recipe(MODEL, "other"):
        assert load_recipe(MODEL)["ctx"] == 999


def test_a_failed_save_as_removes_only_its_own_copy():
    _folder("big", files={"files/patch.py": b"x = 1\n"})
    set_default_name(MODEL, "big")
    code, _out = named.save_as(MODEL, {"name": "bad", "recipe_name": "big", **PLAIN, "port": "nope"})
    assert code == 400
    assert not folder_path(MODEL, "bad").exists()
    assert (folder_path(MODEL, "big") / "files" / "patch.py").is_file()


def test_set_default_between_layouts_moves_only_older_files():
    save_recipe(MODEL, PLAIN)  # older default at <id>.toml
    _folder("big")
    code, out = named.set_default(MODEL, "big")
    assert code == 200, out
    assert default_recipe_name(MODEL) == "big"
    assert (recipes_dir() / "named" / MODEL / "default.toml").is_file()
    assert not (recipes_dir() / f"{MODEL}.toml").exists()
    assert is_folder_recipe(MODEL, "big")
    code, out = named.set_default(MODEL, "default")
    assert code == 200, out
    assert (recipes_dir() / f"{MODEL}.toml").is_file()
    assert is_folder_recipe(MODEL, "big")


def test_rename_moves_the_whole_folder():
    _folder("big", files={"files/patch.py": b"x = 1\n"})
    set_default_name(MODEL, "big")
    code, out = named.rename_named(MODEL, "big", {"new_name": "bigger"})
    assert code == 200, out
    assert default_recipe_name(MODEL) == "bigger"
    assert (folder_path(MODEL, "bigger") / "files" / "patch.py").is_file()
    assert not folder_path(MODEL, "big").exists()


def test_rename_never_replaces_a_folder():
    _folder("big")
    _folder("other")
    set_default_name(MODEL, "big")
    code, _out = named.rename_named(MODEL, "big", {"new_name": "other", "overwrite": True})
    assert code == 409
    assert is_folder_recipe(MODEL, "big") and is_folder_recipe(MODEL, "other")


def test_delete_moves_the_folder_into_backups():
    _folder("big")
    _folder("old", files={"files/patch.py": b"x = 1\n"})
    set_default_name(MODEL, "big")
    code, out = named.delete_named(MODEL, "old")
    assert code == 200, out
    assert not folder_path(MODEL, "old").exists()
    moved = list((recipes_dir().parent / "backups").glob(f"*/deleted/{MODEL}/old/files/patch.py"))
    assert len(moved) == 1
    assert [row["name"] for row in out["recipes"]] == ["big"]


def test_the_list_marks_folder_recipes():
    save_recipe(MODEL, PLAIN)
    _folder("big")
    code, out = named.list_named(MODEL)
    assert code == 200
    rows = {row["name"]: row["folder"] for row in out["recipes"]}
    assert rows == {"default": False, "big": True}
