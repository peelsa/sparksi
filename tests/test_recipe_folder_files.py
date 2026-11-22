# SPDX-License-Identifier: Apache-2.0

"""A recipe folder's files: the manifest, the version hash, and the paths each Spark mounts."""

from __future__ import annotations

import copy
import os

import pytest

from sparksi.config import CFG, Config
from sparksi.recipes import folder as rf

MODEL = "pair-fast"
NAME = "big"


def _raw(n: int, **extra):
    nodes = [{"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}"} for i in range(1, n + 1)]
    for index, values in extra.items():
        nodes[int(index[1:]) - 1].update(values)
    return {"nodes": nodes, "paths": {"home": "/home/USER"}}


@pytest.fixture()
def site():
    saved = copy.copy(CFG.__dict__)

    def use(raw):
        CFG.__init__(raw)
        return CFG

    yield use
    CFG.__dict__.clear()
    CFG.__dict__.update(saved)


def _write(root, rel, data=b"x\n", mode=0o644):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, mode)
    return path


def _pair(mounts_head, mounts_worker=None):
    return {
        "kind": "pinned",
        "placement": "both",
        "roles": {
            "head": {"runtime": "docker", "image": "example/img:1", "mounts": list(mounts_head)},
            "worker": {"runtime": "docker", "image": "example/img:1", "mounts": list(mounts_worker or mounts_head)},
        },
    }


def test_scan_lists_files_sorted_with_hashes(tmp_path):
    _write(tmp_path, "files/b.py", b"b\n")
    _write(tmp_path, "files/sub/a.json", b"{}\n")
    _write(tmp_path, "roles/head/start.sh", b"#!/bin/sh\n", 0o755)
    _write(tmp_path, "recipe.toml", b'kind = "pinned"\n')  # settings are not copied
    found = rf.scan(tmp_path)
    assert found.problems == []
    assert [e.path for e in found.entries] == ["files/b.py", "files/sub/a.json", "roles/head/start.sh"]
    assert found.entries[0].sha256 == __import__("hashlib").sha256(b"b\n").hexdigest()
    assert [e.executable for e in found.entries] == [False, False, True]
    text = rf.manifest_text(found.entries)
    assert text.splitlines()[0].endswith("  files/b.py")


def test_version_changes_with_content_and_exec_bit(tmp_path):
    path = _write(tmp_path, "files/a.py", b"one\n")
    first = rf.scan(tmp_path).version
    assert len(first) == rf.VERSION_CHARS
    assert rf.scan(tmp_path).version == first
    path.write_bytes(b"two\n")
    second = rf.scan(tmp_path).version
    assert second != first
    os.chmod(path, 0o755)
    assert rf.scan(tmp_path).version not in (first, second)


def test_links_and_unknown_roles_are_problems(tmp_path):
    _write(tmp_path, "files/real.py")
    (tmp_path / "files" / "link.py").symlink_to(tmp_path / "files" / "real.py")
    _write(tmp_path, "roles/bogus/start.sh")
    _write(tmp_path, "roles/loose.sh")
    problems = rf.scan(tmp_path).problems
    assert any("files/link.py is a link" in p for p in problems)
    assert any("roles/bogus/start.sh" in p for p in problems)
    assert any("roles/loose.sh" in p for p in problems)


def test_resolve_points_each_role_at_its_own_copy(tmp_path, site):
    site(_raw(2))
    _write(tmp_path, "files/patch.py")
    _write(tmp_path, "files/extras/map.json")
    _write(tmp_path, "roles/head/start.sh")
    _write(tmp_path, "roles/worker/start.sh")
    recipe = _pair([
        "@files/patch.py:/opt/app/patch.py:ro",
        "@files/extras:/opt/app/extras:ro",
        "@role/start.sh:/start.sh:ro",
        "/home/USER/.cache/huggingface:/root/.cache/huggingface",
    ])
    before = copy.deepcopy(recipe)
    out = rf.resolve(recipe, MODEL, NAME, tmp_path)
    assert out.conflicts == []
    assert recipe == before  # the saved recipe is never changed
    base = f"/home/USER/.local/share/sparksi/recipes/{MODEL}/{NAME}/{out.version}"
    assert out.nodes == {"head": base, "worker": base}
    assert out.recipe["roles"]["head"]["mounts"] == [
        f"{base}/files/patch.py:/opt/app/patch.py:ro",
        f"{base}/files/extras:/opt/app/extras:ro",
        f"{base}/roles/head/start.sh:/start.sh:ro",
        "/home/USER/.cache/huggingface:/root/.cache/huggingface",
    ]
    assert out.recipe["roles"]["worker"]["mounts"][2] == f"{base}/roles/worker/start.sh:/start.sh:ro"
    assert rf.local_path(f"{base}/roles/head/start.sh", out.local) == str(tmp_path / "roles/head/start.sh")
    assert rf.local_path("/elsewhere/start.sh", out.local) is None


def test_missing_and_escaping_files_are_refused(tmp_path, site):
    site(_raw(2))
    _write(tmp_path, "files/patch.py")
    out = rf.resolve(_pair(["@files/gone.py:/opt/gone.py", "@files/../etc/passwd:/x", "@role/start.sh:/start.sh"]), MODEL, NAME, tmp_path)
    assert "recipe file files/gone.py is missing from the recipe folder" in out.conflicts
    assert "bad recipe file path @files/../etc/passwd" in out.conflicts
    assert "recipe file roles/head/start.sh is missing from the recipe folder" in out.conflicts
    assert "recipe file roles/worker/start.sh is missing from the recipe folder" in out.conflicts


def test_folder_paths_need_a_folder_recipe(site):
    site(_raw(2))
    out = rf.resolve(_pair(["@files/patch.py:/opt/patch.py"]), MODEL, NAME, None)
    assert out.conflicts == ["this recipe names @files/ or @role/ files but is not in a recipe folder"]


def test_a_recipe_without_folder_files_is_unchanged(tmp_path, site):
    site(_raw(2))
    recipe = _pair(["/home/USER/.cache/huggingface:/root/.cache/huggingface"])
    out = rf.resolve(recipe, MODEL, NAME, None)
    assert out.conflicts == [] and out.version == "" and out.recipe == recipe


def test_native_binary_from_the_folder(tmp_path, site):
    site(_raw(1))
    _write(tmp_path, "roles/head/run.sh", b"#!/bin/sh\n", 0o755)
    recipe = {"kind": "pinned", "placement": "head", "roles": {"head": {"runtime": "native", "binary": "@role/run.sh", "args": []}}}
    out = rf.resolve(recipe, MODEL, NAME, tmp_path)
    assert out.conflicts == []
    assert out.recipe["roles"]["head"]["binary"].endswith(f"/{out.version}/roles/head/run.sh")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_spark_gets_a_root(site, n):
    cfg = site(_raw(n))
    roles = ("head", "worker", "node3", "node4")[:n]
    for role in roles:
        assert cfg.recipe_root(role) == "/home/USER/.local/share/sparksi/recipes"
    assert "/home/USER/.local/share/sparksi/recipes" in cfg.mount_roots()


def test_recipe_root_can_be_set_per_spark(site):
    cfg = site(_raw(2, s2={"recipe_root": "/srv/example/recipes"}))
    assert cfg.recipe_root("head") == "/home/USER/.local/share/sparksi/recipes"
    assert cfg.recipe_root("worker") == "/srv/example/recipes"
    assert "/srv/example/recipes" in cfg.mount_roots()


@pytest.mark.parametrize("value", ["/", "relative/path", "~", "/home", "/srv/../etc"])
def test_unusable_recipe_roots_are_empty(site, value):
    cfg = site(_raw(1, s1={"recipe_root": value}))
    assert cfg.recipe_root("head") == ""
    assert "/" not in cfg.mount_roots()


def test_no_root_is_a_conflict(tmp_path, site):
    site(_raw(2, s2={"recipe_root": "/"}))
    _write(tmp_path, "files/patch.py")
    out = rf.resolve(_pair(["@files/patch.py:/opt/patch.py"]), MODEL, NAME, tmp_path)
    assert "the worker Spark has no usable recipe_root in sparksi.toml" in out.conflicts
    assert out.recipe["roles"]["worker"]["mounts"] == ["@files/patch.py:/opt/patch.py"]


def test_config_accepts_the_key():
    assert Config(_raw(1, s1={"recipe_root": "/srv/example/r"})).nodes[0].recipe_root == "/srv/example/r"
