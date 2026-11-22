# SPDX-License-Identifier: Apache-2.0

"""Node agent: copy a recipe folder once, check it by hash, never rewrite a copy."""

from __future__ import annotations

import base64
import hashlib
import os
import stat

import pytest

from sparksi import agent
from sparksi.agent import main as agent_main
from sparksi.config import CFG
from sparksi.remote import _RECIPE_OPS


@pytest.fixture()
def roots(monkeypatch, tmp_path):
    recipe_root = tmp_path / "recipes"
    source = tmp_path / "source"
    recipe_root.mkdir()
    source.mkdir()
    monkeypatch.setitem(agent._CFG, "recipe_roots", [str(recipe_root)])
    monkeypatch.setitem(agent._CFG, "mount_roots", [str(source)])
    yield recipe_root, source
    for current, dirs, _files in os.walk(tmp_path):
        os.chmod(current, 0o755)


FILES = {"files/patch.py": (b"x = 1\n", False), "roles/head/start.sh": (b"#!/bin/sh\necho hi\n", True)}


def _req(op, target, files=FILES, data=True):
    manifest = [{"path": rel, "sha256": hashlib.sha256(blob).hexdigest(), "executable": ex} for rel, (blob, ex) in files.items()]
    req = {"op": op, "dir": str(target), "manifest": manifest}
    if data:
        req["files"] = [{"path": rel, "data": base64.b64encode(blob).decode()} for rel, (blob, _ex) in files.items()]
    return req


def test_stage_writes_a_read_only_copy_once(roots):
    root, _src = roots
    target = root / "pair-fast" / "big" / "abc123"
    out = agent_main(_req("files_stage", target))
    assert out["ok"] and out["state"] == "match" and out["copied"] is True, out
    assert (target / "files" / "patch.py").read_bytes() == b"x = 1\n"
    assert stat.S_IMODE((target / "files" / "patch.py").stat().st_mode) == 0o444
    assert stat.S_IMODE((target / "roles" / "head" / "start.sh").stat().st_mode) == 0o555
    assert stat.S_IMODE(target.stat().st_mode) == 0o555
    before = (target / "files" / "patch.py").stat().st_ino
    again = agent_main(_req("files_stage", target))
    assert again["ok"] and again["state"] == "match" and again["copied"] is False
    assert (target / "files" / "patch.py").stat().st_ino == before
    assert not list(target.parent.glob(".incoming-*"))


def test_an_existing_copy_that_differs_is_reported_not_rewritten(roots):
    root, _src = roots
    target = root / "m" / "r" / "v1"
    agent_main(_req("files_stage", target))
    os.chmod(target / "files", 0o755)
    os.chmod(target / "files" / "patch.py", 0o644)
    (target / "files" / "patch.py").write_bytes(b"edited by hand\n")
    (target / "files" / "extra.py").write_bytes(b"\n")
    out = agent_main(_req("files_stage", target))
    assert out["ok"] and out["state"] == "differs" and out["copied"] is False
    assert out["differs"] == ["files/patch.py"] and out["extra"] == ["files/extra.py"]
    assert (target / "files" / "patch.py").read_bytes() == b"edited by hand\n"
    check = agent_main(_req("files_check", target, data=False))
    assert check["state"] == "differs" and check["differs"] == ["files/patch.py"]


def test_check_reports_a_missing_copy(roots):
    root, _src = roots
    out = agent_main(_req("files_check", root / "m" / "r" / "nope", data=False))
    assert out["ok"] and out["state"] == "missing" and out["missing"] == sorted(FILES)


def test_bad_bytes_leave_nothing_behind(roots):
    root, _src = roots
    target = root / "m" / "r" / "v2"
    req = _req("files_stage", target)
    req["files"][0]["data"] = base64.b64encode(b"tampered\n").decode()
    out = agent_main(req)
    assert out["ok"] is False and "did not arrive intact" in out["error"]
    assert not target.exists()
    assert not list((root / "m" / "r").glob(".incoming-*")) if (root / "m" / "r").exists() else True


@pytest.mark.parametrize("where", ["/etc/sparksi", "{root}/../escape", "{root}"])
def test_copies_stay_inside_the_recipe_roots(roots, where):
    root, _src = roots
    target = where.format(root=root)
    for op in ("files_stage", "files_check"):
        out = agent_main(_req(op, target))
        assert out["ok"] is False and "outside the recipe roots" in out["error"]


@pytest.mark.parametrize("rel", ["../x", "files/../../x", "/abs", "other/x", "files", "files//x"])
def test_bad_relative_paths_are_refused(roots, rel):
    root, _src = roots
    out = agent_main(_req("files_stage", root / "m" / "r" / "v", files={rel: (b"x", False)}))
    assert out["ok"] is False and "manifest" in out["error"]


def test_the_file_list_must_match_the_manifest(roots):
    root, _src = roots
    req = _req("files_stage", root / "m" / "r" / "v")
    req["files"] = req["files"][:1]
    out = agent_main(req)
    assert out["ok"] is False and "does not match its manifest" in out["error"]


def test_no_recipe_roots_refuses_every_copy(roots, monkeypatch):
    root, _src = roots
    monkeypatch.setitem(agent._CFG, "recipe_roots", [])
    out = agent_main(_req("files_stage", root / "m" / "r" / "v"))
    assert out["ok"] is False


def test_read_returns_files_and_reports_links(roots):
    _root, src = roots
    (src / "overlay").mkdir()
    (src / "overlay" / "a.py").write_bytes(b"a\n")
    (src / "overlay" / "sub").mkdir()
    (src / "overlay" / "sub" / "b.bin").write_bytes(b"\x00\x01")
    (src / "overlay" / "link.py").symlink_to(src / "overlay" / "a.py")
    (src / "start.sh").write_bytes(b"#!/bin/sh\n")
    os.chmod(src / "start.sh", 0o755)
    out = agent_main({"op": "files_read", "paths": [str(src / "overlay"), str(src / "start.sh"), str(src / "gone")]})
    assert out["ok"], out
    files = out["files"]
    assert base64.b64decode(files[str(src / "overlay" / "sub" / "b.bin")]["data"]) == b"\x00\x01"
    assert files[str(src / "start.sh")]["executable"] is True
    assert files[str(src / "gone")] is None
    assert out["links"] == [str(src / "overlay" / "link.py")]


def test_read_and_hash_stay_inside_the_mount_roots(roots):
    _root, src = roots
    assert agent_main({"op": "files_read", "paths": ["/etc/hostname"]})["ok"] is False
    assert agent_main({"op": "file_sha256", "path": "/etc/hostname"})["ok"] is False
    (src / "llama-server").write_bytes(b"binary")
    out = agent_main({"op": "file_sha256", "path": str(src / "llama-server")})
    assert out == {"ok": True, "exists": True, "sha256": hashlib.sha256(b"binary").hexdigest()}
    assert agent_main({"op": "file_sha256", "path": str(src / "nope")}) == {"ok": True, "exists": False, "sha256": ""}


def test_copy_ops_never_fall_back_to_the_controller():
    assert {"files_check", "files_stage", "files_read", "file_sha256"} <= _RECIPE_OPS


def test_the_controller_sends_each_sparks_recipe_root():
    roots = CFG.agent_config()["recipe_roots"]
    assert roots and all(root.endswith("/.local/share/sparksi/recipes") for root in roots)


def test_a_linked_folder_cannot_carry_a_copy_outside(roots, tmp_path):
    root, _src = roots
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "m").symlink_to(outside)
    out = agent_main(_req("files_stage", root / "m" / "r" / "v"))
    assert out["ok"] is False and "outside the recipe roots" in out["error"]
    assert not any(outside.rglob("*"))


def test_the_recipe_root_is_made_on_first_use(roots, monkeypatch, tmp_path):
    fresh = tmp_path / "home" / ".local" / "share" / "sparksi" / "recipes"
    monkeypatch.setitem(agent._CFG, "recipe_roots", [str(fresh)])
    out = agent_main(_req("files_stage", fresh / "m" / "r" / "v"))
    assert out["ok"] and out["copied"] is True, out
