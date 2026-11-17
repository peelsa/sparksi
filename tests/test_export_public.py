# SPDX-License-Identifier: Apache-2.0

"""Tests for scripts/export_public.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "export_public", ROOT / "scripts" / "export_public.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_public"] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path



def test_branch_entry_styles():
    """Flow, scalar, and block styles parse; quotes strip; trailing comment ignored."""
    flow = 'on:\n  push:\n    branches: [main, "release-x", dev]\n'
    assert module._branch_entries(flow) == ["main", "release-x", "dev"]

    scalar = "on:\n  push:\n    branches: release-x\n"
    assert module._branch_entries(scalar) == ["release-x"]

    block = "on:\n  push:\n    branches:\n      - main\n      - feature/**\n"
    assert module._branch_entries(block) == ["main", "feature/**"]

    commented = "on:\n  push:\n    branches: # internal\n      - 'dev'\n"
    assert module._branch_entries(commented) == ["dev"]


def test_flow_lines_rewritten(tmp_path):
    """Both flow-style lines in ci.yml collapse to main; indentation kept; pass."""
    text = (
        "name: ci\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [main, release-x, \"feature/**\"]\n"
        "  push:\n"
        "    branches: [main, release-x, \"feature/**\"]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
    )
    _write(tmp_path / ".github" / "workflows" / "ci.yml", text)
    result = module.public_workflow_branches(tmp_path)
    assert result.status == "pass"
    out = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    expected = (
        "name: ci\n"
        "on:\n"
        "  pull_request:\n"
        "    branches: [main]\n"
        "  push:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
    )
    assert out == expected


def test_block_style_fails(tmp_path):
    """Block-style non-main entry fails; ci.yml unchanged; detail names the file."""
    text = (
        "name: ci\n"
        "on:\n"
        "  push:\n"
        "    branches:\n"
        "      - main\n"
        "      - release-x\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
    )
    path = _write(tmp_path / ".github" / "workflows" / "ci.yml", text)
    result = module.public_workflow_branches(tmp_path)
    assert result.status == "fail"
    assert ".github/workflows/ci.yml: release-x" in result.detail
    assert "release-x" in result.detail
    assert path.read_text(encoding="utf-8") == text


def test_other_workflow_not_rewritten(tmp_path):
    """A second workflow with extra branches fails and is left untouched."""
    ci = (
        "name: ci\n"
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
    )
    _write(tmp_path / ".github" / "workflows" / "ci.yml", ci)
    scorecard = (
        "name: scorecard\n"
        "on:\n"
        "  push:\n"
        "    branches: [main, dev]\n"
        "jobs:\n"
        "  check:\n"
        "    runs-on: ubuntu-latest\n"
    )
    path = _write(tmp_path / ".github" / "workflows" / "scorecard.yml", scorecard)
    result = module.public_workflow_branches(tmp_path)
    assert result.status == "fail"
    assert result.detail == ".github/workflows/scorecard.yml: dev"
    assert path.read_text(encoding="utf-8") == scorecard


def test_repo_ci_passes_after_transform(tmp_path):
    """The repo's own ci.yml passes after transform; the repo copy is not modified."""
    repo_ci = ROOT / ".github" / "workflows" / "ci.yml"
    original = repo_ci.read_text(encoding="utf-8")
    _write(tmp_path / ".github" / "workflows" / "ci.yml", original)
    result = module.public_workflow_branches(tmp_path)
    assert result.status == "pass"
    out = (tmp_path / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    entries = module._branch_entries(out)
    assert entries and set(entries) == {"main"}
    assert repo_ci.read_text(encoding="utf-8") == original


def test_public_layer_moves_and_collides(tmp_path):
    """Nested files move out of public/, the dir is removed, collisions raise."""
    _write(tmp_path / "public" / "docs" / "guide.md", "guide\n")
    _write(tmp_path / "public" / "README", "readme\n")
    moved = module.public_layer(tmp_path)
    assert moved == sorted(["docs/guide.md", "README"])
    assert (tmp_path / "docs" / "guide.md").read_text(encoding="utf-8") == "guide\n"
    assert (tmp_path / "README").read_text(encoding="utf-8") == "readme\n"
    assert not (tmp_path / "public").exists()

    _write(tmp_path / "public" / "README", "other\n")
    with pytest.raises(module.ExportError):
        module.public_layer(tmp_path)


def test_no_public_layer_file_duplicates_a_source_file() -> None:
    """The exporter refuses a public/ file that also exists in the source tree, so keep one copy."""
    root = Path(__file__).resolve().parents[1]
    layer = root / "public"
    dupes = sorted(
        path.relative_to(layer).as_posix()
        for path in layer.rglob("*")
        if path.is_file() and (root / path.relative_to(layer)).exists()
    )
    assert dupes == []


def test_the_only_tracked_planning_file_is_the_shared_lessons_file() -> None:
    """The export keeps planning/ because git archive only sees tracked files; keep it to the lessons file."""
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(["git", "-C", str(root), "ls-files", "planning"], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["planning/pr-lessons.md"]
