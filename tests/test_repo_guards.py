# SPDX-License-Identifier: Apache-2.0

"""Repo-level guards: site identity, stdlib-only node scripts, config defaults."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_identity_guard_is_clean() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_identifiers.py"), str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr


def test_node_scripts_are_stdlib_only() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_stdlib.py"), str(ROOT)],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "stdlib guard: clean" in out.stdout


def test_example_config_matches_defaults() -> None:
    from sparksi.config import CFG, load

    ex = load(ROOT / "config" / "sparksi.example.toml")
    assert ex.node_ids == CFG.node_ids
    assert ex.node_names == CFG.node_names
    assert ex.archive_node == CFG.archive_node
    assert ex.public() == CFG.public()


def test_packaged_examples_match_config() -> None:
    """The wheel ships src/sparksi/examples/; keep it byte-identical to config/."""
    for name in ("catalog.example.yaml", "sparksi.example.toml"):
        packaged = ROOT / "src" / "sparksi" / "examples" / name
        assert packaged.read_bytes() == (ROOT / "config" / name).read_bytes(), (
            f"copy config/{name} to src/sparksi/examples/{name}"
        )


def test_catalog_falls_back_to_packaged_example() -> None:
    from sparksi import catalog

    assert catalog.PACKAGED_EXAMPLE_CATALOG.is_file()
    assert catalog.load_catalog(catalog.PACKAGED_EXAMPLE_CATALOG)["host"]


def test_identity_guard_checks_the_shared_lessons_file_but_not_other_planning_notes(tmp_path) -> None:
    (tmp_path / "planning").mkdir()
    (tmp_path / "planning" / "notes.md").write_text("see /" + "home/someone/x\n")
    script = [sys.executable, str(ROOT / "scripts" / "check_identifiers.py"), str(tmp_path)]
    assert subprocess.run(script, capture_output=True, text=True, check=False).returncode == 0
    (tmp_path / "planning" / "pr-lessons.md").write_text("see /" + "home/someone/x\n")
    proc = subprocess.run(script, capture_output=True, text=True, check=False)
    assert proc.returncode != 0
    assert "planning/pr-lessons.md" in proc.stdout
