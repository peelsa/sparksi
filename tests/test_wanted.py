# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from sparksi.catalog import load_catalog
from sparksi.cli import main
from sparksi.lock import Lock
from sparksi.wanted import resolve_spec


def test_resolve_presets(video_catalog) -> None:
    cat = load_catalog()
    r = resolve_spec(cat, "video")
    assert r["name"] == "video"
    assert r["members"] == [video_catalog]
    assert r["seats"]["video"] == video_catalog
    r = resolve_spec(cat, "trio")
    assert r["name"] == "coresident-3"
    assert "brain-a" in r["members"]


def test_resolve_seats_and_ids(video_catalog) -> None:
    cat = load_catalog()
    r = resolve_spec(cat, "brain,counsel")
    assert r["seats"]["brain"] == "brain-a"
    assert r["seats"]["counsel"] == "counsel-a"
    r = resolve_spec(cat, video_catalog)
    assert video_catalog in r["members"]


def test_wanted_set_requires_unlock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    Lock(tmp_path / "lock").acquire()
    assert main(["wanted", "set", "video"]) == 2
    err = capsys.readouterr().err
    assert "lock is ON" in err


def test_wanted_set_when_unlocked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    Lock(tmp_path / "lock").release(ttl_s=120)
    rc = main(["wanted", "set", "video", "--json"])
    assert rc in (0, 1, 3)  # dry-run may be UNCERTAIN/NO; set still stores
    data = (tmp_path / "wanted.json").read_text()
    assert "video" in data
    assert "last_dry_run" in data
