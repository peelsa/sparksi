# SPDX-License-Identifier: Apache-2.0

"""D4: once a model has a recipe folder, no catalog launch or seat start runs for it.

Covers the library run (``sparksi run`` and ``/api/library/run``) and the
library switch (pair and single seat), which start the catalog command
without going through the recipe plan.
"""

from __future__ import annotations

import pytest

from sparksi import cli, jobs
from sparksi.config import CFG
from sparksi.launch import plan_run, plan_stop, plan_switch
from sparksi.recipes.service import FOLDER_STARTS
from sparksi.recipes.store import RECIPE_FILE, folder_path

MODEL = "demo"
START = ["/srv/example/bin/start-demo", "--launch"]
STOP = ["/srv/example/bin/stop-demo"]


def _row(placement: str) -> dict:
    head = CFG.head.name
    row = {
        "id": MODEL, "title": "Example", "placement": placement, "port": 8014, "live": False,
        "estimate_gib": 8, "listen_host": head, "on": {head: "local"},
        "launch": {"start": START, "stop": STOP, "host": head},
    }
    if placement == "single":
        row["seats"] = {head: {"port": 8014, "start": START, "stop": STOP}}
    return row


def _catalog() -> dict:
    hosts = {node.name: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4} for node in CFG.nodes}
    return {"library": [], "hosts": hosts, "exclusive_groups": {}, "never_touch": {}}


@pytest.fixture()
def folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))

    def make() -> None:
        root = folder_path(MODEL, "folder-one")
        root.mkdir(parents=True)
        (root / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")

    return make


def test_without_a_folder_the_catalog_starts_are_unchanged(folder):
    assert FOLDER_STARTS not in plan_run(_catalog(), _row("single"), {}, lock_on=False)["reasons"]
    assert plan_switch(_catalog(), _row("single"), [], lock_on=False, host=CFG.head.name)["verdict"] == "READY"


def test_a_library_run_is_refused_once_a_folder_recipe_exists(folder):
    folder()
    for placement in ("single", "pair"):
        got = plan_run(_catalog(), _row(placement), {}, lock_on=False)
        assert got["verdict"] == "REFUSE" and got["argv"] is None
        assert FOLDER_STARTS in got["reasons"]


@pytest.mark.parametrize("placement", ["single", "pair"])
def test_a_library_switch_is_refused_once_a_folder_recipe_exists(folder, placement):
    folder()
    got = plan_switch(_catalog(), _row(placement), [], lock_on=False, host=CFG.head.name)
    assert got["verdict"] == "REFUSE" and got["reasons"] == [FOLDER_STARTS]
    assert got["steps"] == [] and got["preview"] == []


def test_stop_still_works_for_a_copy_started_the_old_way(folder):
    folder()
    assert plan_stop(_row("single"), lock_on=False)["verdict"] == "READY"


def test_the_jobs_and_the_cli_run_nothing(folder, monkeypatch, capsys):
    folder()
    row = _row("single")
    monkeypatch.setattr(jobs, "fleet", lambda catalog, runner=None: {"library": [row], "probes": {}})
    monkeypatch.setattr(jobs, "monitor_loaded", lambda: None)

    def never(*_args, **_kwargs):
        raise AssertionError("nothing may start")

    monkeypatch.setattr(jobs, "_start_argv", never)
    monkeypatch.setattr(jobs, "_begin", never)
    for got in (
        jobs.start_run(_catalog(), MODEL, lock_on=False),
        jobs.start_switch(_catalog(), MODEL, lock_on=False, host=CFG.head.name),
    ):
        assert got["verdict"] == "REFUSE" and FOLDER_STARTS in got["reasons"]
    monkeypatch.setattr(cli, "load_catalog", _catalog)

    class _Unlocked:
        def read(self):
            return type("State", (), {"locked": False})()

    monkeypatch.setattr(cli, "Lock", _Unlocked)
    assert cli.main(["run", MODEL]) == 2
    assert FOLDER_STARTS in capsys.readouterr().out
