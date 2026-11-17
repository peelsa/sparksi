# SPDX-License-Identifier: Apache-2.0

"""Batch fixes: ledger, refusals, adopted restart, and the CLI wait path."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sparksi.agent import _recipe_run
from sparksi.config import CFG
from sparksi.ledger import (
    day_key,
    day_view,
    fold,
    parse_metrics,
    poll_once,
    previous_calendar_day,
    read,
    targets_from_runs,
)
from sparksi.recipes.adopt import adopt_running
from sparksi.recipes.memory import node_peak_gib, reset_for_tests as reset_memory
from sparksi.recipes.reasons import join_reasons, reason_code
from sparksi.recipes.runs import get_run, public_run, reset_for_tests as reset_runs, start_rendered
from sparksi.recipes.service import plan_model, preview_recipe


def test_day_key_follows_the_local_calendar_across_dst() -> None:
    from datetime import timezone

    zone = ZoneInfo("America/Chicago")
    spring = datetime(2026, 3, 8, 12, 0, tzinfo=zone)
    winter = datetime(2026, 1, 15, 12, 0, tzinfo=zone)
    assert spring.utcoffset() == timedelta(hours=-5)
    assert winter.utcoffset() == timedelta(hours=-6)
    # During daylight time a fixed six-hour offset is already the previous date.
    utc = datetime(2026, 7, 15, 5, 30, tzinfo=timezone.utc)
    local = utc.astimezone(zone)
    fixed = utc.astimezone(timezone(timedelta(hours=-6)))
    assert local.day == 15
    assert fixed.day == 14
    assert day_key(local) == "2026-07-15"
    evening = datetime(2026, 11, 1, 23, 0, tzinfo=zone)
    assert day_key(evening) == "2026-11-01"
    assert day_key(previous_calendar_day(evening)) == "2026-10-31"
    assert day_key(previous_calendar_day(spring)) == "2026-03-07"


def test_first_sight_is_not_credited_and_a_reset_is(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    moment = datetime(2026, 11, 1, 12, 0, tzinfo=ZoneInfo("America/Chicago"))
    first = fold("glm", role="head", port=8014, prompt=1000, generation=500, now=moment)
    assert first["life_gen"] == 0
    assert first["life_prompt"] == 0
    second = fold("glm", role="head", port=8014, prompt=1100, generation=560, now=moment + timedelta(seconds=15))
    assert second["life_gen"] == 60
    assert second["life_prompt"] == 100
    reset = fold("glm", role="head", port=8014, prompt=4, generation=2, now=moment + timedelta(seconds=30))
    assert reset["life_gen"] == 62
    view = day_view(moment + timedelta(seconds=30))
    assert view["ok"] is True
    assert view["window"] == "today"
    assert view["days"][0]["date"] == "2026-11-01"
    assert view["days"][0]["total"]["generation"] == "62"
    before = read()
    again = day_view(moment)
    assert again["days"][0]["total"] == view["days"][0]["total"]
    assert read() == before


def test_split_run_targets_come_from_the_run_record() -> None:
    head = CFG.head.name
    worker = CFG.worker.name
    runs = [{
        "id": "split1",
        "model_id": "glm-5.3-exl3",
        "phase": "ready",
        "placement": "both",
        "port": 8014,
        "nodes": [
            {"role": "head", "host": head},
            {"role": "worker", "host": worker},
        ],
    }]
    found = targets_from_runs(runs)
    assert {(item["role"], item["port"]) for item in found} == {("head", 8014), ("worker", 8014)}
    assert {item["model_id"] for item in found} == {"glm-5.3-exl3"}


def test_poll_folds_both_sides_and_a_read_does_not_write(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    head = CFG.head.name
    worker = CFG.worker.name
    runs = [{
        "model_id": "glm",
        "phase": "ready",
        "placement": "both",
        "port": 8014,
        "nodes": [
            {"role": "head", "host": head},
            {"role": "worker", "host": worker},
        ],
    }]
    blob = "\n".join([
        "# HELP vllm:generation_tokens_total generated",
        'vllm:prompt_tokens_total{model="glm"} 10',
        'vllm:generation_tokens_total{model="glm"} 4',
    ])
    calls: list[tuple[str, int]] = []

    def fetch(address: str, port: int) -> str:
        calls.append((address, port))
        return blob

    moment = datetime(2026, 11, 5, 9, 0, tzinfo=ZoneInfo("America/Chicago"))
    assert poll_once(runs, fetch, moment) == 2
    assert len(calls) == 2
    assert {port for _addr, port in calls} == {8014}
    assert day_view(moment)["days"][0]["total"]["generation"] == "0"
    later = "\n".join([
        'vllm:prompt_tokens_total{model="glm"} 12',
        'vllm:generation_tokens_total{model="glm"} 9',
    ])
    poll_once(runs, lambda _a, _p: later, moment + timedelta(seconds=15))
    body = day_view(moment + timedelta(seconds=15))
    assert body["days"][0]["total"]["generation"] == "10"
    assert body["decode_tok_s"] is not None and body["decode_tok_s"] > 0
    stored = json.dumps(read())
    day_view()
    assert json.dumps(read()) == stored


def test_metrics_parser_accepts_engine_prefixes() -> None:
    text = "\n".join([
        'tensorfold:prompt_tokens_total 3',
        'sglang:generation_tokens_total 8',
    ])
    assert parse_metrics(text) == {"prompt": 3.0, "generation": 8.0}


def test_refusal_codes_and_separated_text() -> None:
    memory = "the head has only 2.4 GiB free right now and this model needs 21 GiB plus 5 GiB margin"
    port = "port 8014 is already listening on the head"
    off = "port 8188 is off limits"
    assert reason_code(memory) == "memory_not_free"
    assert reason_code(port) == "port_conflict"
    assert reason_code(off) == "off_limits"
    assert "free right now" in memory
    joined = join_reasons(["Example Flash is already running", memory])
    assert " · " in joined
    assert "running the head" not in joined


def test_refused_preview_is_http_200(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    head = CFG.head.name

    class Free:
        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            return {"ok": True, "paths": paths, "listening": {}, "http": {}, "containers": [], "mem_available_gib": 80}

        def capture(self, host, argv, timeout=8):
            return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}

    catalog = {
        "library": [{
            "id": "small",
            "title": "Small",
            "format": "gguf",
            "gib": 8,
            "engines": ["llamacpp"],
            "local": {head: "/srv/example/cache/small.gguf"},
        }],
        "hosts": {head: {"uma_gib": 121}},
        "never_touch": {},
    }
    code, body = preview_recipe(catalog, Free(), "small", {"engine": "llamacpp", "placement": "head", "port": 8188})
    assert code == 200, body
    assert body["refused"] is True
    assert any(item["code"] == "off_limits" for item in body["reasons"])
    plan_code, planned = plan_model(catalog, Free(), "small", {"engine": "llamacpp", "placement": "head", "port": 8188})
    assert plan_code == 400
    assert any(item["code"] == "off_limits" for item in planned["reasons"])
    reset_runs()


def _restart_catalog(head: str) -> dict:
    return {
        "library": [{
            "id": "demo",
            "title": "Example Flash",
            "format": "fp8",
            "gib": 40,
            "engines": ["vllm"],
            "placement": "head",
            "port": 8013,
            "local": {head: "/srv/example/cache/weights"},
        }],
        "hosts": {head: {"uma_gib": 128}},
        "never_touch": {},
    }


class _RestartNodes:
    def __init__(self, head: str) -> None:
        del head
        self.up = ["sparksi-demo"]
        self.mem_free = 2.5
        self.ops: list[str] = []
        self.hold_port = True

    def invoke(self, host, req, timeout=20):
        self.ops.append(str(req.get("op") or ""))
        names = list(self.up)
        if req.get("op") == "probe":
            ports = {}
            if req.get("container_ports") and self.hold_port and "sparksi-demo" in names:
                ports["sparksi-demo"] = 8013
            asked = [str(port) for port in (req.get("ports") or [])]
            return {
                "ok": True,
                "containers": names,
                "container_ports": ports,
                "listening": {port: True for port in asked},
                "http": {port: {"status": 200} for port in asked},
                "paths": {str(path): {"exists": True} for path in (req.get("paths") or [])},
                "mem_total_gib": 128,
                "mem_available_gib": self.mem_free,
            }
        if req.get("op") == "stop":
            argv = [str(part) for part in (req.get("argv") or [])]
            if len(argv) >= 4 and argv[:3] == ["docker", "rm", "-f"]:
                self.up = [name for name in names if name != argv[3]]
            return {"ok": True}
        if req.get("op") == "run":
            return {"ok": True, "phase": "running"}
        return {"ok": True, "containers": names}

    def capture(self, host, argv, timeout=8):
        return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}


def test_replace_credit_is_only_the_memory_still_held(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    from sparksi.recipes.runs import _save
    from sparksi.recipes.service import release_credit

    stopped = _save({
        "id": "old1",
        "model_id": "demo",
        "phase": "stopped",
        "stop_requested": True,
        "stopped_at": 100.0,
        "placement": "head",
        "port": 8013,
        "started": 10.0,
        "nodes": [{
            "role": "head",
            "host": CFG.head.name,
            "phase": "stopped",
            "container": "sparksi-demo",
            "mem_baseline_gib": 10,
            "host_used_gib": 70,
            "free_at_stop_gib": 4,
        }],
    })
    # Host delta is 60. Nothing has come back.
    assert release_credit(stopped, "head", 4.0, {}) == 60
    # Forty GiB came back, so only the rest of the charge is still missing.
    assert release_credit(stopped, "head", 44.0, {}) == 20
    # The whole charge is already in free memory. Crediting it again would double-count.
    assert release_credit(stopped, "head", 64.0, {}) == 0
    _save({
        "id": "other1",
        "model_id": "other",
        "phase": "ready",
        "placement": "head",
        "port": 8014,
        "started": 101.0,
        "nodes": [{"role": "head", "host": CFG.head.name, "phase": "started"}],
    })
    assert release_credit(get_run("old1"), "head", 4.0, {}) == 0
    reset_runs()


def test_restart_credit_and_port_follow_the_named_run(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    reset_memory()
    head = CFG.head.name
    nodes = _RestartNodes(head)
    catalog = _restart_catalog(head)
    created = adopt_running(catalog, nodes)
    assert created and node_peak_gib("demo", "head") is None
    from sparksi.recipes.service import halt_run

    code, stopped = halt_run(catalog, nodes, created[0]["id"])
    assert code == 200, stopped
    saved = get_run(created[0]["id"])
    assert saved["nodes"][0]["free_at_stop_gib"] == 2.5
    # The container is gone and the port is still taken by something else.
    nodes.up = []
    nodes.hold_port = False
    nodes.mem_free = 2.5
    listening = {head: {8013}, "head": {8013}}
    monkeypatch.setattr(
        "sparksi.recipes.service._facts",
        lambda catalog, runner: (listening, {}, {}, set()),
    )
    weights = {head: {"/srv/example/cache/weights": True}}
    listening = {head: {8013}, "head": {8013}}
    monkeypatch.setattr(
        "sparksi.recipes.service._facts",
        lambda catalog, runner: (listening, {}, weights, set()),
    )
    bare = {"engine": "vllm", "placement": "head", "port": 8013}
    code, plain = plan_model(catalog, nodes, "demo", bare)
    assert code == 400, plain
    assert any("free right now" in item for item in plain["conflicts"])
    assert any("already listening" in item for item in plain["conflicts"])
    # Naming the stopped run credits memory that has not come back.
    # The port stays taken: nothing of this run is still listening.
    code, replaced = plan_model(catalog, nodes, "demo", {**bare, "replace_run_id": saved["id"]})
    assert code == 400, replaced
    assert not any("free right now" in item for item in replaced["conflicts"])
    assert any("already listening" in item for item in replaced["conflicts"])
    from sparksi.recipes.service import _without_replaced_port

    hidden = _without_replaced_port(listening, saved, nodes)
    assert 8013 in hidden[head]
    nodes.up = ["sparksi-demo"]
    nodes.hold_port = True
    owned = _without_replaced_port(listening, saved, nodes)
    assert 8013 not in owned[head]
    assert "run" not in nodes.ops
    preview_recipe(catalog, nodes, "demo", {**bare, "replace_run_id": saved["id"]})
    assert "run" not in nodes.ops
    reset_runs()
    reset_memory()


def test_started_by_is_set_on_a_launch(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()

    class Runner:
        def invoke(self, host, req, timeout=120):
            return {"ok": True, "phase": "running"}

        def run_argv(self, host, argv, timeout=60):
            return {"ok": True}

    recipe = {
        "id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "timeout_s": 30,
        "health_paths": [],
        "order": ["head"],
        "nodes": [{
            "role": "head",
            "host": CFG.head.name,
            "argv": ["docker", "run", "-d", "--name", "sparksi-demo", "example:1", "vllm", "serve"],
            "env": {},
            "container": "sparksi-demo",
            "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
        }],
    }
    run = start_rendered(recipe, Runner(), sync=False)
    assert run["started_by"] == "sparksi"
    assert public_run(run)["started_by"] == "sparksi"
    reset_runs()


def test_a_running_container_is_removed_only_for_a_stopped_replace(monkeypatch, tmp_path) -> None:
    from sparksi.agent import _recipe_run, _write_run

    monkeypatch.setattr("sparksi.agent._runs_dir", lambda: tmp_path)
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr("sparksi.agent.subprocess.run", fake_run)
    monkeypatch.setattr("sparksi.agent._inspect_container", lambda name: {"status": "running"})
    argv = ["docker", "run", "-d", "--name", "sparksi-glm", "example:1", "serve"]
    refused = _recipe_run({"run_id": "new1", "argv": argv, "container": "sparksi-glm"})
    assert refused["ok"] is False
    assert refused["error"] == "container sparksi-glm is already running"
    assert seen == []
    named = _recipe_run({
        "run_id": "new2",
        "replace_run_id": "old-head",
        "argv": argv,
        "container": "sparksi-glm",
    })
    assert named["ok"] is False
    assert named["error"] == "container sparksi-glm is already running"
    assert seen == []
    _write_run({"run_id": "old-head", "phase": "running", "argv": argv})
    still = _recipe_run({
        "run_id": "new3",
        "replace_run_id": "old-head",
        "argv": argv,
        "container": "sparksi-glm",
    })
    assert still["ok"] is False
    assert seen == []
    _write_run({"run_id": "old-head", "phase": "stopped", "argv": argv})
    allowed = _recipe_run({
        "run_id": "new4",
        "replace_run_id": "old-head",
        "argv": argv,
        "container": "sparksi-glm",
    })
    assert allowed["ok"] is True
    assert seen[0] == ["docker", "rm", "-f", "sparksi-glm"]
    assert seen[1][:2] == ["docker", "run"]


def test_ledger_read_does_not_create_a_directory(tmp_path, monkeypatch) -> None:
    home = tmp_path / "empty-home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SPARKSI_STATE_DIR", raising=False)
    from sparksi.ledger import ledger_path, read

    path = ledger_path()
    assert read()["series"] == {}
    assert not path.exists()
    assert not path.parent.exists()


def test_recipe_run_removed_name_is_not_docker_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("sparksi.agent._runs_dir", lambda: tmp_path)
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))

        class Result:
            returncode = 0
            stdout = "ok"
            stderr = ""

        return Result()

    monkeypatch.setattr("sparksi.agent.subprocess.run", fake_run)
    monkeypatch.setattr("sparksi.agent._inspect_container", lambda name: {"status": "exited"})
    res = _recipe_run({
        "run_id": "again",
        "argv": ["docker", "run", "-d", "--name", "sparksi-glm", "--ctx", "8192", "example:1", "serve"],
        "container": "sparksi-glm",
    })
    assert res["ok"] is True
    assert seen[0] == ["docker", "rm", "sparksi-glm"]
    assert seen[1][0:2] == ["docker", "run"]
    assert "--ctx" in seen[1]
    assert ["docker", "start", "sparksi-glm"] not in seen
