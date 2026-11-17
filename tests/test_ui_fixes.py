# SPDX-License-Identifier: Apache-2.0

"""Shared ports, a two-Spark stop, and the opt-in video service."""

from __future__ import annotations

import time

from sparksi.config import CFG
from sparksi.recipes.runs import _save, reset_for_tests as reset_runs
from sparksi.recipes.service import halt_run, list_model_cards, plan_model, service_action
from sparksi.recipes.store import save_recipe
from sparksi.recipes.validate import check_port

IMAGE = "example.invalid/vllm:1"
HEAD = "spark-a"
WORKER = "spark-b"


def _role(**over) -> dict:
    role = {
        "runtime": "docker",
        "image": IMAGE,
        "env": {},
        "mounts": ["/srv/example/cache:/root/.cache/huggingface"],
        "network": "host",
        "gpus": "all",
        "container": "sparksi-example",
        "args": ["vllm", "serve", "/models/example", "--port", "8014"],
        "devices": [],
    }
    role.update(over)
    return role


def _catalog(*rows) -> dict:
    host = CFG.head.name
    library = []
    for row in rows:
        item = {
            "id": row["id"],
            "title": row.get("title") or row["id"],
            "format": "fp8",
            "gib": 8,
            "engines": ["vllm"],
            "local": {host: "/srv/example/cache/weights"},
            "port": row.get("port", 8014),
        }
        item.update(row)
        library.append(item)
    return {
        "library": library,
        "hosts": {
            host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
            CFG.worker.name: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
        },
        "exclusive_groups": {},
        "never_touch": {},
    }


def test_saved_recipes_share_a_port_until_one_is_running(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    save_recipe("left-model", {"kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014, "roles": {"head": _role()}})
    save_recipe("right-model", {"kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014, "roles": {"head": _role(container="sparksi-right")}})
    catalog = _catalog(
        {"id": "left-model", "title": "Left Recipe"},
        {"id": "right-model", "title": "Right Recipe"},
    )

    class PlanRunner:
        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            return {"ok": True, "paths": paths, "listening": {}, "http": {}}

        def capture(self, host, argv, timeout=8):
            if list(argv[:3]) == ["docker", "image", "inspect"]:
                return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}
            return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": None}

    code, planned = plan_model(catalog, PlanRunner(), "left-model", {"kind": "pinned", "engine": "vllm", "placement": "head", "roles": {"head": _role()}})
    assert code == 200, planned.get("conflicts")
    assert planned["port"] == 8014
    assert not any("another recipe" in item for item in planned["conflicts"])

    class Runner:
        def __init__(self) -> None:
            self.hosts: list[str] = []

        def invoke(self, host, req, timeout=20):
            if req.get("op") == "stop":
                self.hosts.append(str(host))
            return {"ok": True}

    run = _save({
        "id": "left-run",
        "model_id": "left-model",
        "engine": "vllm",
        "placement": "both",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "nodes": [
            {"role": "worker", "host": WORKER, "run_id": "left-run-worker", "stop_argv": ["docker", "stop", "sparksi-right"], "phase": "started"},
            {"role": "head", "host": HEAD, "run_id": "left-run-head", "stop_argv": ["docker", "stop", "sparksi-example"], "phase": "started"},
        ],
    })
    code, blocked = plan_model(catalog, PlanRunner(), "right-model", {
        "kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014, "roles": {"head": _role(container="sparksi-right")},
    })
    assert code == 400
    assert any("port 8014 is in use by Left Recipe (running)" in item for item in blocked["conflicts"]), blocked["conflicts"]

    listening = {HEAD: {8014}, WORKER: set()}
    heard = check_port(8014, listening, [HEAD])
    assert heard == ["port 8014 is already listening on head"]

    runner = Runner()
    stopped = halt_run(catalog, runner, run["id"])
    assert stopped[0] == 200, stopped[1]
    assert runner.hosts == [WORKER, HEAD]
    reset_runs()


def test_video_service_stays_protected_until_opted_in(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    saved = CFG.raw.get("services")
    catalog = {
        "library": [{
            "id": "video-h3",
            "title": "Video",
            "port": 8188,
            "listen_host": CFG.head.name,
            "local": {CFG.head.name: "/srv/example/video"},
            "launch": {"host": CFG.head.name, "start": ["/srv/example/bin/studio-start"], "stop": ["/srv/example/bin/studio-stop"]},
        }],
        "hosts": {CFG.head.name: {"uma_gib": 121}},
        "never_touch": {"do_not_stop": ["video-h3"]},
        "tenants": {},
        "residents": [],
        "recipes": {},
    }

    class Runner:
        def __init__(self, free=80, comfy=False) -> None:
            self.free = free
            self.comfy = comfy
            self.calls = []

        def invoke(self, host, req, timeout=30):
            self.calls.append((host, req))
            http = {}
            if self.comfy:
                http = {"8188": {"status": 200, "path": "/system_stats", "comfy": True}}
            return {"ok": True, "mem_available_gib": self.free, "http": http, "listening": {"8188": True}}

    try:
        CFG.raw["services"] = {}
        runner = Runner()
        code, payload = service_action(catalog, runner, "video-h3", "start")
        assert code == 400
        assert payload.get("protected") is True
        assert runner.calls == []
        listed = list_model_cards(catalog, runner)
        row = next(item for item in listed["services"] if item["id"] == "video-h3")
        assert row["protected"] is True
        assert row.get("controllable") is not True

        CFG.raw["services"] = {
            "video-h3": {
                "user_controllable": True,
                "reserve_gib": 40,
                "start": ["/srv/example/bin/studio-start"],
                "stop": ["/srv/example/bin/studio-stop"],
            }
        }
        quiet = Runner(free=80, comfy=False)
        code, denied = service_action(catalog, quiet, "video-h3", "start")
        assert code == 200, denied
        assert quiet.calls[-1][1]["argv"] == ["/srv/example/bin/studio-start"]
        off = list_model_cards(catalog, quiet)
        shown = next(item for item in off["services"] if item["id"] == "video-h3")
        assert shown["controllable"] is True
        assert shown["status"] == "Off"

        up = Runner(free=80, comfy=True)
        listed_up = list_model_cards(catalog, up)
        assert next(item for item in listed_up["services"] if item["id"] == "video-h3")["status"] == "Running"

        low = Runner(free=12)
        code, tight = service_action(catalog, low, "video-h3", "start")
        assert code == 400
        assert "GiB free" in tight["error"]
        assert all(req.get("op") != "service" for _host, req in low.calls)

        _save({
            "id": "head-run",
            "model_id": "left-model",
            "title": "Left Recipe",
            "placement": "head",
            "port": 8014,
            "phase": "ready",
            "started": time.time(),
            "nodes": [{"role": "head", "host": HEAD}],
        })
        busy = Runner(free=80)
        code, held = service_action(catalog, busy, "video-h3", "start")
        assert code == 400
        assert "running on the head" in held["error"]
        reset_runs()

        CFG.raw["services"]["video-h3"]["start"] = ["/srv/example/bin/video-mode.sh", "video"]
        refused = service_action(catalog, Runner(free=80), "video-h3", "start")
        assert refused[0] == 400
        assert "video-mode.sh" in refused[1]["error"]
    finally:
        if saved is None:
            CFG.raw.pop("services", None)
        else:
            CFG.raw["services"] = saved
        reset_runs()
