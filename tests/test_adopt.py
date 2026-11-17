# SPDX-License-Identifier: Apache-2.0

"""Running Sparksi containers are adopted when this process has no run for them."""

from __future__ import annotations

import time

import pytest

from sparksi.agent import _ports_from_inspect
from sparksi.config import CFG
from sparksi.recipes.adopt import adopt_running, adoption_view, baseline_for, note_idle, read_idle
from sparksi.recipes.runs import _save, get_run, list_runs, reset_for_tests as reset_runs, tick
from sparksi.recipes.service import _live_usage, _role_charge, _run_status, halt_run, plan_model, preview_recipe, runs_index


def _catalog() -> dict:
    head = CFG.head.name
    worker = CFG.worker.name
    return {
        "library": [
            {
                "id": "demo",
                "title": "Example Flash",
                "format": "fp8",
                "gib": 20,
                "engines": ["vllm"],
                "placement": "both",
                "port": 8014,
                "local": {head: "/srv/example/cache/weights", worker: "/srv/example/cache/weights"},
            },
            {
                "id": "other",
                "title": "Other",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "port": 8013,
                "local": {head: "/srv/example/cache/other"},
            },
            {
                "id": "comfy",
                "title": "Comfy",
                "kind": "comfy",
                "port": 8188,
                "local": {head: "/srv/example/cache/comfy"},
            },
        ],
        "hosts": {
            head: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
            worker: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
        },
        "exclusive_groups": {},
        "never_touch": {},
    }


class Nodes:
    def __init__(self) -> None:
        head = CFG.head.name
        worker = CFG.worker.name
        self.up = {
            head: ["sparksi-demo-head", "helper", "sparksi-comfy", "sparksi-other-head"],
            worker: ["sparksi-demo-worker", "helper"],
        }
        self.ports = {head: {"sparksi-demo-head": 8013, "sparksi-comfy": 8188}, worker: {}}
        self.stopped: list[tuple[str, list[str]]] = []
        self.argv_stops: list[tuple[str, list[str]]] = []
        self.listening = True
        self.http_status = 200
        self.mem_total = 128
        self.mem_free = 40

    def invoke(self, host, req, timeout=20):
        op = req.get("op")
        names = list(self.up.get(host) or [])
        if op == "probe":
            ports = {}
            if req.get("container_ports"):
                for name in names:
                    if name in (self.ports.get(host) or {}):
                        ports[name] = self.ports[host][name]
            asked = [str(port) for port in (req.get("ports") or [])]
            http = {port: {"status": self.http_status} for port in asked} if self.listening else {}
            return {
                "ok": True,
                "containers": names,
                "container_ports": ports,
                "listening": {port: self.listening for port in asked},
                "http": http,
                "mem_total_gib": self.mem_total,
                "mem_available_gib": self.mem_free,
            }
        if op == "status":
            container = str(req.get("container") or "")
            if container and container not in names:
                return {"ok": True, "container_status": "missing", "log_tail": ""}
            return {"ok": True, "phase": "running", "container_status": "running", "log_tail": "up"}
        if op == "stop":
            self.stopped.append((str(host), [str(part) for part in (req.get("argv") or [])]))
            container = ""
            argv = [str(part) for part in (req.get("argv") or [])]
            if len(argv) >= 4 and argv[:3] == ["docker", "rm", "-f"]:
                container = argv[3]
            if container:
                self.up[host] = [name for name in (self.up.get(host) or []) if name != container]
            return {"ok": True}
        return {"ok": True, "containers": names, "listening": {}, "http": {}}

    def run_argv(self, host, argv, timeout=20):
        self.argv_stops.append((str(host), [str(part) for part in argv]))
        return {"ok": True}


def _ready(monkeypatch, tmp_path) -> tuple[Nodes, dict, dict]:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    nodes = Nodes()
    catalog = _catalog()
    created = adopt_running(catalog, nodes)
    assert len(created) == 1
    run = created[0]
    assert run["phase"] == "ready", run
    return nodes, catalog, run


def test_container_port_comes_from_env_before_the_command_flag() -> None:
    blob = "PORT=8013\nVLLM_PORT=8013\nSERVER_PORT=8013\nMASTER_PORT=29501\n---CMD---\n--port\n8014\n"
    assert _ports_from_inspect(blob)[0] == 8013


def test_adopt_running_pair_is_ready_on_both_sides(monkeypatch, tmp_path) -> None:
    nodes, catalog, run = _ready(monkeypatch, tmp_path)
    assert run["started_by"] == "external"
    assert run["port"] == 8013
    assert run["port"] != 8014
    assert run["placement"] == "both"
    assert [node["role"] for node in run["nodes"]] == ["head", "worker"]
    assert [node["container"] for node in run["nodes"]] == ["sparksi-demo-head", "sparksi-demo-worker"]
    status = _run_status(run)
    assert status == "Running on the head and the worker, port 8013."
    again = adopt_running(catalog, nodes)
    assert again == []
    listed = runs_index(nodes, catalog)
    assert len(listed["runs"]) == 1
    assert listed["runs"][0]["model_id"] == "demo"
    assert listed["runs"][0]["started_by"] == "external"
    assert {node["role"] for node in listed["runs"][0]["nodes"]} == {"head", "worker"}
    reset_runs()


def test_stop_on_an_adopted_run_stops_head_then_worker(monkeypatch, tmp_path) -> None:
    nodes, catalog, run = _ready(monkeypatch, tmp_path)
    code, body = halt_run(catalog, nodes, run["id"])
    assert code == 200, body
    assert body["stopped"] is True
    stopped = get_run(run["id"])
    assert stopped["phase"] == "stopped"
    assert not stopped.get("error")
    assert stopped["phase"] != "failed"
    assert [node["phase"] for node in stopped["nodes"]] == ["stopped", "stopped"]
    assert [host for host, _argv in nodes.stopped] == [CFG.head.name, CFG.worker.name]
    assert nodes.stopped[0][1] == ["docker", "rm", "-f", "sparksi-demo-head"]
    assert nodes.stopped[1][1] == ["docker", "rm", "-f", "sparksi-demo-worker"]
    _code, planned = plan_model(catalog, nodes, "demo", {"engine": "vllm", "placement": "head", "port": 8013})
    assert not any("stop it first" in item for item in planned.get("conflicts") or [])
    reset_runs()


def test_adopted_run_port_blocks_a_second_start(monkeypatch, tmp_path) -> None:
    nodes, catalog, run = _ready(monkeypatch, tmp_path)
    assert run["phase"] == "ready"
    ports, gib = _live_usage("other", catalog)
    assert 8013 in ports["head"]
    assert gib["head"] > 0
    code, same = plan_model(catalog, nodes, "demo", {"engine": "vllm", "placement": "both", "port": 8014})
    blob = " ".join(same.get("conflicts") or [])
    assert code == 400
    assert "Example Flash is already running — stop it first or use Restart with changes" in blob
    code, other = plan_model(catalog, nodes, "other", {"engine": "vllm", "placement": "head", "port": 8013})
    other_blob = " ".join(other.get("conflicts") or [])
    assert code == 400
    assert "8013" in other_blob
    assert "in use" in other_blob
    reset_runs()


def test_vanished_adopted_containers_end_stopped(monkeypatch, tmp_path) -> None:
    nodes, _catalog_body, run = _ready(monkeypatch, tmp_path)
    nodes.up = {CFG.head.name: ["helper"], CFG.worker.name: []}
    current = tick(get_run(run["id"]), nodes)
    assert current["phase"] == "stopped", current
    assert current["phase"] != "failed"
    assert not current.get("error")
    assert not current.get("failed_side")
    later = tick(get_run(run["id"]), nodes)
    assert later["phase"] == "stopped"
    assert not later.get("error")
    reset_runs()


def _adopted(phase: str = "waiting_health", **over) -> dict:
    head = CFG.head.name
    worker = CFG.worker.name
    run = {
        "id": "adopt1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "phase": phase,
        "started_by": "external",
        "started": time.time(),
        "timeout_s": 1800,
        "health_paths": ["/health"],
        "seen": {},
        "nodes": [{
            "role": "head",
            "host": head,
            "phase": "started",
            "container": "sparksi-demo",
            "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
        }],
    }
    run.update(over)
    if run.get("placement") == "both":
        run["nodes"] = [
            {
                "role": "head",
                "host": head,
                "phase": "started",
                "container": "sparksi-demo-head",
                "stop_argv": ["docker", "rm", "-f", "sparksi-demo-head"],
            },
            {
                "role": "worker",
                "host": worker,
                "phase": "started",
                "container": "sparksi-demo-worker",
                "stop_argv": ["docker", "rm", "-f", "sparksi-demo-worker"],
            },
        ]
        run["seen"] = {"head": True, "worker": True}
    return run


def test_adopted_timeout_does_not_stop_containers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    nodes = Nodes()
    nodes.up[CFG.head.name] = ["sparksi-demo"]
    nodes.listening = False
    saved = _save(_adopted(started=time.time() - 120, timeout_s=30))
    current = tick(saved, nodes)
    assert current["phase"] == "waiting_health", current
    assert current["phase"] != "timed_out"
    assert current["health_note"] == "Not answering health checks"
    assert _run_status(current) == "Not answering health checks"
    assert nodes.stopped == []
    assert nodes.argv_stops == []
    reset_runs()


def test_adopted_health_failure_does_not_stop_containers(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    nodes = Nodes()
    nodes.up[CFG.head.name] = ["sparksi-demo"]
    nodes.http_status = 500
    saved = _save(_adopted())
    current = tick(saved, nodes)
    assert current["phase"] == "waiting_health", current
    assert current["phase"] != "failed"
    assert current["health_note"] == "Not answering health checks"
    assert nodes.stopped == []
    assert nodes.argv_stops == []
    reset_runs()


def test_adopted_pair_death_does_not_stop_the_other_side(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    nodes = Nodes()
    nodes.up[CFG.worker.name] = ["helper"]
    saved = _save(_adopted(phase="ready", placement="both"))
    current = tick(saved, nodes)
    assert current["phase"] == "ready", current
    assert current["phase"] != "failed"
    assert current["health_note"] == "Worker side stopped"
    assert _run_status(current) == "Worker side stopped"
    assert nodes.stopped == []
    assert nodes.argv_stops == []
    assert "sparksi-demo-head" in nodes.up[CFG.head.name]
    reset_runs()


def test_adopted_real_usage_refuses_a_model_the_estimate_would_allow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    head = CFG.head.name
    catalog = {
        "library": [
            {
                "id": "wide",
                "title": "Wide",
                "format": "fp8",
                "gib": 82,
                "engines": ["vllm"],
                "local": {head: "/srv/example/cache/wide"},
            },
            {
                "id": "small",
                "title": "Small",
                "format": "gguf",
                "gib": 8,
                "engines": ["llamacpp"],
                "local": {head: "/srv/example/cache/small.gguf"},
            },
        ],
        "hosts": {head: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    _save({
        "id": "adopt1",
        "model_id": "wide",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "phase": "ready",
        "started_by": "external",
        "started": time.time(),
        "nodes": [{
            "role": "head",
            "host": head,
            "phase": "started",
            "container": "sparksi-wide",
            "container_gib": 40,
            "mem_baseline_gib": 10,
            "host_used_gib": 119,
        }],
    })
    assert _role_charge(get_run("adopt1"), "head", catalog) == 109

    class Present:
        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            return {"ok": True, "paths": paths, "listening": {}, "http": {}, "containers": []}

        def capture(self, host, argv, timeout=8):
            return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}

    ask = {"engine": "llamacpp", "placement": "head", "port": 8010}
    code, refused = plan_model(catalog, Present(), "small", ask)
    assert code == 400, refused
    assert any("over budget" in item for item in refused["conflicts"]), refused["conflicts"]
    lighter = get_run("adopt1")
    lighter["nodes"][0].pop("container_gib")
    lighter["nodes"][0].pop("mem_baseline_gib")
    lighter["nodes"][0].pop("host_used_gib")
    _save(lighter)
    assert _role_charge(get_run("adopt1"), "head", catalog) == 90
    code, allowed = plan_model(catalog, Present(), "small", ask)
    assert code == 200, allowed.get("conflicts")
    reset_runs()


def test_plan_and_preview_do_not_adopt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    nodes = Nodes()
    catalog = _catalog()
    plan_model(catalog, nodes, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    preview_recipe(catalog, nodes, "demo", {"engine": "vllm", "placement": "head"})
    assert list_runs() == []
    listed = runs_index(nodes, catalog)
    assert any(item["model_id"] == "demo" and item["started_by"] == "external" for item in listed["runs"])
    reset_runs()


def test_live_free_memory_refuses_a_start_the_budget_would_allow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    head = CFG.head.name

    class Free:
        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            return {
                "ok": True,
                "paths": paths,
                "listening": {},
                "http": {},
                "containers": [],
                "mem_total_gib": 200,
                "mem_available_gib": 2.5,
            }

        def capture(self, host, argv, timeout=8):
            return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}

    catalog = {
        "library": [{
            "id": "small",
            "title": "Small",
            "format": "gguf",
            "gib": 13,
            "engines": ["llamacpp"],
            "local": {head: "/srv/example/cache/small.gguf"},
        }],
        "hosts": {head: {"uma_gib": 200, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    code, planned = plan_model(catalog, Free(), "small", {
        "engine": "llamacpp", "placement": "head", "port": 8010,
    })
    assert code == 400, planned
    assert "the head has only 2.5 GiB free right now and this model needs 21 GiB plus 5 GiB margin" in planned["conflicts"]
    assert any(item.get("code") == "memory_not_free" for item in planned["reasons"])
    assert not any("over budget" in item for item in planned["conflicts"])
    assert list_runs() == []
    reset_runs()


def test_idle_baseline_is_persisted_then_config_then_ten(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    head = CFG.head.name
    note_idle(head, 8)
    assert baseline_for(head) == 8
    monkeypatch.setitem(CFG.raw["controller"], "idle_baseline_gib", 12)
    assert baseline_for(head) == 8
    idle = read_idle()
    assert idle[head] == 8
    from sparksi.recipes.adopt import idle_path
    idle_path().unlink()
    assert baseline_for(head) == 12
    monkeypatch.delitem(CFG.raw["controller"], "idle_baseline_gib", raising=False)
    assert baseline_for(head) == 10

    class Busy:
        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            names = ["sparksi-wide"] if host == head else []
            return {
                "ok": True,
                "paths": paths,
                "listening": {},
                "http": {},
                "containers": names,
                "container_ports": {"sparksi-wide": 8013} if names else {},
                "mem_total_gib": 121.7,
                "mem_available_gib": 2.5,
            }

        def capture(self, host, argv, timeout=8):
            return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}

    catalog = {
        "library": [
            {
                "id": "wide",
                "title": "Wide",
                "format": "fp8",
                "gib": 82,
                "engines": ["vllm"],
                "local": {head: "/srv/example/cache/wide"},
            },
            {
                "id": "small",
                "title": "Small",
                "format": "gguf",
                "gib": 13,
                "engines": ["llamacpp"],
                "local": {head: "/srv/example/cache/small.gguf"},
            },
        ],
        "hosts": {head: {"uma_gib": 121.7, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    view = adoption_view(catalog, Busy())
    assert len(view) == 1
    assert view[0]["nodes"][0]["mem_baseline_gib"] == 10
    assert _role_charge(view[0], "head", catalog) == pytest.approx(109.2)
    code, planned = plan_model(catalog, Busy(), "small", {
        "engine": "llamacpp", "placement": "head", "port": 8010,
    })
    assert code == 400, planned
    assert list_runs() == []
    assert not (tmp_path / "recipe-runs").exists() or not list((tmp_path / "recipe-runs").glob("*.json"))
    reset_runs()


def test_plan_before_a_poll_sees_running_containers_and_saves_nothing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    nodes = Nodes()
    nodes.mem_total = 121
    nodes.mem_free = 80
    catalog = _catalog()
    code, same = plan_model(catalog, nodes, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    assert code == 400, same
    assert any("stop it first" in item for item in same.get("conflicts") or [])
    code, other = plan_model(catalog, nodes, "other", {"engine": "vllm", "placement": "head", "port": 8013})
    assert code == 400, other
    blob = " ".join(other.get("conflicts") or [])
    assert "8013" in blob and "in use" in blob
    preview_recipe(catalog, nodes, "demo", {"engine": "vllm", "placement": "head"})
    assert list_runs() == []
    folder = tmp_path / "recipe-runs"
    assert not folder.exists() or not list(folder.glob("*.json"))
    reset_runs()


def test_status_poll_records_idle_and_does_not_overwrite_it(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    head = CFG.head.name
    nodes = Nodes()
    nodes.up[head] = ["helper"]
    nodes.up[CFG.worker.name] = []
    nodes.mem_total = 20
    nodes.mem_free = 12
    adopt_running(_catalog(), nodes)
    assert read_idle()[head] == 8
    nodes.up[head] = ["sparksi-demo"]
    nodes.mem_free = 1
    adopt_running(_catalog(), nodes)
    assert read_idle()[head] == 8
    reset_runs()


def test_non_sparksi_containers_are_ignored(monkeypatch, tmp_path) -> None:
    _nodes, _catalog_body, run = _ready(monkeypatch, tmp_path)
    ids = {item["model_id"] for item in list_runs() if item.get("phase") not in ("stopped", "failed")}
    assert ids == {"demo"}
    assert all(str(node.get("container") or "").startswith("sparksi-demo-") for node in run["nodes"])
    reset_runs()
