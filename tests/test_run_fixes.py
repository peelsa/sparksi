# SPDX-License-Identifier: Apache-2.0

"""Live start and stop fixes: RoCE launch policy, tailnet health, entrypoint, plan."""

from __future__ import annotations

import time

from sparksi.agent import _probe, _refuse_recipe_argv, docker_option_refusal
from sparksi.config import CFG
from sparksi.jobs import _probe_node
from sparksi.recipes.importer import _role_from_docker_run, parse_docker_inspect
from sparksi.recipes.memory import MEMORY_MARGIN_GIB, record_node_peak, record_peak, reset_for_tests as reset_memory, role_need_gib
from sparksi.recipes.pinned import recipe_allows_roce, role_argv, validate_pinned
from sparksi.recipes.runs import _save, reset_for_tests as reset_runs, tick
from sparksi.recipes.service import plan_model

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
        "args": ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"],
        "devices": [],
    }
    role.update(over)
    return role


def _recipe(placement: str, **role_over) -> dict:
    head = _role(**role_over)
    roles = {"head": head}
    if placement == "both":
        roles["worker"] = _role(**role_over)
    return {
        "kind": "pinned",
        "engine": "vllm",
        "placement": placement,
        "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": roles,
    }


def _roce_over() -> dict:
    return {
        "cap_add": ["IPC_LOCK"],
        "security_opt": ["label=disable"],
        "devices": ["/dev/infiniband"],
        "extra_docker_args": "--ulimit memlock=-1 --ulimit stack=67108864",
    }


def test_two_node_roce_passes_both_checks_and_other_caps_stay_refused() -> None:
    recipe = _recipe("both", **_roce_over())
    assert recipe_allows_roce(recipe) is True
    assert validate_pinned(recipe) == []
    for name, bind in (("head", "100.64.0.11"), ("worker", "100.64.0.12")):
        argv = role_argv(recipe["roles"][name], bind)
        assert _refuse_recipe_argv(argv, allow_roce=True) is None, argv
        assert "--cap-add" in argv and "IPC_LOCK" in argv
        assert "/dev/infiniband" in argv
        refused = _refuse_recipe_argv(argv, allow_roce=False)
        assert refused is not None and "--cap-add" in refused
    single = _recipe("head", cap_add=["SYS_ADMIN"])
    assert any("refusing --cap-add SYS_ADMIN" in item for item in validate_pinned(single))
    single_argv = role_argv(single["roles"]["head"], "100.64.0.11")
    agent_refused = _refuse_recipe_argv(single_argv, allow_roce=False)
    assert agent_refused is not None and "refusing --cap-add SYS_ADMIN" in agent_refused
    assert docker_option_refusal("--cap-add", "SYS_ADMIN", allow_roce=True) == "refusing --cap-add SYS_ADMIN"
    assert docker_option_refusal("--ulimit", "nofile=1024", allow_roce=True) == "refusing --ulimit nofile=1024"
    assert any("refusing --ulimit nofile=1024" in item for item in validate_pinned(
        _recipe("both", extra_docker_args="--ulimit nofile=1024"),
    ))


def test_readiness_probes_the_tailnet_address(monkeypatch) -> None:
    seen: dict = {}

    def port_open(port, host="127.0.0.1"):
        seen["host"] = host
        return True

    def http(port, path="/v1/models", host="127.0.0.1"):
        seen.setdefault("calls", []).append((host, path))
        return {"path": path, "status": 200, "models": ["example"]}

    monkeypatch.setattr("sparksi.agent._port_open", port_open)
    monkeypatch.setattr("sparksi.agent._http_identity", http)
    monkeypatch.setattr("sparksi.agent._running_containers", lambda: [])
    monkeypatch.setattr("sparksi.agent._ips", lambda: {"lan": None, "tailscale": None, "direct": []})
    monkeypatch.setattr("sparksi.agent._gpu", lambda: {})
    monkeypatch.setattr("sparksi.agent._meminfo", lambda: {})
    probed = _probe({"ports": [8010], "listen_host": "100.64.0.11", "http_paths": {"8010": "/health"}})
    assert seen["host"] == "100.64.0.11"
    assert seen["calls"][0] == ("100.64.0.11", "/health")
    assert probed["http"]["8010"]["status"] == 200
    _probe({"ports": [8010]})
    assert seen["host"] == "127.0.0.1"

    class Runner:
        def invoke(self, host, req, timeout=20):
            seen["req"] = dict(req)
            return {"ok": True, "listening": {"8010": True}, "http": {"8010": {"status": 200}}}

    _probe_node(Runner(), HEAD, 8010, "/v1/models")
    assert seen["req"]["listen_host"] == CFG.head.tailscale
    assert seen["req"]["http_paths"]["8010"] == "/v1/models"


def test_entrypoint_is_captured_and_a_shell_is_flagged() -> None:
    inspected = parse_docker_inspect({
        "Name": "/example",
        "Config": {
            "Image": IMAGE,
            "Entrypoint": ["vllm"],
            "Cmd": ["serve", "/models/example", "--port", "8014"],
            "Env": [],
        },
        "HostConfig": {"NetworkMode": "host"},
    })
    assert inspected["entrypoint"] == "vllm"
    assert inspected["args"][:2] == ["serve", "/models/example"]
    assert "bash" not in inspected["args"]
    argv = role_argv({**_role(args=inspected["args"], entrypoint="vllm")}, "100.64.0.11")
    assert argv[argv.index("--entrypoint") + 1] == "vllm"
    assert _refuse_recipe_argv(argv) is None, argv

    flattened = parse_docker_inspect({
        "Name": "/example",
        "Config": {"Image": IMAGE, "Entrypoint": ["/bin/bash"], "Cmd": ["vllm", "serve"], "Env": []},
        "HostConfig": {"NetworkMode": "host"},
    })
    assert flattened["entrypoint"] == "/bin/bash"
    shell = _recipe("head", args=["bash", "vllm", "serve", "--port", "8014"])
    assert any("refusing shell entrypoint bash" in item for item in validate_pinned(shell))
    overridden = _role_from_docker_run(
        ["docker", "run", "--entrypoint", "vllm", "--network", "host", IMAGE, "serve", "--port", "8014"],
        {},
        [],
    )
    assert overridden["entrypoint"] == "vllm"
    assert overridden["args"][:1] == ["serve"]
    assert overridden["args"][:2] != ["bash", "vllm"]


def _catalog(model_id: str, **row_over) -> dict:
    host = CFG.head.name
    row = {
        "id": model_id,
        "title": row_over.pop("title", "Example"),
        "format": "fp8",
        "gib": row_over.pop("gib", 8),
        "engines": row_over.pop("engines", ["vllm"]),
        "local": {host: "/srv/example/cache/weights"},
        "port": 8014,
    }
    row.update(row_over)
    return {
        "library": [row],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {},
        "never_touch": {},
    }


def test_pinned_port_is_not_remapped_and_a_live_model_is_refused(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    _save({
        "id": "live1",
        "model_id": "other-model",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "measured_gib": 22,
        "reservation_gib": 22,
        "nodes": [],
    })
    code, planned = plan_model(_catalog("demo"), None, "demo", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "roles": {"head": _role()},
    })
    assert code == 400
    assert planned["port"] == 8014
    assert any("8014" in item for item in planned["conflicts"])
    assert not any(item.startswith("port 8010") for item in planned["conflicts"])
    reset_runs()
    _save({
        "id": "same1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "waiting_health",
        "started": time.time(),
        "nodes": [],
    })
    _code, again = plan_model(_catalog("demo"), None, "demo", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "roles": {"head": _role()},
    })
    assert again["port"] == 8014
    assert not any("already running" in item for item in again["conflicts"])
    assert "head" in again["run_status"]
    assert "8014" in again["run_status"]
    reset_runs()


def test_second_start_is_refused_while_the_model_is_active(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.service import begin_run

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    host = CFG.head.name
    _save({
        "id": "busy1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "waiting_health",
        "started": time.time(),
        "nodes": [{"role": "head", "host": host, "phase": "started"}],
    })
    catalog = _catalog("demo", title="Example Flash")
    code, planned = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    blob = " ".join(planned["conflicts"])
    assert code == 400
    assert planned["verdict"] == "REFUSE"
    assert "Example Flash is already loading — stop it first or use Restart with changes" in blob
    started = {"called": False}

    def no_start(*_args, **_kwargs):
        started["called"] = True
        raise AssertionError("start should wait until the active run has stopped")

    monkeypatch.setattr("sparksi.recipes.service.start_rendered", no_start)
    code, body = begin_run(catalog, None, {"id": "demo", "engine": "vllm", "placement": "head", "port": 8014})
    assert code == 400
    assert started["called"] is False
    assert "already loading" in " ".join(body.get("conflicts") or [])
    _save({
        "id": "busy1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "nodes": [{"role": "head", "host": host, "phase": "started"}],
    })
    _code, running = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    assert "Example Flash is already running — stop it first or use Restart with changes" in " ".join(running["conflicts"])
    current = _save({
        "id": "busy1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "stopped",
        "started": time.time(),
        "nodes": [{"role": "head", "host": host, "phase": "stopped"}],
    })
    assert current["phase"] == "stopped"
    _code, again = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    assert not any("stop it first" in item for item in again["conflicts"])
    reset_runs()


def test_own_run_keeps_its_port_and_skips_self_budget(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.importer import _remap
    from sparksi.recipes.store import save_recipe

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    reset_memory()
    host = CFG.head.name
    record_node_peak("demo", "head", 112.6, solo=True)
    _save({
        "id": "self1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "waiting_health",
        "started": time.time(),
        "nodes": [{"role": "head", "host": host, "phase": "started"}],
    })
    save_recipe("demo", _recipe("head"))
    _code, planned = plan_model(_catalog("demo", title="Example"), None, "demo", {})
    blob = " ".join(planned["conflicts"])
    assert planned["port"] == 8014
    assert "remapped" not in blob
    assert "over budget" not in blob
    assert "already running" not in blob
    assert planned["run_status"].startswith("Loading on the head")
    assert "8014" in planned["run_status"]
    kept, note = _remap(8014, listening={8014}, reserved=set(), claimed=set(), own_ports={8014})
    assert kept == 8014 and note == ""

    reset_runs()
    _save({
        "id": "other1",
        "model_id": "other-model",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "nodes": [{"role": "head", "host": host}],
    })
    catalog = _catalog("demo", title="Example")
    catalog["library"].append({
        "id": "other-model",
        "title": "Other",
        "format": "fp8",
        "gib": 8,
        "engines": ["vllm"],
        "local": {host: "/srv/example/cache/other"},
        "port": 8014,
    })
    _code, other = plan_model(catalog, None, "demo", {})
    other_blob = " ".join(other["conflicts"])
    assert other["port"] == 8014
    assert "remapped" not in other_blob
    assert "Other" in other_blob
    held, held_note = _remap(
        8014,
        listening={8014},
        reserved=set(),
        claimed={8014},
        holders={8014: "port 8014 is in use by Other (running)"},
    )
    assert held == 8014
    assert "Other" in held_note
    assert "remapped" not in held_note
    reset_runs()
    reset_memory()


def test_exclusive_group_blocks_a_pinned_start(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    host = CFG.head.name
    catalog = {
        "library": [
            {
                "id": "left-model",
                "title": "Left Model",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "local": {host: "/srv/example/cache/left"},
                "port": 8014,
                "exclusive": "big-pair",
            },
            {
                "id": "right-model",
                "title": "Right Model",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "local": {host: "/srv/example/cache/right"},
                "port": 8015,
                "exclusive": "big-pair",
            },
        ],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {"big-pair": ["left-model", "right-model"]},
        "never_touch": {},
    }
    _save({
        "id": "left-run",
        "model_id": "left-model",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "nodes": [],
    })
    _code, planned = plan_model(catalog, None, "right-model", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8015,
        "roles": {"head": _role(args=["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8015"])},
    })
    assert any("left-model (Left Model) is already running" in item for item in planned["conflicts"])
    reset_runs()


def test_single_node_exit_fails_the_run_with_the_log(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()

    class Runner:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "status":
                return {
                    "ok": True,
                    "phase": "running",
                    "log_tail": "loading weights\nboom",
                    "container_status": "exited",
                    "exit_code": 1,
                }
            return {"ok": True, "listening": {}, "http": {}}

    run = {
        "id": "solo1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8010,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 30,
        "health_paths": ["/health"],
        "nodes": [{
            "role": "head",
            "host": HEAD,
            "run_id": "solo1-head",
            "container": "sparksi-demo",
            "stop_argv": [],
            "phase": "started",
        }],
    }
    failed = tick(run, Runner())
    assert failed["phase"] == "failed"
    assert failed["ok"] is False
    assert "exited with code 1" in failed["error"]
    assert "boom" in failed["error"]
    reset_runs()


def test_vllm_reservation_and_measured_peak(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    uma = 121.0
    assert role_need_gib(
        engine="vllm", gpu_mem_util=0.75, uma_gib=uma,
        weights_gib=16, reserve_gib=8, measured_gib=105, placement="both",
    ) == 105 + MEMORY_MARGIN_GIB
    assert role_need_gib(
        engine="llamacpp", gpu_mem_util=None, uma_gib=uma,
        weights_gib=8, reserve_gib=8, measured_gib=22, placement="head",
    ) == 22 + MEMORY_MARGIN_GIB
    host = CFG.head.name
    tight = {
        "library": [{
            "id": "wide",
            "title": "Wide",
            "format": "fp8",
            "gib": 8,
            "engines": ["vllm"],
            "local": {host: "/srv/example/cache/weights"},
        }],
        "hosts": {host: {"uma_gib": 128, "floor_gib": 40, "margin_gib": 4}},
        "never_touch": {},
    }
    _code, vllm = plan_model(tight, None, "wide", {
        "engine": "vllm", "placement": "head", "port": 8010, "ctx": 8192, "gpu_mem_util": 0.75,
    })
    assert any("needs 101.0 GiB" in item for item in vllm["conflicts"]), vllm["conflicts"]
    llama_cat = {
        "library": [{
            "id": "small",
            "title": "Small",
            "format": "gguf",
            "gib": 8,
            "engines": ["llamacpp"],
            "local": {host: "/srv/example/cache/small.gguf"},
        }],
        "hosts": {host: {"uma_gib": 36, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    _code, before = plan_model(llama_cat, None, "small", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert not any("needs 22.0 GiB" in item for item in before["conflicts"])
    _save({
        "id": "busy1",
        "model_id": "other-model",
        "engine": "llamacpp",
        "placement": "head",
        "port": 8020,
        "phase": "ready",
        "started": time.time(),
        "measured_gib": 30,
        "nodes": [],
    })
    held = {
        "library": [{
            "id": "next-model",
            "title": "Next",
            "format": "gguf",
            "gib": 8,
            "engines": ["llamacpp"],
            "local": {host: "/srv/example/cache/next.gguf"},
        }],
        "hosts": {host: {"uma_gib": 50, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    _code, crowded = plan_model(held, None, "next-model", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert any("has 10.0 GiB" in item for item in crowded["conflicts"]), crowded["conflicts"]
    reset_runs()
    record_peak("small", 22)
    _code, after = plan_model(llama_cat, None, "small", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert "margin capped" in (after.get("memory_note") or ""), after
    assert not any("needs 27.0 GiB" in item for item in after["conflicts"]), after["conflicts"]
    reset_memory()
    reset_runs()


def test_repo_and_revision_round_trip_and_name_one_snapshot(tmp_path, monkeypatch) -> None:
    from sparksi.recipes.pinned import snapshot_directory
    from sparksi.recipes.store import load_recipe, save_recipe

    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path))
    revision = "ab" * 20
    save_recipe("example", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "repo": "org/example",
        "revision": revision,
        "roles": {"head": {"runtime": "docker", "image": IMAGE, "args": ["vllm", "serve", "--port", "8014"], "env": {}}},
    })
    loaded = load_recipe("example")
    assert loaded["repo"] == "org/example"
    assert loaded["revision"] == revision
    path = snapshot_directory(loaded["repo"], loaded["revision"])
    assert path.endswith(f"/models--org--example/snapshots/{revision}")
    assert path != "/home/USER/.cache/huggingface"
    assert not path.rstrip("/").endswith("/hub")
    page = open("src/sparksi/static/index.html", encoding="utf-8").read()
    assert 'repo.id = "field-repo"' in page
    assert 'revision.id = "field-revision"' in page
    assert "body.repo" in page and "body.revision" in page


def test_unknown_recipe_keys_are_refused(tmp_path, monkeypatch, caplog) -> None:
    import logging

    from sparksi.recipes.store import load_recipe, recipe_path, save_recipe

    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path))
    try:
        save_recipe("example", {"engine": "vllm", "not_a_field": "dropped"})
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("unknown key was stored")
    assert message == "unknown recipe key not_a_field"
    recipe_path("example").write_text('engine = "vllm"\nnot_a_field = "dropped"\n', encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="sparksi"):
        loaded = load_recipe("example")
    assert loaded.get("engine") == "vllm"
    assert "not_a_field" not in loaded
    assert "unknown keys: not_a_field" in caplog.text


def test_importer_fills_repo_and_revision_from_refs_main(tmp_path, monkeypatch) -> None:
    import json

    from sparksi.recipes.importer import import_model
    from sparksi.recipes.pinned import snapshot_directory
    from sparksi.recipes.store import load_recipe

    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path))
    revision = "cd" * 20
    hub = CFG.path("hf_hub")
    ref = f"{hub}/models--org--example/refs/main"
    inspect = {
        "Name": "/example-vllm",
        "Config": {
            "Image": IMAGE,
            "Entrypoint": None,
            "Cmd": ["vllm", "serve", "--port", "8014"],
            "Env": [],
        },
        "HostConfig": {
            "NetworkMode": "host",
            "Binds": ["/srv/example/cache:/root/.cache/huggingface"],
        },
    }

    class Runner:
        def capture(self, host, argv, timeout=8):
            if list(argv) == ["cat", ref]:
                return {"ok": True, "code": 0, "stdout": revision + "\n", "stderr": ""}
            if list(argv)[:2] == ["docker", "inspect"]:
                return {"ok": True, "code": 0, "stdout": json.dumps(inspect), "stderr": ""}
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}

    catalog = {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "repo": "org/example",
            "local": {HEAD: "/srv/example/cache/weights"},
            "identity": {"container": "example-vllm"},
            "launch": {"host": HEAD},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    result = import_model(catalog, Runner(), "example-vllm", "container:example-vllm", dry_run=False)
    recipe = result["recipe"]
    assert recipe["repo"] == "org/example"
    assert recipe["revision"] == revision
    path = snapshot_directory(recipe["repo"], recipe["revision"])
    assert path.endswith(f"/models--org--example/snapshots/{revision}")
    assert path != "/home/USER/.cache/huggingface"
    saved = load_recipe("example-vllm")
    assert saved["repo"] == "org/example"
    assert saved["revision"] == revision


def test_user_port_is_written_into_the_start_script_env() -> None:
    """A script that reads PORT sees the user port in the environment.

    ``--port`` after the script is ignored by that script, so it is not kept.
    """
    from sparksi.recipes.pinned import apply_overrides, role_argv, validate_pinned

    recipe = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "imported_from": "container:example at 2026-11-01T00:00:00Z",
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "container": "sparksi-demo",
                "args": ["bash", "/start.sh", "--port", "8013"],
                "env": {"PORT": "8013", "VLLM_PORT": "8013", "MASTER_PORT": "29501", "NCCL_PORT": "8013"},
            }
        },
    }
    updated = apply_overrides(recipe, {"port": 8014}, {"port"})
    head = updated["roles"]["head"]
    assert updated["port"] == 8014
    assert head["env"]["PORT"] == "8014"
    assert head["env"]["VLLM_PORT"] == "8014"
    assert head["env"]["MASTER_PORT"] == "29501"
    assert head["env"]["NCCL_PORT"] == "8013"
    assert head["args"] == ["bash", "/start.sh"]
    argv = role_argv(head, "100.64.0.11")
    assert "PORT=8014" in argv
    assert "--port" not in argv
    stale = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "args": ["bash", "/start.sh", "--port", "8014"],
                "env": {"PORT": "8013"},
                "network": "host",
            }
        },
    }
    notes = validate_pinned(stale)
    assert "the recipe also sets PORT=8013; the model would listen there" in notes


def test_script_entrypoint_skips_trailing_port_and_context_flags() -> None:
    """A script that reads PORT does not gain ``--port``, ``--host``, or ``--max-model-len``.

    A command entrypoint still receives those flags. A script that already
    accepts trailing args keeps ``--max-model-len``. ``MAX_MODEL_LEN`` is
    updated only when the recipe already documents it.
    """
    from sparksi.recipes.pinned import apply_overrides

    script = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "entrypoint": "bash",
                "container": "sparksi-demo",
                "args": ["/start.sh", "--host", "0.0.0.0", "--port", "8013", "--max-model-len", "4096"],
                "env": {"PORT": "8013", "SERVER_PORT": "8013", "MAX_MODEL_LEN": "4096", "MASTER_PORT": "29501"},
            }
        },
    }
    updated = apply_overrides(script, {"port": 8015, "ctx": 8192}, {"port", "ctx"})
    head = updated["roles"]["head"]
    assert head["args"] == ["/start.sh"]
    assert "--port" not in head["args"]
    assert "--host" not in head["args"]
    assert "--max-model-len" not in head["args"]
    assert head["env"]["PORT"] == "8015"
    assert head["env"]["SERVER_PORT"] == "8015"
    assert head["env"]["MAX_MODEL_LEN"] == "8192"
    assert head["env"]["MASTER_PORT"] == "29501"

    plain = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "entrypoint": "bash",
                "args": ["/start.sh"],
                "env": {"PORT": "8013"},
            }
        },
    }
    kept = apply_overrides(plain, {"port": 8015, "ctx": 8192}, {"port", "ctx"})
    assert kept["roles"]["head"]["args"] == ["/start.sh"]
    assert "MAX_MODEL_LEN" not in kept["roles"]["head"]["env"]
    assert kept["roles"]["head"]["env"]["PORT"] == "8015"

    forwarded = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "entrypoint": "bash",
                "args": ["/start.sh", "--mode", "fast"],
                "env": {"PORT": "8013"},
            }
        },
    }
    passed = apply_overrides(forwarded, {"port": 8015, "ctx": 8192}, {"port", "ctx"})
    args = passed["roles"]["head"]["args"]
    assert args[:3] == ["/start.sh", "--mode", "fast"]
    assert "--port" not in args
    assert "--host" not in args
    assert args[args.index("--max-model-len") + 1] == "8192"

    command = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8013,
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "args": ["vllm", "serve", "/models/example"],
                "env": {"PORT": "8013"},
            }
        },
    }
    server = apply_overrides(command, {"port": 8015, "ctx": 8192}, {"port", "ctx"})
    argv = server["roles"]["head"]["args"]
    assert argv[argv.index("--port") + 1] == "8015"
    assert argv[argv.index("--max-model-len") + 1] == "8192"


def test_script_context_defaults_to_the_recipe_env(monkeypatch, tmp_path) -> None:
    """A script's MAX_MODEL_LEN is the context default until the user picks another."""
    from sparksi.recipes.defaults import _pinned_defaults
    from sparksi.recipes.pinned import apply_overrides
    from sparksi.recipes.service import plan_model
    from sparksi.recipes.store import save_recipe

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    recipe = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z",
        "roles": {
            "head": {
                "runtime": "docker",
                "image": IMAGE,
                "network": "host",
                "entrypoint": "bash",
                "args": ["/start.sh"],
                "env": {"PORT": "8014", "MAX_MODEL_LEN": "850000"},
            }
        },
    }
    defaults = _pinned_defaults(recipe, 1048576)
    assert defaults["values"]["ctx"] == 850000
    assert defaults["values"]["ctx"] != 1048576
    untouched = apply_overrides(recipe, {"port": 8015}, {"port"})
    assert untouched["roles"]["head"]["env"]["MAX_MODEL_LEN"] == "850000"
    assert "--max-model-len" not in untouched["roles"]["head"]["args"]
    assert "1048576" not in str(untouched["roles"]["head"]["env"])
    changed = apply_overrides(recipe, {"ctx": 16384}, {"ctx"})
    assert changed["roles"]["head"]["env"]["MAX_MODEL_LEN"] == "16384"
    save_recipe("demo", recipe)
    catalog = _catalog("demo", title="Example Flash")
    _code, planned = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8014})
    assert planned["defaults"]["values"]["ctx"] == 850000
    rendered = planned.get("draft") or planned.get("recipe") or {}
    env = ((rendered.get("nodes") or [{}])[0].get("env") or {})
    assert env.get("MAX_MODEL_LEN") == "850000"
    assert "1048576" not in str(env)
    reset_runs()


def test_health_reports_listening_on_the_script_port(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()

    class Runner:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "status":
                return {"ok": True, "phase": "running", "log_tail": "", "container_status": "running"}
            ports = [str(item) for item in (req.get("ports") or [])]
            listening = {port: port == "8013" for port in ports}
            return {"ok": True, "listening": listening, "http": {}, "mem_total_gib": 128, "mem_available_gib": 40}

    run = {
        "id": "portmix",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 1800,
        "health_paths": ["/health"],
        "nodes": [{
            "role": "head",
            "host": HEAD,
            "run_id": "portmix-head",
            "container": "sparksi-demo",
            "argv": ["bash", "/start.sh", "--port", "8014"],
            "env": {"PORT": "8013", "MASTER_PORT": "29501"},
            "stop_argv": [],
            "phase": "started",
        }],
    }
    failed = tick(run, Runner())
    assert failed["phase"] == "failed"
    assert failed["error"] == "the model is listening on 8013, not 8014"
    reset_runs()


def test_loading_503_waits_and_other_5xx_fails() -> None:
    from sparksi.jobs import _health_verdict

    class Probe:
        def __init__(self, code, listening=True):
            self.code = code
            self.listening = listening

        def invoke(self, host, req, timeout=20):
            port = str((req.get("ports") or [0])[0])
            http = {}
            if self.listening and self.code is not None:
                http[port] = {"status": self.code}
            return {"ok": True, "listening": {port: self.listening}, "http": http}

    assert _health_verdict(Probe(503), HEAD, 8010, ["/health"]) == "wait"
    assert _health_verdict(Probe(None, listening=False), HEAD, 8010, ["/health"]) == "refused"
    assert _health_verdict(Probe(500), HEAD, 8010, ["/health"]) == "fail"
    assert _health_verdict(Probe(200), HEAD, 8010, ["/health"]) == "ready"


def test_missing_container_and_repeated_refusal_fail_the_run(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.runs import REFUSED_PROBES

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()

    class Runner:
        def __init__(self):
            self.missing = False
            self.listening = True

        def invoke(self, host, req, timeout=20):
            if req.get("op") == "status":
                if self.missing:
                    return {"ok": True, "log_tail": "gone", "container_status": "missing"}
                return {"ok": True, "phase": "running", "log_tail": "up", "container_status": "running"}
            port = str((req.get("ports") or [""])[0]) if req.get("ports") else ""
            listening = {port: self.listening} if port else {}
            http = {port: {"status": 200}} if port and self.listening else {}
            return {"ok": True, "listening": listening, "http": http, "mem_total_gib": 128, "mem_available_gib": 40}

    runner = Runner()
    run = {
        "id": "solo2",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8010,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 30,
        "health_paths": ["/health"],
        "measured_by_role": {},
        "nodes": [{
            "role": "head",
            "host": HEAD,
            "run_id": "solo2-head",
            "container": "sparksi-demo",
            "stop_argv": [],
            "phase": "started",
            "mem_baseline_gib": 80,
        }],
    }
    runner.missing = True
    missing = tick(run, runner)
    assert missing["phase"] == "failed"
    assert "container missing" in missing["error"]
    assert "gone" in missing["error"]
    assert missing["measured_by_role"]["head"] == 8

    reset_runs()
    runner.missing = False
    runner.listening = True
    alive = dict(run)
    alive["phase"] = "waiting_health"
    alive["nodes"] = [dict(run["nodes"][0])]
    ready = tick(alive, runner)
    assert ready["phase"] == "ready"
    runner.listening = False
    current = ready
    for _ in range(REFUSED_PROBES - 1):
        current = tick(current, runner)
        assert current["phase"] == "ready"
    failed = tick(current, runner)
    assert failed["phase"] == "failed"
    assert "health refused" in failed["error"]
    reset_runs()


def test_shell_script_entrypoint_is_allowed_only_after_the_bind_check(tmp_path, monkeypatch) -> None:
    from sparksi.agent import _refuse_recipe_argv
    from sparksi.recipes.engines.vllm import build as build_vllm
    from sparksi.recipes.importer import parse_docker_inspect
    from sparksi.recipes.spec import Spec

    captured = parse_docker_inspect({
        "Name": "/example",
        "Config": {
            "Image": IMAGE,
            "Entrypoint": ["vllm"],
            "Cmd": ["bash", "/start.sh"],
            "Env": [],
        },
        "HostConfig": {"NetworkMode": "host"},
    })
    assert captured["entrypoint"] == "bash"
    assert captured["args"] == ["/start.sh"]

    script = tmp_path / "start.sh"
    script.write_text("#!/bin/sh\nvllm serve /models/example --port 8014\n", encoding="utf-8")
    mount = f"{script}:/start.sh"

    class Reader:
        def capture(self, host, argv, timeout=8):
            if argv[:2] == ["cat", str(script)]:
                return {"ok": True, "code": 0, "stdout": script.read_text(encoding="utf-8"), "stderr": ""}
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing"}

    # The host path is outside the mount roots, so the script is refused.
    assert any("not under a mount root" in item for item in validate_pinned(_recipe("head", **{
        "entrypoint": "bash", "args": ["/start.sh"], "mounts": [mount],
    }), runner=Reader()))

    from sparksi.agent import configure
    from sparksi.recipes import pinned as pinned_mod
    monkeypatch.setattr(pinned_mod, "roots_for", lambda role: [str(tmp_path), "/srv/example/cache"])
    configure({"mount_roots": [str(tmp_path), "/srv/example/cache"]})
    clean = validate_pinned(_recipe("head", entrypoint="bash", args=["/start.sh", "--host", "100.64.0.11", "--port", "8014"], mounts=[mount]), runner=Reader())
    assert not any("shell entrypoint" in item for item in clean), clean
    inline = validate_pinned(_recipe("head", entrypoint="bash", args=["-c", "vllm serve"], mounts=[mount]), runner=Reader())
    assert any("refusing shell entrypoint bash" in item for item in inline)
    noisy = validate_pinned(_recipe("head", entrypoint="bash", args=["/start.sh", "--port", "8014;rm"], mounts=[mount]), runner=Reader())
    assert any("refusing shell entrypoint bash" in item for item in noisy)

    spec = Spec(
        model_id="demo", placement="head", port=8010,
        paths={"head": "/models/demo", "worker": "/models/demo"},
        hosts={"head": "head", "worker": "worker"},
        binds={"head": "100.64.0.11", "worker": "100.64.0.12"},
        fabric_ip={"head": "192.0.2.11", "worker": "192.0.2.12"},
        fabric_iface={"head": "eth0", "worker": "eth0"},
        fabric_hcas={"head": ["roce0"], "worker": ["roce0"]},
        image=IMAGE, hf_cache=str(tmp_path), ctx=8192, hints={"mtp": False}, scripts={},
    )
    base = build_vllm(spec).nodes[0].argv
    image_at = base.index(IMAGE)
    argv = base[:image_at] + ["-v", mount, "--entrypoint", "bash", IMAGE, "/start.sh", "--port", "8014"]
    assert _refuse_recipe_argv(argv) is None, _refuse_recipe_argv(argv)
    refused = _refuse_recipe_argv(base[:image_at] + ["--entrypoint", "bash", IMAGE, "vllm", "serve"])
    assert refused and "shell entrypoint" in refused
    configure({"mount_roots": ["/home/USER/.cache/huggingface"]})


def test_evict_helper_mismatch_warns_and_does_not_block(monkeypatch, tmp_path) -> None:
    from sparksi.evict import module_sha256
    from sparksi.recipes.runs import start_rendered

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    monkeypatch.setattr("sparksi.recipes.runs.threading.Thread.start", lambda self: None)

    class Runner:
        def __init__(self, digest):
            self.digest = digest
            self.started = False

        def invoke(self, host, req, timeout=20):
            if req.get("op") == "evict_sha256":
                return {"ok": True, "sha256": self.digest}
            if req.get("op") == "probe":
                return {"ok": True, "mem_total_gib": 128, "mem_available_gib": 100}
            if req.get("op") == "run":
                self.started = True
                return {"ok": True, "phase": "running"}
            return {"ok": True}

        def run_argv(self, host, argv, timeout=60):
            return {"ok": True, "code": 0}

    bad = Runner("0" * 64)
    recipe = {
        "id": "example",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "timeout_s": 30,
        "health_paths": [],
        "order": ["head"],
        "nodes": [{
            "role": "head",
            "host": HEAD,
            "argv": ["docker", "run", "-d", "--network", "host", IMAGE, "vllm", "serve", "--port", "8014"],
            "env": {},
            "container": "sparksi-example",
            "stop_argv": ["docker", "rm", "-f", "sparksi-example"],
            "evict_page_cache": True,
            "evict_path": "/srv/example/cache/weights",
        }],
    }
    run = start_rendered(recipe, bad, sync=False)
    assert bad.started is True
    assert run["phase"] != "failed"
    assert any("does not match the controller" in item for item in run.get("warnings") or [])
    same = Runner(module_sha256())
    quiet = start_rendered(recipe, same, sync=False)
    assert not any("does not match" in item for item in quiet.get("warnings") or [])
    reset_runs()


def test_live_measured_use_blocks_a_pair_inside_the_margin(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    host = CFG.head.name
    worker = CFG.worker.name
    _save({
        "id": "live-pair",
        "model_id": "already",
        "engine": "vllm",
        "placement": "both",
        "port": 8020,
        "phase": "ready",
        "started": time.time(),
        "measured_by_role": {"head": 100, "worker": 100},
        "nodes": [],
    })
    record_node_peak("next-pair", "head", 105)
    record_node_peak("next-pair", "worker", 105)
    catalog = {
        "library": [{
            "id": "next-pair",
            "title": "Next",
            "format": "fp8",
            "gib": 8,
            "engines": ["vllm"],
            "local": {host: "/srv/example/cache/a", worker: "/srv/example/cache/a"},
        }],
        "hosts": {
            host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
            worker: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
        },
        "never_touch": {},
    }
    _code, planned = plan_model(catalog, None, "next-pair", {
        "engine": "vllm", "placement": "both", "port": 8010, "ctx": 8192, "gpu_mem_util": 0.75,
    })
    assert any("over budget" in item for item in planned["conflicts"]), planned["conflicts"]
    reset_runs()
    reset_memory()


def test_stop_wins_a_race_with_a_slow_probe(monkeypatch, tmp_path) -> None:
    import threading

    from sparksi.recipes.runs import stop_recorded
    from sparksi.recipes.service import _model_running

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    entered = threading.Event()
    release = threading.Event()

    class Runner:
        def invoke(self, host, req, timeout=20):
            if req.get("op") == "status":
                entered.set()
                release.wait(5)
                return {"ok": True, "log_tail": "No such container", "container_status": "missing"}
            if req.get("op") == "stop":
                return {"ok": True}
            return {"ok": True, "listening": {}, "http": {}, "containers": []}

    run = _save({
        "id": "race1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8010,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 30,
        "health_paths": ["/health"],
        "version": 1,
        "nodes": [{
            "role": "head",
            "host": HEAD,
            "run_id": "race1-head",
            "container": "sparksi-demo",
            "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
            "phase": "started",
        }],
    })
    runner = Runner()
    box: dict[str, dict] = {}

    def monitor() -> None:
        box["run"] = tick(run, runner)

    worker = threading.Thread(target=monitor)
    worker.start()
    assert entered.wait(2)
    stopped = stop_recorded(run, runner)
    assert stopped["phase"] == "stopped"
    assert stopped["ok"] is True
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    final = box["run"]
    assert final["phase"] == "stopped"
    assert final.get("error") in (None, "")
    assert not _model_running("demo")
    reset_runs()


def test_requested_stop_is_not_a_container_failure(monkeypatch, tmp_path) -> None:
    import threading

    from sparksi.recipes.runs import stop_recorded

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()

    def race(placement: str, hosts: list[tuple[str, str]]) -> dict:
        entered = threading.Event()
        release = threading.Event()

        class Runner:
            def invoke(self, host, req, timeout=20):
                if req.get("op") == "status":
                    entered.set()
                    release.wait(5)
                    return {"ok": True, "log_tail": "No such container", "container_status": "missing"}
                if req.get("op") == "stop":
                    return {"ok": True}
                names = [] if host == WORKER else ["sparksi-demo"]
                return {"ok": True, "listening": {}, "http": {}, "containers": names}

        nodes = []
        for role, host in hosts:
            nodes.append({
                "role": role,
                "host": host,
                "run_id": f"stop-{role}",
                "container": "sparksi-demo",
                "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
                "phase": "started",
            })
        run = _save({
            "id": f"stop-{placement}",
            "model_id": "demo",
            "engine": "vllm",
            "placement": placement,
            "port": 8010,
            "phase": "waiting_health",
            "started": time.time(),
            "timeout_s": 30,
            "health_paths": ["/health"],
            "seen": {"worker": True, "head": True},
            "version": 1,
            "nodes": nodes,
        })
        box: dict[str, dict] = {}

        def monitor() -> None:
            box["run"] = tick(run, Runner())

        thread = threading.Thread(target=monitor)
        thread.start()
        assert entered.wait(2)
        stop_recorded(run, Runner())
        release.set()
        thread.join(5)
        return box["run"]

    single = race("head", [("head", HEAD)])
    assert single["phase"] == "stopped", single
    assert "container missing" not in str(single.get("error") or "")
    pair = race("both", [("worker", WORKER), ("head", HEAD)])
    assert pair["phase"] == "stopped", pair
    assert "failed first" not in str(pair.get("error") or "")
    reset_runs()


def test_worker_disappearing_during_stop_is_not_a_pair_failure(monkeypatch, tmp_path) -> None:
    """A user stop kills the worker first. The health loop ticks before the head stop is finished.

    The failure check must not leave the run failed, including when that tick's save
    lands after phase=stopped was written. Node entries end as stopped.
    """
    import json
    import threading

    from sparksi.recipes.runs import _watch, get_run, stop_recorded, tick

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    up = {"worker": True, "head": True}
    entered = threading.Event()
    worker_killed = threading.Event()
    stop_finished = threading.Event()

    class Runner:
        def __init__(self) -> None:
            self.nested = False

        def invoke(self, host, req, timeout=20):
            role = "worker" if host == WORKER else "head"
            op = req.get("op")
            if op == "status" and role == "worker" and not self.nested:
                entered.set()
                assert worker_killed.wait(5)
                # Another health pass runs while this one is still in the worker probe,
                # after the worker container is gone and before the outer pass finishes.
                self.nested = True
                tick(get_run("both-stop"), self)
                self.nested = False
                assert stop_finished.wait(5)
            if op == "status":
                if up[role]:
                    return {"ok": True, "log_tail": "", "container_status": "running", "phase": "running"}
                return {
                    "ok": True,
                    "log_tail": "killed",
                    "container_status": "missing",
                    "exit_code": 137,
                    "phase": "exited",
                }
            if op == "probe":
                names = ["sparksi-demo"] if up[role] else []
                listening = {"8010": True} if host == HEAD and up["head"] else {}
                http = {"8010": {"status": 200}} if listening else {}
                return {"ok": True, "listening": listening, "http": http, "containers": names}
            if op == "stop":
                up[role] = False
                if role == "worker":
                    worker_killed.set()
                return {"ok": True, "exit_code": 137}
            return {"ok": True, "listening": {}, "http": {}, "containers": []}

    stop = ["docker", "rm", "-f", "sparksi-demo"]
    run = _save({
        "id": "both-stop",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "both",
        "port": 8010,
        "phase": "ready",
        "ok": True,
        "started": time.time(),
        "timeout_s": 1800,
        "health_paths": ["/health"],
        "seen": {"worker": True, "head": True},
        "version": 1,
        "nodes": [
            {
                "role": "worker",
                "host": WORKER,
                "run_id": "both-stop-worker",
                "container": "sparksi-demo",
                "stop_argv": stop,
                "phase": "started",
            },
            {
                "role": "head",
                "host": HEAD,
                "run_id": "both-stop-head",
                "container": "sparksi-demo",
                "stop_argv": stop,
                "phase": "started",
            },
        ],
    })
    runner = Runner()
    health = threading.Thread(target=_watch, args=(run["id"], runner, 0.05), name="health", daemon=True)
    health.start()
    assert entered.wait(2)
    stopped = stop_recorded(run, runner)
    stop_finished.set()
    health.join(5)
    assert not health.is_alive()
    final = get_run("both-stop")
    disk = json.loads((tmp_path / "recipe-runs" / "both-stop.json").read_text(encoding="utf-8"))
    assert stopped["phase"] == "stopped", stopped
    assert final["phase"] == "stopped", final
    assert final.get("ok") is True
    assert not final.get("error")
    assert final.get("failed_side") in (None, "")
    assert "failed first" not in str(final.get("error") or "")
    assert [node.get("phase") for node in final["nodes"]] == ["stopped", "stopped"]
    assert disk["phase"] == "stopped"
    assert not disk.get("error")
    assert [node.get("phase") for node in disk["nodes"]] == ["stopped", "stopped"]
    reset_runs()


def test_container_memory_is_not_charged_for_a_neighbor(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.memory import consider_peak, node_peak_gib
    from sparksi.recipes.runs import _note_memory

    assert consider_peak(21, 72.7, solo=False) is None
    assert consider_peak(21, 25, solo=False) == 25
    assert consider_peak(21, 72.7, solo=True) == 72.7
    assert consider_peak(None, 21, solo=False) == 21

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    neighbor = _save({
        "id": "other",
        "model_id": "other",
        "engine": "vllm",
        "placement": "head",
        "port": 8020,
        "phase": "ready",
        "started": time.time(),
        "nodes": [{"role": "head", "host": HEAD, "phase": "started"}],
    })
    assert neighbor["id"] == "other"
    run = {
        "id": "demo-run",
        "model_id": "demo",
        "measured_gib": 0,
        "measured_by_role": {},
        "nodes": [],
    }
    node = {"role": "head", "host": HEAD, "mem_baseline_gib": 10, "measured_gib": 21}
    _note_memory(run, node, used=82)
    assert "head" not in run["measured_by_role"]
    assert node_peak_gib("demo", "head") is None
    node["container_gib"] = 21.5
    _note_memory(run, node, used=82)
    assert run["measured_by_role"]["head"] == 21.5
    node["container_gib"] = 72.7
    _note_memory(run, node, used=82)
    assert run["measured_by_role"]["head"] == 21.5
    neighbor["phase"] = "stopped"
    _save(neighbor)
    reset_runs()
    reset_memory()
    alone = {"id": "solo", "model_id": "solo", "measured_gib": 0, "measured_by_role": {}, "nodes": []}
    solo_node = {"role": "head", "host": HEAD, "mem_baseline_gib": 10}
    _note_memory(alone, solo_node, used=31)
    assert alone["measured_by_role"]["head"] == 21
    assert node_peak_gib("solo", "head") == 21
    reset_runs()
    reset_memory()


def test_unified_memory_peak_uses_the_host_delta_when_alone(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.memory import node_peak_gib, record_node_peak
    from sparksi.recipes.runs import _note_memory

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    _save({
        "id": "neighbor",
        "model_id": "neighbor",
        "engine": "vllm",
        "placement": "head",
        "port": 8020,
        "phase": "ready",
        "started": time.time(),
        "nodes": [{"role": "head", "host": HEAD}],
    })
    shared = {
        "id": "wide-run",
        "model_id": "wide",
        "reservation_gib": 110,
        "measured_gib": 0,
        "measured_by_role": {},
        "nodes": [],
    }
    low = {"role": "head", "host": HEAD, "container_gib": 11, "mem_baseline_gib": 10}
    _note_memory(shared, low, used=130)
    assert "head" not in shared["measured_by_role"]
    assert node_peak_gib("wide", "head") is None
    record_node_peak("wide", "head", 112)
    low["measured_gib"] = 112
    low["container_gib"] = 9
    _note_memory(shared, low, used=130)
    assert low["measured_gib"] == 112
    assert node_peak_gib("wide", "head") == 112
    reset_runs()
    reset_memory()
    solo = {
        "id": "solo-run",
        "model_id": "solo-model",
        "measured_gib": 0,
        "measured_by_role": {},
        "nodes": [],
    }
    solo_node = {"role": "head", "host": HEAD, "container_gib": 11, "mem_baseline_gib": 10}
    _note_memory(solo, solo_node, used=113)
    assert solo["measured_by_role"]["head"] == 103
    assert node_peak_gib("solo-model", "head") == 103
    reset_runs()
    reset_memory()


def test_live_charge_uses_peak_and_host_delta(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.memory import record_node_peak
    from sparksi.recipes.service import _live_usage, _role_charge

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    host = CFG.head.name
    record_node_peak("resident", "head", 112)
    run = _save({
        "id": "resident-run",
        "model_id": "resident",
        "engine": "llamacpp",
        "placement": "head",
        "port": 8021,
        "phase": "ready",
        "started": time.time(),
        "nodes": [{
            "role": "head",
            "host": host,
            "container_gib": 11,
            "mem_baseline_gib": 8,
            "host_used_gib": 120,
        }],
    })
    assert _role_charge(run, "head", {}) == 112
    _ports, held = _live_usage("someone-else", {})
    assert held["head"] == 112
    catalog = {
        "library": [{
            "id": "next-small",
            "title": "Next Small",
            "format": "gguf",
            "gib": 8,
            "engines": ["llamacpp"],
            "local": {host: "/srv/example/cache/next.gguf"},
        }],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "never_touch": {},
    }
    _code, planned = plan_model(catalog, None, "next-small", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert any("over budget" in item for item in planned["conflicts"]), planned["conflicts"]
    assert "margin capped" not in (planned.get("memory_note") or "")
    reset_runs()
    reset_memory()


def test_solo_peak_caps_the_margin_and_a_shared_node_keeps_it(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.memory import record_node_peak

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    host = CFG.head.name

    def catalog() -> dict:
        return {
            "library": [{
                "id": "fitted",
                "title": "Fitted",
                "format": "gguf",
                "gib": 8,
                "engines": ["llamacpp"],
                "local": {host: "/srv/example/cache/fitted.gguf"},
            }],
            "hosts": {host: {"uma_gib": 50, "floor_gib": 6, "margin_gib": 4}},
            "never_touch": {},
        }

    record_node_peak("fitted", "head", 20)
    _code, fits = plan_model(catalog(), None, "fitted", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert "margin capped" in (fits.get("memory_note") or ""), fits
    assert not any("over budget" in item for item in fits["conflicts"]), fits["conflicts"]

    reset_memory()
    record_node_peak("fitted", "head", 50)
    _code, too_big = plan_model(catalog(), None, "fitted", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert any("over budget" in item for item in too_big["conflicts"]), too_big["conflicts"]
    assert "margin capped" not in (too_big.get("memory_note") or "")

    reset_memory()
    record_node_peak("fitted", "head", 20)
    _save({
        "id": "other-run",
        "model_id": "other-model",
        "engine": "llamacpp",
        "placement": "head",
        "port": 8022,
        "phase": "ready",
        "started": time.time(),
        "measured_gib": 18,
        "nodes": [],
    })
    _code, shared = plan_model(catalog(), None, "fitted", {
        "engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192,
    })
    assert any("needs 25.0 GiB" in item for item in shared["conflicts"]), shared["conflicts"]
    assert "margin capped" not in (shared.get("memory_note") or "")
    reset_runs()
    reset_memory()


def test_solo_peak_above_the_budget_uses_live_free_memory(monkeypatch, tmp_path) -> None:
    """An idle node admits a solo peak above the budget when free memory covers it."""
    import json

    from sparksi.recipes import memory as memory
    from sparksi.recipes.service import reset_facts_cache

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_memory()
    host = CFG.head.name
    (tmp_path / "recipe-memory.json").write_text(
        json.dumps({"above": {"head": 50}}),
        encoding="utf-8",
    )
    memory._LOADED = False
    assert memory.node_peak_is_solo("above", "head") is True
    assert memory.node_peak_gib("above", "head") == 50

    def catalog() -> dict:
        return {
            "library": [
                {
                    "id": "above",
                    "title": "Above",
                    "format": "gguf",
                    "gib": 8,
                    "engines": ["llamacpp"],
                    "local": {host: "/srv/example/cache/above.gguf"},
                },
                {
                    "id": "plain",
                    "title": "Plain",
                    "format": "fp8",
                    "gib": 8,
                    "engines": ["vllm"],
                    "local": {host: "/srv/example/cache/plain"},
                },
            ],
            "hosts": {host: {"uma_gib": 50, "floor_gib": 6, "margin_gib": 4}},
            "never_touch": {},
        }

    class Runner:
        def __init__(self, free: float) -> None:
            self.free = free

        def invoke(self, host, req, timeout=20):
            paths = {str(path): {"exists": True} for path in (req.get("paths") or [])}
            return {
                "ok": True,
                "paths": paths,
                "mem_total_gib": 50,
                "mem_available_gib": self.free,
                "listening": {},
                "http": {},
            }

        def capture(self, host, argv, timeout=8):
            return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": None}

    def plan(model_id: str, body: dict, free: float):
        reset_facts_cache()
        return plan_model(catalog(), Runner(free), model_id, body)

    above = {"engine": "llamacpp", "placement": "head", "port": 8010, "ctx": 8192}
    code, ready = plan("above", above, 60)
    assert code == 200, ready.get("conflicts")
    assert ready["verdict"] == "READY"
    assert "admitted on measured solo peak" in (ready.get("memory_note") or "")

    code, tight = plan("above", above, 40)
    assert code == 400, tight
    assert tight["verdict"] == "REFUSE"
    assert any("over budget" in item for item in tight["conflicts"]), tight["conflicts"]
    assert "admitted on measured solo peak" not in (tight.get("memory_note") or "")

    _save({
        "id": "other-run",
        "model_id": "other-model",
        "engine": "llamacpp",
        "placement": "head",
        "port": 8022,
        "phase": "ready",
        "started": time.time(),
        "measured_gib": 18,
        "nodes": [],
    })
    code, shared = plan("above", above, 60)
    assert code == 400, shared
    assert any("over budget" in item for item in shared["conflicts"]), shared["conflicts"]
    assert "admitted on measured solo peak" not in (shared.get("memory_note") or "")
    reset_runs()

    code, estimate = plan("plain", {
        "engine": "vllm", "placement": "head", "port": 8012, "ctx": 8192, "gpu_mem_util": 0.9,
    }, 60)
    assert any("needs 50.0 GiB" in item for item in estimate["conflicts"]), estimate["conflicts"]
    assert "admitted on measured solo peak" not in (estimate.get("memory_note") or "")
    reset_runs()
    reset_memory()
    reset_facts_cache()


def test_exclusive_conflict_names_the_running_recipe(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    host = CFG.head.name
    catalog = {
        "library": [
            {
                "id": "first-model",
                "title": "First Recipe",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "local": {host: "/srv/example/cache/first"},
                "port": 8014,
                "exclusive": "big-pair",
            },
            {
                "id": "second-model",
                "title": "Second Recipe",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "local": {host: "/srv/example/cache/second"},
                "port": 8014,
                "exclusive": "big-pair",
            },
            {
                "id": "third-model",
                "title": "Third Recipe",
                "format": "fp8",
                "gib": 8,
                "engines": ["vllm"],
                "local": {host: "/srv/example/cache/third"},
                "port": 8016,
                "exclusive": "big-pair",
            },
        ],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {"big-pair": ["first-model", "second-model", "third-model"]},
        "never_touch": {},
    }
    _save({
        "id": "second-run",
        "model_id": "second-model",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "phase": "ready",
        "started": time.time(),
        "nodes": [],
    })
    _code, planned = plan_model(catalog, None, "third-model", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8016,
        "roles": {"head": _role(args=["vllm", "serve", "/models/example", "--port", "8016"])},
    })
    text = " ".join(planned["conflicts"])
    assert "second-model (Second Recipe) is already running" in text
    assert "First Recipe" not in text
    reset_runs()
