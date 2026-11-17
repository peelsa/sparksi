# SPDX-License-Identifier: Apache-2.0

"""Importer fixes: source choice, scripts, compose units, worker inspect, env, ports."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sparksi.config import CFG
from sparksi.recipes.importer import (
    ENV_VALUE_ALLOW,
    env_value_allowed,
    image_env,
    import_all,
    import_model,
    parse_env_file,
    parse_script,
    parse_script_compose,
    parse_unit,
    without_image_env,
)
from sparksi.recipes.store import recipe_path, save_recipe, value_looks_like_token
from sparksi.recipes.pinned import FABRIC_ONLY, fabric_only, apply_token_cache, role_argv, validate_pinned

ROOT = Path(__file__).resolve().parent / "fixtures" / "import"
IMAGE = "example.invalid/vllm:patched"


def _inspect(name: str, cmd: list[str], env: list[str], *, mounts: list[str] | None = None) -> str:
    body = {
        "Name": f"/{name}",
        "Config": {"Image": IMAGE, "Entrypoint": None, "Cmd": cmd, "Env": env},
        "HostConfig": {
            "NetworkMode": "host",
            "Privileged": False,
            "CapAdd": None,
            "SecurityOpt": None,
            "IpcMode": "host",
            "ShmSize": 17179869184,
            "DeviceRequests": [{"Driver": "nvidia", "Capabilities": [["gpu"]]}],
            "Devices": [],
            "Binds": mounts or ["/srv/example/cache:/root/.cache/huggingface"],
        },
        "Mounts": [],
    }
    return json.dumps([body])


def _image(env: list[str]) -> str:
    return json.dumps([{"Config": {"Env": env}}])


class Runner:
    def __init__(self, answers: list) -> None:
        self.answers = answers
        self.calls: list[tuple[str, list[str]]] = []

    def capture(self, host, argv, timeout=8):
        self.calls.append((str(host), list(argv)))
        for want_host, want_argv, value in self.answers:
            if list(argv) == list(want_argv) and want_host in ("*", host):
                return value
        return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}

    def set(self, host: str, argv: list[str], value: dict) -> None:
        self.answers = [item for item in self.answers if not (item[0] == host and list(item[1]) == argv)]
        self.answers.append((host, argv, value))


def _ok(stdout: str) -> dict:
    return {"ok": True, "code": 0, "stdout": stdout, "stderr": ""}


def test_script_expands_variables_and_strips_setsid() -> None:
    native = parse_script((ROOT / "llama-setsid.sh").read_text(encoding="utf-8"))
    assert native is not None
    assert native["binary"] == "/srv/example/builds/llama/bin/llama-server"
    assert native["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/lib"
    assert native["env"]["MODEL_ALIAS"] == "example"
    assert "BIN" not in native["env"]
    assert "--port" in native["args"]
    assert native["args"][native["args"].index("--port") + 1] == "8014"

    chained = parse_script((ROOT / "llama-exec-setsid.sh").read_text(encoding="utf-8"))
    assert chained is not None
    assert chained["binary"] == "/srv/example/builds/llama/bin/llama-server"
    assert chained["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/lib"
    with pytest.raises(ValueError, match="unresolved shell variable BIN"):
        parse_script((ROOT / "llama-unresolved.sh").read_text(encoding="utf-8"))


def test_container_is_preferred_over_a_start_script() -> None:
    head = _inspect(
        "example-vllm",
        ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
        ["HF_HUB_OFFLINE=1"],
    )
    catalog = {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "identity": {"container": "example-vllm"},
            "launch": {"host": "head", "start": ["/srv/example/bin/start.sh"], "stop": ["/srv/example/bin/stop.sh"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    present = Runner([
        ("*", ["docker", "inspect", "example-vllm"], _ok(head)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(["PATH=/usr/bin"]))),
    ])
    result = import_model(catalog, present, "example-vllm", dry_run=True)
    assert result["conflicts"] == [], result["conflicts"]
    assert result["recipe"]["imported_from"].startswith("container:example-vllm")
    assert present.calls[0][1][:2] == ["docker", "inspect"]
    assert not any(argv[0] == "cat" for _host, argv in present.calls)

    absent = Runner([
        ("*", ["docker", "inspect", "example-vllm"], {"ok": False, "code": 1, "stdout": "", "error": "no such container"}),
        ("*", ["cat", "/srv/example/bin/start.sh"], _ok((ROOT / "llama-start.sh").read_text(encoding="utf-8"))),
    ])
    fallen = import_model(catalog, absent, "example-vllm", dry_run=True)
    assert fallen["recipe"]["engine"] == "llamacpp"
    assert fallen["recipe"]["imported_from"].startswith("script:")
    assert any(argv[:2] == ["docker", "inspect"] for _host, argv in absent.calls)
    assert any(argv[0] == "cat" for _host, argv in absent.calls)


def test_unit_wrapper_finds_compose_from_the_working_directory() -> None:
    unit = (ROOT / "unit-wrapper.service").read_text(encoding="utf-8")
    script = (ROOT / "compose-up.sh").read_text(encoding="utf-8")
    assert parse_unit(unit)["working_directory"] == "/srv/example/stack"
    assert parse_script_compose(script)["bare"] == "1"
    assert parse_script_compose(script)["directory"] == "/srv/example/other"
    compose = (ROOT / "compose-config.json").read_text(encoding="utf-8")
    catalog = {
        "library": [{
            "id": "example-fold",
            "title": "Example",
            "format": "fp8",
            "placement": "pair",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights", "worker": "/srv/example/cache/weights"},
            "launch": {"host": "head", "start": ["sudo", "systemctl", "start", "example-fold"], "stop": ["sudo", "systemctl", "stop", "example-fold"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner([
        ("*", ["systemctl", "cat", "example-fold.service"], _ok(unit)),
        ("*", ["cat", "/srv/example/stack/up.sh"], _ok(script)),
        ("*", ["test", "-f", "/srv/example/stack/docker-compose.yml"], {"ok": False, "code": 1, "stdout": "", "error": "missing"}),
        ("*", ["test", "-f", "/srv/example/stack/compose.yaml"], _ok("")),
        ("*", ["docker", "compose", "-f", "/srv/example/stack/compose.yaml", "config", "--format", "json"], _ok(compose)),
    ])
    result = import_model(catalog, runner, "example-fold", dry_run=True)
    assert result["conflicts"] == [], result["conflicts"]
    assert result["recipe"]["placement"] == "both"
    assert set(result["recipe"]["roles"]) == {"head", "worker"}
    assert any(argv[-1] == "/srv/example/stack/compose.yaml" for _host, argv in runner.calls)
    assert not any("/srv/example/other" in part for _host, argv in runner.calls for part in argv)


def test_worker_container_is_inspected_on_the_worker(monkeypatch) -> None:
    saved = copy.deepcopy(CFG.raw)
    source = CFG.source
    raw = copy.deepcopy(saved)
    raw["nodes"][0]["id"] = "head"
    raw["nodes"][0]["name"] = "head"
    raw["nodes"][1]["id"] = "worker"
    raw["nodes"][1]["name"] = "worker"
    CFG.__init__(raw, source=source)
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", "/tmp/sparksi-importer-recipes")
    try:
        head = _inspect(
            "example-vllm",
            ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
            ["HF_HUB_OFFLINE=1", "NOISE=hello$HOME"],
            mounts=["/srv/example/cache:/root/.cache/huggingface", "/srv/example/builds:/srv/example/builds"],
        )
        worker = _inspect(
            "example-vllm",
            ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
            ["HF_HUB_OFFLINE=1"],
            mounts=["/srv/example/worker:/opt/fold"],
        )
        baked = _image(["NOISE=hello$HOME", "PATH=/usr/bin"])
        catalog = {
            "library": [{
                "id": "example-pair",
                "title": "Example",
                "format": "fp8",
                "placement": "pair",
                "port": 8014,
                "weights_gib": 1,
                "local": {"head": "/srv/example/cache/weights", "worker": "/srv/example/cache/weights"},
                "identity": {"container": "example-vllm", "worker_container": "example-vllm"},
                "launch": {
                    "host": "head",
                    "start": ["/srv/example/bin/start.sh"],
                    "stop": ["/srv/example/bin/stop.sh"],
                },
            }],
            "host": {"reserved_ports": [8188]},
            "never_touch": {},
        }
        runner = Runner([
            ("head", ["docker", "inspect", "example-vllm"], _ok(head)),
            ("worker", ["docker", "inspect", "example-vllm"], _ok(worker)),
            ("head", ["docker", "image", "inspect", IMAGE], _ok(baked)),
            ("worker", ["docker", "image", "inspect", IMAGE], _ok(baked)),
        ])
        result = import_model(catalog, runner, "example-pair", dry_run=True)
        hosts = [(host, argv) for host, argv in runner.calls if argv[:2] == ["docker", "inspect"]]
        assert hosts == [("head", ["docker", "inspect", "example-vllm"]), ("worker", ["docker", "inspect", "example-vllm"])]
        assert result["recipe"]["placement"] == "both"
        assert result["recipe"]["roles"]["worker"]["image"] == IMAGE
        assert "NOISE" not in result["recipe"]["roles"]["head"]["env"]
        assert result["recipe"]["roles"]["head"]["env"]["HF_HUB_OFFLINE"] == "1"
        assert result["conflicts"] == [], result["conflicts"]
        assert not any(argv[0] == "cat" for _host, argv in runner.calls)
    finally:
        CFG.__init__(saved, source=source)


def test_env_port_is_remapped_and_image_env_is_dropped() -> None:
    baked = ["PATH=/usr/bin", "PROMPT=say$hello", "HOME=/root"]
    container_env = baked + ["PORT=8888", "HF_HUB_OFFLINE=1"]
    assert without_image_env(
        {"PROMPT": "say$hello", "PORT": "8888", "HF_HUB_OFFLINE": "1"},
        image_env(_image(baked)),
    ) == {"PORT": "8888", "HF_HUB_OFFLINE": "1"}
    head = _inspect("example-env", ["bash", "/start.sh"], container_env)
    catalog = {
        "library": [{
            "id": "example-env",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "identity": {"container": "example-env"},
            "launch": {"host": "head", "start": ["docker", "start", "example-env"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner([
        ("*", ["docker", "inspect", "example-env"], _ok(head)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(baked))),
    ])
    result = import_model(catalog, runner, "example-env", dry_run=True)
    assert result["conflicts"] == [], result["conflicts"]
    assert result["recipe"]["port"] == 8010
    assert result["port_note"] == "imported port 8888 remapped to 8010"
    assert result["recipe"]["roles"]["head"]["env"]["PORT"] == "8010"
    assert "PROMPT" not in result["recipe"]["roles"]["head"]["env"]

    off = _inspect("example-env", ["bash", "/start.sh"], ["PORT=8188", "HF_HUB_OFFLINE=1"])
    runner.set("*", ["docker", "inspect", "example-env"], _ok(off))
    refused = import_model(catalog, runner, "example-env", dry_run=True)
    assert any("8188" in item for item in refused["conflicts"])
    assert refused["recipe"]["port"] == 8188
    assert refused["recipe"]["roles"]["head"]["env"]["PORT"] == "8188"
    assert refused["port_note"] == ""


def test_import_collapses_script_port_and_port_env() -> None:
    baked = ["PATH=/usr/bin"]
    head = _inspect(
        "example-exl",
        ["bash", "/start.sh", "--port", "8014"],
        ["PORT=8013", "MASTER_PORT=29501", "HF_HUB_OFFLINE=1"],
    )
    catalog = {
        "library": [{
            "id": "example-exl",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "identity": {"container": "example-exl"},
            "launch": {"host": "head", "start": ["docker", "start", "example-exl"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner([
        ("*", ["docker", "inspect", "example-exl"], _ok(head)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(baked))),
    ])
    result = import_model(catalog, runner, "example-exl", dry_run=True)
    assert result["conflicts"] == [], result["conflicts"]
    recipe = result["recipe"]
    role = recipe["roles"]["head"]
    assert recipe["port"] == 8014
    assert role["env"]["PORT"] == "8014"
    assert role["env"]["MASTER_PORT"] == "29501"
    assert role["args"] == ["bash", "/start.sh"]
    assert "--port" not in role["args"]


def test_import_all_skips_services_and_names_security_flags() -> None:
    head = _inspect(
        "example-vllm",
        ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
        ["HF_HUB_OFFLINE=1"],
    )
    catalog = {
        "library": [
            {
                "id": "example-vllm",
                "title": "Example",
                "format": "fp8",
                "placement": "single",
                "port": 8014,
                "weights_gib": 1,
                "local": {"head": "/srv/example/cache/weights"},
                "identity": {"container": "example-vllm"},
                "launch": {"host": "head", "start": ["docker", "start", "example-vllm"]},
            },
            {
                "id": "video-seat",
                "title": "Video",
                "placement": "single",
                "port": 8188,
                "local": {"head": "/srv/example/cache/video"},
                "launch": {"host": "head", "start": ["/srv/example/bin/video-mode.sh", "video"]},
            },
            {
                "id": "kept-out",
                "title": "Kept",
                "format": "fp8",
                "placement": "single",
                "port": 8015,
                "identity": {"container": "kept-out"},
                "launch": {"host": "head", "start": ["docker", "start", "kept-out"]},
            },
            {
                "id": "web-tool",
                "title": "Web",
                "format": "fp8",
                "placement": "single",
                "port": 8016,
                "launch": {"host": "head", "start": ["/srv/example/web/start.sh"], "stop": ["/srv/example/web/stop.sh"]},
            },
        ],
        "host": {"reserved_ports": [8188]},
        "never_touch": {"do_not_stop": ["kept-out"]},
    }
    runner = Runner([
        ("*", ["docker", "inspect", "example-vllm"], _ok(head)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(["PATH=/usr/bin"]))),
    ])
    rows = import_all(catalog, runner, dry_run=True)
    assert [row["id"] for row in rows] == ["example-vllm"]
    recipe = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example-vllm at 2026-10-31T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": {"head": {
            "runtime": "docker",
            "image": IMAGE,
            "env": {},
            "mounts": ["/srv/example/cache:/root/.cache/huggingface"],
            "network": "host",
            "gpus": "all",
            "container": "example-vllm",
            "args": ["vllm", "serve", "--host", "0.0.0.0", "--port", "8014"],
            "cap_add": ["SYS_ADMIN"],
            "security_opt": ["seccomp=unconfined"],
        }},
    }
    conflicts = validate_pinned(recipe)
    assert "refusing --cap-add SYS_ADMIN" in conflicts
    assert "refusing --security-opt seccomp=unconfined" in conflicts


def test_speculative_model_path_follows_the_bind_mount() -> None:
    saved_roots = list(CFG.head.mount_roots)
    if "/srv/example/models" not in CFG.head.mount_roots:
        CFG.head.mount_roots.append("/srv/example/models")

    def recipe(mounts: list[str], model: str, *, native: bool = False) -> dict:
        if native:
            role = {
                "runtime": "native",
                "binary": "llama-server",
                "args": ["-m", "/srv/example/cache/example.gguf", "--host", "0.0.0.0", "--port", "8014", "--model-draft", model],
                "env": {},
                "mounts": [],
            }
        else:
            role = {
                "runtime": "docker",
                "image": IMAGE,
                "env": {},
                "mounts": mounts,
                "network": "host",
                "gpus": "all",
                "container": "example-vllm",
                "args": ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014", "--speculative-draft-model-path", model],
            }
        return {
            "kind": "pinned",
            "engine": "llamacpp" if native else "vllm",
            "placement": "head",
            "port": 8014,
            "imported_from": "container:example-vllm at 2026-10-31T00:00:00Z",
            "port_note": "",
            "rendezvous_ports": [],
            "roles": {"head": role},
        }

    try:
        assert validate_pinned(recipe(["/srv/example/models/draft:/draft"], "/draft")) == []
        outside = validate_pinned(recipe(["/var/outside/draft:/draft"], "/draft"))
        assert any("under a mount root" in item for item in outside)
        uncovered = validate_pinned(recipe(["/srv/example/cache:/root/.cache/huggingface"], "/draft"))
        assert any("not covered by a mount" in item for item in uncovered)
        assert validate_pinned(recipe([], "/srv/example/builds/draft.gguf", native=True)) == []
    finally:
        CFG.head.mount_roots[:] = saved_roots


def test_script_strips_redirections_and_keeps_exported_library_path() -> None:
    text = (ROOT / "llama-redirect.sh").read_text(encoding="utf-8")
    native = parse_script(text)
    assert native is not None
    assert native["binary"].endswith("/llama-server")
    assert native["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/lib:/usr/local/cuda/lib64"
    joined = " ".join(native["args"])
    assert ">>" not in joined
    assert "2>" not in joined
    assert "&" not in joined
    assert "leftover" not in joined
    assert "HF_TOKEN" not in native["env"]
    assert native["env_pass"] == ["HF_TOKEN"]
    assert "HF_TOKEN" not in json.dumps(native["env"])

    prefix = parse_script(
        "BIN_DIR=/srv/example/builds/llama\n"
        "BIN=$BIN_DIR/bin/llama-server\n"
        'LD_LIBRARY_PATH="$BIN_DIR/lib:/usr/local/cuda/lib64" setsid "$BIN" --port 8014 '
        ">>/srv/example/logs/server.log 2>&1 &\n"
    )
    assert prefix is not None
    assert prefix["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/lib:/usr/local/cuda/lib64"
    assert ">>" not in " ".join(prefix["args"])

    with pytest.raises(ValueError, match="unresolved shell variable HF_TOKEN"):
        parse_script(
            "BIN=/srv/example/builds/llama/bin/llama-server\n"
            'setsid "$BIN" --port 8014 --token "$HF_TOKEN"\n'
        )


def test_nested_parameter_expansion_and_secret_indirection() -> None:
    text = (ROOT / "llama-nested.sh").read_text(encoding="utf-8")
    native = parse_script(text)
    assert native is not None
    assert native["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/bin:/usr/local/cuda/lib64"
    assert "}" not in native["env"]["LD_LIBRARY_PATH"]
    assert native["env"]["NESTED"] == "x"
    assert "HF_TOKEN" not in native["env"]
    assert "_CLI_HF_TOKEN" not in native["env"]
    assert "HF_TOKEN" in native["env_pass"]
    assert "HF_TOKEN" not in json.dumps(native["env"])
    assert "\x1e" not in json.dumps(native)

    bee = parse_script(
        "B=bee\n"
        'export NESTED="${A:-${B:-x}}"\n'
        "BIN=/srv/example/builds/llama/bin/llama-server\n"
        'setsid "$BIN" --port 8014\n'
    )
    assert bee is not None
    assert bee["env"]["NESTED"] == "bee"
    alpha = parse_script(
        "A=alpha\n"
        "B=bee\n"
        'export NESTED="${A:-${B:-x}}"\n'
        "BIN=/srv/example/builds/llama/bin/llama-server\n"
        'setsid "$BIN" --port 8014\n'
    )
    assert alpha is not None
    assert alpha["env"]["NESTED"] == "alpha"

    with pytest.raises(ValueError, match="unbalanced shell parameter expansion"):
        parse_script(
            'export LD_LIBRARY_PATH="${BIN_DIR"\n'
            "BIN=/srv/example/builds/llama/bin/llama-server\n"
            'setsid "$BIN" --port 8014\n'
        )
    with pytest.raises(ValueError, match="unresolved shell variable"):
        parse_script(
            '_CLI_HF_TOKEN="${HF_TOKEN:-}"\n'
            "BIN=/srv/example/builds/llama/bin/llama-server\n"
            'setsid "$BIN" --port 8014 --token "$_CLI_HF_TOKEN"\n'
        )


def test_container_name_is_sparksi_and_stop_uses_it() -> None:
    head = _inspect(
        "example-compose",
        ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
        ["HF_HUB_OFFLINE=1", "HF_TOKEN=should-not-be-stored"],
    )
    catalog = {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "identity": {"container": "example-compose"},
            "launch": {"host": "head", "start": ["docker", "start", "example-compose"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner([
        ("*", ["docker", "inspect", "example-compose"], _ok(head)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(["PATH=/usr/bin"]))),
    ])
    result = import_model(catalog, runner, "example-vllm", dry_run=True)
    role = result["recipe"]["roles"]["head"]
    assert role["container"] == "sparksi-example-vllm"
    assert "HF_TOKEN" not in role["env"]
    assert role["env_pass"] == ["HF_TOKEN"]
    assert "should-not-be-stored" not in json.dumps(role)
    argv = role_argv(role, "100.64.0.11")
    assert argv[argv.index("--name") + 1] == "sparksi-example-vllm"
    assert "HF_TOKEN=should-not-be-stored" not in argv
    assert argv[argv.index("-e") + 1] == "HF_TOKEN" or "HF_TOKEN" in argv
    from sparksi.recipes.engines.common import docker_stop
    assert docker_stop(role["container"]) == ["docker", "rm", "-f", "sparksi-example-vllm"]


def _pair_role(**over) -> dict:
    role = {
        "runtime": "docker",
        "image": IMAGE,
        "env": {},
        "mounts": ["/srv/example/cache:/root/.cache/huggingface"],
        "network": "host",
        "gpus": "all",
        "container": "example-vllm",
        "args": ["vllm", "serve", "/srv/example/cache/weights", "--host", "0.0.0.0", "--port", "8014"],
        "devices": ["/dev/infiniband"],
    }
    role.update(over)
    return role


def _pair_recipe(placement: str, **role_over) -> dict:
    head = _pair_role(**role_over)
    worker = _pair_role(**role_over)
    roles = {"head": head} if placement != "both" else {"head": head, "worker": worker}
    return {
        "kind": "pinned",
        "engine": "vllm",
        "placement": placement,
        "port": 8014,
        "imported_from": "container:example-vllm at 2026-10-31T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": roles,
    }


def test_roce_permissions_are_allowed_only_on_both_nodes() -> None:
    allowed = _pair_recipe("both", cap_add=["CAP_IPC_LOCK"], security_opt=["label=disable"])
    assert validate_pinned(allowed) == []
    for role_name in ("head", "worker"):
        argv = role_argv(allowed["roles"][role_name], "100.64.0.11" if role_name == "head" else "100.64.0.12")
        assert argv[argv.index("--cap-add") + 1] == "CAP_IPC_LOCK"
        assert argv[argv.index("--security-opt") + 1] == "label=disable"
    single = _pair_recipe("head", cap_add=["IPC_LOCK"], security_opt=["label=disable"])
    single_conflicts = validate_pinned(single)
    assert FABRIC_ONLY in single_conflicts
    assert not any("refusing --cap-add IPC_LOCK" in item or "refusing --security-opt label=disable" in item for item in single_conflicts)
    # Saving keeps the flags; the refusal moves to Dry run and Load.
    assert validate_pinned(single, for_save=True) == []
    assert fabric_only(single) is True and fabric_only(allowed) is False
    other = _pair_recipe("both", cap_add=["SYS_ADMIN", "IPC_LOCK"], security_opt=["seccomp=unconfined"])
    other_conflicts = validate_pinned(other)
    assert any("refusing --cap-add SYS_ADMIN" in item for item in other_conflicts)
    assert any("refusing --security-opt seccomp=unconfined" in item for item in other_conflicts)
    assert not any(item == "refusing --cap-add IPC_LOCK" for item in other_conflicts)
    other_single = _pair_recipe("head", cap_add=["SYS_ADMIN", "IPC_LOCK"])
    assert any("refusing --cap-add SYS_ADMIN" in item for item in validate_pinned(other_single, for_save=True))
    privileged = _pair_recipe("both", privileged=True, cap_add=["IPC_LOCK"], security_opt=["label=disable"])
    assert any("refusing --privileged" in item for item in validate_pinned(privileged))


def test_mount_refusal_lists_every_tmp_patch() -> None:
    mounts = [f"/tmp/patch-{index}.py:/opt/patch-{index}.py" for index in range(4)]
    recipe = _pair_recipe("head", mounts=mounts)
    conflicts = validate_pinned(recipe)
    for index in range(4):
        assert any(f"/tmp/patch-{index}.py" in item and "outside model roots" in item for item in conflicts), conflicts
    assert not any("allowing /tmp" in item for item in conflicts)


def _pinned_from_role(role: dict, *, placement: str = "head", port: int = 8014) -> dict:
    roles = {"head": role}
    if placement == "both":
        roles["worker"] = copy.deepcopy(role)
    return {
        "kind": "pinned",
        "engine": "vllm",
        "placement": placement,
        "port": port,
        "imported_from": "script:/srv/example/builds/start.sh at 2026-10-31T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": roles,
    }


def test_overlay_function_uses_the_flag_branch(monkeypatch) -> None:
    text = (ROOT / "overlay-start.sh").read_text(encoding="utf-8")
    monkeypatch.setenv("FP8_DENSE", "false")
    monkeypatch.setenv("DO_LAUNCH", "false")
    role = parse_script(text)
    assert role is not None
    assert role["runtime"] == "docker"
    assert role["image"] == "example.invalid/vllm:patched"
    assert role["mounts"] == [
        "/srv/example/cache:/root/.cache/huggingface",
        "/srv/example/builds/files/overlay/first.patch:/opt/overlay/first.patch",
        "/srv/example/builds/files/overlay/second.patch:/opt/overlay/second.patch",
    ]
    assert role["args"][:3] == ["vllm", "serve", "/srv/example/cache/weights"]
    assert "}" not in " ".join(role["mounts"] + role["args"])
    assert "OV" not in role["env"]
    assert validate_pinned(_pinned_from_role(role)) == []

    skipped = text.replace('FP8_DENSE="${FP8_DENSE:-true}"', "FP8_DENSE=false", 1)
    with pytest.raises(ValueError, match="unresolved shell variable OV"):
        parse_script(skipped)

    unset = text.replace('FP8_DENSE="${FP8_DENSE:-true}"\n', "")
    monkeypatch.setenv("FP8_DENSE", "true")
    with pytest.raises(ValueError, match="unresolved shell variable FP8_DENSE"):
        parse_script(unset)

    with pytest.raises(ValueError, match="could not check /srv/example/builds/missing.sh on the node"):
        parse_script(
            "if [ -f /srv/example/builds/missing.sh ]; then\n"
            "  OV=/srv/example/builds/files/overlay\n"
            "fi\n"
            "exec docker run --network host example.invalid/vllm:patched vllm serve --port 8014\n"
        )


def _script_role(host_file: Path, container: str, args: list[str], *, extra: list[str] | None = None) -> dict:
    mounts = [f"{host_file}:{container}"]
    for item in extra or []:
        mounts.append(item)
    return {
        "runtime": "docker",
        "image": IMAGE,
        "env": {},
        "mounts": mounts,
        "network": "host",
        "gpus": "all",
        "container": "example-vllm",
        "args": args,
    }


class _NodeFiles:
    """Remote cat replies. A path omitted here cannot be read, even if a local file exists."""

    def __init__(self, files: dict[str, tuple[str, str]]) -> None:
        self.files = files
        self.calls: list[tuple[str, list[str]]] = []

    def capture(self, host, argv, timeout=8):
        self.calls.append((str(host), list(argv)))
        if list(argv[:1]) != ["cat"] or len(argv) != 2:
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}
        rule = self.files.get(argv[1])
        if rule is None:
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}
        want_host, text = rule
        if want_host not in ("*", host):
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}
        return _ok(text)


def test_validator_refuses_wildcard_bind_inside_mounted_script(tmp_path) -> None:
    start = "/srv/example/builds/start.sh"
    sourced = "/srv/example/builds/extra.sh"
    worker = "/srv/example/builds/worker.sh"
    clean = "/srv/example/builds/clean.sh"
    star = "/srv/example/builds/star.sh"
    bound = "/srv/example/builds/bind.sh"
    head_name = CFG.head.name
    worker_name = CFG.worker.name
    files = {
        start: ("*", "#!/bin/sh\n# --host 0.0.0.0\nvllm serve model --host 0.0.0.0 --port 8014\n"),
        sourced: ("*", "#!/bin/sh\nexec vllm serve model --host=0.0.0.0 --port 8014\n"),
        worker: (worker_name, "#!/bin/sh\nvllm serve model --host :: --port 8014\n"),
        clean: ("*", "#!/bin/sh\n# --host 0.0.0.0\nvllm serve model --port 8014\n"),
        star: ("*", "#!/bin/sh\nserve --host '*'\n"),
        bound: ("*", "#!/bin/sh\nserve --bind ::\n"),
    }
    remote = _NodeFiles(files)
    direct = validate_pinned(
        _pinned_from_role(_script_role(Path(start), "/start.sh", ["bash", "/start.sh"])),
        runner=remote,
    )
    assert any(item == f"refusing bind 0.0.0.0 in {start}:3" for item in direct), direct
    assert any(call[0] == head_name and call[1] == ["cat", start] for call in remote.calls)

    parent = "/srv/example/builds/parent.sh"
    remote.files[parent] = ("*", "#!/bin/sh\nsource /opt/extra.sh\n")
    mounted = _script_role(
        Path(parent),
        "/start.sh",
        ["bash", "/start.sh"],
        extra=[f"{sourced}:/opt/extra.sh"],
    )
    followed = validate_pinned(_pinned_from_role(mounted), runner=remote)
    assert any(item == f"refusing bind 0.0.0.0 in {sourced}:2" for item in followed), followed

    head = _script_role(Path(clean), "/start.sh", ["bash", "/start.sh"])
    rank = _script_role(Path(worker), "/worker.sh", ["bash", "/worker.sh"])
    both = _pinned_from_role(head, placement="both")
    both["roles"]["worker"] = rank
    ranked = validate_pinned(both, runner=remote)
    assert any(item == f"refusing bind :: in {worker}:2" for item in ranked), ranked
    assert any(call[0] == worker_name and call[1] == ["cat", worker] for call in remote.calls)
    assert not any(clean in item for item in ranked)

    assert validate_pinned(
        _pinned_from_role(_script_role(Path(clean), "/start.sh", ["bash", "/start.sh"])),
        runner=remote,
    ) == []
    missing = "/srv/example/builds/no-such-start.sh"
    unread = validate_pinned(
        _pinned_from_role(_script_role(Path(missing), "/start.sh", ["bash", "/start.sh"])),
        runner=remote,
    )
    assert any(item == f"could not read mounted script {missing} on head" for item in unread), unread

    starred = validate_pinned(
        _pinned_from_role(_script_role(Path(star), "/star.sh", ["bash", "/star.sh"])),
        runner=remote,
    )
    assert any(item == f"refusing bind * in {star}:2" for item in starred), starred
    binds = validate_pinned(
        _pinned_from_role(_script_role(Path(bound), "/bind.sh", ["/bind.sh"])),
        runner=remote,
    )
    assert any(item == f"refusing bind :: in {bound}:2" for item in binds), binds

    decoy = tmp_path / "local-start.sh"
    decoy.write_text("#!/bin/sh\nserve --host 0.0.0.0 --port 8014\n", encoding="utf-8")
    saved = list(CFG.head.mount_roots)
    CFG.head.mount_roots.append(str(tmp_path))
    try:
        local_only = validate_pinned(
            _pinned_from_role(_script_role(decoy, "/start.sh", ["bash", "/start.sh"])),
            runner=remote,
        )
    finally:
        CFG.head.mount_roots[:] = saved
    assert any(item == f"could not read mounted script {decoy} on head" for item in local_only), local_only
    assert not any(item.startswith("refusing bind") for item in local_only)

    image_only = _script_role(Path("/srv/example/cache/weights"), "/root/.cache/huggingface", ["bash", "/opt/inside.sh"])
    image_only["mounts"] = ["/srv/example/cache:/root/.cache/huggingface"]
    assert validate_pinned(_pinned_from_role(image_only), runner=remote) == []


def _absent(blob: str, secret: str) -> None:
    if secret and secret in blob:
        raise AssertionError("a secret value was included in output")


def test_env_allowlist_rejects_secrets_and_unknown_keys() -> None:
    for name in ENV_VALUE_ALLOW:
        assert env_value_allowed(name), name
    for name in (
        "NCCL_IB_DISABLE",
        "SPECULATIVE_CONFIG",
        "SPEC_DRAFT",
        "MTP_DRAFT",
        "MTP_NUM_SPECULATIVE_" + "TOKENS",
        "MAX_NUM_BATCHED_" + "TOKENS",
        "MAX_NUM_SEQS",
        "MODEL_ID",
        "FP8_DENSE",
        "WORKER_USER",
    ):
        assert env_value_allowed(name), name
    for name in (
        "HF_TOKEN",
        "HF_HUB_OFFLINE",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "API_KEY",
        "DB_PASSWORD",
        "MY_PASS",
        "LOGIN_CREDENTIAL",
        "UNKNOWN_SETTING",
        "EXTRA_DOCKER_ARGS",
    ):
        assert not env_value_allowed(name), name


def test_env_file_drops_token_values_and_unknown_keys(tmp_path, monkeypatch, caplog) -> None:
    import logging

    secret = "hf_exampletokenvalue"
    sentinel = "unknown-value-sentinel"
    env_text = (ROOT / "stack.env").read_text(encoding="utf-8")
    values, secrets, _review = parse_env_file(env_text)
    assert values["IMAGE"] == "example.invalid/vllm:patched"
    assert values["MODEL"] == "/srv/example/cache/weights"
    assert values["KV_CACHE_DTYPE"] == "fp8"
    assert "HF_TOKEN" not in values
    assert "UNKNOWN_SETTING" not in values
    assert secrets == ["HF_TOKEN", "UNKNOWN_SETTING"]
    _absent(json.dumps(values), secret)
    _absent(json.dumps(values), sentinel)

    expanded = "HF_TOKEN=" + secret + "\nMODEL=$(uname)\n"
    with pytest.raises(ValueError, match="refusing expansion in env file for MODEL") as caught:
        parse_env_file(expanded)
    _absent(str(caught.value), secret)

    script = "/srv/example/builds/stack/start.sh"
    env_path = "/srv/example/builds/stack/.env"
    catalog = {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "launch": {
                "host": CFG.head.name,
                "start": [script],
                "stop": ["/srv/example/builds/stack/stop.sh"],
            },
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner([
        ("*", ["cat", script], _ok((ROOT / "dotenv-start.sh").read_text(encoding="utf-8"))),
        ("*", ["test", "-f", env_path], _ok("")),
        ("*", ["cat", env_path], _ok(env_text)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(["PATH=/usr/bin"]))),
        ("*", ["printenv", "HF_TOKEN"], _ok("")),
    ])
    caplog.set_level(logging.DEBUG)
    dry = import_model(catalog, runner, "example-vllm", f"script:{script}", dry_run=True)
    blob = json.dumps(dry)
    argv = role_argv(dry["recipe"]["roles"]["head"], "100.64.0.11")
    captured = "\n".join([blob, " ".join(argv), caplog.text, "\n".join(dry.get("conflicts") or [])])
    _absent(captured, secret)
    _absent(captured, sentinel)
    assert dry["conflicts"] == [], dry["conflicts"]
    assert dry["recipe"]["token_note"] == "Token required"
    env = dry["recipe"]["roles"]["head"]["env"]
    assert env["MODEL"] == "/srv/example/cache/weights"
    assert env["SERVED_MODEL_NAME"] == "example"
    assert env["NCCL_IB_DISABLE"] == "0"
    assert "HF_TOKEN" not in env
    assert "UNKNOWN_SETTING" not in env
    passed = dry["recipe"]["roles"]["head"]["env_pass"]
    assert "HF_TOKEN" in passed
    assert "UNKNOWN_SETTING" in passed
    assert "-e" in argv and "HF_TOKEN" in argv
    assert "HF_TOKEN=" not in " ".join(argv)
    assert "--env-file" not in argv
    assert dry["recipe"]["roles"]["head"]["image"] == IMAGE
    assert any(call[1] == ["test", "-f", env_path] for call in runner.calls)

    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path))
    saved = import_model(catalog, runner, "example-vllm", f"script:{script}", dry_run=False)
    _absent(json.dumps(saved), secret)
    stored = recipe_path("example-vllm").read_text(encoding="utf-8")
    _absent(stored, secret)
    _absent(stored, sentinel)

    blocked = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "roles": {"head": {"env": {"MODEL": secret}}},
    }
    with pytest.raises(ValueError, match="refusing to save env MODEL") as save_error:
        save_recipe("example-vllm", blocked)
    _absent(str(save_error.value), secret)
    noisy = "aB3dE6gH9jK2mN5pQ8sT1vX4yZ7cF0hU"
    assert value_looks_like_token(noisy)
    with pytest.raises(ValueError, match="refusing to save env TOKEN_LIKE") as entropy_error:
        save_recipe("example-vllm", {"roles": {"head": {"env": {"TOKEN_LIKE": noisy}}}})
    _absent(str(entropy_error.value), noisy)


def test_shell_idioms_resolve_and_name_the_line() -> None:
    script = "/srv/example/builds/stack/start.sh"
    env_path = "/srv/example/builds/stack/.env"
    env_text = "IMAGE=example.invalid/vllm:patched\nMODEL=/srv/example/cache/weights\nPORT=8014\n"
    runner = Runner([
        ("*", ["test", "-f", env_path], _ok("")),
        ("*", ["cat", env_path], _ok(env_text)),
    ])
    role = parse_script(
        (ROOT / "shell-start.sh").read_text(encoding="utf-8"),
        runner=runner,
        host=CFG.head.name,
        cwd="/srv/example/builds/stack",
        path=script,
    )
    assert role is not None
    assert role["image"] == IMAGE
    assert role["args"][role["args"].index("--port") + 1] == "8014"
    assert "HF_TOKEN" in role["env_pass"]
    assert "$*" not in json.dumps(role)
    assert "BASH_SOURCE" not in json.dumps(role)

    directory = "/srv/example/builds/stack"
    for idiom in (
        'SCRIPT_DIR="$(dirname "$0")"\n',
        'SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"\n',
        'SCRIPT_DIR="$(realpath "$(dirname "$0")")"\n',
        'SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"\n',
    ):
        found = parse_script(
            idiom + 'exec docker run --network host "$SCRIPT_DIR/app:latest" vllm serve /srv/example/cache/weights --port 8014\n',
            path=script,
        )
        assert found is not None
        assert found["image"] == f"{directory}/app:latest"

    with pytest.raises(ValueError, match=r"unresolved shell command substitution in /srv/example/builds/stack/start.sh:1"):
        parse_script('FOO="$(uname)"\nexec docker run --network host example.invalid/vllm:patched vllm serve --port 8014\n', path=script)
    with pytest.raises(ValueError, match=r"unresolved shell variable MISSING in /srv/example/builds/fail.sh:3"):
        parse_script(
            "#!/bin/sh\nBIN=/srv/example/builds/llama/bin/llama-server\nsetsid \"$MISSING\" --port 8014\n",
            path="/srv/example/builds/fail.sh",
        )
    with pytest.raises(ValueError, match=r"unresolved shell variable NOPE in /srv/example/builds/cond.sh:2"):
        parse_script(
            "#!/bin/sh\nif [ \"$NOPE\" = x ]; then\n  true\nfi\nexec docker run --network host example.invalid/vllm:patched vllm serve --port 8014\n",
            path="/srv/example/builds/cond.sh",
        )


def _token_recipe_catalog() -> dict:
    return {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "repo": "example/model",
            "placement": "single",
            "port": 8014,
            "weights_gib": 1,
            "local": {"head": "/srv/example/cache/weights"},
            "launch": {
                "host": CFG.head.name,
                "start": ["/srv/example/builds/stack/start.sh"],
                "stop": ["/srv/example/builds/stack/stop.sh"],
            },
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }


def _snapshot_answers(root: str, revision: str, *, shards: list[str] | None = None, skip: set[str] | None = None) -> list:
    snap = f"{root}/snapshots/{revision}"
    skip = skip or set()
    rows = [
        ("*", ["cat", f"{root}/refs/main"], _ok(revision + "\n")),
        ("*", ["test", "-d", snap], _ok("")),
    ]
    files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    if shards is None:
        files.append("model.safetensors")
    for name in files:
        if name in skip:
            continue
        rows.append(("*", ["test", "-f", f"{snap}/{name}"], _ok("")))
    if shards is not None:
        weight_map = {f"layer.{index}": name for index, name in enumerate(shards)}
        rows.append((
            "*",
            ["cat", f"{snap}/model.safetensors.index.json"],
            _ok(json.dumps({"weight_map": weight_map})),
        ))
        for name in shards:
            if name in skip:
                continue
            rows.append(("*", ["test", "-f", f"{snap}/{name}"], _ok("")))
    return rows


def test_complete_cache_drops_the_token_and_incomplete_cache_refuses() -> None:
    secret = "hf_exampletokenvalue"
    hub = str(CFG.path("hf_hub") or "").rstrip("/")
    root = f"{hub}/models--example--model"
    revision = "0123456789abcdef0123456789abcdef01234567"
    script = "/srv/example/builds/stack/start.sh"
    env_path = "/srv/example/builds/stack/.env"
    env_text = (ROOT / "stack.env").read_text(encoding="utf-8")
    base = [
        ("*", ["cat", script], _ok((ROOT / "dotenv-start.sh").read_text(encoding="utf-8"))),
        ("*", ["test", "-f", env_path], _ok("")),
        ("*", ["cat", env_path], _ok(env_text)),
        ("*", ["docker", "image", "inspect", IMAGE], _ok(_image(["PATH=/usr/bin"]))),
        *_snapshot_answers(root, revision),
    ]
    cached_runner = Runner(base)
    cached = import_model(_token_recipe_catalog(), cached_runner, "example-vllm", f"script:{script}", dry_run=True)
    blob = json.dumps(cached)
    _absent(blob, secret)
    assert cached["conflicts"] == [], cached["conflicts"]
    role = cached["recipe"]["roles"]["head"]
    assert role["env"]["HF_HUB_OFFLINE"] == "1"
    assert role["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in role.get("env_pass") or []
    assert "HF_TOKEN" not in role["env"]
    assert cached["recipe"]["token_note"] == "Token not required"
    assert not any(part.rsplit("/", 1)[-1] in ("token", "token.json") for _host, argv in cached_runner.calls for part in argv)

    missing = [item for item in base if not (item[1][:2] == ["test", "-f"] and item[1][-1].endswith("/model.safetensors"))]
    missing.append(("*", ["printenv", "HF_TOKEN"], {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "unset"}))
    refused_runner = Runner(missing)
    refused = import_model(_token_recipe_catalog(), refused_runner, "example-vllm", f"script:{script}", dry_run=True)
    text = json.dumps(refused)
    _absent(text, secret)
    assert "HF token not available on head; weights not fully cached" in refused["conflicts"]
    assert refused["recipe"]["token_note"] == "Token required"
    joined = " ".join(" ".join(argv) for _host, argv in refused_runner.calls)
    assert ["printenv", "HF_TOKEN"] in [argv for _host, argv in refused_runner.calls]
    assert not any(part.endswith("/token") for _host, argv in refused_runner.calls for part in argv)
    assert secret not in joined


def test_incomplete_worker_cache_names_the_worker() -> None:
    hub = str(CFG.path("hf_hub") or "").rstrip("/")
    root = f"{hub}/models--example--model"
    revision = "0123456789abcdef0123456789abcdef01234567"
    fail = {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "missing"}
    head_files = _snapshot_answers(root, revision)
    runner = Runner([
        *[(CFG.head.name, argv, value) for _host, argv, value in head_files],
        (CFG.worker.name, ["cat", f"{root}/refs/main"], fail),
        (CFG.worker.name, ["printenv", "HUGGING_FACE_HUB_TOKEN"], fail),
    ])
    recipe = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "both",
        "port": 8014,
        "roles": {
            "head": {
                "runtime": "docker", "image": IMAGE, "env": {}, "env_pass": ["HF_TOKEN"],
                "mounts": ["/srv/example/cache:/root/.cache/huggingface"], "network": "host",
                "args": ["vllm", "serve", "/srv/example/cache/weights", "--port", "8014"],
            },
            "worker": {
                "runtime": "docker", "image": IMAGE, "env": {}, "env_pass": ["HUGGING_FACE_HUB_TOKEN"],
                "mounts": ["/srv/example/cache:/root/.cache/huggingface"], "network": "host",
                "args": ["vllm", "serve", "/srv/example/cache/weights", "--port", "8014"],
            },
        },
    }
    conflicts = apply_token_cache(recipe, runner, repo="example/model")
    assert conflicts == ["HF token not available on worker; weights not fully cached"]
    assert recipe["roles"]["head"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert recipe["roles"]["head"]["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert "HF_TOKEN" not in recipe["roles"]["head"]["env_pass"]
    assert recipe["roles"]["worker"]["env_pass"] == ["HUGGING_FACE_HUB_TOKEN"]
    assert recipe["token_note"] == "Token required"
    assert "hf_" not in json.dumps(conflicts)


def test_bracket_and_or_stay_inside_one_test() -> None:
    taken = parse_script(
        "FLAG=no\n"
        'if [[ "$FLAG" == "yes" || "$FLAG" == "no" ]]; then\n'
        "  IMAGE=example.invalid/vllm:taken\n"
        "else\n"
        "  IMAGE=example.invalid/vllm:skipped\n"
        "fi\n"
        'exec docker run --network host "$IMAGE" vllm serve /srv/example/cache/weights --port 8014\n',
        path="/srv/example/builds/start.sh",
    )
    assert taken is not None
    assert taken["image"] == "example.invalid/vllm:taken"
    skipped = parse_script(
        "FLAG=no\n"
        'if [[ "$FLAG" == "yes" && "$FLAG" == "no" ]]; then\n'
        "  IMAGE=example.invalid/vllm:taken\n"
        "else\n"
        "  IMAGE=example.invalid/vllm:skipped\n"
        "fi\n"
        'exec docker run --network host "$IMAGE" vllm serve /srv/example/cache/weights --port 8014\n',
        path="/srv/example/builds/start.sh",
    )
    assert skipped is not None
    assert skipped["image"] == "example.invalid/vllm:skipped"


def test_extra_docker_args_are_reviewed_and_limited() -> None:
    script = "/srv/example/builds/stack/start.sh"
    env_path = "/srv/example/builds/stack/.env"
    good = (
        "IMAGE=example.invalid/vllm:patched\n"
        "MODEL=/srv/example/cache/weights\n"
        "PORT=8014\n"
        "EXTRA_DOCKER_ARGS=--gpus all --shm-size 16g --network host\n"
    )
    bad = good.replace("--network host", "--privileged", 1)

    def run(env_text: str):
        runner = Runner([
            ("*", ["test", "-f", env_path], _ok("")),
            ("*", ["cat", env_path], _ok(env_text)),
        ])
        text = (
            "#!/bin/sh\n"
            "cd /srv/example/builds/stack\n"
            "set -a\n"
            ". ./.env\n"
            "set +a\n"
            "exec docker run --network host \\\n"
            "  -v /srv/example/cache:/root/.cache/huggingface \\\n"
            '  "$IMAGE" vllm serve "$MODEL" --port "$PORT"\n'
        )
        return parse_script(text, runner=runner, host=CFG.head.name, cwd="/srv/example/builds/stack", path=script)

    role = run(good)
    assert role is not None
    assert "EXTRA_DOCKER_ARGS" not in role["env"]
    assert role["extra_docker_args"] == "--gpus all --shm-size 16g --network host"
    argv = role_argv(role, "100.64.0.11")
    assert "--shm-size" in argv and "16g" in argv
    assert validate_pinned(_pinned_from_role(role)) == []
    refused = run(bad)
    assert refused is not None
    conflicts = validate_pinned(_pinned_from_role(refused))
    assert any(item == "refusing EXTRA_DOCKER_ARGS --privileged" for item in conflicts), conflicts
    assert "--privileged" not in role_argv(refused, "100.64.0.11")


def test_token_like_values_and_missing_shards_fail_closed() -> None:
    secret = "hf_exampletokenvalue"
    values, secrets, review = parse_env_file(
        "MODEL=" + secret + "\n"
        + "MAX_NUM_BATCHED_" + "TOKENS=8192\n"
        + "MTP_NUM_SPECULATIVE_" + "TOKENS=3\n"
        + "EXTRA_DOCKER_ARGS=--ipc host\n"
    )
    assert "MODEL" in secrets
    assert secret not in json.dumps(values)
    assert secret not in json.dumps(review)
    assert values["MAX_NUM_BATCHED_" + "TOKENS"] == "8192"
    assert values["MTP_NUM_SPECULATIVE_" + "TOKENS"] == "3"
    assert review["EXTRA_DOCKER_ARGS"] == "--ipc host"

    hub = str(CFG.path("hf_hub") or "").rstrip("/")
    root = f"{hub}/models--example--model"
    revision = "0123456789abcdef0123456789abcdef01234567"
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    answers = _snapshot_answers(root, revision, shards=shards, skip={"model-00002-of-00002.safetensors"})
    runner = Runner([*[(CFG.head.name, argv, value) for _host, argv, value in answers]])
    recipe = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "roles": {"head": {
            "runtime": "docker", "image": IMAGE, "env": {}, "env_pass": ["HF_TOKEN"],
            "mounts": ["/srv/example/cache:/root/.cache/huggingface"], "network": "host",
            "args": ["vllm", "serve", "/srv/example/cache/weights", "--port", "8014"],
        }},
    }
    runner.answers.append((CFG.head.name, ["printenv", "HF_TOKEN"], {
        "ok": True, "code": 0, "stdout": secret, "stderr": "",
    }))
    conflicts = apply_token_cache(recipe, runner, repo="example/model")
    blob = json.dumps({"conflicts": conflicts, "recipe": recipe})
    _absent(blob, secret)
    assert conflicts == []
    assert recipe["token_note"] == "Token required"
    assert "HF_HUB_OFFLINE" not in recipe["roles"]["head"]["env"]
    assert recipe["roles"]["head"]["env_pass"] == ["HF_TOKEN"]
    assert any(argv[-1].endswith("model-00002-of-00002.safetensors") for _host, argv in runner.calls)


def test_recipe_run_reuses_a_stopped_container(monkeypatch, tmp_path):
    """An existing container name is removed so the new docker run argv is what starts."""
    from sparksi.agent import _recipe_run

    monkeypatch.setattr("sparksi.agent._runs_dir", lambda: tmp_path)
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(list(argv))
        class R:
            returncode = 0
            stdout = "started"
            stderr = ""
        return R()

    monkeypatch.setattr("sparksi.agent.subprocess.run", fake_run)
    monkeypatch.setattr(
        "sparksi.agent._inspect_container",
        lambda name: {"status": "exited", "exit_code": 0},
    )
    res = _recipe_run({
        "run_id": "reuse-1",
        "argv": ["docker", "run", "-d", "--name", "sparksi-x", "--flag", "changed", "image", "cmd"],
        "container": "sparksi-x",
    })
    assert res["ok"] is True and not res.get("reused")
    assert seen[0] == ["docker", "rm", "sparksi-x"]
    assert seen[1][:4] == ["docker", "run", "-d", "--name"]
    assert "--flag" in seen[1]

    seen.clear()
    monkeypatch.setattr(
        "sparksi.agent._inspect_container",
        lambda name: {"status": "running", "exit_code": 0},
    )
    res = _recipe_run({
        "run_id": "reuse-2",
        "argv": ["docker", "run", "-d", "--name", "sparksi-x", "image", "cmd"],
        "container": "sparksi-x",
    })
    assert res["ok"] is False
    assert res["error"] == "container sparksi-x is already running"
    assert seen == []

    seen.clear()
    monkeypatch.setattr("sparksi.agent._inspect_container", lambda name: {"status": "missing"})
    _recipe_run({
        "run_id": "reuse-3",
        "argv": ["docker", "run", "-d", "--name", "sparksi-x", "image", "cmd"],
    })
    assert seen[0][:2] == ["docker", "run"]
