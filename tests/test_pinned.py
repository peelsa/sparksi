# SPDX-License-Identifier: Apache-2.0

"""Pinned recipes, the read-only importer, service allowlist, and catalog ports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sparksi.agent import _refuse_recipe_argv, configure, main as agent_main
from sparksi.models import ModelCard
from sparksi.recipes.args import apply_edits, parse_launch, rewrite_bind
from sparksi.recipes.importer import (
    import_model,
    parse_compose_config,
    parse_docker_inspect,
    parse_ps,
    parse_script,
    parse_unit,
    readonly_argv,
)
from sparksi.recipes.pinned import render_pinned, role_argv, validate_pinned
from sparksi.recipes.store import load_recipe, recipe_path
from sparksi.recipes.validate import ctx_ceiling, validate_fields

ROOT = Path(__file__).resolve().parent / "fixtures" / "import"
HEAD = "spark-a"
IMAGE = "example.invalid/vllm:patched"
ALLOWED = ["example.invalid/vllm:1"]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CARDS_DIR", str(tmp_path / "cards"))
    configure({
        "mount_roots": [
            "/home/USER/.cache/huggingface",
            "/srv/example/cache",
            "/srv/example/builds",
            "/srv/example/worker",
        ],
        "service_commands": [],
    })
    yield


def _card() -> ModelCard:
    return ModelCard(
        id="example-vllm",
        title="Example",
        format="fp8",
        engines=["vllm"],
        files={HEAD: "/srv/example/cache/weights"},
        presence={HEAD: True},
        weights_gib=1,
    )


def _role(**over) -> dict:
    base = {
        "runtime": "docker",
        "image": IMAGE,
        "env": {"HF_HUB_OFFLINE": "1"},
        "mounts": ["/srv/example/cache:/root/.cache/huggingface"],
        "network": "host",
        "gpus": "all",
        "ipc": "host",
        "shm": "16g",
        "container": "example-vllm",
        "args": [
            "vllm", "serve", "/srv/example/cache/weights",
            "--host", "0.0.0.0",
            "--port", "8014",
            "--max-model-len", "8192",
        ],
    }
    base.update(over)
    return base


def _recipe(**over) -> dict:
    base = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example-vllm at 2026-10-31T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": {"head": _role()},
    }
    base.update(over)
    return base


def test_docker_inspect_compose_unit_and_script_parse() -> None:
    inspected = json.loads((ROOT / "docker-inspect.json").read_text(encoding="utf-8"))
    role = parse_docker_inspect(inspected)
    assert role["runtime"] == "docker"
    assert role["image"] == IMAGE
    assert role["network"] == "host"
    assert role["gpus"] == "all"
    assert role["ipc"] == "host"
    assert role["shm"] == "16g"
    assert role["container"] == "example-vllm"
    assert role["env"]["HF_HUB_OFFLINE"] == "1"
    assert "PATH" not in role["env"]
    assert any(item.startswith("/srv/example/builds/patches/") for item in role["mounts"])
    assert "--tool-call-parser" in role["args"]
    assert "qwen3_coder" in role["args"]
    command, known, extra = parse_launch(role["args"])
    assert command == ["vllm", "serve", "/srv/example/cache/weights"]
    assert known["ctx"] == 850000
    assert known["port"] == 8888
    assert known["tool_parser"] == "qwen3_coder"
    assert known["reasoning_parser"] == "glm45"
    assert known["speculative"]["method"] == "dflash"
    assert known["speculative"]["model"] == "/srv/example/builds/drafter"
    assert known["speculative"]["tokens"] == 3
    assert "--trust-remote-code" in extra
    assert "--attention-backend FLASH_ATTN" in extra
    assert "--tensor-parallel-size 2" in extra
    assert "vllm" not in extra
    assert "serve" not in extra
    assert apply_edits(role["args"], {}, extra, "vllm") == role["args"]

    roles = parse_compose_config((ROOT / "compose-config.json").read_text(encoding="utf-8"))
    assert set(roles) == {"head", "worker"}
    assert roles["head"]["image"] == "example.invalid/tensorfold:1"
    assert roles["worker"]["mounts"] == ["/srv/example/worker:/opt/fold"]
    assert roles["head"]["gpus"] == "all"

    pointed = parse_unit((ROOT / "unit.service").read_text(encoding="utf-8"))
    assert pointed["compose"] == "/srv/example/builds/compose.yaml"

    native = parse_script((ROOT / "llama-start.sh").read_text(encoding="utf-8"))
    assert native is not None
    assert native["runtime"] == "native"
    assert native["binary"] == "/srv/example/builds/llama/bin/llama-server"
    assert native["env"]["LD_LIBRARY_PATH"] == "/srv/example/builds/llama/lib"
    assert "--spec-type" in native["args"]
    assert native["args"][native["args"].index("--spec-draft-n-max") + 1] == "2"
    assert "--alias" in native["args"]

    from_ps = parse_ps((ROOT / "ps.txt").read_text(encoding="utf-8"))
    assert from_ps is not None
    assert from_ps["binary"].endswith("/llama-server")


def test_pinned_render_rewrites_only_the_bind_and_edits_known_flags() -> None:
    recipe = _recipe()
    body = render_pinned(_card(), recipe)
    argv = body["argv_preview"]["head"]
    assert argv[argv.index("--host") + 1] == "100.64.0.11"
    assert "0.0.0.0" not in argv
    assert "--privileged" not in argv
    assert body["imported_from"].startswith("container:example-vllm")
    assert body["fields"]["command"] == "vllm serve /srv/example/cache/weights"
    assert body["fields"]["extra_args"] == []
    assert "--max-model-len" in argv
    edited = {
        **recipe,
        "roles": {
            "head": {
                **recipe["roles"]["head"],
                "args": [
                    "vllm", "serve", "/srv/example/cache/weights",
                    "--host", "0.0.0.0",
                    "--port", "8016",
                    "--max-model-len", "16384",
                    "--kv-cache-dtype", "bf16",
                    "--trust-remote-code",
                ],
            }
        },
    }
    from sparksi.recipes.pinned import apply_overrides

    updated = apply_overrides(edited, {"ctx": 32768, "port": 8016, "kv_cache_dtype": "bf16"}, {"ctx", "port", "kv_cache_dtype"})
    args = updated["roles"]["head"]["args"]
    assert args[args.index("--max-model-len") + 1] == "32768"
    assert args[args.index("--kv-cache-dtype") + 1] == "bf16"
    assert "--trust-remote-code" in args
    rendered = rewrite_bind(args, "100.64.0.11")
    assert rendered[rendered.index("--host") + 1] == "100.64.0.11"
    # An mpirun host list is not the listen bind.
    kept = rewrite_bind(["mpirun", "--host", "192.0.2.11:1,192.0.2.12:1"], "100.64.0.11")
    assert "192.0.2.11:1,192.0.2.12:1" in kept


def test_pinned_validation_refuses_every_escape() -> None:
    assert validate_pinned(_recipe()) == []
    cases = [
        ({"privileged": True}, "--privileged"),
        ({"pid": "host"}, "--pid=host"),
        ({"cap_add": ["SYS_ADMIN"]}, "--cap-add"),
        ({"security_opt": ["seccomp=unconfined"]}, "--security-opt"),
        ({"devices": ["/dev/sda"]}, "--device"),
        ({"network": "bridge"}, "--network"),
        ({"mounts": ["/:/host"]}, "refusing mount /"),
        ({"mounts": ["/etc:/etc"]}, "refusing mount /etc"),
        ({"mounts": ["/root:/root"]}, "refusing mount /root"),
        ({"mounts": ["/var/run/docker.sock:/var/run/docker.sock"]}, "docker.sock"),
        ({"mounts": ["/tmp:/tmp"]}, "outside model roots"),
        ({"args": ["vllm", "serve", "--host", "0.0.0.0", "--port", "8014", "dgx-announce"]}, "dgx-announce"),
    ]
    for patch, needle in cases:
        recipe = _recipe(roles={"head": _role(**patch)})
        conflicts = validate_pinned(recipe)
        assert any(needle in item for item in conflicts), (patch, conflicts)
    off = _recipe(port=8188, roles={"head": _role(args=["vllm", "serve", "--host", "0.0.0.0", "--port", "8188"])})
    assert any("8188" in item for item in validate_pinned(off))
    wild = role_argv(_role(args=["vllm", "serve", "--host", "0.0.0.0", "--port", "8014"]), "0.0.0.0")
    assert "0.0.0.0" in wild
    safe = role_argv(_role(), "100.64.0.11")
    assert "0.0.0.0" not in safe
    native = _recipe(engine="llamacpp", roles={"head": {
        "runtime": "native",
        "binary": "/tmp/llama-server",
        "args": ["-m", "/srv/example/cache/example.gguf", "--host", "0.0.0.0", "--port", "8014"],
        "env": {},
    }})
    assert any("binary" in item for item in validate_pinned(native))
    allowed = _recipe(engine="llamacpp", roles={"head": {
        "runtime": "native",
        "binary": "/srv/example/builds/llama/bin/llama-server",
        "args": ["-m", "/srv/example/cache/example.gguf", "--host", "0.0.0.0", "--port", "8014", "-c", "262144"],
        "env": {"LD_LIBRARY_PATH": "/srv/example/builds/llama/lib"},
    }})
    assert validate_pinned(allowed) == []
    on_path = _recipe(engine="llamacpp", roles={"head": {
        "runtime": "native",
        "binary": "llama-server",
        "args": ["-m", "/srv/example/cache/example.gguf", "--host", "0.0.0.0", "--port", "8014"],
        "env": {},
    }})
    assert validate_pinned(on_path) == []


def test_ctx_caps_and_port_remap() -> None:
    assert ctx_ceiling(None) == 2097152
    assert ctx_ceiling(8192) == 8192
    wide, bad = validate_fields("vllm", {"ctx": 850000}, allowed_images=ALLOWED)
    assert wide["ctx"] == 850000 and bad == []
    _cleaned, bad = validate_fields("vllm", {"ctx": 3000000}, allowed_images=ALLOWED)
    assert any("2097152" in item for item in bad)
    _cleaned, bad = validate_fields("vllm", {"ctx": 9000}, allowed_images=ALLOWED, ctx_max=8192)
    assert any("8192" in item for item in bad)
    huge = _recipe(roles={"head": _role(args=[
        "vllm", "serve", "--host", "0.0.0.0", "--port", "8014", "--max-model-len", "850000",
    ])})
    assert validate_pinned(huge) == []
    capped = validate_pinned(huge, ctx_max=8192)
    assert any("8192" in item for item in capped)

    class Runner:
        def __init__(self) -> None:
            self.argv: list[list[str]] = []

        def capture(self, host, argv, timeout=8):
            self.argv.append(list(argv))
            assert argv[0] in ("docker", "systemctl", "cat", "ps")
            assert "start" not in argv and "stop" not in argv and "up" not in argv
            if argv[:2] == ["docker", "inspect"]:
                return {"ok": True, "code": 0, "stdout": (ROOT / "docker-inspect.json").read_text(encoding="utf-8"), "stderr": ""}
            return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}

    catalog = {
        "library": [{
            "id": "example-vllm",
            "title": "Example",
            "format": "fp8",
            "placement": "single",
            "port": 8888,
            "weights_gib": 1,
            "local": {HEAD: "/srv/example/cache/weights"},
            "identity": {"container": "example-vllm"},
            "launch": {"host": HEAD, "start": ["docker", "start", "example-vllm"], "stop": ["docker", "stop", "example-vllm"]},
        }],
        "host": {"reserved_ports": [8188]},
        "never_touch": {},
    }
    runner = Runner()
    result = import_model(catalog, runner, "example-vllm", "container:example-vllm", dry_run=True)
    assert result["saved"] is False
    assert not recipe_path("example-vllm").exists()
    assert result["port_note"] == "imported port 8888 remapped to 8010"
    recipe = result["recipe"]
    assert recipe["port"] == 8010
    assert recipe["kind"] == "pinned"
    assert recipe["engine"] == "vllm"
    assert "8010" in recipe["roles"]["head"]["args"]
    assert "8888" not in recipe["roles"]["head"]["args"]
    assert 29501 in recipe["rendezvous_ports"]
    assert result["conflicts"] == []
    assert runner.argv[0][:2] == ["docker", "inspect"]
    assert readonly_argv("container", "example-vllm") == ["docker", "inspect", "example-vllm"]
    saved = import_model(catalog, runner, "example-vllm", "container:example-vllm", dry_run=False)
    assert saved["saved"] is True
    assert load_recipe("example-vllm")["kind"] == "pinned"
    blocked = _recipe(port=8188)
    blocked["roles"]["head"]["args"] = ["vllm", "serve", "--port", "8188"]
    # 8188 is not remapped onto a model port by validation.
    assert any("8188" in item for item in validate_pinned(blocked))


def test_catalog_port_is_the_default_when_it_is_free() -> None:
    from sparksi.catalog import load_catalog
    from sparksi.models import build_cards
    from sparksi.recipes.service import _choose

    catalog = load_catalog()
    cards = build_cards(catalog, presence={row["id"]: {host: True for host in (row.get("local") or {})} for row in catalog["library"]}, probe=False)
    single = next(item for item in cards if item.id == "single-gguf")
    assert single.catalog_port == 8014
    _engine, _placement, port, conflicts = _choose(single, {"engine": "llamacpp"}, {HEAD: set(), "spark-b": set()}, set(), catalog)
    assert conflicts == []
    assert port == 8014
    pair = next(item for item in cards if item.id == "pair-fast")
    assert pair.catalog_port == 8888
    _engine, _placement, port, conflicts = _choose(pair, {"engine": "vllm"}, {HEAD: set(), "spark-b": set()}, set(), catalog)
    assert port == 8010
    assert conflicts == []


def test_service_allowlist_is_verbatim() -> None:
    from sparksi.catalog import load_catalog
    from sparksi.recipes.service import list_model_cards, service_action

    configure({"service_commands": [["/bin/true"]]})
    refused = agent_main({"op": "service", "argv": ["/bin/false"]})
    assert refused["ok"] is False
    assert refused["error"] == "not in the service allowlist"
    started = agent_main({"op": "service", "argv": ["/bin/true"]})
    assert started["ok"] is True, started
    configure({"service_commands": [["/home/USER/scripts/video-mode.sh", "video"], ["/bin/true", "--port", "8188"]]})
    blocked = agent_main({"op": "service", "argv": ["/home/USER/scripts/video-mode.sh", "video"]})
    assert blocked["ok"] is False
    assert "forbidden" in blocked["error"] or "video-mode" in blocked["error"]
    port = agent_main({"op": "service", "argv": ["/bin/true", "--port", "8188"]})
    assert port["ok"] is False
    assert "8188" in port["error"]
    announced = agent_main({"op": "service", "argv": ["/bin/true"]})
    assert announced["ok"] is False

    catalog = load_catalog()
    body = list_model_cards(catalog, None)
    services = {row["id"]: row for row in body["services"]}
    assert "video-seat" in services and services["video-seat"]["protected"] is True
    assert "studio-b" in services and services["studio-b"]["protected"] is False
    assert "svg-b" in services
    assert "video-seat" not in {row["id"] for row in body["models"]}
    assert "pair-fast" in {row["id"] for row in body["models"]}

    class Record:
        def __init__(self) -> None:
            self.calls = []

        def invoke(self, host, req, timeout=30):
            self.calls.append((host, req))
            return {"ok": True, "pid": 1}

    record = Record()
    code, payload = service_action(catalog, record, "video-seat", "start")
    assert code == 400
    assert payload.get("protected") is True
    assert record.calls == []
    code, payload = service_action(catalog, record, "studio-b", "start")
    assert code == 400
    assert "allowlist" in payload["error"]
    assert record.calls == []
    configure({"service_commands": []})
    from sparksi.config import CFG

    real = CFG.service_commands
    CFG.service_commands = lambda: [["sudo", "systemctl", "start", "studio-b"]]  # type: ignore[method-assign]
    try:
        code, payload = service_action(catalog, record, "studio-b", "start")
    finally:
        CFG.service_commands = real  # type: ignore[method-assign]
    assert code == 200, payload
    assert record.calls[0][1]["op"] == "service"
    assert record.calls[0][1]["argv"] == ["sudo", "systemctl", "start", "studio-b"]


def test_agent_allows_a_native_binary_under_a_mount_root() -> None:
    assert _refuse_recipe_argv(["/srv/example/builds/bin/engine", "--port", "8010"]) is None
    refused = _refuse_recipe_argv(["/opt/engine", "--port", "8010"])
    assert refused and "engine" in refused
    assert _refuse_recipe_argv(["llama-server", "--port", "8010"]) is None


def test_generated_allowlist_covers_the_fleet_flags() -> None:
    cleaned, bad = validate_fields(
        "vllm",
        {
            "tool_parser": "qwen3_coder",
            "reasoning_parser": "glm45",
            "speculative": {"method": "dflash2", "model": "/srv/example/builds/drafter", "tokens": 4},
            "extra_flags": ["--enable-auto-tool-choice", "--trust-remote-code"],
            "attention_backend": "FLASH_ATTN",
            "max_num_batched_tokens": 8192,
            "mamba_ssm_dtype": "bfloat16",
            "quantization": "fp8",
            "load_format": "safetensors",
            "limit_mm_per_prompt": "image=2",
        },
        allowed_images=ALLOWED,
    )
    assert bad == [], bad
    assert cleaned["speculative"]["method"] == "dflash2"
    assert cleaned["tool_parser"] == "qwen3_coder"
    cleaned, bad = validate_fields(
        "llamacpp",
        {"binary": "/srv/example/builds/llama/bin/llama-server", "ld_library_path": "/srv/example/builds/llama/lib"},
        allowed_images=ALLOWED,
        mount_roots=["/srv/example/builds", "/home/USER/.cache/huggingface"],
    )
    assert bad == [], bad
    assert cleaned["binary"].endswith("llama-server")
