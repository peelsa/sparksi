# SPDX-License-Identifier: Apache-2.0

"""Production startup loads mount roots, and a second process can keep its own state."""

from __future__ import annotations

import json
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sparksi.agent import _refuse_recipe_argv, docker_option_refusal
from sparksi.cli import build_parser, main
from sparksi.config import CFG, reload
from sparksi.recipes.memory import reset_for_tests as reset_memory
from sparksi.recipes.runs import reset_for_tests as reset_runs, runs_dir
from sparksi.recipes.service import reset_facts_cache
from sparksi.serve import LISTENERS, STATE, _STOP, listen_targets, start_announce

IMAGE = "example.invalid/vllm:1"


def _quote(text: str) -> str:
    return json.dumps(str(text))


def test_serve_flags_keep_their_defaults() -> None:
    bare = build_parser().parse_args(["serve"])
    assert bare.port == 7500
    assert bare.state_dir == ""
    assert bare.bind is None
    assert bare.no_announce is False
    chosen = build_parser().parse_args([
        "serve", "--port", "7601", "--state-dir", "/srv/example/state",
        "--bind", "127.0.0.1", "--bind", "100.64.0.2,100.64.0.3", "--no-announce",
    ])
    assert chosen.port == 7601
    assert chosen.state_dir == "/srv/example/state"
    assert chosen.bind == ["127.0.0.1", "100.64.0.2,100.64.0.3"]
    assert chosen.no_announce is True


def test_shell_entrypoint_ignores_placeholder_mount_roots() -> None:
    """A script under the baked-in example path is refused until roots are loaded."""
    import sparksi.agent as agent

    saved_roots = list(agent._CFG["mount_roots"])
    saved_flag = agent._MOUNTS_CONFIGURED
    argv = [
        "docker", "run", "--network", "host",
        "-v", "/srv/example/builds/start.sh:/start.sh",
        "--entrypoint", "bash",
        IMAGE,
        "/start.sh",
    ]
    try:
        # The stand-in list would accept this path. The shell check must not.
        agent._CFG["mount_roots"] = [
            "/home/USER/.cache/huggingface",
            "/srv/example/cache",
            "/srv/example/builds",
            "/srv/example/worker",
        ]
        agent._MOUNTS_CONFIGURED = False
        refused = _refuse_recipe_argv(argv)
        assert refused == "refusing shell entrypoint bash: mount roots are not configured"
        empty = docker_option_refusal("--entrypoint", "bash", allow_roce=False, argv=argv, roots=[])
        assert empty == "refusing shell entrypoint bash: mount roots are not configured"
        allowed = docker_option_refusal(
            "--entrypoint", "bash", allow_roce=False, argv=argv, roots=["/srv/example/builds"],
        )
        assert allowed is None
    finally:
        agent._CFG["mount_roots"] = saved_roots
        agent._MOUNTS_CONFIGURED = saved_flag


def _role_table(mounts: list[str], args: list[str], entry: str, container: str) -> str:
    mount_text = ", ".join(_quote(item) for item in mounts)
    arg_text = ", ".join(_quote(item) for item in args)
    return "\n".join([
        'runtime = "docker"',
        f"image = {_quote(IMAGE)}",
        'network = "host"',
        'gpus = "all"',
        f"container = {_quote(container)}",
        f"entrypoint = {_quote(entry)}",
        f"args = [{arg_text}]",
        f"mounts = [{mount_text}]",
    ])


def _write_recipe(folder, model_id: str, placement: str, port: int, roles: dict[str, str]) -> None:
    lines = [
        'kind = "pinned"',
        'engine = "vllm"',
        f"placement = {_quote(placement)}",
        f"port = {port}",
        'imported_from = "container:example at 2026-11-01T00:00:00Z"',
    ]
    for name, body in roles.items():
        lines.extend(["", f"[roles.{name}]", body])
    (folder / f"{model_id}.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_service_startup_plans_with_configured_mount_roots(tmp_path, monkeypatch) -> None:
    """The serve entry point loads a site file the way a process start does.

    Plan is the HTTP route. A bash script under a node's mount_roots is READY.
    The same script outside those roots is refused. Docker and two-node plans
    go through that same startup.
    """
    import sparksi.agent as agent

    cache = tmp_path / "cache"
    roots = tmp_path / "roots"
    outside = tmp_path / "outside"
    recipes = tmp_path / "recipes"
    state = tmp_path / "state"
    for folder in (cache, roots, outside, recipes, state):
        folder.mkdir()
    (cache / "hub").mkdir()
    script = roots / "start.sh"
    script.write_text("#!/bin/sh\nexec vllm serve /models/example --port 8015\n", encoding="utf-8")
    outside_script = outside / "start.sh"
    outside_script.write_text("#!/bin/sh\nexec vllm serve /models/example --port 8017\n", encoding="utf-8")
    cache_mount = f"{cache}:/root/.cache/huggingface"
    script_mount = f"{script}:/start.sh"
    outside_mount = f"{outside_script}:/start.sh"
    _write_recipe(recipes, "docker-demo", "head", 8014, {
        "head": _role_table([cache_mount], ["serve", "/models/example", "--port", "8014"], "vllm", "sparksi-docker-demo"),
    })
    _write_recipe(recipes, "script-demo", "head", 8015, {
        "head": _role_table(
            [cache_mount, script_mount], ["/start.sh", "--port", "8015"], "bash", "sparksi-script-demo",
        ),
    })
    _write_recipe(recipes, "pair-demo", "both", 8016, {
        "head": _role_table([cache_mount], ["serve", "/models/example", "--port", "8016"], "vllm", "sparksi-pair-head"),
        "worker": _role_table([cache_mount], ["serve", "/models/example", "--port", "8016"], "vllm", "sparksi-pair-worker"),
    })
    _write_recipe(recipes, "outside-demo", "head", 8017, {
        "head": _role_table(
            [cache_mount, outside_mount], ["/start.sh", "--port", "8017"], "bash", "sparksi-outside-demo",
        ),
    })

    site = tmp_path / "sparksi.toml"
    site.write_text(
        "\n".join([
            "[controller]",
            'name = "controller"',
            'bind = ["127.0.0.1"]',
            'bind_deny = ["0.0.0.0", "::"]',
            f"recipes_dir = {_quote(recipes)}",
            "",
            "[[nodes]]",
            'id = "a"',
            'name = "spark-a"',
            'ssh = "local"',
            'tailscale = "100.64.0.11"',
            "uma_gib = 40",
            f"mount_roots = [{_quote(roots)}]",
            "",
            "[[nodes]]",
            'id = "b"',
            'name = "spark-b"',
            'ssh = "local"',
            'tailscale = "100.64.0.12"',
            "uma_gib = 40",
            f"mount_roots = [{_quote(roots)}]",
            "",
            "[paths]",
            'home = "/home/USER"',
            f"hf_hub = {_quote(cache / 'hub')}",
            "",
        ]),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    weight = cache / "weights"
    pair_weight = cache / "pair"
    catalog.write_text(
        "\n".join([
            "schema_version: 1",
            "host:",
            "  uma_gib: 40",
            "  bind_allow: ['127.0.0.1']",
            "  bind_deny: ['0.0.0.0', '::']",
            "  reserved_ports: [8000, 8001, 8004, 8007, 8011, 8188]",
            "tenants:",
            "  os:",
            "    always: true",
            "    measure: catalog_min",
            "    min_gib: 6",
            "residents: []",
            "recipes: {}",
            "hosts:",
            "  spark-a:",
            "    uma_gib: 40",
            "    floor_gib: 6",
            "    margin_gib: 4",
            "    ssh: local",
            "  spark-b:",
            "    uma_gib: 40",
            "    floor_gib: 6",
            "    margin_gib: 4",
            "    ssh: local",
            "library:",
            "  - id: docker-demo",
            "    title: Docker",
            "    format: fp8",
            "    gib: 8",
            "    engines: [vllm]",
            "    port: 8014",
            f"    local: {{spark-a: {_quote(weight)}}}",
            "  - id: script-demo",
            "    title: Script",
            "    format: fp8",
            "    gib: 8",
            "    engines: [vllm]",
            "    port: 8015",
            f"    local: {{spark-a: {_quote(weight)}}}",
            "  - id: pair-demo",
            "    title: Pair",
            "    format: fp8",
            "    gib: 30",
            "    engines: [vllm]",
            "    port: 8016",
            f"    local: {{spark-a: {_quote(pair_weight)}, spark-b: {_quote(pair_weight)}}}",
            "  - id: outside-demo",
            "    title: Outside",
            "    format: fp8",
            "    gib: 8",
            "    engines: [vllm]",
            "    port: 8017",
            f"    local: {{spark-a: {_quote(weight)}}}",
            "",
        ]),
        encoding="utf-8",
    )

    class Runner:
        def invoke(self, host, req, timeout=20):
            paths = req.get("paths") if isinstance(req, dict) else None
            found = {str(path): {"exists": True} for path in (paths or [])}
            return {"ok": True, "paths": found, "listening": {}, "http": {}}

        def capture(self, host, argv, timeout=8):
            parts = [str(item) for item in argv]
            if parts[:1] == ["cat"] and len(parts) == 2:
                path = parts[1]
                if path == str(script):
                    return {"ok": True, "code": 0, "stdout": script.read_text(encoding="utf-8"), "stderr": ""}
                if path == str(outside_script):
                    return {"ok": True, "code": 0, "stdout": outside_script.read_text(encoding="utf-8"), "stderr": ""}
                return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}
            if parts[:3] == ["docker", "image", "inspect"]:
                return {"ok": True, "code": 0, "stdout": "{}", "stderr": ""}
            return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": None}

    saved_roots = list(agent._CFG["mount_roots"])
    saved_flag = agent._MOUNTS_CONFIGURED
    saved_runner = STATE.runner
    saved_lock = getattr(STATE, "lock", None)
    saved_catalog = getattr(STATE, "catalog", None)
    monkeypatch.setenv("SPARKSI_CONFIG", str(site))
    monkeypatch.setenv("SPARKSI_CATALOG", str(catalog))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(state))
    monkeypatch.delenv("SPARKSI_RECIPES_DIR", raising=False)
    reset_runs()
    reset_memory()
    reset_facts_cache()
    STATE.runner = Runner()

    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    box: dict = {}

    def _run() -> None:
        try:
            box["code"] = main(["serve", "--port", str(port), "--state-dir", str(state)])
        except Exception as exc:  # noqa: BLE001 — the assertion reports the startup failure
            box["error"] = exc

    thread = threading.Thread(target=_run, name="sparksi-test-serve", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if box.get("error") or box.get("code") not in (None, 0):
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError(f"serve did not listen: {box}")
        if box.get("error") or (box.get("code") not in (None, 0)):
            raise AssertionError(f"serve exited: {box}")

        assert agent._MOUNTS_CONFIGURED is True
        assert str(roots) in agent._CFG["mount_roots"]
        assert "/srv/example/cache" not in agent._CFG["mount_roots"]
        assert STATE.lock is not None
        assert STATE.lock.path.parent == state
        assert runs_dir() == state / "recipe-runs"
        assert CFG.controller.get("recipes_dir") == str(recipes)

        def plan(model_id: str) -> tuple[int, dict]:
            req = Request(
                f"http://127.0.0.1:{port}/api/models/{model_id}/plan",
                data=b"{}",
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(req, timeout=30) as resp:
                    return resp.status, json.loads(resp.read().decode())
            except HTTPError as exc:
                return exc.code, json.loads(exc.read().decode())

        docker_code, docker = plan("docker-demo")
        assert docker_code == 200, docker.get("conflicts")
        assert docker["verdict"] == "READY"
        docker_argv = docker["recipe"]["nodes"][0]["argv"]
        assert docker_argv[docker_argv.index("--entrypoint") + 1] == "vllm"

        script_code, scripted = plan("script-demo")
        assert script_code == 200, scripted.get("conflicts")
        assert scripted["verdict"] == "READY"
        assert scripted["kind"] == "pinned"
        script_argv = scripted["recipe"]["nodes"][0]["argv"]
        assert script_argv[script_argv.index("--entrypoint") + 1] == "bash"
        assert "/start.sh" in script_argv

        pair_code, pair = plan("pair-demo")
        assert pair_code == 200, pair.get("conflicts")
        assert pair["verdict"] == "READY"
        assert pair["placement"] == "both"
        assert [node["role"] for node in pair["recipe"]["nodes"]] == ["worker", "head"]

        outside_code, refused = plan("outside-demo")
        assert outside_code == 400, refused
        assert refused["verdict"] == "REFUSE"
        assert any("not under a mount root" in item for item in refused["conflicts"]), refused["conflicts"]
    except URLError as exc:
        raise AssertionError(f"serve did not answer: {box}") from exc
    finally:
        _STOP.set()
        for srv in list(LISTENERS):
            srv.shutdown()
        thread.join(timeout=8)
        LISTENERS.clear()
        STATE.runner = saved_runner
        STATE.lock = saved_lock
        STATE.catalog = saved_catalog
        monkeypatch.delenv("SPARKSI_CONFIG", raising=False)
        monkeypatch.delenv("SPARKSI_CATALOG", raising=False)
        monkeypatch.delenv("SPARKSI_STATE_DIR", raising=False)
        reload()
        agent._CFG["mount_roots"] = saved_roots
        agent._MOUNTS_CONFIGURED = saved_flag
        reset_runs()
        reset_memory()
        reset_facts_cache()


def test_bind_keeps_staging_on_loopback_without_announce(tmp_path, monkeypatch) -> None:
    """``--bind 127.0.0.1`` replaces the catalog addresses. ``--no-announce`` stays quiet."""
    import sparksi.agent as agent

    site = tmp_path / "sparksi.toml"
    site.write_text(
        "\n".join([
            "[controller]",
            'name = "controller"',
            'bind = ["127.0.0.1", "100.64.0.1"]',
            'bind_deny = ["0.0.0.0", "::"]',
            "",
        ]),
        encoding="utf-8",
    )
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        "\n".join([
            "schema_version: 1",
            "host:",
            "  bind_allow: ['127.0.0.1', '100.64.0.1']",
            "  bind_deny: ['0.0.0.0', '::']",
            "  reserved_ports: [8000, 8188]",
            "tenants:",
            "  os:",
            "    always: true",
            "    measure: catalog_min",
            "    min_gib: 6",
            "residents: []",
            "recipes: {}",
            "",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr("sparksi.serve.on_controller", lambda: True)
    wider = listen_targets(None, {
        "controller": {"bind_allow": ["127.0.0.1", "100.64.0.1"], "bind_deny": ["0.0.0.0", "::"]},
    })
    assert wider == ["127.0.0.1", "100.64.0.1"]
    assert listen_targets(["127.0.0.1"], {}) == ["127.0.0.1"]

    calls: list[str] = []
    monkeypatch.setattr("sparksi.announce_rx.start_listener", lambda: calls.append("on"))
    start_announce(False)
    assert calls == []
    start_announce(True)
    assert calls == ["on"]
    calls.clear()

    class Runner:
        def invoke(self, host, req, timeout=20):
            return {"ok": True, "paths": {}}

        def capture(self, host, argv, timeout=8):
            return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": None}

    saved_roots = list(agent._CFG["mount_roots"])
    saved_flag = agent._MOUNTS_CONFIGURED
    saved_runner = STATE.runner
    monkeypatch.setenv("SPARKSI_CONFIG", str(site))
    monkeypatch.setenv("SPARKSI_CATALOG", str(catalog))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    STATE.runner = Runner()
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    box: dict = {}

    def _run() -> None:
        try:
            box["code"] = main([
                "serve", "--port", str(port), "--state-dir", str(tmp_path / "state"),
                "--bind", "127.0.0.1", "--no-announce",
            ])
        except Exception as exc:  # noqa: BLE001
            box["error"] = exc

    thread = threading.Thread(target=_run, name="sparksi-bind-test", daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if box.get("error") or box.get("code") not in (None, 0):
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                time.sleep(0.05)
        else:
            raise AssertionError(f"serve did not listen: {box}")
        addrs = sorted(srv.server_address[0] for srv in LISTENERS)
        assert addrs == ["127.0.0.1"], addrs
        assert calls == []
    finally:
        _STOP.set()
        for srv in list(LISTENERS):
            srv.shutdown()
        thread.join(timeout=8)
        LISTENERS.clear()
        STATE.runner = saved_runner
        monkeypatch.delenv("SPARKSI_CONFIG", raising=False)
        monkeypatch.delenv("SPARKSI_CATALOG", raising=False)
        monkeypatch.delenv("SPARKSI_STATE_DIR", raising=False)
        reload()
        agent._CFG["mount_roots"] = saved_roots
        agent._MOUNTS_CONFIGURED = saved_flag
