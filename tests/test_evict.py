# SPDX-License-Identifier: Apache-2.0

"""Pre-launch page-cache eviction is optional, argv-only, and never blocks a start."""

from __future__ import annotations

import os

from sparksi.config import CFG
import sparksi.evict as evict
from sparksi.evict import advise_tree, evict_argv, eviction_block, main
from sparksi.recipes.importer import parse_script
from sparksi.recipes.pinned import resolve_evict_path
from sparksi.recipes.runs import public_run, reset_for_tests, start_rendered
from sparksi.recipes.store import load_recipe, save_recipe

IMAGE = "example.invalid/vllm:patched"


def test_evict_command_is_argv_only_and_confined(tmp_path) -> None:
    path = "/srv/example/cache/weights"
    argv = evict_argv(path)
    assert argv == ["python3", "-m", "sparksi.evict", path]
    assert "-c" not in argv
    assert "sh" not in argv
    assert "sudo" not in argv
    assert "drop_caches" not in " ".join(argv)
    assert main(["sparksi.evict", "/tmp/weights"]) == 1
    assert main(["sparksi.evict"]) == 2
    outside = resolve_evict_path(
        {"runtime": "native", "args": ["-m", "/tmp/weights"], "mounts": []},
        "head",
    )
    assert outside == ""
    inside = resolve_evict_path(
        {
            "runtime": "docker",
            "args": ["vllm", "serve", "/opt/weights"],
            "mounts": ["/srv/example/cache/weights:/opt/weights"],
        },
        "head",
    )
    assert inside == "/srv/example/cache/weights"
    weight = tmp_path / "layer.safetensors"
    weight.write_bytes(b"weights")
    calls: list[str] = []

    def advise(fd, offset, length, advice):
        calls.append(os.read(fd, 16).decode())

    real_advise = os.posix_fadvise
    real_refuse = evict._refused
    os.posix_fadvise = advise
    evict._refused = lambda path: ""
    try:
        assert advise_tree(str(weight)) == 0
    finally:
        os.posix_fadvise = real_advise
        evict._refused = real_refuse
    assert calls == ["weights"]
    source = open(evict.__file__, encoding="utf-8").read()
    assert "drop_caches" not in source
    assert "sudo " not in source


def test_evict_failure_does_not_block_launch(monkeypatch, tmp_path) -> None:
    import threading

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    reset_for_tests()
    monkeypatch.setattr(threading, "Thread", lambda *args, **kwargs: type("T", (), {"start": lambda self: None})())
    seen: list[list[str]] = []

    class Runner:
        def run_argv(self, host, argv, timeout=60):
            seen.append(list(argv))
            return {"ok": False, "code": 1, "error": "advise failed", "stdout": "", "stderr": "advise failed"}

        def invoke(self, host, req, timeout=120):
            return {"ok": True, "phase": "running"}

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
            "host": CFG.head.name,
            "argv": ["docker", "run", "-d", "--network", "host", IMAGE, "vllm", "serve", "--port", "8014"],
            "env": {},
            "container": "sparksi-example",
            "stop_argv": ["docker", "rm", "-f", "sparksi-example"],
            "evict_page_cache": True,
            "evict_path": "/srv/example/cache/weights",
        }],
    }
    run = start_rendered(recipe, Runner(), sync=False)
    assert run["phase"] != "failed"
    assert seen == [["python3", "-m", "sparksi.evict", "/srv/example/cache/weights"]]
    public = public_run(run)
    warning = public["nodes"][0]["evict_warning"]
    assert warning.startswith("page cache not cleared on head:")
    assert "advise failed" in warning

    skipped: list[list[str]] = []

    class Quiet:
        def run_argv(self, host, argv, timeout=60):
            skipped.append(list(argv))
            return {"ok": True, "code": 0}

        def invoke(self, host, req, timeout=120):
            return {"ok": True}

    bare = dict(recipe)
    bare["nodes"] = [dict(recipe["nodes"][0], evict_path="/tmp/weights")]
    quiet = start_rendered(bare, Quiet(), sync=False)
    assert quiet["phase"] != "failed"
    assert skipped == []
    assert "outside mount roots" in public_run(quiet)["nodes"][0]["evict_warning"]
    reset_for_tests()


def test_evict_field_round_trips_and_importer_maps_it(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path))
    save_recipe("example", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "evict_page_cache": True,
        "roles": {"head": {"runtime": "docker", "image": IMAGE, "args": ["--port", "8014"], "env": {}}},
    })
    loaded = load_recipe("example")
    assert loaded["evict_page_cache"] is True

    env_path = "/srv/example/builds/stack/.env"
    runner_answers = {
        ("test", "-f", env_path): {"ok": True, "code": 0, "stdout": "", "stderr": ""},
        ("cat", env_path): {"ok": True, "code": 0, "stdout": "EVICT_PAGE_CACHE=true\nIMAGE=example.invalid/vllm:patched\n", "stderr": ""},
    }

    class Runner:
        def capture(self, host, argv, timeout=8):
            return runner_answers.get(tuple(argv), {"ok": False, "code": 1, "stdout": "", "error": "missing"})

    role = parse_script(
        "#!/bin/sh\ncd /srv/example/builds/stack\nset -a\n. ./.env\nset +a\n"
        'exec docker run --network host "$IMAGE" vllm serve /srv/example/cache/weights --port 8014\n',
        runner=Runner(),
        host=CFG.head.name,
        cwd="/srv/example/builds/stack",
        path="/srv/example/builds/stack/start.sh",
    )
    assert role is not None
    assert role["evict_page_cache"] is True
    assert "EVICT_PAGE_CACHE" not in role["env"]


def test_repo_and_revision_resolve_one_snapshot(monkeypatch, tmp_path) -> None:
    """A saved repo and revision name one snapshot. The cache mount is not a fallback."""
    import threading

    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    revision = "ab" * 20
    save_recipe("example", {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "repo": "org/example",
        "revision": revision,
        "evict_page_cache": True,
        "roles": {"head": {"runtime": "docker", "image": IMAGE, "args": ["vllm", "serve", "--port", "8014"], "env": {}}},
    })
    loaded = load_recipe("example")
    assert loaded["repo"] == "org/example"
    assert loaded["revision"] == revision
    cache = "/home/USER/.cache/huggingface"
    role = {
        "runtime": "docker",
        "args": ["vllm", "serve", "--port", "8014"],
        "mounts": [f"{cache}:/root/.cache/huggingface"],
    }
    assert resolve_evict_path(role, "head") == ""
    path = resolve_evict_path(role, "head", repo=loaded["repo"], revision=loaded["revision"])
    assert path.endswith(f"/models--org--example/snapshots/{revision}")
    assert path != cache
    assert not path.rstrip("/").endswith("/hub")

    many = tmp_path / "hub"
    for name in ("models--one--a", "models--two--b"):
        folder = many / name / "snapshots" / "rev"
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(b"weights")
    assert eviction_block(str(many)) == "directory holds more than one model's snapshots"
    calls: list[str] = []

    def advise(fd, offset, length, advice):
        calls.append("advised")

    real_advise = os.posix_fadvise
    real_refuse = evict._refused
    os.posix_fadvise = advise
    evict._refused = lambda item: ""
    try:
        assert advise_tree(str(many)) == 1
    finally:
        os.posix_fadvise = real_advise
        evict._refused = real_refuse
    assert calls == []

    reset_for_tests()
    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    seen: list[list[str]] = []

    class Runner:
        def run_argv(self, host, argv, timeout=60):
            seen.append(list(argv))
            return {"ok": True, "code": 0}

        def invoke(self, host, req, timeout=120):
            return {"ok": True, "phase": "running"}

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
            "host": CFG.head.name,
            "argv": ["docker", "run", "-d", "--network", "host", IMAGE, "vllm", "serve", "--port", "8014"],
            "env": {},
            "container": "sparksi-example",
            "stop_argv": ["docker", "rm", "-f", "sparksi-example"],
            "evict_page_cache": True,
            "evict_path": "",
        }],
    }
    run = start_rendered(recipe, Runner(), sync=False)
    assert run["phase"] != "failed"
    assert seen == []
    assert "no specific model snapshot" in public_run(run)["nodes"][0]["evict_warning"]
    reset_for_tests()
