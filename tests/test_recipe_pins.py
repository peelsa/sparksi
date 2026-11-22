# SPDX-License-Identifier: Apache-2.0

"""A recipe records the image build and native binary it runs, and its own KV reserve."""

from __future__ import annotations

import json

import pytest

from sparksi.config import CFG
from sparksi.recipes import service
from sparksi.recipes.pins import pin_conflicts
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.recipes.service import plan_model, recipe_reserve_gib
from sparksi.recipes.store import save_recipe

MODEL = "demo"
IMAGE = "example.invalid/vllm:1"
BUILD = "sha256:" + "a" * 64
OTHER = "sha256:" + "b" * 64
BINARY = "/srv/example/bin/server"
DIGEST = "c" * 64


def _catalog() -> dict:
    host = CFG.head.name
    return {
        "library": [{
            "id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "engines": ["vllm"],
            "local": {host: "/srv/example/cache/weights"}, "port": 8014,
        }],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
        "exclusive_groups": {},
        "never_touch": {},
    }


def _recipe(**head) -> dict:
    return {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-11-01T00:00:00Z",
        "port_note": "",
        "rendezvous_ports": [],
        "roles": {"head": {
            "runtime": "docker",
            "image": IMAGE,
            "env": {},
            "mounts": ["/srv/example/cache:/root/.cache/huggingface"],
            "network": "host",
            "gpus": "all",
            "container": "sparksi-example",
            "args": ["vllm", "serve", "/models/example", "--host", "0.0.0.0", "--port", "8014"],
            "devices": [],
            **head,
        }},
    }


class _Spark:
    """Read-only answers: every path exists, the image has build ``image``, the binary hashes to ``digest``."""

    def __init__(self, image: str = BUILD, digest: str | None = DIGEST, down: bool = False):
        self.image, self.digest, self.down = image, digest, down

    def invoke(self, host, req, timeout=20):
        op = req.get("op")
        if op == "probe":
            return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}
        if op == "file_sha256":
            if self.down:
                return {"ok": False, "error": "ssh failed"}
            if self.digest is None:
                return {"ok": True, "exists": False, "sha256": ""}
            return {"ok": True, "exists": True, "sha256": self.digest}
        raise AssertionError(f"nothing may run: {op}")

    def capture(self, host, argv, timeout=8):
        if list(argv[:3]) == ["docker", "image", "inspect"]:
            if self.down:
                return {"ok": False, "code": 255, "stdout": "", "stderr": "ssh failed"}
            return {"ok": True, "code": 0, "stdout": json.dumps([{"Id": self.image}]), "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "not here"}

    def run_argv(self, host, argv, timeout=3600):
        raise AssertionError("nothing may run")

    def agent_call(self, host, op, payload=None, timeout=60):
        raise AssertionError("nothing may run")


@pytest.fixture()
def recipes(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    reset_runs()
    yield
    reset_runs()


def test_a_matching_image_build_plans(recipes):
    save_recipe(MODEL, _recipe(image_id=BUILD))
    code, planned = plan_model(_catalog(), _Spark(), MODEL, {})
    assert code == 200, planned["conflicts"]


def test_a_retagged_image_refuses_the_plan(recipes):
    save_recipe(MODEL, _recipe(image_id=BUILD))
    code, planned = plan_model(_catalog(), _Spark(image=OTHER), MODEL, {})
    host = CFG.head.name
    assert code == 400
    assert f"{host} has a different build of image {IMAGE} than this recipe records, so it will not start" in planned["conflicts"]


def test_a_recipe_without_a_build_is_not_checked(recipes):
    save_recipe(MODEL, _recipe())
    code, planned = plan_model(_catalog(), _Spark(image=OTHER), MODEL, {})
    assert code == 200, planned["conflicts"]


def test_a_badly_written_build_refuses():
    assert pin_conflicts(None, _recipe(image_id="v1")) == [
        "the head role's image_id must be sha256: and 64 hex characters, with an image",
    ]


def test_a_native_binary_is_checked_by_sha256():
    native = _recipe(binary=BINARY, binary_sha256=DIGEST)
    host = CFG.head.name
    assert pin_conflicts(_Spark(), native) == []
    assert pin_conflicts(_Spark(digest="d" * 64), native) == [
        f"{BINARY} on {host} does not match the SHA-256 this recipe records, so it will not start",
    ]
    assert pin_conflicts(_Spark(digest=None), native) == [
        f"{BINARY} on {host} does not match the SHA-256 this recipe records, so it will not start",
    ]
    assert pin_conflicts(_Spark(down=True), native) == [f"could not check {BINARY} on {host}"]


def test_a_spark_that_does_not_answer_refuses():
    host = CFG.head.name
    assert pin_conflicts(_Spark(down=True), _recipe(image_id=BUILD)) == [f"could not check the build of image {IMAGE} on {host}"]


def test_the_recipe_kv_reserve_wins_over_the_card(recipes, monkeypatch):
    seen: list[float] = []
    real = service._memory_conflicts

    def spy(card, *args, **kwargs):
        seen.append(float(card.kv_reserve_gib))
        return real(card, *args, **kwargs)

    monkeypatch.setattr(service, "_memory_conflicts", spy)
    save_recipe(MODEL, {**_recipe(), "memory": {"kv_reserve_gib": 15}})
    code, planned = plan_model(_catalog(), _Spark(), MODEL, {})
    assert seen == [15.0]
    save_recipe(MODEL, _recipe())
    seen.clear()
    plan_model(_catalog(), _Spark(), MODEL, {})
    assert seen and seen[0] != 15.0


def test_a_bad_kv_reserve_is_one_sentence():
    assert recipe_reserve_gib({}) == (None, "")
    assert recipe_reserve_gib({"memory": {"kv_reserve_gib": "lots"}}) == (None, "memory.kv_reserve_gib must be a number of GiB")
    assert recipe_reserve_gib({"memory": {"kv_reserve_gib": 500}})[1] == "memory.kv_reserve_gib must be between 0 and 128 GiB"
    assert recipe_reserve_gib({"memory": {"kv_reserve_gib": 0}}) == (0.0, "")
