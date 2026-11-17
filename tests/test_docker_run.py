# SPDX-License-Identifier: Apache-2.0

"""The node agent refuses docker run escapes. Generated recipes still pass."""

from __future__ import annotations

from sparksi.agent import _refuse_recipe_argv, configure
from sparksi.recipes.engines.llamacpp import build as build_llamacpp
from sparksi.recipes.engines.sglang import build as build_sglang
from sparksi.recipes.engines.trtllm import build as build_trtllm
from sparksi.recipes.engines.vllm import build as build_vllm
from sparksi.recipes.spec import Spec

HEAD = "head"
WORKER = "worker"
IMAGE = "example.invalid/vllm:1"
SGLANG = "example.invalid/sglang:1"
TRT = "example.invalid/tensorrt-llm:1"
CACHE = "/home/USER/.cache/huggingface"


def _spec(**over) -> Spec:
    base = dict(
        model_id="demo",
        placement="head",
        port=8010,
        paths={"head": "/models/demo", "worker": "/models/demo"},
        hosts={"head": HEAD, "worker": WORKER},
        binds={"head": "100.64.0.11", "worker": "100.64.0.12"},
        fabric_ip={"head": "192.0.2.11", "worker": "192.0.2.12"},
        fabric_iface={"head": "eth0", "worker": "eth0"},
        fabric_hcas={"head": ["roce0"], "worker": ["roce0"]},
        image=IMAGE,
        hf_cache=CACHE,
        ctx=8192,
        hints={"mtp": False},
        scripts={},
    )
    base.update(over)
    return Spec(**base)


def setup_module() -> None:
    configure({"mount_roots": [CACHE]})


def test_generated_recipes_pass() -> None:
    cases = [
        (_spec(), build_vllm),
        (_spec(placement="both"), build_vllm),
        (_spec(image=SGLANG), build_sglang),
        (_spec(image=SGLANG, placement="both"), build_sglang),
        (_spec(image=TRT), build_trtllm),
        (_spec(image=TRT, placement="both"), build_trtllm),
        (_spec(paths={"head": "/models/demo.gguf"}), build_llamacpp),
        (_spec(paths={"head": "/models/demo.gguf"}, hints={"mtp": True}), build_llamacpp),
    ]
    for spec, builder in cases:
        built = builder(spec)
        # Two-node fabric recipes are the RoCE placement. The launch request
        # carries that bit; the argv alone does not turn it on.
        allow = spec.placement == "both"
        for node in built.nodes:
            assert _refuse_recipe_argv(node.argv, allow_roce=allow) is None, node.argv
            if node.stop_argv:
                assert _refuse_recipe_argv(node.stop_argv) is None, node.stop_argv


def _with(argv: list[str], *extra: str) -> list[str]:
    return ["docker", "run", *extra, *argv[2:]]


def test_docker_run_refuses_host_escapes() -> None:
    base = build_vllm(_spec()).nodes[0].argv
    assert _refuse_recipe_argv(base) is None
    refused = [
        (_with(base, "--privileged"), "--privileged"),
        (_with(base, "--pid=host"), "--pid=host"),
        (_with(base, "--pid", "host"), "--pid=host"),
        (_with(base, "--cap-add", "SYS_ADMIN"), "--cap-add"),
        (_with(base, "--cap-add=ALL"), "--cap-add"),
        (_with(base, "--security-opt", "seccomp=unconfined"), "--security-opt"),
        (_with(base, "--device", "/dev/sda"), "--device"),
        (_with(base, "--device", "/dev/mem"), "--device"),
        (_with(base, "--network", "bridge"), "--network"),
        (_with(base, "--network=container:other"), "--network"),
        (_with(base, "--net", "none"), "--network"),
        (_with(base, "--entrypoint", "/bin/sh"), "shell entrypoint"),
        (_with(base, "-v", "/:/host"), "refusing mount /"),
        (_with(base, "-v", "/etc:/etc"), "refusing mount /etc"),
        (_with(base, "--volume", "/root:/root"), "refusing mount /root"),
        (_with(base, "-v", "/var/run/docker.sock:/var/run/docker.sock"), "docker.sock"),
        (_with(base, "-v", "/tmp:/tmp"), "outside model roots"),
        (_with(base, "--mount", "type=bind,source=/,target=/host"), "refusing mount /"),
        (_with(base, "--mount", "type=bind,source=/etc,target=/etc"), "refusing mount /etc"),
    ]
    for argv, needle in refused:
        reason = _refuse_recipe_argv(argv)
        assert reason and needle in reason, (argv, reason)
    assert _refuse_recipe_argv(_with(base, "--device", "/dev/nvidia0")) is None
    assert _refuse_recipe_argv(_with(base, "--device", "/dev/dri/card0")) is None
    assert _refuse_recipe_argv(_with(base, "--network", "host")) is None
    assert _refuse_recipe_argv(_with(base, "-v", f"{CACHE}:/root/.cache/huggingface")) is None


def test_entrypoint_is_allowed_only_for_two_node_trtllm() -> None:
    both = build_trtllm(_spec(image=TRT, placement="both"))
    for node in both.nodes:
        argv = ["docker", "run", "--entrypoint", "/bin/true", *node.argv[2:]]
        assert _refuse_recipe_argv(argv, allow_roce=True) is None, (node.role, _refuse_recipe_argv(argv, allow_roce=True))
    single = build_trtllm(_spec(image=TRT)).nodes[0].argv
    refused = _refuse_recipe_argv(["docker", "run", "--entrypoint", "/bin/true", *single[2:]])
    assert refused == "refusing --entrypoint"
