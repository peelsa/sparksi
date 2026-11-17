# SPDX-License-Identifier: Apache-2.0

"""SGLang ``launch_server`` in a pinned container. Two-node uses the same NCCL env as vLLM."""

from __future__ import annotations

from sparksi.recipes.engines.common import (
    container_name,
    docker_argv,
    docker_stop,
    fabric_env,
    launch_for,
    path_for,
    roles_for,
)
from sparksi.recipes.spec import Built, Spec
from sparksi.recipes.validate import normalize_speculative


def _serve(spec: Spec, role: str) -> list[str]:
    command = [
        "python", "-m", "sglang.launch_server",
        "--model-path", path_for(spec, role),
        "--host", spec.binds.get(role) or "",
        "--port", str(int(spec.port)),
        "--mem-fraction-static", f"{float(spec.gpu_mem_util):.2f}",
        "--context-length", str(int(spec.ctx)),
    ]
    mode, _why = normalize_speculative(spec.speculative)
    if mode and mode.get("method"):
        method = str(mode["method"])
        algo = "NEXTN" if method in ("mtp", "nextn") else method
        command.extend(["--speculative-algorithm", algo])
        if mode.get("tokens"):
            command.extend(["--speculative-num-steps", str(int(mode["tokens"]))])
        if mode.get("model"):
            command.extend(["--speculative-draft-model-path", str(mode["model"])])
    if spec.tool_parser:
        command.extend(["--tool-call-parser", spec.tool_parser])
    if spec.reasoning_parser:
        command.extend(["--reasoning-parser", spec.reasoning_parser])
    if spec.placement == "both":
        command.extend([
            "--tp", "2",
            "--nnodes", "2",
            "--node-rank", "0" if role == "head" else "1",
            "--dist-init-addr", f"{spec.fabric_ip.get('head') or ''}:{int(spec.dist_port)}",
        ])
    command.extend(spec.extra_flags)
    return command


def build(spec: Spec) -> Built:
    name = container_name(spec)
    nodes = []
    for role in roles_for(spec.placement):
        role_env = fabric_env(spec, role) if spec.placement == "both" else {}
        argv = docker_argv(spec, _serve(spec, role), env=role_env)
        nodes.append(launch_for(spec, role, argv, role_env, container=name, stop=docker_stop(name)))
    return Built(nodes=nodes, order=roles_for(spec.placement), experimental=False, health_paths=["/health", "/v1/models"])
