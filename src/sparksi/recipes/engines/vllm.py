# SPDX-License-Identifier: Apache-2.0

"""vLLM serve argv. Single node, or native two-node tensor parallel with no Ray."""

from __future__ import annotations

from sparksi.recipes.args import speculative_json
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


def _spec_mode(value: object) -> tuple[dict | None, str]:
    return normalize_speculative(value)


def _serve(spec: Spec, role: str) -> list[str]:
    command = [
        "vllm", "serve", path_for(spec, role),
        "--served-model-name", spec.model_id,
        "--host", spec.binds.get(role) or "",
        "--port", str(int(spec.port)),
    ]
    if spec.kv_cache_memory_bytes:
        command.extend(["--kv-cache-memory-bytes", str(int(spec.kv_cache_memory_bytes))])
    else:
        command.extend(["--gpu-memory-utilization", f"{float(spec.gpu_mem_util):.2f}"])
    command.extend([
        "--max-model-len", str(int(spec.ctx)),
        "--max-num-seqs", str(int(spec.max_num_seqs)),
        "--kv-cache-dtype", spec.kv_cache_dtype,
        "--enable-chunked-prefill",
    ])
    if spec.prefix_cache and not spec.forbid_prefix:
        command.append("--enable-prefix-caching")
    mode, _why = _spec_mode(spec.speculative)
    if mode and mode.get("method"):
        command.extend(["--speculative-config", speculative_json(mode)])
    if spec.tool_parser:
        command.extend(["--enable-auto-tool-choice", "--tool-call-parser", spec.tool_parser])
    if spec.reasoning_parser:
        command.extend(["--reasoning-parser", spec.reasoning_parser])
    if spec.max_num_batched_tokens:
        command.extend(["--max-num-batched-tokens", str(int(spec.max_num_batched_tokens))])
    if spec.attention_backend:
        command.extend(["--attention-backend", spec.attention_backend])
    if spec.mamba_ssm_dtype:
        command.extend(["--mamba-ssm-cache-dtype", spec.mamba_ssm_dtype])
    if spec.quantization:
        command.extend(["--quantization", spec.quantization])
    if spec.load_format:
        command.extend(["--load-format", spec.load_format])
    if spec.limit_mm_per_prompt:
        command.extend(["--limit-mm-per-prompt", spec.limit_mm_per_prompt])
    if spec.placement == "both":
        command.extend([
            "--tensor-parallel-size", "2",
            "--nnodes", "2",
            "--node-rank", "0" if role == "head" else "1",
            "--master-addr", spec.fabric_ip.get("head") or "",
            "--master-port", str(int(spec.master_port)),
        ])
        if role == "worker":
            command.append("--headless")
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
