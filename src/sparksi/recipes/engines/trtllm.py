# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM serve argv.

Single node follows ``trtllm-serve --backend pytorch``. Two-node follows the
DGX Spark playbook's ``mpirun trtllm-llmapi-launch trtllm-serve --tp_size 2``
line. The playbook only reaches the second Spark after a shell entrypoint
installs OpenSSH and starts sshd; that script is not an argv list, so the
two-node recipe is marked experimental and does not run it.
"""

from __future__ import annotations

from sparksi.recipes.engines.common import (
    container_name,
    docker_stop,
    fabric_env,
    launch_for,
    path_for,
    roles_for,
)
from sparksi.recipes.spec import Built, Spec

EXPERIMENTAL_BOTH = (
    "experimental: two-node TensorRT-LLM serves with the DGX Spark playbook command "
    "mpirun trtllm-llmapi-launch trtllm-serve --tp_size 2 --backend pytorch "
    "in nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6. The playbook reaches the worker "
    "only after a shell entrypoint installs OpenSSH and starts sshd. That entrypoint "
    "is not an argv list, so this recipe does not install packages or start sshd. "
    "The worker container is started with the playbook fabric env and stays up; "
    "the head runs the verified mpirun line."
)


def _serve_flags(spec: Spec, role: str) -> list[str]:
    return [
        path_for(spec, role),
        "--host", spec.binds.get("head") or "",
        "--port", str(int(spec.port)),
        "--backend", "pytorch",
        "--max_seq_len", str(int(spec.ctx)),
        "--kv_cache_free_gpu_memory_fraction", f"{float(spec.gpu_mem_util):.2f}",
    ]


def _single(spec: Spec, role: str) -> list[str]:
    command = ["trtllm-serve", *_serve_flags(spec, role), *spec.extra_flags]
    return _docker(spec, role, command, env={})


def _docker(spec: Spec, role: str, command: list[str], env: dict[str, str], extra: list[str] | None = None) -> list[str]:
    argv = [
        "docker", "run", "-d",
        "--name", container_name(spec),
        "--gpus", "all",
        "--ipc", "host",
        "--network", "host",
        "--ulimit", "memlock=-1",
        "--shm-size", "16g",
        "-e", "HF_HUB_OFFLINE=1",
    ]
    for key, value in env.items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(extra or [])
    argv.extend(["-v", f"{spec.hf_cache}:/root/.cache/huggingface", spec.image])
    argv.extend(command)
    return argv


def _both_env(spec: Spec, role: str) -> dict[str, str]:
    env = fabric_env(spec, role)
    iface = spec.fabric_iface.get(role) or ""
    env["OMPI_MCA_btl_tcp_if_include"] = iface
    env["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
    env["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
    return env


def _head_both(spec: Spec) -> list[str]:
    head_ip = spec.fabric_ip.get("head") or ""
    worker_ip = spec.fabric_ip.get("worker") or ""
    command = [
        "mpirun",
        "--host", f"{head_ip}:1,{worker_ip}:1",
        "-n", "2",
        "--map-by", "ppr:1:node",
        "trtllm-llmapi-launch",
        "trtllm-serve",
        *_serve_flags(spec, "head"),
        "--tp_size", "2",
        *spec.extra_flags,
    ]
    extra = [
        "--ulimit", "stack=67108864",
        "--device", "/dev/infiniband:/dev/infiniband",
    ]
    return _docker(spec, "head", command, env=_both_env(spec, "head"), extra=extra)


def _worker_both(spec: Spec) -> list[str]:
    """Hold the playbook container. sshd is not started; see ``EXPERIMENTAL_BOTH``."""
    extra = [
        "--ulimit", "stack=67108864",
        "--device", "/dev/infiniband:/dev/infiniband",
    ]
    return _docker(spec, "worker", ["sleep", "infinity"], env=_both_env(spec, "worker"), extra=extra)


def build(spec: Spec) -> Built:
    name = container_name(spec)
    if spec.placement != "both":
        role = "worker" if spec.placement == "worker" else "head"
        argv = _single(spec, role)
        node = launch_for(spec, role, argv, {}, container=name, stop=docker_stop(name))
        return Built(nodes=[node], order=[role], experimental=False, health_paths=["/health", "/v1/models"])
    nodes = []
    for role in roles_for("both"):
        argv = _head_both(spec) if role == "head" else _worker_both(spec)
        env = _both_env(spec, role)
        nodes.append(launch_for(spec, role, argv, env, container=name, stop=docker_stop(name)))
    return Built(
        nodes=nodes,
        order=roles_for("both"),
        experimental=True,
        experimental_reason=EXPERIMENTAL_BOTH,
        health_paths=["/health", "/v1/models"],
    )
