# SPDX-License-Identifier: Apache-2.0

"""Shared argv pieces. Strings are appended as elements and never joined for a shell."""

from __future__ import annotations

from sparksi.recipes.spec import Launch, Spec

CONTAINER_MOUNT = "/root/.cache/huggingface"


def container_name(spec: Spec) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in spec.model_id)
    return f"sparksi-{safe}-{int(spec.port)}"


def roles_for(placement: str) -> list[str]:
    if placement == "both":
        return ["worker", "head"]
    if placement == "worker":
        return ["worker"]
    return ["head"]


def fabric_env(spec: Spec, role: str) -> dict[str, str]:
    iface = spec.fabric_iface.get(role) or ""
    hcas = ",".join(spec.fabric_hcas.get(role) or [])
    return {
        "NCCL_SOCKET_IFNAME": iface,
        "GLOO_SOCKET_IFNAME": iface,
        "TP_SOCKET_IFNAME": iface,
        "UCX_NET_DEVICES": iface,
        "NCCL_IB_HCA": hcas,
        "NCCL_IB_DISABLE": "0",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
    }


def docker_argv(
    spec: Spec,
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    extra: list[str] | None = None,
) -> list[str]:
    """``docker run -d`` with the shared Spark flags. ``command`` is the container argv."""
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
    for key, value in (env or {}).items():
        argv.extend(["-e", f"{key}={value}"])
    argv.extend(extra or [])
    argv.extend(["-v", f"{spec.hf_cache}:{CONTAINER_MOUNT}", spec.image])
    argv.extend(command)
    return argv


def launch_for(spec: Spec, role: str, argv: list[str], env: dict[str, str], *, container: str, stop: list[str]) -> Launch:
    rank = 0 if role == "head" else 1
    return Launch(
        role=role,
        rank=rank,
        host=spec.hosts.get(role) or role,
        argv=list(argv),
        env=dict(env),
        container=container,
        stop_argv=list(stop),
        bind=spec.binds.get(role) or "",
    )


def docker_stop(name: str) -> list[str]:
    return ["docker", "rm", "-f", name]


def path_for(spec: Spec, role: str) -> str:
    return spec.paths.get(role) or spec.paths.get("head") or ""
