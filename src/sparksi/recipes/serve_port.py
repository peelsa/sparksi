# SPDX-License-Identifier: Apache-2.0

"""One serve port for a recipe.

A start script often ignores an appended ``--port`` flag and reads ``PORT``,
``VLLM_PORT``, or ``SERVER_PORT`` instead. Those flags are not written after
the script. The number Sparksi health-checks is the number in that environment.
Fabric ports (master and NCCL) are left alone.
"""

from __future__ import annotations

import os
from typing import Any

SERVE_PORT_ENV = frozenset({
    "PORT",
    "SERVER_PORT",
    "VLLM_PORT",
    "SGLANG_PORT",
    "TRTLLM_PORT",
    "LLAMA_PORT",
    "LISTEN_PORT",
    "API_PORT",
    "HTTP_PORT",
    "WEB_PORT",
    "APP_PORT",
    "UVICORN_PORT",
})


def _skipped(name: str) -> bool:
    upper = str(name or "").upper()
    if not upper:
        return True
    if upper in {"MASTER_PORT", "MASTER_ADDR", "DIST_PORT"}:
        return True
    if upper.startswith("MASTER_"):
        return True
    if "NCCL" in upper or "RDMA" in upper or "GLOO" in upper or "UCX" in upper:
        return True
    return False


def is_serve_env(name: str, value: str, saved_port: int | None = None) -> bool:
    """True when this variable is the port the server process would bind."""
    if _skipped(name):
        return False
    upper = str(name or "").upper()
    if upper in SERVE_PORT_ENV:
        return True
    if saved_port is None:
        return False
    if str(value) != str(saved_port):
        return False
    return "PORT" in upper


def _as_port(value: Any) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def rewrite_port_args(args: list[str], port: int) -> list[str]:
    """Replace ``--port`` / ``-p`` values. Other tokens stay put."""
    out: list[str] = []
    index = 0
    while index < len(args):
        part = args[index]
        nxt = args[index + 1] if index + 1 < len(args) else None
        if part in ("--port", "-p") and nxt is not None and _as_port(nxt):
            out.extend([part, str(port)])
            index += 2
            continue
        if str(part).startswith("--port=") and _as_port(str(part).split("=", 1)[1]):
            out.append(f"--port={port}")
            index += 1
            continue
        out.append(part)
        index += 1
    return out


_SHELLS = frozenset({"bash", "sh", "dash", "ash"})
_PORT_FROM_ENV = frozenset({"PORT", "VLLM_PORT", "SERVER_PORT"})
_CTX_FLAGS = frozenset({"--max-model-len", "--context-length", "--max_seq_len", "--ctx-size"})
_PORT_FLAGS = frozenset({"--port", "-p"})
_HOST_FLAGS = frozenset({"--host", "--bind"})
# Flags Sparksi appends. A script that already carries some other argument
# is one that accepts trailing args, so a context flag can stay.
_APPENDED = _CTX_FLAGS | _PORT_FLAGS | _HOST_FLAGS | frozenset({
    "--gpu-memory-utilization",
    "--mem-fraction-static",
    "--kv_cache_free_gpu_memory_fraction",
    "--kv-cache-memory-bytes",
    "--kv-cache-memory",
    "--kv-cache-dtype",
    "--max-num-seqs",
    "--tool-call-parser",
    "--reasoning-parser",
    "--speculative-config",
    "--spec-type",
    "--spec-draft-n-max",
    "--model-draft",
    "--speculative-algorithm",
    "--speculative-num-steps",
    "--speculative-draft-model-path",
})


def _script_token(token: str) -> bool:
    base = os.path.basename(str(token or ""))
    return base.endswith(".sh") or base.endswith(".bash")


def _script_index(role: dict[str, Any]) -> int | None:
    """Index of the script path when this role is ``bash /start.sh``."""
    args = [str(part) for part in (role.get("args") or [])]
    entry = os.path.basename(str(role.get("entrypoint") or "").strip())
    if entry in _SHELLS and args and _script_token(args[0]):
        return 0
    if args and os.path.basename(args[0]) in _SHELLS and len(args) > 1 and _script_token(args[1]):
        return 1
    if entry in _SHELLS and len(args) > 1 and os.path.basename(args[0]) in _SHELLS and _script_token(args[1]):
        return 1
    return None


def _forwards_args(tail: list[str]) -> bool:
    """True when the script is already given an argument Sparksi did not invent."""
    index = 0
    while index < len(tail):
        token = tail[index]
        flag = token.split("=", 1)[0]
        if flag not in _APPENDED:
            return True
        if "=" in token:
            index += 1
            continue
        if index + 1 < len(tail) and not str(tail[index + 1]).startswith("-"):
            index += 2
            continue
        return True
    return False


def _rewrite_script_tail(
    tail: list[str],
    *,
    drop_port: bool,
    drop_host: bool,
    drop_ctx: bool,
) -> tuple[list[str], str]:
    """Drop ignored flags. The returned string is a context length that was present."""
    kept: list[str] = []
    ctx = ""
    index = 0
    while index < len(tail):
        token = tail[index]
        flag, sep, inline = token.partition("=")
        paired = not sep and index + 1 < len(tail) and not str(tail[index + 1]).startswith("-")
        value = inline if sep else (tail[index + 1] if paired else "")
        span = 2 if paired else 1
        if flag in _CTX_FLAGS and value:
            ctx = value
        host_list = flag in _HOST_FLAGS and "," in str(value)
        drop = (
            (drop_port and flag in _PORT_FLAGS)
            or (drop_host and flag in _HOST_FLAGS and not host_list)
            or (drop_ctx and flag in _CTX_FLAGS)
        )
        if not drop:
            kept.extend(tail[index:index + span])
        index += span
    return kept, ctx


def quiet_script_args(role: dict[str, Any]) -> None:
    """Do not leave flags a ``bash /start.sh`` entry ignores.

    Port and host flags are dropped when the script reads ``PORT``,
    ``VLLM_PORT``, or ``SERVER_PORT``. ``--max-model-len`` is kept only when
    the script already accepts trailing args. A documented ``MAX_MODEL_LEN``
    is updated from that flag, and is not invented when the recipe has none.
    """
    if not isinstance(role, dict):
        return
    index = _script_index(role)
    if index is None:
        return
    args = [str(part) for part in (role.get("args") or [])]
    head, tail = args[: index + 1], args[index + 1 :]
    env = {str(key): str(value) for key, value in (role.get("env") or {}).items()}
    uses_port = any(str(key).upper() in _PORT_FROM_ENV for key in env)
    kept, ctx = _rewrite_script_tail(
        tail,
        drop_port=uses_port,
        drop_host=uses_port,
        drop_ctx=not _forwards_args(tail),
    )
    if ctx:
        for key in list(env):
            if str(key).upper() == "MAX_MODEL_LEN":
                env[key] = ctx
                role["env"] = env
                break
    role["args"] = head + kept


def align_role(role: dict[str, Any], port: int, *, saved_port: int | None = None) -> None:
    """Write ``port`` into this role's serve-port argument and environment."""
    if not isinstance(role, dict) or not port:
        return
    role["args"] = rewrite_port_args([str(part) for part in (role.get("args") or [])], port)
    env = {str(key): str(value) for key, value in (role.get("env") or {}).items()}
    for key, value in list(env.items()):
        if is_serve_env(key, value, saved_port):
            env[key] = str(port)
    role["env"] = env
    quiet_script_args(role)


def align_recipe(recipe: dict[str, Any], port: int | None = None, *, saved_port: int | None = None) -> None:
    """Make every role agree with the recipe serve port."""
    if not isinstance(recipe, dict):
        return
    number = _as_port(port if port not in (None, "") else recipe.get("port"))
    if not number:
        return
    recipe["port"] = number
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    for role in roles.values():
        align_role(role, number, saved_port=saved_port)


def serve_port_conflicts(recipe: dict[str, Any]) -> list[str]:
    """Plain-words refusals when a launch would bind somewhere else."""
    number = _as_port(recipe.get("port") if isinstance(recipe, dict) else 0)
    if not number or not isinstance(recipe, dict):
        return []
    found: list[str] = []
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    saved = _as_port(recipe.get("saved_port")) or None
    for role in roles.values():
        if not isinstance(role, dict):
            continue
        for key, value in (role.get("env") or {}).items():
            if not is_serve_env(str(key), str(value), saved):
                continue
            if _as_port(value) and _as_port(value) != number:
                found.append(f"the recipe also sets {key}={value}; the model would listen there")
        args = [str(part) for part in (role.get("args") or [])]
        index = 0
        while index < len(args):
            part = args[index]
            if part in ("--port", "-p") and index + 1 < len(args):
                other = _as_port(args[index + 1])
                if other and other != number:
                    found.append(f"the recipe also sets --port {other}; the model would listen there")
                index += 2
                continue
            if part.startswith("--port="):
                other = _as_port(part.split("=", 1)[1])
                if other and other != number:
                    found.append(f"the recipe also sets --port {other}; the model would listen there")
            index += 1
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)
    return unique


def other_serve_ports(run: dict[str, Any]) -> list[int]:
    """Serve ports named on the launch that are not the port Sparksi is waiting on."""
    expected = _as_port(run.get("port") if isinstance(run, dict) else 0)
    found: list[int] = []
    nodes = []
    if isinstance(run, dict):
        nodes.extend(run.get("nodes") or [])
        roles = run.get("roles") if isinstance(run.get("roles"), dict) else {}
        nodes.extend(roles.values())
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for key, value in (node.get("env") or {}).items():
            if not is_serve_env(str(key), str(value), expected or None):
                continue
            number = _as_port(value)
            if number and number != expected and number not in found:
                found.append(number)
        args = [str(part) for part in (node.get("argv") or node.get("args") or [])]
        index = 0
        while index < len(args):
            part = args[index]
            if part in ("--port", "-p") and index + 1 < len(args):
                number = _as_port(args[index + 1])
                if number and number != expected and number not in found:
                    found.append(number)
                index += 2
                continue
            index += 1
    return found
