# SPDX-License-Identifier: Apache-2.0

"""llama-server on one Spark. Two-node is refused."""

from __future__ import annotations

from sparksi.recipes.engines.common import launch_for, path_for
from sparksi.recipes.spec import Built, Spec
from sparksi.recipes.validate import normalize_speculative


def build(spec: Spec) -> Built:
    if spec.placement == "both":
        raise ValueError("llamacpp runs on one Spark")
    role = "worker" if spec.placement == "worker" else "head"
    binary = spec.binary or "llama-server"
    env: dict[str, str] = {}
    if spec.ld_library_path:
        env["LD_LIBRARY_PATH"] = spec.ld_library_path
    kv = spec.kv_cache_dtype if spec.kv_cache_dtype in ("q8_0", "f16", "bf16") else "q8_0"
    argv = [
        binary,
        "-m", path_for(spec, role),
        "--host", spec.binds.get(role) or "",
        "--port", str(int(spec.port)),
        "-ngl", "99",
        "-fa", "on",
        "-c", str(int(spec.ctx)),
        "-np", "1",
        "-ctk", kv,
        "-ctv", kv,
        "--jinja",
    ]
    mode, _why = normalize_speculative(spec.speculative)
    if spec.speculative is None and spec.hints.get("mtp"):
        mode = {"method": "mtp", "model": "", "tokens": 0}
    if mode and mode.get("method"):
        method = "draft-mtp" if mode["method"] == "mtp" else str(mode["method"])
        argv.extend(["--spec-type", method])
        if mode.get("tokens"):
            argv.extend(["--spec-draft-n-max", str(int(mode["tokens"]))])
        if mode.get("model"):
            argv.extend(["--model-draft", str(mode["model"])])
    if spec.temperature is not None:
        argv.extend(["--temp", str(spec.temperature)])
    if spec.top_p is not None:
        argv.extend(["--top-p", str(spec.top_p)])
    if spec.top_k is not None:
        argv.extend(["--top-k", str(int(spec.top_k))])
    if spec.presence_penalty is not None:
        argv.extend(["--presence-penalty", str(spec.presence_penalty)])
    if spec.alias:
        argv.extend(["--alias", spec.alias])
    if spec.reasoning:
        argv.extend(["--reasoning", spec.reasoning])
    argv.extend(spec.extra_flags)
    node = launch_for(spec, role, argv, env, container="", stop=[])
    return Built(nodes=[node], order=[role], experimental=False, health_paths=["/health", "/v1/models"])
