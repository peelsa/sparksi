# SPDX-License-Identifier: Apache-2.0

"""Field allowlists. Values become argv elements and are never shell-joined."""

from __future__ import annotations

import os
from typing import Any

from sparksi import protect
from sparksi.never import launcher_refusal

PORT_MIN = 8010
PORT_MAX = 8099
CTX_MIN = 256
CTX_ABS_MAX = 2097152
ENGINES = ("vllm", "sglang", "trtllm", "llamacpp", "custom")
PINNED_ENGINES = ("vllm", "sglang", "trtllm", "llamacpp", "tensorfold", "other")

KV_DTYPES = {"fp8", "auto", "bf16", "fp8_e4m3", "fp8_e5m2", "q8_0"}
TOOL_PARSERS = {"hermes", "llama3_json", "pythonic", "mistral", "qwen3_coder", "glm47"}
REASONING_PARSERS = {"deepseek_r1", "qwen3", "granite", "glm45"}
SPEC_METHODS = {"mtp", "dflash", "dflash2", "nextn", "draft-mtp", "eagle", "draft"}
ATTENTION_BACKENDS = {"FLASH_ATTN", "FLASHINFER", "TORCH_SDPA", "FLEX_ATTENTION", "TRITON_ATTN"}
SSM_DTYPES = {"float32", "bfloat16", "float16", "auto"}
QUANTIZATIONS = {"awq", "gptq", "fp8", "bitsandbytes", "compressed-tensors", "auto", "experts_int8"}
LOAD_FORMATS = {"auto", "pt", "safetensors", "dummy", "bitsandbytes", "runai_streamer", "tensorizer"}
REASONING_MODES = {"off", "on", "auto"}

# Flags that may be appended as their own argv element. No values, no shell.
EXTRA_FLAGS = {
    "vllm": {
        "--trust-remote-code",
        "--enable-expert-parallel",
        "--disable-log-requests",
        "--async-scheduling",
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--no-enable-prefix-caching",
    },
    "sglang": {"--trust-remote-code", "--enable-cache-report", "--enable-mixed-chunk"},
    "trtllm": {"--trust_remote_code"},
    "llamacpp": {"--no-mmap", "--metrics", "--jinja", "--flash-attn"},
    "custom": set(),
}

# Fields a recipe may set, per engine. Anything else is rejected.
FIELDS = {
    "vllm": {
        "ctx", "gpu_mem_util", "max_num_seqs", "kv_cache_dtype", "prefix_cache",
        "kv_cache_memory_bytes", "speculative", "tool_parser", "reasoning_parser",
        "image", "timeout_s", "extra_flags",
        "max_num_batched_tokens", "attention_backend", "mamba_ssm_dtype",
        "quantization", "load_format", "limit_mm_per_prompt",
        "placement", "port", "engine",
    },
    "sglang": {
        "ctx", "gpu_mem_util", "speculative", "tool_parser", "reasoning_parser",
        "image", "timeout_s", "extra_flags", "placement", "port", "engine",
    },
    "trtllm": {"ctx", "gpu_mem_util", "image", "timeout_s", "extra_flags", "placement", "port", "engine"},
    "llamacpp": {
        "ctx", "speculative", "timeout_s", "extra_flags", "placement", "port", "engine",
        "binary", "ld_library_path", "temperature", "top_p", "top_k", "presence_penalty",
        "alias", "reasoning", "kv_cache_dtype", "max_num_seqs",
    },
    "custom": {"timeout_s", "placement", "port", "engine"},
}

SHELL_META = set(";|&$`<>\n\r\\")


def _meta(value: str) -> bool:
    return any(ch in SHELL_META for ch in value)


def _listen_word(host: str) -> str:
    """The role word (head, worker, node3, node4) when the host is one of this site's Sparks."""
    from sparksi.config import CFG

    for role, node in CFG.role_nodes():
        if str(host) in (node.name, node.id, role):
            return role
    return str(host)


def check_port(
    port: Any,
    listening: dict[str, set[int]] | None = None,
    hosts: list[str] | None = None,
    *,
    reserved: set[int] | None = None,
    claimed: set[int] | None = None,
    holders: dict[int, str] | None = None,
) -> list[str]:
    conflicts: list[str] = []
    try:
        number = int(port)
    except (TypeError, ValueError):
        return ["port must be an integer"]
    if protect.is_off_limits(number):
        conflicts.append("port 8188 is off limits")
        return conflicts
    if number < PORT_MIN or number > PORT_MAX:
        conflicts.append(f"port {number} is outside {PORT_MIN}-{PORT_MAX}")
    if number in (reserved or ()):
        conflicts.append(f"port {number} is reserved")
    if listening:
        keys = list(hosts) if hosts is not None else list(listening)
        for host in keys:
            if number in (listening.get(host) or set()):
                conflicts.append(f"port {number} is already listening on {_listen_word(host)}")
                break
    if holders and number in holders:
        conflicts.append(str(holders[number]))
    elif number in (claimed or ()):
        conflicts.append(f"port {number} is used by another recipe")
    return conflicts


def next_free_port(
    listening: set[int],
    reserved: set[int] | None = None,
    claimed: set[int] | None = None,
) -> int | None:
    skip = set(listening)
    skip |= set(reserved or ())
    skip |= set(claimed or ())
    skip |= protect.off_limits_ports()
    for port in range(PORT_MIN, PORT_MAX + 1):
        if port not in skip:
            return port
    return None


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def ctx_ceiling(model_max: int | None) -> int:
    """Model maximum when the card knows one, otherwise 2097152."""
    if model_max in (None, ""):
        return CTX_ABS_MAX
    try:
        number = int(model_max)
    except (TypeError, ValueError):
        return CTX_ABS_MAX
    if number < CTX_MIN:
        return CTX_ABS_MAX
    return min(number, CTX_ABS_MAX)


def form_for(engine: str, *, pinned: bool = False, ctx_max: int | None = None) -> dict[str, Any]:
    """Which recipe fields the page may edit for this engine."""
    if pinned:
        return {
            "pinned": True,
            "ctx": True,
            "ctx_max": ctx_ceiling(ctx_max),
            "max_num_seqs": True,
            "gpu_mem_util": True,
            "kv_cache_memory_bytes": True,
            "kv_cache_dtype": sorted(KV_DTYPES),
            "speculative": True,
            "speculative_methods": sorted(SPEC_METHODS),
            "tool_parser": sorted(TOOL_PARSERS),
            "reasoning_parser": sorted(REASONING_PARSERS),
            "image": True,
            "image_free": True,
            "extra_args": True,
            "env": True,
            "extra_flags": [],
        }
    if engine not in FIELDS:
        return {}
    allowed = FIELDS[engine]
    return {
        "ctx": "ctx" in allowed,
        "ctx_max": ctx_ceiling(ctx_max),
        "max_num_seqs": "max_num_seqs" in allowed,
        "gpu_mem_util": "gpu_mem_util" in allowed,
        "kv_cache_memory_bytes": "kv_cache_memory_bytes" in allowed,
        "kv_cache_dtype": sorted(KV_DTYPES) if "kv_cache_dtype" in allowed else [],
        "speculative": "speculative" in allowed,
        "speculative_methods": sorted(SPEC_METHODS) if "speculative" in allowed else [],
        "tool_parser": sorted(TOOL_PARSERS) if "tool_parser" in allowed else [],
        "reasoning_parser": sorted(REASONING_PARSERS) if "reasoning_parser" in allowed else [],
        "image": "image" in allowed,
        "extra_flags": sorted(EXTRA_FLAGS.get(engine) or ()),
        "max_num_batched_tokens": "max_num_batched_tokens" in allowed,
        "attention_backend": sorted(ATTENTION_BACKENDS) if "attention_backend" in allowed else [],
        "mamba_ssm_dtype": sorted(SSM_DTYPES) if "mamba_ssm_dtype" in allowed else [],
        "quantization": sorted(QUANTIZATIONS) if "quantization" in allowed else [],
        "load_format": sorted(LOAD_FORMATS) if "load_format" in allowed else [],
        "limit_mm_per_prompt": "limit_mm_per_prompt" in allowed,
        "binary": "binary" in allowed,
        "ld_library_path": "ld_library_path" in allowed,
        "temperature": "temperature" in allowed,
        "top_p": "top_p" in allowed,
        "top_k": "top_k" in allowed,
        "presence_penalty": "presence_penalty" in allowed,
        "alias": "alias" in allowed,
        "reasoning": sorted(REASONING_MODES) if "reasoning" in allowed else [],
    }


def _under_root(path: str, roots: list[str]) -> bool:
    text = os.path.normpath(str(path or ""))
    if not text.startswith("/"):
        return False
    for root in roots:
        base = os.path.normpath(str(root or ""))
        if base in ("", ".", "/"):
            continue
        if text == base or text.startswith(base + os.sep):
            return True
    return False


def binary_allowed(path: str, roots: list[str]) -> bool:
    """``llama-server`` on PATH, or an absolute path under a configured root."""
    text = str(path or "")
    if text == "llama-server":
        return True
    if _meta(text) or not text.startswith("/"):
        return False
    return _under_root(text, roots)


def library_path_allowed(path: str, roots: list[str]) -> bool:
    """One directory, or several joined by ``:``, each under a configured root."""
    text = str(path or "")
    if not text or _meta(text):
        return False
    parts = [part for part in text.split(":") if part]
    return bool(parts) and all(_under_root(part, roots) for part in parts)


def normalize_speculative(value: Any) -> tuple[dict[str, Any] | None, str]:
    """Return a method/model/tokens dict, or an error. A bool maps to MTP or off."""
    if value in (None, ""):
        return None, ""
    if isinstance(value, bool):
        if value:
            return {"method": "mtp", "model": "", "tokens": 1}, ""
        return {"method": "", "model": "", "tokens": 0}, ""
    if not isinstance(value, dict):
        return None, "speculative must be method, draft model, and token count"
    method = str(value.get("method") or "")
    model = str(value.get("model") or "")
    try:
        tokens = int(value.get("tokens") or 0)
    except (TypeError, ValueError):
        return None, "speculative tokens must be an integer"
    if method and method not in SPEC_METHODS:
        return None, "speculative method is not allowed"
    if model and (_meta(model) or not model.startswith("/")):
        return None, "speculative model must be an absolute path"
    if tokens < 0 or tokens > 16:
        return None, "speculative tokens must be from 0 to 16"
    if method and tokens < 1:
        tokens = 1
    return {"method": method, "model": model, "tokens": tokens}, ""


def validate_fields(
    engine: str,
    fields: dict[str, Any] | None,
    *,
    allowed_images: list[str],
    forbid_prefix: bool = False,
    max_ctx: int | None = None,
    ctx_max: int | None = None,
    mount_roots: list[str] | None = None,
    pinned: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return cleaned fields and every conflict. Unknown keys and flags are rejected."""
    raw = dict(fields or {})
    conflicts: list[str] = []
    if engine not in FIELDS:
        return {}, [f"unknown engine {engine}"]
    allowed = FIELDS[engine]
    unknown = sorted(key for key in raw if key not in allowed)
    for key in unknown:
        conflicts.append(f"unknown field {key}")
    cleaned: dict[str, Any] = {}

    if "ctx" in raw and "ctx" in allowed:
        try:
            ctx = int(raw["ctx"])
        except (TypeError, ValueError):
            conflicts.append("ctx must be an integer")
        else:
            cap = max_ctx if max_ctx is not None else ctx_max
            if cap is not None:
                limit = min(int(cap), CTX_ABS_MAX)
                if ctx < CTX_MIN or ctx > limit:
                    if ctx > limit:
                        conflicts.append(f"ctx is above the model maximum {limit}")
                    else:
                        conflicts.append(f"ctx must be from {CTX_MIN} to {limit}")
                else:
                    cleaned["ctx"] = ctx
            elif ctx < CTX_MIN or ctx > CTX_ABS_MAX:
                conflicts.append(f"ctx must be from {CTX_MIN} to {CTX_ABS_MAX}")
            else:
                cleaned["ctx"] = ctx
    if "gpu_mem_util" in raw and "gpu_mem_util" in allowed:
        try:
            util = float(raw["gpu_mem_util"])
        except (TypeError, ValueError):
            conflicts.append("gpu_mem_util must be a number")
        else:
            if util < 0.3 or util > 0.95:
                conflicts.append("gpu_mem_util must be from 0.3 to 0.95")
            else:
                cleaned["gpu_mem_util"] = util
    if "max_num_seqs" in raw and "max_num_seqs" in allowed:
        try:
            seqs = int(raw["max_num_seqs"])
        except (TypeError, ValueError):
            conflicts.append("max_num_seqs must be an integer")
        else:
            if seqs < 1 or seqs > 16:
                conflicts.append("max_num_seqs must be from 1 to 16")
            else:
                cleaned["max_num_seqs"] = seqs
    if "kv_cache_dtype" in raw and "kv_cache_dtype" in allowed:
        dtype = str(raw["kv_cache_dtype"])
        if dtype not in KV_DTYPES:
            conflicts.append(f"kv_cache_dtype must be one of {', '.join(sorted(KV_DTYPES))}")
        else:
            cleaned["kv_cache_dtype"] = dtype
    if "prefix_cache" in raw and "prefix_cache" in allowed:
        value = raw["prefix_cache"]
        if not isinstance(value, bool):
            conflicts.append("prefix_cache must be a boolean")
        elif value and forbid_prefix:
            conflicts.append("prefix caching is forbidden for this model")
        else:
            cleaned["prefix_cache"] = value
    if "speculative" in raw and raw.get("speculative") not in (None, "") and "speculative" in allowed:
        spec, why = normalize_speculative(raw["speculative"])
        if why:
            conflicts.append(why)
        elif spec is not None:
            cleaned["speculative"] = spec
    if "tool_parser" in raw and raw.get("tool_parser") not in (None, "") and "tool_parser" in allowed:
        parser = str(raw["tool_parser"])
        if parser not in TOOL_PARSERS:
            conflicts.append("tool_parser is not an allowed parser")
        else:
            cleaned["tool_parser"] = parser
    if "reasoning_parser" in raw and raw.get("reasoning_parser") not in (None, "") and "reasoning_parser" in allowed:
        parser = str(raw["reasoning_parser"])
        if parser not in REASONING_PARSERS:
            conflicts.append("reasoning_parser is not an allowed parser")
        else:
            cleaned["reasoning_parser"] = parser
    if "kv_cache_memory_bytes" in raw and raw.get("kv_cache_memory_bytes") not in (None, "") and "kv_cache_memory_bytes" in allowed:
        try:
            nbytes = int(raw["kv_cache_memory_bytes"])
        except (TypeError, ValueError):
            conflicts.append("kv_cache_memory_bytes must be an integer")
        else:
            if nbytes < 1 or nbytes > 2**40:
                conflicts.append("kv_cache_memory_bytes is out of range")
            else:
                cleaned["kv_cache_memory_bytes"] = nbytes
    if "image" in raw and raw.get("image") not in (None, "") and "image" in allowed:
        image = str(raw["image"])
        if ":latest" in image or image.endswith(":latest"):
            conflicts.append("image tag must be pinned")
        elif _meta(image):
            conflicts.append("image contains shell metacharacters")
        elif not pinned and image not in allowed_images:
            conflicts.append("image is not in the allowed list")
        else:
            cleaned["image"] = image
    if "timeout_s" in raw and raw.get("timeout_s") not in (None, ""):
        try:
            timeout = int(raw["timeout_s"])
        except (TypeError, ValueError):
            conflicts.append("timeout_s must be an integer")
        else:
            if timeout < 1 or timeout > 86400:
                conflicts.append("timeout_s must be from 1 to 86400")
            else:
                cleaned["timeout_s"] = timeout
    if "extra_flags" in raw and "extra_flags" in allowed:
        flags = raw["extra_flags"]
        if flags in (None, ""):
            flags = []
        if not isinstance(flags, list) or any(not isinstance(item, str) for item in flags):
            conflicts.append("extra_flags must be a list of strings")
        else:
            allow = EXTRA_FLAGS.get(engine) or set()
            bad = [item for item in flags if item not in allow]
            if bad:
                conflicts.append("unknown flag " + bad[0])
            elif any(_meta(item) for item in flags):
                conflicts.append("unknown flag")
            else:
                cleaned["extra_flags"] = list(flags)
    if "max_num_batched_tokens" in raw and raw.get("max_num_batched_tokens") not in (None, "") and "max_num_batched_tokens" in allowed:
        try:
            batched = int(raw["max_num_batched_tokens"])
        except (TypeError, ValueError):
            conflicts.append("max_num_batched_tokens must be an integer")
        else:
            if batched < 1 or batched > CTX_ABS_MAX:
                conflicts.append("max_num_batched_tokens is out of range")
            else:
                cleaned["max_num_batched_tokens"] = batched
    if "attention_backend" in raw and raw.get("attention_backend") not in (None, "") and "attention_backend" in allowed:
        backend = str(raw["attention_backend"])
        if backend not in ATTENTION_BACKENDS:
            conflicts.append("attention_backend is not allowed")
        else:
            cleaned["attention_backend"] = backend
    if "mamba_ssm_dtype" in raw and raw.get("mamba_ssm_dtype") not in (None, "") and "mamba_ssm_dtype" in allowed:
        dtype = str(raw["mamba_ssm_dtype"])
        if dtype not in SSM_DTYPES:
            conflicts.append("mamba_ssm_dtype is not allowed")
        else:
            cleaned["mamba_ssm_dtype"] = dtype
    if "quantization" in raw and raw.get("quantization") not in (None, "") and "quantization" in allowed:
        quant = str(raw["quantization"])
        if quant not in QUANTIZATIONS:
            conflicts.append("quantization is not allowed")
        else:
            cleaned["quantization"] = quant
    if "load_format" in raw and raw.get("load_format") not in (None, "") and "load_format" in allowed:
        fmt = str(raw["load_format"])
        if fmt not in LOAD_FORMATS:
            conflicts.append("load_format is not allowed")
        else:
            cleaned["load_format"] = fmt
    if "limit_mm_per_prompt" in raw and raw.get("limit_mm_per_prompt") not in (None, "") and "limit_mm_per_prompt" in allowed:
        limit = str(raw["limit_mm_per_prompt"])
        ok = bool(limit) and all(
            part and "=" in part and part.split("=", 1)[0].isalpha() and part.split("=", 1)[1].isdigit()
            for part in limit.split(",")
        )
        if _meta(limit) or not ok:
            conflicts.append("limit_mm_per_prompt must look like image=2")
        else:
            cleaned["limit_mm_per_prompt"] = limit
    roots = list(mount_roots or [])
    if "binary" in raw and raw.get("binary") not in (None, "") and "binary" in allowed:
        binary = str(raw["binary"])
        if not binary_allowed(binary, roots):
            conflicts.append("binary must be llama-server or an absolute path under a mount root")
        else:
            cleaned["binary"] = binary
    if "ld_library_path" in raw and raw.get("ld_library_path") not in (None, "") and "ld_library_path" in allowed:
        lib = str(raw["ld_library_path"])
        if not library_path_allowed(lib, roots):
            conflicts.append("LD_LIBRARY_PATH must be under a mount root")
        else:
            cleaned["ld_library_path"] = lib
    if "temperature" in raw and raw.get("temperature") not in (None, "") and "temperature" in allowed:
        try:
            temp = float(raw["temperature"])
        except (TypeError, ValueError):
            conflicts.append("temperature must be a number")
        else:
            if temp < 0 or temp > 2:
                conflicts.append("temperature must be from 0 to 2")
            else:
                cleaned["temperature"] = temp
    if "top_p" in raw and raw.get("top_p") not in (None, "") and "top_p" in allowed:
        try:
            top_p = float(raw["top_p"])
        except (TypeError, ValueError):
            conflicts.append("top_p must be a number")
        else:
            if top_p < 0 or top_p > 1:
                conflicts.append("top_p must be from 0 to 1")
            else:
                cleaned["top_p"] = top_p
    if "top_k" in raw and raw.get("top_k") not in (None, "") and "top_k" in allowed:
        try:
            top_k = int(raw["top_k"])
        except (TypeError, ValueError):
            conflicts.append("top_k must be an integer")
        else:
            if top_k < 0 or top_k > 1000:
                conflicts.append("top_k must be from 0 to 1000")
            else:
                cleaned["top_k"] = top_k
    if "presence_penalty" in raw and raw.get("presence_penalty") not in (None, "") and "presence_penalty" in allowed:
        try:
            penalty = float(raw["presence_penalty"])
        except (TypeError, ValueError):
            conflicts.append("presence_penalty must be a number")
        else:
            if penalty < -2 or penalty > 2:
                conflicts.append("presence_penalty must be from -2 to 2")
            else:
                cleaned["presence_penalty"] = penalty
    if "alias" in raw and raw.get("alias") not in (None, "") and "alias" in allowed:
        alias = str(raw["alias"])
        if _meta(alias) or any(ch.isspace() for ch in alias):
            conflicts.append("alias must be one token")
        else:
            cleaned["alias"] = alias
    if "reasoning" in raw and raw.get("reasoning") not in (None, "") and "reasoning" in allowed:
        mode = str(raw["reasoning"])
        if mode not in REASONING_MODES:
            conflicts.append("reasoning must be off, on, or auto")
        else:
            cleaned["reasoning"] = mode
    for key in ("placement", "port", "engine"):
        if key in raw:
            cleaned[key] = raw[key]
    return cleaned, conflicts


def memory_conflict(weights_gib: float, reserve_gib: float, placement: str, budgets: dict[str, float]) -> list[str]:
    conflicts = []
    text = str(placement or "")
    multi = ["head", "worker"] if text == "both" else [part for part in text.split("+") if part] if "+" in text else []
    if text in ("head", "worker", "node3", "node4"):
        need = float(weights_gib) + float(reserve_gib)
        budget = float(budgets.get(placement, 0.0))
        if need > budget:
            conflicts.append(
                f"over budget: {placement} needs {need:.1f} GiB and has {budget:.1f} GiB"
            )
    elif multi:
        per = float(weights_gib) / float(len(multi)) + float(reserve_gib)
        for role in multi:
            budget = float(budgets.get(role, 0.0))
            if per > budget:
                conflicts.append(f"over budget: {role} needs {per:.1f} GiB and has {budget:.1f} GiB")
                break
    return conflicts


def argv_backstop(argv_lists: list[list[str]]) -> list[str]:
    """``launcher_refusal`` plus the ports and binds a recipe must not carry."""
    conflicts = []
    for argv in argv_lists:
        blocked = launcher_refusal(argv)
        if blocked:
            conflicts.append(blocked)
            continue
        for index, part in enumerate(argv):
            if part in ("--port", "-p") and index + 1 < len(argv) and protect.is_off_limits(argv[index + 1]):
                conflicts.append("port 8188 is off limits")
            if part in ("--host",) and index + 1 < len(argv) and argv[index + 1] in ("0.0.0.0", "::", "*"):
                conflicts.append("refusing bind 0.0.0.0")
            if "dgx-announce" in part:
                conflicts.append("refusing dgx-announce")
            name = part.rsplit("/", 1)[-1]
            if name in ("video-mode.sh", "dgx-announce", "dgx-announce.service"):
                conflicts.append("refusing forbidden command")
    return conflicts
