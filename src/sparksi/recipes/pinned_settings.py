# SPDX-License-Identifier: Apache-2.0

"""Main settings read out of a pinned launch command, for the recipe list and name suggestion.

Read only: nothing here rewrites argv. Editing still goes through args.py.
A pinned recipe keeps its settings inside the command (vLLM or llama.cpp flags,
TensorFold flags, or the environment a start.sh reads), so the list columns and
the suggested name would stay blank without this.
"""

from __future__ import annotations

import re
from typing import Any

from sparksi.recipes.pinned import _model_from_args, known_fields

_HF_CACHE = re.compile(r"models--([^/]+?)--([^/]+)")
_BARE_FOLDERS = {"", "models", "model", "weights", "snapshot", "snapshots"}


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _flag(args: list[str], *names: str) -> str | None:
    """The value after ``--flag value`` or in ``--flag=value``; the first name found wins."""
    for name in names:
        for index, arg in enumerate(args):
            if arg == name and index + 1 < len(args) and not args[index + 1].startswith("-"):
                return args[index + 1]
            if arg.startswith(name + "="):
                return arg.split("=", 1)[1]
    return None


def _hf_repo(path: str) -> str:
    match = _HF_CACHE.search(path or "")
    return f"{match.group(1)}/{match.group(2)}" if match else ""


def _last_part(path: str) -> str:
    tail = str(path or "").rstrip("/").rsplit("/", 1)[-1]
    return tail[:-5] if tail.lower().endswith(".gguf") else tail


def _head(recipe: dict[str, Any]) -> dict[str, Any]:
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    head = roles.get("head") if isinstance(roles.get("head"), dict) else next(iter(roles.values()), {})
    return head if isinstance(head, dict) else {}


def _known(recipe: dict[str, Any], head: dict[str, Any], args: list[str]) -> dict[str, Any]:
    """known_fields on a copy whose head args are a clean list; {} when the file is too odd to read."""
    safe = {key: value for key, value in recipe.items() if key not in ("roles", "rendezvous_ports")}
    safe["roles"] = {"head": {"args": args, "image": str(head.get("image") or "")}}
    try:
        known = known_fields(safe)
    except (TypeError, ValueError, AttributeError, KeyError):
        return {}
    return known if isinstance(known, dict) else {}


def pinned_settings(recipe: dict[str, Any]) -> dict[str, Any]:
    """Return ctx, max_num_seqs, kv_cache_dtype, speculative (method), tp, model and quant; unknown keys are left out."""
    if not isinstance(recipe, dict):
        return {}
    head = _head(recipe)
    raw_args = head.get("args")
    raw_env = head.get("env")
    # A hand-edited or damaged file must not break the recipe list: odd shapes read as empty.
    args = [str(part) for part in raw_args] if isinstance(raw_args, (list, tuple)) else []
    env = {str(k): str(v) for k, v in raw_env.items()} if isinstance(raw_env, dict) else {}
    known = _known(recipe, head, args)
    spec = known.get("speculative") if isinstance(known.get("speculative"), dict) else {}

    candidates: dict[str, list[Any]] = {
        "ctx": [recipe.get("ctx"), known.get("ctx"), _flag(args, "--context"), env.get("MAX_MODEL_LEN")],
        "max_num_seqs": [recipe.get("max_num_seqs"), known.get("max_num_seqs"), _flag(args, "--parallel"), env.get("MAX_NUM_SEQS")],
        "tp": [_flag(args, "--tensor-parallel-size", "-tp", "--tp"), env.get("TP"), env.get("TENSOR_PARALLEL_SIZE")],
    }
    out: dict[str, Any] = {}
    for key, values in candidates.items():
        for value in values:
            number = _as_int(value)
            if number is not None:
                out[key] = number
                break

    drafter = _flag(args, "--drafter")
    texts: dict[str, list[Any]] = {
        "kv_cache_dtype": [recipe.get("kv_cache_dtype"), known.get("kv_cache_dtype"), env.get("KV_CACHE_DTYPE"), env.get("TF_GLM_KV")],
        "speculative": [
            spec.get("method"),
            ("dflash" if "dflash" in drafter.lower() else "draft") if drafter else "",
            env.get("SPEC_METHOD"),
        ],
    }
    for key, values in texts.items():
        for value in values:
            text = str(value or "").strip()
            if text and text.lower() not in ("auto", "none", "off", "0"):
                out[key] = text
                break

    path = _model_from_args(args) or env.get("MODEL_DIR") or env.get("MODEL_PATH") or ""
    repo = (recipe.get("repo") if isinstance(recipe.get("repo"), str) else "").strip() or _hf_repo(path)
    names = [_flag(args, "--served-model-name", "--alias", "-a"), env.get("SERVED_MODEL_NAME")]
    tail = _last_part(path)
    model = repo or next((str(n).strip() for n in names if str(n or "").strip()), "")
    if not model and tail.lower() not in _BARE_FOLDERS:
        model = tail
    if model:
        out["model"] = model
    quant = [_flag(args, "--quantization", "-q") or env.get("QUANTIZATION") or "", repo.rsplit("/", 1)[-1] if repo else tail]
    quant_text = " ".join(part for part in quant if part and part.lower() not in _BARE_FOLDERS)
    if quant_text:
        out["quant"] = quant_text
    return out
