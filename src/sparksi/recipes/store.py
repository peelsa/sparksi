# SPDX-License-Identifier: Apache-2.0

"""Per-model recipe files. Only ``~/.config/sparksi/recipes/<id>.toml`` is written."""

from __future__ import annotations

import json
import math
import os
import re
import tomllib
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from sparksi.log import logger

KEYS = (
    "kind",
    "engine",
    "placement",
    "port",
    "repo",
    "revision",
    "ctx",
    "gpu_mem_util",
    "max_num_seqs",
    "kv_cache_dtype",
    "prefix_cache",
    "kv_cache_memory_bytes",
    "speculative",
    "tool_parser",
    "reasoning_parser",
    "image",
    "timeout_s",
    "extra_flags",
    "imported_from",
    "copied_from",
    "port_note",
    "token_note",
    "evict_page_cache",
    "rendezvous_ports",
    "roles",
    "binary",
    "ld_library_path",
    "max_num_batched_tokens",
    "attention_backend",
    "mamba_ssm_dtype",
    "quantization",
    "load_format",
    "limit_mm_per_prompt",
    "temperature",
    "top_p",
    "top_k",
    "presence_penalty",
    "alias",
    "reasoning",
)


def recipes_dir() -> Path:
    """Recipe files. ``SPARKSI_RECIPES_DIR`` wins, then ``[controller] recipes_dir``."""
    explicit = os.environ.get("SPARKSI_RECIPES_DIR")
    if explicit:
        return Path(explicit).expanduser()
    from sparksi.config import CFG

    configured = str(CFG.controller.get("recipes_dir") or "").strip()
    if configured:
        return Path(configured).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return xdg / "sparksi" / "recipes"


DEFAULT_NAME = "default"
# Letters, digits, ".", "-" and "_", up to 80, starting with a letter or digit (so a
# name is never "." or ".." and never starts with a dot). Dots allow names like glm-5.3.
_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")
_ACTIVE: ContextVar[tuple[str, str] | None] = ContextVar("sparksi_active_recipe", default=None)


def valid_recipe_name(name: Any) -> bool:
    return isinstance(name, str) and bool(_NAME_RE.fullmatch(name))


def named_dir(model_id: str) -> Path:
    return recipes_dir() / "named" / model_id


def default_recipe_name(model_id: str) -> str:
    try:
        data = json.loads((named_dir(model_id) / "index.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return DEFAULT_NAME
    name = data.get("default") if isinstance(data, dict) else None
    if valid_recipe_name(name):
        return str(name)
    return DEFAULT_NAME


def set_default_name(model_id: str, name: str) -> None:
    target = named_dir(model_id)
    target.mkdir(parents=True, exist_ok=True)
    tmp = target / "index.json.tmp"
    tmp.write_text(json.dumps({"default": name}), encoding="utf-8")
    os.replace(tmp, target / "index.json")


def named_recipe_path(model_id: str, name: str) -> Path:
    if name == default_recipe_name(model_id):
        return recipes_dir() / f"{model_id}.toml"
    return named_dir(model_id) / f"{name}.toml"


@contextmanager
def using_recipe(model_id: str, name: str | None):
    token = _ACTIVE.set((model_id, name) if name else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def recipe_names(model_id: str) -> list[str]:
    names: list[str] = []
    default = default_recipe_name(model_id)
    if (recipes_dir() / f"{model_id}.toml").is_file():
        names.append(default)
    seen = set(names)
    for path in sorted(named_dir(model_id).glob("*.toml")):
        stem = path.stem
        if valid_recipe_name(stem) and stem != default and stem not in seen:
            names.append(stem)
            seen.add(stem)
    return names


def recipe_path(model_id: str) -> Path:
    active = _ACTIVE.get()
    if active is not None and active[0] == model_id:
        return named_recipe_path(model_id, active[1])
    return recipes_dir() / f"{model_id}.toml"


def load_recipe(model_id: str) -> dict[str, Any]:
    path = recipe_path(model_id)
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    unknown = sorted(str(key) for key in data if str(key) not in KEYS)
    if unknown:
        logger().warning("recipe %s has unknown keys: %s", model_id, ", ".join(unknown))
    return {key: data[key] for key in KEYS if key in data}


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return json.dumps(list(value))
    raise TypeError(type(value).__name__)


def _emit(key: str, value: Any) -> str:
    return f"{key} = {_scalar(value)}"


def _emit_table(prefix: str, data: dict[str, Any], lines: list[str]) -> None:
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    for key, value in data.items():
        if isinstance(value, dict):
            tables.append((str(key), value))
        elif value is not None:
            scalars.append((str(key), value))
    for key, value in scalars:
        lines.append(_emit(key, value))
    for key, value in tables:
        path = f"{prefix}.{key}" if prefix else key
        lines.append("")
        lines.append(f"[{path}]")
        _emit_table(path, value, lines)


def _shannon(text: str) -> float:
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def value_looks_like_token(value: str) -> bool:
    """A Hugging Face token prefix, or a long high-entropy env string."""
    text = str(value or "").strip()
    if re.search(r"(?i)hf_[A-Za-z0-9]{8,}", text):
        return True
    if len(text) < 32 or "/" in text or " " in text or "." in text:
        return False
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", text):
        return False
    if sum(ch.isdigit() for ch in text) < 4:
        return False
    return _shannon(text) >= 4.0


def _env_strings(value: Any) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if not isinstance(value, dict):
        return found
    env = value.get("env")
    if isinstance(env, dict):
        for key, item in env.items():
            if isinstance(item, str):
                found.append((str(key), item))
    for child in value.values():
        if isinstance(child, dict):
            found.extend(_env_strings(child))
    return found


def _refuse_token_values(fields: dict[str, Any]) -> None:
    for key, value in _env_strings(fields):
        if value_looks_like_token(value):
            raise ValueError(f"refusing to save env {key}: value looks like a token")


def save_recipe(model_id: str, fields: dict[str, Any]) -> Path:
    """Write the local recipe file and nothing else.

    An env value that looks like a token is refused. The message names the key only.
    """
    unknown = sorted(str(key) for key in fields if str(key) not in KEYS)
    if unknown:
        listed = ", ".join(unknown)
        raise ValueError(f"unknown recipe key {listed}" if len(unknown) == 1 else f"unknown recipe keys: {listed}")
    _refuse_token_values(fields)
    path = recipe_path(model_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Local launch recipe. Sparksi does not write sparksi.toml or catalog.yaml.", ""]
    tables: list[tuple[str, dict[str, Any]]] = []
    for key in KEYS:
        if key not in fields or fields[key] is None:
            continue
        value = fields[key]
        if isinstance(value, dict):
            tables.append((key, value))
            continue
        lines.append(_emit(key, value))
    for key, value in tables:
        lines.append("")
        lines.append(f"[{key}]")
        _emit_table(key, value, lines)
    text = "\n".join(lines) + "\n"
    tmp = path.with_suffix(".toml.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path
