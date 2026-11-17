# SPDX-License-Identifier: Apache-2.0

"""Parse and rewrite well-known server flags. Everything else stays a raw token.

The browser edits fields. This module rewrites the matching argv elements and
leaves every other element in the extra-arguments list, one token per row.
Nothing is joined for a shell.
"""

from __future__ import annotations

import json
from typing import Any

# Flag that consumes the next argv element. ``--host`` is kept out of the
# extra list because render always rewrites it to the tailnet bind.
VALUE_FLAGS = {
    "--max-model-len": "ctx",
    "--context-length": "ctx",
    "--max_seq_len": "ctx",
    "--ctx-size": "ctx",
    "-c": "ctx",
    "--gpu-memory-utilization": "gpu_mem_util",
    "--mem-fraction-static": "gpu_mem_util",
    "--kv_cache_free_gpu_memory_fraction": "gpu_mem_util",
    "--kv-cache-memory-bytes": "kv_cache_memory_bytes",
    "--kv-cache-memory": "kv_cache_memory_bytes",
    "--kv-cache-dtype": "kv_cache_dtype",
    "-ctk": "kv_cache_dtype",
    "-ctv": "kv_cache_dtype",
    "--max-num-seqs": "max_num_seqs",
    "-np": "max_num_seqs",
    "--port": "port",
    "-p": "port",
    "--tool-call-parser": "tool_parser",
    "--reasoning-parser": "reasoning_parser",
    "--host": "bind",
    "--speculative-config": "speculative_config",
    "--spec-type": "spec_type",
    "--spec-draft-n-max": "spec_tokens",
    "--model-draft": "spec_model",
    "--speculative-algorithm": "spec_algo",
    "--speculative-num-steps": "spec_tokens",
    "--speculative-draft-model-path": "spec_model",
}

# Preferred flag when a field is set and the imported argv has no match.
PREFERRED = {
    "vllm": {
        "ctx": "--max-model-len",
        "gpu_mem_util": "--gpu-memory-utilization",
        "kv_cache_memory_bytes": "--kv-cache-memory-bytes",
        "kv_cache_dtype": "--kv-cache-dtype",
        "max_num_seqs": "--max-num-seqs",
        "port": "--port",
        "tool_parser": "--tool-call-parser",
        "reasoning_parser": "--reasoning-parser",
    },
    "sglang": {
        "ctx": "--context-length",
        "gpu_mem_util": "--mem-fraction-static",
        "port": "--port",
        "tool_parser": "--tool-call-parser",
        "reasoning_parser": "--reasoning-parser",
    },
    "trtllm": {
        "ctx": "--max_seq_len",
        "gpu_mem_util": "--kv_cache_free_gpu_memory_fraction",
        "port": "--port",
    },
    "llamacpp": {
        "ctx": "-c",
        "kv_cache_dtype": "-ctk",
        "max_num_seqs": "-np",
        "port": "--port",
    },
    "tensorfold": {
        "ctx": "--max-model-len",
        "port": "--port",
    },
    "other": {
        "ctx": "--max-model-len",
        "port": "--port",
    },
}

_SCALAR = (
    "ctx",
    "gpu_mem_util",
    "kv_cache_memory_bytes",
    "kv_cache_dtype",
    "max_num_seqs",
    "port",
    "tool_parser",
    "reasoning_parser",
)


def _coerce(field: str, raw: str) -> Any:
    if field in ("ctx", "kv_cache_memory_bytes", "max_num_seqs", "port"):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if field == "gpu_mem_util":
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    return raw


def _spec_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    spec = {"method": "", "model": "", "tokens": 0}
    blob = raw.get("speculative_config")
    if blob:
        try:
            data = json.loads(str(blob))
        except (json.JSONDecodeError, TypeError, ValueError):
            data = {}
        if isinstance(data, dict):
            spec["method"] = str(data.get("method") or "")
            spec["model"] = str(data.get("model") or "")
            try:
                spec["tokens"] = int(data.get("num_speculative_tokens") or data.get("tokens") or 0)
            except (TypeError, ValueError):
                spec["tokens"] = 0
    if raw.get("spec_type") or raw.get("spec_algo"):
        spec["method"] = str(raw.get("spec_type") or raw.get("spec_algo") or spec["method"])
    if raw.get("spec_model"):
        spec["model"] = str(raw["spec_model"])
    if raw.get("spec_tokens") not in (None, ""):
        try:
            spec["tokens"] = int(raw["spec_tokens"])
        except (TypeError, ValueError):
            pass
    return spec


def _paired(token: str, nxt: str | None) -> bool:
    return nxt is not None and not str(nxt).startswith("-")


def row_tokens(row: str) -> list[str]:
    """One Extra arguments row back into argv elements."""
    text = str(row or "").strip()
    if not text:
        return []
    if text.startswith("-") and " " in text:
        flag, value = text.split(" ", 1)
        return [flag, value]
    return [text]


def parse_launch(args: list[str]) -> tuple[list[str], dict[str, Any], list[str]]:
    """Command prefix, known fields, and extra rows.

    A flag and its value share one row. The leading non-flag tokens (``vllm serve <model>``)
    are the command. Rebuilding the rows plus the known flags is the stored argv.
    """
    command: list[str] = []
    index = 0
    while index < len(args) and not str(args[index]).startswith("-"):
        command.append(args[index])
        index += 1
    raw: dict[str, Any] = {}
    rows: list[str] = []
    while index < len(args):
        token = args[index]
        nxt = args[index + 1] if index + 1 < len(args) else None
        if token in VALUE_FLAGS and _paired(token, nxt):
            raw[VALUE_FLAGS[token]] = nxt
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            flag, value = token.split("=", 1)
            if flag in VALUE_FLAGS:
                raw[VALUE_FLAGS[flag]] = value
                index += 1
                continue
            rows.append(token)
            index += 1
            continue
        if str(token).startswith("-") and _paired(token, nxt):
            rows.append(f"{token} {nxt}")
            index += 2
            continue
        rows.append(token)
        index += 1
    fields: dict[str, Any] = {}
    for key in _SCALAR:
        if key in raw and raw[key] not in (None, ""):
            fields[key] = _coerce(key, str(raw[key]))
    spec = _spec_from_raw(raw)
    if spec["method"] or spec["model"] or spec["tokens"]:
        fields["speculative"] = spec
    return command, fields, rows


def parse_args(args: list[str]) -> tuple[dict[str, Any], list[str]]:
    """Return ``(known fields, extra rows)``. ``--host`` is not an extra row."""
    _command, fields, rows = parse_launch(args)
    return fields, rows


def speculative_json(spec: dict[str, Any]) -> str:
    payload: dict[str, Any] = {"method": spec.get("method") or "mtp"}
    if spec.get("model"):
        payload["model"] = spec["model"]
    tokens = int(spec.get("tokens") or 1)
    payload["num_speculative_tokens"] = tokens
    return json.dumps(payload, separators=(",", ":"))


def _field_value(field: str, token: str, current: str, updates: dict[str, Any]) -> str | None:
    """New argv text for one known flag, or None to drop it."""
    if field == "bind":
        return current
    if field == "speculative_config":
        spec = updates.get("speculative")
        if not isinstance(spec, dict):
            return current
        if not spec.get("method"):
            return None
        return speculative_json(spec)
    if field == "spec_type":
        spec = updates.get("speculative")
        if not isinstance(spec, dict):
            return current
        if not spec.get("method"):
            return None
        return str(spec["method"])
    if field == "spec_algo":
        spec = updates.get("speculative")
        if not isinstance(spec, dict):
            return current
        if not spec.get("method"):
            return None
        method = str(spec["method"])
        if method in ("mtp", "nextn"):
            return "NEXTN"
        return method
    if field == "spec_tokens":
        spec = updates.get("speculative")
        if not isinstance(spec, dict):
            return current
        if not spec.get("method"):
            return None
        return str(int(spec.get("tokens") or 1))
    if field == "spec_model":
        spec = updates.get("speculative")
        if not isinstance(spec, dict):
            return current
        model = str(spec.get("model") or "")
        return model or None
    if field not in updates:
        return current
    value = updates[field]
    if value in (None, ""):
        return None
    if field == "gpu_mem_util":
        return f"{float(value):.2f}"
    return str(value)


def _append_missing(out: list[str], updates: dict[str, Any], seen: set[str], engine: str) -> None:
    prefer = PREFERRED.get(engine) or PREFERRED["other"]
    for field, flag in prefer.items():
        if field in seen or field not in updates:
            continue
        value = updates[field]
        if value in (None, ""):
            continue
        text = f"{float(value):.2f}" if field == "gpu_mem_util" else str(value)
        out.extend([flag, text])
    spec = updates.get("speculative")
    if not isinstance(spec, dict) or not spec.get("method"):
        return
    if "speculative_config" in seen or "spec_type" in seen or "spec_algo" in seen:
        return
    if engine == "llamacpp":
        method = "draft-mtp" if spec["method"] == "mtp" else str(spec["method"])
        out.extend(["--spec-type", method])
        if spec.get("tokens"):
            out.extend(["--spec-draft-n-max", str(int(spec["tokens"]))])
        if spec.get("model"):
            out.extend(["--model-draft", str(spec["model"])])
    elif engine == "sglang":
        method = "NEXTN" if spec["method"] in ("mtp", "nextn") else str(spec["method"])
        out.extend(["--speculative-algorithm", method])
        if spec.get("tokens"):
            out.extend(["--speculative-num-steps", str(int(spec["tokens"]))])
        if spec.get("model"):
            out.extend(["--speculative-draft-model-path", str(spec["model"])])
    else:
        out.extend(["--speculative-config", speculative_json(spec)])


def apply_edits(
    args: list[str],
    updates: dict[str, Any],
    extra: list[str] | None,
    engine: str,
) -> list[str]:
    """Rewrite known flags from ``updates``. ``extra`` replaces extra rows.

    ``extra is None`` keeps the imported rows. Each row is a flag, or a flag and its value.
    The leading command is kept. An unchanged parse rebuilds the stored argv.
    """
    out: list[str] = []
    seen: set[str] = set()
    rows = [str(item) for item in extra] if extra is not None else None
    used = 0
    index = 0
    while index < len(args) and not str(args[index]).startswith("-"):
        out.append(args[index])
        index += 1
    while index < len(args):
        token = args[index]
        nxt = args[index + 1] if index + 1 < len(args) else None
        if token in VALUE_FLAGS and _paired(token, nxt):
            field = VALUE_FLAGS[token]
            seen.add(field)
            value = _field_value(field, token, str(nxt), updates)
            if value is not None:
                out.extend([token, value])
            index += 2
            continue
        if token.startswith("--") and "=" in token:
            flag, current = token.split("=", 1)
            if flag in VALUE_FLAGS:
                field = VALUE_FLAGS[flag]
                seen.add(field)
                value = _field_value(field, flag, current, updates)
                if value is not None:
                    out.append(f"{flag}={value}")
                index += 1
                continue
        span = 2 if str(token).startswith("-") and _paired(token, nxt) else 1
        if rows is None:
            out.extend(args[index:index + span])
        elif used < len(rows):
            out.extend(row_tokens(rows[used]))
            used += 1
        index += span
    if rows is not None:
        while used < len(rows):
            out.extend(row_tokens(rows[used]))
            used += 1
    _append_missing(out, updates, seen, engine)
    return out


def _listen_host(value: str) -> bool:
    """A server bind, not an mpirun host list (``ip:slots,ip:slots``)."""
    text = str(value or "")
    if text in ("0.0.0.0", "::", "*"):
        return True
    if "," in text or (":" in text and text != "::"):
        return False
    return True


def rewrite_bind(args: list[str], bind: str) -> list[str]:
    """Replace the listen ``--host`` / ``--bind`` with the node's tailnet address.

    ``mpirun --host ip:1,ip:1`` is left alone.
    """
    if not bind:
        return list(args)
    out: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in ("--host", "--bind") and index + 1 < len(args):
            if _listen_host(args[index + 1]):
                out.extend([token, bind])
            else:
                out.extend([token, args[index + 1]])
            index += 2
            continue
        if token.startswith("--host=") or token.startswith("--bind="):
            flag, value = token.split("=", 1)
            out.append(f"{flag}={bind}" if _listen_host(value) else token)
            index += 1
            continue
        out.append(token)
        index += 1
    return out


def rendezvous_ports(args: list[str]) -> list[int]:
    """Master and dist ports carried beside the service port. An off-limits port is never one."""
    flags = {"--master-port", "--rdzv-port", "--dist-port"}
    found: list[int] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in flags and index + 1 < len(args):
            try:
                number = int(args[index + 1])
            except (TypeError, ValueError):
                number = 0
            if number and number not in found:
                found.append(number)
            index += 2
            continue
        if token == "--dist-init-addr" and index + 1 < len(args) and ":" in args[index + 1]:
            try:
                number = int(args[index + 1].rsplit(":", 1)[-1])
            except (TypeError, ValueError):
                number = 0
            if number and number not in found:
                found.append(number)
        index += 1
    return found
