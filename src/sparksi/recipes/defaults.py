# SPDX-License-Identifier: Apache-2.0

"""Default recipe fields. The page shows these; it does not choose them.

Priority, highest first:

1. The saved local recipe (``Save as default for this model``).
2. Settings already known on the catalog card or its launch config.
3. The engine template, sized for the placement and the free memory.
"""

from __future__ import annotations

from typing import Any

from sparksi.models import ModelCard, _load_override, _max_ctx_from
from sparksi.recipes.store import load_recipe
from sparksi.recipes.validate import EXTRA_FLAGS, FIELDS

SOURCE_SAVED = "saved recipe"
SOURCE_LAUNCH = "launch config"
SOURCE_CATALOG = "catalog"
SOURCE_ENGINE = "engine defaults"

_RANK = {SOURCE_ENGINE: 0, SOURCE_CATALOG: 1, SOURCE_LAUNCH: 2, SOURCE_SAVED: 3}
_TITLE = {
    SOURCE_ENGINE: "Engine defaults",
    SOURCE_CATALOG: "Catalog",
    SOURCE_LAUNCH: "Launch config",
    SOURCE_SAVED: "Saved recipe",
}

# Preferred length on a row or launch dict. Not the architecture maximum.
_CTX_KEYS = ("ctx", "default_ctx", "context_length", "n_ctx")
_UTIL_KEYS = ("gpu_mem_util", "gpu_memory_utilization", "mem_fraction")
_DTYPE_KEYS = ("kv_cache_dtype", "kv_dtype")
_TOOL_KEYS = ("tool_parser", "tool_call_parser")
_REASON_KEYS = ("reasoning_parser",)
_IMAGE_KEYS = ("image",)
_SEQ_KEYS = ("max_num_seqs", "concurrency")
_SPEC_KEYS = ("speculative",)
_BYTES_KEYS = ("kv_cache_memory_bytes",)

_ARGV_FIELDS = {
    "--max-model-len": "ctx",
    "--context-length": "ctx",
    "--max_seq_len": "ctx",
    "--gpu-memory-utilization": "gpu_mem_util",
    "--mem-fraction-static": "gpu_mem_util",
    "--kv_cache_free_gpu_memory_fraction": "gpu_mem_util",
    "--kv-cache-dtype": "kv_cache_dtype",
    "--max-num-seqs": "max_num_seqs",
    "--tool-call-parser": "tool_parser",
    "--reasoning-parser": "reasoning_parser",
    "--kv-cache-memory-bytes": "kv_cache_memory_bytes",
}

_CTX_LABELS = {
    4096: "4K",
    8192: "8K",
    16384: "16K",
    32768: "32K",
    65536: "64K",
    131072: "128K",
    262144: "256K",
    524288: "512K",
    1048576: "1M",
}

# Fields the page can show. Empty strings stay so a select can read "off".
_BLANK_OK = {"tool_parser", "reasoning_parser", "extra_flags"}


def _roles(placement: str) -> tuple[str, ...]:
    """Roles a placement covers: "both" is head+worker, "a+b" is each part, else one role (head by default)."""
    text = str(placement or "")
    if text == "both":
        return ("head", "worker")
    if "+" in text:
        parts = tuple(part for part in text.split("+") if part)
        if parts:
            return parts
    return (text,) if text in ("head", "worker", "node3", "node4") else ("head",)


def sized_memory_fraction(
    weights_gib: float,
    placement: str,
    budgets: dict[str, float],
    umas: dict[str, float],
) -> float:
    """Share of UMA the engine should claim.

    One Spark starts at 0.80 and two Sparks at 0.85. A model that needs more
    than that share, and still fits in the free budget, steps up. A tight
    budget steps down. The result stays on a 0.05 step inside 0.30–0.95.
    """
    roles = _roles(placement)
    base = 0.85 if len(roles) >= 2 else 0.80
    weight = float(weights_gib) / float(len(roles))
    picked: list[float] = []
    for role in roles:
        uma = float(umas.get(role) or 121.0) or 121.0
        budget = float(budgets.get(role, uma))
        room = max(0.0, min(budget, 0.95 * uma))
        practice = base * uma
        target = practice if weight + 8.0 <= practice else max(practice, weight + 8.0)
        target = min(target, room)
        frac = (target / uma) if uma else base
        frac = min(0.95, max(0.30, frac))
        snapped = round(frac * 20) / 20
        snapped = min(0.95, max(0.30, snapped))
        while snapped > 0.30 and (snapped * uma) > budget + 0.01:
            snapped = round(snapped - 0.05, 2)
        picked.append(snapped)
    return round(min(picked) if picked else base, 2)


def sized_ctx(
    weights_gib: float,
    placement: str,
    budgets: dict[str, float],
    max_ctx: int | None,
) -> int:
    """8K unless the free memory after the weights cannot hold the KV reserve."""
    roles = _roles(placement)
    weight = float(weights_gib) / float(len(roles))
    leftover: float | None = None
    for role in roles:
        room = float(budgets.get(role, 111.0)) - weight
        leftover = room if leftover is None else min(leftover, room)
    ctx = 4096 if leftover is not None and leftover < 8 else 8192
    if max_ctx is not None and ctx > int(max_ctx):
        ctx = int(max_ctx)
    return ctx


def _first(raw: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in raw and raw.get(key) not in (None, ""):
            return raw.get(key)
    return None


def _put(found: dict[str, Any], sources: dict[str, str], key: str, value: Any, source: str) -> None:
    if value in (None, ""):
        return
    found[key] = value
    sources[key] = source


def _from_mapping(raw: dict[str, Any], source: str, found: dict[str, Any], sources: dict[str, str]) -> None:
    if not isinstance(raw, dict):
        return
    _put(found, sources, "ctx", _as_int(_first(raw, _CTX_KEYS)), source)
    _put(found, sources, "gpu_mem_util", _as_float(_first(raw, _UTIL_KEYS)), source)
    dtype = _first(raw, _DTYPE_KEYS)
    if isinstance(dtype, str) and dtype.strip():
        _put(found, sources, "kv_cache_dtype", dtype.strip(), source)
    tool = _first(raw, _TOOL_KEYS)
    if isinstance(tool, str) and tool.strip():
        _put(found, sources, "tool_parser", tool.strip(), source)
    reason = _first(raw, _REASON_KEYS)
    if isinstance(reason, str) and reason.strip():
        _put(found, sources, "reasoning_parser", reason.strip(), source)
    image = _first(raw, _IMAGE_KEYS)
    if isinstance(image, str) and image.strip():
        _put(found, sources, "image", image.strip(), source)
    _put(found, sources, "max_num_seqs", _as_int(_first(raw, _SEQ_KEYS)), source)
    spec = _first(raw, _SPEC_KEYS)
    if isinstance(spec, dict):
        _put(found, sources, "speculative", spec, source)
    elif spec not in (None, ""):
        flag = _as_bool(spec)
        if flag is not None:
            _put(found, sources, "speculative", flag, source)
    _put(found, sources, "kv_cache_memory_bytes", _as_int(_first(raw, _BYTES_KEYS)), source)
    flags = raw.get("extra_flags")
    if isinstance(flags, list) and flags and all(isinstance(item, str) for item in flags):
        _put(found, sources, "extra_flags", [str(item) for item in flags], source)


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
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


def _argv_lists(row: dict[str, Any]) -> list[list[str]]:
    lists: list[list[str]] = []
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    start = launch.get("start") if isinstance(launch, dict) else None
    if isinstance(start, list) and start:
        lists.append([str(part) for part in start])
    seats = row.get("seats") if isinstance(row.get("seats"), dict) else {}
    for seat in seats.values():
        if isinstance(seat, dict) and isinstance(seat.get("start"), list) and seat["start"]:
            lists.append([str(part) for part in seat["start"]])
    return lists


def _looks_like_image(part: str, allow: set[str]) -> bool:
    if part in allow:
        return True
    if part.startswith(("-", "/")) or "=" in part:
        return False
    return "/" in part and ":" in part


def _from_argv(
    argv: list[str],
    engine: str,
    allowed_images: list[str],
    found: dict[str, Any],
    sources: dict[str, str],
) -> None:
    allow = set(allowed_images or [])
    flags = EXTRA_FLAGS.get(engine) or set()
    extra: list[str] = []
    llama = any(part == "llama-server" or part.endswith("/llama-server") for part in argv)
    index = 0
    while index < len(argv):
        part = argv[index]
        if part in flags and part not in extra:
            extra.append(part)
        nxt = argv[index + 1] if index + 1 < len(argv) else None
        field = _ARGV_FIELDS.get(part)
        if part == "-c" and llama:
            field = "ctx"
        if field and nxt is not None and not str(nxt).startswith("-"):
            if field == "ctx":
                _put(found, sources, field, _as_int(nxt), SOURCE_LAUNCH)
            elif field == "gpu_mem_util":
                _put(found, sources, field, _as_float(nxt), SOURCE_LAUNCH)
            elif field in ("max_num_seqs", "kv_cache_memory_bytes"):
                _put(found, sources, field, _as_int(nxt), SOURCE_LAUNCH)
            else:
                _put(found, sources, field, str(nxt), SOURCE_LAUNCH)
            index += 2
            continue
        if part in ("--speculative-config", "--spec-type"):
            _put(found, sources, "speculative", True, SOURCE_LAUNCH)
        if _looks_like_image(part, allow):
            _put(found, sources, "image", part, SOURCE_LAUNCH)
        index += 1
    if extra:
        _put(found, sources, "extra_flags", extra, SOURCE_LAUNCH)


def _known(
    card: ModelCard,
    row: dict[str, Any],
    resident: dict[str, Any],
    override: dict[str, Any],
    engine: str,
    allowed_images: list[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    found: dict[str, Any] = {}
    sources: dict[str, str] = {}
    if card.ctx:
        _put(found, sources, "ctx", int(card.ctx), SOURCE_CATALOG)
    if card.hints.get("mtp"):
        _put(found, sources, "speculative", True, SOURCE_CATALOG)
    need = [str(item) for item in (card.flags.get("need") or [])]
    allow = EXTRA_FLAGS.get(engine) or set()
    needed = [item for item in need if item in allow]
    if needed:
        _put(found, sources, "extra_flags", needed, SOURCE_CATALOG)
    _from_mapping(row, SOURCE_CATALOG, found, sources)
    _from_mapping(resident, SOURCE_CATALOG, found, sources)
    _from_mapping(override, SOURCE_CATALOG, found, sources)
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    if isinstance(launch, dict):
        _from_mapping({key: value for key, value in launch.items() if key not in ("start", "stop", "host")}, SOURCE_LAUNCH, found, sources)
    for argv in _argv_lists(row):
        _from_argv(argv, engine, allowed_images, found, sources)
    return found, sources


def _engine_values(
    engine: str,
    card: ModelCard,
    placement: str,
    budgets: dict[str, float],
    umas: dict[str, float],
    image: str,
    max_ctx: int | None,
) -> dict[str, Any]:
    ctx = sized_ctx(card.weights_gib, placement, budgets, max_ctx)
    util = sized_memory_fraction(card.weights_gib, placement, budgets, umas)
    if engine == "vllm":
        values: dict[str, Any] = {
            "ctx": ctx,
            "gpu_mem_util": util,
            "max_num_seqs": 4,
            "kv_cache_dtype": "fp8",
            "speculative": False,
            "tool_parser": "",
            "reasoning_parser": "",
            "extra_flags": [],
        }
    elif engine == "sglang":
        values = {
            "ctx": ctx,
            "gpu_mem_util": util,
            "speculative": False,
            "tool_parser": "",
            "reasoning_parser": "",
            "extra_flags": [],
        }
    elif engine == "trtllm":
        values = {"ctx": ctx, "gpu_mem_util": util, "extra_flags": []}
    elif engine == "llamacpp":
        values = {"ctx": ctx, "speculative": False, "extra_flags": []}
    else:
        values = {}
    if image and engine in ("vllm", "sglang", "trtllm"):
        values["image"] = image
    return values


def _summary(origin: str, values: dict[str, Any]) -> str:
    bits: list[str] = []
    ctx = values.get("ctx")
    if isinstance(ctx, int) and ctx > 0:
        bits.append(f"{_CTX_LABELS.get(ctx, str(ctx))} context")
    dtype = values.get("kv_cache_dtype")
    if isinstance(dtype, str) and dtype:
        bits.append(f"{dtype} KV")
    if values.get("kv_cache_memory_bytes"):
        bits.append("KV bytes")
    elif values.get("gpu_mem_util") is not None:
        bits.append(f"{float(values['gpu_mem_util']):.2f} memory")
    title = _TITLE.get(origin, "Engine defaults")
    if not bits:
        return title
    return title + ": " + ", ".join(bits)


def _max_ctx(card: ModelCard, row: dict[str, Any], override: dict[str, Any]) -> int | None:
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    for source in (override, launch if isinstance(launch, dict) else {}, row):
        found = _max_ctx_from(source)
        if found:
            return found
    if card.max_ctx and int(card.max_ctx) >= 256:
        return int(card.max_ctx)
    return None


def _pinned_defaults(saved: dict[str, Any], max_ctx: int | None) -> dict[str, Any] | None:
    """A pinned recipe is one source: imported from the capture."""
    if saved.get("kind") != "pinned" or not isinstance(saved.get("roles"), dict):
        return None
    from sparksi.recipes.args import parse_launch

    note = str(saved.get("imported_from") or "an imported launch")
    origin = note if str(note).startswith("imported from") else f"imported from {note}"
    roles = saved["roles"]
    head = roles.get("head") if isinstance(roles.get("head"), dict) else next(iter(roles.values()), {})
    if not isinstance(head, dict):
        head = {}
    command, known, rows = parse_launch([str(part) for part in (head.get("args") or [])])
    values: dict[str, Any] = {}
    source: dict[str, str] = {}
    for key in (
        "ctx", "gpu_mem_util", "kv_cache_memory_bytes", "kv_cache_dtype",
        "max_num_seqs", "tool_parser", "reasoning_parser", "speculative",
    ):
        if known.get(key) not in (None, ""):
            values[key] = known[key]
            source[key] = origin
    if head.get("image"):
        values["image"] = str(head["image"])
        source["image"] = origin
    values["command"] = " ".join(command)
    values["extra_args"] = list(rows)
    source["command"] = origin
    source["extra_args"] = origin
    worker = roles.get("worker") if isinstance(roles.get("worker"), dict) else {}
    if worker:
        worker_command, _known, worker_rows = parse_launch([str(part) for part in (worker.get("args") or [])])
        values["command_worker"] = " ".join(worker_command)
        values["extra_args_worker"] = list(worker_rows)
    env = head.get("env") if isinstance(head.get("env"), dict) else {}
    if env:
        values["env"] = {str(key): str(value) for key, value in env.items()}
        source["env"] = origin
    if values.get("ctx") in (None, ""):
        for key, raw in env.items():
            if str(key).upper() != "MAX_MODEL_LEN":
                continue
            number = _as_int(raw)
            if number:
                values["ctx"] = number
                source["ctx"] = origin
            break
    from sparksi.recipes.pinned import roce_permissions_line

    line = roce_permissions_line(roles if isinstance(roles, dict) else {})
    if line and str(saved.get("placement") or "") == "both":
        values["roce_permissions"] = line
        source["roce_permissions"] = origin
    token_note = str(saved.get("token_note") or "")
    if token_note:
        values["token_note"] = token_note
        source["token_note"] = origin
    for key in ("repo", "revision"):
        text = str(saved.get(key) or "").strip()
        if text:
            values[key] = text
            source[key] = origin
    if saved.get("evict_page_cache"):
        values["evict_page_cache"] = True
        source["evict_page_cache"] = origin
    summary = _summary("imported", values).replace("Engine defaults", "Imported recipe", 1)
    try:
        port = int(saved.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    return {
        "values": values,
        "source": source,
        "origin": origin,
        "summary": summary,
        "max_ctx": max_ctx,
        "pinned": True,
        "engine": str(saved.get("engine") or ""),
        "placement": str(saved.get("placement") or ""),
        "port": port,
    }


def derive_defaults(
    card: ModelCard,
    engine: str,
    placement: str,
    *,
    saved: dict[str, Any] | None = None,
    row: dict[str, Any] | None = None,
    resident: dict[str, Any] | None = None,
    override: dict[str, Any] | None = None,
    budgets: dict[str, float] | None = None,
    umas: dict[str, float] | None = None,
    image: str = "",
    allowed_images: list[str] | None = None,
) -> dict[str, Any]:
    """Values, a source per field, and the one-line summary the page shows."""
    row = row or {}
    resident = resident or {}
    if override is None:
        override = _load_override(card.id)
    if saved is None:
        saved = load_recipe(card.id)
    budgets = budgets if budgets is not None else {"head": 111.0, "worker": 111.0}
    umas = umas if umas is not None else {"head": 121.0, "worker": 121.0}
    max_ctx = _max_ctx(card, row, override if override is not None else {})
    pinned = _pinned_defaults(saved or {}, max_ctx)
    if pinned is not None:
        return pinned
    allowed = FIELDS.get(engine)
    if not allowed:
        return {
            "values": {},
            "source": {},
            "origin": SOURCE_ENGINE,
            "summary": _TITLE[SOURCE_ENGINE],
            "max_ctx": max_ctx,
        }
    found, sources = _known(card, row, resident, override, engine, list(allowed_images or []))
    for key, value in (saved or {}).items():
        if key in ("engine", "placement", "port") or value in (None, ""):
            continue
        found[key] = value
        sources[key] = SOURCE_SAVED
    engine_values = _engine_values(engine, card, placement, budgets, umas, image, max_ctx)
    for key, value in engine_values.items():
        if key not in found:
            found[key] = value
            sources[key] = SOURCE_ENGINE
    values = {key: found[key] for key in found if key in allowed and (found[key] not in (None, "") or key in _BLANK_OK)}
    source = {key: sources[key] for key in values if key in sources}
    if not source:
        origin = SOURCE_ENGINE
    else:
        origin = max(source.values(), key=lambda name: _RANK.get(name, 0))
    return {
        "values": values,
        "source": source,
        "origin": origin,
        "summary": _summary(origin, values),
        "max_ctx": max_ctx,
    }
