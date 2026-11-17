# SPDX-License-Identifier: Apache-2.0

"""Turn a card plus structured fields into per-node argv. Hosts come from config."""

from __future__ import annotations

from typing import Any

from sparksi import protect
from sparksi.config import CFG
from sparksi.health import health_paths_for, health_timeout_for
from sparksi.models import ModelCard
from sparksi.placement import engine_view, is_multi, needs_text
from sparksi.recipes.engines import BUILDERS
from sparksi.recipes.spec import Spec
from sparksi.recipes.validate import PORT_MIN, normalize_speculative


def _evict_on(value: Any) -> bool:
    from sparksi.recipes.pinned import flag_on

    return flag_on(value)


def _evict_path(path: str, role: str) -> str:
    from sparksi.recipes.pinned import resolve_evict_path

    return resolve_evict_path(
        {"runtime": "native", "args": ["-m", str(path or "")], "mounts": []},
        role,
    )


def _role_host(role: str) -> str:
    return CFG.node_by_role(role).name


def _bind(role: str) -> str:
    return CFG.node_by_role(role).tailscale


def _fabric(role: str) -> tuple[str, str, list[str]]:
    fab = CFG.node_by_role(role).fabric
    return fab.ip, fab.interface, list(fab.hcas)


def _relabel(body: dict[str, Any], stands_for: dict[str, str]) -> None:
    """Give each built node the role of the Spark it runs on (identity for head/worker/both)."""
    if all(role == real for role, real in stands_for.items()):
        return
    for node in body.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("role") or "") in stands_for:
            node["role"] = stands_for[str(node["role"])]
    if isinstance(body.get("order"), list):
        body["order"] = [stands_for.get(str(role), role) for role in body["order"]]
    if isinstance(body.get("argv_preview"), dict):
        body["argv_preview"] = {stands_for.get(str(role), role): argv for role, argv in body["argv_preview"].items()}


def default_image(engine: str) -> str:
    return CFG.engine_image(engine)


def spec_from(
    card: ModelCard,
    *,
    engine: str,
    placement: str,
    port: int,
    fields: dict[str, Any],
) -> Spec:
    image = str(fields.get("image") or default_image(engine) or "")
    # Builders know "head", "worker" and "both". A third or fourth Spark, or a
    # pair other than head+worker, is built under those words and each builder
    # role reads the Spark it stands for (placement.engine_view).
    built_as, stands_for = engine_view(placement)
    if not built_as:
        raise ValueError(needs_text(placement) or f"cannot place a run on {placement!r}")
    paths = {}
    hosts = {}
    binds = {}
    fabric_ip = {}
    fabric_iface = {}
    fabric_hcas = {}
    roles = ["head", "worker"] if built_as == "both" else [built_as]
    for role in roles:
        real = stands_for.get(role, role)
        host = _role_host(real)
        hosts[role] = host
        binds[role] = _bind(real)
        ip, iface, hcas = _fabric(real)
        fabric_ip[role] = ip
        fabric_iface[role] = iface
        fabric_hcas[role] = hcas
        if host in card.files:
            paths[role] = card.files[host]
    forbid = "prefix_cache" in (card.flags.get("forbid") or []) or "prefix-caching" in (card.flags.get("forbid") or [])
    need_prefix = "prefix_cache" in (card.flags.get("need") or [])
    prefix = bool(fields.get("prefix_cache")) if "prefix_cache" in fields else (need_prefix and not forbid)
    ctx = int(fields.get("ctx") or card.ctx or 8192)
    return Spec(
        model_id=card.id,
        placement=built_as,
        port=int(port),
        paths=paths,
        hosts=hosts,
        binds=binds,
        fabric_ip=fabric_ip,
        fabric_iface=fabric_iface,
        fabric_hcas=fabric_hcas,
        image=image,
        hf_cache=CFG.hf_cache(),
        ctx=ctx,
        gpu_mem_util=float(fields.get("gpu_mem_util") if fields.get("gpu_mem_util") is not None else 0.80),
        max_num_seqs=int(fields.get("max_num_seqs") or 4),
        kv_cache_dtype=str(fields.get("kv_cache_dtype") or "fp8"),
        prefix_cache=prefix,
        kv_cache_memory_bytes=fields.get("kv_cache_memory_bytes"),
        speculative=fields["speculative"] if "speculative" in fields else None,
        tool_parser=str(fields.get("tool_parser") or ""),
        reasoning_parser=str(fields.get("reasoning_parser") or ""),
        extra_flags=list(fields.get("extra_flags") or []),
        binary=str(fields.get("binary") or ""),
        ld_library_path=str(fields.get("ld_library_path") or ""),
        max_num_batched_tokens=fields.get("max_num_batched_tokens"),
        attention_backend=str(fields.get("attention_backend") or ""),
        mamba_ssm_dtype=str(fields.get("mamba_ssm_dtype") or ""),
        quantization=str(fields.get("quantization") or ""),
        load_format=str(fields.get("load_format") or ""),
        limit_mm_per_prompt=str(fields.get("limit_mm_per_prompt") or ""),
        temperature=fields.get("temperature"),
        top_p=fields.get("top_p"),
        top_k=fields.get("top_k"),
        presence_penalty=fields.get("presence_penalty"),
        alias=str(fields.get("alias") or ""),
        reasoning=str(fields.get("reasoning") or ""),
        hints=dict(card.hints),
        master_port=CFG.master_port(),
        dist_port=CFG.sglang_dist_port(),
        scripts=dict(card.scripts),
        forbid_prefix=forbid,
    )


def _public_fields(spec: Spec, timeout: Any) -> dict[str, Any]:
    mode, _why = normalize_speculative(spec.speculative)
    if spec.speculative is None and spec.hints.get("mtp"):
        mode = {"method": "mtp", "model": "", "tokens": 1}
    return {
        "ctx": spec.ctx,
        "gpu_mem_util": spec.gpu_mem_util,
        "max_num_seqs": spec.max_num_seqs,
        "kv_cache_dtype": spec.kv_cache_dtype,
        "kv_cache_memory_bytes": spec.kv_cache_memory_bytes,
        "speculative": mode or {"method": "", "model": "", "tokens": 0},
        "tool_parser": spec.tool_parser,
        "reasoning_parser": spec.reasoning_parser,
        "image": spec.image,
        "extra_flags": list(spec.extra_flags),
        "timeout_s": timeout,
        "max_num_batched_tokens": spec.max_num_batched_tokens,
        "attention_backend": spec.attention_backend,
        "mamba_ssm_dtype": spec.mamba_ssm_dtype,
        "quantization": spec.quantization,
        "load_format": spec.load_format,
        "limit_mm_per_prompt": spec.limit_mm_per_prompt,
        "binary": spec.binary,
        "ld_library_path": spec.ld_library_path,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "presence_penalty": spec.presence_penalty,
        "alias": spec.alias,
        "reasoning": spec.reasoning,
    }


def render(card: ModelCard, *, engine: str, placement: str, port: int, fields: dict[str, Any], catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    builder = BUILDERS[engine]
    spec = spec_from(card, engine=engine, placement=placement, port=port, fields=fields)
    built = builder(spec)
    _built_as, stands_for = engine_view(placement)
    row = {
        "id": card.id,
        "port": int(port),
        "kind": engine,
        "placement": "pair" if is_multi(placement) else "single",
        "timeout_s": fields.get("timeout_s") if fields.get("timeout_s") is not None else card.timeout_s,
        "probe": None,
    }
    paths = health_paths_for(row)
    if engine == "custom":
        # A custom row's own probe path replaces the engine default when it has one.
        custom = health_paths_for({"port": int(port), "kind": "custom", "probe": fields.get("probe")})
        if custom:
            paths = custom
        elif not paths:
            paths = list(built.health_paths)
    timeout = health_timeout_for(catalog, {"id": card.id, "timeout_s": row["timeout_s"], "start_timeout_s": card.timeout_s})
    body = built.to_dict()
    _relabel(body, stands_for)
    body.update({
        "id": card.id,
        "engine": engine,
        "placement": placement,
        "port": int(port),
        "timeout_s": timeout,
        "health_paths": paths or list(built.health_paths),
        "image": spec.image,
        "ctx": spec.ctx,
        "experimental": built.experimental,
        "experimental_reason": built.experimental_reason,
        "fields": _public_fields(spec, timeout),
        "evict_page_cache": _evict_on(fields.get("evict_page_cache")),
    })
    built_role = {real: role for role, real in stands_for.items()}
    for node in body.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        role_name = str(node.get("role") or "head")
        node["evict_page_cache"] = body["evict_page_cache"]
        path = spec.paths.get(built_role.get(role_name, role_name), "")
        node["evict_path"] = _evict_path(path, role_name) if body["evict_page_cache"] else ""
    if isinstance(body.get("fields"), dict):
        body["fields"]["evict_page_cache"] = body["evict_page_cache"]
    if protect.is_off_limits(port) or int(port) < PORT_MIN:
        body["health_paths"] = []
    return body
