# SPDX-License-Identifier: Apache-2.0

"""Fail-closed fit check. Never OK from free/MemAvailable alone.

Scores: always-on floor + measured live GPU + cost of *starts* + margin
versus MemTotal. Does not stack gpu_memory_utilization fractions as if
each one reserved the whole box. MemAvailable is reported, not used.
"""

from __future__ import annotations

from typing import Any

from sparksi.catalog import recipe_for, resident_by_id


def _uma_comparable(catalog: dict[str, Any]) -> tuple[bool, str]:
    host = catalog.get("host") or {}
    uma = host.get("uma")
    if uma is None:
        uma = host.get("uma_gib") is not None and str(host.get("swap", "")).lower() in (
            "off",
            "false",
            "0",
        )
    if uma:
        return True, (
            "UMA: floor (always-on RSS/catalog_min) + live GPU (measured) "
            "+ start costs (named recipe or per-id) + margin"
        )
    return False, "cannot compare RSS floor to VRAM recipe honestly (host is not UMA)"


def _named_recipe(catalog: dict[str, Any], rid: str) -> bool:
    return rid in (catalog.get("recipes") or {})


def cost_id(catalog: dict[str, Any], mid: str) -> float | None:
    """GiB to seat one catalog id that is not already live."""
    recipes = catalog.get("recipes") or {}
    for rec in recipes.values():
        members = list(rec.get("members") or [])
        est = rec.get("estimate_gib")
        if members == [mid] and est is not None and rec.get("confidence") in ("declared", "measured", None):
            if rec.get("confidence") in ("declared", "measured"):
                return float(est)
    try:
        row = resident_by_id(catalog, mid) or {}
    except Exception:
        row = {}
    host = catalog.get("host") or {}
    uma = float(host.get("uma_gib") or 121)
    util = row.get("gpu_memory_utilization")
    if util is not None:
        return round(float(util) * uma, 2)
    return None


def _live_gpu_gib(residents: list[dict[str, Any]]) -> float:
    total = 0.0
    for r in residents:
        if not r.get("resident"):
            continue
        g = r.get("gpu_gib")
        if g is None:
            continue
        try:
            total += float(g)
        except (TypeError, ValueError):
            continue
    return round(total, 3)


def evaluate(
    catalog: dict[str, Any],
    set_name: str,
    *,
    floor: dict[str, Any],
    mem: dict[str, Any],
    residents: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    rid, recipe = recipe_for(catalog, set_name)
    host = catalog.get("host") or {}
    margin = float(host.get("dry_run_margin_gib") or 4.0)
    total = mem.get("mem_total_gib")
    if total is None:
        total = host.get("uma_gib")

    named = _named_recipe(catalog, rid)
    estimate = recipe.get("estimate_gib")
    confidence = recipe.get("confidence") or "unknown"
    source = recipe.get("source") or "none"
    recipe_unit = recipe.get("unit") or "declared_vram_gib"
    members = list(recipe.get("members") or [])

    live = [r for r in (residents or []) if r.get("resident")]
    live_ids = {r.get("id") for r in live if r.get("id")}
    live_gpu = _live_gpu_gib(live)
    starts = [m for m in members if m not in live_ids]

    reasons: list[str] = []
    comparable, compare_note = _uma_comparable(catalog)

    start_cost: float | None
    if named and estimate is None:
        start_cost = None
    elif not members and estimate is None:
        start_cost = None
    elif named and estimate is not None and not any(m in live_ids for m in members):
        start_cost = float(estimate)
        source = recipe.get("source") or "catalog"
        confidence = recipe.get("confidence") or "declared"
    else:
        parts: list[float] = []
        missing: list[str] = []
        for mid in starts:
            c = cost_id(catalog, mid)
            if c is None:
                missing.append(mid)
            else:
                parts.append(float(c))
        if missing:
            start_cost = None
            reasons.append("no start cost for: " + ", ".join(missing))
            if not named:
                confidence = "unknown"
        else:
            start_cost = round(sum(parts), 2) if starts else 0.0
            if starts:
                source = "start_cost"
                confidence = "declared"
                estimate = start_cost
                recipe_unit = "start_cost_gib"

    if named and estimate is None and start_cost is None:
        reasons.append(f"recipe {rid} has no estimate_gib ({recipe.get('notes') or 'unknown runtime'})")
    if confidence not in ("declared", "measured"):
        reasons.append(f"recipe confidence is {confidence!r}, not declared/measured")
    if not floor.get("complete"):
        holes = floor.get("reasons") or ["floor incomplete"]
        reasons.append("tenant floor incomplete: " + "; ".join(holes))
    if floor.get("total_gib") is None:
        reasons.append("tenant floor total unknown")
    if total is None:
        reasons.append("MemTotal unknown")
    if not comparable:
        reasons.append(compare_note)

    avail = mem.get("mem_avail_gib")
    free = mem.get("mem_free_gib")

    needed = None
    if comparable and start_cost is not None and floor.get("total_gib") is not None:
        needed = round(float(floor["total_gib"]) + live_gpu + float(start_cost) + margin, 3)

    can_decide = (
        comparable
        and start_cost is not None
        and confidence in ("declared", "measured")
        and floor.get("complete")
        and floor.get("total_gib") is not None
        and total is not None
        and needed is not None
    )

    if can_decide:
        if needed > float(total):
            verdict = "NO"
            reasons.append(
                f"floor {floor['total_gib']} + live GPU {live_gpu} + starts {start_cost} "
                f"+ margin {margin} = {needed} > total {total}"
            )
        else:
            verdict = "OK"
            reasons.append(
                f"floor {floor['total_gib']} + live GPU {live_gpu} + starts {start_cost} "
                f"+ margin {margin} = {needed} <= total {total}"
            )
    else:
        verdict = "UNCERTAIN"
        if not reasons:
            reasons.append("insufficient inputs")
        if needed is None and start_cost is not None and floor.get("total_gib") is not None and not comparable:
            reasons.append("refusing to add RSS floor to VRAM recipe without UMA")

    floor_parts = []
    for p in floor.get("parts") or []:
        floor_parts.append(
            {
                **p,
                "label": (
                    f"{p.get('gib')} GiB {p.get('unit') or '?'} "
                    f"{p.get('confidence')} ({p.get('instrument') or p.get('measure')})"
                ),
            }
        )

    return {
        "set": rid,
        "requested": set_name,
        "recipe": {
            "estimate_gib": estimate,
            "unit": recipe_unit,
            "source": source,
            "confidence": confidence,
            "members": members,
            "notes": recipe.get("notes"),
            "start_cost_gib": start_cost,
            "starts": starts,
            "live_gpu_gib": live_gpu,
        },
        "floor": {**floor, "parts": floor_parts or floor.get("parts") or []},
        "margin_gib": margin,
        "needed_gib": needed,
        "mem_total_gib": total,
        "mem_avail_gib": avail,
        "mem_free_gib": free,
        "avail_not_used_for_verdict": True,
        "comparable": comparable,
        "compare_note": compare_note,
        "verdict": verdict,
        "reasons": reasons,
        "residents_informational": [
            {
                "id": r.get("id"),
                "seat": r.get("seat") or r.get("role"),
                "port": r.get("port"),
                "identity": r.get("identity"),
                "gpu_gib": r.get("gpu_gib"),
            }
            for r in live
        ],
        "note": "Apply keeps live models; fit is floor + live GPU + new starts + margin",
    }
