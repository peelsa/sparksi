# SPDX-License-Identifier: Apache-2.0

"""Read-only compare of the side catalog against the model inventory file ([paths] model_inventory)."""

from __future__ import annotations

from typing import Any

COMPARE_FIELDS = ("served_name", "port", "unit", "weights", "gpu_memory_utilization")


def diff_catalog_vs_truth(catalog: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    cat_roles = {}
    for row in catalog.get("residents") or []:
        role = row.get("role")
        if role in ("brain", "vision", "counsel"):
            cat_roles[role] = row

    truth_roles = {}
    for row in truth.get("models") or []:
        role = row.get("role")
        if role:
            truth_roles[role] = row

    matched: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    catalog_only = sorted(set(cat_roles) - set(truth_roles))
    truth_only = sorted(set(truth_roles) - set(cat_roles))

    for role in sorted(set(cat_roles) & set(truth_roles)):
        c, t = cat_roles[role], truth_roles[role]
        fields: dict[str, Any] = {}
        bad = False
        for f in COMPARE_FIELDS:
            cv, tv = c.get(f), t.get(f)
            same = cv == tv
            fields[f] = {"catalog": cv, "truth": tv, "match": same}
            if not same:
                bad = True
        entry = {"role": role, "catalog_id": c.get("id"), "fields": fields}
        if bad:
            mismatches.append(entry)
        else:
            matched.append(entry)

    stack = {
        "catalog_uma_gib": (catalog.get("host") or {}).get("uma_gib"),
        "truth_approx_vram_gb": (truth.get("stack") or {}).get("approx_vram_gb"),
        "truth_stack_id": (truth.get("stack") or {}).get("id"),
        "truth_mode": (truth.get("stack") or {}).get("mode"),
    }

    return {
        "source": "read-only; the model inventory file was not written",
        "matched": matched,
        "mismatches": mismatches,
        "catalog_only_roles": catalog_only,
        "truth_only_roles": truth_only,
        "stack": stack,
        "drift": bool(mismatches or catalog_only or truth_only),
    }
