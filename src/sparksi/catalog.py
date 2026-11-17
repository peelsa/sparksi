# SPDX-License-Identifier: Apache-2.0

"""Load and validate the side-view catalog."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from sparksi import protect
from sparksi.config import CFG, env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Search order: $SPARKSI_CATALOG, [controller] catalog, ~/.config/sparksi/catalog.yaml,
# ./config/catalog.yaml, then the shipped example. A source checkout uses config/;
# a wheel install uses the copy packaged under sparksi/examples/ (kept byte-identical by a test).
PACKAGED_EXAMPLE_CATALOG = Path(__file__).resolve().parent / "examples" / "catalog.example.yaml"
_CHECKOUT_EXAMPLE_CATALOG = PROJECT_ROOT / "config" / "catalog.example.yaml"
EXAMPLE_CATALOG = _CHECKOUT_EXAMPLE_CATALOG if _CHECKOUT_EXAMPLE_CATALOG.is_file() else PACKAGED_EXAMPLE_CATALOG



class CatalogError(ValueError):
    pass


def catalog_path() -> Path:
    explicit = env("SPARKSI_CATALOG") or str(CFG.controller.get("catalog") or "")
    if explicit:
        return Path(explicit).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    for cand in (xdg / "sparksi" / "catalog.yaml", Path.cwd() / "config" / "catalog.yaml"):
        if cand.is_file():
            return cand
    return EXAMPLE_CATALOG


def model_inventory_path() -> Path:
    return Path(env("SPARKSI_MODEL_INVENTORY") or CFG.path("model_inventory"))


def _normalize_rows(data: dict[str, Any]) -> None:
    for row in data.get("library") or []:
        if not isinstance(row, dict):
            continue
        for old, new in (CFG.compat.get("legacy_row_keys") or {}).items():
            if old in row and new not in row:
                row[new] = row.pop(old)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CatalogError(f"missing yaml: {path}")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise CatalogError(f"expected mapping in {path}")
    return data


def load_catalog(path: Path | None = None) -> dict[str, Any]:
    data = load_yaml(path or catalog_path())
    for key in ("host", "tenants", "residents", "recipes"):
        if key not in data:
            raise CatalogError(f"catalog missing {key}")
    if not isinstance(data["residents"], list):
        raise CatalogError("residents must be a list")
    if not isinstance(data["recipes"], dict):
        raise CatalogError("recipes must be a mapping")
    if not isinstance(data["tenants"], dict):
        raise CatalogError("tenants must be a mapping")
    _normalize_rows(data)
    return data


def _synthetic_recipe(catalog: dict[str, Any], members: list[str]) -> dict[str, Any] | None:
    uma = float((catalog.get("host") or {}).get("uma_gib") or 121)
    est = 0.0
    for mid in members:
        row = resident_by_id(catalog, mid) if "residents" in catalog else None
        if not row:
            return None
        if protect.is_video_kind(row.get("kind")):
            return {
                "estimate_gib": None,
                "unit": "declared_vram_gib",
                "source": "none",
                "confidence": "unknown",
                "members": members,
                "notes": "no trusted video service runtime ceiling",
            }
        util = row.get("gpu_memory_utilization")
        if util is None:
            return None
        est += float(util) * uma
    return {
        "estimate_gib": round(est, 2),
        "unit": "declared_vram_gib",
        "source": "gpu_memory_utilization",
        "confidence": "declared",
        "members": members,
        "notes": "declared from catalog gpu_memory_utilization × UMA",
    }


def recipe_for(catalog: dict[str, Any], name: str) -> tuple[str, dict[str, Any]]:
    recipes: dict[str, Any] = catalog["recipes"]
    if name in recipes:
        return name, recipes[name]
    for rid, rec in recipes.items():
        aliases = rec.get("aliases") or []
        if name in aliases:
            return rid, rec
    tokens = [t.strip() for t in str(name).split(",") if t.strip()]
    syn = _synthetic_recipe(catalog, tokens) if tokens else None
    if syn:
        return name, syn
    known = sorted(list(recipes) + [a for r in recipes.values() for a in (r.get("aliases") or [])])
    raise CatalogError(f"unknown set {name!r}. known: {', '.join(known)}")


def core_recipe(catalog: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """The co-resident core stack: the recipe marked ``core: true``."""
    for rid, rec in (catalog.get("recipes") or {}).items():
        if isinstance(rec, dict) and rec.get("core"):
            return rid, rec
    return None


def core_ids(catalog: dict[str, Any]) -> frozenset[str]:
    found = core_recipe(catalog)
    return frozenset(found[1].get("members") or []) if found else frozenset()


def core_recipe_names(catalog: dict[str, Any]) -> set[str]:
    found = core_recipe(catalog)
    if not found:
        return set()
    rid, rec = found
    return {rid, *[str(a) for a in rec.get("aliases") or []]}


def resident_by_id(catalog: dict[str, Any], rid: str) -> dict[str, Any] | None:
    for row in catalog["residents"]:
        if row.get("id") == rid:
            return row
    return None


def listen_host_for(row: dict[str, Any]) -> str:
    return str(row.get("listen_host") or "127.0.0.1")
