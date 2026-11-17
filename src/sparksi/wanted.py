# SPDX-License-Identifier: Apache-2.0

"""Wanted set: persist desired residents. Does not load or evict."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparksi.catalog import CatalogError, recipe_for, resident_by_id
from sparksi.lock import state_dir


def wanted_path() -> Path:
    return state_dir() / "wanted.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_wanted() -> dict[str, Any] | None:
    p = wanted_path()
    if not p.is_file():
        return None
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def save_wanted(payload: dict[str, Any]) -> dict[str, Any]:
    p = wanted_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(p)
    return payload


def seats_map(catalog: dict[str, Any]) -> dict[str, str]:
    return dict(catalog.get("seats") or {})


def id_to_seat(catalog: dict[str, Any], rid: str) -> str | None:
    for seat, sid in seats_map(catalog).items():
        if sid == rid:
            return seat
    row = resident_by_id(catalog, rid)
    if row and row.get("role") in ("brain", "vision", "counsel", "video"):
        return row.get("role")
    return None


def resolve_spec(catalog: dict[str, Any], spec: str) -> dict[str, Any]:
    spec = (spec or "").strip()
    if not spec:
        raise CatalogError("empty wanted spec")
    # recipe name / alias
    try:
        rid, rec = recipe_for(catalog, spec)
        members = list(rec.get("members") or [])
        seats = rec.get("seats") or [id_to_seat(catalog, m) for m in members]
        return {
            "name": rid,
            "spec": spec,
            "members": members,
            "seats": {s: m for s, m in zip(seats, members) if s},
        }
    except CatalogError:
        pass

    tokens = [t.strip() for t in spec.split(",") if t.strip()]
    members: list[str] = []
    seats: dict[str, str] = {}
    smap = seats_map(catalog)
    for tok in tokens:
        if tok in smap:
            cid = smap[tok]
            members.append(cid)
            seats[tok] = cid
            continue
        row = resident_by_id(catalog, tok)
        if row:
            members.append(tok)
            seat = id_to_seat(catalog, tok)
            if seat:
                seats[seat] = tok
            continue
        raise CatalogError(
            f"unknown wanted token {tok!r} (not a recipe, seat, or catalog id)"
        )
    name = spec if len(tokens) > 1 else (tokens[0] if tokens else spec)
    return {"name": name, "spec": spec, "members": members, "seats": seats}


def apply_set(
    catalog: dict[str, Any],
    spec: str,
    dry_run: dict[str, Any],
) -> dict[str, Any]:
    resolved = resolve_spec(catalog, spec)
    payload = {
        "name": resolved["name"],
        "spec": resolved["spec"],
        "members": resolved["members"],
        "seats": resolved["seats"],
        "updated_at": _now(),
        "last_dry_run": {
            "set": dry_run.get("set"),
            "verdict": dry_run.get("verdict"),
            "needed_gib": dry_run.get("needed_gib"),
            "reasons": dry_run.get("reasons"),
            "comparable": dry_run.get("comparable"),
        },
    }
    return save_wanted(payload)


def seat_view(catalog: dict[str, Any], residents: list[dict[str, Any]], wanted: dict[str, Any] | None) -> list[dict[str, Any]]:
    by_id = {r.get("id"): r for r in residents}
    smap = seats_map(catalog)
    wanted_seats = (wanted or {}).get("seats") or {}
    rows = []
    seen = set()
    for seat, cid in {**smap, **wanted_seats}.items():
        if seat in seen:
            continue
        seen.add(seat)
        cid = wanted_seats.get(seat, cid)
        row = by_id.get(cid) or resident_by_id(catalog, cid) or {}
        live = by_id.get(cid) or {}
        rows.append(
            {
                "seat": seat,
                "id": cid,
                "port": live.get("port") or row.get("port"),
                "resident": bool(live.get("resident")),
                "identity": live.get("identity"),
            }
        )
    return rows
