# SPDX-License-Identifier: Apache-2.0

"""Assemble status / inventory / dry-run payloads."""

from __future__ import annotations

import time
from typing import Any

from sparksi.catalog import load_yaml, model_inventory_path
from sparksi import compat
from sparksi import protect
from sparksi.compat import with_legacy_fields
from sparksi.dryrun import evaluate
from sparksi.lock import Lock
from sparksi.probe import conflicts, port_listening, snapshot, tenant_floor
from sparksi.truth_diff import diff_catalog_vs_truth

# The protected video port (preset on) is listed when a node reports it listening. Status does not connect.


def _load_model_inventory() -> tuple[dict[str, Any] | None, str | None]:
    try:
        return load_yaml(model_inventory_path()), None
    except Exception as e:
        return None, str(e)


def _compat_mem(mem: dict[str, Any] | None) -> dict[str, Any]:
    mem = mem or {}
    return {
        "mem_total_gib": mem.get("total_gib"),
        "mem_avail_gib": mem.get("available_gib"),
        "mem_free_gib": mem.get("free_gib"),
        "mem_used_gib": mem.get("used_gib"),
        "swap_total_gib": mem.get("swap_total_gib"),
        "swap_free_gib": mem.get("swap_free_gib"),
    }


def _node_list(nodes: dict[str, Any] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if isinstance(nodes, dict):
        rows = nodes.get("nodes") or []
    else:
        rows = nodes or []
    return [row for row in rows if isinstance(row, dict)]


def _listed_ports(nodes: list[dict[str, Any]]) -> set[int]:
    found: set[int] = set()
    for node in nodes:
        for port in node.get("ports") or []:
            try:
                found.add(int(port))
            except (TypeError, ValueError):
                continue
    return found


def _node_for(row: dict[str, Any], nodes: list[dict[str, Any]]) -> dict[str, Any]:
    host = str(row.get("listen_host") or "")
    by_key: dict[str, dict[str, Any]] = {}
    for node in nodes:
        for key in (node.get("id"), node.get("name"), node.get("role")):
            if key:
                by_key[str(key)] = node
    if host and host in by_key:
        return by_key[host]
    for node in nodes:
        if node.get("role") == "head":
            return node
    return nodes[0] if nodes else {}


def _residents_from_nodes(catalog: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Catalog rows scored from node facts. A missing unit or weight path is stale."""
    seats = catalog.get("seats") or {}
    out: list[dict[str, Any]] = []
    for row in catalog.get("residents") or []:
        if not isinstance(row, dict):
            continue
        node = _node_for(row, nodes)
        unit_name = row.get("unit") or None
        weights = row.get("weights") or None
        unit_info = (node.get("units") or {}).get(unit_name) if unit_name else None
        load_state = (unit_info or {}).get("load_state")
        unit_missing = bool(unit_name) and load_state in ("not-found", "missing")
        path_state = (node.get("paths") or {}).get(weights) if weights else None
        path_missing = bool(weights) and path_state is False
        stale = bool(unit_missing or path_missing)
        try:
            port = int(row["port"]) if row.get("port") not in (None, "") else None
        except (TypeError, ValueError):
            port = None
        listening = bool(port) and port in _listed_ports([node])
        container = row.get("container")
        names = {item.get("name") for item in (node.get("containers") or []) if isinstance(item, dict)}
        container_up = bool(container) and container in names
        resident = (listening or container_up) and not stale
        if unit_name:
            active = (unit_info or {}).get("active_state") or "unknown"
            if load_state in ("not-found", "missing"):
                active = "inactive"
            unit = {
                "unit": unit_name,
                "is_active": active,
                "is_enabled": "unknown",
                "active_state": active,
                "sub_state": None,
                "unit_file_state": load_state,
            }
        else:
            unit = None
        rid = row.get("id")
        seat = None
        for seat_name, seat_id in seats.items():
            if seat_id == rid:
                seat = seat_name
                break
        if seat is None and row.get("role") in ("brain", "vision", "counsel", "video"):
            seat = row.get("role")
        out.append({
            "id": rid,
            "kind": row.get("kind"),
            "role": row.get("role"),
            "seat": seat,
            "served_name": row.get("served_name"),
            "port": port,
            "resident": resident,
            "stale": stale,
            "listening": listening,
            "http_ok": None,
            "identity": None,
            "gpu_apps": [],
            "gpu_gib": None,
            "unit": unit,
            "smi_error": None,
            "path_missing": path_missing,
            "unit_missing": unit_missing,
        })
    return out


def _units_from_nodes(catalog: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, str]] = {}
    for node in nodes:
        for name, info in (node.get("units") or {}).items():
            if isinstance(info, dict):
                found[str(name)] = info
    rows: list[dict[str, Any]] = []
    for name in catalog.get("units_status") or []:
        info = found.get(str(name)) or {}
        active = info.get("active_state") or "unknown"
        if info.get("load_state") in ("not-found", "missing"):
            active = "inactive"
        rows.append({
            "unit": str(name),
            "is_active": active,
            "is_enabled": "unknown",
            "active_state": active,
            "sub_state": None,
            "unit_file_state": info.get("load_state"),
        })
    return rows


def _apply_nodes(
    snap: dict[str, Any],
    catalog: dict[str, Any],
    nodes: list[dict[str, Any]],
    truth: dict[str, Any] | None,
    docker_fn=None,
) -> None:
    """Replace controller-local memory and residents with each Spark's own facts."""
    head = next((node for node in nodes if node.get("role") == "head"), nodes[0] if nodes else None)
    snap["nodes"] = nodes
    snap["mem"] = _compat_mem((head or {}).get("mem") if head and head.get("reachable") else None)
    if head and not head.get("reachable"):
        snap["mem_error"] = head.get("error")
    snap["residents"] = _residents_from_nodes(catalog, nodes)
    snap["units"] = _units_from_nodes(catalog, nodes)
    snap["conflicts"] = conflicts(catalog, snap["residents"], snap["units"], truth)
    listed = _listed_ports(nodes)

    video_port = protect.video_port()

    def listen_fn(port: int, host: str = "127.0.0.1") -> bool:
        if video_port is not None and int(port) == video_port:
            return video_port in listed
        return port_listening(port, host)

    snap["floor"] = tenant_floor(catalog, listen_fn=listen_fn, docker_fn=docker_fn)
    snap["video_service_live"] = video_port is not None and video_port in listed
    with_legacy_fields(snap)
    snap["ts"] = snap.get("ts") or time.strftime("%Y-%m-%dT%H:%M:%S%z")


def assemble_status(
    catalog: dict[str, Any],
    truth: dict[str, Any] | None,
    lock: Lock,
    truth_err: str | None = None,
    nodes: dict[str, Any] | list[dict[str, Any]] | None = None,
    docker_fn=None,
) -> dict[str, Any]:
    """``docker_fn`` replaces the ~2 s ``docker stats`` sample (the server passes a cached one)."""
    from sparksi.wanted import load_wanted, seat_view

    node_rows = _node_list(nodes)
    if node_rows:
        snap = {
            "ts": "",
            "host": catalog.get("host"),
            "mem": _compat_mem(None),
            "floor": {},
            "residents": [],
            "units": [],
            "conflicts": [],
        }
        _apply_nodes(snap, catalog, node_rows, truth, docker_fn=docker_fn)
    else:
        snap = snapshot(catalog, truth, docker_fn=docker_fn)
        video_port = protect.video_port()
        snap["video_service_live"] = video_port is not None and bool(port_listening(video_port, "127.0.0.1"))
        with_legacy_fields(snap)
    snap["lock"] = lock.read().to_dict()
    wanted = load_wanted()
    snap["wanted"] = wanted
    snap["seats"] = seat_view(catalog, snap["residents"], wanted)
    if truth_err:
        snap["truth_error"] = truth_err
    snap["slice"] = 4
    snap["mutating"] = False
    # last dry-run is stored wanted; label stale for UI
    if isinstance(wanted, dict) and wanted.get("last_dry_run"):
        snap["last_dry_run_stale"] = True
    return snap


def assemble_inventory(
    catalog: dict[str, Any],
    truth: dict[str, Any] | None,
    truth_err: str | None = None,
) -> dict[str, Any]:
    snap = snapshot(catalog, truth)
    diff = None
    if truth is not None:
        diff = diff_catalog_vs_truth(catalog, truth)
    return with_legacy_fields({
        "slice": 2,
        "catalog_host": catalog.get("host"),
        "catalog_tenants": list((catalog.get("tenants") or {}).keys()),
        "catalog_residents": [
            {
                "id": r.get("id"),
                "role": r.get("role"),
                "port": r.get("port"),
                "unit": r.get("unit"),
                "served_name": r.get("served_name"),
            }
            for r in catalog.get("residents") or []
        ],
        "recipes": {
            k: {
                "estimate_gib": v.get("estimate_gib"),
                "source": v.get("source"),
                "confidence": v.get("confidence"),
                "aliases": v.get("aliases") or [],
            }
            for k, v in (catalog.get("recipes") or {}).items()
        },
        "live": {
            "residents": snap["residents"],
            "units": snap["units"],
            "floor": snap["floor"],
            "conflicts": snap["conflicts"],
            "mem": snap["mem"],
        },
        "diff_vs_model_inventory": diff,
        "truth_error": truth_err,
        "wrote_model_inventory": False,
    })


def run_dry_run(
    catalog: dict[str, Any],
    set_name: str,
    truth: dict[str, Any] | None,
    truth_err: str | None = None,
) -> dict[str, Any]:
    snap = snapshot(catalog, truth)
    result = evaluate(
        catalog,
        set_name,
        floor=snap["floor"],
        mem=snap["mem"],
        residents=snap["residents"],
    )
    result["conflicts"] = snap["conflicts"]
    if truth_err:
        result["truth_error"] = truth_err
    return result


def format_status(data: dict[str, Any]) -> str:
    mem = data.get("mem") or {}
    lock = data.get("lock") or {}
    floor = data.get("floor") or {}
    wanted = data.get("wanted") or {}
    lines = [
        "sparksi status  mutating=no",
        f"  host: {(data.get('host') or {}).get('name')}  "
        f"UMA={(data.get('host') or {}).get('uma_gib')} GiB  swap={(data.get('host') or {}).get('swap')}",
        f"  mem:  total={mem.get('mem_total_gib')}  "
        f"used={mem.get('mem_used_gib')}  "
        f"avail={mem.get('mem_avail_gib')}  free={mem.get('mem_free_gib')}  "
        f"(avail is NOT a fit signal)",
        f"  lock: {lock.get('on')}  holder={lock.get('holder')}  {lock.get('reason')}",
        f"  wanted: {wanted.get('name') or '(unset)'}",
    ]
    for node in data.get("nodes") or []:
        nmem = node.get("mem") or {}
        flag = "up" if node.get("reachable") else "down"
        err = f"  {node.get('error')}" if node.get("error") else ""
        lines.append(
            f"  node {node.get('role')} {node.get('name')}: {flag}  "
            f"total={nmem.get('total_gib')} used={nmem.get('used_gib')} avail={nmem.get('available_gib')}{err}"
        )
    for s in data.get("seats") or []:
        live = "up" if s.get("resident") else "down"
        lines.append(f"    {s.get('seat')} → {s.get('id')} :{s.get('port')}  {live}")
    parts = floor.get("parts") or []
    bits = []
    for p in parts:
        bits.append(f"{p['name']}={p.get('gib')} {p.get('unit') or p.get('confidence')}")
    lines.append(
        f"  floor: {floor.get('total_gib')} GiB  complete={floor.get('complete')}  ({', '.join(bits)})"
    )
    lines.append("  residents (name → id → port):")
    any_r = False
    for r in data.get("residents") or []:
        if not r.get("resident"):
            continue
        any_r = True
        unit = (r.get("unit") or {}).get("is_active")
        seat = r.get("seat") or "-"
        lines.append(
            f"    - {seat} → {r['id']} :{r.get('port')}  http={'ok' if r.get('http_ok') else 'no'}  "
            f"served={r.get('identity')}  gpu={r.get('gpu_gib')} GiB  unit={unit}"
            + ("  stale" if r.get("stale") else "")
        )
    if not any_r:
        lines.append("    (none live)")
    for r in data.get("residents") or []:
        if r.get("stale"):
            lines.append(f"    - {r.get('id')} :{r.get('port')}  stale")
    conf = data.get("conflicts") or []
    lines.append("  conflict: " + ("; ".join(conf) if conf else "(none)"))
    return "\n".join(lines)


def format_inventory(data: dict[str, Any]) -> str:
    lines = ["sparksi inventory  (side view; the model inventory file is not written)"]
    diff = compat.field(data, "diff_vs_model_inventory")
    if diff is None:
        lines.append(f"  truth: unavailable ({data.get('truth_error')})")
    else:
        lines.append(f"  drift: {diff.get('drift')}  stack={diff.get('stack')}")
        for m in diff.get("matched") or []:
            lines.append(f"  match: {m['role']} {m.get('catalog_id')}")
        for m in diff.get("mismatches") or []:
            bad = [f for f, v in (m.get("fields") or {}).items() if not v.get("match")]
            lines.append(f"  mismatch: {m['role']} fields={bad}")
        if diff.get("catalog_only_roles"):
            lines.append(f"  catalog_only: {diff['catalog_only_roles']}")
        if diff.get("truth_only_roles"):
            lines.append(f"  truth_only: {diff['truth_only_roles']}")
    lines.append("  live residents:")
    live = (data.get("live") or {}).get("residents") or []
    shown = False
    for r in live:
        if r.get("resident"):
            shown = True
            lines.append(f"    - {r['id']} :{r.get('port')}  {r.get('identity')}")
    if not shown:
        lines.append("    (none)")
    conf = (data.get("live") or {}).get("conflicts") or []
    lines.append("  conflict: " + ("; ".join(conf) if conf else "(none)"))
    return "\n".join(lines)


def format_dry_run(data: dict[str, Any]) -> str:
    rec = data.get("recipe") or {}
    floor = data.get("floor") or {}
    lines = [
        f"sparksi dry-run  set={data.get('set')}",
        f"  floor: {floor.get('total_gib')} GiB  complete={floor.get('complete')}",
    ]
    for p in floor.get("parts") or []:
        unit = p.get("unit") or "?"
        inst = p.get("instrument") or p.get("measure")
        lines.append(
            f"    {p['name']}: {p.get('gib')} GiB  {unit}/{p.get('confidence')}  "
            f"({inst})  {p.get('detail')}"
        )
    lines.append(
        f"  recipe: {rec.get('estimate_gib')} GiB  {rec.get('unit') or 'declared_vram_gib'}  "
        f"source={rec.get('source')}  confidence={rec.get('confidence')}"
    )
    if rec.get("notes"):
        lines.append(f"    {rec['notes']}")
    lines.append(f"  compare: {data.get('compare_note')}")
    lines.append(
        f"  margin: {data.get('margin_gib')}  needed: {data.get('needed_gib')}  "
        f"total: {data.get('mem_total_gib')} GiB MemTotal"
    )
    lines.append(f"  avail={data.get('mem_avail_gib')} GiB MemAvailable (not used for verdict)")
    lines.append(f"  verdict: {data.get('verdict')}")
    for r in data.get("reasons") or []:
        lines.append(f"    - {r}")
    live = data.get("residents_informational") or []
    if live:
        lines.append("  current residents (informational, no eviction):")
        for r in live:
            seat = r.get("seat") or "-"
            lines.append(f"    - {seat} → {r.get('id')} :{r.get('port')}  {r.get('identity')}")
    return "\n".join(lines)


def format_wanted(data: dict[str, Any]) -> str:
    w = data.get("wanted") or {}
    lock = data.get("lock") or {}
    last = w.get("last_dry_run") if isinstance(w, dict) else None
    lines = [
        f"sparksi wanted  name={w.get('name') if w else '(unset)'}",
        f"  lock: {lock.get('on')}  holder={lock.get('holder')}",
        f"  spec: {(w or {}).get('spec')}",
        f"  members: {(w or {}).get('members')}",
    ]
    for s in data.get("seats") or []:
        live = "up" if s.get("resident") else "down"
        lines.append(f"  {s.get('seat')} → {s.get('id')} :{s.get('port')}  {live}")
    if last:
        lines.append(
            f"  last dry-run: {(last or {}).get('verdict')}  set={(last or {}).get('set')}  "
            f"needed={(last or {}).get('needed_gib')}"
        )
        for r in (last or {}).get("reasons") or []:
            lines.append(f"    - {r}")
    else:
        lines.append("  last dry-run: (none)")
    conf = data.get("conflicts") or []
    lines.append("  conflict: " + ("; ".join(conf) if conf else "(none)"))
    lines.append("  (wanted is desired state only — no load/evict)")
    return "\n".join(lines)
