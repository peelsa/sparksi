# SPDX-License-Identifier: Apache-2.0

"""Group fleet data by machine. Pure: no I/O, no addresses in the result.

Inputs are the public shapes the dashboard already fetches:

* ``nodes`` from ``GET /api/config`` (``role``, ``id``, ``name``).
* ``models`` from the catalog (``id``, ``title``, ``port``, ``placement``,
  optional ``seats``).
* ``live`` from ``GET /api/monitor`` (``hosts``, ``loaded``).
* ``library`` rows from ``GET /api/fleet`` (``id``, ``on``, ``gib``).

``controller`` and ``extras`` are quiet name-only lines. They have no
announcer, so they do not grow stats or models.
"""

from __future__ import annotations

from typing import Any

_STAT_KEYS = (
    "mem_used_gib",
    "mem_total_gib",
    "mem_pct",
    "cpu_pct",
    "gpu_pct",
    "gpu_temp_f",
    "soc_temp_f",
)


def group_machines(
    nodes: list[dict[str, Any]],
    models: list[dict[str, Any]],
    live: dict[str, Any] | None = None,
    library: list[dict[str, Any]] | None = None,
    controller: str | dict[str, Any] | None = None,
    extras: list[Any] | None = None,
) -> dict[str, Any]:
    """Return ``{"machines": [...], "loads": [...]}``.

    Head and worker keep config order. A pair model is listed on every
    machine it seats, and ``loads`` contains that model once. A host with
    an age that is no longer fresh is ``last heard Ns ago``. A host with
    no age is ``no report yet``.

    A catalog model that declares ``identity.served`` or
    ``identity.served_prefix`` matches a loaded row only by that served
    name (the ``served`` field, not a truncated ``name``). A shared port
    is not enough. A loaded row that no catalog model claims still appears
    on its machine as read-only, marked ``Not in catalog``.
    """
    live = live or {}
    library = library or []
    hosts = {_host_key(row): row for row in live.get("hosts") or [] if _host_key(row)}
    loaded = [row for row in live.get("loaded") or []]
    lib_by_id = {str(row.get("id")): row for row in library if row.get("id")}

    machines: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("role") not in ("head", "worker", "node3", "node4"):
            continue
        machines.append(_machine(node, models, hosts, loaded, lib_by_id, site=nodes))
    if controller:
        name = controller.get("name") if isinstance(controller, dict) else str(controller)
        if name:
            machines.append(_quiet("controller", str(name)))
    for extra in extras or []:
        name = extra.get("name") if isinstance(extra, dict) else str(extra)
        if name:
            machines.append(_quiet("extra", str(name)))
    return {"machines": machines, "loads": _loads(machines)}


def _host_key(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("name") or "")


def _quiet(role: str, name: str) -> dict[str, Any]:
    return {
        "role": role,
        "id": "",
        "name": name,
        "quiet": True,
        "report": "missing",
        "status": "no report yet",
        "age_s": None,
        "dimmed": False,
        "stats": None,
        "disk": None,
        "archive_mounted": None,
        "models": [],
    }


def _machine(
    node: dict[str, Any],
    models: list[dict[str, Any]],
    hosts: dict[str, dict[str, Any]],
    loaded: list[dict[str, Any]],
    lib_by_id: dict[str, dict[str, Any]],
    site: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    host = hosts.get(str(node.get("id") or "")) or hosts.get(str(node.get("name") or ""))
    report, status, age_s, dimmed = _report(host)
    seated = [model for model in models if _seated(model, node, site)]
    shown: list[dict[str, Any]] = []
    if report != "missing":
        matched: set[int] = set()
        for model in seated:
            live = _match_loaded(model, node, loaded)
            if live is not None:
                matched.add(id(live))
            shown.append(_model_view(model, node, loaded, lib_by_id.get(str(model.get("id")))))
        for rec in loaded:
            if not _on_node(rec, node) or id(rec) in matched:
                continue
            shown.append(_unlisted_view(rec))
    return {
        "role": node.get("role"),
        "id": str(node.get("id") or ""),
        "name": str(node.get("name") or node.get("role") or ""),
        "quiet": False,
        "report": report,
        "status": status,
        "age_s": age_s,
        "dimmed": dimmed,
        "stats": None if report == "missing" else _stats(host),
        "disk": None if report == "missing" else _disk(host),
        "archive_mounted": None if report == "missing" else _archive(host),
        "models": shown,
    }


def _report(host: dict[str, Any] | None) -> tuple[str, str, int | None, bool]:
    if not host or host.get("age") is None:
        return "missing", "no report yet", None, False
    age = host.get("age")
    seconds = int(float(f"{float(age):.0f}"))
    if host.get("fresh"):
        return "fresh", f"fresh {seconds}s", seconds, False
    return "stale", f"last heard {seconds}s ago", seconds, True


def _keys(node: dict[str, Any]) -> set[str]:
    return {str(key) for key in (node.get("name"), node.get("id"), node.get("role")) if key}


def _named(model: dict[str, Any]) -> set[str]:
    """Hosts a model names: its run nodes, its seats, then its hosts."""
    out: set[str] = set()
    for item in model.get("nodes") or []:
        if isinstance(item, dict):
            out.update(str(item[key]) for key in ("host", "name") if item.get(key))
    out.update(str(key) for key in (model.get("seats") or {}))
    out.update(str(host) for host in model.get("hosts") or [])
    return out


def _seated(model: dict[str, Any], node: dict[str, Any], site: list[dict[str, Any]] | None = None) -> bool:
    if _is_pair(model):
        # A pair sits on the two Sparks it names. One that names fewer of this site's Sparks is head+worker.
        named = _named(model)
        if sum(1 for other in site or [node] if _keys(other) & named) >= 2:
            return bool(_keys(node) & named)
        return node.get("role") in ("head", "worker")
    seats = model.get("seats") or {}
    if seats:
        keys = {str(node.get("name") or ""), str(node.get("id") or ""), str(node.get("role") or "")}
        return any(str(key) in keys for key in seats)
    hosts = [str(h) for h in model.get("hosts") or []]
    if hosts:
        return str(node.get("name") or "") in hosts or str(node.get("id") or "") in hosts
    return node.get("role") == "head"


def _is_pair(model: dict[str, Any]) -> bool:
    if model.get("pair") is True or str(model.get("placement") or "").lower() == "pair":
        return True
    seats = model.get("seats") or {}
    return len(seats) >= 2


def _port(model: dict[str, Any], node: dict[str, Any]) -> int | None:
    seats = model.get("seats") or {}
    for key in (node.get("name"), node.get("id"), node.get("role")):
        seat = seats.get(key) if key else None
        if isinstance(seat, dict) and seat.get("port") is not None:
            return int(seat["port"])
        if isinstance(seat, int):
            return seat
    if model.get("port") is None:
        return None
    return int(model["port"])


def _model_view(
    model: dict[str, Any],
    node: dict[str, Any],
    loaded: list[dict[str, Any]],
    library_row: dict[str, Any] | None,
) -> dict[str, Any]:
    live = _match_loaded(model, node, loaded)
    state, rate = _state(live)
    presence = None
    if library_row:
        on = library_row.get("on") or {}
        for key in (node.get("name"), node.get("id"), node.get("role")):
            if key and key in on:
                presence = on[key]
                break
    return {
        "id": str(model.get("id") or ""),
        "name": str(model.get("title") or model.get("name") or model.get("id") or ""),
        "port": _port(model, node),
        "pair": _is_pair(model),
        "state": state,
        "rate": rate,
        "presence": presence,
        "gib": None if not library_row else library_row.get("gib"),
    }


def _on_node(rec: dict[str, Any], node: dict[str, Any]) -> bool:
    keys = {str(node.get("id") or ""), str(node.get("name") or "")}
    keys.discard("")
    host = str(rec.get("host") or "")
    span = {str(item) for item in rec.get("span") or []}
    return bool(keys & ({host} | span))


def _served_identity(model: dict[str, Any]) -> tuple[set[str], str] | None:
    """Served names and prefix, or None when a bare port match is allowed.

    ``identity`` is the catalog field. ``identities`` is accepted as the
    same dict so a row can declare either name.
    """
    raw = model.get("identity")
    if not isinstance(raw, dict):
        raw = model.get("identities")
    if not isinstance(raw, dict):
        return None
    served = raw.get("served") or []
    if isinstance(served, str):
        served = [served]
    names = {str(item).strip().lower() for item in served if str(item).strip()}
    prefix = str(raw.get("served_prefix") or "").strip()
    if not names and not prefix:
        return None
    return names, prefix


def _loaded_label(rec: dict[str, Any]) -> str:
    """Prefer ``served``. ``name`` may be cut short with an ellipsis."""
    served = rec.get("served")
    if served not in (None, ""):
        return str(served).strip()
    return str(rec.get("name") or "").strip()


def _match_loaded(model: dict[str, Any], node: dict[str, Any], loaded: list[dict[str, Any]]) -> dict[str, Any] | None:
    port = _port(model, node)
    title = str(model.get("title") or model.get("name") or "").strip().lower()
    ident = str(model.get("id") or "")
    declared = _served_identity(model)
    for rec in loaded:
        if not _on_node(rec, node):
            continue
        names = {
            str(rec.get("name") or "").strip().lower(),
            str(rec.get("id") or "").strip(),
            _loaded_label(rec).lower(),
        }
        names.discard("")
        if ident and ident in names:
            return rec
        if title and title in names:
            return rec
        if declared is not None:
            want, prefix = declared
            label = _loaded_label(rec)
            folded = label.lower()
            if folded and folded in want:
                return rec
            if prefix and folded.startswith(prefix.lower()):
                return rec
            continue
        if port is not None and rec.get("port") is not None and int(rec["port"]) == port:
            return rec
    return None


def _unlisted_view(rec: dict[str, Any]) -> dict[str, Any]:
    state, rate = _state(rec)
    label = _loaded_label(rec) or "unnamed"
    port = None if rec.get("port") is None else int(rec["port"])
    return {
        "id": f"unlisted:{port}:{label}",
        "name": label,
        "port": port,
        "pair": False,
        "state": state,
        "rate": rate,
        "presence": None,
        "gib": None,
        "catalog": False,
        "note": "Not in catalog",
    }


def _state(rec: dict[str, Any] | None) -> tuple[str, str]:
    if not rec:
        return "idle", ""
    raw = str(rec.get("state") or "").upper()
    rate = ""
    tok = rec.get("tok_s")
    if tok not in (None, ""):
        rate = f"{float(tok):.0f} t/s"
    if raw == "DOWN":
        return "down", ""
    if raw in ("GEN", "GENERATING", "VIDEO"):
        return "generating", rate
    if raw == "BUSY":
        return "busy", rate
    return "idle", ""


def _stats(host: dict[str, Any] | None) -> dict[str, Any] | None:
    if not host:
        return None
    if not any(host.get(key) is not None for key in _STAT_KEYS):
        return None
    return {key: host.get(key) for key in _STAT_KEYS if key in host}


def _disk(host: dict[str, Any] | None) -> dict[str, Any] | None:
    if not host:
        return None
    if host.get("disk_used_gib") is None and host.get("disk_total_gib") is None:
        return None
    return {"used_gib": host.get("disk_used_gib"), "total_gib": host.get("disk_total_gib")}


def _archive(host: dict[str, Any] | None) -> bool | None:
    if not host:
        return None
    if "mount_archive" in host:
        return bool(host.get("mount_archive"))
    if "archive_mounted" in host:
        return bool(host.get("archive_mounted"))
    return None


def _loads(machines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for machine in machines:
        if machine.get("quiet"):
            continue
        for model in machine.get("models") or []:
            if model.get("catalog") is False:
                continue
            rec = found.setdefault(
                model["id"],
                {
                    "id": model["id"],
                    "name": model["name"],
                    "port": model["port"],
                    "pair": bool(model.get("pair")),
                    "on": [],
                },
            )
            role = machine.get("role")
            if role and role not in rec["on"]:
                rec["on"].append(role)
    return list(found.values())
