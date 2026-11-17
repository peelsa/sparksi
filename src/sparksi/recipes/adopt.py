# SPDX-License-Identifier: Apache-2.0

"""Notice Sparksi containers this process did not start, and record them as runs.

A status poll asks each node which ``sparksi-*`` containers are up. A recipe
with live containers and no active run is adopted: ``started_by`` is
``external``, the port is the one the container is actually listening on, and
the normal health check runs. When those containers disappear, the run is
stopped. It is not recorded as a failure.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from sparksi import protect
from sparksi.config import CFG
from sparksi.placement import placement_for, roles_of
from sparksi.models import row_is_protected, row_is_service
from sparksi.recipes.engines.common import docker_stop
from sparksi.recipes.runs import _save, _stopped_copy, list_runs, ready_check_due, tick
from sparksi.recipes.service import valid_id
from sparksi.recipes.store import load_recipe

_QUIET = frozenset({"stopped", "failed", "timed_out", "cancelled"})
_IDLE_FALLBACK_GIB = 10.0


def idle_path() -> Path:
    """Persisted per-node idle used-memory. Lives beside the run records."""
    from sparksi.recipes.runs import runs_dir

    return runs_dir().parent / "idle-baseline.json"


def read_idle() -> dict[str, float]:
    path = idle_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    found: dict[str, float] = {}
    for key, value in data.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number >= 0:
            found[str(key)] = number
    return found


def configured_idle_gib() -> float:
    """Idle used-memory from config, or 10 GiB when it was never set."""
    raw = CFG.controller.get("idle_baseline_gib")
    if raw in (None, ""):
        return _IDLE_FALLBACK_GIB
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return _IDLE_FALLBACK_GIB
    if number < 0:
        return _IDLE_FALLBACK_GIB
    return number


def baseline_for(host: str) -> float:
    """Persisted idle used-memory for this node, else the config default."""
    saved = read_idle().get(str(host or ""))
    if saved is not None:
        return saved
    return configured_idle_gib()


def note_idle(host: str, used: float) -> None:
    """Remember this node's used memory while no Sparksi model container is up."""
    if not host:
        return
    try:
        number = float(used)
    except (TypeError, ValueError):
        return
    if number < 0:
        return
    data = read_idle()
    data[str(host)] = round(number, 2)
    path = idle_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        return


def _safe(model_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in str(model_id)).strip(".-") or "model"


def _candidates(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in catalog.get("library") or []:
        if not isinstance(row, dict):
            continue
        model_id = str(row.get("id") or "")
        if not valid_id(model_id):
            continue
        if row_is_service(row) or row_is_protected(row, catalog):
            continue
        if CFG.controllable_service(model_id):
            continue
        rows.append(row)
    return rows


class _Sight:
    """What one node probe showed: Sparksi containers, and host memory in use."""

    def __init__(self, names: dict[str, int], used: float | None) -> None:
        self.names = names
        self.used = used


def _used_gib(probe: dict[str, Any]) -> float | None:
    try:
        total = float(probe.get("mem_total_gib"))
        avail = float(probe.get("mem_available_gib"))
    except (TypeError, ValueError):
        return None
    return max(0.0, total - avail)


def _sight(runner: Any, host: str) -> _Sight | None:
    """Sparksi containers and used memory. ``None`` when the node could not be asked."""
    try:
        out = runner.invoke(host, {"op": "probe", "ports": [], "container_ports": True}, timeout=8)
    except Exception:  # noqa: BLE001 — one dark node does not hide the other
        return None
    if not isinstance(out, dict) or out.get("containers") is None:
        return None
    ports = out.get("container_ports") if isinstance(out.get("container_ports"), dict) else {}
    found: dict[str, int] = {}
    for name in out.get("containers") or []:
        text = str(name or "")
        if not text.startswith("sparksi-"):
            continue
        try:
            port = int(ports.get(text) or ports.get(name) or 0)
        except (TypeError, ValueError):
            port = 0
        found[text] = port
    return _Sight(found, _used_gib(out))


def _sights(runner: Any) -> dict[str, _Sight]:
    found: dict[str, _Sight] = {}
    for host in CFG.node_names:
        if not host:
            continue
        sight = _sight(runner, host)
        if sight is not None:
            found[host] = sight
    return found


def _names(sights: dict[str, _Sight]) -> dict[str, dict[str, int]]:
    return {host: sight.names for host, sight in sights.items()}


def _remember_idle(sights: dict[str, _Sight]) -> None:
    """A node with no Sparksi model container contributes its used memory as idle."""
    for host, sight in sights.items():
        if sight.names or sight.used is None:
            continue
        note_idle(host, sight.used)


def _stamp_memory(run: dict[str, Any], sights: dict[str, _Sight]) -> None:
    """Charge host used-memory above the persisted idle baseline, or the default."""
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        host = str(node.get("host") or "")
        sight = sights.get(host)
        if sight is None or sight.used is None:
            continue
        node["host_used_gib"] = sight.used
        node["mem_baseline_gib"] = baseline_for(host)
        # Idle baseline, not a pre-start sample. The host delta includes every
        # other process and must not be stored as this model's peak.
        node["baseline_kind"] = "idle"


def _explicit_names(recipe: dict[str, Any]) -> dict[str, str]:
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    names: dict[str, str] = {}
    for role in ("head", "worker"):
        body = roles.get(role) if isinstance(roles.get(role), dict) else {}
        name = str(body.get("container") or "")
        if name.startswith("sparksi-"):
            names[role] = name
    return names


def _pack(nodes: list[tuple[str, str, str, int]]) -> tuple[str, int, list[tuple[str, str, str, int]]] | None:
    rank = {role: index for index, role in enumerate(CFG.roles)}
    ordered = sorted(nodes, key=lambda item: rank.get(item[0], len(rank)))
    port = 0
    for role, _host, _name, number in ordered:
        if role == "head" and number and not protect.is_off_limits(number):
            port = number
            break
    if not port:
        for _role, _host, _name, number in ordered:
            if number and not protect.is_off_limits(number):
                port = number
                break
    if not port:
        return None
    roles = {role for role, _host, _name, _port in ordered}
    placement = placement_for(roles) or ordered[0][0]
    return placement, port, ordered


def _match(
    model_id: str,
    recipe: dict[str, Any],
    found: dict[str, dict[str, int]],
) -> tuple[str, int, list[tuple[str, str, str, int]]] | None:
    head = CFG.head.name
    worker = CFG.worker.name if CFG.has_worker else ""
    head_found = found.get(head) or {}
    worker_found = (found.get(worker) or {}) if worker else {}

    def present(host_found: dict[str, int], name: str) -> bool:
        return name in host_found

    explicit = _explicit_names(recipe)
    placement = str(recipe.get("placement") or "")
    if explicit:
        nodes: list[tuple[str, str, str, int]] = []
        for role, name in explicit.items():
            if role == "worker" and not worker:
                continue
            host = worker if role == "worker" else head
            host_found = worker_found if role == "worker" else head_found
            if present(host_found, name):
                nodes.append((role, host, name, int(host_found.get(name) or 0)))
        if placement == "both" and len(explicit) >= 2 and len(nodes) < 2:
            return None
        if not nodes:
            return None
        return _pack(nodes)

    safe = _safe(model_id)
    pair_head = f"sparksi-{safe}-head"
    pair_worker = f"sparksi-{safe}-worker"
    if present(head_found, pair_head) and present(worker_found, pair_worker):
        return _pack([
            ("head", head, pair_head, int(head_found.get(pair_head) or 0)),
            ("worker", worker, pair_worker, int(worker_found.get(pair_worker) or 0)),
        ])
    single = f"sparksi-{safe}"
    on_head = present(head_found, single)
    on_worker = present(worker_found, single)
    if on_head and on_worker:
        return _pack([
            ("head", head, single, int(head_found.get(single) or 0)),
            ("worker", worker, single, int(worker_found.get(single) or 0)),
        ])
    if on_head and placement != "worker":
        return _pack([("head", head, single, int(head_found.get(single) or 0))])
    if on_worker and placement != "head":
        return _pack([("worker", worker, single, int(worker_found.get(single) or 0))])
    if placement == "both":
        return None
    if present(head_found, pair_head) and placement != "worker":
        return _pack([("head", head, pair_head, int(head_found.get(pair_head) or 0))])
    if present(worker_found, pair_worker) and placement != "head":
        return _pack([("worker", worker, pair_worker, int(worker_found.get(pair_worker) or 0))])
    return _match_extra(placement, found, single, pair_head, pair_worker)


def _match_extra(
    placement: str,
    found: dict[str, dict[str, int]],
    single: str,
    pair_head: str,
    pair_worker: str,
) -> tuple[str, int, list[tuple[str, str, str, int]]] | None:
    """A run placed on a third or fourth Spark, or on a pair other than head+worker.

    Only reached when the head/worker checks above found nothing, so a
    two-Spark site never gets here with a match.
    """
    wanted = roles_of(placement)
    if not wanted or placement in ("head", "worker", "both"):
        return None
    nodes: list[tuple[str, str, str, int]] = []
    if len(wanted) == 1:
        host = CFG.node_by_role(wanted[0]).name
        here = found.get(host) or {}
        if single in here:
            nodes.append((wanted[0], host, single, int(here.get(single) or 0)))
        return _pack(nodes) if nodes else None
    # A rendered pair uses one container name on every Spark; older pairs used -head/-worker.
    names = {wanted[0]: (single, pair_head), wanted[1]: (single, pair_worker)} if len(wanted) == 2 else {}
    for role, choices in names.items():
        host = CFG.node_by_role(role).name
        here = found.get(host) or {}
        name = next((item for item in choices if item in here), "")
        if name:
            nodes.append((role, host, name, int(here.get(name) or 0)))
    return _pack(nodes) if len(nodes) == len(names) and nodes else None


def _active(model_id: str) -> dict[str, Any] | None:
    found = None
    for run in list_runs():
        if str(run.get("model_id") or "") != model_id or str(run.get("phase") or "") in _QUIET:
            continue
        if found is None or float(run.get("started") or 0) >= float(found.get("started") or 0):
            found = run
    return found


def containers_gone(run: dict[str, Any], runner: Any) -> bool:
    """True when every adopted container is absent from a successful probe."""
    if str(run.get("started_by") or "") != "external" or runner is None:
        return False
    from sparksi.jobs import _probe_node

    saw = False
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        name = str(node.get("container") or "")
        host = str(node.get("host") or "")
        if not name or not host:
            continue
        try:
            probe = _probe_node(runner, host, None, None)
        except Exception:  # noqa: BLE001 — an unread node is not proof the container died
            return False
        containers = probe.get("containers") if isinstance(probe, dict) else None
        if containers is None:
            return False
        saw = True
        names = {str(item) for item in containers}
        if name in names or any(item.endswith("/" + name) for item in names):
            return False
    return saw


def _live_external(runner: Any, *, watch: bool) -> None:
    """Stop an adopted run whose containers are gone, and health-check on a poll."""
    for run in list_runs():
        phase = str(run.get("phase") or "")
        if str(run.get("started_by") or "") != "external":
            continue
        if run.get("stop_requested") or phase in _QUIET or phase == "stopping":
            continue
        if not ready_check_due(run, mark=False):
            continue  # a serving run is checked at the ready pace, not on every poll
        if containers_gone(run, runner):
            _save(_stopped_copy(run))
            continue
        if watch and phase in ("starting", "waiting_health", "ready"):
            tick(run, runner)


def _engine(row: dict[str, Any], recipe: dict[str, Any]) -> str:
    if recipe.get("engine"):
        return str(recipe["engine"])
    engines = row.get("engines") if isinstance(row.get("engines"), list) else []
    if engines:
        return str(engines[0])
    return "vllm"


def _record(
    row: dict[str, Any],
    recipe: dict[str, Any],
    placement: str,
    port: int,
    nodes: list[tuple[str, str, str, int]],
) -> dict[str, Any]:
    run_id = uuid.uuid4().hex[:12]
    model_id = str(row.get("id") or "")
    stored = []
    for role, host, name, _port in nodes:
        stored.append({
            "role": role,
            "host": host,
            "run_id": f"{run_id}-{role}",
            "argv": [],
            "env": {},
            "container": name,
            "stop_argv": docker_stop(name),
            "phase": "started",
            "log_tail": "",
        })
    try:
        timeout_s = int(recipe.get("timeout_s") or 1800)
    except (TypeError, ValueError):
        timeout_s = 1800
    return {
        "id": run_id,
        "model_id": model_id,
        "engine": _engine(row, recipe),
        "placement": placement,
        "port": port,
        "phase": "waiting_health",
        "ok": None,
        "error": None,
        "failed_side": None,
        "stopped_sides": [],
        "health": {},
        "nodes": stored,
        "order": [role for role, _host, _name, _port in nodes],
        "health_paths": ["/health", "/v1/models"],
        "timeout_s": timeout_s,
        "experimental": False,
        "experimental_reason": "",
        "started": time.time(),
        "seen": {},
        "started_by": "external",
    }


def _matched_runs(catalog: dict[str, Any], sights: dict[str, _Sight], *, persist_ids: bool) -> list[dict[str, Any]]:
    """Run records for live containers that have no active run. Nothing is saved here."""
    if not sights:
        return []
    found = _names(sights)
    built: list[dict[str, Any]] = []
    for row in _candidates(catalog):
        model_id = str(row.get("id") or "")
        if _active(model_id):
            continue
        recipe = load_recipe(model_id)
        matched = _match(model_id, recipe, found)
        if matched is None:
            continue
        placement, port, nodes = matched
        run = _record(row, recipe, placement, port, nodes)
        if not persist_ids:
            run["id"] = f"view-{model_id}"
            for node in run.get("nodes") or []:
                if isinstance(node, dict):
                    node["run_id"] = f"view-{model_id}-{node.get('role')}"
        _stamp_memory(run, sights)
        # The containers are already up. Planners treat the view as running.
        if not persist_ids:
            run["phase"] = "ready"
        built.append(run)
    return built


def adoption_view(catalog: dict[str, Any], runner: Any) -> list[dict[str, Any]]:
    """In-memory runs for containers that are up. Nothing is written."""
    if runner is None or not isinstance(catalog, dict):
        return []
    return _matched_runs(catalog, _sights(runner), persist_ids=False)


def adopt_running(catalog: dict[str, Any], runner: Any, *, watch: bool = False) -> list[dict[str, Any]]:
    """Adopt live Sparksi containers that have no active run. Returns the new runs.

    Only a status poll calls this. An idle node, one with no Sparksi model
    container, stores its used memory as the baseline for later charges.
    """
    if runner is None or not isinstance(catalog, dict):
        return []
    sights = _sights(runner)
    _remember_idle(sights)
    _live_external(runner, watch=watch)
    created: list[dict[str, Any]] = []
    for run in _matched_runs(catalog, sights, persist_ids=True):
        saved = _save(run)
        created.append(tick(saved, runner))
    return created
