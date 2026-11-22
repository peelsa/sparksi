# SPDX-License-Identifier: Apache-2.0

"""HTTP-facing recipe operations. The browser sends fields, never a command string."""

from __future__ import annotations

import dataclasses
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from sparksi import protect
from sparksi import compat
from sparksi.config import ALL_ROLES, CFG
from sparksi.models import (
    KV_RESERVE_DEFAULT,
    ModelCard,
    _library_rows,
    build_cards,
    card_by_id,
    card_from_row,
    host_budget,
    probe_presence,
    running_on,
)
from sparksi.never import Refused, assert_bind_allowed
from sparksi.nodes import node_report
from sparksi.placement import all_placements, is_multi, needs_text, placement_for, roles_of, where_words
from sparksi.recipes.defaults import derive_defaults
from sparksi.recipes.memory import (
    MEMORY_MARGIN_GIB,
    SOLO_FREE_TOLERANCE_GIB,
    estimate_gib,
    node_peak_gib,
    node_peak_is_solo,
    role_need_gib,
)
from sparksi.recipes.folder import Resolved, check_sparks, copy_to_sparks, resolve
from sparksi.recipes.pins import pin_conflicts
from sparksi.recipes.pinned import FABRIC_ONLY, apply_overrides, apply_token_cache, fabric_only, folder_files, render_pinned, validate_pinned
from sparksi.recipes.render import render
from sparksi.recipes.runs import get_run, list_runs, pop_run_view, public_run, push_run_view, refresh_run, start_rendered, stop_recorded
from sparksi.recipes.store import active_recipe_name, folder_path, is_folder_recipe, load_recipe, save_recipe
from sparksi.recipes.validate import (
    ENGINES,
    PINNED_ENGINES,
    PORT_MAX,
    PORT_MIN,
    argv_backstop,
    check_port,
    form_for,
    next_free_port,
    validate_fields,
)

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,80}$")
_FORBIDDEN_NAMES = {"video-mode.sh", "dgx-announce", "dgx-announce.service", "tmux"}
# Request keys that are not recipe fields. They must not fail validation.
_CONTROL = {"id", "model_id", "sync", "from_ui", "probe", "action", "replace_run_id"}
# Present-but-empty values clear a saved setting instead of falling through.
_CLEARABLE = {
    "tool_parser", "reasoning_parser", "image", "kv_cache_memory_bytes", "extra_flags",
    "speculative_method", "speculative_model", "speculative_tokens",
    "extra_args", "extra_args_worker", "env", "env_head", "env_worker",
}
_ALWAYS = _CLEARABLE
# Page loads must not open a fresh SSH session every time.
FACTS_TTL_S = 15.0
_FACTS_LOCK = threading.Lock()
_FACTS_LOAD = threading.Lock()
_FACTS: dict[str, Any] = {"mono": 0.0, "key": None, "listening": None, "running": None, "found": None, "down": None}


def valid_id(model_id: str) -> bool:
    return bool(_ID.match(model_id or ""))


def reset_facts_cache() -> None:
    """Drop the short presence and listening cache. Tests call this with the node cache."""
    with _FACTS_LOCK:
        _FACTS["mono"] = 0.0
        _FACTS["key"] = None
        _FACTS["listening"] = None
        _FACTS["running"] = None
        _FACTS["found"] = None
        _FACTS["down"] = None


def _report(catalog: dict[str, Any], runner: Any) -> dict[str, Any]:
    try:
        return node_report(catalog, runner, force=True)
    except Exception:  # noqa: BLE001 — a dark fleet is an empty report, not a 500
        return {"nodes": []}


def _tailnet_address(address: str) -> bool:
    """True for a tailnet (100.64.0.0/10) listen address. Loopback and wildcards are not."""
    import ipaddress

    text = str(address or "").strip().strip("[]")
    if text in ("", "*", "0.0.0.0", "::"):
        return False
    try:
        return ipaddress.ip_address(text) in ipaddress.ip_network("100.64.0.0/10")
    except ValueError:
        return False


def _local_ports(node: dict[str, Any]) -> set[int]:
    """Ports bound on loopback or a wildcard. A tailnet-only proxy does not count."""
    listeners = node.get("listeners")
    if not isinstance(listeners, list):
        found: set[int] = set()
        for port in node.get("ports") or []:
            try:
                found.add(int(port))
            except (TypeError, ValueError):
                continue
        return found
    by_port: dict[int, list[str]] = {}
    for item in listeners:
        if not isinstance(item, dict):
            continue
        try:
            port = int(item.get("port"))
        except (TypeError, ValueError):
            continue
        by_port.setdefault(port, []).append(str(item.get("address") or ""))
    return {port for port, addresses in by_port.items() if any(not _tailnet_address(addr) for addr in addresses)}


def _identity_path(row: dict[str, Any]) -> str:
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    custom = ident.get("http") or row.get("probe")
    if not custom:
        return ""
    text = str(custom)
    return text if text.startswith("/") else "/" + text


def _probe_paths(rows: list[dict[str, Any]]) -> dict[str, str]:
    """One identity path per port. A shared port keeps a path only when every row agrees."""
    grouped: dict[str, list[str]] = {}
    for row in rows:
        try:
            port = str(int(row.get("port") or 0))
        except (TypeError, ValueError):
            continue
        if port == "0":
            continue
        path = _identity_path(row)
        if path:
            grouped.setdefault(port, []).append(path)
    out: dict[str, str] = {}
    for port, paths in grouped.items():
        if paths and all(path == paths[0] for path in paths):
            out[port] = paths[0]
    return out


def _rows_on(node: dict[str, Any], catalog: dict[str, Any]) -> list[dict[str, Any]]:
    from sparksi.models import _row_hosts

    keys = {str(node.get("name") or ""), str(node.get("id") or ""), str(node.get("role") or "")}
    keys.discard("")
    found: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _library_rows(catalog):
        row_id = str(row.get("id") or "")
        if not row_id or row_id in seen:
            continue
        if not keys.intersection(_row_hosts(row)):
            continue
        try:
            int(row.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if not int(row.get("port") or 0):
            continue
        seen.add(row_id)
        found.append(row)
    return found


def _probe_rows(runner: Any, host: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Read-only agent probe. Loopback open plus the row identity path. Never starts anything."""
    invoke = getattr(runner, "invoke", None)
    if invoke is None or not rows or not host:
        return {}
    ports: list[int] = []
    for row in rows:
        try:
            port = int(row.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if port and port not in ports:
            ports.append(port)
    if not ports:
        return {}
    try:
        result = invoke(host, {
            "op": "probe",
            "ports": ports,
            "paths": [],
            "http_paths": _probe_paths(rows),
        }, 20)
    except Exception:  # noqa: BLE001 — a dark node confirms nothing
        return {}
    return result if isinstance(result, dict) else {}


def _confirmed_ports(rows: list[dict[str, Any]], probe: dict[str, Any]) -> set[int]:
    """Ports whose catalog identity answered. A bound socket alone is not enough."""
    from sparksi.library import identity_check

    found: set[int] = set()
    for row in rows:
        ok, _holder = identity_check(row, probe)
        if not ok:
            continue
        try:
            found.add(int(row.get("port")))
        except (TypeError, ValueError):
            continue
    return found


def _stamp(node: dict[str, Any], ports: set[int], into: dict[str, set[int]]) -> None:
    for key in (node.get("name"), node.get("id"), node.get("role")):
        if key:
            into[str(key)] = set(ports)


def _listening_maps(
    catalog: dict[str, Any],
    runner: Any,
) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """``(bound, running)``.

    ``bound`` is a loopback or wildcard listen, so a free-port check still sees
    a taken port. ``running`` is the subset whose identity answered on
    127.0.0.1. A tailnet-only proxy is in neither: it is not the service.
    """
    report = _report(catalog, runner)
    bound: dict[str, set[int]] = {}
    running: dict[str, set[int]] = {}
    for node in report.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        host = str(node.get("name") or node.get("id") or "")
        rows = _rows_on(node, catalog)
        _stamp(node, _local_ports(node), bound)
        _stamp(node, _confirmed_ports(rows, _probe_rows(runner, host, rows)), running)
    return bound, running


def _listening_map(catalog: dict[str, Any], runner: Any) -> dict[str, set[int]]:
    """Identity-confirmed ports. A tailnet proxy or a dead loopback backend is absent."""
    _bound, running = _listening_maps(catalog, runner)
    return running


def _weight_files(catalog: dict[str, Any]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for row in _library_rows(catalog):
        for host, path in card_from_row(row).files.items():
            if not path:
                continue
            bucket = grouped.setdefault(str(host), [])
            if path not in bucket:
                bucket.append(path)
    return grouped


def _copy_facts() -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, dict[str, bool | None]], set[str]]:
    listening = {key: set(ports) for key, ports in (_FACTS["listening"] or {}).items()}
    running = {key: set(ports) for key, ports in (_FACTS["running"] or {}).items()}
    found = {host: dict(paths) for host, paths in (_FACTS["found"] or {}).items()}
    down = set(_FACTS["down"] or ())
    return listening, running, found, down


def _load_facts(
    catalog: dict[str, Any],
    runner: Any,
    grouped: dict[str, list[str]],
) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, dict[str, bool | None]], set[str]]:
    """Bound ports, confirmed rows, and weight presence. Each side probes together."""

    def listen() -> tuple[dict[str, set[int]], dict[str, set[int]]]:
        try:
            return _listening_maps(catalog, runner)
        except Exception:  # noqa: BLE001
            return {}, {}

    def presence() -> tuple[dict[str, dict[str, bool | None]], set[str]]:
        try:
            return probe_presence(runner, grouped)
        except Exception:  # noqa: BLE001 — a dark node is a reason, not a crash
            return {}, set(grouped)

    with ThreadPoolExecutor(max_workers=2) as pool:
        left = pool.submit(listen)
        right = pool.submit(presence)
        found, down = right.result()
        bound, running = left.result()
        return bound, running, found, down


def _facts(
    catalog: dict[str, Any],
    runner: Any,
) -> tuple[dict[str, set[int]], dict[str, set[int]], dict[str, dict[str, bool | None]], set[str]]:
    """Presence, bound ports, and confirmed rows, reused for about 15 seconds."""
    grouped = _weight_files(catalog)
    key = tuple((host, tuple(paths)) for host, paths in sorted(grouped.items()))
    now = time.monotonic()
    with _FACTS_LOCK:
        if (
            _FACTS["found"] is not None
            and _FACTS["key"] == key
            and now - float(_FACTS["mono"]) < FACTS_TTL_S
        ):
            return _copy_facts()
    with _FACTS_LOAD:
        now = time.monotonic()
        with _FACTS_LOCK:
            if (
                _FACTS["found"] is not None
                and _FACTS["key"] == key
                and now - float(_FACTS["mono"]) < FACTS_TTL_S
            ):
                return _copy_facts()
        bound, running, found, down = _load_facts(catalog, runner, grouped)
        with _FACTS_LOCK:
            _FACTS["mono"] = time.monotonic()
            _FACTS["key"] = key
            _FACTS["listening"] = bound
            _FACTS["running"] = running
            _FACTS["found"] = found
            _FACTS["down"] = down
            return _copy_facts()


def _model_view(
    catalog: dict[str, Any],
    runner: Any,
) -> tuple[list[ModelCard], dict[str, set[int]], dict[str, set[int]]]:
    if runner is None:
        return build_cards(catalog, None, probe=False), {}, {}
    bound, running, found, down = _facts(catalog, runner)
    cards = build_cards(
        catalog,
        runner,
        listening=bound,
        running=running,
        probe=False,
        probed=found,
        down=down,
    )
    return cards, bound, running


def _hosts_for(placement: str) -> list[str]:
    return [CFG.node_by_role(role).name for role in (roles_of(placement) or ("head",))]


def _own_ports(model_id: str) -> set[int]:
    """Ports of this model's own active run. They are not a conflict with itself."""
    found: set[int] = set()
    for run in list_runs():
        if str(run.get("model_id") or "") != str(model_id or ""):
            continue
        if run.get("phase") in _QUIET_RUNS:
            continue
        try:
            port = int(run.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if port:
            found.add(port)
    return found


def stopped_replace(model_id: str, replace_run_id: str) -> dict[str, Any] | None:
    """The stopped run a Restart named, or None when this start is not that replacement.

    Credit and the port exception apply only when the caller names this run.
    A later start of the same model does not inherit them.
    """
    text = str(replace_run_id or "").strip()
    if not text:
        return None
    run = get_run(text)
    if not run or str(run.get("model_id") or "") != str(model_id or ""):
        return None
    if not run.get("stop_requested") or str(run.get("phase") or "") != "stopped":
        return None
    return run


def _free_at_stop(run: dict[str, Any], role: str) -> float | None:
    for node in _role_nodes(run, role):
        found = _as_gib(node.get("free_at_stop_gib"))
        if found is not None:
            return found
    return _as_gib(run.get("free_at_stop_gib"))


def _started_after(run: dict[str, Any], when: float) -> bool:
    try:
        started = float(run.get("started") or 0)
    except (TypeError, ValueError):
        return False
    return started + 1e-9 >= when


def release_credit(
    released: dict[str, Any] | None,
    role: str,
    free_now: float | None,
    catalog: dict[str, Any] | None = None,
) -> float:
    """GiB of the stopped run still missing from free memory.

    ``charge - (free_now - free_at_stop)``, never above the charge and never
    below zero. Nothing is credited once another run has started on this role
    after the stop, or when the stop did not record free memory.
    """
    if released is None or free_now is None or role not in _run_roles(released):
        return 0.0
    try:
        stopped_at = float(released.get("stopped_at") or 0)
    except (TypeError, ValueError):
        stopped_at = 0.0
    replaced_id = str(released.get("id") or "")
    for other in list_runs():
        if str(other.get("id") or "") == replaced_id:
            continue
        if role not in _run_roles(other):
            continue
        if _started_after(other, stopped_at):
            return 0.0
    charge = _role_charge(released, role, catalog)
    free_stop = _free_at_stop(released, role)
    if free_stop is None or charge <= 0:
        return 0.0
    returned = float(free_now) - float(free_stop)
    return max(0.0, min(charge, charge - returned))


def container_holds_port(runner: Any, run: dict[str, Any], port: int) -> set[str]:
    """Host keys whose listener for ``port`` is still this run's container."""
    if runner is None or not port:
        return set()
    held: set[str] = set()
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        name = str(node.get("container") or "")
        host = str(node.get("host") or "")
        role = str(node.get("role") or "")
        if not name or not host:
            continue
        try:
            probe = runner.invoke(host, {"op": "probe", "ports": [], "container_ports": True}, timeout=8)
        except Exception:  # noqa: BLE001 — an unread node is not this container
            continue
        if not isinstance(probe, dict):
            continue
        names = {str(item) for item in (probe.get("containers") or [])}
        present = name in names or any(item.endswith("/" + name) for item in names)
        if not present:
            continue
        ports = probe.get("container_ports") if isinstance(probe.get("container_ports"), dict) else {}
        try:
            number = int(ports.get(name) or 0)
        except (TypeError, ValueError):
            number = 0
        if number != int(port):
            continue
        keys = {host, role}
        try:
            node_obj = CFG.node_by_role(role if role in CFG.roles else "head")
            if role in CFG.roles:
                keys.update({node_obj.name, node_obj.id, role})
        except Exception:  # noqa: BLE001
            pass
        held.update(key for key in keys if key)
    return held


def _without_replaced_port(
    listening: dict[str, set[int]],
    released: dict[str, Any] | None,
    runner: Any,
) -> dict[str, set[int]]:
    """Drop the replaced run's port only where its own container is still listening."""
    if released is None:
        return listening
    try:
        port = int(released.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    hosts = container_holds_port(runner, released, port)
    if not port or not hosts:
        return listening
    hidden = {str(key): set(ports) for key, ports in (listening or {}).items()}
    for key in hosts:
        if key in hidden:
            hidden[key].discard(port)
    return hidden


def _hide_own_ports(listening: dict[str, set[int]], own: set[int]) -> dict[str, set[int]]:
    if not own:
        return listening
    hidden: dict[str, set[int]] = {}
    for host, ports in (listening or {}).items():
        hidden[str(host)] = set(ports) - own
    return hidden


def _probe_ports(running: dict[str, set[int]], node: Any, role: str, skip_id: str) -> set[int]:
    ports = set(running.get(node.name) or running.get(node.id) or running.get(role) or ())
    return ports - _own_ports(skip_id)


def _budgets(catalog: dict[str, Any], running: dict[str, set[int]], skip_id: str) -> dict[str, float]:
    """Budget after other rows and other live runs. This model's own run is not a charge."""
    live_ports, live_gib = _live_usage(skip_id, catalog)
    budgets = {}
    for role, node in CFG.role_nodes():
        ports = _probe_ports(running, node, role, skip_id)
        ports -= live_ports.get(role) or set()
        held = running_on(catalog, node.name, ports, skip_id=skip_id)
        held += float(live_gib.get(role) or 0.0)
        budgets[role] = host_budget(catalog, node.name, held)
    return budgets


def _image_state(runner: Any, host: str, image: str) -> bool | None:
    """True or False from ``docker image inspect``; None when SSH itself gave no answer."""
    capture = getattr(runner, "capture", None)
    if capture is None:
        return False
    try:
        result = capture(host, ["docker", "image", "inspect", image], 8)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(result, dict):
        return None
    if result.get("code") == 255:
        return None
    return bool(result.get("ok") or result.get("code") == 0)


def _never(catalog: dict[str, Any], model_id: str, argv_lists: list[list[str]]) -> list[str]:
    touch = catalog.get("never_touch") or {}
    banned = {str(item) for item in (touch.get("do_not_stop") or [])}
    conflicts = []
    if model_id in banned:
        conflicts.append(f"refusing {model_id}")
    names = {str(path).rsplit("/", 1)[-1] for path in (touch.get("do_not_invoke") or [])}
    names.update(_FORBIDDEN_NAMES)
    for argv in argv_lists:
        for part in argv:
            base = str(part).rsplit("/", 1)[-1]
            if base in names or "dgx-announce" in str(part):
                conflicts.append("refusing forbidden command")
                return conflicts
    return conflicts


def _bind_conflicts(placement: str) -> list[str]:
    roles = list(roles_of(placement) or ("head",))
    conflicts = []
    hosts = []
    for role in roles:
        bind = CFG.node_by_role(role).tailscale
        if not bind:
            conflicts.append(f"no tailnet address for the {role}")
            continue
        hosts.append(bind)
    if hosts:
        try:
            assert_bind_allowed(hosts)
        except Refused as exc:
            conflicts.append(str(exc))
    return conflicts


def _fabric_conflicts(placement: str) -> list[str]:
    roles = roles_of(placement)
    if len(roles) < 2:
        return []
    conflicts = []
    for role in roles:
        fab = CFG.node_by_role(role).fabric
        if not fab.ip:
            conflicts.append(f"no fabric address for the {role}")
        if not fab.interface:
            conflicts.append(f"no fabric interface for the {role}")
        if not fab.hcas:
            conflicts.append(f"no RoCE HCAs for the {role}")
    return conflicts


def _reserved_ports(catalog: dict[str, Any]) -> set[int]:
    found = set(protect.off_limits_ports())
    host = catalog.get("host") or {}
    blobs = list(host.get("reserved_ports") or [])
    blobs.extend(CFG.controller.get("reserved_ports") or [])
    for item in blobs:
        try:
            found.add(int(item))
        except (TypeError, ValueError):
            continue
    return found


def _occupied(listening: dict[str, set[int]]) -> set[int]:
    found: set[int] = set()
    for ports in listening.values():
        found.update(int(port) for port in (ports or ()))
    return found


_QUIET_RUNS = frozenset({"stopped", "failed", "timed_out", "cancelled"})


def _as_gib(value: Any) -> float | None:
    """A GiB reading, including zero. Missing or negative values are absent."""
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < 0:
        return None
    return number


def _run_roles(run: dict[str, Any]) -> tuple[str, ...]:
    return roles_of(str(run.get("placement") or "head")) or ("head",)


def _role_nodes(run: dict[str, Any], role: str) -> list[dict[str, Any]]:
    found = []
    for node in run.get("nodes") or []:
        if isinstance(node, dict) and str(node.get("role") or "") == role:
            found.append(node)
    return found


def _recipe_estimate(run: dict[str, Any], catalog: dict[str, Any] | None) -> float:
    """Estimate for one node of a live run: planned reservation, or weights plus reserve."""
    planned = 0.0
    if run.get("reservation_gib") not in (None, ""):
        try:
            planned = max(0.0, float(run["reservation_gib"]) - MEMORY_MARGIN_GIB)
        except (TypeError, ValueError):
            planned = 0.0
    weights = 0.0
    row = _library_row(catalog or {}, str(run.get("model_id") or ""))
    if row:
        for key in ("gib", "weights_gib", "estimate_gib"):
            if row.get(key) not in (None, ""):
                try:
                    weights = float(row[key])
                except (TypeError, ValueError):
                    weights = 0.0
                break
    placement = str(run.get("placement") or "head")
    share = weights / float(max(1, len(roles_of(placement)))) if weights else 0.0
    base = (share + float(KV_RESERVE_DEFAULT)) if weights else 0.0
    return max(planned, base)


def _live_container(run: dict[str, Any], role: str) -> float | None:
    """Latest container reading. Per-node memory, else the run's stored sample."""
    for node in _role_nodes(run, role):
        found = _as_gib(node.get("container_gib"))
        if found:
            return found
    by_role = run.get("measured_by_role") if isinstance(run.get("measured_by_role"), dict) else {}
    if role in by_role:
        return _as_gib(by_role.get(role))
    if not by_role:
        return _as_gib(run.get("measured_gib"))
    return None


def _host_delta(run: dict[str, Any], role: str) -> float | None:
    """Host used-memory above this run's idle baseline, when both were recorded."""
    found: list[float] = []
    for node in _role_nodes(run, role):
        base = _as_gib(node.get("mem_baseline_gib"))
        used = _as_gib(node.get("host_used_gib"))
        if base is None or used is None:
            continue
        found.append(max(0.0, used - base))
    if not found:
        return None
    return max(found)


def _role_charge(run: dict[str, Any], role: str, catalog: dict[str, Any] | None = None) -> float:
    """Charge for one live run on one node.

    Use the container's own memory or the host's used memory above its idle
    baseline, whichever is larger and was actually measured. Never charge less
    than the stored peak or the catalog estimate. The container reading misses
    unified memory, so the host delta is kept when it is higher.
    """
    container = _live_container(run, role) or 0.0
    delta = _host_delta(run, role)
    measured = container if delta is None else max(container, delta)
    peak = node_peak_gib(str(run.get("model_id") or ""), role) or 0.0
    estimate = _recipe_estimate(run, catalog)
    return max(0.0, measured, float(peak), estimate)


def _host_pressure(runs: list[dict[str, Any]], role: str) -> float | None:
    """Host used-memory above the idle baseline, when both numbers were recorded."""
    baselines: list[float] = []
    useds: list[float] = []
    for run in runs:
        if role not in _run_roles(run):
            continue
        for node in _role_nodes(run, role):
            base = _as_gib(node.get("mem_baseline_gib"))
            used = _as_gib(node.get("host_used_gib"))
            if base is not None:
                baselines.append(base)
            if used is not None:
                useds.append(used)
    if not baselines or not useds:
        return None
    return max(0.0, max(useds) - min(baselines))


def _live_runs(skip_id: str) -> list[dict[str, Any]]:
    found = []
    for run in list_runs():
        if str(run.get("model_id") or "") == skip_id or run.get("phase") in _QUIET_RUNS:
            continue
        found.append(run)
    return found


def _live_usage(skip_id: str, catalog: dict[str, Any] | None = None) -> tuple[dict[str, set[int]], dict[str, float]]:
    """Ports and GiB to hold back for other live runs, per node.

    Each run is charged at max(container, stored peak, recipe estimate). When
    the node recorded a pre-start baseline and a later host used-memory, the
    hold is at least that host delta, which sees unified memory the container
    counter misses.
    """
    ports: dict[str, set[int]] = {role: set() for role in CFG.roles}
    charges = {role: 0.0 for role in CFG.roles}
    runs = _live_runs(skip_id)
    for run in runs:
        try:
            port = int(run.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        for role in _run_roles(run):
            charge = _role_charge(run, role, catalog)
            if charge <= 0:
                continue
            charges[role] = charges.get(role, 0.0) + charge
            if port:
                ports.setdefault(role, set()).add(port)
    gib = {role: 0.0 for role in CFG.roles}
    for role in CFG.roles:
        pressure = _host_pressure(runs, role)
        gib[role] = charges.get(role, 0.0) if pressure is None else max(charges.get(role, 0.0), pressure)
    return ports, gib


def _role_busy(catalog: dict[str, Any], running: dict[str, set[int]], skip_id: str, role: str) -> bool:
    """True when another run or a confirmed catalog row already holds memory on this node."""
    live_ports, live_gib = _live_usage(skip_id, catalog)
    if float(live_gib.get(role) or 0.0) > 0:
        return True
    node = CFG.node_by_role(role)
    ports = _probe_ports(running, node, role, skip_id)
    ports -= live_ports.get(role) or set()
    return running_on(catalog, node.name, ports, skip_id=skip_id) > 0


def _active_run(model_id: str) -> dict[str, Any] | None:
    found = None
    for run in list_runs():
        if str(run.get("model_id") or "") != model_id or run.get("phase") in _QUIET_RUNS:
            continue
        if found is None or float(run.get("started") or 0) >= float(found.get("started") or 0):
            found = run
    return found


def _duplicate_start(card: Any) -> str:
    """Refuse a second start while this model is already loading or running."""
    run = _active_run(str(getattr(card, "id", "") or ""))
    if not run:
        return ""
    phase = str(run.get("phase") or "")
    if phase not in ("starting", "waiting_health", "ready"):
        return ""
    title = str(getattr(card, "title", "") or getattr(card, "id", "") or "this model")
    word = "running" if phase == "ready" else "loading"
    return f"{title} is already {word} — stop it first or use Restart with changes"


def _run_status(run: dict[str, Any]) -> str:
    """Plain status for the card while this model itself is the active run."""
    note = str(run.get("health_note") or "").strip()
    if note:
        return note
    phase = str(run.get("phase") or "")
    where = where_words(str(run.get("placement") or "head"))
    try:
        port = int(run.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if phase == "ready":
        word = "Running"
    elif phase == "waiting_health":
        word = "Loading"
    elif phase == "starting":
        word = "Starting"
    else:
        word = phase.replace("_", " ").capitalize() or "Running"
    if port:
        return f"{word} on {where}, port {port}."
    return f"{word} on {where}."


def _model_running(model_id: str) -> bool:
    return _active_run(model_id) is not None


def _run_label(run: dict[str, Any], by_id: dict[str, dict[str, Any]]) -> str:
    """Recipe id plus that recipe's own display name."""
    model_id = str(run.get("model_id") or "")
    row = by_id.get(model_id) or {}
    title = str(run.get("title") or row.get("title") or "")
    if model_id and title and title != model_id:
        return f"{model_id} ({title})"
    return model_id or title


def _exclusive_conflict(catalog: dict[str, Any], model_id: str, listening: dict[str, set[int]]) -> str | None:
    """Another member of this row's exclusive group is live.

    The name is the running run's recipe id and display name. Members that
    share a port must not lend each other a title.
    """
    row = _library_row(catalog, model_id)
    group = row.get("exclusive")
    if not group:
        return None
    members = [str(mid) for mid in ((catalog.get("exclusive_groups") or {}).get(group) or [])]
    member_ids = set(members)
    by_id = {str(item.get("id")): item for item in (catalog.get("library") or []) if isinstance(item, dict)}
    runs = _live_runs(model_id)
    for run in runs:
        rid = str(run.get("model_id") or "")
        if rid and rid in member_ids:
            return f"{_run_label(run, by_id)} is already running"
    for other_id in members:
        if other_id == model_id:
            continue
        other = by_id.get(other_id) or {}
        try:
            port = int(other.get("port") or 0)
        except (TypeError, ValueError):
            port = 0
        if not port:
            continue
        host = str(other.get("listen_host") or CFG.head.name)
        keys = {host, CFG.role_for(host)}
        for node in CFG.nodes:
            if host in (node.name, node.id):
                keys.update((node.name, node.id))
        listening_here = any(port in (listening.get(key) or ()) for key in keys)
        if not listening_here:
            continue
        for run in runs:
            try:
                run_port = int(run.get("port") or 0)
            except (TypeError, ValueError):
                run_port = 0
            if run_port == port:
                return f"{_run_label(run, by_id)} is already running"
        title = str(other.get("title") or "")
        label = f"{other_id} ({title})" if title and title != other_id else other_id
        return f"{label} is already listening on :{port}"
    return None


def _plan_util(requested: dict[str, Any], cleaned: dict[str, Any] | None, pinned_recipe: dict[str, Any] | None) -> float | None:
    for source in (cleaned or {}, requested, pinned_recipe or {}):
        if not isinstance(source, dict) or source.get("gpu_mem_util") in (None, ""):
            continue
        try:
            return float(source["gpu_mem_util"])
        except (TypeError, ValueError):
            continue
    if isinstance(pinned_recipe, dict):
        from sparksi.recipes.args import parse_args

        roles = pinned_recipe.get("roles") if isinstance(pinned_recipe.get("roles"), dict) else {}
        head = roles.get("head") if isinstance(roles.get("head"), dict) else {}
        if not head and roles:
            sample = next(iter(roles.values()))
            head = sample if isinstance(sample, dict) else {}
        known, _extra = parse_args([str(part) for part in (head.get("args") or [])])
        if known.get("gpu_mem_util") not in (None, ""):
            try:
                return float(known["gpu_mem_util"])
            except (TypeError, ValueError):
                return None
    return None


def _gib_words(value: float) -> str:
    text = f"{float(value):.1f}"
    if text.endswith(".0"):
        return text[:-2]
    return text


def _live_memory_gate(role: str, free: float, need: float) -> str:
    """Refuse when live free memory cannot cover the model plus the safety margin.

    The sentence keeps ``free right now`` so a caller that matches that
    substring still sees a memory refusal. The need and the margin are both named.
    """
    if free + 1e-9 >= need + MEMORY_MARGIN_GIB:
        return ""
    return (
        f"the {role} has only {_gib_words(free)} GiB free right now "
        f"and this model needs {_gib_words(need)} GiB plus {_gib_words(MEMORY_MARGIN_GIB)} GiB margin"
    )


def _live_free_gib(runner: Any, role: str) -> float | None:
    """Host MemAvailable in GiB. Eviction has no sized credit to add."""
    if runner is None:
        return None
    host = CFG.node_by_role(role).name
    try:
        probe = runner.invoke(host, {"op": "probe", "ports": [], "paths": []}, timeout=8)
    except Exception:  # noqa: BLE001 — a missed probe is not free memory
        return None
    if not isinstance(probe, dict):
        return None
    return _as_gib(probe.get("mem_available_gib"))


def _memory_conflicts(
    card: ModelCard,
    engine: str,
    placement: str,
    budgets: dict[str, float],
    util: float | None,
    measured: float | None,
    umas: dict[str, float],
    *,
    catalog: dict[str, Any] | None = None,
    running: dict[str, set[int]] | None = None,
    runner: Any = None,
    released: dict[str, Any] | None = None,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Conflicts, notes, and the GiB admitted on each role.

    A solo peak that fits the budget is admitted with the margin capped.
    A solo peak above the budget is still admitted on an idle node when live
    free memory covers that peak, within a small tolerance. A shared node,
    and a recipe with no solo peak, keep the budget and the full margin.
    """
    conflicts: list[str] = []
    notes: list[str] = []
    needs: dict[str, float] = {}
    roles = roles_of(placement)
    catalog = catalog or {}
    running = running or {}
    for role in roles:
        uma = float(umas.get(role) or CFG.node_by_role(role).uma_gib or 121)
        measured_role = node_peak_gib(card.id, role)
        if measured_role is None:
            measured_role = measured
        if measured_role is not None and measured_role <= 0:
            measured_role = None
        estimate = estimate_gib(
            engine=engine,
            gpu_mem_util=util,
            uma_gib=uma,
            weights_gib=card.weights_gib,
            reserve_gib=card.kv_reserve_gib,
            placement=placement,
        )
        full = role_need_gib(
            engine=engine,
            gpu_mem_util=util,
            uma_gib=uma,
            weights_gib=card.weights_gib,
            reserve_gib=card.kv_reserve_gib,
            measured_gib=measured_role,
            placement=placement,
        )
        free = _live_free_gib(runner, role)
        if free is not None and released is not None:
            free += release_credit(released, role, free, catalog)
        blocked = _live_memory_gate(role, free, estimate) if free is not None else ""
        if blocked:
            conflicts.append(blocked)
            needs[role] = full
            break
        budget = float(budgets.get(role, 0.0))
        busy = _role_busy(catalog, running, card.id, role)
        peak_drives = measured_role is not None and measured_role + 1e-9 >= estimate
        solo = bool(measured_role is not None and node_peak_is_solo(card.id, role))
        if solo and peak_drives and not busy and float(measured_role) > budget + 1e-9:
            free = _live_free_gib(runner, role)
            if free is not None and free + SOLO_FREE_TOLERANCE_GIB + 1e-9 >= float(measured_role):
                needs[role] = float(measured_role)
                notes.append(f"{role} admitted on measured solo peak")
                continue
        if peak_drives and not busy and measured_role is not None and float(measured_role) <= budget + 1e-9:
            need = min(full, budget)
            notes.append(
                f"{role} solo peak {measured_role:.1f} GiB fits the {budget:.1f} GiB budget; margin capped"
            )
        else:
            need = full
        needs[role] = need
        if need > budget + 1e-9:
            conflicts.append(f"over budget: {role} needs {need:.1f} GiB and has {budget:.1f} GiB")
            break
    return conflicts, notes, needs


def _placement_roles(placement: str) -> set[str]:
    if placement == "pair":
        return {"head", "worker"}
    return set(roles_of(placement) or ("head",))


def _placements_overlap(left: str, right: str) -> bool:
    return bool(_placement_roles(left) & _placement_roles(right))


def _run_title(run: dict[str, Any], catalog: dict[str, Any] | None) -> str:
    model_id = str(run.get("model_id") or "")
    row = _library_row(catalog or {}, model_id)
    title = str(row.get("title") or run.get("title") or model_id or "a recipe")
    return title


def _live_port_holders(model_id: str, placement: str, catalog: dict[str, Any] | None = None) -> dict[int, str]:
    """Active runs that hold a port on an overlapping Spark. Saved recipes do not."""
    found: dict[int, str] = {}
    for run in list_runs():
        if str(run.get("model_id") or "") == model_id or run.get("phase") in _QUIET_RUNS:
            continue
        if not _placements_overlap(str(run.get("placement") or "head"), placement):
            continue
        try:
            port = int(run.get("port") or 0)
        except (TypeError, ValueError):
            continue
        if not port or port in found:
            continue
        found[port] = f"port {port} is in use by {_run_title(run, catalog)} (running)"
    return found


def _claimed_ports(model_id: str) -> set[int]:
    """Ports held by an active run. Another saved recipe does not claim a port."""
    return set(_live_port_holders(model_id, placement_for(CFG.roles)))


def _service_host(catalog: dict[str, Any], card: ModelCard) -> str:
    row = _library_row(catalog, card.id)
    host = str(row.get("listen_host") or "")
    if host:
        return host
    if card.files:
        return next(iter(card.files))
    return CFG.head.name


def _service_running(catalog: dict[str, Any], card: ModelCard, running: dict[str, set[int]]) -> bool:
    """Same confirmed set the fit check charges. A tailnet proxy is not up."""
    try:
        port = int(card.catalog_port or 0)
    except (TypeError, ValueError):
        return False
    if not port:
        return False
    host = _service_host(catalog, card)
    keys = {host, CFG.role_for(host)}
    for node in CFG.nodes:
        if host in (node.name, node.id):
            keys.add(node.name)
            keys.add(node.id)
    return any(port in (running.get(key) or ()) for key in keys if key)


def list_model_cards(catalog: dict[str, Any], runner: Any) -> dict[str, Any]:
    cards, _bound, running = _model_view(catalog, runner)
    reserved = sorted(
        port for port in _reserved_ports(catalog) if PORT_MIN <= port <= PORT_MAX or protect.is_off_limits(port)
    )
    models = []
    services = []
    for card in cards:
        if card.service:
            body = card.to_service()
            opt = CFG.controllable_service(card.id)
            if opt:
                live = _controllable_http_up(runner)
                body["controllable"] = True
                body["running"] = live
                body["status"] = "Running" if live else "Off"
            else:
                live = _service_running(catalog, card, running)
                body["running"] = live
                body["status"] = "up" if live else "down"
            body["host"] = _service_host(catalog, card)
            services.append(body)
        else:
            models.append(card.to_dict())
    return {
        "models": models,
        "services": services,
        "reserved_ports": reserved,
        "port_min": PORT_MIN,
        "port_max": PORT_MAX,
    }


def _fold_speculative(fields: dict[str, Any]) -> dict[str, Any]:
    if "speculative" in fields and "speculative_method" not in fields:
        return fields
    if not any(key in fields for key in ("speculative_method", "speculative_model", "speculative_tokens")):
        return fields
    folded = dict(fields)
    tokens = folded.get("speculative_tokens") or 0
    try:
        tokens = int(tokens or 0)
    except (TypeError, ValueError):
        tokens = 0
    folded["speculative"] = {
        "method": str(folded.get("speculative_method") or ""),
        "model": str(folded.get("speculative_model") or ""),
        "tokens": tokens,
    }
    return folded


_SPECULATIVE_RAW = frozenset({"speculative_method", "speculative_model", "speculative_tokens"})
_RECIPE_LEVEL = frozenset({"evict_page_cache", "repo", "revision"})


def _engine_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """The fields the engine allowlist judges.

    Control keys, the raw speculative form, and recipe-level keys
    (stored or rendered outside the engine argv) are not engine fields.
    An empty speculative form means off, so it is not sent either.
    """
    folded = _fold_speculative(dict(fields))
    spec = folded.get("speculative")
    if isinstance(spec, dict) and not spec.get("method"):
        folded.pop("speculative", None)
    return {k: v for k, v in folded.items() if k not in _CONTROL and k not in _SPECULATIVE_RAW and k not in _RECIPE_LEVEL}


def _leaving_pinned(saved: dict[str, Any], explicit: set[str], requested: dict[str, Any]) -> bool:
    if saved.get("kind") != "pinned":
        return False
    if "kind" in explicit and str(requested.get("kind") or "") == "generated":
        return True
    if "engine" in explicit and str(requested.get("engine") or "") not in ("", str(saved.get("engine") or "")):
        return True
    return False


def _merge_saved(
    model_id: str,
    query: dict[str, Any],
    body: dict[str, Any] | None = None,
    card: ModelCard | None = None,
) -> dict[str, Any]:
    saved = load_recipe(model_id)
    merged: dict[str, Any] = dict(card.catalog_fields) if card is not None else {}
    # A saved recipe, pinned or generated, wins over the catalog row.
    merged.update(saved)
    explicit = _explicit_keys(query, body)
    overlay = _fold_speculative(dict(query or {}))
    if body:
        overlay.update(_fold_speculative(dict(body)))
    if _leaving_pinned(saved, explicit, overlay):
        merged.pop("kind", None)
        merged.pop("roles", None)
        merged.pop("imported_from", None)
        merged.pop("port_note", None)
        merged.pop("rendezvous_ports", None)
    for key, value in overlay.items():
        if key in _ALWAYS or value not in (None, ""):
            merged[key] = value
    if "speculative_method" in overlay or "speculative" in overlay:
        merged["speculative"] = overlay.get("speculative", merged.get("speculative"))
    return merged


def _explicit_keys(query: dict[str, Any] | None, body: dict[str, Any] | None) -> set[str]:
    keys: set[str] = set()
    for key, value in (query or {}).items():
        if key in _ALWAYS or value not in (None, ""):
            keys.add(key)
    if body:
        keys.update(str(key) for key, value in body.items() if value not in (None, "") or key in _CLEARABLE)
    for key in _CLEARABLE:
        if query and key in query:
            keys.add(key)
        if body and key in body:
            keys.add(key)
    if any(key in keys for key in ("speculative_method", "speculative_model", "speculative_tokens")):
        keys.add("speculative")
    return keys


def _library_row(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    for row in _library_rows(catalog):
        if str(row.get("id") or "") == model_id:
            return row
    return {}


def _resident_row(catalog: dict[str, Any], model_id: str) -> dict[str, Any]:
    for row in catalog.get("residents") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == model_id:
            return row
    return {}


def _umas(catalog: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for role, node in CFG.role_nodes():
        spec = ((catalog.get("hosts") or {}).get(node.name) or {})
        try:
            out[role] = float(spec.get("uma_gib") or node.uma_gib or 121)
        except (TypeError, ValueError):
            out[role] = 121.0
    return out


def _recipe_defaults(
    catalog: dict[str, Any],
    card: ModelCard,
    engine: str,
    placement: str,
    listening: dict[str, set[int]],
) -> dict[str, Any]:
    return derive_defaults(
        card,
        engine,
        placement,
        saved=load_recipe(card.id),
        row=_library_row(catalog, card.id),
        resident=_resident_row(catalog, card.id),
        budgets=_budgets(catalog, listening, card.id),
        umas=_umas(catalog),
        image=CFG.engine_image(engine) if engine in ENGINES else "",
        allowed_images=CFG.allowed_images(),
    )


def _compose_fields(
    defaults: dict[str, Any],
    raw: dict[str, Any],
    explicit: set[str],
) -> dict[str, Any]:
    """Default values, with only the keys the request actually sent on top."""
    values = dict(defaults.get("values") or {})
    for key, value in raw.items():
        if key not in explicit or key in _CONTROL or key in ("engine", "placement", "port"):
            continue
        if key == "extra_flags" and value in ("", None):
            values[key] = []
        else:
            values[key] = value
    return values


def _placement_words() -> str:
    """The refusal for a saved placement this site cannot use, naming the ones it can."""
    site = all_placements()
    if site == ["head", "worker", "both"]:
        return "placement must be head, worker, or both"
    return "placement must be one of " + ", ".join(site)


def _cheapest(card: ModelCard) -> str:
    """One Spark before two, in node order: head, worker, node3, node4, then pairs."""
    for candidate in all_placements():
        if (card.placements.get(candidate) or {}).get("allowed"):
            return candidate
    return ""


def _choose(
    card: ModelCard,
    requested: dict[str, Any],
    listening: dict[str, set[int]],
    explicit: set[str],
    catalog: dict[str, Any],
) -> tuple[str, str, int, list[str]]:
    conflicts: list[str] = []
    pinned = str(requested.get("kind") or "") == "pinned" and isinstance(requested.get("roles"), dict)
    engine = str(requested.get("engine") or (card.engines[0] if card.engines else ""))
    if pinned:
        if engine not in PINNED_ENGINES:
            conflicts.append(f"unknown engine {engine or '(none)'}")
    elif engine not in card.engines:
        conflicts.append(f"engine {engine or '(none)'} is not offered for this model")
    site = all_placements()
    if "placement" in explicit:
        placement = str(requested.get("placement") or "")
    else:
        saved = str(requested.get("placement") or "")
        saved_ok = saved in site and (card.placements.get(saved) or {}).get("allowed")
        placement = saved if saved_ok else _cheapest(card)
    if placement not in site:
        # A placement for Sparks this site does not have (a two-Spark recipe on
        # one Spark) says so; anything else keeps the head's reason.
        missing = needs_text(placement) if placement and roles_of(placement, ALL_ROLES) else ""
        placement = "head"
        reason = missing or str((card.placements.get("head") or {}).get("reason") or "no placement can run this model")
        conflicts.append(reason)
    option = card.placements.get(placement) or {}
    if placement in site and not option.get("allowed"):
        conflicts.append(str(option.get("reason") or f"{placement} is not available"))
    if engine == "llamacpp" and is_multi(placement):
        conflicts.append("llamacpp runs on one Spark")
    hosts = _hosts_for(placement if placement in site else "head")
    reserved = _reserved_ports(catalog)
    holders = _live_port_holders(card.id, placement, catalog)
    claimed = set(holders)
    # This model's own listener is not a conflict and must not move its saved port.
    listening = _hide_own_ports(listening, _own_ports(card.id))
    scan = list(hosts)
    port_args = dict(reserved=reserved, claimed=claimed, holders=holders)

    def port_conflicts(number: int) -> list[str]:
        return check_port(number, listening, scan, **port_args)

    saved_port = 0
    if requested.get("port") not in (None, ""):
        try:
            saved_port = int(requested.get("port"))
        except (TypeError, ValueError):
            saved_port = 0
    if "port" not in explicit:
        port = 0
        if saved_port:
            # A saved recipe port stays put. Another model is a note, not a new port.
            port = saved_port
            conflicts.extend(port_conflicts(saved_port))
        else:
            if card.catalog_port and not port_conflicts(int(card.catalog_port)):
                port = int(card.catalog_port)
            if not port:
                port = next_free_port(_occupied(listening) | claimed, reserved, claimed) or 0
                if not port:
                    conflicts.append("pinned recipe needs a port" if pinned else "no free port in 8010-8099")
    else:
        try:
            port = int(requested["port"])
        except (TypeError, ValueError):
            conflicts.append("port must be an integer")
            port = 0
        conflicts.extend(port_conflicts(port))
    return engine, placement, int(port), conflicts


def plan_model(
    catalog: dict[str, Any],
    runner: Any,
    model_id: str,
    body: dict[str, Any] | None = None,
    *,
    query: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Plan a start. Running containers are visible, but this call does not save a run."""
    if not valid_id(model_id):
        return 400, {"error": "bad model id", "conflicts": ["bad model id"]}
    view: list[dict[str, Any]] = []
    if runner is not None:
        from sparksi.recipes.adopt import adoption_view

        view = adoption_view(catalog, runner)
    push_run_view(view)
    try:
        return _plan_model(catalog, runner, model_id, body, query=query)
    finally:
        pop_run_view()


def _plan_model(
    catalog: dict[str, Any],
    runner: Any,
    model_id: str,
    body: dict[str, Any] | None = None,
    *,
    query: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    memory_note = ""
    if not valid_id(model_id):
        return 400, {"error": "bad model id", "conflicts": ["bad model id"]}
    cards, listening, running = _model_view(catalog, runner)
    card = card_by_id(cards, model_id)
    if card is None:
        return 404, {"error": "unknown model", "conflicts": ["unknown model"]}
    explicit = _explicit_keys(query, body)
    requested = _merge_saved(model_id, query or {}, body, card)
    reserve, reserve_conflict = recipe_reserve_gib(requested)
    if reserve is not None:
        # The recipe's own KV reserve wins; the model card's is only the fallback.
        card = dataclasses.replace(card, kv_reserve_gib=reserve)
    replaced = stopped_replace(card.id, str((body or {}).get("replace_run_id") or ""))
    listening = _without_replaced_port(listening, replaced, runner)
    engine, placement, port, conflicts = _choose(card, requested, listening, explicit, catalog)
    pinned = str(requested.get("kind") or "") == "pinned" and isinstance(requested.get("roles"), dict)
    active = _active_run(card.id)
    duplicate = _duplicate_start(card)
    if duplicate:
        conflicts.append(duplicate)
    if reserve_conflict:
        conflicts.append(reserve_conflict)
    if pinned:
        exclusive = _exclusive_conflict(catalog, card.id, listening)
        if exclusive:
            conflicts.append(exclusive)
    defaults = _recipe_defaults(catalog, card, engine, placement, running)
    raw = dict(query or {})
    if body:
        raw.update(body)
    if pinned:
        edit_fields = dict(requested)
        edit_keys = set(explicit)
        if port:
            edit_fields["port"] = port
            edit_keys.add("port")
        pinned_recipe = apply_overrides(requested, edit_fields, edit_keys)
        pinned_recipe["port"] = port or pinned_recipe.get("port")
        # A two-node import stays both so RoCE flags are judged on that placement.
        if "placement" not in explicit and str(requested.get("placement") or "") == "both":
            pinned_recipe["placement"] = "both"
        else:
            pinned_recipe["placement"] = placement
        conflicts.extend(apply_token_cache(
            pinned_recipe,
            runner,
            repo=str(pinned_recipe.get("repo") or card.repo or ""),
            revision=str(pinned_recipe.get("revision") or card.revision or ""),
        ))
        note = str(pinned_recipe.get("token_note") or "")
        if note and isinstance(defaults, dict):
            values = defaults.setdefault("values", {})
            if isinstance(values, dict):
                values["token_note"] = note
        resolved = resolve_folder(card.id, pinned_recipe)
        conflicts.extend(resolved.conflicts)
        pinned_recipe = resolved.recipe
        files_view = folder_view(resolved)
        if files_view:
            # Read-only here: a missing copy is made at start; a copy that differs refuses.
            files_view["sparks"] = check_sparks(runner, files_view)
            for row in files_view["sparks"].values():
                if row.get("state") == "differs":
                    conflicts.append(f"{row['host']}'s copy of this recipe's files does not match the recipe folder")
        with folder_files(resolved.local):
            conflicts.extend(validate_pinned(pinned_recipe, ctx_max=card.max_ctx, runner=runner))
        conflicts.extend(pin_conflicts(runner, pinned_recipe))
        cleaned = {}
        fields = {}
    else:
        pinned_recipe = None
        files_view = None
        fields = _compose_fields(defaults, _fold_speculative(raw), explicit)
        cleaned, field_conflicts = validate_fields(
            engine if engine in ENGINES else "vllm",
            _engine_fields(fields),
            allowed_images=CFG.allowed_images(),
            forbid_prefix="prefix_cache" in (card.flags.get("forbid") or []),
            max_ctx=defaults.get("max_ctx") if defaults.get("max_ctx") is not None else card.max_ctx,
            mount_roots=CFG.mount_roots(),
        )
        conflicts.extend(field_conflicts)
    if pinned or engine in ("vllm", "sglang", "trtllm", "llamacpp", "custom"):
        conflicts.extend(_bind_conflicts(placement))
        if not pinned:
            conflicts.extend(_fabric_conflicts(placement))
        budgets = _budgets(catalog, running, card.id)
        util = _plan_util(requested, cleaned if not pinned else None, pinned_recipe if pinned else None)
        mem_conflicts, mem_notes, mem_needs = _memory_conflicts(
            card, engine, placement, budgets, util, None, _umas(catalog),
            catalog=catalog, running=running, runner=runner, released=replaced,
        )
        conflicts.extend(mem_conflicts)
        memory_note = " ".join(mem_notes)
        reservation = None
        roles = roles_of(placement)
        if roles:
            reservation = mem_needs.get(roles[0])
        for role in roles:
            host = CFG.node_by_role(role).name
            if host in card.files and card.presence.get(host) is False:
                conflicts.append(f"weights are not on the {role}")
        recipe = None
        # A pinned command names the Sparks it was written for: one role, or head+worker.
        if pinned and pinned_recipe is not None and (placement == "both" or placement in ALL_ROLES) and roles_of(placement):
            try:
                recipe = render_pinned(card, pinned_recipe, catalog=catalog)
            except ValueError as exc:
                conflicts.append(str(exc))
                recipe = None
            if recipe and files_view:
                recipe["files"] = files_view
            if recipe and runner is not None:
                for name, role in (pinned_recipe.get("roles") or {}).items():
                    if not isinstance(role, dict) or role.get("runtime") == "native":
                        continue
                    image = str(role.get("image") or "")
                    host = CFG.node_by_role(str(name)).name
                    if not image:
                        continue
                    state = _image_state(runner, host, image)
                    if state is None:
                        if not any(c.startswith("could not reach") for c in conflicts):
                            conflicts.append(f"could not check image {image} on {host}: no answer over SSH")
                    elif state is False:
                        conflicts.append(f"image {image} is missing on {host}")
        elif not pinned:
            image = str(cleaned.get("image") or ("" if engine in ("llamacpp", "custom") else CFG.engine_image(engine)))
            if engine in ("vllm", "sglang", "trtllm"):
                if not image:
                    conflicts.append("no pinned image for this engine")
                elif runner is not None:
                    for host in _hosts_for(placement):
                        state = _image_state(runner, host, image)
                        if state is None:
                            if not any(c.startswith("could not reach") for c in conflicts):
                                conflicts.append(f"could not check image {image} on {host}: no answer over SSH")
                        elif state is False:
                            conflicts.append(f"image {image} is missing on {host}")
            if engine in card.engines and roles_of(placement):
                try:
                    recipe = render(card, engine=engine, placement=placement, port=port or 8010, fields={**fields, **cleaned}, catalog=catalog)
                except ValueError as exc:
                    conflicts.append(str(exc))
                    recipe = None
        if recipe:
            if reservation is not None:
                recipe["reservation_gib"] = reservation
            if memory_note:
                recipe["memory_note"] = memory_note
            conflicts.extend(argv_backstop([list(node.get("argv") or []) for node in recipe.get("nodes") or []]))
            conflicts.extend(_never(catalog, card.id, [list(node.get("argv") or []) for node in recipe.get("nodes") or []] + [list(node.get("stop_argv") or []) for node in recipe.get("nodes") or []]))
            if recipe.get("experimental") and recipe.get("experimental_reason"):
                recipe["plan_note"] = recipe["experimental_reason"]
    else:
        recipe = None
    # Preserve order while dropping repeats.
    if active and not _live_runs(card.id):
        # Its own load is not an over-budget warning about itself.
        conflicts = [item for item in conflicts if not str(item).startswith("over budget:")]
    seen: set[str] = set()
    unique = []
    for item in conflicts:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    ok = not unique
    form = form_for(engine) if engine in ENGINES else {}
    if card.max_ctx:
        form = {**form, "ctx_max": card.max_ctx}
    payload = {
        "ok": ok,
        "verdict": "READY" if ok else "REFUSE",
        "conflicts": unique,
        "id": card.id,
        "engine": engine,
        "placement": placement,
        "port": port,
        "experimental": bool(recipe and recipe.get("experimental")),
        "experimental_reason": (recipe or {}).get("experimental_reason") or "",
        "recipe": recipe if ok else None,
        "draft": recipe,
        "kind": "pinned" if pinned else "generated",
        "imported_from": (recipe or {}).get("imported_from") or "",
        "port_note": (recipe or {}).get("port_note") or "",
        "form": form_for(engine, pinned=pinned, ctx_max=(defaults.get("max_ctx") if defaults.get("max_ctx") is not None else card.max_ctx)) if pinned or engine in ENGINES else form,
        "defaults": defaults,
        "allowed_images": CFG.allowed_images(),
        "memory_note": memory_note,
        "run_status": _run_status(active) if active else "",
        "files": files_view,
    }
    from sparksi.recipes.reasons import annotate

    payload["reasons"] = annotate(unique)
    if not ok and recipe and recipe.get("experimental"):
        payload["experimental_reason"] = recipe.get("experimental_reason") or ""
    return (200 if ok else 400), payload


def preview_recipe(catalog: dict[str, Any], runner: Any, model_id: str, query: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """A preview is a read of the plan.

    A refusal is HTTP 200 with ``refused`` true and ``reasons``. Callers that
    only look at the status were treating a normal "this would not start" as
    a broken request. Start and plan stay HTTP 400 on refusal: the CLI and
    the load button treat a non-200 as "do not start".
    """
    code, payload = plan_model(catalog, runner, model_id, query=query)
    if code == 404:
        return code, payload
    if not payload.get("ok"):
        body = dict(payload)
        body["refused"] = True
        return 200, body
    recipe = dict(payload.get("recipe") or {})
    recipe["conflicts"] = []
    recipe["verdict"] = "READY"
    recipe["form"] = payload.get("form") or form_for(str(recipe.get("engine") or ""))
    recipe["defaults"] = payload.get("defaults") or {}
    recipe["allowed_images"] = payload.get("allowed_images") or CFG.allowed_images()
    return 200, recipe


RESERVE_MAX_GIB = 128.0


def recipe_reserve_gib(recipe: dict[str, Any]) -> tuple[float | None, str]:
    """The recipe's ``[memory] kv_reserve_gib`` and a sentence when it is not usable."""
    memory = recipe.get("memory")
    if memory in (None, ""):
        return None, ""
    if not isinstance(memory, dict):
        return None, "memory must be a table"
    value = memory.get("kv_reserve_gib")
    if value in (None, ""):
        return None, ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, "memory.kv_reserve_gib must be a number of GiB"
    if not 0 <= number <= RESERVE_MAX_GIB:
        return None, f"memory.kv_reserve_gib must be between 0 and {RESERVE_MAX_GIB:g} GiB"
    return number, ""


def resolve_folder(model_id: str, recipe: dict[str, Any]) -> Resolved:
    """The recipe with its folder files mapped to Spark paths, for the recipe being read now."""
    name = active_recipe_name(model_id)
    root = folder_path(model_id, name) if is_folder_recipe(model_id, name) else None
    return resolve(recipe, model_id, name, root)


def folder_view(resolved: Resolved) -> dict[str, Any] | None:
    """What a plan says about the recipe folder: None for a recipe that uses no folder files."""
    if not resolved.version:
        return None
    return {
        "folder": resolved.folder,
        "version": resolved.version,
        "count": len(resolved.entries),
        "nodes": dict(resolved.nodes),
        "manifest": [
            {"path": entry.path, "sha256": entry.sha256, "size": entry.size, "executable": entry.executable}
            for entry in resolved.entries
        ],
    }


def write_recipe(
    model_id: str, body: dict[str, Any], runner: Any = None, *, copied_from: str = ""
) -> tuple[int, dict[str, Any]]:
    """Save one recipe. ``copied_from`` comes from Save as only; a request body never sets it."""
    if not valid_id(model_id):
        return 400, {"error": "bad model id"}
    from sparksi.catalog import load_catalog

    catalog = load_catalog()
    found = card_by_id(build_cards(catalog, None, probe=False), model_id)
    saved = load_recipe(model_id)
    body = _fold_speculative(dict(body))
    body.pop("copied_from", None)
    if str(body.get("kind") or saved.get("kind") or "") == "pinned" and not _leaving_pinned(saved, set(body), body):
        base = saved if saved.get("kind") == "pinned" else body
        if body.get("roles"):
            base = {**base, "roles": body["roles"], "kind": "pinned"}
        updated = apply_overrides(base, body, set(body))
        updated["kind"] = "pinned"
        if body.get("imported_from"):
            updated["imported_from"] = str(body["imported_from"])
        if copied_from:
            updated["copied_from"] = str(copied_from)
        if "port_note" in body:
            updated["port_note"] = str(body.get("port_note") or "")
        if body.get("rendezvous_ports") is not None:
            updated["rendezvous_ports"] = list(body.get("rendezvous_ports") or [])
        token_conflicts: list[str] = []
        if found is not None:
            token_conflicts = apply_token_cache(
                updated,
                runner,
                repo=str(updated.get("repo") or found.repo or ""),
                revision=str(updated.get("revision") or found.revision or ""),
            )
        # The saved recipe keeps @files/ and @role/ sources; they are checked as the Spark paths they become.
        resolved = resolve_folder(model_id, updated)
        with folder_files(resolved.local):
            conflicts = token_conflicts + resolved.conflicts + validate_pinned(
                resolved.recipe, ctx_max=found.max_ctx if found is not None else None, runner=runner, for_save=True,
            )
        if "port" in body:
            conflicts.extend(check_port(body.get("port"), reserved=_reserved_ports(catalog)))
        if conflicts:
            return 400, {"error": conflicts[0], "conflicts": conflicts}
        try:
            path = save_recipe(model_id, updated)
        except ValueError as exc:
            return 400, {"error": str(exc), "conflicts": [str(exc)]}
        return 200, {
            "ok": True,
            "path": str(path.name),
            "recipe": load_recipe(model_id),
            "not_loadable": FABRIC_ONLY if fabric_only(updated) else "",
        }
    engine = str(body.get("engine") or "vllm")
    cleaned, conflicts = validate_fields(
        engine,
        _engine_fields(body),
        allowed_images=CFG.allowed_images(),
        max_ctx=found.max_ctx if found is not None else None,
        mount_roots=CFG.mount_roots(),
    )
    if "port" in body:
        conflicts.extend(check_port(body.get("port"), reserved=_reserved_ports(catalog)))
    if "placement" in body and str(body.get("placement")) not in all_placements():
        conflicts.append(_placement_words())
    if conflicts:
        return 400, {"error": conflicts[0], "conflicts": conflicts}
    stored = dict(cleaned)
    stored["engine"] = engine
    for key in ("repo", "revision"):
        if key in body:
            text = str(body.get(key) or "").strip()
            if text:
                stored[key] = text
    if "placement" in body:
        stored["placement"] = str(body["placement"])
    if "evict_page_cache" in body:
        from sparksi.recipes.pinned import flag_on

        stored["evict_page_cache"] = flag_on(body.get("evict_page_cache"))
    if "port" in body:
        try:
            stored["port"] = int(body["port"])
        except (TypeError, ValueError):
            return 400, {"error": "port must be an integer", "conflicts": ["port must be an integer"]}
    try:
        path = save_recipe(model_id, stored)
    except ValueError as exc:
        return 400, {"error": str(exc), "conflicts": [str(exc)]}
    return 200, {"ok": True, "path": str(path.name), "recipe": load_recipe(model_id)}


def begin_run(catalog: dict[str, Any], runner: Any, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    model_id = str(body.get("id") or body.get("model_id") or "")
    code, payload = plan_model(catalog, runner, model_id, body)
    if not payload.get("ok"):
        return code, payload
    recipe = payload["recipe"]
    if recipe.get("files"):
        # Copy the folder to every Spark the recipe uses and check it by hash first.
        # A container is never started on folder paths that did not check out.
        why, states = copy_to_sparks(runner, recipe["files"])
        if why:
            return 409, {"ok": False, "error": why, "conflicts": [why], "files_sparks": states}
    run = start_rendered(
        recipe,
        runner,
        sync=bool(body.get("sync")),
        replace_run_id=str(body.get("replace_run_id") or ""),
    )
    return 200, {"ok": True, "run": public_run(run), "experimental": payload.get("experimental"), "experimental_reason": payload.get("experimental_reason") or ""}


def runs_index(runner: Any = None, catalog: dict[str, Any] | None = None) -> dict[str, Any]:
    if runner is not None and catalog is not None:
        from sparksi.recipes.adopt import adopt_running

        adopt_running(catalog, runner, watch=True)
    rows = []
    for run in list_runs():
        if runner is not None:
            try:
                run = refresh_run(run, runner)
            except Exception:  # noqa: BLE001 — status still returns the recorded run
                pass
        rows.append(public_run(run))
    return {"runs": rows}


def run_one(run_id: str, runner: Any = None) -> tuple[int, dict[str, Any]]:
    if not valid_id(run_id):
        return 400, {"error": "bad run id"}
    run = get_run(run_id)
    if not run:
        return 404, {"error": "unknown run"}
    if runner is not None:
        try:
            run = refresh_run(run, runner)
        except Exception:  # noqa: BLE001
            pass
    return 200, {"run": public_run(run)}


def _head_run_name() -> str:
    """Display name of an active run on the head, or empty."""
    for run in list_runs():
        if run.get("phase") in _QUIET_RUNS:
            continue
        if not _placements_overlap(str(run.get("placement") or "head"), "head"):
            continue
        return _run_title(run, {})
    return ""


def _head_free_gib(runner: Any) -> float | None:
    if runner is None:
        return None
    try:
        probe = runner.invoke(
            CFG.head.name,
            {"op": "probe", "ports": [], "paths": [], "listen_host": "127.0.0.1"},
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(probe, dict):
        return None
    return _as_gib(probe.get("mem_available_gib"))


def _controllable_http_up(runner: Any) -> bool:
    """True only when loopback /system_stats on the head looks like the service.

    A bare TCP listener is not enough: a tailnet proxy can bind the same port.
    """
    port = protect.video_port()
    if runner is None or port is None:
        return False
    try:
        probe = runner.invoke(
            CFG.head.name,
            {
                "op": "probe",
                "ports": [port],
                "paths": [],
                "listen_host": "127.0.0.1",
                "http_paths": {str(port): "/system_stats"},
            },
            timeout=8,
        )
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(probe, dict):
        return False
    http = probe.get("http") if isinstance(probe.get("http"), dict) else {}
    stat = http.get(str(port)) or http.get(port) or {}
    if not isinstance(stat, dict):
        return False
    return stat.get("status") == 200 and compat.agent_video_flag(stat)


def _run_configured_service(
    runner: Any,
    host: str,
    argv: list[str],
) -> tuple[int, dict[str, Any]]:
    if any(os.path.basename(part) == "video-mode.sh" for part in argv):
        return 400, {"error": "refusing video-mode.sh; configure a service-only command", "ok": False}
    if argv not in CFG.service_commands():
        return 400, {"error": "not in the service allowlist"}
    payload = {"op": "service", "argv": argv}
    call = getattr(runner, "agent_call", None)
    try:
        result = call(host, "service", {"argv": argv}) if call is not None else runner.invoke(host, payload, timeout=30)
    except Exception as exc:  # noqa: BLE001 — the page shows the failure
        return 400, {"error": str(exc), "ok": False}
    if not isinstance(result, dict) or not result.get("ok"):
        return 400, {"error": (result or {}).get("error") or "service command failed", "ok": False}
    return 200, {"ok": True}


def service_action(catalog: dict[str, Any], runner: Any, model_id: str, action: str) -> tuple[int, dict[str, Any]]:
    """Start or stop a catalog service. The argv must match ``service_commands`` exactly."""
    from sparksi.models import row_is_protected, row_is_service

    if action not in ("start", "stop"):
        return 400, {"error": "action must be start or stop"}
    if not valid_id(model_id):
        return 400, {"error": "bad model id"}
    row = None
    for item in catalog.get("library") or []:
        if isinstance(item, dict) and str(item.get("id") or "") == model_id:
            row = item
            break
    if row is None:
        return 404, {"error": "unknown model"}
    if not row_is_service(row):
        return 400, {"error": "not a service"}
    opt = CFG.controllable_service(model_id)
    if opt:
        argv = list(opt[action])
        if action == "start":
            busy = _head_run_name()
            if busy:
                return 400, {"error": f"{busy} is running on the head", "ok": False}
            free = _head_free_gib(runner)
            reserve = float(opt["reserve_gib"])
            if free is None:
                return 400, {"error": "could not read free memory on the head", "ok": False}
            if free + 1e-9 < reserve:
                return 400, {
                    "error": f"the head has {free:.1f} GiB free; this service needs {reserve:.0f} GiB",
                    "ok": False,
                }
        code, body = _run_configured_service(runner, CFG.head.name, argv)
        if code != 200:
            return code, body
        body["action"] = action
        body["id"] = model_id
        return code, body
    try:
        port = int(row.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if protect.is_off_limits(port) or row_is_protected(row, catalog):
        return 400, {
            "error": "port 8188 is off limits" if protect.is_off_limits(port) else "protected",
            "protected": True,
        }
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    argv = [str(part) for part in (launch.get(action) or [])]
    if not argv:
        return 400, {"error": "no command"}
    if any("dgx-announce" in part or part.rsplit("/", 1)[-1] == "video-mode.sh" for part in argv):
        return 400, {"error": "refusing forbidden command"}
    if any(
        protect.is_off_limits(part) or (":" in str(part) and protect.is_off_limits(str(part).rsplit(":", 1)[1]))
        for part in argv
    ):
        return 400, {"error": "port 8188 is off limits"}
    if argv not in CFG.service_commands():
        return 400, {"error": "not in the service allowlist"}
    host = str(launch.get("host") or CFG.head.name)
    payload = {"op": "service", "argv": argv}
    call = getattr(runner, "agent_call", None)
    try:
        result = call(host, "service", {"argv": argv}) if call is not None else runner.invoke(host, payload, timeout=30)
    except Exception as exc:  # noqa: BLE001 — the page shows the failure
        return 400, {"error": str(exc), "ok": False}
    if not isinstance(result, dict) or not result.get("ok"):
        return 400, {"error": (result or {}).get("error") or "service command failed", "ok": False}
    return 200, {"ok": True, "action": action, "id": model_id}


def halt_run(catalog: dict[str, Any], runner: Any, run_id: str) -> tuple[int, dict[str, Any]]:
    if not valid_id(run_id):
        return 400, {"error": "bad run id"}
    run = get_run(run_id)
    if not run:
        return 404, {"error": "unknown run"}
    if protect.is_off_limits(run.get("port") or 0):
        return 400, {"error": "port 8188 is off limits", "stopped": False}
    blocked = _never(catalog, str(run.get("model_id") or ""), [list(node.get("stop_argv") or []) for node in run.get("nodes") or []])
    if blocked:
        return 400, {"error": blocked[0], "stopped": False}
    updated = stop_recorded(run, runner)
    code = 200 if updated.get("phase") == "stopped" else 400
    body = {"run": public_run(updated), "stopped": updated.get("phase") == "stopped"}
    if updated.get("error"):
        body["error"] = updated["error"]
    return code, body
