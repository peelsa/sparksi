# SPDX-License-Identifier: Apache-2.0

"""Find models serving on this machine that Sparksi did not start."""

from __future__ import annotations

from sparksi import protect

ACTIVE = {"starting", "waiting_health", "copying", "ready", "stopping"}

_LABEL_MAX = 200
_LABEL_DEFAULT = "unnamed"
_NO_DOCKER = "docker could not be asked on this machine"
_NOT_OURS = "not a container Sparksi can stop; stop it on the machine"
_OURS = "a Sparksi container; stop it from its run"
_DISABLED = "stopping models started outside Sparksi is off on this controller ([controller] stop_unmanaged)"
_PROTECTED = "protected on this controller; stop it on the machine"
_SPARKSI_PORT = "served by a Sparksi container; stop it from its run"
_UNKNOWN = "docker could not be asked, so Sparksi cannot tell whether this is its own run"
# Web front ends and Sparksi's own helpers are never offered an emergency stop.
_KEEP_SUBSTRINGS = ("openwebui", "open-webui", "announce")


def _as_int(value: object) -> int | None:
    """Return value as int when possible, else None."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clean(value: object) -> str:
    """Return value as a stripped string."""
    return value.strip() if isinstance(value, str) else ""


def _label(row: dict) -> str:
    """Build a display label from a loaded row."""
    text = _clean(row.get("served")) or _clean(row.get("name"))
    if not text:
        return _LABEL_DEFAULT
    return text[:_LABEL_MAX]


def _node_matches(node_id: str, node_name: str, row: dict) -> bool:
    """Check whether a loaded row is on the given node."""
    host = _clean(row.get("host"))
    if host == node_id or host == node_name:
        return True
    span = row.get("span")
    if not isinstance(span, list):
        return False
    for entry in span:
        name = _clean(entry)
        if name == node_id or name == node_name:
            return True
    return False


def _run_hosts(run: object) -> set[str] | None:
    """Return hosts a run is on, or None for every node."""
    if not isinstance(run, dict):
        return set()
    nodes = run.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return None
    hosts: set[str] = set()
    for entry in nodes:
        if not isinstance(entry, dict):
            continue
        host = _clean(entry.get("host"))
        if host:
            hosts.add(host)
    return hosts or None


def _run_active(run: object) -> bool:
    """Check whether a run is in an active phase."""
    if not isinstance(run, dict):
        return False
    phase = run.get("phase")
    return isinstance(phase, str) and phase in ACTIVE


def _run_port(run: object) -> int | None:
    """Return the run port when int-able."""
    if not isinstance(run, dict):
        return None
    return _as_int(run.get("port"))


def find(
    loaded: list[dict],
    runs: list[dict],
    nodes: list[dict],
) -> list[dict]:
    """List (node, port) pairs serving models without an active Sparksi run."""
    active: list[tuple[set[str] | None, int]] = []
    for run in runs:
        if not _run_active(run):
            continue
        port = _run_port(run)
        if port is None:
            continue
        active.append((_run_hosts(run), port))

    items: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = _clean(node.get("id"))
        if not node_id:
            continue
        node_name = _clean(node.get("name")) or node_id
        role = _clean(node.get("role"))
        for row in loaded:
            if not isinstance(row, dict):
                continue
            port = _as_int(row.get("port"))
            if port is None or protect.is_off_limits(port):
                continue
            if not _node_matches(node_id, node_name, row):
                continue
            if (node_id, port) in seen:
                continue
            taken = False
            for hosts, run_port in active:
                if run_port != port:
                    continue
                if hosts is None or node_id in hosts or node_name in hosts:
                    taken = True
                    break
            if taken:
                continue
            seen.add((node_id, port))
            items.append(
                {
                    "host": node_id,
                    "host_name": node_name,
                    "role": role,
                    "port": port,
                    "label": _label(row),
                }
            )
    return items


def attach_containers(
    items: list[dict],
    containers_by_host: dict[str, dict[str, int]] | None,
) -> list[dict]:
    """Attach the container serving each item's port, when known."""
    out: list[dict] = []
    for item in items:
        host = item.get("host")
        port = item.get("port")
        container: str | None = None
        names = None if containers_by_host is None else containers_by_host.get(host)
        if names is None:
            why = _NO_DOCKER
        else:
            for name in sorted(names):
                if names[name] == port:
                    container = name
                    break
            why = "" if container is not None else _NOT_OURS
        out.append({**item, "container": container, "why": why})
    return out


def sparksi_containers(runs: object) -> set[str]:
    """Container names in Sparksi's run records, any phase."""
    names: set[str] = set()
    if not isinstance(runs, list):
        return names
    for run in runs:
        if not isinstance(run, dict):
            continue
        nodes = run.get("nodes")
        if not isinstance(nodes, list):
            continue
        for node in nodes:
            if not isinstance(node, dict):
                continue
            container = node.get("container")
            if isinstance(container, str):
                name = container.strip()
                if name:
                    names.add(name)
    return names


def mark_stoppable(
    items: list[dict],
    enabled: bool,
    ours: frozenset[str] | set[str] = frozenset(),
) -> list[dict]:
    """Mark which items Sparksi may stop on this controller."""
    out: list[dict] = []
    for item in items:
        container = item.get("container")
        stoppable = bool(enabled) and bool(container)
        new = {**item, "stoppable": stoppable}
        if container in ours:
            new["stoppable"] = False
            new["why"] = _OURS
        elif not enabled and container:
            new["why"] = _DISABLED
        out.append(new)
    return out


def _kept(text: object, keep: frozenset[str] | set[str]) -> bool:
    low = _clean(text).lower()
    if not low:
        return False
    return low in keep or any(s in low for s in _KEEP_SUBSTRINGS) or any(s in low for s in protect.process_substrings())


def mark_emergency(
    items: list[dict],
    ours: frozenset[str] | set[str] = frozenset(),
    keep: frozenset[str] | set[str] = frozenset(),
    keep_ports: frozenset[int] | set[int] = frozenset(),
    sparksi_ports: dict[str, set[int]] | None = None,
) -> list[dict]:
    """Mark which items may get the emergency stop: never Sparksi's own, protected ones, or ones with a Stop already.

    ``keep`` holds lower-cased names the catalog protects (never_touch.do_not_stop);
    ``keep_ports`` the ports of catalog rows the config protects; ``sparksi_ports`` the
    ports each host's running ``sparksi-*`` containers serve (a failed, timed-out or
    half-up run included). A host missing from it means docker could not be asked there,
    and then nothing on that host is offered.
    """
    sparksi_ports = sparksi_ports or {}
    out: list[dict] = []
    for item in items:
        container = item.get("container")
        new = {**item, "emergency": False}
        if item.get("stoppable") or (container and container in ours):
            out.append(new)
            continue
        port = _as_int(item.get("port"))
        host_ports = sparksi_ports.get(str(item.get("host") or ""))
        if host_ports is None:
            new["why"] = _UNKNOWN
            out.append(new)
            continue
        if port in host_ports:
            new["why"] = _SPARKSI_PORT
            out.append(new)
            continue
        if protect.is_off_limits(port) or port in keep_ports or _kept(item.get("label"), keep) or _kept(container, keep):
            new["why"] = _PROTECTED
            out.append(new)
            continue
        new["emergency"] = True
        out.append(new)
    return out


def same_item(
    items: list[dict],
    host: str,
    port: int,
    container: str | None,
) -> dict | None:
    """Return the item matching host, port and container exactly."""
    wanted = _as_int(port)
    if wanted is None:
        return None
    for item in items:
        if item.get("host") != host:
            continue
        if _as_int(item.get("port")) != wanted:
            continue
        if item.get("container") != container:
            continue
        return item
    return None
