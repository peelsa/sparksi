# SPDX-License-Identifier: Apache-2.0

"""Explicit serving display.

One function turns monitor rows and run records into a stable state for each
role. A stopped run is never the label for a later container.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

_ACTIVE = frozenset({"starting", "waiting_health", "copying", "ready", "stopping"})
_LOADING = frozenset({"starting", "waiting_health", "copying", "stopping"})
STATES = ("idle", "loading", "ready", "answering", "split", "observed", "stale")
# Role -> node id for a third and fourth Spark, set for the length of one reconcile.
_EXTRA: ContextVar[dict[str, str]] = ContextVar("serving_extra_ids", default={})


def reconcile(
    loaded: list[Any] | None,
    runs: list[Any] | None,
    *,
    hosts: list[Any] | None = None,
    head_id: str = "head",
    worker_id: str = "worker",
    extra_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """One row per Spark: head, then worker, then any extra roles (node3, node4).

    ``worker_id`` is "" on a one-Spark site, which drops the worker row.
    ``extra_ids`` maps an extra role to its node id. Each row has state,
    reason, text, title, port, and run_id.
    """
    rows = [item for item in loaded or [] if isinstance(item, dict)]
    recorded = [item for item in runs or [] if isinstance(item, dict)]
    extras = {str(role): str(node) for role, node in (extra_ids or {}).items() if role not in ("head", "worker")}
    roles = ["head"] + (["worker"] if worker_id else []) + list(extras)
    token = _EXTRA.set(extras)
    try:
        return [_role(role, rows, recorded, hosts, head_id, worker_id) for role in roles]
    finally:
        _EXTRA.reset(token)


def attach(
    body: dict[str, Any],
    loaded: list[Any] | None,
    hosts: list[Any] | None,
    *,
    head_id: str,
    worker_id: str,
    extra_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Add serving when the body does not already carry it. Existing keys stay."""
    out = dict(body)
    if isinstance(out.get("serving"), list):
        return out
    recorded = out.get("runs") if isinstance(out.get("runs"), list) else []
    out["serving"] = reconcile(loaded, recorded, hosts=hosts, head_id=head_id, worker_id=worker_id, extra_ids=extra_ids)
    return out


def _role(
    role: str,
    rows: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    hosts: list[Any] | None,
    head_id: str,
    worker_id: str,
) -> dict[str, Any]:
    # The pair spans the head and worker only; a third or fourth Spark never inherits it.
    pair = _pair_row(rows, head_id, worker_id) if role in ("head", "worker") else None
    own = _confirming(rows, role, head_id, worker_id)
    source = own or pair
    if source is None:
        return _blank(role)
    if _announced_stale(source, hosts, head_id):
        view = _view(
            role,
            "stale",
            _title(source, role, head_id, worker_id),
            "Report is stale",
            _port(source),
            "",
            "The report that carried this process is stale.",
        )
        _reject(view, runs)
        return view
    # A split is decided before any run match. One side down is not a ready run.
    if pair is not None and _split(pair):
        up = _side_up(pair, role)
        view = _view(
            role,
            "split",
            _title(pair, role, head_id, worker_id),
            f"Split · {role} {'up' if up else 'not confirmed'} · port {_port(pair)}",
            _port(pair),
            "",
            _split_reason(runs, pair),
        )
        _reject(view, runs)
        return view
    matched = _matching_run(runs, own, role, head_id, worker_id) if own is not None else None
    if matched is not None:
        state = _state_for_run(matched)
        view = _view(
            role,
            state,
            _run_title(matched),
            _run_text(state, own or source),
            _port(own or source),
            str(matched.get("id") or ""),
            _run_reason(state),
        )
        _reject(view, runs)
        return view
    if own is not None:
        named = bool(_observed_identity(own, role, head_id, worker_id))
        view = _view(
            role,
            "answering" if named else "observed",
            _title(own, role, head_id, worker_id),
            f"Answering · port {_port(own)}" if named else f"Up · identity not reported · port {_port(own)}",
            _port(own),
            "",
            "This process is up. No active run owns it.",
        )
        _reject(view, runs)
        return view
    return _blank(role)


def _blank(role: str) -> dict[str, Any]:
    return _view(role, "idle", "", "Nothing running", 0, "", "Nothing is answering on this role.")


def _view(role: str, state: str, title: str, text: str, port: int, run_id: str, reason: str) -> dict[str, Any]:
    return {
        "role": role,
        "state": state,
        "reason": reason,
        "text": text,
        "title": title,
        "port": port,
        "run_id": run_id,
    }


def _reject(view: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    """Refuse a state that would label a live process with a closed run."""
    state = view.get("state")
    if state not in STATES:
        raise ValueError("unknown serving state")
    run_id = str(view.get("run_id") or "")
    if state in {"split", "observed", "idle", "stale", "answering"} and run_id:
        raise ValueError("this state cannot carry a run id")
    if not run_id:
        return
    run = next((item for item in runs if str(item.get("id") or "") == run_id), None)
    if run is None or str(run.get("phase") or "") not in _ACTIVE:
        raise ValueError("only an active run can label the current instance")


def _pair_row(rows: list[dict[str, Any]], head_id: str, worker_id: str) -> dict[str, Any] | None:
    for row in rows:
        if not _is_pair(row, head_id, worker_id):
            continue
        if _side_up(row, "head") or _side_up(row, "worker"):
            return row
    return None


def _confirming(rows: list[dict[str, Any]], role: str, head_id: str, worker_id: str) -> dict[str, Any] | None:
    for row in rows:
        if _role_up(row, role, head_id, worker_id):
            return row
    return None


def _is_pair(row: dict[str, Any], head_id: str, worker_id: str) -> bool:
    if not worker_id:
        return False
    span = [str(item) for item in row.get("span") or [] if item]
    if head_id not in span:
        return False
    return worker_id in span or str(row.get("worker_host") or "") == worker_id


def _split(row: dict[str, Any]) -> bool:
    return _side_up(row, "head") != _side_up(row, "worker")


def _side_up(row: dict[str, Any], role: str) -> bool:
    return row.get("head_up") is True if role == "head" else row.get("worker_up") is True


def _role_up(row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    """Pair rows use each side's flag. A single host uses head_up for that host."""
    if str(row.get("state") or "") == "DOWN":
        return False
    if _is_pair(row, head_id, worker_id):
        return role in ("head", "worker") and _side_up(row, role)
    extra = _EXTRA.get()
    expected = head_id if role == "head" else (worker_id if role == "worker" else extra.get(role, ""))
    if not expected:
        return False
    if str(row.get("host") or head_id) != expected:
        return False
    return row.get("head_up") is True


def _announced_stale(row: dict[str, Any], hosts: list[Any] | None, head_id: str) -> bool:
    """Live only when the announcing host is present and fresh is exactly true."""
    if not hosts:
        return True
    host_id = str(row.get("host") or head_id)
    for host in hosts:
        if not isinstance(host, dict) or str(host.get("id") or "") != host_id:
            continue
        return host.get("fresh") is not True
    return True


def _matching_run(
    runs: list[dict[str, Any]], row: dict[str, Any], role: str, head_id: str, worker_id: str
) -> dict[str, Any] | None:
    best = None
    for run in runs:
        if not _same_instance(run, row, role, head_id, worker_id):
            continue
        if best is None or int(run.get("started") or 0) >= int(best.get("started") or 0):
            best = run
    return best


def _same_instance(run: dict[str, Any], row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    if str(run.get("phase") or "") not in _ACTIVE:
        return False
    if not _run_uses_role(run, role) or not _same_identity(run, row, role, head_id, worker_id):
        return False
    if _port(run) != _port(row) or _port(row) == 0:
        return False
    return _containers_agree(run, row, role)


def _run_uses_role(run: dict[str, Any], role: str) -> bool:
    nodes = [node for node in run.get("nodes") or [] if isinstance(node, dict)]
    if nodes:
        return any(node.get("role") == role for node in nodes)
    place = str(run.get("placement") or "head")
    if place in {"both", "pair"}:
        return True
    return place == role


def _same_identity(run: dict[str, Any], row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    served = _observed_identity(row, role, head_id, worker_id)
    if not served:
        return False
    names = {str(run.get("model_id") or ""), str(run.get("title") or "")}
    names.discard("")
    return served in names


def _observed_identity(row: dict[str, Any], role: str, head_id: str, worker_id: str) -> str:
    """Served name counts only for a role that is up, and not a static truth hint."""
    if not _role_up(row, role, head_id, worker_id):
        return ""
    if str(row.get("container_source") or "") == "truth" and row.get("head_up") is not True:
        return ""
    return _identity(row)


def _containers_agree(run: dict[str, Any], row: dict[str, Any], role: str) -> bool:
    if str(row.get("container_source") or "") != "engine":
        return True
    live = str(row.get("container") or "") if role == "head" else str(row.get("worker_name") or "")
    if not live:
        return True
    for node in run.get("nodes") or []:
        if not isinstance(node, dict) or node.get("role") != role:
            continue
        recorded = str(node.get("container") or "")
        if recorded and recorded != live:
            return False
    return True


def _state_for_run(run: dict[str, Any]) -> str:
    phase = str(run.get("phase") or "")
    if phase == "ready":
        return "ready"
    if phase in _LOADING:
        return "loading"
    raise ValueError("only an active run can label the current instance")


def _run_text(state: str, row: dict[str, Any]) -> str:
    port = _port(row)
    if state == "ready":
        return f"Ready · port {port}"
    if state == "answering":
        return f"Answering · port {port}"
    return f"Loading · port {port}"


def _run_reason(state: str) -> str:
    if state == "ready":
        return "The recorded run is ready and this role is confirmed."
    return "The recorded run is still starting."


def _split_reason(runs: list[dict[str, Any]], row: dict[str, Any]) -> str:
    if _closed_on_port(runs, _port(row)):
        return (
            "One side of this pair is up and the other is not confirmed. "
            "A closed run on this port is an earlier instance."
        )
    return "One side of this pair is up and the other is not confirmed."


def _closed_on_port(runs: list[dict[str, Any]], port: int) -> bool:
    return any(str(run.get("phase") or "") not in _ACTIVE and _port(run) == port for run in runs)


def _run_title(run: dict[str, Any]) -> str:
    return str(run.get("title") or run.get("model_id") or "Run")


def _title(row: dict[str, Any], role: str, head_id: str, worker_id: str) -> str:
    observed = _observed_identity(row, role, head_id, worker_id)
    if observed:
        return observed
    for key in ("name", "container"):
        value = row.get(key)
        if value:
            return str(value)
    return "Live process"


def _identity(row: dict[str, Any]) -> str:
    value = row.get("served")
    return str(value).strip() if value else ""


def _port(row: dict[str, Any]) -> int:
    try:
        return int(row.get("port") or 0)
    except (TypeError, ValueError):
        return 0
