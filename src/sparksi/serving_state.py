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
# Model id -> (served names, served prefixes) from its catalog row and recipes, for one reconcile.
_SERVED: ContextVar[dict[str, tuple[set[str], list[str]]]] = ContextVar("serving_served_names", default={})
# Every run record of this reconcile, so a newer run that reuses a container name wins it.
_RUNS: ContextVar[list[dict[str, Any]]] = ContextVar("serving_runs", default=[])
_NAME_FLAGS = ("--served-model-name", "--alias", "-a")
# TensorFold names the served model with --name inside the container args. Only there:
# elsewhere --name is docker's container name, never a served name.
_TENSORFOLD_FLAGS = _NAME_FLAGS + ("--name",)


def reconcile(
    loaded: list[Any] | None,
    runs: list[Any] | None,
    *,
    hosts: list[Any] | None = None,
    head_id: str = "head",
    worker_id: str = "worker",
    extra_ids: dict[str, str] | None = None,
    served_names: dict[str, tuple[set[str], list[str]]] | None = None,
) -> list[dict[str, Any]]:
    """One row per Spark: head, then worker, then any extra roles (node3, node4).

    ``worker_id`` is "" on a one-Spark site, which drops the worker row.
    ``extra_ids`` maps an extra role to its node id. ``served_names`` maps a
    model id to the served names and prefixes its catalog row and recipes
    declare (see ``served_names_for``); without it a run may serve only its
    model id or title. Each row has state, reason, text, title, port, and run_id.
    """
    rows = [item for item in loaded or [] if isinstance(item, dict)]
    recorded = [item for item in runs or [] if isinstance(item, dict)]
    extras = {str(role): str(node) for role, node in (extra_ids or {}).items() if role not in ("head", "worker")}
    roles = ["head"] + (["worker"] if worker_id else []) + list(extras)
    token = _EXTRA.set(extras)
    served = _SERVED.set(served_names if served_names is not None else {})
    every = _RUNS.set(recorded)
    try:
        return [_role(role, rows, recorded, hosts, head_id, worker_id) for role in roles]
    finally:
        _RUNS.reset(every)
        _SERVED.reset(served)
        _EXTRA.reset(token)


def attach(
    body: dict[str, Any],
    loaded: list[Any] | None,
    hosts: list[Any] | None,
    *,
    head_id: str,
    worker_id: str,
    extra_ids: dict[str, str] | None = None,
    served_names: dict[str, tuple[set[str], list[str]]] | None = None,
) -> dict[str, Any]:
    """Add serving when the body does not already carry it. Existing keys stay."""
    out = dict(body)
    if isinstance(out.get("serving"), list):
        return out
    recorded = out.get("runs") if isinstance(out.get("runs"), list) else []
    if served_names is None:
        served_names = _LazyServedNames()
    out["serving"] = reconcile(
        loaded,
        recorded,
        hosts=hosts,
        head_id=head_id,
        worker_id=worker_id,
        extra_ids=extra_ids,
        served_names=served_names,
    )
    return out


class _LazyServedNames(dict):
    """Declared names per model, read from the catalog once and only when a container must decide."""

    def __init__(self) -> None:
        super().__init__()
        self._catalog: dict[str, Any] | None = None

    def __missing__(self, model_id: str) -> tuple[set[str], list[str]]:
        if self._catalog is None:
            self._catalog = _catalog()
        found = served_names_for([model_id], self._catalog)[model_id]
        self[model_id] = found
        return found


def _catalog() -> dict[str, Any]:
    try:
        from sparksi.catalog import load_catalog

        return load_catalog()
    except Exception:  # noqa: BLE001 — no catalog: runs may serve only their model id or title
        return {}


def served_names_for(
    model_ids: list[str], catalog: dict[str, Any] | None = None
) -> dict[str, tuple[set[str], list[str]]]:
    """Served names (folded) and prefixes each model may answer as: catalog row and every saved recipe."""
    if not model_ids:
        return {}
    if catalog is None:
        catalog = _catalog()
    rows = {str(row.get("id") or ""): row for row in catalog.get("library") or [] if isinstance(row, dict)}
    out: dict[str, tuple[set[str], list[str]]] = {}
    for model_id in model_ids:
        names: set[str] = {model_id}
        prefixes: list[str] = []
        row = rows.get(model_id) or {}
        ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
        served = ident.get("served") or []
        names.update(str(item) for item in ([served] if isinstance(served, str) else served))
        if row.get("title"):
            names.add(str(row["title"]))
        if ident.get("served_prefix"):
            prefixes.append(str(ident["served_prefix"]))
        names.update(_recipe_served(model_id))
        out[model_id] = ({name.strip().casefold() for name in names if str(name).strip()}, prefixes)
    return out


def _recipe_served(model_id: str) -> set[str]:
    """Names a saved recipe (generated, copy or pinned) serves as: its alias and served-name flags."""
    try:
        from sparksi.recipes.store import load_recipe, recipe_names, using_recipe

        found: set[str] = set()
        for name in recipe_names(model_id):
            with using_recipe(model_id, name):
                recipe = load_recipe(model_id)
            if recipe.get("alias"):
                found.add(str(recipe["alias"]))
            roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
            flags = _TENSORFOLD_FLAGS if str(recipe.get("engine") or "") == "tensorfold" else _NAME_FLAGS
            for side in roles.values():
                if not isinstance(side, dict):
                    continue
                args = [str(part) for part in side.get("args") or []] if isinstance(side.get("args"), list) else []
                for index, arg in enumerate(args):
                    if arg in flags and index + 1 < len(args):
                        found.add(args[index + 1])
                    for flag in flags:
                        if arg.startswith(flag + "="):
                            found.add(arg.split("=", 1)[1])
                env = side.get("env") if isinstance(side.get("env"), dict) else {}
                if env.get("SERVED_MODEL_NAME"):
                    found.add(str(env["SERVED_MODEL_NAME"]))
        return found
    except Exception:  # noqa: BLE001 — an unreadable recipe adds no names
        return set()


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
        # Newest start wins; an equal start is broken by id so the answer never depends on order.
        if best is None or _run_order(run) > _run_order(best):
            best = run
    return best


def _run_order(run: dict[str, Any]) -> tuple[float, str]:
    try:
        started = float(run.get("started") or 0)
    except (TypeError, ValueError):
        started = 0.0
    return started, str(run.get("id") or "")


def _same_instance(run: dict[str, Any], row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    if str(run.get("phase") or "") not in _ACTIVE:
        return False
    if not _run_uses_role(run, role) or not _same_identity(run, row, role, head_id, worker_id):
        return False
    if _port(run) != _port(row) or _port(row) == 0:
        return False
    return _containers_agree(run, row, role, head_id, worker_id)


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
    if served in names:
        return True
    # Two catalog rows or a recipe copy can serve one id, so the served name cannot pick
    # between their runs. The run's own container then decides, but only for a name this
    # model may serve: a served name it never declares vetoes a container-name match.
    if not _may_serve(run, served):
        return False
    return _container_owner(run, row, role, head_id, worker_id)


def _may_serve(run: dict[str, Any], served: str) -> bool:
    model_id = str(run.get("model_id") or "")
    folded = served.strip().casefold()
    known = _SERVED.get()
    lazy = isinstance(known, _LazyServedNames)
    names, prefixes = known[model_id] if lazy or model_id in known else (set(), [])
    allowed = {model_id.casefold(), str(run.get("title") or "").casefold()} | set(names)
    allowed.discard("")
    if folded in allowed:
        return True
    return any(prefix and folded.startswith(prefix.casefold()) for prefix in prefixes)


def _live_container(row: dict[str, Any], role: str, head_id: str, worker_id: str) -> str:
    """The engine-reported container for this role, or "" when it is not from the engine."""
    if str(row.get("container_source") or "") != "engine":
        return ""
    if _is_pair(row, head_id, worker_id) and role == "worker":
        return str(row.get("worker_name") or "")
    return str(row.get("container") or "")


def _container_owner(run: dict[str, Any], row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    live = _live_container(row, role, head_id, worker_id)
    if not live:
        return False
    if role in (run.get("stopped_sides") or []):
        return False
    from sparksi.library import _superseded_by

    for node in run.get("nodes") or []:
        if not isinstance(node, dict) or node.get("role") != role:
            continue
        if str(node.get("phase") or "") in ("stopped", "cancelled") or str(node.get("container") or "") != live:
            continue
        # The same rule as run_live: a newer run that reuses this host and name owns it.
        if _superseded_by(run, str(node.get("host") or ""), live, _RUNS.get()):
            continue
        return True
    return False


def _observed_identity(row: dict[str, Any], role: str, head_id: str, worker_id: str) -> str:
    """Served name counts only for a role that is up, and not a static truth hint."""
    if not _role_up(row, role, head_id, worker_id):
        return ""
    if str(row.get("container_source") or "") == "truth" and row.get("head_up") is not True:
        return ""
    return _identity(row)


def _containers_agree(run: dict[str, Any], row: dict[str, Any], role: str, head_id: str, worker_id: str) -> bool:
    live = _live_container(row, role, head_id, worker_id)
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
