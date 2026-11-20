# SPDX-License-Identifier: Apache-2.0

"""Plan copy, delete, and run. Callers execute only a READY plan."""

from __future__ import annotations

from typing import Any

from sparksi import protect
from sparksi.config import CFG
from sparksi.health import health_paths_for, health_timeout_for
from sparksi.library import archive_mounted, external_refusal, local_path
from sparksi.never import launcher_refusal


def _head() -> str:
    """Node name of the head (first [[nodes]] entry)."""
    return CFG.head.name


def _worker() -> str:
    """Node name of the pair worker (second [[nodes]] entry); "" on a one-Spark site."""
    return CFG.worker.name if CFG.has_worker else ""

PAIR_PORTS = (8888, 8014, 8015)
MIN_FREE_RATIO = 0.20


def _listening(probe: dict[str, Any], port: int) -> bool:
    return bool((probe.get("listening") or {}).get(str(port)))


def pair_live(probes: dict[str, dict[str, Any]]) -> bool:
    for probe in probes.values():
        for port in PAIR_PORTS:
            if _listening(probe or {}, port):
                return True
    return False


def video_service_live(probes: dict[str, dict[str, Any]]) -> bool:
    port = protect.video_port()
    if port is None:
        return False
    for probe in probes.values():
        if _listening(probe or {}, port):
            return True
    return False


def _exclusive_conflict(catalog: dict[str, Any], row: dict[str, Any], probes: dict[str, dict[str, Any]]) -> str | None:
    group = row.get("exclusive")
    if not group:
        return None
    members = ((catalog.get("exclusive_groups") or {}).get(group) or [])
    by_id = {r.get("id"): r for r in (catalog.get("library") or [])}
    for mid in members:
        if mid == row.get("id"):
            continue
        other = by_id.get(mid) or {}
        port = other.get("port")
        host = other.get("listen_host") or _head()
        if port and _listening(probes.get(host) or {}, int(port)):
            return f"{other.get('title') or mid} is already listening on :{port}"
    return None


def memory_verdict(catalog: dict[str, Any], row: dict[str, Any]) -> tuple[str, str]:
    est = row.get("estimate_gib")
    if est is None:
        return "UNCERTAIN", "no declared VRAM for this row"
    hosts = list((row.get("local") or {}).keys()) or [_head()]
    if row.get("placement") != "pair":
        hosts = list(row.get("hosts") or hosts[:1] or [_head()])
    for host in hosts:
        spec = (catalog.get("hosts") or {}).get(host) or {}
        uma = float(spec.get("uma_gib") or 0)
        floor = float(spec.get("floor_gib") or 6)
        margin = float(spec.get("margin_gib") or 4)
        if uma <= 0:
            return "UNCERTAIN", f"no UMA size for {host}"
        budget = uma - floor - margin
        if float(est) > budget:
            return "NO", f"{row.get('id')} needs {est} GiB on {host}; budget is {budget:.0f} GiB declared"
    return "OK", "declared VRAM fits the catalog budget on each charged host"


def charged_hosts(row: dict[str, Any], host: str | None = None) -> list[str]:
    if row.get("placement") == "pair":
        return list((row.get("local") or {}).keys()) or [h for h in (_head(), _worker()) if h]
    if host:
        return [host]
    return list(row.get("hosts") or [_head()])


def _row_tar(row: dict[str, Any]) -> str:
    """Catalog key is ``archive_tar``. ``tar`` remains as a fallback for older rows."""
    return str(row.get("archive_tar") or row.get("tar") or "")


def plan_stage(
    catalog: dict[str, Any],
    row: dict[str, Any],
    host: str,
    probes: dict[str, dict[str, Any]],
    *,
    lock_on: bool,
    refill: bool = False,
) -> dict[str, Any]:
    """Copy plan for one host. ``refill`` re-copies the shards of a flat row that differ from storage."""
    reasons: list[str] = []
    if lock_on:
        reasons.append("lock is ON")
    if host not in (catalog.get("hosts") or {}):
        reasons.append(f"unknown host {host}")
    if pair_live(probes):
        reasons.append("a pair model is listening; copy waits until it is stopped")
    probe = probes.get(host) or {}
    if not probe.get("ok"):
        reasons.append(probe.get("error") or f"no probe for {host}")
    if not archive_mounted(probe):
        reasons.append(f"{host} has no archive mount")
    path = local_path(row, host)
    if not path:
        reasons.append(f"{row.get('id')} has no local path on {host}")
    on = (row.get("on") or {}).get(host)
    if on == "live" or (on == "local" and not (refill and row.get("archive_flat"))):
        reasons.append(f"already present on {host}")
    if row.get("live"):
        reasons.append("weights are in use")
    need = int(row.get("bytes") or 0)
    free = float(probe.get("disk_free_gib") or 0) * 1024**3
    total = float(probe.get("disk_total_gib") or 0) * 1024**3
    if need and total and free - need < MIN_FREE_RATIO * total:
        reasons.append("copy would drop free space through 20%")
    req = None
    if row.get("archive_flat"):
        flat = probe.get("flat") or (probes.get(CFG.archive_node) or {}).get("flat") or {}
        if not flat.get("ok"):
            reasons.append("archive flat folder is incomplete")
        req = {
            "op": "stage_flat",
            "src": row["archive_flat"],
            "dest": path,
            "bytes": need,
            "shards": int(row.get("shards") or 120),
            "exact": bool(refill),
            "min_free_ratio": MIN_FREE_RATIO,
            "refuse_if_listening": list(PAIR_PORTS),
        }
    elif _row_tar(row) and row.get("sha256") and row.get("source_path"):
        req = {
            "op": "stage_tar",
            "tar": _row_tar(row),
            "sha256": row["sha256"],
            "dest": row["source_path"] if host == _head() or row.get("placement") != "pair" else path,
            "bytes": need,
            "min_free_ratio": MIN_FREE_RATIO,
            "refuse_if_listening": list(PAIR_PORTS),
        }
        # single-host releases restore to ea:source on whichever Spark was asked
        req["dest"] = path
    else:
        reasons.append("no archive copy for this row")
    verdict = "READY" if not reasons else "REFUSE"
    return {"verdict": verdict, "reasons": reasons, "host": host, "id": row.get("id"), "request": req if verdict == "READY" else None}


def plan_drop(
    row: dict[str, Any],
    host: str,
    probes: dict[str, dict[str, Any]],
    *,
    lock_on: bool,
    confirm: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    path = local_path(row, host)
    if lock_on:
        reasons.append("lock is ON")
    if not path:
        reasons.append(f"no local path on {host}")
    elif confirm != path:
        reasons.append("confirm must equal the local path")
    if (row.get("on") or {}).get(host) == "live" or row.get("live"):
        reasons.append("weights are in use")
    elif row.get("external") and host in ((row.get("external_engine") or {}).get("hosts") or []):
        reasons.append("weights are in use by an engine started outside a Sparksi run")
    if not archive_mounted(probes.get(host) or {}):
        reasons.append(f"{host} has no archive mount")
    archived = row.get("archive_state") == "archived" or bool(_row_tar(row) or row.get("archive_flat"))
    if row.get("archive_flat"):
        flat = (probes.get(CFG.archive_node) or {}).get("flat") or {}
        if not flat.get("ok"):
            reasons.append("archive flat folder is not complete")
            archived = False
    if not archived:
        reasons.append("no intact archive copy; refusing to delete the only copy")
    video_root = CFG.path("video_service_root")
    if path and video_root and (path == video_root or path.startswith(video_root.rstrip("/") + "/")):
        reasons.append("refusing the protected video service's files")
    req = None
    if not reasons and path:
        req = {"op": "drop", "path": path, "confirm": path, "refuse_if_listening": list(PAIR_PORTS)}
        if _row_tar(row) and row.get("sha256"):
            req["tar"] = _row_tar(row)
            req["sha256"] = row["sha256"]
        if row.get("archive_flat"):
            req["verify_flat"] = row["archive_flat"]
            req["shards"] = int(row.get("shards") or 120)
    return {
        "verdict": "READY" if not reasons else "REFUSE",
        "reasons": reasons,
        "host": host,
        "id": row.get("id"),
        "request": req,
    }


def _live_exclusive(catalog: dict[str, Any], row: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group = row.get("exclusive")
    if not group:
        return []
    members = set((catalog.get("exclusive_groups") or {}).get(group) or [])
    out = []
    for other in rows:
        if other.get("id") == row.get("id"):
            continue
        if other.get("id") in members and other.get("live"):
            out.append(other)
    return out


def _one_per_port(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A pair and a single model can claim one port. Stop the pair when both look live."""
    grouped: dict[Any, list[dict[str, Any]]] = {}
    order: list[Any] = []
    for row in rows:
        key = row.get("port")
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)
    out = []
    for key in order:
        group = grouped[key]
        pairs = [row for row in group if row.get("placement") == "pair"]
        out.append(pairs[0] if len(group) > 1 and pairs else group[0])
    return out


def _missing_hosts(row: dict[str, Any]) -> list[str]:
    # Rows with no weight copy (a video service seat, or a harness whose start
    # script downloads its own checkpoint) set ``no_stage: true`` in the catalog.
    if row.get("no_stage"):
        return []
    missing = []
    for host in charged_hosts(row):
        if (row.get("on") or {}).get(host) == "absent":
            missing.append(host)
    return missing


def _short(host: str) -> str:
    """Short node id (config ``id``) for a node name."""
    return CFG.id_for(host)


def seat_for(row: dict[str, Any], host: str) -> dict[str, Any] | None:
    seats = row.get("seats") or {}
    if isinstance(seats, dict) and host in seats and isinstance(seats[host], dict):
        return seats[host]
    return None


def _budget(catalog: dict[str, Any], host: str) -> float:
    spec = (catalog.get("hosts") or {}).get(host) or {}
    return float(spec.get("uma_gib") or 121) - float(spec.get("floor_gib") or 6) - float(spec.get("margin_gib") or 4)


def _live_on_host(row: dict[str, Any], host: str) -> bool:
    if (row.get("on") or {}).get(host) == "live":
        return True
    return bool(row.get("live")) and (row.get("listen_host") or _head()) == host


def _outside_block(target: dict[str, Any], rows: list[dict[str, Any]], hosts: list[str]) -> str | None:
    """Refuse to start, replace or evict around an engine Sparksi did not start.

    The target itself, or any row an outside engine serves on one of ``hosts``.
    Sparksi never plans a stop for it, so loading there would collide with it.
    """
    mine = external_refusal(target)
    if mine:
        return mine
    wanted = {h for h in hosts if h}
    for other in rows or []:
        if other.get("id") == target.get("id") or not other.get("external"):
            continue
        engine = other.get("external_engine") or {}
        held = set(engine.get("hosts") or [other.get("listen_host") or _head()])
        if held & wanted:
            title = other.get("title") or other.get("id")
            port = engine.get("port") or other.get("port")
            on = ", ".join(_short(h) for h in sorted(held & wanted))
            return (
                f"{title} is running on {on} (port {port}), started outside a Sparksi run; "
                "Sparksi will not stop it. Stop it where it was started, then load again."
            )
    return None


def _stop_step(other: dict[str, Any], host: str) -> dict[str, Any] | None:
    if other.get("external"):
        return None  # never a catalog stop against an engine Sparksi did not start
    seat = seat_for(other, host) or {}
    launch = other.get("launch") or {}
    stop = list(seat.get("stop") or launch.get("stop") or [])
    if not stop:
        return None
    title = other.get("title") or other.get("id")
    return {
        "op": "stop",
        "id": other.get("id"),
        "host": host if seat else (launch.get("host") or host),
        "port": seat.get("port") or other.get("port"),
        "argv": stop,
        "text": f"on {_short(host)}, stop {title}",
    }


def monitor_loaded() -> list[dict[str, Any]] | None:
    """Loaded rows from the announce card. None when that card cannot be read."""
    try:
        from sparksi.dash import current

        data = current()
    except Exception:
        return None
    if not isinstance(data, dict) or not data.get("ok"):
        return None
    loaded = data.get("loaded") or []
    return list(loaded) if isinstance(loaded, list) else None


def pair_holds_worker(rows: list[dict[str, Any]], loaded: list[dict[str, Any]] | None = None) -> bool:
    """The worker node is busy when a loaded model spans it.

    A pair and a single model can share one port, so a pair row's live flag is not the seat.
    Without a monitor sample, fall back to that flag.
    """
    if not CFG.has_worker:
        return False
    if loaded is None:
        return any(row.get("placement") == "pair" and row.get("live") for row in rows)
    for rec in loaded:
        span = rec.get("span") or []
        if rec.get("host") == CFG.worker.id or CFG.worker.id in span:
            return True
    return False


def _pair_sides(target: dict[str, Any]) -> list[dict[str, Any]]:
    """Head and worker of a pair. Names come from the row and the node config."""
    launch = target.get("launch") or {}
    ident = target.get("identity") or {}
    return [
        {
            "role": "head",
            "host": launch.get("host") or _head(),
            "container": ident.get("container"),
            "port": target.get("port"),
        },
        {
            "role": "worker",
            "host": _worker(),
            "container": ident.get("worker_container") or ident.get("container"),
            "port": None,
        },
    ]


def _start_fields(catalog: dict[str, Any], target: dict[str, Any], placement: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "health_timeout_s": health_timeout_for(catalog, target),
        "http_paths": health_paths_for(target),
        "placement": placement,
    }
    if placement == "pair":
        launch = target.get("launch") or {}
        fields["sides"] = _pair_sides(target)
        fields["stop_argv"] = list(launch.get("stop") or [])
    return fields


def _unknown_holder(target: dict[str, Any], host: str, steps: list[dict[str, Any]]) -> str | None:
    """Refuse to start into a port that answers as something no planned stop will free."""
    holder = (target.get("port_holder") or {}).get(host)
    if not holder:
        return None
    if any(step.get("op") == "stop" for step in steps):
        return None
    return f"port {target.get('port')} on {_short(host)} is held by {holder}; stop it first"


def plan_switch(
    catalog: dict[str, Any],
    target: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    lock_on: bool,
    host: str = "",
    loaded: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Load a harness on one Spark, or as a pair across both. The other Spark stays up."""
    reasons: list[str] = []
    if lock_on:
        reasons.append("lock is ON")
    placement = target.get("placement") or "single"
    steps: list[dict[str, Any]] = []
    if placement == "pair" and not CFG.has_worker:
        return {
            "verdict": "REFUSE",
            "reasons": ["this model runs on two Sparks; this site has 1 Spark"],
            "id": target.get("id"),
            "host": host,
            "steps": [],
            "preview": [],
        }
    if placement == "pair":
        hosts = [(target.get("launch") or {}).get("host") or _head(), _worker()]
    else:
        hosts = [host or target.get("listen_host") or _head()]
    outside = _outside_block(target, rows, hosts)
    if outside:
        return {"verdict": "REFUSE", "reasons": [outside], "id": target.get("id"), "host": host, "steps": [], "preview": []}
    if placement == "pair":
        launch = target.get("launch") or {}
        start = list(launch.get("start") or [])
        if not start:
            reasons.append("no harness; this archive can be copied and deleted only")
        blocked = launcher_refusal(start)
        if blocked:
            return {
                "verdict": "REFUSE",
                "reasons": [blocked],
                "id": target.get("id"),
                "host": host,
                "steps": [],
                "preview": [],
            }
        if target.get("live") and not _missing_hosts(target):
            return {"verdict": "NOOP", "reasons": ["already in memory"], "id": target.get("id"), "steps": [], "preview": []}
        for other in rows:
            if other.get("id") == target.get("id") or not other.get("live"):
                continue
            if other.get("placement") == "pair" or other.get("placement") == "single":
                for side in (_head(), _worker()):
                    if other.get("placement") == "single" and not _live_on_host(other, side):
                        continue
                    if other.get("placement") == "pair" and side != _head():
                        continue
                    step = _stop_step(other, side if other.get("placement") == "single" else (launch.get("host") or _head()))
                    if step and step["id"] not in {s["id"] for s in steps}:
                        steps.append(step)
        unknown = _unknown_holder(target, launch.get("host") or _head(), steps)
        if unknown:
            return {"verdict": "REFUSE", "reasons": [unknown], "id": target.get("id"), "steps": [], "preview": []}
        missing = _missing_hosts(target)
        if missing:
            steps.append({"op": "copy", "id": target.get("id"), "hosts": missing, "text": "copy weights to " + ", ".join(missing)})
        steps.append({
            "op": "start",
            "id": target.get("id"),
            "host": launch.get("host") or _head(),
            "port": target.get("port"),
            "argv": start,
            "timeout_s": target.get("start_timeout_s"),
            "text": f"start {target.get('title') or target.get('id')} as a pair",
            **_start_fields(catalog, target, "pair"),
        })
    else:
        spark = host or target.get("listen_host") or _head()
        seat = seat_for(target, spark)
        if not seat or not list(seat.get("start") or []):
            return {
                "verdict": "REFUSE",
                "reasons": [f"no launcher on {_short(spark)}"],
                "id": target.get("id"),
                "host": spark,
                "steps": [],
                "preview": [],
            }
        blocked = launcher_refusal(list(seat.get("start") or []))
        if blocked:
            return {
                "verdict": "REFUSE",
                "reasons": [blocked],
                "id": target.get("id"),
                "host": spark,
                "steps": [],
                "preview": [],
            }
        # The worker's one GPU seat is the pair worker. A single model on the head does not take it.
        if spark == _worker() and pair_holds_worker(rows, loaded):
            return {
                "verdict": "REFUSE",
                "reasons": [f"{_short(spark)} is in a pair; stop that model first"],
                "id": target.get("id"),
                "host": spark,
                "steps": [],
                "preview": [],
            }
        if _live_on_host(target, spark):
            return {"verdict": "NOOP", "reasons": [f"already in memory on {_short(spark)}"], "id": target.get("id"), "steps": [], "preview": []}
        others = [
            row for row in rows
            if row.get("id") != target.get("id") and row.get("placement") != "pair" and _live_on_host(row, spark)
        ]
        used = sum(float(row.get("estimate_gib") or 0) for row in others)
        need = float(target.get("estimate_gib") or 0)
        same_port = [row for row in others if (seat_for(row, spark) or {}).get("port", row.get("port")) == seat.get("port")]
        stop_these = others if (used + need > _budget(catalog, spark) or same_port) else []
        if same_port and not (used + need > _budget(catalog, spark)):
            stop_these = same_port
        for other in stop_these:
            step = _stop_step(other, spark)
            if step:
                steps.append(step)
        unknown = _unknown_holder(target, spark, steps)
        if unknown:
            return {"verdict": "REFUSE", "reasons": [unknown], "id": target.get("id"), "host": spark, "steps": [], "preview": []}
        if (target.get("on") or {}).get(spark) == "absent":
            steps.append({"op": "copy", "id": target.get("id"), "hosts": [spark], "text": f"copy weights to {_short(spark)}"})
        steps.append({
            "op": "start",
            "id": target.get("id"),
            "host": spark,
            "port": seat.get("port"),
            "argv": list(seat.get("start") or []),
            "text": f"on {_short(spark)}, start {target.get('title') or target.get('id')}",
            **_start_fields(catalog, target, "single"),
        })
        if target.get("start_note"):
            steps.append({"op": "note", "text": str(target["start_note"])})
    hard = [r for r in reasons if r != "lock is ON"]
    verdict = "READY" if not reasons else "REFUSE"
    shown = steps if not hard else []
    return {
        "verdict": verdict,
        "reasons": reasons,
        "id": target.get("id"),
        "host": host,
        "steps": shown,
        "preview": [s["text"] for s in shown],
    }


def plan_run(
    catalog: dict[str, Any],
    row: dict[str, Any],
    probes: dict[str, dict[str, Any]],
    *,
    lock_on: bool,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if lock_on:
        reasons.append("lock is ON")
    launch = row.get("launch") or {}
    hosts = [launch.get("host") or _head()] + ([_worker()] if row.get("placement") == "pair" else [])
    outside = _outside_block(row, list(rows or []), hosts)
    if outside:
        reasons.append(outside)
    start = list(launch.get("start") or [])
    if not start:
        reasons.append("no launcher; this archive can be copied and deleted only")
    if any(str(p).endswith("video-mode.sh") or Path_name(str(p)) == "tmux" for p in start):
        reasons.append("refusing a forbidden launcher")
    if video_service_live(probes) and not protect.is_video_id(row.get("id")):
        reasons.append("the protected video service is resident")
    conflict = _exclusive_conflict(catalog, row, probes)
    if conflict:
        reasons.append(conflict)
    for host in charged_hosts(row):
        if (row.get("on") or {}).get(host) == "absent":
            reasons.append(f"weights are absent on {host}")
    mem, why = memory_verdict(catalog, row)
    if mem != "OK":
        reasons.append(why)
    return {
        "verdict": "READY" if not reasons else "REFUSE",
        "reasons": reasons,
        "id": row.get("id"),
        "host": launch.get("host") or _head(),
        "argv": start if not reasons else None,
        "memory": mem,
    }


def plan_stop(row: dict[str, Any], *, lock_on: bool) -> dict[str, Any]:
    outside = external_refusal(row)
    if outside:
        return {"verdict": "REFUSE", "reasons": [outside], "external": True, "id": row.get("id"),
                "host": (row.get("launch") or {}).get("host") or _head(), "argv": None}
    reasons: list[str] = []
    if lock_on:
        reasons.append("lock is ON")
    if str(row.get("id") or "").lower() in protect.protected_ids():
        reasons.append("refusing to stop the protected video service")
    launch = row.get("launch") or {}
    seat = seat_for(row, row.get("listen_host") or _head()) or {}
    stop = list(seat.get("stop") or launch.get("stop") or [])
    if not stop:
        reasons.append("no stop command")
    blocked = launcher_refusal(stop)
    if blocked:
        reasons.append(blocked)
    return {
        "verdict": "READY" if not reasons else "REFUSE",
        "reasons": reasons,
        "id": row.get("id"),
        "host": launch.get("host") or _head(),
        "argv": stop if not reasons else None,
    }


def Path_name(text: str) -> str:
    return text.rstrip("/").split("/")[-1]
