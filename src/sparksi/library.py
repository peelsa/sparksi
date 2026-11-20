# SPDX-License-Identifier: Apache-2.0

"""Library rows: catalog overlay plus archive release metas, with per-host presence.

The archive store is configured in ``[archive]`` (owner node, root, label).
Older catalog row keys can be mapped to ``archive_tar``/``archive_flat`` with
``[compat] legacy_row_keys`` in sparksi.toml.
"""

from __future__ import annotations

import ipaddress
import time
from typing import Any

from sparksi import compat, protect
from sparksi.config import CFG
from sparksi.remote import SshRunner


def archive_mounted(probe: dict[str, Any] | None) -> bool:
    """Probe says the archive is reachable. Accepts the legacy key if configured."""
    probe = probe or {}
    if "mount_archive" in probe:
        return bool(probe.get("mount_archive"))
    legacy = str(CFG.compat.get("legacy_mount_key") or "")
    return bool(legacy and probe.get(legacy))


def _compat_mount(card: dict[str, Any], value: Any) -> dict[str, Any]:
    """Compat shim: also publish the old mount key for older API readers."""
    legacy = str(CFG.compat.get("legacy_mount_key") or "")
    if legacy:
        card[legacy] = value
    return card

GiB = 1024**3
PROBE_PORTS = [8007, 8011, 8014, 8015, 8195, 8765, 8888]  # plus any protect preset port, see probe_ports()


def probe_ports() -> list[int]:
    return sorted(set(PROBE_PORTS) | protect.off_limits_ports())


def overlay_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in catalog.get("library") or []:
        if not isinstance(raw, dict) or not raw.get("id"):
            continue
        row = dict(raw)
        row["source"] = "catalog"
        row["launcher"] = "yes" if (row.get("launch") or {}).get("start") else "no"
        row.setdefault("archive_tar", row.get("archive_flat") or None)
        rows.append(row)
    return rows


def release_row(meta: dict[str, Any]) -> dict[str, Any] | None:
    slug = str(meta.get("slug") or "")
    source = str(meta.get("source") or "")
    if not slug or not source.startswith("/"):
        return None
    nbytes = int(meta.get("bytes") or 0)
    return {
        "id": slug,
        "title": meta.get("title") or slug,
        "placement": "single",
        "gib": round(nbytes / GiB, 1),
        "bytes": nbytes,
        "archive_tar": meta.get("tar"),
        "sha256": meta.get("sha256") or "",
        "tar": meta.get("tar"),
        "source_path": source,
        "estimate_gib": None,
        "launcher": "no",
        "launch": None,
        "local": {name: source for name in CFG.node_names},
        "hosts": list(CFG.node_names),
        "from_release": True,
    }


def merge_rows(overlay: list[dict[str, Any]], releases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {r["id"] for r in overlay}
    known_tar = {r.get("archive_tar") for r in overlay if r.get("archive_tar")}
    out = list(overlay)
    for meta in releases:
        row = release_row(meta)
        if not row or row["id"] in known or (row.get("archive_tar") or row.get("tar")) in known_tar:
            continue
        out.append(row)
    return out


def host_ids(catalog: dict[str, Any]) -> list[str]:
    return list((catalog.get("hosts") or {}).keys()) or list(CFG.node_names)


def local_path(row: dict[str, Any], host: str) -> str | None:
    local = row.get("local") or {}
    if host in local:
        return str(local[host])
    if row.get("placement") == "single" and host in (row.get("hosts") or []):
        return str(local.get(host) or "")
    return None


def http_paths(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Per-port identity path.

    A row's declared ``http`` path wins. A port where a row declares served names
    is asked /v1/models, so the answer carries the names to match: /health returns
    200 with no names. Other ports keep the agent's default (/health, then /v1/models).
    """
    out: dict[str, str] = {}
    served_ports: list[str] = []
    for row in rows:
        ident = row.get("identity") or {}
        path = ident.get("http")
        if path and row.get("port"):
            out[str(row["port"])] = str(path)
        elif ident.get("served") and row.get("port"):
            served_ports.append(str(row["port"]))
    for port in served_ports:
        out.setdefault(port, "/v1/models")
    return out


def _probe_request(catalog: dict[str, Any], host: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths = []
    for row in rows:
        p = local_path(row, host)
        if p:
            paths.append(p)
    req: dict[str, Any] = {
        "op": "probe", "ports": probe_ports(), "paths": paths, "http_paths": http_paths(rows), "mounts": True,
    }
    if host == CFG.archive_node:
        req["scan_releases"] = True
        flat_row = next((r for r in rows if r.get("archive_flat")), None)
        if flat_row:
            req["scan_flat"] = flat_row["archive_flat"]
            req["shards"] = int(flat_row.get("shards") or 120)
    return req


def identity_check(row: dict[str, Any], probe: dict[str, Any] | None) -> tuple[bool, str | None]:
    """(answers as this row, what holds the port) on one host.

    A bound port is not enough: :8014 is shared by four harnesses and tailscaled
    can hold :8188. The row's catalog ``identity`` decides: ``served`` ids must
    appear in /v1/models, ``container`` must be running, ``http`` must return 200
    (and /system_stats must look like the video service). No identity: any /v1/models id.
    """
    probe = probe or {}
    port = row.get("port")
    if not port or not (probe.get("listening") or {}).get(str(port)):
        return False, None
    ident = row.get("identity") or {}
    h = (probe.get("http") or {}).get(str(port)) or {}
    models = [str(m) for m in h.get("models") or []]
    status = h.get("status")
    if models:
        holder = ", ".join(models)
    elif status is not None:
        holder = f"HTTP {status} on {h.get('path') or '/v1/models'}"
    else:
        holder = "port bound, no HTTP answer"
    if ident.get("http"):
        ok = status == 200 and (h.get("path") == ident["http"])
        if ok and ident["http"] == "/system_stats":
            ok = compat.agent_video_flag(h)
    elif ident.get("served"):
        want = {str(x).lower() for x in ident["served"]}
        ok = any(m.lower() in want for m in models)
    else:
        ok = bool(models)
    containers = probe.get("containers")
    if ok and ident.get("container") and containers is not None:
        ok = ident["container"] in containers
    return ok, holder


EXTERNAL_NOTE = "Started outside a Sparksi run"
EXTERNAL_REFUSAL = "{title} was started outside a Sparksi run; Sparksi will not stop or replace it. Stop it where it was started."


def _served_answers(row: dict[str, Any], probe: dict[str, Any] | None) -> bool:
    """The row's declared served name answers on its port. Container names are not checked."""
    probe = probe or {}
    port = row.get("port")
    ident = row.get("identity") or {}
    served = ident.get("served")
    if not port or ident.get("http") or not served:
        return False
    if not (probe.get("listening") or {}).get(str(port)):
        return False
    h = (probe.get("http") or {}).get(str(port)) or {}
    if h.get("status") != 200:
        return False
    want = {str(x).lower() for x in ([served] if isinstance(served, str) else served)}
    return any(str(m).lower() in want for m in h.get("models") or [])


def _sparksi_containers(runs: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """(host, container) of every run side that is not known stopped: Sparksi owns those."""
    out: set[tuple[str, str]] = set()
    for run in runs or []:
        if not isinstance(run, dict) or str(run.get("phase") or "") in _ENDED:
            continue
        for node in run.get("nodes") or []:
            if isinstance(node, dict) and node.get("container"):
                out.add((str(node.get("host") or ""), str(node["container"])))
    return out


def _mounting(probe: dict[str, Any] | None, path: str) -> list[str] | None:
    """Running containers on one host that mount ``path`` or a folder inside it. None: unknown."""
    mounts = (probe or {}).get("mounts")
    if not isinstance(mounts, dict) or not path:
        return None
    root = path.rstrip("/")
    names = []
    for name, sources in mounts.items():
        if any(str(src).rstrip("/") == root or str(src).startswith(root + "/") for src in sources or []):
            names.append(str(name))
    return sorted(names)


def external_engine(
    row: dict[str, Any], probes: dict[str, dict[str, Any]], runs: list[dict[str, Any]] | None = None
) -> dict[str, Any] | None:
    """An engine Sparksi did not start that serves this row, or None.

    Only for a row whose catalog pins its container names (otherwise the catalog
    identity already decides). It needs all of: the pinned container is not
    running on the head, the row's served name answers on its port there, and a
    running container on the head mounts this row's weights. A sparksi- container
    or one a not-stopped run recorded is Sparksi's own and never counts.
    """
    ident = row.get("identity") or {}
    pinned = ident.get("container") or ident.get("worker_container")
    if not pinned or protect.is_off_limits(row.get("port")):
        return None
    head = row.get("listen_host") or CFG.head.name
    probe = probes.get(head) or {}
    listed = probe.get("containers")
    if listed is None or (ident.get("container") and ident["container"] in listed):
        return None
    if not _served_answers(row, probe):
        return None
    owned = _sparksi_containers(list(runs or []))
    sides: dict[str, list[str]] = {}
    for host, other in probes.items():
        if host != head and row.get("placement") != "pair":
            continue
        found = _mounting(other, local_path(row, host) or "")
        if not found:
            continue
        found = [n for n in found if not n.startswith("sparksi-") and (host, n) not in owned]
        if found:
            sides[host] = found
    if head not in sides:
        return None
    held = [head] + sorted(h for h in sides if h != head)
    if row.get("placement") == "pair" and CFG.has_worker:
        # A pair holds the worker too, even when that Spark's probe could not say so.
        for name in (CFG.head.name, CFG.worker.name):
            if name not in held:
                held.append(name)
    return {
        "hosts": held,
        "containers": sides,
        "port": row.get("port"),
        "note": EXTERNAL_NOTE,
    }


def external_refusal(row: dict[str, Any] | None) -> str | None:
    """The refusal for any stop, restart or replace of a row an outside engine serves."""
    if not row or not row.get("external"):
        return None
    return EXTERNAL_REFUSAL.format(title=row.get("title") or row.get("id") or "This model")


def _worker_ok(row: dict[str, Any], probe: dict[str, Any] | None) -> bool:
    want = (row.get("identity") or {}).get("worker_container")
    if not want:
        return True
    containers = (probe or {}).get("containers")
    return containers is None or want in containers


def _serving(row: dict[str, Any], probes: dict[str, dict[str, Any]], head: str, head_ok: bool) -> bool | None:
    """Is any side of this row up? True, False, or None when the container lists are unknown.

    The head answering as this row counts. So does the row's own container running on the
    head (loading or not answering yet) and, for a pair, its worker container on another
    host: a half-up pair can still be unloaded. None lets a stop through rather than
    refusing it on a guess.
    """
    if head_ok:
        return True
    ident = row.get("identity") or {}
    sides = [(head, ident.get("container"))]
    if row.get("placement") == "pair":
        sides += [(host, ident.get("worker_container")) for host in probes if host != head]
    unknown = False
    for host, name in sides:
        if not name:
            continue
        listed = (probes.get(host) or {}).get("containers")
        if listed is None:
            unknown = True
        elif name in listed:
            return True
    return None if unknown else False


def catalog_serving(row: dict[str, Any], probes: dict[str, dict[str, Any]]) -> bool | None:
    """Is a side up under the row's catalog identity alone (a harness launch)? True, False or None.

    It ignores Sparksi runs; ``annotate`` adds those as ``serving``. None means a container
    list could not be read.
    """
    head = row.get("listen_host") or CFG.head.name
    head_ok, _holder = identity_check(row, probes.get(head))
    return _serving(row, probes, head, head_ok)


_ENDED = ("stopped", "cancelled")
_CLOSED = ("failed", "timed_out")


def _superseded_by(run: dict[str, Any], host: str, container: str, runs: list[dict[str, Any]]) -> str:
    """The id of a newer run that reuses this host and container name, or ""."""
    started = float(run.get("started") or 0)
    for other in runs:
        if not isinstance(other, dict) or other.get("id") == run.get("id"):
            continue
        if float(other.get("started") or 0) <= started:
            continue
        for node in other.get("nodes") or []:
            if isinstance(node, dict) and node.get("host") == host and node.get("container") == container:
                return str(other.get("id") or "")
    return ""


def run_sides(run: dict[str, Any], probes: dict[str, dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, str]:
    """Each side of a run against the real container lists.

    "stopped" (its stop answered ok), "superseded" (a newer run reuses the name on that
    host, so this run neither claims nor stops it), "up", "gone", or "unknown" (no name
    recorded, or the machine's container list could not be read).
    """
    stopped_sides = run.get("stopped_sides") or []
    out: dict[str, str] = {}
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        role = str(node.get("role") or "")
        host = str(node.get("host") or "")
        name = str(node.get("container") or "")
        if str(node.get("phase") or "") == "stopped" or role in stopped_sides:
            out[role] = "stopped"
        elif name and _superseded_by(run, host, name, runs):
            out[role] = "superseded"
        else:
            listed = (probes.get(host) or {}).get("containers")
            if not name or listed is None:
                out[role] = "unknown"
            else:
                out[role] = "up" if name in listed else "gone"
    return out


def run_liveness(run: dict[str, Any], probes: dict[str, dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Is this run live? A failed or timed_out run is live while any side is up or unknown.

    A stopped or cancelled run is not. A run still starting, loading, ready or stopping is
    live whatever the lists say: its container may not exist yet.
    """
    sides = run_sides(run, probes, runs)
    phase = str(run.get("phase") or "")
    if phase in _ENDED:
        return {"id": run.get("id"), "live": False, "up": False, "sides": sides}
    up = any(state == "up" for state in sides.values())
    live = any(state in ("up", "unknown") for state in sides.values()) if phase in _CLOSED else True
    return {"id": run.get("id"), "live": live, "up": up, "sides": sides}


def newest_run(runs: list[dict[str, Any]], row_id: str) -> dict[str, Any] | None:
    """The newest run of this model (the one its card shows), or None."""
    found = None
    for run in runs:
        if isinstance(run, dict) and run.get("model_id") == row_id:
            if found is None or float(run.get("started") or 0) >= float(found.get("started") or 0):
                found = run
    return found


def annotate(
    rows: list[dict[str, Any]], probes: dict[str, dict[str, Any]], runs: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Presence, ``serving`` and, when ``runs`` is given, ``run_live`` for the newest run of each row.

    ``serving`` is True when the catalog identity answers or when a side of the newest run is
    up in the container lists: a Sparksi-launched run uses its own container names, which the
    catalog identity of a harness launch does not know.
    """
    runs = list(runs or [])
    out = []
    for row in rows:
        item = dict(row)
        presence = {}
        holders: dict[str, str] = {}
        head = row.get("listen_host") or CFG.head.name
        head_ok, head_holder = identity_check(row, probes.get(head))
        if head_holder:
            holders[head] = head_holder
        # ``live`` also needs local weights, so Unload reads ``serving`` instead.
        item["serving"] = _serving(row, probes, head, head_ok)
        run = newest_run(runs, str(row.get("id") or "")) if runs else None
        if run is not None:
            item["run_live"] = run_liveness(run, probes, runs)
            if item["run_live"]["up"]:
                item["serving"] = True
        # Display only: an outside engine serves this row. It never sets live, so no
        # stop or switch plan treats it as Sparksi's to stop; external_refusal guards those.
        outside = None
        if not head_ok and not (item.get("run_live") or {}).get("up"):
            outside = external_engine(row, probes, runs)
        item["external"] = outside is not None
        if outside is not None:
            item["external_engine"] = outside
            item["external_note"] = EXTERNAL_NOTE
            item["serving"] = True
        for host, probe in probes.items():
            path = local_path(row, host)
            stat = ((probe or {}).get("paths") or {}).get(path or "", {})
            here = bool(stat.get("exists") and stat.get("nonempty"))
            if row.get("placement") == "pair":
                live = here and head_ok and (host == head or _worker_ok(row, probe))
            else:
                live = here and host == head and head_ok
            presence[host] = "live" if live else ("local" if here else "absent")
        item["on"] = presence
        item["live"] = any(v == "live" for v in presence.values())
        item["port_holder"] = holders
        item["archive_state"] = "archived" if item.get("archive_tar") else "none"
        if item.get("archive_flat"):
            flat = (probes.get(CFG.archive_node) or {}).get("flat") or {}
            item["archive_state"] = "archived" if flat.get("ok") else "incomplete"
        out.append(item)
    return out


def fleet(catalog: dict[str, Any], runner: Any | None = None) -> dict[str, Any]:
    """Probe both Sparks and return host cards plus library rows."""
    runner = runner or SshRunner(catalog)
    base = overlay_rows(catalog)
    hosts = host_ids(catalog)
    first_host = CFG.archive_node if CFG.archive_node in hosts else hosts[0]
    first = runner.invoke(first_host, _probe_request(catalog, first_host, base), timeout=45)
    releases = (first or {}).get("releases") or []
    rows = merge_rows(base, releases)
    probes = {}
    for host in hosts:
        if host == first_host:
            # Reuse the scan; fill any release paths the first probe did not stat.
            probes[host] = first
            missing = [p for p in (local_path(r, host) for r in rows) if p and p not in (first.get("paths") or {})]
            if missing:
                extra = runner.invoke(
                    host,
                    {"op": "probe", "ports": probe_ports(), "paths": missing, "http_paths": http_paths(rows)},
                    timeout=30,
                )
                paths = dict(first.get("paths") or {})
                paths.update(extra.get("paths") or {})
                probes[host] = dict(first)
                probes[host]["paths"] = paths
        else:
            probes[host] = runner.invoke(host, _probe_request(catalog, host, rows), timeout=30)
    from sparksi.recipes.runs import list_runs  # late: library stays importable without the run store

    rows = annotate(rows, probes, list_runs())
    cards = []
    for host in hosts:
        probe = probes.get(host) or {}
        spec = (catalog.get("hosts") or {}).get(host) or {}
        cards.append(_compat_mount(
            {
                "id": host,
                "ok": bool(probe.get("ok")),
                "error": probe.get("error"),
                "uma_gib": spec.get("uma_gib"),
                "mem_total_gib": probe.get("mem_total_gib"),
                "mem_available_gib": probe.get("mem_available_gib"),
                "disk_free_gib": probe.get("disk_free_gib"),
                "disk_total_gib": probe.get("disk_total_gib"),
                "disk_used_gib": probe.get("disk_used_gib"),
                "disk_used_pct": probe.get("disk_used_pct"),
                "mount_archive": archive_mounted(probe),
                "ips": probe.get("ips") or {},
                "listening": probe.get("listening") or {},
                "gpu": probe.get("gpu") or {},
            },
            archive_mounted(probe),
        ))
    return {
        "hosts": cards,
        "library": rows,
        "probes": probes,
        "storage": storage(catalog, probes),
        "machines": machines(probes),
    }


_TS_CACHE: dict[str, Any] = {"t": 0.0, "peers": None}


def _is_lan(addr: str) -> bool:
    """LAN endpoint: a configured prefix, or any private non-CGNAT IPv4."""
    prefixes = [str(p) for p in CFG.paths.get("lan_prefixes") or []]
    if prefixes:
        return any(addr.startswith(p) for p in prefixes)
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return ip.version == 4 and ip.is_private and ip not in ipaddress.ip_network("100.64.0.0/10")


def _tailscale_peers(ttl: float = 60.0) -> dict[str, dict[str, Any]]:
    """HostName -> {tailscale, lan} from this host's ``tailscale status --json`` (cached)."""
    import json as _json
    import subprocess
    import time

    now = time.monotonic()
    if _TS_CACHE["peers"] is not None and now - _TS_CACHE["t"] < ttl:
        return _TS_CACHE["peers"]
    peers: dict[str, dict[str, Any]] = {}
    try:
        raw = subprocess.run(["tailscale", "status", "--json"], capture_output=True, text=True, timeout=3).stdout
        data = _json.loads(raw or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        data = {}
    for node in list((data.get("Peer") or {}).values()) + [data.get("Self") or {}]:
        name = node.get("HostName")
        if not name:
            continue
        ts = next((ip for ip in node.get("TailscaleIPs") or [] if "." in ip), None)
        cur = str(node.get("CurAddr") or "")
        host_part = cur.rsplit(":", 1)[0] if cur else ""
        lan = host_part if _is_lan(host_part) else None
        peers[name] = {"tailscale": ts, "lan": lan, "online": bool(node.get("Online", True))}
    _TS_CACHE.update(t=now, peers=peers)
    return peers


def machines(probes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """LAN and Tailscale IPs, read live: nodes from their probe, the controller locally,
    and any ``[extra_machines] tailscale`` peers from tailscale status."""
    from sparksi.agent import _ips

    out: list[dict[str, Any]] = []
    for host, probe in probes.items():
        ips = (probe or {}).get("ips") or {}
        out.append({"id": host, "lan": ips.get("lan"), "tailscale": ips.get("tailscale"),
                    "direct": ips.get("direct") or [], "source": "probe"})
    local = _ips()
    out.append({"id": CFG.controller_name, "role": "sparksi controller", "lan": local.get("lan"),
                "tailscale": local.get("tailscale"), "direct": [], "source": "local"})
    extra = (CFG.raw.get("extra_machines") or {}).get("tailscale") or {}
    peers = _tailscale_peers() if extra else {}
    for ts_name, shown in extra.items():
        peer = peers.get(ts_name)
        if peer:
            out.append({"id": str(shown), "lan": peer.get("lan"), "tailscale": peer.get("tailscale"), "direct": [],
                        "online": peer.get("online"), "source": "tailscale status (LAN = direct endpoint)"})
    return out


def archive_default() -> dict[str, Any]:
    return {"host": CFG.archive_node, "label": str(CFG.archive.get("label") or "Archive")}


def storage(catalog: dict[str, Any], probes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Live storage rows: each node's root disk plus the archive store."""
    out: list[dict[str, Any]] = []
    for host, probe in probes.items():
        probe = probe or {}
        total = probe.get("disk_total_gib")
        free = probe.get("disk_free_gib")
        used = probe.get("disk_used_gib")
        out.append(
            {
                "id": f"{host}:/",
                "label": f"{host} local disk (/)",
                "kind": "local",
                "host": host,
                "ok": bool(probe.get("ok")) and total is not None,
                "total_gib": total,
                "used_gib": used,
                "free_gib": free,
                "used_pct": probe.get("disk_used_pct"),
            }
        )
    spec = {**archive_default(), **(catalog.get("archive_store") or {})}
    owner = str(spec["host"])
    store = (probes.get(owner) or {}).get("archive_store") or {}
    row = {
            "id": CFG.archive_id,
            "label": spec["label"],
            "kind": "archive",
            "host": owner,
            "ok": bool(store),
            "error": None if store else f"archive not mounted on {owner}",
            "path": store.get("path"),
            "device": store.get("device"),
            "fstype": store.get("fstype"),
            "total_gib": store.get("total_gib"),
            "used_gib": store.get("used_gib"),
            "free_gib": store.get("free_gib"),
            "used_pct": store.get("used_pct"),
            "total_bytes": store.get("total_bytes"),
            "used_bytes": store.get("used_bytes"),
            "free_bytes": store.get("free_bytes"),
            "avail_bytes": store.get("avail_bytes"),
            "mounted_on": {h: archive_mounted(p) for h, p in probes.items()},
    }
    legacy_id = str(CFG.compat.get("legacy_archive_id") or "")
    if legacy_id:
        # Compat shim: older readers matched the store by its previous id.
        row["legacy_id"] = legacy_id
    out.append(row)
    return out


def row_by_id(rows: list[dict[str, Any]], rid: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get("id") == rid:
            return row
    return None


def _cell(state: str, note: str, can_copy: bool = False, refill: bool = False) -> dict[str, Any]:
    return {"state": state, "note": note, "can_copy": can_copy, "refill": refill}


def _sync_cell(row: dict[str, Any], host: str, path: str, probe: dict[str, Any], runner: Any) -> dict[str, Any]:
    if not probe.get("ok"):
        return _cell("unchecked", f"could not reach {host}")
    on = (row.get("on") or {}).get(host) or "absent"
    archived = row.get("archive_state") == "archived"
    if row.get("archive_state") == "incomplete":
        return _cell("unchecked", "the storage copy is incomplete")
    if on == "absent":
        if not archived:
            return _cell("absent", "not in storage either")
        if not archive_mounted(probe):
            return _cell("missing", "not on this Spark; storage is not mounted here")
        return _cell("missing", "not on this Spark", can_copy=True)
    if not archived:
        return _cell("not_in_storage", "only on this Spark")
    if not row.get("archive_flat"):
        return _cell("present", "on this Spark; size not compared")
    listing = runner.invoke(
        host,
        {"op": "list_shards", "src": row["archive_flat"], "dest": path, "shards": int(row.get("shards") or 120), "exact": True},
        timeout=120,
    ) or {}
    if not listing.get("ok"):
        return _cell("unchecked", str(listing.get("error") or "could not list shards"))
    pending = list(listing.get("pending") or [])
    files_differ = list(listing.get("files_differ") or [])
    total = int(listing.get("total") or 0)
    if not total:
        return _cell("unchecked", "no shards found in storage")
    if not pending and not files_differ:
        return _cell("in_sync", f"all {total} shards match")
    if pending:
        note = f"{len(pending)} of {total} shards missing or a different size"
    else:
        note = ", ".join(files_differ) + " differ"
    if files_differ and pending:
        note += "; " + ", ".join(files_differ) + " differ"
    cell = _cell("differs", note, can_copy=True, refill=True)
    if on == "live":
        cell["can_copy"] = False
        cell["note"] += "; in use"
    return cell


def sync_report(catalog: dict[str, Any], runner: Any | None = None) -> dict[str, Any]:
    """Read-only compare of each Spark against storage. Copies nothing."""
    runner = runner or SshRunner(catalog)
    snap = fleet(catalog, runner)
    hosts = host_ids(catalog)
    out = []
    for row in snap.get("library") or []:
        cells: dict[str, dict[str, Any]] = {}
        for host in hosts:
            path = local_path(row, host)
            if not path:
                continue
            cells[host] = _sync_cell(row, host, path, (snap.get("probes") or {}).get(host) or {}, runner)
        # A row that is neither in storage nor on any Spark that answered is not part of the sync.
        if row.get("archive_state") == "none" and all(cell["state"] in ("absent", "unchecked") for cell in cells.values()):
            continue
        out.append({
            "id": row.get("id"),
            "title": row.get("title") or row.get("id"),
            "gib": row.get("gib"),
            "archive_state": row.get("archive_state"),
            "hosts": cells,
        })
    return {"hosts": hosts, "rows": out, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
