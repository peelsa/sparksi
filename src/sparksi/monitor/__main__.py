# SPDX-License-Identifier: Apache-2.0

"""Entry: python3 -m sparksi.monitor  (or the sparksi-monitor console script).

1 Hz box-drawing dashboard. Read-only everywhere. Ctrl+C exits cleanly.
"""

import fcntl
import json
import os
import sys
import time
import urllib.error
import urllib.request

from sparksi import protect
from sparksi.config import CFG, CFG_ERRORS

from .models import decode_rates, discover_loaded, sample_key
from .peer import STALE_SECS, Peer
from .render import NOMINAL_GB, FAINT, bottom, c_to_f, divider, system_lines, text, top

# Every pane is an SSH poll. Destinations are the [[nodes]] ssh entries in config.
HEAD_SSH = os.environ.get("SPARKSI_MON_HEAD") or CFG.head.ssh
# "" on a one-Spark site: the worker pane is then an idle NoPeer.
WORKER_SSH = os.environ.get("SPARKSI_MON_WORKER") or (CFG.worker.ssh if CFG.has_worker else "")


class NoPeer:
    """Stand-in for a Spark this site does not have. Never polls anything."""

    host = ""

    def start(self):
        return None

    def stop(self):
        return None

    def snapshot(self):
        return None, None


def _worker_peer():
    return Peer(host=WORKER_SSH) if WORKER_SSH else NoPeer()


def _extra_peers():
    """(host id, Peer) for a third and fourth Spark."""
    return [(node.id, Peer(host=node.ssh)) for node in CFG.nodes[2:]]

_prev_map = {}


def _log_err():
    """Never dump tracebacks into the desktop widget."""
    try:
        path = os.path.expanduser("~/.cache/sparksi/monitor.err")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import traceback
        with open(path, "a") as ef:
            ef.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            traceback.print_exc(file=ef)
    except Exception:
        pass


def _host_tag(rec):
    return (rec.get("host") or CFG.head.id)[:4]


def _worker_line(rec):
    name = (rec.get("worker_name") or "worker")[:18]
    host = rec.get("worker_host") or (CFG.worker.id if CFG.has_worker else "")
    up = rec.get("worker_up")
    if up is None:
        return text(f"worker {host} {name} ?", FAINT)
    if up:
        return text(f"worker {host} {name} up")
    return text(f"worker {host} {name} down", FAINT)


def _activity(rec, dec):
    """idle / GEN / busy / SPLIT / UP from head counters and worker liveness."""
    if rec.get("span"):
        head_up = rec.get("head_up", True)
        worker_up = rec.get("worker_up")
        if worker_up is False and head_up:
            return "SPLIT"
        if worker_up and not head_up:
            return "SPLIT"
    m = rec.get("metrics")
    if not m:
        return "UP"
    run = int(m.get("num_requests_running", 0))
    wait = int(m.get("num_requests_waiting", 0))
    if run == 0 and wait == 0:
        return "idle"
    if dec >= 1.0:
        return "GEN"
    return "busy"


def _model_lines(rec, now, prev_map):
    lines = []
    title = rec.get("name") or f":{rec.get('port')}"
    if len(title) > 22:
        title = title[:21] + "…"
    role = (rec.get("role") or "svc")[:7]
    host = _host_tag(rec)

    if protect.is_video_kind(rec["kind"]):
        st = "VIDEO" if rec.get("mode") == "video" else "UP"
        lines.append(divider(title, st))
        lines.append(text(
            f"{role:<7s} {host:<4s} rss{rec.get('rss_gb') or 0.0:5.1f}G :{rec['port']}"
        ))
        return lines

    if rec.get("span") and not rec.get("head_up", True):
        members = "+".join(rec.get("span") or [])
        lines.append(divider(title, "SPLIT"))
        lines.append(text(f"span   {members:<8s} head down", FAINT))
        lines.append(_worker_line(rec))
        return lines

    m = rec.get("metrics")
    if not m:
        state = "SPLIT" if rec.get("span") and rec.get("worker_up") is False else "UP"
        lines.append(divider(title, state))
        sid = (rec.get("served") or "")[:12]
        if rec.get("span"):
            members = "+".join(rec.get("span") or [])
            lines.append(text(f"span   {members:<8s} :{rec['port']} {sid}", FAINT))
            lines.append(_worker_line(rec))
        else:
            lines.append(text(f"{role:<7s} {host:<4s} :{rec['port']} {sid}", FAINT))
        return lines

    dec, acc_pct = decode_rates(rec, now, prev_map)
    run = int(m.get("num_requests_running", 0))
    wait = int(m.get("num_requests_waiting", 0))
    kv = m.get("kv_cache_usage_perc", 0.0) * 100.0
    state = _activity(rec, dec)
    lines.append(divider(title, state))
    if rec.get("span"):
        members = "+".join(rec.get("span") or [])
        lines.append(text(
            f"span   {members:<8s} {dec:5.1f}t/s KV{kv:3.0f}% r{run} w{wait}"
        ))
        lines.append(_worker_line(rec))
    else:
        lines.append(text(
            f"{role:<7s} {host:<4s} {dec:5.1f}t/s KV{kv:3.0f}% r{run} w{wait}"
        ))
    draft_tot = m.get("spec_decode_num_draft_tokens_total", 0.0)
    if draft_tot > 0 or acc_pct is not None:
        dfl = f"{acc_pct:4.1f}% accept" if acc_pct is not None else "  -  (no data)"
        lines.append(text(f"DFlash    {dfl}"))
    return lines


def _view(peer):
    """(view dict with fresh flag, raw snapshot, age)."""
    data, age = peer.snapshot()
    fresh = data is not None and age is not None and age <= STALE_SECS
    view = dict(data) if data else {}
    view["fresh"] = fresh
    return view, data, age


def _host_pane(title, data, age):
    lines = []
    if data is not None and age is not None and age <= STALE_SECS:
        lines.append(divider(title, f"{age:0.0f}s"))
    else:
        lines.append(divider(f"{title} --", f"{age:0.0f}s" if age is not None else ""))
    lines.extend(_peer_body(data, age))
    return lines


def _peer_body(peer_data, age):
    """Content rows for one Spark (divider drawn by caller)."""
    if peer_data is None or age is None or age > STALE_SECS:
        return [text("peer unreachable", FAINT)]
    total, avail = peer_data["mem"] or (0.0, 0.0)
    used = total - avail
    pct = used / total * 100.0 if total > 0 else 0.0
    g = peer_data["gpu"] or [None, None, None]
    return system_lines(
        mem_used=used, mem_total=total, mem_pct=pct, mem_avail=avail,
        cpu_pct=peer_data["cpu_pct"] if peer_data["cpu_pct"] is not None else 0.0,
        cpu_mhz=peer_data["cpu_mhz"],
        gpu_util=g[0], gpu_temp=g[1], gpu_mhz=g[2],
        zones=peer_data["zones"],
    )


def _host_snapshot(title, data, age):
    fresh = data is not None and age is not None and age <= STALE_SECS
    row = {"id": title, "age": None if age is None else round(age, 2), "fresh": fresh}
    if not fresh:
        row["unreachable"] = True
        return row
    total, avail = data.get("mem") or (0.0, 0.0)
    used = total - avail
    pct = used / total * 100.0 if total else 0.0
    reserved = max(0.0, NOMINAL_GB - total)
    used128 = NOMINAL_GB - avail
    g = data.get("gpu") or [None, None, None]
    gpu_c = g[1] if len(g) > 1 else None
    zones = []
    for item in data.get("zones") or []:
        label, celsius = item[0], item[1]
        zones.append({"label": label, "c": celsius, "f": c_to_f(celsius)})
    soc_f, soc_zone = None, None
    if zones:
        hot = max(zones, key=lambda z: z["c"])
        soc_f = hot["f"]
        soc_zone = hot["label"]
    row.update({
        "unreachable": False,
        "mem_used_gib": used,
        "mem_total_gib": total,
        "mem_pct": pct,
        "mem_avail_gib": avail,
        "sys_used_gib": used128,
        "sys_pct": used128 / NOMINAL_GB * 100.0,
        "sys_nominal_gib": NOMINAL_GB,
        "reserved_gib": reserved,
        "cpu_pct": data.get("cpu_pct"),
        "cpu_mhz": data.get("cpu_mhz"),
        "gpu_pct": g[0] if g else None,
        "gpu_temp_c": gpu_c,
        "gpu_temp_f": None if gpu_c is None else c_to_f(gpu_c),
        "gpu_mhz": g[2] if len(g) > 2 else None,
        "zones": zones,
        "soc_temp_f": soc_f,
        "soc_zone": soc_zone,
    })
    if data.get("power_w") is not None:
        row["power_w"] = data.get("power_w")
    budget = data.get("budget") or {}
    consumers = list(budget.get("consumers") or [])
    row["consumers"] = consumers
    row["cache_gib"] = budget.get("cache_gib")
    row["model_gib"] = budget.get("model_gib")
    row["nonserve_gib"] = round(sum(c.get("gib") or 0 for c in consumers), 2)
    return row


def _loaded_row(rec, now, prev_map):
    dec, acc_pct = (None, None)
    if not protect.is_video_kind(rec.get("kind")) and rec.get("metrics"):
        dec, acc_pct = decode_rates(rec, now, prev_map)
    state = _activity(rec, dec or 0.0)
    if protect.is_video_kind(rec.get("kind")):
        state = "VIDEO" if rec.get("mode") == "video" else "UP"
    m = rec.get("metrics") or {}
    kv = m.get("kv_cache_usage_perc")
    return {
        "name": rec.get("name"),
        "role": rec.get("role"),
        "host": rec.get("host"),
        "port": rec.get("port"),
        "served": rec.get("served"),
        "kind": rec.get("kind"),
        "span": rec.get("span"),
        "container": rec.get("container"),
        "container_source": rec.get("container_source") or "",
        "listen": rec.get("listen") or "",
        "bind": rec.get("bind") or "",
        "route": rec.get("route") or "",
        "head_up": rec.get("head_up", True),
        "worker_host": rec.get("worker_host"),
        "worker_name": rec.get("worker_name"),
        "worker_up": rec.get("worker_up"),
        "state": state,
        "tok_s": dec,
        "kv_pct": None if kv is None else kv * 100.0,
        "running": int(m.get("num_requests_running", 0)) if m else 0,
        "waiting": int(m.get("num_requests_waiting", 0)) if m else 0,
        "dflash_accept": acc_pct,
        "rss_gb": rec.get("rss_gb"),
        "mode": rec.get("mode"),
    }


def _attach_endpoints(rows, head_view, worker_view, extra_views=None):
    """Publish a client URL only after client_endpoint accepts the address."""
    from sparksi.monitor.models import client_endpoint

    views = dict(extra_views or {})
    for rec in rows:
        host = rec.get("host") or CFG.head.id
        spanned = bool(rec.get("span"))
        if spanned or host == CFG.head.id:
            node = CFG.head
            view = head_view
        elif host in views:
            node = next((item for item in CFG.nodes if item.id == host), CFG.head)
            view = views[host]
        elif CFG.has_worker:
            node = CFG.worker
            view = worker_view
        else:
            node = CFG.head
            view = head_view
        rec["announce_from"] = str((view or {}).get("announce_from") or "")
        rec["endpoint"] = client_endpoint(
            str(rec.get("listen") or ""),
            rec.get("port"),
            str(rec.get("bind") or ""),
            str(rec.get("route") or ""),
            node.tailscale,
        )


def snapshot(head, worker, prev_map=None, now=None, extras=None):
    """Numbers the TUI draws. prev_map holds decode-rate samples.

    ``worker`` may be None on a one-Spark site. ``extras`` is
    ``[(host_id, peer), ...]`` for a third and fourth Spark.
    """
    prev_map = _prev_map if prev_map is None else prev_map
    now = time.monotonic() if now is None else now
    head_v, head_data, head_age = _view(head)
    worker_v, worker_data, worker_age = _view(worker or NoPeer())
    extra_seen = [(str(host_id), _view(peer or NoPeer())) for host_id, peer in (extras or [])]
    loaded = discover_loaded(head_v, worker_v, [(host_id, seen[0]) for host_id, seen in extra_seen])
    live_keys = {sample_key(x) for x in loaded}
    for key in list(prev_map.keys()):
        if key not in live_keys:
            prev_map.pop(key, None)
    rows = [_loaded_row(rec, now, prev_map) for rec in loaded]
    _attach_endpoints(rows, head_v, worker_v, {host_id: seen[0] for host_id, seen in extra_seen})
    mode = (head_v.get("video_service") or {}).get("mode") or "auto"
    run_tot = sum(r["running"] for r in rows)
    wait_tot = sum(r["waiting"] for r in rows)
    return {
        "ts": time.strftime("%H:%M:%S"),
        "hosts": [
            _host_snapshot(CFG.head.id, head_data, head_age),
            *([_host_snapshot(CFG.worker.id, worker_data, worker_age)] if CFG.has_worker else []),
            *[_host_snapshot(host_id, seen[1], seen[2]) for host_id, seen in extra_seen],
        ],
        "loaded": rows,
        "loaded_summary": {
            "count": len(rows),
            "running": run_tot,
            "waiting": wait_tot,
            "mode": mode,
        },
    }


def _lines_host(host):
    age = host.get("age")
    age_s = f"{age:0.0f}s" if age is not None else ""
    if not host.get("fresh"):
        return [divider(f"{host['id']} --", age_s), text("peer unreachable", FAINT)]
    lines = [divider(host["id"], age_s)]
    lines.extend(system_lines(
        mem_used=host["mem_used_gib"],
        mem_total=host["mem_total_gib"],
        mem_pct=host["mem_pct"],
        mem_avail=host["mem_avail_gib"],
        cpu_pct=host["cpu_pct"] if host["cpu_pct"] is not None else 0.0,
        cpu_mhz=host["cpu_mhz"],
        gpu_util=host["gpu_pct"],
        gpu_temp=host["gpu_temp_c"],
        gpu_mhz=host["gpu_mhz"],
        zones=[(z["label"], z["c"]) for z in host.get("zones") or []],
    ))
    return lines


def _lines_loaded(rec):
    """Same strings as the previous _model_lines, from a snapshot row."""
    lines = []
    title = rec.get("name") or f":{rec.get('port')}"
    if len(title) > 22:
        title = title[:21] + "…"
    role = (rec.get("role") or "svc")[:7]
    host = (rec.get("host") or CFG.head.id)[:4]
    state = rec.get("state") or "UP"
    if protect.is_video_kind(rec.get("kind")):
        lines.append(divider(title, state))
        lines.append(text(
            f"{role:<7s} {host:<4s} rss{rec.get('rss_gb') or 0.0:5.1f}G :{rec['port']}"
        ))
        return lines
    if rec.get("span") and not rec.get("head_up", True):
        members = "+".join(rec.get("span") or [])
        lines.append(divider(title, "SPLIT"))
        lines.append(text(f"span   {members:<8s} head down", FAINT))
        lines.append(_worker_line(rec))
        return lines
    if rec.get("tok_s") is None:
        lines.append(divider(title, state))
        sid = (rec.get("served") or "")[:12]
        if rec.get("span"):
            members = "+".join(rec.get("span") or [])
            lines.append(text(f"span   {members:<8s} :{rec['port']} {sid}", FAINT))
            lines.append(_worker_line(rec))
        else:
            lines.append(text(f"{role:<7s} {host:<4s} :{rec['port']} {sid}", FAINT))
        return lines
    dec = rec.get("tok_s") or 0.0
    kv = rec.get("kv_pct") or 0.0
    run = rec.get("running") or 0
    wait = rec.get("waiting") or 0
    lines.append(divider(title, state))
    if rec.get("span"):
        members = "+".join(rec.get("span") or [])
        lines.append(text(
            f"span   {members:<8s} {dec:5.1f}t/s KV{kv:3.0f}% r{run} w{wait}"
        ))
        lines.append(_worker_line(rec))
    else:
        lines.append(text(
            f"{role:<7s} {host:<4s} {dec:5.1f}t/s KV{kv:3.0f}% r{run} w{wait}"
        ))
    acc = rec.get("dflash_accept")
    if acc is not None:
        lines.append(text(f"DFlash    {acc:4.1f}% accept"))
    return lines


def frame_from_snapshot(snap):
    lines = [top("DGX SPARK", snap.get("ts") or "")]
    for host in snap.get("hosts") or []:
        lines.extend(_lines_host(host))
    summary = snap.get("loaded_summary") or {}
    mode_tag = summary.get("mode") or "auto"
    loaded = snap.get("loaded") or []
    if not loaded:
        lines.append(divider("LOADED", mode_tag))
        lines.append(text("none (trio/video idle)", FAINT))
    else:
        for rec in loaded:
            lines.extend(_lines_loaded(rec))
    lines.append(divider(
        "LOADED",
        f"{summary.get('count', len(loaded))} in mem · {summary.get('running', 0)}r/{summary.get('waiting', 0)}w · {mode_tag}",
    ))
    lines.append(bottom())
    return lines


def build_frame(head, worker, extras=None):
    return frame_from_snapshot(snapshot(head, worker, _prev_map, extras=extras))


def _monitor_api():
    """The page's poller. None when dgx-mem is down, so the window may fall back."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:7500/api/monitor", timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError, TimeoutError, ValueError):
        return None
    if not isinstance(data, dict) or not data.get("ok") or not data.get("hosts"):
        return None
    return data


def _hold_instance():
    """One monitor window. A second process exits instead of opening more sockets."""
    path = os.path.expanduser("~/.local/state/dgx-monitor/instance.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def _print_snapshot() -> None:
    snap = _monitor_api()
    if snap is not None:
        json.dump(snap, sys.stdout)
        sys.stdout.write("\n")
        return
    head = Peer(host=HEAD_SSH)
    worker = _worker_peer()
    extras = _extra_peers()
    everyone = [head, worker, *[peer for _id, peer in extras]]
    for peer in everyone:
        peer.start()
    polled = [head] + ([worker] if WORKER_SSH else []) + [peer for _id, peer in extras]
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if all(peer.snapshot()[0] for peer in polled):
            break
        time.sleep(0.4)
    try:
        json.dump(snapshot(head, worker, {}, extras=extras), sys.stdout)
        sys.stdout.write("\n")
    finally:
        for peer in everyone:
            peer.stop()


def main() -> int | None:
    if CFG_ERRORS:
        # The import kept the built-in example so it could finish. Do not poll it.
        sys.stderr.write("\n".join(CFG_ERRORS) + "\n")
        return 1
    if "--snapshot" in sys.argv:
        _print_snapshot()
        return
    held = _hold_instance()
    if held is None:
        sys.stderr.write("DGX Spark Monitor is already running.\n")
        return
    head = None
    worker = None
    extras = []
    hide_cursor = sys.stdout.isatty()
    try:
        if hide_cursor:
            sys.stdout.write("\033[?25l")
        while True:
            try:
                snap = _monitor_api()
                if snap is not None:
                    if head is not None:
                        head.stop()
                        worker.stop()
                        for _id, peer in extras:
                            peer.stop()
                        head = None
                        worker = None
                        extras = []
                    lines = frame_from_snapshot(snap)
                else:
                    if head is None:
                        head = Peer(host=HEAD_SSH)
                        worker = _worker_peer()
                        extras = _extra_peers()
                        head.start()
                        worker.start()
                        for _id, peer in extras:
                            peer.start()
                    lines = build_frame(head, worker, extras)
                sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\n")
                sys.stdout.flush()
            except Exception:
                _log_err()
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if head is not None:
            head.stop()
            worker.stop()
            for _id, peer in extras:
                peer.stop()
        if hide_cursor:
            sys.stdout.write("\033[?25h")
            sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())
