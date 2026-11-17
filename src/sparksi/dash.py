# SPDX-License-Identifier: Apache-2.0

"""Monitor card from Spark announce packets.

sparksi.monitor still shapes a pair into one span. This module does not open SSH.
A host with no packet for 10 seconds is "no announce". Both silent is unreachable.
"""

from __future__ import annotations

from typing import Any

from sparksi import protect
from sparksi.announce_rx import INBOX
from sparksi.config import CFG

_prev: dict[str, Any] = {}


def _fresh(peer: Any, stale_secs: float) -> bool:
    data, age = peer.snapshot()
    return data is not None and age is not None and age <= stale_secs


def _keep(rec: dict[str, Any], head_fresh: bool, worker_fresh: bool, extra_fresh: dict[str, bool] | None = None) -> bool:
    head_id = CFG.head.id
    worker_id = CFG.worker.id if CFG.has_worker else ""
    host = rec.get("host") or head_id
    span = rec.get("span") or []
    if not head_fresh and (host == head_id or head_id in span):
        return False
    if worker_id and not worker_fresh and host == worker_id:
        return False
    if extra_fresh and host in extra_fresh and not extra_fresh[host]:
        return False
    return True


def _llama_ports(peer: Any) -> dict[int, dict[str, Any]]:
    """Ports whose announce row came from llama-server, with their metrics."""
    data, _age = peer.snapshot()
    out: dict[int, dict[str, Any]] = {}
    for port, rec in ((data or {}).get("http") or {}).items():
        m = (rec or {}).get("metrics") or {}
        if "llama_process_start" in m:
            try:
                out[int(port)] = m
            except (TypeError, ValueError):
                continue
    return out


def _llama_row(rec: dict[str, Any], m: dict[str, Any]) -> None:
    """Live status for a llama-server row from what dgx-announce measured.

    llama-server has no vLLM counters, so sparksi.monitor reads it as idle.
    Running comes from /slots, tok/s is the Spark's own decode rate, KV is
    the running request's slot context fill (kv_kind "ctx"), and waiting is
    unknown (None) because /slots does not show the queue.
    """
    rec["engine"] = "llama-server"
    run = m.get("num_requests_running")
    rec["running"] = int(run) if run is not None else 0
    rec["waiting"] = None
    tps = m.get("llama_gen_tps")
    rec["tok_s"] = float(tps) if tps is not None else (0.0 if run is not None else None)
    ctx = m.get("llama_ctx_used")
    rec["kv_pct"] = None if ctx is None else float(ctx) * 100.0
    rec["kv_kind"] = "ctx"
    if m.get("llama_health_ok") == 0:
        rec["state"] = "DOWN"
    elif run is None:
        rec["state"] = "UP"
    elif rec["running"] and (rec["tok_s"] or 0.0) >= 1.0:
        rec["state"] = "GEN"
    elif rec["running"]:
        rec["state"] = "busy"
    else:
        rec["state"] = "idle"


def monitor_card(head: Any, worker: Any, extras: list[tuple[str, Any]] | None = None) -> dict[str, Any]:
    """Same /api/monitor shape as the SSH poll, from whatever is already in hand.

    ``worker`` is None on a one-Spark site; ``extras`` is ``[(node id, peer)]``
    for a third and fourth Spark.
    """
    try:
        from sparksi.monitor.__main__ import snapshot
        from sparksi.monitor.peer import STALE_SECS
    except Exception as exc:
        return {
            "ok": False,
            "error": f"sparksi.monitor is not importable ({exc})",
            "ts": "",
            "hosts": [],
            "loaded": [],
            "loaded_summary": {"count": 0, "running": 0, "waiting": 0, "mode": "auto"},
        }
    head_fresh = _fresh(head, STALE_SECS)
    worker_fresh = _fresh(worker, STALE_SECS) if worker is not None else False
    extra_fresh = {str(node_id): _fresh(peer, STALE_SECS) for node_id, peer in (extras or [])}
    data = snapshot(head, worker, _prev, extras=extras)
    data["loaded"] = [rec for rec in data.get("loaded") or [] if _keep(rec, head_fresh, worker_fresh, extra_fresh)]
    llama = {CFG.head.id: _llama_ports(head)}
    if worker is not None and CFG.has_worker:
        llama[CFG.worker.id] = _llama_ports(worker)
    for node_id, peer in extras or []:
        llama[str(node_id)] = _llama_ports(peer)
    for rec in data["loaded"]:
        if rec.get("span") or protect.is_video_kind(rec.get("kind")):
            continue
        m = llama.get(rec.get("host") or CFG.head.id, {}).get(rec.get("port"))
        if m:
            _llama_row(rec, m)
    summary = data.get("loaded_summary") or {}
    summary["count"] = len(data["loaded"])
    summary["running"] = sum(int(rec.get("running") or 0) for rec in data["loaded"])
    summary["waiting"] = sum(int(rec.get("waiting") or 0) for rec in data["loaded"])
    data["loaded_summary"] = summary
    # Unreachable only when every Spark is silent; one quiet Spark is "no announce".
    both = not head_fresh and not (worker_fresh if worker is not None else False) and not any(extra_fresh.values())
    for host in data.get("hosts") or []:
        if host.get("fresh"):
            continue
        if both:
            host["unreachable"] = True
            host["detail"] = "unreachable"
        else:
            host["unreachable"] = False
            host["detail"] = "no announce"
    data["ok"] = True
    return data


def current() -> dict[str, Any]:
    worker = INBOX.peer(CFG.worker.id) if CFG.has_worker else None
    extras = [(node.id, INBOX.peer(node.id)) for node in CFG.nodes[2:]]
    return monitor_card(INBOX.peer(CFG.head.id), worker, extras)
