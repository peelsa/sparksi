# SPDX-License-Identifier: Apache-2.0

"""Live-model discovery: HTTP liveness + vLLM metrics + the video service. Read-only.

Each host contributes its own answering ports. A span is one logical model
whose API is on the head and whose worker container is headless on the peer.
Two hosts that both answer HTTP stay two rows.
"""

import json
import os
import re
import subprocess
import time
import urllib.request

from sparksi import compat, protect
from sparksi.config import CFG

from .truth import SERVED_DISPLAY, models_from_text

TP_RE = re.compile(r"TP=\d+", re.IGNORECASE)
# Node ids (config [[nodes]] id): first is the head, second the pair worker.
HEAD_HOST = CFG.head.id
# "" on a one-Spark site: there is no worker, so nothing can span.
PEER_HOST = CFG.worker.id if CFG.has_worker else ""

VIDEO_MODE_FILE = os.path.join(compat.video_service_root(CFG.path("video_service_root")), "state", "mode")

WANTED_METRICS = {
    "generation_tokens_total", "prompt_tokens_total",
    "num_requests_running", "num_requests_waiting",
    "kv_cache_usage_perc", "prefix_cache_hits_total",
    "prefix_cache_queries_total", "num_preemptions_total",
    "request_success_total",
    "spec_decode_num_draft_tokens_total",
    "spec_decode_num_accepted_tokens_total",
}


def parse_metrics_text(text):
    """Sum Prometheus text by base metric name. Empty input → {}.

    vLLM prefixes names with ``vllm:``. Lines that are not a wanted metric
    are ignored. A listening port with no matching samples returns {}.
    """
    agg = {}
    if not text:
        return agg
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        key, val = parts
        base = key.split("{", 1)[0]
        if base.startswith("vllm:"):
            base = base[5:]
        if base not in WANTED_METRICS:
            continue
        try:
            agg[base] = agg.get(base, 0.0) + float(val)
        except ValueError:
            continue
    return agg


def served_from_models_json(text):
    """First served id in an OpenAI /v1/models body, or None."""
    if not text or not str(text).strip():
        return None
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None
    items = data.get("data") or []
    if not items or not isinstance(items, list):
        return None
    mid = items[0].get("id") if isinstance(items[0], dict) else None
    return str(mid) if mid else None


def read_vllm_metrics(port):
    """vLLM Prometheus text summed by base metric name, or None if down."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/metrics", timeout=1.5
        ) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    agg = parse_metrics_text(text)
    return agg or None


def read_served_model(port):
    """Served model id from /v1/models, or None."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/v1/models", timeout=1.0
        ) as resp:
            text = resp.read().decode("utf-8", "replace")
    except Exception:
        return None
    return served_from_models_json(text)


def read_video_mode():
    try:
        with open(VIDEO_MODE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip().lower() or None
    except OSError:
        return None


def video_service_from_peer(peer):
    """Video service row from the head dump (mode file, RSS, port 8188)."""
    if not protect.video_on():
        return None
    info = (peer or {}).get("video_service") or {}
    rss_kb = int(info.get("rss_kb") or 0)
    mode = info.get("mode")
    if not info.get("listening") and rss_kb <= 0:
        return None
    label = "Video service" if mode == "video" else "Video service (not in video mode)"
    return {
        "kind": compat.VIDEO_SERVICE_KIND,
        "video_service": True,
        "legacy_kind": compat.LEGACY_VIDEO_SERVICE_KIND,
        compat.LEGACY_ROW_FLAG: True,
        "role": "video",
        "port": protect.video_port(),
        "name": label,
        "rss_gb": rss_kb / (1024 ** 2) if rss_kb else 0.0,
        "mode": mode,
        "host": HEAD_HOST,
        "head_up": True,
    }


def model_title(mdl, served_id=None):
    if served_id:
        key = served_id.strip().lower()
        if key in SERVED_DISPLAY:
            return SERVED_DISPLAY[key]
        title = served_id.strip()
    else:
        title = (mdl or {}).get("name") or f":{(mdl or {}).get('port', '?')}"
    if len(title) > 22:
        title = title[:21] + "…"
    return title


def sample_key(rec):
    """Delta-map key. Host is part of it so two :8014s do not share a counter."""
    return f"{rec.get('host') or HEAD_HOST}:{rec.get('port')}"


def _span_declared(meta, title):
    display = (meta or {}).get("display") or ""
    if TP_RE.search(display):
        return True
    return (title or "").upper().endswith("TP2")


def _container_up(containers, name):
    """True when docker ps shows name in state Up. Missing or Exited → False."""
    if not name:
        return False
    want = str(name).strip()
    for c in containers or []:
        cname = (c.get("name") or "").strip()
        if cname != want and not cname.endswith("/" + want):
            continue
        return (c.get("status") or "").startswith("Up")
    return False


def _vllm_record(meta, port, served, metrics, host):
    title = model_title(meta, served)
    container = (meta or {}).get("container")
    return {
        "kind": "vllm" if metrics is not None else "openai",
        "role": (meta or {}).get("role") or ("proxy" if port == 8001 else "svc"),
        "port": port,
        "name": title,
        "served": served,
        "metrics": metrics,
        "host": host,
        "head_up": True,
        "container": container,
        # Truth file is keyed by port. It is not the process that answered.
        "container_source": "truth" if container else "",
    }


_ENGINE_SUFFIXES = ("-head", "-worker", "-rank0", "-rank1")
_engine_cache = {"t": 0.0, "rows": None}


def _stem(name):
    for suf in _ENGINE_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _guess_worker_name(local_name):
    if local_name.endswith("-head"):
        return _stem(local_name) + "-worker"
    if local_name.endswith("-rank0"):
        return _stem(local_name) + "-rank1"
    return local_name


def _is_multi(eng):
    """Container claims a multi-spark job: TP>=2, or a -head / -rank0 name."""
    try:
        tp = int(eng.get("tp") or 0)
    except (TypeError, ValueError):
        tp = 0
    name = eng.get("name") or ""
    return tp >= 2 or name.endswith("-head") or name.endswith("-rank0")


def _partner(local_name, containers):
    """Peer container that is the other half of this engine, if it is Up."""
    stem = _stem(local_name)
    headed = local_name.endswith("-head") or local_name.endswith("-rank0")
    for c in containers or []:
        if not (c.get("status") or "").startswith("Up"):
            continue
        pname = c.get("name") or ""
        if pname == local_name:
            return c
        if _stem(pname) == stem and (
            pname.endswith("-worker") or pname.endswith("-rank1") or headed
        ):
            return c
    return None


def _engine_port_tp(name):
    try:
        env = subprocess.run(
            ["docker", "inspect", "--format",
             "{{range .Config.Env}}{{println .}}{{end}}", name],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None, None
    port = tp = None
    for line in env.splitlines():
        if line.startswith("PORT="):
            try:
                port = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("TP=") or line.startswith("TENSOR_PARALLEL_SIZE="):
            try:
                tp = int(line.split("=", 1)[1].strip())
            except ValueError:
                pass
    return port, tp


def _scan_local_engines():
    """Up containers on this host, with PORT and TP from their env. Read-only."""
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=2,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    rows = []
    for line in out.splitlines():
        name, _, status = line.strip().partition("\t")
        if not name or not status.startswith("Up"):
            continue
        port, tp = _engine_port_tp(name)
        rows.append({"name": name, "status": status, "port": port, "tp": tp})
    return rows


def read_local_engines():
    """Cached scan. The 1 s frame must not inspect docker on every tick."""
    now = time.monotonic()
    if _engine_cache["rows"] is not None and now - _engine_cache["t"] < 3.0:
        return _engine_cache["rows"]
    rows = _scan_local_engines()
    _engine_cache["t"] = now
    _engine_cache["rows"] = rows
    return rows


def _apply_engine_span(rec, engines, containers, fresh, worker_answers):
    """Pair a live head container with its worker by PORT and name stem.

    `pair-exl3-head` + `pair-exl3-worker`, or the same name on both hosts
    (`vllm-fn`). A stale peer poll leaves worker_up unknown.
    """
    if worker_answers or rec.get("port") == 8001 or not PEER_HOST:
        return rec
    for eng in engines or []:
        if eng.get("port") != rec.get("port") or not _is_multi(eng):
            continue
        partner = _partner(eng["name"], containers) if fresh else None
        rec["span"] = [HEAD_HOST, PEER_HOST]
        rec["worker_host"] = PEER_HOST
        rec["container"] = eng["name"]
        rec["container_source"] = "engine"
        if not fresh:
            rec["worker_up"] = None
            rec["worker_name"] = _guess_worker_name(eng["name"])
        elif partner:
            rec["worker_up"] = True
            rec["worker_name"] = partner["name"]
        else:
            rec["worker_up"] = False
            rec["worker_name"] = _guess_worker_name(eng["name"])
        return rec
    return rec


def _apply_span(rec, meta, containers, fresh, worker_answers):
    """Attach span fields when this head row owns a headless peer worker.

    A peer that also answers HTTP on the same port is an independent model
    and is left alone. Proxy :8001 is never a span. When the peer poll is
    stale, a declared span still groups, with worker_up left unknown.
    """
    if rec.get("port") == 8001 or protect.is_video_kind(rec.get("kind")) or not PEER_HOST:
        return rec
    if worker_answers:
        return rec
    container = (meta or {}).get("container") or rec.get("container")
    declared = _span_declared(meta, rec.get("name"))
    worker_up = _container_up(containers, container) if fresh else None
    if not declared and not worker_up:
        return rec
    rec["span"] = [HEAD_HOST, PEER_HOST]
    rec["worker_up"] = worker_up
    rec["worker_name"] = container or "worker"
    rec["worker_host"] = PEER_HOST
    return rec


def _http_rows(http, by_port, host):
    rows = []
    for port in sorted(http or {}):
        if protect.is_off_limits(port):
            continue
        info = http[port] or {}
        served = info.get("served")
        metrics = info.get("metrics")
        if not served and not metrics:
            continue
        meta = by_port.get(port, {})
        rec = _vllm_record(meta, port, served, metrics, host)
        listen = info.get("listen")
        if isinstance(listen, str) and listen:
            rec["listen"] = listen
        if info.get("bind"):
            rec["bind"] = str(info.get("bind"))
        if info.get("route"):
            rec["route"] = str(info.get("route"))
        rows.append(rec)
    return rows


_WILD_LISTEN = {"", "0.0.0.0", "::", "*", "127.0.0.1", "::1", "localhost"}


def client_endpoint(listen: str, port, bind: str = "", route: str = "", configured: str = "") -> str:
    """Client URL only for an address a probe reached.

    A specific bind uses the dialed address. A wildcard bind uses the
    configured client address only when this sample's route is that same
    address. Loopback, a missing route, and a route that is not the
    configured address publish nothing. The packet source is not an address.
    """
    try:
        number = int(port)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    kind = str(bind or "")
    if kind == "loopback":
        return ""
    if kind == "wildcard":
        advertised = str(route or "").strip()
        want = str(configured or "").strip()
        if not want or advertised != want or not _endpoint_host(advertised):
            return ""
        return _endpoint_url(advertised, number)
    bound = str(listen or "")
    if not bound or bound in _WILD_LISTEN or not _endpoint_host(bound):
        return ""
    return _endpoint_url(bound, number)


def _endpoint_host(host: str) -> bool:
    if not host or host in _WILD_LISTEN:
        return False
    return all(ch.isalnum() or ch in ".:-" for ch in host)


def _endpoint_url(host: str, port: int) -> str:
    shown = f"[{host}]" if ":" in host else host
    return f"http://{shown}:{port}/v1"


def discover_loaded(head=None, worker=None, extras=None):
    """Loaded/serving records from the per-Spark snapshots.

    Each argument is a peer dump (`http`, `containers`, `engines`, `video_service`,
    `truth`, `fresh`). Nothing is probed on the machine running the TUI.
    A vLLM row requires HTTP on that host. Docker names mark a headless
    worker. ``extras`` is ``[(host_id, dump), ...]`` for a third and fourth
    Spark; their models are listed as single-Spark rows. Order: spans, then
    head, then worker, then the extra Sparks in config order.
    """
    head = head or {}
    worker = (worker or {}) if PEER_HOST else {}
    worker_fresh = bool(worker.get("fresh"))
    containers = worker.get("containers") or [] if worker_fresh else []
    worker_http = worker.get("http") or {} if worker_fresh else {}
    engines = head.get("engines") or []

    known, _ports = models_from_text(head.get("truth") or "")
    by_port = {m["port"]: m for m in known}
    loaded = []
    seen = set()

    video = video_service_from_peer(head)
    if video:
        loaded.append(video)
        seen.add(protect.video_port())

    for rec in _http_rows(head.get("http") or {}, by_port, HEAD_HOST):
        if rec["port"] in seen:
            continue
        meta = by_port.get(rec["port"], {})
        worker_answers = rec["port"] in worker_http and (
            worker_http[rec["port"]].get("served") or worker_http[rec["port"]].get("metrics")
        )
        rec = _apply_span(rec, meta, containers, worker_fresh, bool(worker_answers))
        rec = _apply_engine_span(
            rec, engines, containers, worker_fresh, bool(worker_answers)
        )
        loaded.append(rec)
        seen.add(rec["port"])

    if worker_fresh:
        for meta in known:
            port = meta["port"]
            if port in seen or port == 8001:
                continue
            if not _span_declared(meta, meta.get("name")):
                continue
            if not _container_up(containers, meta.get("container")):
                continue
            loaded.append({
                "kind": "vllm",
                "role": meta.get("role") or "svc",
                "port": port,
                "name": model_title(meta, meta.get("served_name")),
                "served": meta.get("served_name"),
                "metrics": None,
                "host": HEAD_HOST,
                "head_up": False,
                "container": meta.get("container"),
                "container_source": "truth" if meta.get("container") else "",
                "span": [HEAD_HOST, PEER_HOST],
                "worker_up": True,
                "worker_name": meta.get("container") or "worker",
                "worker_host": PEER_HOST,
            })
            seen.add(port)

        for eng in engines:
            port = eng.get("port")
            if not port or port in seen or port == 8001 or not _is_multi(eng):
                continue
            partner = _partner(eng["name"], containers)
            if not partner:
                continue
            loaded.append({
                "kind": "vllm",
                "role": "svc",
                "port": port,
                "name": _stem(eng["name"]) or f":{port}",
                "served": None,
                "metrics": None,
                "host": HEAD_HOST,
                "head_up": False,
                "container": eng["name"],
                "container_source": "engine",
                "span": [HEAD_HOST, PEER_HOST],
                "worker_up": True,
                "worker_name": partner["name"],
                "worker_host": PEER_HOST,
            })
            seen.add(port)

        for rec in _http_rows(worker_http, by_port, PEER_HOST):
            if rec["port"] in seen and any(
                r.get("span") and r["port"] == rec["port"] for r in loaded
            ):
                continue
            loaded.append(rec)

    extra_order = {}
    for index, (host_id, dump) in enumerate(extras or []):
        extra_order[str(host_id)] = 3 + index
        dump = dump or {}
        if not dump.get("fresh"):
            continue
        for rec in _http_rows(dump.get("http") or {}, by_port, str(host_id)):
            loaded.append(rec)

    def _order(rec):
        if rec.get("span"):
            group = 0
        elif rec.get("host") in extra_order:
            group = extra_order[rec.get("host")]
        elif PEER_HOST and rec.get("host") == PEER_HOST:
            group = 2
        else:
            group = 1
        return (group, rec.get("port") or 0)

    loaded.sort(key=_order)
    return loaded


def decode_rates(rec, now, prev_map):
    """Instant decode tok/s + MTP accept % from counter deltas. Updates
    prev_map (host:port -> sample). Returns (dec_tps, acc_pct)."""
    m = rec.get("metrics") or {}
    gen = m.get("generation_tokens_total", 0.0)
    draft = m.get("spec_decode_num_draft_tokens_total", 0.0)
    acc = m.get("spec_decode_num_accepted_tokens_total", 0.0)
    dec, acc_pct = 0.0, None
    key = sample_key(rec)
    prev = prev_map.get(key)
    if prev is not None:
        dt = now - prev["t"]
        if dt > 0:
            dec = max(0.0, (gen - prev["gen"]) / dt)
        draft_d = draft - prev.get("draft", 0.0)
        acc_d = acc - prev.get("acc", 0.0)
        if draft_d > 0:
            acc_pct = 100.0 * acc_d / draft_d
    if acc_pct is None and draft > 0:
        acc_pct = 100.0 * acc / draft
    prev_map[key] = {"gen": gen, "draft": draft, "acc": acc, "t": now}
    return dec, acc_pct
