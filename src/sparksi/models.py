# SPDX-License-Identifier: Apache-2.0

"""Installed-model list and launch cards.

``list_models`` is the disk inventory the overview already reads.
``build_cards`` turns catalog library rows into launch cards: format, engines,
per-node files, and which placements fit. An optional card file under
``~/.config/sparksi/cards/<id>.toml`` overlays a row. This module does not
write the catalog or sparksi.toml.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparksi import compat, protect
from sparksi.catalog import load_catalog, load_yaml, model_inventory_path
from sparksi.config import ALL_ROLES, CFG
from sparksi.dryrun import cost_id
from sparksi.placement import is_multi, multis, needs_text, role_words, roles_of
from sparksi.probe import compute_apps, http_get, port_listening
from sparksi.tokens import fmt_count, report as token_report

# Optional disk inventory script on the node (see [paths] disk_inventory).
DGX_MODELS = Path(CFG.path("disk_inventory") or "/nonexistent/dgx_models.py")
SYS_PY = "/usr/bin/python3"
GiB = 1024**3
MODALITY_KEYS = ("audio", "image", "video")
_DISK_LOCK = threading.Lock()
_DISK_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_DISK_TTL_S = 45.0


def _bit(v: Any) -> int:
    if v in (1, True, "1", "yes", "true", "True"):
        return 1
    return 0


def infer_modalities(path: str | None) -> dict[str, int]:
    """Best-effort 0/1 from weights dir. Tokenizer audio tokens alone are not enough."""
    out = {k: 0 for k in MODALITY_KEYS}
    if not path:
        return out
    root = Path(path).expanduser()
    if not root.exists():
        return out
    names = {p.name.lower() for p in root.iterdir()} if root.is_dir() else {root.name.lower()}
    blob = ""
    if root.is_dir():
        for name in (
            "processor_config.json",
            "preprocessor_config.json",
            "video_preprocessor_config.json",
            "audio_preprocessor_config.json",
            "config.json",
        ):
            p = root / name
            if p.is_file():
                blob += " " + p.read_text(encoding="utf-8", errors="ignore").lower()
    else:
        blob = root.name.lower()
    if (
        "audio_processor" in blob
        or "audio_seq_length" in blob
        or "audio_ms_per_token" in blob
        or "audio_model.py" in names
        or "audio_preprocessor_config.json" in names
        or "audio_kwargs" in blob
    ):
        out["audio"] = 1
    if (
        "image_processor" in blob
        or "image_token_id" in blob
        or "image_processing.py" in names
        or "vision_config" in blob
        or "gemma4_vision" in blob
    ):
        out["image"] = 1
    if (
        "video_processor" in blob
        or "video_token_id" in blob
        or "video_processing.py" in names
        or "video_preprocessor_config.json" in names
        or "video_context_token" in blob
    ):
        out["video"] = 1
    return out


def resolve_modalities(
    declared: dict[str, Any] | None,
    path: str | None,
) -> dict[str, int]:
    inferred = infer_modalities(path)
    out = {k: inferred[k] for k in MODALITY_KEYS}
    if isinstance(declared, dict):
        for k in MODALITY_KEYS:
            if k in declared:
                out[k] = _bit(declared.get(k))
    return out


def _with_modalities(row: dict[str, Any], mods: dict[str, int]) -> dict[str, Any]:
    row.update(mods)
    row["modalities"] = {k: mods[k] for k in MODALITY_KEYS}
    row["aiv"] = " ".join(str(mods[k]) for k in MODALITY_KEYS)
    return row


def _infer_quant(path: str | None) -> str:
    p = (path or "").lower()
    if "nvfp4" in p or "fp4" in p:
        return "nvfp4"
    if "mxfp4" in p:
        return "mxfp4"
    if "fp8" in p:
        return "fp8"
    if "int8" in p:
        return "int8"
    if "gguf" in p or p.endswith(".gguf"):
        return "gguf"
    if "bf16" in p:
        return "bf16"
    if "fp16" in p:
        return "fp16"
    return ""


def gpu_size_parts(
    disk_gib: float | None,
    gpu_gib: float | None,
    kv_frac: float | None = None,
) -> dict[str, Any]:
    """Size column: loaded weights + KV/runtime = nvidia-smi GPU total.

    When not resident, show disk only. When resident, KV is the remainder of
    process GPU memory after weights so the three numbers add up.
    """
    if gpu_gib is None:
        return {
            "weight_gib": disk_gib,
            "kv_gib": None,
            "gpu_gib": None,
            "kv_frac": kv_frac,
            "size_label": f"{disk_gib:.1f} GiB" if disk_gib is not None else "—",
        }
    g = round(float(gpu_gib), 1)
    if disk_gib is not None and disk_gib >= 0:
        w = round(min(float(disk_gib), g), 1)
        kv = round(g - w, 1)
    elif kv_frac is not None:
        kv = round(max(0.0, min(g, g * float(kv_frac))), 1)
        w = round(g - kv, 1)
    else:
        w, kv = g, 0.0
    return {
        "weight_gib": w,
        "kv_gib": kv,
        "gpu_gib": g,
        "kv_frac": kv_frac,
        "size_label": f"{w:.1f} + {kv:.1f} = {g:.1f} GiB",
    }


def _kv_frac(port: int | None) -> float | None:
    if not port:
        return None
    try:
        if not port_listening(int(port), "127.0.0.1"):
            return None
    except Exception:
        return None
    st, body = http_get(f"http://127.0.0.1:{int(port)}/metrics")
    if not body:
        return None
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        if "kv_cache_usage_perc" not in line or "{" not in line:
            continue
        try:
            val = float(line.rsplit(None, 1)[-1])
        except ValueError:
            continue
        if val > 1.5:
            val = val / 100.0
        return max(0.0, min(1.0, val))
    return None


def _attach_gpu_sizes(rows: list[dict[str, Any]], catalog: dict[str, Any] | None = None) -> None:
    apps, _ = compute_apps()
    by_port: dict[int, dict[str, Any]] = {}
    by_name: dict[str, dict[str, Any]] = {}
    for a in apps or []:
        if a.get("port") is not None:
            by_port[int(a["port"])] = a
        if a.get("served_model"):
            by_name[str(a["served_model"])] = a
    for row in rows:
        port = row.get("port")
        app = None
        if port is not None:
            app = by_port.get(int(port))
        if app is None and row.get("served_name"):
            app = by_name.get(str(row["served_name"]))
        gpu = app.get("used_gpu_mem_gib") if app else None
        kv_frac = _kv_frac(port) if gpu is not None and port else None
        parts = gpu_size_parts(row.get("size_gib"), gpu, kv_frac)
        if gpu is None:
            cid = row.get("catalog_id") or row.get("id")
            planned = cost_id(catalog or {}, str(cid)) if cid else None
            if planned is not None:
                parts["gpu_gib"] = round(float(planned), 1)
        row.update(parts)


def _dir_size_gib(path: str | None) -> float | None:
    if not path:
        return None
    p = Path(path).expanduser()
    if not p.exists():
        return None
    try:
        out = subprocess.check_output(["du", "-sb", str(p)], text=True, timeout=30)
        return round(int(out.split()[0]) / GiB, 2)
    except Exception:
        return None


def _dgx_models_fetch() -> list[dict[str, Any]]:
    if not DGX_MODELS.is_file():
        return []
    try:
        p = subprocess.run(
            [SYS_PY, str(DGX_MODELS), "--json", "--once"],
            capture_output=True,
            text=True,
            timeout=90,
        )
    except Exception:
        return []
    if p.returncode != 0 or not p.stdout.strip():
        return []
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        return []
    rows = data.get("models")
    return rows if isinstance(rows, list) else []


def _dgx_models_json() -> list[dict[str, Any]]:
    """Disk inventory. Cached — dgx_models.py --once is ~12s and must not gate unlock."""
    global _DISK_CACHE
    now = time.monotonic()
    with _DISK_LOCK:
        if _DISK_CACHE and (now - _DISK_CACHE[0]) < _DISK_TTL_S:
            return _DISK_CACHE[1]
        stale = _DISK_CACHE[1] if _DISK_CACHE else None
    rows = _dgx_models_fetch()
    if not rows and stale:
        rows = stale
    with _DISK_LOCK:
        _DISK_CACHE = (time.monotonic(), rows)
    return rows


def _truth_display() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        truth = load_yaml(model_inventory_path())
    except Exception:
        return out
    for m in truth.get("models") or []:
        name = m.get("served_name")
        disp = m.get("display")
        if name and disp:
            out[str(name)] = str(disp)
        w = m.get("weights")
        if w and disp:
            try:
                out[str(Path(w).expanduser().resolve())] = str(disp)
            except OSError:
                pass
    return out


def _match_disk(row: dict[str, Any], disk: list[dict[str, Any]]) -> dict[str, Any] | None:
    weights = row.get("weights") or row.get("path")
    served = row.get("served_name") or row.get("id")
    wres = None
    if weights:
        try:
            wres = str(Path(weights).expanduser().resolve())
        except OSError:
            wres = str(weights)
    for d in disk:
        dp = str(d.get("path") or "")
        if wres and dp:
            try:
                if str(Path(dp).expanduser().resolve()) == wres:
                    return d
            except OSError:
                if dp == wres:
                    return d
        if served and served in (
            d.get("served_name"),
            d.get("live_served"),
            d.get("id"),
        ):
            return d
        cid = row.get("id")
        if cid and cid in dp:
            return d
    return None


def _token_fields(row: dict[str, Any] | None) -> dict[str, Any]:
    if not row:
        return {
            "life_prompt": None,
            "life_gen": None,
            "life_requests": None,
            "engine_prompt": None,
            "engine_gen": None,
            "tok_label": "—",
        }
    return {
        "life_prompt": row.get("life_prompt"),
        "life_gen": row.get("life_gen"),
        "life_requests": row.get("life_requests"),
        "engine_prompt": row.get("engine_prompt"),
        "engine_gen": row.get("engine_gen"),
        "tok_label": (
            f"{fmt_count(row.get('life_gen'))} out · {fmt_count(row.get('life_prompt'))} in"
        ),
    }


def list_models(catalog: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cat = catalog or load_catalog()
    disk = _dgx_models_json()
    displays = _truth_display()
    ledger = token_report(cat, sample=False)
    out: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    for r in cat.get("residents") or []:
        if r.get("kind") == "proxy":
            continue
        cid = r.get("id")
        weights = r.get("weights")
        hit = _match_disk(r, disk)
        path = (hit or {}).get("path") or weights
        if path:
            seen_paths.add(str(path))
        quant = (hit or {}).get("quant") or _infer_quant(path)
        if quant in (None, "?"):
            quant = _infer_quant(path)
        size = (hit or {}).get("size_gib")
        if size is None and hit and hit.get("size_bytes"):
            size = round(int(hit["size_bytes"]) / GiB, 2)
        if size is None:
            size = _dir_size_gib(path)
        model = (
            displays.get(r.get("served_name") or "")
            or (displays.get(str(Path(path).resolve())) if path else None)
            or (hit or {}).get("id")
            or (Path(path).name if path else None)
            or cid
        )
        if protect.is_video_id(cid):
            model = r.get("display") or "Video service"
            if not path:
                path = str(Path(compat.video_service_root(CFG.path("video_service_root"))) / "models")
            if not quant:
                quant = "int8+nvfp4"
            if size is None:
                size = _dir_size_gib(path)
        mods = resolve_modalities(r.get("modalities"), path)
        out.append(
            _with_modalities(
                {
                    "id": cid,
                    "catalog_id": cid,
                    "model": model,
                    "path": path,
                    "role": r.get("role"),
                    "prod_role": r.get("role") if r.get("role") in ("brain", "vision", "counsel") else None,
                    "size_gib": size,
                    "quant": quant or "",
                    "model_type": (hit or {}).get("model_type") or r.get("kind"),
                    "ctx": (hit or {}).get("max_position_embeddings"),
                    "port": (hit or {}).get("live_port") or r.get("port"),
                    "served_name": r.get("served_name"),
                    "live": bool((hit or {}).get("live_port") or (hit or {}).get("live_served")),
                    "selectable": r.get("kind") in ("vllm", "llamacpp") or protect.is_video_kind(r.get("kind")),
                    "notes": r.get("notes") or "",
                    **(
                        _token_fields(
                            ledger.get(str(cid))
                            or {
                                "life_prompt": 0,
                                "life_gen": 0,
                                "life_requests": 0,
                                "engine_prompt": 0,
                                "engine_gen": 0,
                            }
                        )
                        if r.get("kind") == "vllm"
                        else _token_fields(None)
                    ),
                },
                mods,
            )
        )

    for d in disk:
        dp = str(d.get("path") or "")
        if dp and dp in seen_paths:
            continue
        if d.get("role") == "drafter":
            continue
        quant = d.get("quant") if d.get("quant") not in (None, "?") else _infer_quant(dp)
        size = d.get("size_gib")
        if size is None and d.get("size_bytes"):
            size = round(int(d["size_bytes"]) / GiB, 2)
        mods = resolve_modalities(d.get("modalities"), dp or None)
        out.append(
            _with_modalities(
                {
                    "id": d.get("id"),
                    "catalog_id": None,
                    "model": d.get("id") or Path(dp).name if dp else "",
                    "path": dp or None,
                    "role": d.get("role"),
                    "prod_role": d.get("prod_role"),
                    "size_gib": size,
                    "quant": quant or "",
                    "model_type": d.get("model_type"),
                    "ctx": d.get("max_position_embeddings"),
                    "port": d.get("live_port"),
                    "served_name": d.get("served_name") or d.get("live_served"),
                    "live": bool(d.get("live_port") or d.get("live_served")),
                    "selectable": False,
                    "notes": d.get("notes") or "",
                    **_token_fields(ledger.get(str(d.get("id") or ""))),
                },
                mods,
            )
        )
    _attach_gpu_sizes(out, cat)
    return out


# --- launch cards -------------------------------------------------------

FORMATS = ("nvfp4", "mxfp4", "fp8", "bf16", "gguf", "exl3")
TWO_NODE_ENGINES = ("vllm", "sglang", "trtllm")
KV_RESERVE_DEFAULT = 8.0
PLACEMENTS = ("head", "worker", "both")  # the two-Spark set; placement.all_placements() lists a site's own


@dataclass
class ModelCard:
    id: str
    title: str
    repo: str = ""
    revision: str = ""
    format: str = ""
    weights_gib: float = 0.0
    hints: dict[str, bool] = field(default_factory=lambda: {"moe": False, "mtp": False, "vision": False})
    engines: list[str] = field(default_factory=list)
    files: dict[str, str] = field(default_factory=dict)
    flags: dict[str, list[str]] = field(default_factory=lambda: {"forbid": [], "need": []})
    ctx: int | None = None
    max_ctx: int | None = None
    timeout_s: int | None = None
    presence: dict[str, bool | None] = field(default_factory=dict)
    placements: dict[str, dict[str, Any]] = field(default_factory=dict)
    scripts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    kv_reserve_gib: float = KV_RESERVE_DEFAULT
    catalog_port: int | None = None
    max_ctx: int | None = None
    service: bool = False
    protected: bool = False
    catalog_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Public card. Launch scripts stay off the wire."""
        return {
            "id": self.id,
            "title": self.title,
            "repo": self.repo,
            "revision": self.revision,
            "format": self.format,
            "weights_gib": self.weights_gib,
            "hints": dict(self.hints),
            "engines": list(self.engines),
            "files": dict(self.files),
            "flags": {"forbid": list(self.flags.get("forbid") or []), "need": list(self.flags.get("need") or [])},
            "ctx": self.ctx,
            "max_ctx": self.max_ctx,
            "timeout_s": self.timeout_s,
            "kv_reserve_gib": self.kv_reserve_gib,
            "presence": dict(self.presence),
            "placements": {key: dict(value) for key, value in self.placements.items()},
            "service": self.service,
            "protected": self.protected,
            "port": self.catalog_port,
        }

    def to_service(self) -> dict[str, Any]:
        """Service row for the Models page. Start and stop stay on the server."""
        return {
            "id": self.id,
            "title": self.title,
            "port": self.catalog_port,
            "service": True,
            "protected": self.protected,
        }


def cards_dir() -> Path:
    explicit = os.environ.get("SPARKSI_CARDS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return xdg / "sparksi" / "cards"


def _library_rows(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in catalog.get("library") or []:
        if isinstance(raw, dict) and raw.get("id"):
            rows.append(raw)
    return rows


def _blob(row: dict[str, Any]) -> str:
    parts = [
        str(row.get("id") or ""),
        str(row.get("title") or ""),
        str(row.get("notes") or ""),
        str(row.get("weights_note") or ""),
        str(row.get("kind") or ""),
    ]
    local = row.get("local") or {}
    if isinstance(local, dict):
        parts.extend(str(path) for path in local.values())
    launch = row.get("launch") or {}
    if isinstance(launch, dict):
        for key in ("start", "stop"):
            parts.extend(str(part) for part in launch.get(key) or [])
    return " ".join(parts).lower()


def infer_format(row: dict[str, Any]) -> str:
    declared = str(row.get("format") or "").lower()
    if declared in FORMATS:
        return declared
    blob = _blob(row)
    for fmt in ("nvfp4", "mxfp4", "exl3", "gguf", "fp8", "bf16"):
        if fmt in blob:
            return fmt
    return ""


def _tool_row(row: dict[str, Any]) -> bool:
    try:
        if protect.is_off_limits(int(row.get("port") or 0)):
            return True
    except (TypeError, ValueError):
        pass
    if protect.is_video_kind(row.get("kind")):
        return True
    ident = row.get("identity") or {}
    if isinstance(ident, dict) and ident.get("http") in ("/", "/system_stats") and not infer_format(row):
        return True
    return False


def infer_engines(row: dict[str, Any], fmt: str) -> list[str]:
    declared = row.get("engines")
    if isinstance(declared, list) and declared:
        return [str(item) for item in declared if str(item)]
    if _tool_row(row):
        return ["custom"]
    if fmt == "gguf":
        engines = ["llamacpp"]
    elif fmt in FORMATS or "vllm" in _blob(row):
        engines = ["vllm", "sglang", "trtllm"]
    else:
        engines = []
    start = ((row.get("launch") or {}).get("start")) or _any_seat(row, "start")
    if start and "custom" not in engines:
        engines.append("custom")
    if not engines:
        engines = ["custom"] if start else ["vllm"]
    return engines


def _any_seat(row: dict[str, Any], key: str) -> list[str]:
    seats = row.get("seats") or {}
    if not isinstance(seats, dict):
        return []
    for seat in seats.values():
        if isinstance(seat, dict) and seat.get(key):
            return [str(part) for part in seat[key]]
    return []


def infer_hints(row: dict[str, Any]) -> dict[str, bool]:
    blob = _blob(row)
    mods = row.get("modalities") if isinstance(row.get("modalities"), dict) else {}
    vision = bool(mods.get("image")) or "vision" in blob or str(row.get("role") or "") == "vision"
    return {
        "moe": "moe" in blob or "a22b" in blob or "a3b" in blob,
        "mtp": "mtp" in blob,
        "vision": vision,
    }


def _scripts(row: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Per-host start and stop argv copied off the catalog row."""
    out: dict[str, dict[str, list[str]]] = {}
    seats = row.get("seats") or {}
    hosts = set()
    local = row.get("local") or {}
    if isinstance(local, dict):
        hosts.update(str(host) for host in local)
    if isinstance(seats, dict):
        hosts.update(str(host) for host in seats)
    launch = row.get("launch") or {}
    launch_host = str(launch.get("host") or "") if isinstance(launch, dict) else ""
    if launch_host:
        hosts.add(launch_host)
    for host in hosts:
        seat = seats.get(host) if isinstance(seats, dict) else None
        start = list((seat or {}).get("start") or []) if isinstance(seat, dict) else []
        stop = list((seat or {}).get("stop") or []) if isinstance(seat, dict) else []
        if not start and isinstance(launch, dict) and (not launch_host or launch_host == host):
            start = list(launch.get("start") or [])
            stop = list(launch.get("stop") or stop)
        if start or stop:
            out[host] = {
                "start": [str(part) for part in start],
                "stop": [str(part) for part in stop],
            }
    return out


def _weights(row: dict[str, Any]) -> float:
    for key in ("weights_gib", "gib", "estimate_gib"):
        if row.get(key) not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _files(row: dict[str, Any]) -> dict[str, str]:
    local = row.get("local") or {}
    files: dict[str, str] = {}
    if isinstance(local, dict):
        for host, path in local.items():
            if path:
                files[str(host)] = str(path)
    return files


def _flags(row: dict[str, Any]) -> dict[str, list[str]]:
    raw = row.get("flags") if isinstance(row.get("flags"), dict) else {}
    forbid = [str(item) for item in (raw.get("forbid") or [])]
    need = [str(item) for item in (raw.get("need") or [])]
    return {"forbid": forbid, "need": need}


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _max_ctx_from(raw: dict[str, Any] | None) -> int | None:
    """Architecture maximum, when the row records one. A preferred length is not a maximum."""
    if not isinstance(raw, dict):
        return None
    for key in ("max_ctx", "max_model_len", "max_position_embeddings", "n_ctx_train"):
        if key not in raw or raw.get(key) in (None, ""):
            continue
        number = _as_int(raw.get(key))
        if number is not None and number >= 256:
            return number
    return None


def _load_override(model_id: str) -> dict[str, Any]:
    path = cards_dir() / f"{model_id}.toml"
    if not path.is_file():
        return {}
    try:
        with open(path, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _apply_override(card: ModelCard, over: dict[str, Any]) -> None:
    if not over:
        return
    if over.get("title"):
        card.title = str(over["title"])
    if "repo" in over:
        card.repo = str(over.get("repo") or "")
    if "revision" in over:
        card.revision = str(over.get("revision") or "")
    if str(over.get("format") or "") in FORMATS:
        card.format = str(over["format"])
    if over.get("weights_gib") not in (None, ""):
        try:
            card.weights_gib = float(over["weights_gib"])
        except (TypeError, ValueError):
            pass
    if isinstance(over.get("engines"), list) and over["engines"]:
        card.engines = [str(item) for item in over["engines"] if str(item)]
    if isinstance(over.get("hints"), dict):
        for key in ("moe", "mtp", "vision"):
            if key in over["hints"]:
                card.hints[key] = bool(over["hints"][key])
    if isinstance(over.get("flags"), dict):
        card.flags = _flags({"flags": over["flags"]})
    if isinstance(over.get("files"), dict):
        for host, path in over["files"].items():
            if path:
                card.files[str(host)] = str(path)
            elif str(host) in card.files:
                card.files.pop(str(host), None)
    ctx = _as_int(over.get("ctx") if "ctx" in over else over.get("default_ctx"))
    if ctx is not None:
        card.ctx = ctx
    max_ctx = _max_ctx_from(over)
    if max_ctx is not None:
        card.max_ctx = max_ctx
    timeout = _as_int(over.get("timeout_s"))
    if timeout is not None:
        card.timeout_s = timeout
    if over.get("kv_reserve_gib") not in (None, ""):
        try:
            card.kv_reserve_gib = float(over["kv_reserve_gib"])
        except (TypeError, ValueError):
            pass


def row_is_service(row: dict[str, Any]) -> bool:
    """A catalog row that is not a model server: image tools, web apps, batch jobs."""
    kind = str(row.get("kind") or "")
    if kind in {"service", "tool", "job"} or protect.is_video_kind(kind):
        return True
    if _tool_row(row):
        return True
    fmt = infer_format(row)
    if fmt in FORMATS:
        return False
    try:
        port = int(row.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if protect.is_off_limits(port) or (port and (port < 8010 or port > 8099)):
        engines = infer_engines(row, fmt)
        if not any(engine in ("vllm", "sglang", "trtllm", "llamacpp") for engine in engines):
            return True
    return False


def row_is_protected(row: dict[str, Any], catalog: dict[str, Any] | None = None) -> bool:
    """Off-limits ports (protect preset) and never_touch rows are listed and cannot be started or stopped."""
    try:
        port = int(row.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    if protect.is_off_limits(port):
        return True
    parts: list[str] = []
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    for key in ("start", "stop"):
        parts.extend(str(part) for part in (launch.get(key) or []))
    blob = " ".join(parts)
    if "dgx-announce" in blob or any(part.rsplit("/", 1)[-1] == "video-mode.sh" for part in parts):
        return True
    touch = (catalog or {}).get("never_touch") or {}
    banned = {str(item) for item in (touch.get("do_not_stop") or [])}
    return str(row.get("id") or "") in banned


def _catalog_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Defaults that sit under a saved recipe and above the engine defaults."""
    out: dict[str, Any] = {}
    ctx = _as_int(row.get("ctx") if row.get("ctx") is not None else row.get("default_ctx"))
    if ctx:
        out["ctx"] = ctx
    for src, dest in (
        ("gpu_memory_utilization", "gpu_mem_util"),
        ("gpu_mem_util", "gpu_mem_util"),
        ("kv_cache_dtype", "kv_cache_dtype"),
        ("max_num_seqs", "max_num_seqs"),
    ):
        if row.get(src) not in (None, ""):
            out[dest] = row[src]
    kind = str(row.get("kind") or "")
    if kind in {"vllm", "sglang", "trtllm", "llamacpp", "custom"}:
        out["engine"] = kind
    return out


def card_from_row(row: dict[str, Any], catalog: dict[str, Any] | None = None) -> ModelCard:
    fmt = infer_format(row)
    card = ModelCard(
        id=str(row.get("id")),
        title=str(row.get("title") or row.get("id")),
        repo=str(row.get("repo") or ""),
        revision=str(row.get("revision") or ""),
        format=fmt,
        weights_gib=_weights(row),
        hints=infer_hints(row),
        engines=infer_engines(row, fmt),
        files=_files(row),
        flags=_flags(row),
        ctx=_as_int(row.get("ctx") if row.get("ctx") is not None else row.get("default_ctx")),
        max_ctx=_max_ctx_from(row) or _max_ctx_from(row.get("launch") if isinstance(row.get("launch"), dict) else None),
        timeout_s=_as_int(row.get("timeout_s") if row.get("timeout_s") is not None else row.get("start_timeout_s")),
        scripts=_scripts(row),
        kv_reserve_gib=KV_RESERVE_DEFAULT,
        catalog_port=_as_int(row.get("port")),
        service=row_is_service(row),
        protected=row_is_protected(row, catalog),
        catalog_fields=_catalog_fields(row),
    )
    _apply_override(card, _load_override(card.id))
    return card


def host_budget(catalog: dict[str, Any], host: str, running_gib: float = 0.0) -> float:
    """UMA minus floor minus margin minus what is already running, in GiB."""
    spec = ((catalog.get("hosts") or {}).get(host) or {}) if catalog else {}
    node = None
    for item in CFG.nodes:
        if host in (item.name, item.id):
            node = item
            break
    try:
        uma = float(spec.get("uma_gib") or (node.uma_gib if node else 121) or 121)
    except (TypeError, ValueError):
        uma = 121.0
    try:
        floor = float(spec.get("floor_gib") if spec.get("floor_gib") is not None else 6)
    except (TypeError, ValueError):
        floor = 6.0
    try:
        margin = float(spec.get("margin_gib") if spec.get("margin_gib") is not None else 4)
    except (TypeError, ValueError):
        margin = 4.0
    return max(0.0, uma - floor - margin - float(running_gib or 0))


def _row_hosts(row: dict[str, Any]) -> list[str]:
    place = str(row.get("placement") or "")
    if place == "pair" or (place not in ("head", "worker", "both") and is_multi(place, ALL_ROLES)):
        local = row.get("local") or {}
        if isinstance(local, dict) and local:
            return [str(host) for host in local]
        # The Sparks the placement names ("pair" is head+worker); a missing one is left out.
        covered = roles_of("both" if place == "pair" else place, ALL_ROLES)
        return [node.name for node in (CFG.node_for_role(role) for role in covered) if node is not None]
    hosts = row.get("hosts") or []
    if hosts:
        return [str(host) for host in hosts]
    listen = row.get("listen_host")
    return [str(listen)] if listen else [CFG.head.name]


def _rss_tenant(catalog: dict[str, Any], row: dict[str, Any]) -> dict[str, Any] | None:
    """A tenant with ``measure: rss`` that this row is. Match by port, then id."""
    tenants = catalog.get("tenants") if isinstance(catalog.get("tenants"), dict) else {}
    try:
        port = int(row.get("port") or 0)
    except (TypeError, ValueError):
        port = 0
    row_id = str(row.get("id") or "")
    for name, spec in (tenants or {}).items():
        if not isinstance(spec, dict) or spec.get("measure") != "rss":
            continue
        try:
            spec_port = int(spec.get("port") or 0)
        except (TypeError, ValueError):
            spec_port = 0
        if (spec_port and spec_port == port) or str(name) == row_id:
            return spec
    return None


def _resident_floor(row: dict[str, Any]) -> float | None:
    if row.get("resident_min_gib") in (None, ""):
        return None
    try:
        return float(row.get("resident_min_gib"))
    except (TypeError, ValueError):
        return None


def _row_charge(
    catalog: dict[str, Any],
    row: dict[str, Any],
    *,
    rss_fn: Any,
) -> float:
    """GiB to hold against the budget for one row that is actually running.

    A row with ``resident_min_gib``, or a tenant measured by RSS, is charged
    the measured RSS and at least ``resident_min_gib``. Other rows keep
    ``estimate_gib``.
    """
    try:
        estimate = float(row.get("estimate_gib") or 0)
    except (TypeError, ValueError):
        estimate = 0.0
    floor = _resident_floor(row)
    tenant = _rss_tenant(catalog, row)
    if floor is None and tenant is None:
        return estimate
    spec = tenant or {}
    try:
        gib, _hits, err = rss_fn(
            comms=list(spec.get("comm") or []),
            cmdline_substr=list(spec.get("cmdline_substr") or []),
            user=spec.get("user"),
        )
    except TypeError:
        gib, err = rss_fn(), None
    if gib is None or err:
        return floor if floor is not None else estimate
    measured = float(gib)
    if floor is not None:
        return max(measured, floor)
    return measured


def running_on(
    catalog: dict[str, Any],
    host: str,
    listening: set[int],
    skip_id: str | None = None,
    *,
    rss_fn: Any = None,
) -> float:
    """GiB of other library rows whose port is confirmed listening here.

    Rows that share one port are counted once, at the larger charge, so a
    pair and a single model on the same port are not added twice. ``listening``
    must already be identity-confirmed; a tailnet proxy alone is not enough.
    """
    if rss_fn is None:
        from sparksi.probe import rss_match_gib

        rss_fn = rss_match_gib
    by_port: dict[int, float] = {}
    for row in _library_rows(catalog):
        if skip_id and str(row.get("id")) == skip_id:
            continue
        if host not in _row_hosts(row):
            continue
        try:
            port = int(row.get("port"))
        except (TypeError, ValueError):
            continue
        if port not in listening:
            continue
        charge = _row_charge(catalog, row, rss_fn=rss_fn)
        by_port[port] = max(by_port.get(port, 0.0), charge)
    return sum(by_port.values())


def _gib(value: float) -> str:
    text = f"{value:.1f}".rstrip("0").rstrip(".")
    return text or "0"


def derive_placements(
    card: ModelCard,
    budgets: dict[str, float],
    presence: dict[str, bool | None] | None = None,
    down: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Which single-Spark and multi-Spark placements this card may use, each with a reason."""
    presence = presence if presence is not None else card.presence
    unreachable = set(down or ())
    need = float(card.weights_gib) + float(card.kv_reserve_gib)
    two_node = any(engine in TWO_NODE_ENGINES for engine in card.engines)
    only_llama = "llamacpp" in card.engines and not two_node
    out: dict[str, dict[str, Any]] = {}
    single_ok = False

    def single_entry(role: str, host: str) -> dict[str, Any]:
        budget = float(budgets.get(role, budgets.get(host, 0.0)))
        here = presence.get(host)
        reason = ""
        allowed = False
        if host not in card.files:
            reason = f"weights are not on {role_words(role)}"
        elif here is None:
            if host in unreachable:
                reason = f"could not reach {role_words(role)}"
            else:
                reason = f"weights were not checked on {role_words(role)}"
        elif here is False:
            reason = f"weights are not on {role_words(role)}"
        elif need > budget:
            reason = (
                f"{role_words(role)} cannot hold this model: weights plus a {_gib(card.kv_reserve_gib)} GiB"
                f" KV reserve need {_gib(need)} GiB and {role_words(role)} has {_gib(budget)} GiB"
            )
        else:
            allowed = True
        return {"allowed": allowed, "reason": reason, "host": host}

    def pair_entry(role_a: str, role_b: str) -> dict[str, Any]:
        names = {role: CFG.node_by_role(role).name for role in (role_a, role_b)}
        reason = ""
        allowed = False
        if only_llama:
            reason = "llamacpp runs on one Spark"
        elif not two_node:
            reason = "this service runs on one Spark"
        else:
            absent = [
                role for role in (role_a, role_b)
                if names[role] not in card.files or presence.get(names[role]) is False
            ]
            unknown = [
                role for role in (role_a, role_b)
                if presence.get(names[role]) is None and names[role] in card.files
            ]
            # Head+worker keeps its words; any other pair names its Sparks.
            legacy = (role_a, role_b) == ("head", "worker")

            def which(roles: list[str]) -> str:
                if legacy and len(roles) >= 2:
                    return "both Sparks"
                return " and ".join(role_words(role) for role in roles)

            if single_ok:
                reason = "this model fits on one Spark"
            elif absent:
                reason = "weights are not on " + ("both Sparks" if legacy else which(absent))
            elif unknown:
                down_roles = [role for role in unknown if names[role] in unreachable]
                if len(down_roles) >= 2:
                    reason = "could not reach " + which(down_roles)
                elif down_roles:
                    reason = f"could not reach {role_words(down_roles[0])}"
                else:
                    reason = "weights were not checked on " + ("both Sparks" if legacy else which(unknown))
            else:
                per = float(card.weights_gib) / 2.0 + float(card.kv_reserve_gib)
                tight = []
                for role in (role_a, role_b):
                    budget = float(budgets.get(role, budgets.get(names[role], 0.0)))
                    if per > budget:
                        tight.append((role, budget))
                if tight:
                    role, budget = tight[0]
                    reason = (
                        f"the two Sparks cannot hold this model: each needs {_gib(per)} GiB"
                        f" and {role_words(role)} has {_gib(budget)} GiB"
                    )
                else:
                    allowed = True
        return {"allowed": allowed, "reason": reason, "hosts": [names[role_a], names[role_b]]}

    for role, node in CFG.role_nodes():
        out[role] = single_entry(role, node.name)
        single_ok = single_ok or out[role]["allowed"]

    for placement in multis():
        roles = roles_of(placement)
        out[placement] = pair_entry(*roles)

    if "both" not in out:
        # A one-Spark site: the two-Spark form is listed, never allowed, with plain words.
        out["both"] = {"allowed": False, "reason": needs_text("both"), "hosts": [CFG.head.name]}
    return out


def probe_presence(
    runner: Any,
    files: dict[str, list[str]],
    timeout: int = 8,
) -> tuple[dict[str, dict[str, bool | None]], set[str]]:
    """One probe per node, in parallel. ``files`` maps a host to the paths to stat.

    The second value is the set of hosts that did not answer. A dark node stays
    unknown (``None``) and is listed there; it does not raise.
    """
    out: dict[str, dict[str, bool | None]] = {host: {path: None for path in paths} for host, paths in files.items()}
    down: set[str] = set()
    if runner is None:
        return out, down

    def one(host: str, paths: list[str]) -> dict[str, Any]:
        try:
            result = runner.invoke(host, {"op": "probe", "ports": [], "paths": paths}, timeout=timeout)
        except Exception as exc:  # noqa: BLE001 — a dark node is unknown, not a crash
            return {"ok": False, "error": str(exc)}
        return result if isinstance(result, dict) else {"ok": False}

    grouped = {host: list(dict.fromkeys(paths)) for host, paths in files.items() if paths}
    results: dict[str, dict[str, Any]] = {}
    if len(grouped) <= 1:
        results = {host: one(host, paths) for host, paths in grouped.items()}
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(grouped)) as pool:
            futures = {pool.submit(one, host, paths): host for host, paths in grouped.items()}
            for future in futures:
                host = futures[future]
                try:
                    results[host] = future.result()
                except Exception as exc:  # noqa: BLE001 — one Spark must not sink the other
                    results[host] = {"ok": False, "error": str(exc)}
    for host, paths in grouped.items():
        result = results.get(host) or {}
        stats = result.get("paths") if isinstance(result.get("paths"), dict) else {}
        if result.get("ok") is False and not stats:
            down.add(host)
            continue
        for path in paths:
            stat = stats.get(path) or {}
            if stat:
                out[host][path] = bool(stat.get("exists"))
            else:
                out[host][path] = False
    return out, down


def build_cards(
    catalog: dict[str, Any] | None = None,
    runner: Any = None,
    *,
    presence: dict[str, dict[str, bool | None]] | None = None,
    listening: dict[str, set[int]] | None = None,
    running: dict[str, set[int]] | None = None,
    probe: bool = True,
    probed: dict[str, dict[str, bool | None]] | None = None,
    down: set[str] | None = None,
) -> list[ModelCard]:
    """Cards for every library row, with placements filled in."""
    catalog = catalog if catalog is not None else load_catalog()
    cards = [card_from_row(row, catalog) for row in _library_rows(catalog)]
    grouped: dict[str, list[str]] = {}
    for card in cards:
        for host, path in card.files.items():
            grouped.setdefault(host, [])
            if path not in grouped[host]:
                grouped[host].append(path)
    found: dict[str, dict[str, bool | None]] = {}
    unreachable = set(down or ())
    if presence is None and probed is not None:
        found = probed
    elif presence is None and probe and runner is not None and grouped:
        found, unreachable = probe_presence(runner, grouped)
    for card in cards:
        if presence is not None and card.id in presence:
            card.presence = dict(presence[card.id])
        else:
            card.presence = {host: (found.get(host) or {}).get(path) for host, path in card.files.items()}
        budgets = {}
        charge = running if running is not None else listening
        for role, node in CFG.role_nodes():
            ports = set()
            if charge:
                ports = set(charge.get(node.name) or charge.get(node.id) or charge.get(role) or [])
            held = running_on(catalog, node.name, ports, skip_id=card.id) if charge is not None else 0.0
            budgets[role] = host_budget(catalog, node.name, held)
        card.placements = derive_placements(card, budgets, card.presence, down=unreachable)
    return cards


def card_by_id(cards: list[ModelCard], model_id: str) -> ModelCard | None:
    for card in cards:
        if card.id == model_id:
            return card
    return None
