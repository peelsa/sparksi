# SPDX-License-Identifier: Apache-2.0

"""Protection presets: named sets of protected-service data. All are off by default.

One preset exists, ``video-service``: a video generation service that Sparksi
must never start, stop, evict or reuse the port of. A fresh install has no
reserved port 8188, no process-name matching and no video row.

Turn it on with ``[protect] presets = ["video-service"]`` in sparksi.toml, or
``SPARKSI_PROTECT=video-service`` (comma-separated; added to the file's list).
It also turns on by itself while an install still uses a legacy marker (see
``LEGACY_MARKERS``), so an existing install is never left unprotected;
``sparksi serve`` logs that once at start. Every helper below returns False or
an empty value while the preset is off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

VIDEO_SERVICE = "video-service"


@dataclass(frozen=True)
class Preset:
    """Protected-service data for one preset."""

    ports: frozenset[int]  # off limits: never probed for health, started on, stopped or reused
    kinds: frozenset[str]  # catalog kinds of the service row
    ids: frozenset[str]  # catalog ids of the service row itself
    protected_ids: frozenset[str]  # ids never started, stopped or evicted
    id_substrings: tuple[str, ...]  # an id containing one of these is protected
    argv_substrings: tuple[str, ...]  # a stop/kill/rm/restart argv containing one is refused
    process_substrings: tuple[str, ...]  # lowercased process/container names of the running service
    rss_awk: str  # awk regex the peer dump uses to sum the service's RSS
    tenant_cmdline: tuple[str, ...]  # default cmdline match for its tenant row
    seat: str  # catalog seat it sits in


# The video-service preset. The old names here are the ones existing installs use.
VIDEO = Preset(
    ports=frozenset({8188}),
    kinds=frozenset({"video_service", "comfy"}),
    ids=frozenset({VIDEO_SERVICE, "comfy"}),
    protected_ids=frozenset({VIDEO_SERVICE, "comfy", "comfyui", "h3", "minimax-h3"}),
    id_substrings=(VIDEO_SERVICE, "h3", "comfy"),
    argv_substrings=(VIDEO_SERVICE, "comfy", "8188", "minimax", "h3"),
    process_substrings=("comfy",),
    rss_awk="[c]omfyui|[C]omfyUI",
    tenant_cmdline=("comfy",),
    seat="video",
)

PRESETS: dict[str, Preset] = {VIDEO_SERVICE: VIDEO}

# Legacy markers: any one of these turns the video-service preset on by itself.
LEGACY_MARKERS: dict[str, tuple[str, ...]] = {
    "config_paths": ("comfy_root", "model_truth"),  # [paths] keys, as written in the file
    "config_services": ("video-h3",),  # [services.<id>] tables
    "env": ("SPARKSI_MODEL_TRUTH", "DGX_MEM_MODEL_TRUTH", "SPARKSI_COMFY_STATE"),
    "catalog_kind": ("comfy",),
    "catalog_id": ("comfy",),
    "catalog_do_not_stop": ("comfy",),  # substring of a never_touch.do_not_stop item
    "catalog_tenant": ("comfy",),
    "catalog_seat_video": ("comfy",),
}

_CATALOG_CACHE: dict[tuple[str, int], list[str]] = {}
_UNKNOWN: list[str] = []


def _file_raw() -> dict[str, Any]:
    from sparksi.config import CFG

    raw = getattr(CFG, "file_raw", None)
    return raw if isinstance(raw, dict) else {}


def explicit() -> frozenset[str]:
    """Presets named in ``[protect] presets`` and ``SPARKSI_PROTECT``."""
    names: set[str] = set()
    section = _file_raw().get("protect")
    value = section.get("presets") if isinstance(section, dict) else None
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        names.update(str(item).strip() for item in value)
    names.update(part.strip() for part in os.environ.get("SPARKSI_PROTECT", "").split(","))
    names.discard("")
    _UNKNOWN[:] = sorted(names - set(PRESETS))
    return frozenset(names & set(PRESETS))


def unknown_presets() -> list[str]:
    explicit()
    return list(_UNKNOWN)


def _catalog_reasons() -> list[str]:
    """Legacy markers in the catalog file; cached by path and mtime. Unreadable: none."""
    try:
        from sparksi.catalog import catalog_path, load_yaml

        path = catalog_path()
        key = (str(path), os.stat(path).st_mtime_ns)
    except Exception:
        return []
    if key in _CATALOG_CACHE:
        return _CATALOG_CACHE[key]
    try:
        data = load_yaml(path)
    except Exception:
        data = None
    reasons = catalog_markers(data if isinstance(data, dict) else {})
    _CATALOG_CACHE.clear()
    _CATALOG_CACHE[key] = reasons
    return reasons


def catalog_markers(data: dict[str, Any]) -> list[str]:
    """Legacy markers in a loaded catalog dict."""
    found: list[str] = []
    rows = [row for value in data.values() if isinstance(value, list) for row in value if isinstance(row, dict)]
    for row in rows:
        kind, rid = str(row.get("kind") or ""), str(row.get("id") or "")
        if kind in LEGACY_MARKERS["catalog_kind"]:
            found.append(f"catalog kind {kind}")
        if rid in LEGACY_MARKERS["catalog_id"]:
            found.append(f"catalog id {rid}")
    touch = data.get("never_touch") if isinstance(data.get("never_touch"), dict) else {}
    for item in touch.get("do_not_stop") or []:
        if any(mark in str(item) for mark in LEGACY_MARKERS["catalog_do_not_stop"]):
            found.append(f"catalog never_touch.do_not_stop {item}")
    tenants = data.get("tenants") if isinstance(data.get("tenants"), dict) else {}
    found += [f"catalog tenant {name}" for name in tenants if name in LEGACY_MARKERS["catalog_tenant"]]
    seats = data.get("seats") if isinstance(data.get("seats"), dict) else {}
    if str(seats.get("video") or "") in LEGACY_MARKERS["catalog_seat_video"]:
        found.append(f"catalog seats.video {seats.get('video')}")
    return list(dict.fromkeys(found))


def config_markers(raw: dict[str, Any], environ: Any = None) -> list[str]:
    """Legacy markers in a config file dict (as written) and the environment."""
    environ = os.environ if environ is None else environ
    found: list[str] = []
    paths = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    found += [f"config [paths] {key}" for key in LEGACY_MARKERS["config_paths"] if key in paths]
    services = raw.get("services") if isinstance(raw.get("services"), dict) else {}
    found += [f"config [services.{key}]" for key in LEGACY_MARKERS["config_services"] if key in services]
    found += [f"env {name}" for name in LEGACY_MARKERS["env"] if environ.get(name)]
    return found


def auto_reasons() -> list[str]:
    """Why the video-service preset is on without being named; empty when not."""
    return config_markers(_file_raw()) + _catalog_reasons()


def active() -> frozenset[str]:
    """Explicit presets, plus video-service while a legacy marker is in use."""
    names = explicit()
    if VIDEO_SERVICE not in names and auto_reasons():
        names = names | {VIDEO_SERVICE}
    return names


def video_on() -> bool:
    return VIDEO_SERVICE in active()


def log_auto(logger: Any) -> None:
    """Log once at serve start when the preset turned on by itself, and any unknown name."""
    for name in unknown_presets():
        logger.warning("protect: unknown preset %r ignored", name)
    if VIDEO_SERVICE in explicit():
        return
    reasons = auto_reasons()
    if reasons:
        logger.warning(
            "protect: video-service preset auto-enabled for a legacy install (%s); "
            'add [protect] presets = ["video-service"] to sparksi.toml',
            ", ".join(reasons),
        )


def _video() -> Preset | None:
    return VIDEO if video_on() else None


def video_port() -> int | None:
    """The video service port while the preset is on, else None."""
    p = _video()
    return min(p.ports) if p else None


def off_limits_ports() -> frozenset[int]:
    p = _video()
    return p.ports if p else frozenset()


def is_off_limits(port: Any) -> bool:
    """True when ``port`` belongs to an active preset. Bad values are False."""
    ports = off_limits_ports()
    if not ports:
        return False
    try:
        return int(str(port).strip()) in ports
    except (TypeError, ValueError):
        return False


def is_video_kind(kind: Any) -> bool:
    p = _video()
    return bool(p) and str(kind or "").lower() in p.kinds


def is_video_id(rid: Any) -> bool:
    p = _video()
    return bool(p) and str(rid or "").lower() in p.ids


def video_ids() -> frozenset[str]:
    p = _video()
    return p.ids if p else frozenset()


def protected_ids() -> frozenset[str]:
    p = _video()
    return p.protected_ids if p else frozenset()


def is_protected_id(rid: Any) -> bool:
    p = _video()
    low = str(rid or "").lower()
    return bool(p) and bool(low) and (low in p.protected_ids or any(s in low for s in p.id_substrings))


def is_video_seat(seat: Any) -> bool:
    p = _video()
    return bool(p) and str(seat or "").lower() == p.seat


def is_video_process(name: Any) -> bool:
    p = _video()
    low = str(name or "").lower()
    return bool(p) and any(s in low for s in p.process_substrings)


def argv_substrings() -> tuple[str, ...]:
    p = _video()
    return p.argv_substrings if p else ()


def process_substrings() -> tuple[str, ...]:
    p = _video()
    return p.process_substrings if p else ()


def tenant_cmdline() -> tuple[str, ...]:
    p = _video()
    return p.tenant_cmdline if p else ()


def rss_awk() -> str:
    p = _video()
    return p.rss_awk if p else ""


def agent_values() -> dict[str, Any]:
    """What the stdlib-only node agent needs, sent with each request."""
    return {
        "off_limits_ports": sorted(off_limits_ports()),
        "video_process_substrings": list(process_substrings()),
    }
