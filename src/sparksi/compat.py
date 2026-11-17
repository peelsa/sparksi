# SPDX-License-Identifier: Apache-2.0

"""Compatibility table: legacy names Sparksi still reads.

Config keys, env vars, API fields and catalog values were renamed to generic
names. Old names are still read: when only the old name is set it is used and
one deprecation note is logged at ``sparksi serve`` start; when both are set,
the new one wins. API responses carry both field names for now.

The protected-service data itself (ports, ids, kinds, process names) is a
preset in ``sparksi.protect``, off by default. The two stdlib-only node
scripts (agent.py, announce.py) carry a copy of the few names they need;
tests/test_compat.py keeps the copies equal.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# --- Renamed config keys: new [paths] key -> legacy key that is still read ----
LEGACY_PATH_KEYS: dict[str, str] = {
    "video_service_root": "comfy_root",
    "model_inventory": "model_truth",
}

# --- Renamed env vars: new name -> legacy names still read, in order --------
LEGACY_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "SPARKSI_MODEL_INVENTORY": ("SPARKSI_MODEL_TRUTH", "DGX_MEM_MODEL_TRUTH"),
    "SPARKSI_VIDEO_STATE": ("SPARKSI_COMFY_STATE",),
}

# --- API / JSON fields: new name -> legacy name emitted alongside it ----------
LEGACY_FIELDS: dict[str, str] = {
    "video_service_live": "comfy_live",
    "diff_vs_model_inventory": "diff_vs_model_truth",
    "wrote_model_inventory": "wrote_model_truth",
}

# --- Announce packet key for the video service: new, then legacy -----------
VIDEO_PACKET_KEY = "video"
LEGACY_VIDEO_PACKET_KEY = "comfy"

# --- Agent HTTP probe flag for a /system_stats answer: new, then legacy ----
AGENT_VIDEO_KEY = "video_service"
LEGACY_AGENT_VIDEO_KEY = "comfy"

# --- /api/monitor and probe mirrors: new name, then the legacy one sent alongside ---
# Peer record key: "video_service", plus a copy under the legacy key.
LEGACY_MONITOR_VIDEO_KEY = "comfy"
# Loaded video row: "kind": "video_service" and "video_service": True, plus
# "legacy_kind": "comfy" and "comfy": True for readers of the old names.
LEGACY_VIDEO_SERVICE_KIND = "comfy"
LEGACY_ROW_FLAG = "comfy"
# Probe identity for a /system_stats answer: "identity" is the new label and
# "legacy_identity" carries the old one.
LEGACY_VIDEO_HTTP_LABEL = "comfy"

# --- New names of the video service row (its protection is sparksi.protect) ---
VIDEO_SERVICE_KIND = "video_service"
VIDEO_SERVICE_ID = "video-service"
# HTTP probe label for an answer on /system_stats.
VIDEO_HTTP_LABEL = VIDEO_SERVICE_ID
# Install directory and pid file names, newest first. The first one that exists wins.
VIDEO_SERVICE_ROOT_FALLBACKS = ("~/services/video-service", "~/services/comfyui")
VIDEO_SERVICE_PID_FILES = ("video-service.pid", "comfyui.pid")
MODEL_INVENTORY_FALLBACKS = ("~/ops/model-inventory.yaml", "~/ops/MODEL_TRUTH.yaml")

_NOTES: list[str] = []


def note(old: str, new: str) -> None:
    """Remember one deprecated name in use; ``sparksi serve`` logs these at start."""
    text = f"{old} is deprecated; use {new}"
    if text not in _NOTES:
        _NOTES.append(text)


def deprecation_notes() -> list[str]:
    return list(_NOTES)


def apply_legacy_paths(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Copy of a config file dict with legacy [paths] keys mirrored to the new keys."""
    out = dict(raw or {})
    paths = out.get("paths")
    if not isinstance(paths, dict):
        return out
    paths = dict(paths)
    for new, old in LEGACY_PATH_KEYS.items():
        if old in paths and new not in paths:
            paths[new] = paths[old]
            note(f"[paths] {old}", f"[paths] {new}")
    out["paths"] = paths
    return out


def legacy_env(name: str, environ: Any = None) -> str | None:
    """Value of the first legacy env var for ``name`` that is set and not empty."""
    environ = os.environ if environ is None else environ
    for old in LEGACY_ENV_NAMES.get(name, ()):
        val = environ.get(old)
        if val not in (None, ""):
            note(old, name)
            return val
    return None


def first_existing(candidates: tuple[str, ...]) -> str:
    """The first candidate path that exists, expanded; else the first one."""
    paths = [os.path.expanduser(c) for c in candidates]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]


def video_service_root(configured: str) -> str:
    return configured or first_existing(VIDEO_SERVICE_ROOT_FALLBACKS)


def video_pid_file(state: Path) -> Path:
    for name in VIDEO_SERVICE_PID_FILES:
        if (state / name).is_file():
            return state / name
    return state / VIDEO_SERVICE_PID_FILES[-1]


def with_legacy_fields(payload: dict[str, Any]) -> dict[str, Any]:
    """Add each legacy field name next to its new one (in place; returns payload)."""
    for new, old in LEGACY_FIELDS.items():
        if new in payload:
            payload[old] = payload[new]
    return payload


def field(payload: dict[str, Any], new: str) -> Any:
    """Read a renamed field: the new name, else its legacy name (older servers)."""
    if new in payload:
        return payload[new]
    return payload.get(LEGACY_FIELDS.get(new, new))


def agent_video_flag(answer: dict[str, Any]) -> bool:
    """True when an agent HTTP answer says /system_stats looks like the video service."""
    if AGENT_VIDEO_KEY in answer:
        return bool(answer.get(AGENT_VIDEO_KEY))
    return bool(answer.get(LEGACY_AGENT_VIDEO_KEY))


def video_packet(pkt: dict[str, Any]) -> Any:
    """The video service entry of an announce packet, new key first."""
    if VIDEO_PACKET_KEY in pkt:
        return pkt.get(VIDEO_PACKET_KEY)
    return pkt.get(LEGACY_VIDEO_PACKET_KEY)
