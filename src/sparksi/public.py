# SPDX-License-Identifier: Apache-2.0

"""Public HTTP helpers: video-live gate, secret stripping."""

from __future__ import annotations

from typing import Any

from sparksi import protect
from sparksi.compat import with_legacy_fields
from sparksi.probe import port_listening

SECRET_KEYS = frozenset(
    {
        "unlock_token",
        "unlock.token",
        "token",
        "unlock-token",
    }
)


def video_service_is_live(listen_fn=None) -> bool:
    port = protect.video_port()
    if port is None:
        return False
    fn = listen_fn or port_listening
    return bool(fn(port, "127.0.0.1"))


def refuse_apply_while_video_live(listen_fn=None) -> dict[str, Any] | None:
    """API refuse: :8188 live ⇒ no commit. Independent of victims[]."""
    if not video_service_is_live(listen_fn):
        return None
    return with_legacy_fields({
        "executed": False,
        "mutating": False,
        "verdict": "REFUSE",
        "error": "REFUSE: :8188 / the protected video service is live; video stays",
        "video_service_live": True,
        "reasons": [":8188 is live — POST /api/apply refused regardless of victims[]"],
    })


def strip_secrets(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if str(k) in SECRET_KEYS or "unlock" in str(k).lower() and "token" in str(k).lower():
                continue
            out[k] = strip_secrets(v)
        return out
    if isinstance(obj, list):
        return [strip_secrets(x) for x in obj]
    return obj
