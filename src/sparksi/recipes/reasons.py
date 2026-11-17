# SPDX-License-Identifier: Apache-2.0

"""Machine-readable reason codes for a plan or start refusal.

The words stay in ``conflicts`` for readers that match the text. ``reasons``
is the same list with a code on each line. A downstream retry matches the
substring ``free right now`` in memory refusals, so that phrase stays.
"""

from __future__ import annotations


def reason_code(message: str) -> str:
    """A stable code for one refusal sentence."""
    text = str(message or "").lower()
    if "free right now" in text:
        return "memory_not_free"
    if "8188" in text or "off limits" in text:
        return "off_limits"
    if (
        "already listening" in text
        or " in use" in text
        or "is reserved" in text
        or "no free port" in text
        or "needs a port" in text
        or "used by another recipe" in text
        or ("port" in text and "outside" in text)
        or "port must be an integer" in text
    ):
        return "port_conflict"
    if "already running" in text or "already loading" in text:
        return "already_running"
    if "over budget" in text or "cannot hold" in text:
        return "over_budget"
    if "weights are not" in text or "weights missing" in text:
        return "weights_missing"
    if "image" in text and ("missing" in text or "not in the allowed" in text or "no pinned image" in text):
        return "image_missing"
    if text.startswith("unknown engine") or "is not offered" in text or "is not available" in text:
        return "placement"
    if "llamacpp runs on one" in text:
        return "placement"
    return "refused"


def annotate(conflicts: list[str]) -> list[dict[str, str]]:
    """One ``{code, message}`` per refusal, in the same order."""
    out = []
    for item in conflicts:
        message = str(item or "").strip()
        if not message:
            continue
        out.append({"code": reason_code(message), "message": message})
    return out


def join_reasons(conflicts: list[str]) -> str:
    """Several refusals, with a separator between them."""
    parts = [str(item).strip() for item in conflicts if str(item or "").strip()]
    return " · ".join(parts)
