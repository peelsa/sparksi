# SPDX-License-Identifier: Apache-2.0

"""How long a load may wait, and which HTTP paths count as healthy.

The default budget is half an hour. A catalog recipe's ``timeout_s`` overrides
that. A library row's ``start_timeout_s`` (the TensorFold row carries a long
one) is the budget when no recipe timeout matches. An off-limits port (8188
with the ``video-service`` preset) is never a health target: it may only show
up in a listening-port list.
"""

from __future__ import annotations

from typing import Any

from sparksi import protect

DEFAULT_HEALTH_TIMEOUT_S = 1800


def _as_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def health_timeout_for(
    catalog: dict[str, Any] | None,
    row: dict[str, Any] | None = None,
    step: dict[str, Any] | None = None,
) -> int:
    """Seconds to poll before a load is ``timed_out``.

    A matching recipe ``timeout_s`` wins, then the row's own ``timeout_s``,
    then ``start_timeout_s``. Otherwise the default is 1800.
    """
    row = row or {}
    step = step or {}
    if "health_timeout_s" in step:
        explicit = _as_int(step.get("health_timeout_s"))
        if explicit is not None:
            return explicit
    rid = str(row.get("id") or step.get("id") or "")
    for name, rec in ((catalog or {}).get("recipes") or {}).items():
        if not isinstance(rec, dict):
            continue
        timeout = _as_int(rec.get("timeout_s"))
        if timeout is None:
            continue
        members = [str(item) for item in rec.get("members") or []]
        aliases = [str(item) for item in rec.get("aliases") or []]
        if rid and (rid == str(name) or rid in members or rid in aliases):
            return timeout
    own = _as_int(row.get("timeout_s"))
    if own is not None:
        return own
    start = _as_int(row.get("start_timeout_s"))
    if start is None:
        start = _as_int(step.get("start_timeout_s"))
    if start is not None:
        return start
    return DEFAULT_HEALTH_TIMEOUT_S


def _port(row: dict[str, Any]) -> int | None:
    try:
        if row.get("port") in (None, ""):
            return None
        return int(row["port"])
    except (TypeError, ValueError):
        return None


def health_paths_for(row: dict[str, Any] | None) -> list[str]:
    """HTTP paths to GET on the model port, or none when the port is off limits.

    HTTP engines are probed at ``/health`` and then ``/v1/models``. A catalog
    ``probe`` or ``identity.http`` path replaces that pair. An off-limits port is
    never requested.
    """
    row = row or {}
    port = _port(row)
    if protect.is_off_limits(port):
        return []
    ident = row.get("identity") or {}
    custom = ident.get("http") or row.get("probe")
    if custom:
        text = str(custom)
        return [text if text.startswith("/") else "/" + text]
    kind = str(row.get("kind") or "")
    if kind in ("vllm", "llamacpp", "sglang", "trtllm") or row.get("placement") in ("pair", "both"):
        return ["/health", "/v1/models"]
    return []
