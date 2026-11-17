# SPDX-License-Identifier: Apache-2.0

"""Per-model token ledger.

vLLM Prometheus counters (`prompt_tokens_total`, `generation_tokens_total`)
reset when the engine restarts. We sample them and credit only positive
deltas into a durable lifetime total keyed by catalog id.

Not related to unlock tokens. Field names avoid `token` as a secret key.
"""

from __future__ import annotations

import fcntl
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparksi.catalog import load_catalog
from sparksi.lock import state_dir

from sparksi.config import env, state_subdir

_LOCK = threading.Lock()
_WANTED = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "request_success_total",
)


def ledger_path() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state) / "token-ledger.json"
    durable = state_subdir(Path.home() / ".local" / "state")
    try:
        durable.mkdir(parents=True, exist_ok=True)
        return durable / "token-ledger.json"
    except OSError:
        p = state_dir()
        p.mkdir(parents=True, exist_ok=True)
        return p / "token-ledger.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty_row(mid: str) -> dict[str, Any]:
    return {
        "id": mid,
        "port": None,
        "engine_prompt": 0,
        "engine_gen": 0,
        "engine_requests": 0,
        "life_prompt": 0,
        "life_gen": 0,
        "life_requests": 0,
        "updated_at": None,
    }


def _parse_metrics(text: str) -> dict[str, float]:
    agg: dict[str, float] = {}
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
        if base not in _WANTED:
            continue
        try:
            agg[base] = agg.get(base, 0.0) + float(val)
        except ValueError:
            continue
    return agg


def _credit(prev: int, current: int) -> int:
    """Monotonic counter: credit the delta, or the new value after a reset."""
    if current >= prev:
        return current - prev
    return current


def _load_unlocked(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"models": {}}
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {"models": {}}
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        return {"models": {}}
    return data


def _atomic_write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, payload.encode())
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)


def load_ledger() -> dict[str, Any]:
    with _LOCK:
        return _load_unlocked(ledger_path())


def observe(
    model_id: str,
    *,
    prompt: int,
    gen: int,
    requests: int = 0,
    port: int | None = None,
) -> dict[str, Any]:
    """Fold one engine sample. The first sight of a counter is not credited.

    The durable totals live in the counter-delta ledger. ``requests`` is kept
    on the returned row for older callers and is not a separate series.
    """
    from sparksi.ledger import fold

    mid = str(model_id)
    prompt = max(0, int(prompt))
    gen = max(0, int(gen))
    requests = max(0, int(requests))
    row = fold(mid, role="head", port=int(port or 0), prompt=prompt, generation=gen)
    row["engine_requests"] = requests
    row["life_requests"] = requests
    return row


def snapshot() -> dict[str, dict[str, Any]]:
    """Lifetime rows. Does not sample and does not write."""
    from sparksi.ledger import lifetimes

    rows: dict[str, dict[str, Any]] = {}
    for mid, life in lifetimes().items():
        rows[mid] = {
            "id": mid,
            "life_prompt": int(life.get("prompt") or 0),
            "life_gen": int(life.get("generation") or 0),
        }
    return rows


def catalog_vllm(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    return [r for r in catalog.get("residents") or [] if r.get("kind") == "vllm" and r.get("id")]


def seated_ids(catalog: dict[str, Any]) -> set[str]:
    """Catalog seats whose occupant is a vLLM (brain/vision/counsel, not video)."""
    seats = catalog.get("seats") or {}
    by_id = {r.get("id"): r for r in catalog.get("residents") or []}
    out: set[str] = set()
    for cid in seats.values():
        row = by_id.get(cid)
        if row and row.get("kind") == "vllm":
            out.add(str(cid))
    return out


def _tok_label(row: dict[str, Any]) -> str:
    return f"{fmt_count(row.get('life_gen'))} out · {fmt_count(row.get('life_prompt'))} in"


def report(catalog: dict[str, Any] | None = None, *, sample: bool = True) -> dict[str, dict[str, Any]]:
    """Per-model counts for the API/UI.

    Lifetime totals come from the counter-delta ledger. This function does not
    sample and does not write, including when ``sample`` is true. The poll
    loop is the only writer. Seated vLLM that is idle or down is still listed,
    at zero if never sampled.
    """
    del sample  # reads never fold a counter
    try:
        cat = catalog or load_catalog()
    except Exception:
        cat = {}
    live_rows = snapshot()
    seated = seated_ids(cat)
    want = set(seated)
    by_id = {r.get("id"): r for r in cat.get("residents") or []}
    want.update(str(k) for k in live_rows)
    out: dict[str, dict[str, Any]] = {}
    for cid in sorted(want):
        rec = by_id.get(cid) or {}
        row = dict(live_rows.get(cid) or _empty_row(cid))
        if rec.get("port") and not row.get("port"):
            row["port"] = rec["port"]
        row["tok_label"] = _tok_label(row)
        row["seated"] = cid in seated
        out[cid] = row
    return out


def format_ledger(models: dict[str, dict[str, Any]] | None = None) -> str:
    rows = models if models is not None else report(sample=True)
    if not rows:
        return "tokens: (empty ledger — no vLLM sample yet)"
    lines = [
        f"{'id':22} {'out':>8} {'in':>8} {'req':>5}  port",
    ]
    for mid in sorted(rows):
        r = rows[mid]
        port = r.get("port")
        port_s = f":{port}" if port else "—"
        lines.append(
            f"{mid:22} {fmt_count(r.get('life_gen')):>8} "
            f"{fmt_count(r.get('life_prompt')):>8} "
            f"{fmt_count(r.get('life_requests')):>5}  {port_s}"
        )
    return "\n".join(lines)


def fmt_count(n: int | float | None) -> str:
    if n is None:
        return "—"
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def sample_vllm(catalog: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Poll run-record targets. Catalog seats and loopback ports are not targets.

    ``catalog`` is accepted so older callers keep working. It is not read.
    """
    del catalog
    from sparksi.ledger import poll_once

    poll_once()
    return snapshot()
