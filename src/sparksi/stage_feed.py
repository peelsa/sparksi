# SPDX-License-Identifier: Apache-2.0

"""Loopback stage that records another dashboard's monitor feed.

This is test tooling. It does not change a production ``sparksi serve``.
It binds ``127.0.0.1:7600``, does not listen for announce packets, and writes
only under the state directory you pass. Lock, history, telemetry, the token
ledger, and the job file stay in that directory. The upstream is read with GET ``/api/monitor`` only. ``GET /api/runs`` on the
upstream adopts containers and writes run files, so this process never calls
it. Run rows come from a local snapshot file, or are absent, and are labeled
stale.

    python -m sparksi.stage_feed --upstream http://127.0.0.1:7500 --state-dir /tmp/sparksi-stage

The upstream has to be loopback HTTP. The state directory must not be the
live controller state directory. Production config is read and not written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from sparksi import compat, protect

DEFAULT_PORT = 7600
FRESH_S = 5.0
_STOP_POLL = threading.Event()
_READ_ENV = {
    "SPARKSI_CONFIG",
    "SPARKSI_CATALOG",
    "SPARKSI_RECIPES_DIR",
    "SPARKSI_MODEL_INVENTORY",
    "DGX_MEM_CONFIG",
    "DGX_MEM_CATALOG",
    *compat.LEGACY_ENV_NAMES["SPARKSI_MODEL_INVENTORY"],
    "SPARKSI_PROTECT",  # protect presets stay on in the fresh state
    *protect.LEGACY_MARKERS["env"],
}
_FEED_PATHS = {"/api/monitor"}
_RUN_FIELDS = ("id", "model_id", "phase", "placement", "port", "started")
_NODE_FIELDS = ("role", "phase")


def _opener():
    from urllib.request import HTTPRedirectHandler

    class RefuseRedirect(HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
            raise URLError("redirect refused")

    # An empty proxy map ignores http_proxy and does not open another host.
    return build_opener(RefuseRedirect, ProxyHandler({}))


def production_writer_roots() -> list[Path]:
    """Every directory a writer might use. Call this before changing the environment.

    Path helpers that create directories are not called. The durable directory and
    the runtime directory are both included when they differ.
    """
    from sparksi.config import CFG, env, state_subdir
    from sparksi.jobs import job_path
    from sparksi.ledger import ledger_path as counter_ledger
    from sparksi.llama_day import ledger_path as llama_ledger
    from sparksi.lock import default_lock_path, state_dir
    from sparksi.recipes.runs import runs_dir

    roots: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved not in roots:
            roots.append(resolved)

    add(state_dir())
    add(default_lock_path().parent)
    add(job_path().parent)
    add(counter_ledger().parent)
    add(llama_ledger().parent)
    add(runs_dir())
    configured = str(CFG.controller.get("state_dir") or "")
    if configured:
        add(Path(configured))
    # Same parents history, activity, and the token ledger use, without creating them.
    state = env("SPARKSI_STATE_DIR")
    if state:
        add(Path(state))
    else:
        add(state_subdir(Path.home() / ".local" / "state"))
    home_state = Path.home() / ".local" / "state"
    add(home_state / "sparksi")
    add(home_state / "dgx-mem")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        add(Path(runtime) / "sparksi")
        add(Path(runtime) / "dgx-mem")
    uid_run = Path(f"/run/user/{os.getuid()}")
    if uid_run.is_dir():
        add(uid_run / "sparksi")
        add(uid_run / "dgx-mem")
    for key in ("SPARKSI_STATE_DIR", "DGX_MEM_STATE_DIR", "SPARKSI_LOCK", "DGX_MEM_LOCK"):
        value = os.environ.get(key)
        if value:
            add(Path(value))
            if key.endswith("LOCK"):
                add(Path(value).parent)
    return roots


def _overlaps(chosen: Path, live: Path) -> bool:
    if chosen == live:
        return True
    if live in chosen.parents or chosen in live.parents:
        return True
    return False


def isolated_state(path: Path) -> Path:
    chosen = path.expanduser().resolve()
    for live in production_writer_roots():
        if _overlaps(chosen, live):
            raise SystemExit("state directory must be separate from the live controller state")
    if chosen.exists():
        if not chosen.is_dir() or any(chosen.iterdir()):
            raise SystemExit("state directory must be a fresh empty directory")
    else:
        chosen.mkdir(parents=True)
    return chosen


def loopback_upstream(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "http" or host not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password:
        raise SystemExit("upstream must be loopback http")
    if parsed.path not in {"", "/"}:
        raise SystemExit("upstream must be loopback http")
    port = parsed.port or 80
    return f"http://{host}:{port}"


def _get_json(upstream: str, path: str) -> dict:
    if path not in _FEED_PATHS:
        raise ValueError(path)
    base = urlparse(loopback_upstream(upstream))
    url = f"http://{base.hostname}:{base.port}{path}"
    request = Request(url, method="GET")
    try:
        response = _opener().open(request, timeout=5)
    except HTTPError as exc:
        raise URLError(str(exc)) from exc
    with response:
        if request.get_method() != "GET":
            raise URLError("stage feed only uses GET")
        landed = urlparse(response.geturl())
        if landed.scheme != "http":
            raise URLError("upstream left loopback")
        if (landed.hostname or "").lower() != (base.hostname or "").lower():
            raise URLError("upstream left loopback")
        if landed.port != base.port or landed.path != path:
            raise URLError("unexpected path")
        if (landed.hostname or "").lower() not in {"127.0.0.1", "localhost"}:
            raise URLError("upstream left loopback")
        body = json.loads(response.read().decode("utf-8"))
    if not isinstance(body, dict):
        raise ValueError(path)
    return body


class Feed:
    """Last good observation, dropped as soon as the upstream misses a poll."""

    def __init__(self) -> None:
        self.card: dict = {"hosts": [], "loaded": []}
        self.runs: dict = {"runs": []}
        self.fetched_at = 0.0
        self.live = False

    def note_success(self, card: dict, runs: dict | None = None, now: float | None = None) -> None:
        self.card = card
        if runs is not None:
            self.runs = runs
        self.fetched_at = time.monotonic() if now is None else now
        self.live = True

    def note_failure(self) -> None:
        self.live = False

    def current_card(self, now: float | None = None) -> dict:
        moment = time.monotonic() if now is None else now
        if not self.live or moment - self.fetched_at > FRESH_S:
            return {"hosts": [], "loaded": [], "stale": True}
        return self.card

    def current_runs(self) -> dict:
        # The snapshot does not depend on the monitor poll. It is never a live adopt.
        body = dict(self.runs)
        body["stale"] = True
        return body


def _sanitize_run(row: Any) -> dict | None:
    if not isinstance(row, dict):
        return None
    run_id = str(row.get("id") or "")
    if not run_id:
        return None
    clean: dict = {"id": run_id}
    for key in _RUN_FIELDS:
        if key == "id" or key not in row:
            continue
        clean[key] = row.get(key)
    nodes = []
    for node in row.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nodes.append({key: node.get(key) for key in _NODE_FIELDS})
    clean["nodes"] = nodes
    return clean


def load_run_snapshot(path: Path | None, roots: list[Path]) -> dict:
    """Read a local run list. This does not call the live runs endpoint.

    ``roots`` is the production writer set captured before the environment changes.
    """
    if path is None:
        return {"runs": [], "stale": True, "source": "unavailable"}
    raw_path = path.expanduser()
    chosen = raw_path.resolve()
    for live in roots:
        if chosen == live or live in chosen.parents or chosen in live.parents:
            raise SystemExit("run snapshot must not be a live controller file")
    if raw_path.is_symlink() and not chosen.is_file():
        raise SystemExit("run snapshot must be a file")
    if not chosen.is_file():
        raise SystemExit("run snapshot must be a file")
    try:
        data = json.loads(chosen.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("run snapshot must contain a runs list") from exc
    rows = data.get("runs") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise SystemExit("run snapshot must contain a runs list")
    clean = [item for item in (_sanitize_run(row) for row in rows) if item]
    return {"runs": clean, "stale": True, "source": "snapshot"}


def install_feed(upstream: str, feed: Feed) -> None:
    """Point the collector at ``feed`` and keep filling it from ``upstream``."""
    from sparksi.serve import STATE

    STATE.monitor_fn = feed.current_card
    STATE.observe_fn = feed.current_card
    STATE.runs_fn = feed.current_runs

    def poll() -> None:
        while not _STOP_POLL.is_set():
            try:
                card = _get_json(upstream, "/api/monitor")
            except (OSError, URLError, ValueError, json.JSONDecodeError):
                feed.note_failure()
            else:
                feed.note_success(card, feed.runs)
            _STOP_POLL.wait(1)

    threading.Thread(target=poll, name="sparksi-stage-feed", daemon=True).start()


def scrub_env(state: Path) -> None:
    """Drop state overrides that point outside the fresh directory."""
    for key in list(os.environ):
        if key.startswith("SPARKSI_") or key.startswith("DGX_MEM_"):
            if key not in _READ_ENV:
                os.environ.pop(key, None)
    os.environ["SPARKSI_STATE_DIR"] = str(state)
    os.environ["SPARKSI_LOCK"] = str(state / "lock")


def run(
    upstream: str,
    state_dir: Path,
    port: int = DEFAULT_PORT,
    runs_snapshot: Path | None = None,
) -> None:
    upstream = loopback_upstream(upstream)
    roots = production_writer_roots()
    state = isolated_state(state_dir)
    snapshot = load_run_snapshot(runs_snapshot, roots)
    scrub_env(state)
    from sparksi.history import reset_for_tests
    from sparksi.serve import serve

    reset_for_tests()
    feed = Feed()
    feed.runs = snapshot
    try:
        feed.note_success(_get_json(upstream, "/api/monitor"), snapshot)
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        feed.note_failure()
    _STOP_POLL.clear()
    install_feed(upstream, feed)
    try:
        serve(port=int(port), hosts=["127.0.0.1"], announce=False, readonly=True)
    finally:
        _STOP_POLL.set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Loopback stage fed by a read-only dashboard")
    parser.add_argument("--upstream", required=True, help="loopback http dashboard, for example http://127.0.0.1:7500")
    parser.add_argument("--state-dir", required=True, help="empty directory for this process's state")
    parser.add_argument("--runs-snapshot", default="", help="local JSON file with a runs list; not the live runs endpoint")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)
    snapshot = Path(args.runs_snapshot) if args.runs_snapshot else None
    try:
        run(args.upstream, Path(args.state_dir), args.port, runs_snapshot=snapshot)
    except KeyboardInterrupt:
        _STOP_POLL.set()
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
