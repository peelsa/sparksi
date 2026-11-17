# SPDX-License-Identifier: Apache-2.0

"""Headless start and stop through the controller HTTP API.

The CLI never talks to Docker. The running controller applies the same lock,
memory, port, duplicate, and adoption checks as the dashboard.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote

from sparksi import protect
from sparksi.config import CFG
from sparksi.lock import Lock

_QUIET = frozenset({"stopped", "cancelled", "failed", "timed_out"})
_FAIL = frozenset({"failed", "timed_out"})
_POLL_S = 1.5


def default_server() -> str:
    """Loopback controller. ``SPARKSI_SERVER`` overrides the configured UI port."""
    chosen = (os.environ.get("SPARKSI_SERVER") or "").strip()
    if chosen:
        return chosen.rstrip("/")
    try:
        port = int(CFG.controller.get("ui_port") or 7500)
    except (TypeError, ValueError):
        port = 7500
    return "http://127.0.0.1:" + str(port)


def _read_secret() -> str:
    """Unlock secret from the environment, an explicit file, or the lock file.

    The value is never logged. Callers must not place it on a command line.
    """
    from_env = (os.environ.get("SPARKSI_UNLOCK") or "").strip()
    if from_env:
        return from_env
    explicit = (os.environ.get("SPARKSI_UNLOCK_FILE") or "").strip()
    path = Path(explicit).expanduser() if explicit else Lock().token_path()
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Keep the unlock header on the controller. A redirect must not be followed."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ARG002
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "")
    return text


def _scrub(value: Any, secret: str) -> Any:
    if not secret:
        return value
    if isinstance(value, str):
        return value.replace(secret, "")
    if isinstance(value, list):
        return [_scrub(item, secret) for item in value]
    if isinstance(value, dict):
        return {str(key): _scrub(item, secret) for key, item in value.items()}
    return value


def _base(raw: str) -> str:
    text = (raw or "").strip() or default_server()
    text = text.rstrip("/")
    if not text.startswith(("http://", "https://")) or any(ch in text for ch in " \n\r"):
        raise ValueError("server must be an http address")
    if "@" in text.split("://", 1)[-1]:
        raise ValueError("server address must not carry a secret")
    return text


class Controller:
    def __init__(self, base: str, secret: str = "") -> None:
        self.base = base
        self.secret = secret

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
        payload = None if body is None else json.dumps(body).encode()
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self.secret:
            headers["X-Sparksi-Token"] = self.secret
        req = urllib.request.Request(self.base + path, data=payload, headers=headers, method=method)
        opener = urllib.request.build_opener(_NoRedirect)
        try:
            with opener.open(req, timeout=30) as resp:
                return resp.status, _decode(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, _decode(exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ConnectionError("Could not reach the controller.") from exc

    def models(self) -> dict[str, Any]:
        _code, body = self._request("GET", "/api/models")
        return body

    def runs(self) -> list[dict[str, Any]]:
        _code, body = self._request("GET", "/api/runs")
        rows = body.get("runs") if isinstance(body, dict) else None
        return [item for item in rows or [] if isinstance(item, dict)]

    def run(self, run_id: str) -> tuple[int, dict[str, Any]]:
        return self._request("GET", "/api/runs/" + quote(run_id, safe=""))

    def plan(self, recipe_id: str, ctx: int | None) -> tuple[int, dict[str, Any]]:
        return self._request("POST", "/api/models/" + quote(recipe_id, safe="") + "/plan", _body(recipe_id, ctx))

    def start(self, recipe_id: str, ctx: int | None) -> tuple[int, dict[str, Any]]:
        return self._request("POST", "/api/runs", _body(recipe_id, ctx))

    def stop(self, run_id: str) -> tuple[int, dict[str, Any]]:
        return self._request("POST", "/api/runs/" + quote(run_id, safe="") + "/stop", {})


def _decode(raw: bytes) -> dict[str, Any]:
    try:
        data = json.loads(raw.decode() or "{}")
    except (json.JSONDecodeError, UnicodeError):
        return {"error": "the controller returned an error"}
    return data if isinstance(data, dict) else {"error": "the controller returned an error"}


def _body(recipe_id: str, ctx: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {"id": recipe_id}
    if ctx is not None:
        body["ctx"] = int(ctx)
    return body


def _reason(payload: dict[str, Any]) -> str:
    conflicts = payload.get("conflicts") if isinstance(payload.get("conflicts"), list) else []
    parts = [str(item).strip() for item in conflicts if str(item or "").strip()]
    if parts:
        return " · ".join(parts)
    return str(payload.get("error") or "refused")


def _present(run: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for node in run.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        nodes.append({"role": node.get("role") or "", "state": node.get("phase") or ""})
    return {
        "recipe": run.get("model_id") or "",
        "id": run.get("id") or "",
        "state": run.get("phase") or "",
        "port": run.get("port"),
        "nodes": nodes,
        "started_by": run.get("started_by") or "",
    }


def _reasons(payload: dict[str, Any] | None) -> list[dict[str, str]]:
    """Reason codes from the controller, kept as ``code`` and ``message``."""
    raw = payload.get("reasons") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    found: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if not code:
            continue
        found.append({"code": code, "message": str(item.get("message") or "")})
    return found


def _start_payload(
    recipe_id: str,
    *,
    ok: bool,
    already: bool = False,
    error: str = "",
    run: dict[str, Any] | None = None,
    reasons: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": ok,
        "action": "start",
        "recipe": recipe_id,
        "already": already,
        "error": error,
    }
    # A refusal has no run. An empty object used to sit in the payload.
    if run:
        body["run"] = run
    if reasons:
        body["reasons"] = reasons
    return body


def _stop_payload(
    target: str,
    *,
    ok: bool,
    already: bool = False,
    error: str = "",
    run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": ok,
        "action": "stop",
        "target": target,
        "already": already,
        "error": error,
    }
    if run:
        body["run"] = run
    return body


def _plan_payload(
    recipe_id: str,
    *,
    ok: bool,
    verdict: str = "",
    conflicts: list[Any] | None = None,
    port: Any = None,
    error: str = "",
    reasons: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": ok,
        "action": "plan",
        "recipe": recipe_id,
        "verdict": verdict,
        "conflicts": list(conflicts or []),
        "port": port,
        "error": error,
    }
    if reasons:
        body["reasons"] = reasons
    return body


def _emit(payload: dict[str, Any], as_json: bool, text: str, *, secret: str, ok: bool) -> None:
    cleaned = _scrub(payload, secret)
    if as_json:
        print(json.dumps(cleaned, indent=2, sort_keys=True))
        return
    stream = sys.stdout if ok else sys.stderr
    stream.write(_redact(text, secret) + "\n")


def _client(args: Any) -> tuple[Controller | None, str]:
    try:
        base = _base(getattr(args, "server", "") or "")
    except ValueError as exc:
        return None, str(exc)
    secret = ""
    if getattr(args, "unlock", False):
        secret = _read_secret()
        if not secret:
            return None, "No unlock secret is available."
    return Controller(base, secret), ""


def _forbidden(controller: Controller, recipe_id: str) -> str:
    if protect.is_video_id(recipe_id):
        return "Port 8188 is off limits."
    try:
        listing = controller.models()
    except ConnectionError:
        raise
    for bucket in ("models", "services"):
        for card in listing.get(bucket) or []:
            if not isinstance(card, dict) or str(card.get("id") or "") != recipe_id:
                continue
            try:
                port = int(card.get("port") or 0)
            except (TypeError, ValueError):
                port = 0
            if card.get("protected") or protect.is_off_limits(port) or protect.is_video_kind(card.get("kind")):
                return "Port 8188 is off limits."
    return ""


def _newest(runs: list[dict[str, Any]], recipe_id: str) -> dict[str, Any] | None:
    found = None
    for run in runs:
        if str(run.get("model_id") or "") != recipe_id:
            continue
        if found is None or float(run.get("started") or 0) >= float(found.get("started") or 0):
            found = run
    return found


def _active(runs: list[dict[str, Any]], recipe_id: str) -> dict[str, Any] | None:
    found = None
    for run in runs:
        if str(run.get("model_id") or "") != recipe_id:
            continue
        if str(run.get("phase") or "") in _QUIET:
            continue
        if found is None or float(run.get("started") or 0) >= float(found.get("started") or 0):
            found = run
    return found


def _by_id(runs: list[dict[str, Any]], run_id: str) -> dict[str, Any] | None:
    for run in runs:
        if str(run.get("id") or "") == run_id:
            return run
    return None


def _port_of(run: dict[str, Any]) -> int:
    try:
        return int(run.get("port") or 0)
    except (TypeError, ValueError):
        return 0


def cmd_runs(args: Any) -> int:
    controller, why = _client(args)
    if controller is None:
        _emit({"ok": False, "action": "runs", "error": why, "runs": []}, args.json, why, secret="", ok=False)
        return 2
    try:
        _code, body = controller._request("GET", "/api/runs")
    except ConnectionError as exc:
        _emit({"ok": False, "action": "runs", "error": str(exc), "runs": []}, args.json, str(exc), secret=controller.secret, ok=False)
        return 2
    raw = body.get("runs") if isinstance(body, dict) else None
    rows = [_present(run) for run in raw or [] if isinstance(run, dict) and str(run.get("phase") or "") not in _QUIET]
    serving = body.get("serving") if isinstance(body, dict) else None
    serving_rows = [item for item in serving or [] if isinstance(item, dict)]
    payload = {"ok": True, "action": "runs", "error": "", "runs": rows, "serving": serving_rows}
    if args.json:
        _emit(payload, True, "", secret=controller.secret, ok=True)
    else:
        live = [item for item in serving_rows if item.get("state") not in (None, "", "idle")]
        if not rows and not live:
            print("no runs")
        for run in rows:
            where = ", ".join(node["role"] for node in run["nodes"] if node["role"]) or "head"
            print(f"{run['recipe']}  {run['id']}  {run['state']}  port {run['port']}  {where}  {run['started_by']}")
        for item in live:
            print(f"{item.get('role') or ''}  {item.get('state') or ''}  {item.get('text') or ''}")
    return 0


def cmd_plan(args: Any) -> int:
    controller, why = _client(args)
    if controller is None:
        _emit(_plan_payload(args.recipe_id, ok=False, error=why), args.json, why, secret="", ok=False)
        return 2
    try:
        blocked = _forbidden(controller, args.recipe_id)
        if blocked:
            _emit(_plan_payload(args.recipe_id, ok=False, error=blocked), args.json, blocked, secret=controller.secret, ok=False)
            return 2
        code, planned = controller.plan(args.recipe_id, getattr(args, "ctx", None))
    except ConnectionError as exc:
        _emit(_plan_payload(args.recipe_id, ok=False, error=str(exc)), args.json, str(exc), secret=controller.secret, ok=False)
        return 2
    verdict = str(planned.get("verdict") or "")
    conflicts = planned.get("conflicts") if isinstance(planned.get("conflicts"), list) else []
    ok = 200 <= code < 300 and verdict == "READY"
    payload = _plan_payload(
        args.recipe_id,
        ok=ok,
        verdict=verdict or ("READY" if ok else "REFUSE"),
        conflicts=conflicts,
        port=planned.get("port"),
        error="" if ok else _reason(planned),
        reasons=_reasons(planned),
    )
    text = "ready" if ok else _reason(planned)
    _emit(payload, args.json, text, secret=controller.secret, ok=ok)
    return 0 if ok else 2


def cmd_start(args: Any) -> int:
    controller, why = _client(args)
    recipe_id = args.recipe_id
    if controller is None:
        _emit(_start_payload(recipe_id, ok=False, error=why), args.json, why, secret="", ok=False)
        return 2
    try:
        blocked = _forbidden(controller, recipe_id)
        if blocked:
            return _refused(args, controller, recipe_id, blocked)
        current = _active(controller.runs(), recipe_id)
        if current and str(current.get("phase") or "") == "ready":
            payload = _start_payload(recipe_id, ok=True, already=True, run=_present(current))
            _emit(payload, args.json, f"{recipe_id} is already running", secret=controller.secret, ok=True)
            return 0
        code, started = controller.start(recipe_id, getattr(args, "ctx", None))
    except ConnectionError as exc:
        _emit(_start_payload(recipe_id, ok=False, error=str(exc)), args.json, str(exc), secret=controller.secret, ok=False)
        return 2
    if code in (401, 403):
        message = "Unlock first." if not args.unlock else "Unlock was refused."
        return _refused(args, controller, recipe_id, message)
    if not 200 <= code < 300 or not started.get("ok", False):
        return _refused(args, controller, recipe_id, _reason(started), reasons=_reasons(started))
    run = started.get("run") if isinstance(started.get("run"), dict) else {}
    if not args.wait:
        payload = _start_payload(recipe_id, ok=True, run=_present(run))
        _emit(payload, args.json, f"started {recipe_id}", secret=controller.secret, ok=True)
        return 0
    return _wait_ready(args, controller, recipe_id, str(run.get("id") or ""))


def _wait_ready(args: Any, controller: Controller, recipe_id: str, run_id: str) -> int:
    if not run_id:
        message = "the start did not return a run id"
        payload = _start_payload(recipe_id, ok=False, error=message)
        _emit(payload, args.json, message, secret=controller.secret, ok=False)
        return 2
    deadline = time.monotonic() + max(1, int(args.timeout))
    last: dict[str, Any] = {}
    while True:
        try:
            _code, body = controller.run(run_id) if run_id else (200, {})
        except ConnectionError as exc:
            _emit(_start_payload(recipe_id, ok=False, error=str(exc)), args.json, str(exc), secret=controller.secret, ok=False)
            return 2
        run = body.get("run") if isinstance(body.get("run"), dict) else body
        if isinstance(run, dict):
            last = run
        phase = str((run or {}).get("phase") or "")
        if phase == "ready":
            payload = _start_payload(recipe_id, ok=True, run=_present(run))
            _emit(payload, args.json, f"{recipe_id} is ready", secret=controller.secret, ok=True)
            return 0
        if phase in _FAIL or phase in ("stopped", "cancelled"):
            why = str(run.get("error") or run.get("health_note") or phase or "the run stopped")
            payload = _start_payload(recipe_id, ok=False, error=why, run=_present(run))
            _emit(payload, args.json, why, secret=controller.secret, ok=False)
            return 1
        if time.monotonic() >= deadline:
            message = "timed out; the run was left running"
            payload = _start_payload(recipe_id, ok=False, error=message, run=_present(last))
            _emit(payload, args.json, message, secret=controller.secret, ok=False)
            return 3
        time.sleep(_POLL_S)


def cmd_stop(args: Any) -> int:
    controller, why = _client(args)
    target = args.target
    if controller is None:
        _emit(_stop_payload(target, ok=False, error=why), args.json, why, secret="", ok=False)
        return 2
    try:
        if protect.is_video_id(target):
            return _refused_stop(args, controller, target, "Port 8188 is off limits.")
        runs = controller.runs()
        chosen = _by_id(runs, target) or _active(runs, target) or _newest(runs, target)
        if chosen and protect.is_off_limits(_port_of(chosen)):
            return _refused_stop(args, controller, target, "Port 8188 is off limits.")
        blocked = _forbidden(controller, str((chosen or {}).get("model_id") or target))
        if blocked:
            return _refused_stop(args, controller, target, blocked)
        if chosen is None or str(chosen.get("phase") or "") in _QUIET:
            payload = _stop_payload(target, ok=True, already=True, run=_present(chosen or {}))
            _emit(payload, args.json, f"{target} is not running", secret=controller.secret, ok=True)
            return 0
        code, stopped = controller.stop(str(chosen.get("id") or ""))
    except ConnectionError as exc:
        _emit(_stop_payload(target, ok=False, error=str(exc)), args.json, str(exc), secret=controller.secret, ok=False)
        return 2
    if code in (401, 403):
        message = "Unlock first." if not args.unlock else "Unlock was refused."
        return _refused_stop(args, controller, target, message)
    if not 200 <= code < 300:
        return _refused_stop(args, controller, target, _reason(stopped))
    run = stopped.get("run") if isinstance(stopped.get("run"), dict) else chosen
    if not args.wait:
        payload = _stop_payload(target, ok=True, run=_present(run or {}))
        _emit(payload, args.json, f"stopped {target}", secret=controller.secret, ok=True)
        return 0
    return _wait_stopped(args, controller, target, str((run or {}).get("id") or chosen.get("id") or ""))


def _wait_stopped(args: Any, controller: Controller, target: str, run_id: str) -> int:
    if not run_id:
        message = "the stop did not return a run id"
        payload = _stop_payload(target, ok=False, error=message)
        _emit(payload, args.json, message, secret=controller.secret, ok=False)
        return 2
    deadline = time.monotonic() + max(1, int(args.timeout))
    while True:
        try:
            _code, body = controller.run(run_id)
        except ConnectionError as exc:
            _emit(_stop_payload(target, ok=False, error=str(exc)), args.json, str(exc), secret=controller.secret, ok=False)
            return 2
        run = body.get("run") if isinstance(body.get("run"), dict) else body
        phase = str((run or {}).get("phase") or "")
        if phase in _QUIET or phase == "stopped":
            payload = _stop_payload(target, ok=True, run=_present(run or {}))
            _emit(payload, args.json, f"stopped {target}", secret=controller.secret, ok=True)
            return 0
        if time.monotonic() >= deadline:
            message = "timed out; the run was left as it is"
            payload = _stop_payload(target, ok=False, error=message, run=_present(run or {}))
            _emit(payload, args.json, message, secret=controller.secret, ok=False)
            return 3
        time.sleep(_POLL_S)


def _refused(
    args: Any,
    controller: Controller,
    recipe_id: str,
    message: str,
    reasons: list[dict[str, str]] | None = None,
) -> int:
    payload = _start_payload(recipe_id, ok=False, error=message, reasons=reasons)
    _emit(payload, args.json, message, secret=controller.secret, ok=False)
    return 2


def _refused_stop(args: Any, controller: Controller, target: str, message: str) -> int:
    payload = _stop_payload(target, ok=False, error=message)
    _emit(payload, args.json, message, secret=controller.secret, ok=False)
    return 2
