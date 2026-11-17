# SPDX-License-Identifier: Apache-2.0

"""Every state-changing POST takes the recipe-save rule: same-origin JSON, and no Origin (CLI) passes."""

from __future__ import annotations

import http.client
import json
import threading

import pytest

from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

GUARDED = [
    "/api/library/stage",
    "/api/library/drop",
    "/api/library/run",
    "/api/library/stop",
    "/api/switch",
    "/api/lock",
    "/api/unlock",
    "/api/apply",
    "/api/stop",
    "/api/job/pause",
    "/api/job/cancel",
    "/api/runs",
    "/api/runs/abc/stop",
    "/api/wanted",
    "/api/models/single-a/service",
]


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None


def _post(port: int, path: str, body: str, content_type: str | None, origin: str | None) -> tuple[int, dict]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    headers = {"Host": f"127.0.0.1:{port}"}
    if content_type:
        headers["Content-Type"] = content_type
    if origin is not None:
        headers["Origin"] = origin
    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode()
        try:
            return resp.status, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return resp.status, {"raw": raw}
    finally:
        conn.close()


@pytest.mark.parametrize("path", GUARDED)
def test_cross_origin_post_is_refused(server, path) -> None:
    code, body = _post(server, path, json.dumps({"id": "x", "from_ui": True}), "application/json", "http://evil.example")
    assert code == 403 and "cross-origin" in body.get("error", ""), (path, code, body)


@pytest.mark.parametrize("path", GUARDED)
def test_non_json_post_is_refused_even_without_origin(server, path) -> None:
    # A plain HTML form or a text/plain fetch needs no preflight; it must not reach the action.
    for ctype in ("text/plain", "application/x-www-form-urlencoded", None):
        code, body = _post(server, path, '{"id": "x", "from_ui": true}', ctype, None)
        assert code == 415, (path, ctype, code, body)


@pytest.mark.parametrize("path", ["/api/library/stage", "/api/library/drop", "/api/library/run", "/api/switch"])
def test_cli_and_same_origin_json_still_reach_the_action(server, path) -> None:
    for origin in (None, f"http://127.0.0.1:{server}"):
        code, body = _post(server, path, json.dumps({"from_ui": True}), "application/json", origin)
        # Past the guard: the action's own answer (missing id, or the lock), never 415 or cross-origin.
        assert code in (400, 403) and "cross-origin" not in body.get("error", ""), (path, origin, code, body)
        assert body.get("error") in ("missing id", "Unlock first."), (path, origin, body)


def test_cross_origin_unlock_leaves_the_lock_on(server) -> None:
    code, _ = _post(server, "/api/unlock", json.dumps({"from_ui": True}), "text/plain", "http://evil.example")
    assert code == 415
    code, _ = _post(server, "/api/unlock", json.dumps({"from_ui": True}), "application/json", "http://evil.example")
    assert code == 403
    assert STATE.lock.read().locked is True


@pytest.mark.parametrize("path", ["/api/switch/plan", "/api/models/single-a/plan"])
def test_plan_posts_need_json_too(server, path) -> None:
    code, _ = _post(server, path, json.dumps({"id": "nope"}), "text/plain", None)
    assert code == 415
    code, body = _post(server, path, json.dumps({"id": "nope"}), "application/json", None)
    assert code != 415 and "cross-origin" not in str(body.get("error", "")), body
