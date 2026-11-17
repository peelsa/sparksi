# SPDX-License-Identifier: Apache-2.0

"""Headless start and stop go through the controller HTTP API."""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from sparksi.cli import build_parser, main

SECRET = "example-unlock-secret"
_START_KEYS = {"ok", "action", "recipe", "already", "error", "run"}
_STOP_KEYS = {"ok", "action", "target", "already", "error", "run"}
_PLAN_KEYS = {"ok", "action", "recipe", "verdict", "conflicts", "port", "error"}
_RUN_KEYS = {"recipe", "id", "state", "port", "nodes", "started_by"}


def _sample(
    phase: str = "waiting_health",
    run_id: str = "run-1",
    recipe: str = "demo",
    port: int = 8000,
    started_by: str = "cli",
    error: str = "",
    nodes: list[dict[str, str]] | None = None,
) -> dict:
    return {
        "id": run_id,
        "model_id": recipe,
        "phase": phase,
        "port": port,
        "started": 1,
        "started_by": started_by,
        "error": error,
        "nodes": nodes if nodes is not None else [{"role": "head", "phase": phase}],
    }


class _Box:
    def __init__(self) -> None:
        self.base = ""
        self.state = ""
        self.locked = False
        self.models: dict = {"models": [{"id": "demo", "port": 8000}], "services": []}
        self.runs: list[dict] = []
        self.polls: list[dict] = []
        self.posts: list[dict] = []
        self.gets: list[str] = []
        self.start_code = 200
        self.start_body: dict | None = None
        self.stop_code = 200
        self.stop_body: dict | None = None
        self.plan_code = 200
        self.plan_body: dict = {"ok": True, "verdict": "READY", "conflicts": [], "port": 8000}
        self.redirect = ""


def _serve(box: _Box) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def log_message(self, format: str, *args) -> None:  # noqa: A003
            return

        def _handle(self, method: str) -> None:
            path = self.path.split("?", 1)[0]
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                body = {}
            header = self.headers.get("X-Sparksi-Token") or ""
            if method == "GET":
                box.gets.append(path)
            else:
                box.posts.append({"path": path, "body": body, "header": header})
            if method == "POST" and box.locked and header != SECRET and (path == "/api/runs" or path.endswith("/stop")):
                self._send(403, {"error": "Unlock first."})
                return
            if method == "GET" and path == "/api/models":
                self._send(200, box.models)
                return
            if method == "GET" and path == "/api/runs":
                self._send(200, {"runs": box.runs})
                return
            if method == "GET" and path.startswith("/api/runs/"):
                run = box.polls.pop(0) if box.polls else _sample(phase="waiting_health", run_id=path.rsplit("/", 1)[-1])
                self._send(200, {"run": run})
                return
            if method == "POST" and path.endswith("/plan"):
                self._send(box.plan_code, box.plan_body)
                return
            if method == "POST" and path == "/api/runs":
                if box.redirect:
                    self.send_response(302)
                    self.send_header("Location", box.redirect)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                payload = box.start_body if box.start_body is not None else {"ok": True, "run": _sample()}
                self._send(box.start_code, payload)
                return
            if method == "POST" and path.endswith("/stop"):
                payload = box.stop_body if box.stop_body is not None else {"run": _sample(phase="stopping")}
                self._send(box.stop_code, payload)
                return
            self._send(404, {"error": "not found"})

        def _send(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    box.base = "http://127.0.0.1:" + str(httpd.server_address[1])
    return httpd


@pytest.fixture
def api(tmp_path):
    box = _Box()
    box.state = str(tmp_path)
    httpd = _serve(box)
    try:
        yield box
    finally:
        httpd.shutdown()
        httpd.server_close()


def _call(box: _Box, argv: list[str], *, secret: str | None = None, secret_file: str | None = None, base: str | None = None):
    keys = ("SPARKSI_UNLOCK", "SPARKSI_UNLOCK_FILE", "SPARKSI_SERVER", "SPARKSI_STATE_DIR")
    saved = {key: os.environ.get(key) for key in keys}
    out, err = io.StringIO(), io.StringIO()
    try:
        for key in keys:
            os.environ.pop(key, None)
        os.environ["SPARKSI_STATE_DIR"] = box.state
        if secret is not None:
            os.environ["SPARKSI_UNLOCK"] = secret
        if secret_file is not None:
            os.environ["SPARKSI_UNLOCK_FILE"] = secret_file
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["--server", base or box.base, *argv])
        return code, out.getvalue(), err.getvalue()
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _hurry(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = {"t": 0.0}

    def monotonic() -> float:
        return clock["t"]

    def sleep(seconds: float) -> None:
        clock["t"] += seconds

    monkeypatch.setattr(time, "monotonic", monotonic)
    monkeypatch.setattr(time, "sleep", sleep)


def _quiet(text: str) -> None:
    assert SECRET not in text


def test_run_help_stays_available() -> None:
    help_text = build_parser().format_help()
    assert "legacy" in help_text
    parsed = build_parser().parse_args(["run", "demo"])
    assert parsed.cmd == "run"
    flags = build_parser().parse_args([
        "--server", "http://127.0.0.1:7500", "--unlock",
        "start", "demo", "--wait", "--timeout", "5", "--ctx", "1024", "--json",
    ])
    assert flags.server == "http://127.0.0.1:7500"
    assert flags.unlock is True
    assert flags.timeout == 5
    assert flags.ctx == 1024


def test_start_posts_the_same_request_as_the_dashboard(api: _Box) -> None:
    code, out, err = _call(api, ["start", "demo", "--ctx", "4096", "--json"])
    body = json.loads(out)
    assert code == 0
    assert err == ""
    assert set(body) == _START_KEYS
    assert body["ok"] is True
    assert body["already"] is False
    assert body["run"]["state"] == "waiting_health"
    assert api.posts[-1]["path"] == "/api/runs"
    assert api.posts[-1]["body"] == {"id": "demo", "ctx": 4096}
    assert "from_ui" not in api.posts[-1]["body"]
    assert api.posts[-1]["header"] == ""


def test_refusal_has_no_empty_run_and_401_means_unlock(api: _Box) -> None:
    api.start_code = 400
    api.start_body = {"ok": False, "conflicts": ["the head has only 2.5 GiB free right now and this model needs 21 GiB"]}
    code, out, _err = _call(api, ["start", "demo", "--json"])
    body = json.loads(out)
    assert code == 2
    assert "run" not in body
    assert " · " not in body["error"] or "free right now" in body["error"]

    api.start_code = 401
    api.start_body = {"error": "unauthorized"}
    code, out, _err = _call(api, ["start", "demo", "--json"])
    assert code == 2
    assert json.loads(out)["error"] == "Unlock first."
    assert "run" not in json.loads(out)

    api.stop_code = 401
    api.runs = [_sample(phase="ready")]
    code, out, _err = _call(api, ["stop", "demo", "--json"])
    assert code == 2
    assert json.loads(out)["error"] == "Unlock first."
    assert "run" not in json.loads(out)


def test_wait_without_a_run_id_fails_fast(api: _Box, monkeypatch: pytest.MonkeyPatch) -> None:
    _hurry(monkeypatch)
    api.start_body = {"ok": True, "run": {}}
    code, out, _err = _call(api, ["start", "demo", "--wait", "--timeout", "30", "--json"])
    assert code == 2
    assert "run id" in json.loads(out)["error"]
    assert not any(item.startswith("/api/runs/") for item in api.gets)


def test_start_refusal_exits_2(api: _Box) -> None:
    api.start_code = 400
    api.start_body = {
        "ok": False,
        "conflicts": ["the head has only 2.5 GiB free right now and this model needs 21 GiB"],
    }
    code, _out, err = _call(api, ["start", "demo"])
    assert code == 2
    assert "2.5 GiB free" in err

    api.posts.clear()
    api.runs = [_sample(phase="starting")]
    api.start_body = {
        "ok": False,
        "conflicts": ["demo is already loading — stop it first or use Restart with changes"],
    }
    code, out, _err = _call(api, ["start", "demo", "--json"])
    assert code == 2
    assert "already loading" in out
    assert json.loads(out)["error"]
    assert any(item["path"] == "/api/runs" for item in api.posts)


def test_lock_without_unlock_exits_2(api: _Box) -> None:
    api.locked = True
    code, out, err = _call(api, ["start", "demo", "--json"])
    body = json.loads(out)
    assert code == 2
    assert body["error"] == "Unlock first."
    assert err == ""
    _quiet(out)
    assert api.posts[-1]["header"] == ""

    api.runs = [_sample(phase="ready")]
    code, out, _err = _call(api, ["stop", "demo", "--json"])
    assert code == 2
    assert json.loads(out)["error"] == "Unlock first."

    code, out, _err = _call(api, ["--unlock", "stop", "demo", "--json"], secret="wrong-example-secret")
    assert code == 2
    assert json.loads(out)["error"] == "Unlock was refused."
    assert "wrong-example-secret" not in out


def test_wait_ready_failed_and_timeout(api: _Box, monkeypatch: pytest.MonkeyPatch) -> None:
    api.polls = [_sample(phase="ready")]
    code, out, _err = _call(api, ["start", "demo", "--wait", "--json"])
    assert code == 0
    assert json.loads(out)["run"]["state"] == "ready"

    api.polls = [_sample(phase="failed", error="weights missing")]
    code, out, _err = _call(api, ["start", "demo", "--wait", "--json"])
    assert code == 1
    assert json.loads(out)["error"] == "weights missing"

    api.polls = [_sample(phase="timed_out", error="health timed out")]
    code, out, _err = _call(api, ["start", "demo", "--wait", "--json"])
    assert code == 1
    assert "health timed out" in out

    api.polls = [_sample(phase="stopped")]
    code, _out, _err = _call(api, ["start", "demo", "--wait", "--json"])
    assert code == 1

    _hurry(monkeypatch)
    api.polls = []
    before = len(api.posts)
    code, out, _err = _call(api, ["start", "demo", "--wait", "--timeout", "1", "--json"])
    assert code == 3
    assert json.loads(out)["error"] == "timed out; the run was left running"
    assert not any(item["path"].endswith("/stop") for item in api.posts[before:])


def test_start_is_idempotent_when_ready(api: _Box) -> None:
    api.runs = [_sample(phase="ready", started_by="external")]
    code, out, err = _call(api, ["start", "demo"])
    assert code == 0
    assert "already running" in out
    assert err == ""
    assert not any(item["path"] == "/api/runs" for item in api.posts)

    code, out, _err = _call(api, ["start", "demo", "--json"])
    body = json.loads(out)
    assert code == 0
    assert body["already"] is True
    assert body["run"]["started_by"] == "external"
    assert set(body["run"]) == _RUN_KEYS


def test_stop_idempotent_and_wait_timeout(api: _Box, monkeypatch: pytest.MonkeyPatch) -> None:
    api.runs = [_sample(phase="stopped")]
    code, out, _err = _call(api, ["stop", "demo", "--json"])
    assert code == 0
    assert json.loads(out)["already"] is True
    assert not api.posts

    api.runs = []
    code, out, _err = _call(api, ["stop", "missing", "--json"])
    assert code == 0
    assert "not running" in out or json.loads(out)["already"] is True

    _hurry(monkeypatch)
    api.runs = [_sample(phase="ready")]
    api.polls = []
    code, out, _err = _call(api, ["stop", "run-1", "--wait", "--timeout", "1", "--json"])
    assert code == 3
    assert json.loads(out)["error"] == "timed out; the run was left as it is"
    assert [item["path"] for item in api.posts] == ["/api/runs/run-1/stop"]


def test_stop_reaches_stopped(api: _Box) -> None:
    nodes = [{"role": "head", "phase": "ready"}, {"role": "worker", "phase": "ready"}]
    api.runs = [_sample(phase="ready", nodes=nodes)]
    api.polls = [_sample(phase="stopped", nodes=[{"role": "head", "phase": "stopped"}, {"role": "worker", "phase": "stopped"}])]
    code, out, _err = _call(api, ["stop", "demo", "--wait", "--json"])
    body = json.loads(out)
    assert code == 0
    assert set(body) == _STOP_KEYS
    assert body["already"] is False
    assert body["run"]["state"] == "stopped"
    assert [item["path"] for item in api.posts] == ["/api/runs/run-1/stop"]


def test_port_8188_is_refused(api: _Box) -> None:
    api.models = {
        "models": [],
        "services": [{"id": "draw", "port": 8188, "protected": True, "kind": "comfy"}],
    }
    for argv in (["start", "comfy", "--json"], ["start", "draw", "--json"], ["stop", "draw", "--json"], ["plan", "comfy", "--json"]):
        code, out, _err = _call(api, argv)
        assert code == 2
        assert json.loads(out)["error"] == "Port 8188 is off limits."
    api.runs = [_sample(phase="ready", port=8188, recipe="draw")]
    code, out, _err = _call(api, ["stop", "run-1", "--json"])
    assert code == 2
    assert "8188" in out
    assert api.posts == []


def test_plan_and_start_json_keep_reason_codes(api: _Box) -> None:
    reasons = [{"code": "memory_not_free", "message": "the head has only 2.5 GiB free right now and this model needs 21 GiB plus 5 GiB margin"}]
    api.plan_code = 400
    api.plan_body = {"ok": False, "verdict": "REFUSE", "conflicts": [reasons[0]["message"]], "reasons": reasons, "port": 8010}
    code, out, _err = _call(api, ["plan", "demo", "--json"])
    body = json.loads(out)
    assert code == 2
    assert body["reasons"] == reasons
    assert body["reasons"][0]["code"] == "memory_not_free"

    api.start_code = 400
    api.start_body = {"ok": False, "conflicts": [reasons[0]["message"]], "reasons": reasons}
    code, out, _err = _call(api, ["start", "demo", "--json"])
    started = json.loads(out)
    assert code == 2
    assert started["reasons"] == reasons
    assert "run" not in started


def test_plan_does_not_start_and_runs_lists_fields(api: _Box) -> None:
    code, out, _err = _call(api, ["plan", "demo", "--ctx", "2048", "--json"])
    body = json.loads(out)
    assert code == 0
    assert set(body) == _PLAN_KEYS
    assert body["verdict"] == "READY"
    assert body["port"] == 8000
    assert api.posts[-1]["path"].endswith("/plan")
    assert api.posts[-1]["body"] == {"id": "demo", "ctx": 2048}
    assert not any(item["path"] == "/api/runs" for item in api.posts)

    api.plan_code = 400
    api.plan_body = {"ok": False, "verdict": "REFUSE", "conflicts": ["port 8000 is in use"], "port": 8000}
    code, out, _err = _call(api, ["plan", "demo", "--json"])
    assert code == 2
    assert "in use" in out

    api.runs = [
        _sample(
            phase="ready",
            started_by="external",
            nodes=[{"role": "head", "phase": "ready"}, {"role": "worker", "phase": "ready"}],
        ),
        _sample(phase="failed", run_id="old"),
    ]
    code, out, err = _call(api, ["runs"])
    assert code == 0
    assert err == ""
    assert "demo  run-1  ready  port 8000  head, worker  external" in out
    assert "old" not in out

    code, out, _err = _call(api, ["runs", "--json"])
    body = json.loads(out)
    assert body["ok"] is True
    assert len(body["runs"]) == 1
    assert set(body["runs"][0]) == _RUN_KEYS
    assert body["runs"][0]["nodes"] == [
        {"role": "head", "state": "ready"},
        {"role": "worker", "state": "ready"},
    ]
    assert out.index('"action"') < out.index('"runs"')


def test_unlock_secret_is_sent_and_never_printed(api: _Box, tmp_path) -> None:
    argv = ["--unlock", "start", "demo", "--json"]
    assert SECRET not in argv
    api.locked = True
    code, out, err = _call(api, argv, secret=SECRET)
    assert code == 0
    assert api.posts[-1]["header"] == SECRET
    assert SECRET not in api.posts[-1]["path"]
    assert SECRET not in json.dumps(api.posts[-1]["body"])
    _quiet(out)
    _quiet(err)

    api.posts.clear()
    api.start_code = 400
    api.start_body = {"ok": False, "error": "nope " + SECRET}
    code, out, err = _call(api, ["--unlock", "start", "demo"], secret=SECRET)
    assert code == 2
    assert "nope" in err
    _quiet(out)
    _quiet(err)

    path = tmp_path / "unlock.file"
    path.write_text(SECRET + "\n", encoding="utf-8")
    api.start_code = 200
    api.start_body = None
    api.posts.clear()
    code, out, err = _call(api, argv, secret_file=str(path))
    assert code == 0
    assert api.posts[-1]["header"] == SECRET
    _quiet(out)
    _quiet(err)
    assert str(path) not in out

    before = len(api.posts)
    code, out, _err = _call(api, argv)
    assert code == 2
    assert json.loads(out)["error"] == "No unlock secret is available."
    assert len(api.posts) == before


def test_controller_errors_exit_2(api: _Box) -> None:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    code, out, _err = _call(api, ["runs", "--json"], base="http://127.0.0.1:" + str(port))
    assert code == 2
    assert json.loads(out)["error"] == "Could not reach the controller."

    code, _out, err = _call(api, ["runs"], base="http://user:pass@127.0.0.1:9")
    assert code == 2
    assert "must not carry a secret" in err

    api.redirect = "http://127.0.0.1:9/collect"
    code, _out, _err = _call(api, ["--unlock", "start", "demo", "--json"], secret=SECRET)
    assert code == 2
    assert "/collect" not in api.gets
    assert [item["path"] for item in api.posts if item["path"] == "/api/runs"] == ["/api/runs"]
