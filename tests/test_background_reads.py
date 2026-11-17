# SPDX-License-Identifier: Apache-2.0

"""Reads that reach the Sparks answer from the last result; the refresh runs in the background."""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from urllib.request import Request, urlopen

import pytest

import sparksi.remote as remote_mod
import sparksi.serve as serve_mod
from sparksi.lock import Lock
from sparksi.refresh import PENDING, Refreshing
from sparksi.serve import STATE, DualServer, Handler


def test_refreshing_serves_the_last_value_while_a_slow_refresh_runs() -> None:
    clock = [100.0]
    cache = Refreshing(ttl=5, first_wait=0.2, clock=lambda: clock[0])
    gate = threading.Event()
    calls = []

    def slow():
        calls.append(1)
        if len(calls) > 1:
            gate.wait(5)
        return len(calls)

    assert cache.get("k", slow) == 1  # the first read waits (fast here)
    clock[0] += 6
    t0 = time.perf_counter()
    for _ in range(10):
        assert cache.get("k", slow) == 1  # old value at once; one refresh running
    assert time.perf_counter() - t0 < 0.5 and len(calls) == 2
    gate.set()
    for _ in range(50):
        if cache.get("k", slow) == 2:
            break
        time.sleep(0.02)
    assert cache.get("k", slow) == 2


def test_first_read_gives_up_after_first_wait_and_failures_back_off() -> None:
    clock = [100.0]
    cache = Refreshing(ttl=5, first_wait=0.1, clock=lambda: clock[0])
    gate = threading.Event()
    t0 = time.perf_counter()
    assert cache.get("slow", lambda: gate.wait(5) and "late") is PENDING
    assert time.perf_counter() - t0 < 0.5
    gate.set()

    fails = []

    def down():
        fails.append(1)
        raise OSError("no route to host")

    assert cache.get("down", down) is PENDING
    time.sleep(0.05)
    for _ in range(5):
        assert cache.get("down", down) is PENDING
    assert len(fails) == 1  # a machine that is down is retried once per ttl, not per read
    clock[0] += 6
    cache.get("down", down)
    time.sleep(0.05)
    assert len(fails) == 2


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setattr(serve_mod.HOSTS, "names", lambda: {"localhost", "127.0.0.1"})
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None


def _timed(port: int, path: str) -> tuple[float, dict]:
    t0 = time.perf_counter()
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=20) as resp:
        body = json.loads(resp.read())
    return time.perf_counter() - t0, body


def test_node_reads_never_wait_for_slow_nodes(server, monkeypatch) -> None:
    gate = threading.Event()
    node = {"id": "n1", "name": "n1", "role": "head", "reachable": True, "mem": None}
    monkeypatch.setattr(serve_mod, "node_report", lambda cat, runner, force=False: gate.wait(10) and {"nodes": [node], "cached": False})
    monkeypatch.setattr(serve_mod, "run_readonly", lambda argv: (time.sleep(3), (0, "", ""))[1])  # docker stats samples for seconds
    monkeypatch.setattr(serve_mod, "fleet", lambda cat, runner: gate.wait(10) and {"library": ["x"], "machines": [], "hosts": [], "storage": []})
    for path in ("/api/nodes", "/api/status", "/api/fleet"):
        took, body = _timed(server, path)
        assert took < 1.5, (path, took)
        if path != "/api/status":
            assert body.get("pending") is True, (path, body)
    took, _ = _timed(server, "/api/nodes")
    assert took < 0.9  # still cold: bounded by first_wait, never the node timeout
    gate.set()
    for _ in range(100):
        _, body = _timed(server, "/api/nodes")
        if body.get("nodes"):
            break
        time.sleep(0.05)
    assert body["nodes"] == [node]
    _, fleet_view = _timed(server, "/api/fleet")
    assert fleet_view["library"] == ["x"]


def test_a_write_marks_reads_old(server, monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(serve_mod, "node_report", lambda cat, runner, force=False: seen.append(1) or {"nodes": [], "cached": False})
    _timed(server, "/api/nodes")
    _timed(server, "/api/nodes")
    assert len(seen) == 1  # fresh: served from the cache
    req = Request(f"http://127.0.0.1:{server}/api/lock", data=b"{}", headers={"Content-Type": "application/json"}, method="POST")
    try:
        urlopen(req, timeout=10).read()
    except Exception:  # noqa: BLE001 — the answer does not matter, only that a write happened
        pass
    _timed(server, "/api/nodes")
    for _ in range(50):
        if len(seen) == 2:
            break
        time.sleep(0.02)
    assert len(seen) == 2


def test_slow_requests_are_logged_with_path_status_time_and_host(server, monkeypatch, caplog) -> None:
    monkeypatch.setattr(serve_mod, "SLOW_REQUEST_S", 0.0)
    with caplog.at_level(logging.INFO, logger="sparksi"):
        _timed(server, "/api/config")
        time.sleep(0.05)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("http ")]
    assert any(line.startswith("http GET /api/config 200 ") and line.endswith(f"host=127.0.0.1:{server}") for line in lines), lines
    monkeypatch.setenv("SPARKSI_ACCESS_LOG", "off")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="sparksi"):
        _timed(server, "/api/config")
        time.sleep(0.05)
    assert not [r for r in caplog.records if r.getMessage().startswith("http ")]


def test_an_unreachable_node_is_not_retried_for_reads(monkeypatch) -> None:
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 255, "", "ssh: Could not resolve hostname node-x.invalid")

    monkeypatch.setattr(remote_mod.subprocess, "run", run)
    monkeypatch.setattr(remote_mod, "resolve_target", lambda host, catalog: "node-x.invalid")
    runner = remote_mod.SshRunner({})
    first = runner.invoke("node-x", {"op": "probe"}, timeout=5)
    again = runner.invoke("node-x", {"op": "probe"}, timeout=5)
    assert first["ok"] is False and again["ok"] is False and again.get("unreachable") is True
    assert len(calls) == 1  # the second read did not wait on ssh again
    assert "Could not resolve hostname" in again["error"]
    runner.invoke("node-x", {"op": "stage_flat"}, timeout=5)
    assert len(calls) == 2  # actions always try


def test_recipe_preview_probes_share_recent_answers() -> None:
    calls = []

    class Runner:
        def invoke(self, host, req, timeout=0):
            calls.append((host, req["op"]))
            time.sleep(0.2)
            return {"ok": True, "host": host, "op": req["op"]}

    real = Runner()
    first = serve_mod.CachedReads(real)
    second = serve_mod.CachedReads(real)  # a later request wraps the same runner
    assert first.invoke("h1", {"op": "probe", "ports": [1]}, timeout=5)["ok"] is True
    t0 = time.perf_counter()
    assert second.invoke("h1", {"ports": [1], "op": "probe"}, timeout=5)["ok"] is True
    assert time.perf_counter() - t0 < 0.1 and calls == [("h1", "probe")]
    second.invoke("h2", {"op": "probe", "ports": [1]}, timeout=5)
    second.invoke("h1", {"op": "stage_flat"}, timeout=5)
    second.invoke("h1", {"op": "stage_flat"}, timeout=5)
    assert calls == [("h1", "probe"), ("h2", "probe"), ("h1", "stage_flat"), ("h1", "stage_flat")]  # actions never cached


def test_read_only_captures_share_recent_answers() -> None:
    reads = []

    class Runner:
        def invoke(self, host, req, timeout=0):
            return {"ok": True}

        def capture(self, host, argv, timeout=8):
            reads.append((host, tuple(argv)))
            return {"ok": True, "code": 0, "stdout": "text", "stderr": "", "error": None}

    real = Runner()
    for _ in range(3):
        assert serve_mod.CachedReads(real).capture("h1", ["cat", "/x/start.sh"], 8)["stdout"] == "text"
    serve_mod.CachedReads(real).capture("h1", ["cat", "/x/other.sh"], 8)
    assert reads == [("h1", ("cat", "/x/start.sh")), ("h1", ("cat", "/x/other.sh"))]


def _post(port, path, body):
    data = json.dumps(body).encode("utf-8")
    req = Request(
        "http://127.0.0.1:%d%s" % (port, path),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Origin": "http://127.0.0.1:%d" % port,
        },
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=10)
        payload = resp.read()
        code = resp.getcode()
    except Exception as exc:  # HTTPError and friends: reply code does not matter
        try:
            payload = exc.read()
        except Exception:
            payload = b""
        code = getattr(exc, "code", None)
    try:
        parsed = json.loads(payload.decode("utf-8"))
    except Exception:
        parsed = {}
    return code, parsed


def test_plan_previews_and_scans_do_not_mark_reads_stale(server, monkeypatch):
    port = server
    counters = {"reads": 0, "slow": 0}

    def fake_reads_expire():
        counters["reads"] += 1

    def fake_slow_expire():
        counters["slow"] += 1

    monkeypatch.setattr(serve_mod.READS, "expire", fake_reads_expire)
    monkeypatch.setattr(serve_mod.SLOW_READS, "expire", fake_slow_expire)
    monkeypatch.setattr(serve_mod.discover, "start", lambda *a, **k: (202, {"running": True}))

    _post(port, "/api/models/example/plan", {"preview": True, "engine": "vllm"})
    _post(port, "/api/models/example/plan", {"engine": "vllm"})
    _post(port, "/api/switch/plan", {})
    _post(port, "/api/discover", {})
    assert counters == {"reads": 0, "slow": 0}

    _post(port, "/api/not-a-route", {})
    # Expired before the reply is sent (so this is never a race), and again after.
    assert counters["reads"] >= 1
    assert counters["slow"] >= 1


class _Ssh:
    """Fake ssh: each call pops the next (code, stdout) from ``replies``."""

    def __init__(self, monkeypatch, replies):
        self.replies = list(replies)
        self.calls = 0
        monkeypatch.setattr(remote_mod.subprocess, "run", self.run)
        monkeypatch.setattr(remote_mod, "resolve_target", lambda host, catalog: "spark-x.example")

    def run(self, argv, **kwargs):
        self.calls += 1
        code, out = self.replies.pop(0)
        return subprocess.CompletedProcess(argv, code, out, "ssh: connect to host spark-x.example: No route to host" if code == 255 else "")


def _mark_down(runner, ssh):
    assert runner.invoke("spark-x", {"op": "probe"}, timeout=5)["ok"] is False
    skipped = runner.invoke("spark-x", {"op": "probe"}, timeout=5)
    assert skipped.get("unreachable") is True and ssh.calls == 1  # the skip is on


@pytest.mark.parametrize("kind", ["action", "run_argv", "capture"])
def test_a_successful_call_clears_the_unreachable_skip_at_once(monkeypatch, kind) -> None:
    ssh = _Ssh(monkeypatch, [(255, ""), (0, '{"ok": true}'), (0, '{"ok": true}')])
    runner = remote_mod.SshRunner({})
    _mark_down(runner, ssh)
    if kind == "action":
        assert runner.invoke("spark-x", {"op": "stage_flat"}, timeout=5)["ok"] is True
    elif kind == "run_argv":
        assert runner.run_argv("spark-x", ["true"], timeout=5)["ok"] is True
    else:
        assert runner.capture("spark-x", ["cat", "/proc/meminfo"], timeout=5)["ok"] is True
    again = runner.invoke("spark-x", {"op": "probe"}, timeout=5)
    assert again["ok"] is True and ssh.calls == 3  # probed again, well inside the 30 s


def test_a_failed_action_keeps_the_skip(monkeypatch) -> None:
    ssh = _Ssh(monkeypatch, [(255, ""), (255, "")])
    runner = remote_mod.SshRunner({})
    _mark_down(runner, ssh)
    assert runner.invoke("spark-x", {"op": "stage_flat"}, timeout=5)["ok"] is False
    assert runner.invoke("spark-x", {"op": "probe"}, timeout=5).get("unreachable") is True
    assert ssh.calls == 2
