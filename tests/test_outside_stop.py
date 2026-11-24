# SPDX-License-Identifier: Apache-2.0

"""Emergency stop for a server started outside Sparksi: the agent op on real local
processes, and the controller endpoint with fakes on 1 to 4 Sparks."""

from __future__ import annotations

import copy
import http.client
import json
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from sparksi import agent, unmanaged
from sparksi.config import CFG
from sparksi.serve import STATE

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc")

# A dummy model server: listens on a port, starts one child, optionally ignores SIGTERM.
SERVER = r"""
import signal, socket, subprocess, sys, time
mode = sys.argv[2]
if mode == "ignore":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", int(sys.argv[1])))
s.listen()
code = "import signal, time\n" + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if mode == "ignore" else "") + "time.sleep(600)"
child = subprocess.Popen([sys.executable, "-c", code])
print(child.pid, flush=True)
while True:
    time.sleep(1)
"""


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _gone(pid: int) -> bool:
    info = agent._proc_info(pid)
    return info is None or info["state"] in ("Z", "X")


@pytest.fixture()
def dummy(request):
    """Start a dummy server; yields (port, server Popen, child pid); always cleaned up."""
    started: list[tuple[subprocess.Popen, int]] = []

    def start(mode: str = "plain", extra: str = "x") -> tuple[int, subprocess.Popen, int]:
        port = _free_port()
        proc = subprocess.Popen(
            [sys.executable, "-c", SERVER, str(port), mode, extra],
            stdout=subprocess.PIPE, text=True, start_new_session=True,
        )
        child = int(proc.stdout.readline().strip())
        started.append((proc, child))
        deadline = time.monotonic() + 10
        while not agent._listen_inodes(port):
            assert time.monotonic() < deadline, "dummy server did not listen"
            time.sleep(0.05)
        return port, proc, child

    yield start
    for proc, child in started:
        for pid in (proc.pid, child):
            try:
                os.kill(pid, 9)
            except OSError:
                pass
        proc.wait(timeout=10)


@pytest.fixture()
def agent_cfg(monkeypatch):
    monkeypatch.setitem(agent._CFG, "off_limits_ports", [8188])
    monkeypatch.setitem(agent._CFG, "video_process_substrings", ["comfy"])
    monkeypatch.setattr(agent, "_gpu_apps", lambda: None)


# ---------------------------------------------------------------------------
# Agent op on real local processes
# ---------------------------------------------------------------------------


def test_term_stops_listener_and_child_and_closes_port(dummy, agent_cfg):
    port, proc, child = dummy()
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is True, out
    assert out["signal"] == "TERM"
    assert out["port_closed"] is True
    assert proc.pid in out["pids"] and child in out["pids"]
    proc.wait(timeout=5)
    assert _gone(child)
    assert not agent._listen_inodes(port)


def test_kill_after_grace_when_term_is_ignored(dummy, agent_cfg):
    port, proc, child = dummy("ignore")
    started = time.monotonic()
    out = agent._stop_outside({"port": port, "grace": 1})
    assert out["stopped"] is True, out
    assert out["signal"] == "KILL"
    assert time.monotonic() - started < 15
    proc.wait(timeout=5)
    assert _gone(child)


def test_grace_is_capped(dummy, agent_cfg, monkeypatch):
    waits: list[float] = []
    real = agent._wait_gone
    monkeypatch.setattr(agent, "_wait_gone", lambda targets, seconds: (waits.append(seconds), real(targets, seconds))[1])
    port, _proc, _child = dummy()
    agent._stop_outside({"port": port, "grace": 9999})
    assert waits[0] == agent._OUTSIDE_GRACE_MAX


def test_gpu_memory_reported_freed(dummy, agent_cfg, monkeypatch):
    port, proc, child = dummy()
    answers = iter([{child: 5000, 4242: 100}, {4242: 100}])
    monkeypatch.setattr(agent, "_gpu_apps", lambda: next(answers))
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is True
    assert out["gpu_checked"] is True
    assert out["gpu_freed_mib"] == 5000
    assert out["gpu_pids_left"] == []


@pytest.mark.parametrize("port", [8188, 22, 0, 70000, "x", None])
def test_refuses_off_limits_and_bad_ports(agent_cfg, port):
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is False


def test_nothing_listening(agent_cfg):
    out = agent._stop_outside({"port": _free_port()})
    assert out["stopped"] is False and out["port_closed"] is True


@pytest.mark.parametrize(("extra", "keep"), [("comfy-ui", []), ("keepme-name", ["keepme-name"]), ("open-webui", [])])
def test_protected_process_is_left_alone(dummy, agent_cfg, extra, keep):
    port, proc, child = dummy(extra=extra)
    out = agent._stop_outside({"port": port, "keep": keep})
    assert out["stopped"] is False
    assert "protected" in out["error"]
    assert proc.poll() is None and not _gone(child)


def test_catalog_keep_matches_whole_parts_only(dummy, agent_cfg):
    port, proc, _child = dummy(extra="model-h3x")
    out = agent._stop_outside({"port": port, "keep": ["h3"]})
    assert out["stopped"] is True, out
    proc.wait(timeout=5)


def test_plumbing_listener_is_refused(dummy, agent_cfg, monkeypatch):
    port, proc, child = dummy()
    comm = agent._proc_info(proc.pid)["comm"]
    monkeypatch.setattr(agent, "_OUTSIDE_KEEP_COMM", frozenset({comm}))
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is False and "never stops" in out["error"]
    assert proc.poll() is None


def test_process_inside_a_container_is_refused(dummy, agent_cfg, monkeypatch):
    port, proc, _child = dummy()
    monkeypatch.setattr(agent, "_in_container", lambda info: True)
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is False and "container" in out["error"]
    assert proc.poll() is None


def test_never_signals_its_own_process(agent_cfg):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        out = agent._stop_outside({"port": s.getsockname()[1]})
    assert out["stopped"] is False
    assert "could not find" in out["error"]


def test_other_users_process_uses_exact_sudo_kill_or_returns_the_command(dummy, agent_cfg, monkeypatch):
    port, proc, child = dummy()
    calls: list[list[str]] = []
    real_run = subprocess.run

    def fake_run(argv, **kw):
        if argv and argv[0] == "sudo":
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 1, "", "sudo: a password is required")
        return real_run(argv, **kw)

    monkeypatch.setattr(agent.os, "getuid", lambda: 99999)
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is False
    pids = sorted([proc.pid, child])
    assert calls == [["sudo", "-n", "/usr/bin/kill", "-s", "TERM", "--", *map(str, pids)]]
    assert out["command"] == "sudo /usr/bin/kill -s TERM -- " + " ".join(map(str, pids))
    assert proc.poll() is None


def test_other_users_process_stopped_when_sudo_is_allowed(dummy, agent_cfg, monkeypatch):
    port, proc, child = dummy()
    real_run = subprocess.run

    def fake_run(argv, **kw):
        if argv and argv[0] == "sudo":
            assert argv[:6] == ["sudo", "-n", "/usr/bin/kill", "-s", "TERM", "--"]
            for pid in argv[6:]:
                os.kill(int(pid), 15)
            return subprocess.CompletedProcess(argv, 0, "", "")
        return real_run(argv, **kw)

    monkeypatch.setattr(agent.os, "getuid", lambda: 99999)
    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    out = agent._stop_outside({"port": port})
    assert out["stopped"] is True, out
    proc.wait(timeout=5)


def test_unreadable_owner_returns_the_command(agent_cfg, monkeypatch):
    monkeypatch.setattr(agent, "_listen_inodes", lambda port: {"123"})
    monkeypatch.setattr(agent, "_socket_pids", lambda inodes: (set(), True))
    out = agent._stop_outside({"port": 8020})
    assert out["stopped"] is False
    assert out["command"] == "sudo fuser -k -TERM 8020/tcp"


class _Docker:
    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, "", "")


def _container_env(monkeypatch, docker, running=("model-box",), port=8020):
    monkeypatch.setattr(agent, "_running_containers", lambda: list(running))
    monkeypatch.setattr(agent, "_container_listen_port", lambda name, address: port)
    monkeypatch.setattr(agent, "_probe_address", lambda raw: "127.0.0.1")
    monkeypatch.setattr(agent, "_wait_closed", lambda p, s: True)
    monkeypatch.setattr(agent.subprocess, "run", docker)


def test_container_gets_docker_stop(agent_cfg, monkeypatch):
    docker = _Docker()
    _container_env(monkeypatch, docker)
    out = agent._stop_outside({"port": 8020, "container": "model-box"})
    assert out["stopped"] is True and out["signal"] == "docker stop"
    assert docker.calls == [["docker", "stop", "--time", "20", "model-box"]]


@pytest.mark.parametrize("name", ["sparksi-glm-head", "comfyui", "dgx-announce", "my-open-webui", "keepme", "bad;name"])
def test_protected_containers_are_never_stopped(agent_cfg, monkeypatch, name):
    docker = _Docker()
    _container_env(monkeypatch, docker, running=(name,))
    out = agent._stop_outside({"port": 8020, "container": name, "keep": ["keepme"]})
    assert out["stopped"] is False
    assert docker.calls == []


def test_container_on_another_port_is_refused(agent_cfg, monkeypatch):
    docker = _Docker()
    _container_env(monkeypatch, docker, port=9000)
    out = agent._stop_outside({"port": 8020, "container": "model-box"})
    assert out["stopped"] is False and docker.calls == []


def test_op_ignores_stop_unmanaged(dummy, agent_cfg, monkeypatch):
    monkeypatch.setitem(agent._CFG, "stop_unmanaged", False)
    port, proc, _child = dummy()
    out = agent.main({"op": "stop_outside", "port": port, "_config": {"off_limits_ports": [8188], "video_process_substrings": ["comfy"]}})
    assert out["stopped"] is True, out
    proc.wait(timeout=5)


def test_op_is_a_spark_only_op():
    from sparksi.remote import _RECIPE_OPS

    assert "stop_outside" in _RECIPE_OPS


# ---------------------------------------------------------------------------
# Who is offered the button
# ---------------------------------------------------------------------------


def _item(**kw):
    base = {"host": "s1", "host_name": "spark-1", "role": "head", "port": 8020, "label": "llama", "container": None, "why": "", "stoppable": False}
    base.update(kw)
    return base


def test_mark_emergency():
    items = [
        _item(),
        _item(port=8021, container="model-box"),
        _item(port=8022, container="sparksi-x-head"),
        _item(port=8023, container="model-b", stoppable=True),
        _item(port=8024, label="keepme"),
        _item(port=8025, container="lever-openwebui"),
        _item(port=8026, container="ComfyUI-box"),
    ]
    items.append(_item(port=8027, label="video"))
    out = unmanaged.mark_emergency(items, ours={"sparksi-x-head"}, keep={"keepme"}, keep_ports={8027})
    assert [i["emergency"] for i in out] == [True, True, False, False, False, False, False, False]
    assert out[7]["why"] == unmanaged._PROTECTED
    assert out[4]["why"] == unmanaged._PROTECTED and out[5]["why"] == unmanaged._PROTECTED


# ---------------------------------------------------------------------------
# Controller endpoint, 1 to 4 Sparks
# ---------------------------------------------------------------------------

import sparksi.serve as serve_mod  # noqa: E402
from sparksi.lock import Lock  # noqa: E402
from sparksi.serve import DualServer, Handler  # noqa: E402


def _raw(n: int) -> dict:
    nodes = [{"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}", "tailscale": f"100.64.0.{10 + i}"} for i in range(1, n + 1)]
    return {"nodes": nodes, "paths": {"home": "/home/USER"}}


class FakeRunner:
    def __init__(self, containers: dict[str, dict[str, int]]):
        self.containers = containers
        self.calls: list[tuple[str, dict]] = []
        self.reply = {"ok": True, "stopped": True, "signal": "TERM", "port_closed": True, "gpu_checked": True, "gpu_freed_mib": 5000, "pids": [10, 11]}

    def invoke(self, host, req, timeout=60):
        self.calls.append((host, dict(req)))
        if req.get("op") == "probe":
            return {"ok": True, "unmanaged_containers": dict(self.containers.get(host, {}))}
        if req.get("op") == "stop_outside":
            # The SSH runner resolves a Spark-only op through the catalog hosts, keyed by node name.
            from sparksi.remote import recipe_target_error

            hosts = {node.name: {"ssh": node.ssh} for node in CFG.nodes}
            assert recipe_target_error(host, {"hosts": hosts}) is None, host
            return dict(self.reply)
        return {"ok": False, "error": "unexpected"}

    def stops(self):
        return [c for c in self.calls if c[1].get("op") == "stop_outside"]


@pytest.fixture()
def site(tmp_path, monkeypatch, request):
    n = getattr(request, "param", 2)
    saved = copy.copy(CFG.__dict__)
    CFG.__init__(_raw(n))
    request.addfinalizer(lambda: (CFG.__dict__.clear(), CFG.__dict__.update(saved)))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
    last = CFG.nodes[-1]
    loaded = [{"host": last.id, "port": 8020, "served": "llama"}, {"host": last.id, "port": 8030, "served": "boxed"}]
    monkeypatch.setattr(serve_mod, "_monitor_view", lambda: (loaded, []))
    monkeypatch.setattr("sparksi.recipes.runs.list_runs", lambda: [])
    monkeypatch.setattr(serve_mod, "load_catalog", lambda: {"never_touch": {"do_not_stop": ["keepme"]}})
    notes: list[tuple[str, str, str]] = []
    monkeypatch.setattr(serve_mod, "note_activity", lambda kind, text, machine="": notes.append((kind, text, machine)))
    runner = FakeRunner({last.id: {"boxed-server": 8030}})
    monkeypatch.setattr(STATE, "runner", runner)
    monkeypatch.setattr(serve_mod, "active_runner", lambda cat: runner)
    before = CFG.controller.get("stop_unmanaged")
    serve_mod.READS.clear()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield {"port": httpd.server_address[1], "runner": runner, "host": last.id, "name": last.name, "notes": notes, "loaded": loaded}
    httpd.shutdown()
    httpd.server_close()
    STATE.runner = None
    serve_mod.READS.clear()
    assert CFG.controller.get("stop_unmanaged") == before


def ask(port, method, path, body=None, headers=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    hdrs = {"Content-Type": "application/json", "Origin": "http://127.0.0.1:%d" % port}
    hdrs.update(headers or {})
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=hdrs)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw or b"{}")


def _items(port):
    deadline = time.monotonic() + 3
    while True:
        status, body = ask(port, "GET", "/api/unmanaged")
        if body.get("checked") and body.get("items") or time.monotonic() > deadline:
            return {i["port"]: i for i in body.get("items", [])}
        time.sleep(0.05)


def _body(s, port=8020, container=None, **kw):
    out = {"host": s["host"], "port": port, "container": container, "from_ui": True,
           "confirm": {"model": "llama" if port == 8020 else "boxed", "spark": s["name"], "port": port}}
    out.update(kw)
    return out


NS = [1, 2, 3, 4]


@pytest.mark.parametrize("site", NS, indirect=True)
def test_bare_process_offered_and_stopped_with_lock_on(site):
    items = _items(site["port"])
    assert items[8020]["emergency"] is True and items[8020]["container"] is None
    assert items[8020]["stoppable"] is False
    assert STATE.lock.acquire("test")
    try:
        status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site))
    finally:
        STATE.lock.release()
    assert status == 200, body
    assert body["stopped"] is True and body["gpu_freed_mib"] == 5000
    assert site["runner"].stops() == [(site["name"], {"op": "stop_outside", "port": 8020, "keep": ["keepme"]})]
    assert site["notes"] == [("stop", f"emergency stop llama on {site['name']} port 8020: stopped with TERM, port closed, GPU freed 5000 MiB", site["host"])]


@pytest.mark.parametrize("site", NS, indirect=True)
def test_container_offered_and_stopped(site):
    items = _items(site["port"])
    assert items[8030]["container"] == "boxed-server" and items[8030]["emergency"] is True
    status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site, port=8030, container="boxed-server"))
    assert status == 200, body
    assert site["runner"].stops() == [(site["name"], {"op": "stop_outside", "port": 8030, "keep": ["keepme"], "container": "boxed-server"})]


@pytest.mark.parametrize(
    "confirm",
    [None, {}, {"model": "llama", "spark": "spark-9", "port": 8020}, {"model": "other", "spark": "SPARK", "port": 8020}, {"model": "llama", "spark": "SPARK", "port": 8021}],
)
def test_confirmation_must_name_model_spark_and_port(site, confirm):
    _items(site["port"])
    if isinstance(confirm, dict) and confirm.get("spark") == "SPARK":
        confirm = dict(confirm, spark=site["name"])
    status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site, confirm=confirm))
    assert status == 400 and body["stopped"] is False
    assert site["runner"].stops() == []
    assert site["notes"] == []


def test_failure_is_logged_and_returns_the_command(site):
    _items(site["port"])
    site["runner"].reply = {"ok": False, "stopped": False, "error": "sudo needs a password", "command": "sudo /usr/bin/kill -s TERM -- 10"}
    status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site))
    assert status == 502
    assert body["command"] == "sudo /usr/bin/kill -s TERM -- 10"
    assert site["notes"][0][1].endswith("failed: sudo needs a password")


def test_protected_label_not_offered_and_refused(site, monkeypatch):
    site["loaded"][0]["served"] = "keepme"
    items = _items(site["port"])
    assert items[8020]["emergency"] is False and items[8020]["why"] == unmanaged._PROTECTED
    body = _body(site)
    body["confirm"]["model"] = "keepme"
    status, _ = ask(site["port"], "POST", "/api/outside/stop", body)
    assert status == 409 and site["runner"].stops() == []


def test_protected_catalog_row_port_not_offered(site, monkeypatch):
    cat = {"never_touch": {"do_not_stop": ["video-x"]}, "library": [{"id": "video-x", "port": 8020}, {"id": "free", "port": 8030}]}
    monkeypatch.setattr(serve_mod, "load_catalog", lambda: cat)
    items = _items(site["port"])
    assert items[8020]["emergency"] is False and items[8030]["emergency"] is True
    status, _ = ask(site["port"], "POST", "/api/outside/stop", _body(site))
    assert status == 409 and site["runner"].stops() == []


def test_sparksi_run_never_offered(site, monkeypatch):
    host = site["host"]
    monkeypatch.setattr("sparksi.recipes.runs.list_runs", lambda: [{"id": "r1", "phase": "ready", "port": 8020, "nodes": [{"host": host, "container": "sparksi-x"}]}])
    items = _items(site["port"])
    assert 8020 not in items
    status, _ = ask(site["port"], "POST", "/api/outside/stop", _body(site))
    assert status == 409 and site["runner"].stops() == []


def test_sparksi_container_never_offered(site, monkeypatch):
    host = site["host"]
    site["runner"].containers[host] = {"sparksi-old-head": 8030}
    monkeypatch.setattr("sparksi.recipes.runs.list_runs", lambda: [{"id": "r0", "phase": "stopped", "port": 8030, "nodes": [{"host": host, "container": "sparksi-old-head"}]}])
    items = _items(site["port"])
    assert items[8030]["emergency"] is False
    status, _ = ask(site["port"], "POST", "/api/outside/stop", _body(site, port=8030, container="sparksi-old-head"))
    assert status == 409 and site["runner"].stops() == []


def test_off_limits_port_refused(site):
    status, _ = ask(site["port"], "POST", "/api/outside/stop", _body(site, port=8188))
    assert status == 400 and site["runner"].stops() == []


def test_api_caller_without_token_refused_when_locked(site):
    _items(site["port"])
    assert STATE.lock.acquire("test")
    try:
        status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site, from_ui=False))
    finally:
        STATE.lock.release()
    assert status == 403 and site["runner"].stops() == []


@pytest.mark.parametrize(
    "headers",
    [{"Origin": "http://evil.example"}, {"Sec-Fetch-Site": "cross-site"}, {"Content-Type": "text/plain"}],
)
def test_same_origin_json_only(site, headers):
    _items(site["port"])
    status, _ = ask(site["port"], "POST", "/api/outside/stop", _body(site), headers=headers)
    assert status in (403, 415)
    assert site["runner"].stops() == []


def test_one_stop_at_a_time(site):
    _items(site["port"])
    assert serve_mod._OUTSIDE_STOP.acquire(blocking=False)
    try:
        status, body = ask(site["port"], "POST", "/api/outside/stop", _body(site))
    finally:
        serve_mod._OUTSIDE_STOP.release()
    assert status == 409 and site["runner"].stops() == []
