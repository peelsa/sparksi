# SPDX-License-Identifier: Apache-2.0

import http.client
import json
import threading
import time
import types
from pathlib import Path

import pytest

import sparksi.agent as agent
from sparksi import unmanaged
from sparksi.config import CFG
from sparksi.serve import STATE

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

NODES = [{"id": "spark-a", "name": "Spark A", "role": "head"}, {"id": "spark-b", "name": "Spark B", "role": "worker"}]


def _row(host, port, **kw):
    return {"host": host, "port": port, **kw}


class TestFind:
    def test_a_model_without_a_sparksi_run_is_listed(self):
        out = unmanaged.find([_row("spark-a", 8000, served="org/model-a", name="model-a")], [], NODES)
        assert out == [
            {"host": "spark-a", "host_name": "Spark A", "role": "head", "port": 8000, "label": "org/model-a"}
        ]

    def test_label_falls_back_to_name_then_unnamed(self):
        out = unmanaged.find([_row("spark-a", 8001, name=" model-b "), _row("spark-a", 8002)], [], NODES)
        assert [i["label"] for i in out] == ["model-b", "unnamed"]

    def test_rows_match_by_name_and_span(self):
        out = unmanaged.find([_row("Spark A", 8000, span=["spark-b"])], [], NODES)
        assert [(i["host"], i["port"]) for i in out] == [("spark-a", 8000), ("spark-b", 8000)]

    @pytest.mark.parametrize("phase", sorted(unmanaged.ACTIVE))
    def test_an_active_sparksi_run_on_that_port_and_node_hides_it(self, phase):
        runs = [{"id": "r1", "phase": phase, "port": 8000, "nodes": [{"host": "spark-a"}]}]
        assert unmanaged.find([_row("spark-a", 8000)], runs, NODES) == []

    def test_a_run_with_no_nodes_covers_every_node(self):
        runs = [{"id": "r1", "phase": "ready", "port": 8000}]
        assert unmanaged.find([_row("spark-a", 8000), _row("spark-b", 8000)], runs, NODES) == []

    @pytest.mark.parametrize(
        "run",
        [
            {"id": "r1", "phase": "stopped", "port": 8000, "nodes": [{"host": "spark-a"}]},
            {"id": "r1", "phase": "ready", "port": 8001, "nodes": [{"host": "spark-a"}]},
            {"id": "r1", "phase": "ready", "port": 8000, "nodes": [{"host": "spark-b"}]},
        ],
        ids=["closed run", "other port", "other node"],
    )
    def test_other_runs_do_not_hide_it(self, run):
        assert len(unmanaged.find([_row("spark-a", 8000)], [run], NODES)) == 1

    def test_port_8188_odd_rows_and_unknown_hosts_are_skipped(self):
        rows = [
            _row("spark-a", 8188),
            "x",
            _row("spark-a", "nope"),
            _row("elsewhere", 8000),
            _row("spark-a", 8000),
            _row("spark-a", 8000),
        ]
        assert [(i["host"], i["port"]) for i in unmanaged.find(rows, [], NODES)] == [("spark-a", 8000)]


class TestContainersAndStoppable:
    ITEM = {"host": "spark-a", "host_name": "Spark A", "role": "head", "port": 8000, "label": "model-a"}

    def test_container_found_by_port(self):
        (out,) = unmanaged.attach_containers([self.ITEM], {"spark-a": {"z-box": 8000, "a-box": 8000, "c-box": 9000}})
        assert out["container"] == "a-box" and out["why"] == ""

    def test_no_container_and_no_docker(self):
        (none,) = unmanaged.attach_containers([self.ITEM], {"spark-a": {"c-box": 9000}})
        assert none["container"] is None and "stop it on the machine" in none["why"]
        (nodocker,) = unmanaged.attach_containers([self.ITEM], {"spark-a": None})
        assert nodocker["container"] is None and nodocker["why"] == "docker could not be asked on this machine"

    def test_stoppable_needs_the_setting_and_a_container(self):
        (item,) = unmanaged.attach_containers([self.ITEM], {"spark-a": {"a-box": 8000}})
        assert unmanaged.mark_stoppable([item], True)[0]["stoppable"] is True
        off = unmanaged.mark_stoppable([item], False)[0]
        assert off["stoppable"] is False and "stop_unmanaged" in off["why"]
        (bare,) = unmanaged.attach_containers([self.ITEM], {"spark-a": {}})
        assert unmanaged.mark_stoppable([bare], True)[0]["stoppable"] is False

    def test_same_item_needs_host_port_and_container(self):
        (item,) = unmanaged.attach_containers([self.ITEM], {"spark-a": {"a-box": 8000}})
        assert unmanaged.same_item([item], "spark-a", 8000, "a-box") == item
        assert unmanaged.same_item([item], "spark-a", "8000", "a-box") == item
        for args in (("spark-b", 8000, "a-box"), ("spark-a", 8001, "a-box"), ("spark-a", 8000, "b-box")):
            assert unmanaged.same_item([item], *args) is None


class Recorder:
    def __init__(self, ret=None):
        self.calls = []
        self.ret = ret or types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def __call__(self, *a, **k):
        self.calls.append((a, k))
        return self.ret


def _patch_agent(monkeypatch, containers, listen=None, cfg=True):
    monkeypatch.setitem(agent._CFG, "stop_unmanaged", cfg)
    monkeypatch.setattr(agent, "_running_containers", lambda: dict(containers))
    monkeypatch.setattr(agent, "_container_listen_port", lambda name, addr: listen(name) if listen else 8000)
    monkeypatch.setattr(agent, "_probe_address", lambda raw: "127.0.0.1")
    rec = Recorder()
    monkeypatch.setattr(agent, "subprocess", types.SimpleNamespace(run=rec))
    return rec


class TestAgentStopUnmanaged:
    def test_happy_path(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"model-a-box": 8000}, listen=lambda n: 8000)
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8000})
        assert out["stopped"] is True
        assert rec.calls[0][0][0] == ["docker", "stop", "--time", "30", "model-a-box"]
        assert len(rec.calls) == 1

    def test_refused_when_cfg_off(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"model-a-box": 8000}, cfg=False)
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8000})
        assert out["stopped"] is False
        assert "off" in out.get("error", "")
        assert rec.calls == []

    def test_refused_port_8188(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"model-a-box": 8188}, listen=lambda n: 8188)
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8188})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_sparksi_prefix(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"sparksi-x": 8000}, listen=lambda n: 8000)
        out = agent._stop_unmanaged({"container": "sparksi-x", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_comfy_any_case(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"CoMfY-ui": 8000}, listen=lambda n: 8000)
        out = agent._stop_unmanaged({"container": "CoMfY-ui", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_announce_any_case(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"dgx-Announce": 8000}, listen=lambda n: 8000)
        out = agent._stop_unmanaged({"container": "dgx-Announce", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    @pytest.mark.parametrize("name", ["a;b", "-x"])
    def test_refused_unsafe_name(self, monkeypatch, name):
        rec = _patch_agent(monkeypatch, {name: 8000}, listen=lambda n: 8000)
        out = agent._stop_unmanaged({"container": name, "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_unknown_container(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"other": 8000})
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_when_docker_unavailable(self, monkeypatch):
        monkeypatch.setitem(agent._CFG, "stop_unmanaged", True)
        monkeypatch.setattr(agent, "_running_containers", lambda: None)
        monkeypatch.setattr(agent, "_container_listen_port", lambda n, a: 8000)
        monkeypatch.setattr(agent, "_probe_address", lambda raw: "127.0.0.1")
        rec = Recorder()
        monkeypatch.setattr(agent, "subprocess", types.SimpleNamespace(run=rec))
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []

    def test_refused_port_mismatch(self, monkeypatch):
        rec = _patch_agent(monkeypatch, {"model-a-box": 8000}, listen=lambda n: 9001)
        out = agent._stop_unmanaged({"container": "model-a-box", "port": 8000})
        assert out["stopped"] is False
        assert rec.calls == []


class TestAgentUnmanagedContainerPorts:
    def test_filters_and_maps(self, monkeypatch):
        def listen(name, address):
            return 8000 if name == "model-a-box" else 8188

        monkeypatch.setitem(agent._CFG, "stop_unmanaged", True)
        monkeypatch.setattr(agent, "_container_listen_port", listen)
        monkeypatch.setattr(agent, "_probe_address", lambda raw: "127.0.0.1")
        out = agent._unmanaged_container_ports(
            ["sparksi-x", "model-a-box", "comfyui", "dgx-announce", "bad;name"],
            "127.0.0.1",
        )
        assert out == {"model-a-box": 8000}


# ---------------------------------------------------------------------------
# (C) HTTP
# ---------------------------------------------------------------------------

import sparksi.serve as serve_mod  # noqa: E402
from sparksi.lock import Lock  # noqa: E402
from sparksi.serve import DualServer, Handler  # noqa: E402


def ask(port, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json", "Origin": "http://127.0.0.1:%d" % port}
    conn.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    resp = conn.getresponse()
    raw = resp.read()
    conn.close()
    return resp.status, json.loads(raw or b"{}")


class FakeRunner:
    def __init__(self, containers):
        self.containers = containers
        self.calls = []

    def invoke(self, host, req, timeout=60):
        self.calls.append((host, dict(req)))
        if req.get("op") == "probe":
            return {"ok": True, "unmanaged_containers": dict(self.containers.get(host, {}))}
        if req.get("op") == "stop_unmanaged":
            return {"ok": True, "stopped": True}
        return {"ok": False, "error": "unexpected"}


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()  # unlocked
    head = CFG.head.id
    monkeypatch.setattr(serve_mod, "_monitor_view", lambda: ([{"host": head, "port": 8000, "served": "model-a"}], []))
    monkeypatch.setattr("sparksi.recipes.runs.list_runs", lambda: [])
    runner = FakeRunner({head: {"model-a-box": 8000}})
    monkeypatch.setattr(STATE, "runner", runner)
    monkeypatch.setitem(CFG.controller, "stop_unmanaged", True)
    serve_mod.READS.clear()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], runner, head
    httpd.shutdown()
    httpd.server_close()
    STATE.runner = None
    serve_mod.READS.clear()


def _poll_unmanaged(port, want=True, timeout=2.0):
    import time

    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, body = ask(port, "GET", "/api/unmanaged")
        last = (status, body)
        items = body.get("items", []) if isinstance(body, dict) else []
        if bool(items) == want:
            return status, body
        time.sleep(0.05)
    return last


class TestHttpUnmanaged:
    def test_get_lists_item(self, server):
        port, runner, head = server
        status, body = _poll_unmanaged(port, want=True)
        assert status == 200
        items = body["items"]
        assert len(items) == 1
        item = items[0]
        assert item["host"] == head
        assert item["port"] == 8000
        assert item["label"] == "model-a"
        assert item["container"] == "model-a-box"
        assert item["stoppable"] is True
        assert body.get("enabled") is True

    def test_post_stop(self, server):
        port, runner, head = server
        _poll_unmanaged(port, want=True)
        status, body = ask(
            port,
            "POST",
            "/api/unmanaged/stop",
            {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
        )
        assert status == 200
        assert body["stopped"] is True
        assert (head, {"op": "stop_unmanaged", "container": "model-a-box", "port": 8000}) in runner.calls

    def test_post_stop_locked(self, server):
        port, runner, head = server
        _poll_unmanaged(port, want=True)
        assert STATE.lock.acquire("test")
        try:
            status, body = ask(
                port,
                "POST",
                "/api/unmanaged/stop",
                {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
            )
            assert status == 403
            assert "Unlock first" in body.get("error", "")
            assert not any(c[1].get("op") == "stop_unmanaged" for c in runner.calls)
        finally:
            STATE.lock.release()

    def test_disabled_stoppable_false_and_post_403(self, server, monkeypatch):
        port, runner, head = server
        monkeypatch.setitem(CFG.controller, "stop_unmanaged", False)
        status, body = _poll_unmanaged(port, want=True)
        item = body["items"][0]
        assert item["stoppable"] is False
        assert "stop_unmanaged" in item.get("why", "")
        before = len(runner.calls)
        status2, body2 = ask(
            port,
            "POST",
            "/api/unmanaged/stop",
            {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
        )
        assert status2 == 403
        assert len(runner.calls) == before

    def test_post_mismatch_container_409(self, server):
        port, runner, head = server
        _poll_unmanaged(port, want=True)
        before = len(runner.calls)
        status, body = ask(
            port, "POST", "/api/unmanaged/stop", {"host": head, "port": 8000, "container": "other-box", "from_ui": True}
        )
        assert status == 409
        assert not [c for c in runner.calls[before:] if c[1].get("op") == "stop_unmanaged"]

    def test_post_port_8188_400(self, server):
        port, runner, head = server
        _poll_unmanaged(port, want=True)
        status, body = ask(
            port,
            "POST",
            "/api/unmanaged/stop",
            {"host": head, "port": 8188, "container": "model-a-box", "from_ui": True},
        )
        assert status == 400
        assert not any(c[1].get("op") == "stop_unmanaged" for c in runner.calls)

    def test_active_run_hides_item_and_post_409(self, server, monkeypatch):
        port, runner, head = server
        monkeypatch.setattr(
            "sparksi.recipes.runs.list_runs",
            lambda: [{"id": "r1", "phase": "ready", "port": 8000, "nodes": [{"host": head}]}],
        )
        status, body = _poll_unmanaged(port, want=False)
        assert status == 200
        assert body["items"] == []
        status2, body2 = ask(
            port,
            "POST",
            "/api/unmanaged/stop",
            {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
        )
        assert status2 == 409
        assert not any(c[1].get("op") == "stop_unmanaged" for c in runner.calls)

    def test_cross_origin_post_403(self, server):
        port, runner, head = server
        _poll_unmanaged(port, want=True)
        before = len(runner.calls)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request(
            "POST",
            "/api/unmanaged/stop",
            body=json.dumps({"host": head, "port": 8000, "container": "model-a-box", "from_ui": True}),
            headers={"Content-Type": "application/json", "Origin": "http://evil.example"},
        )
        resp = conn.getresponse()
        resp.read()
        conn.close()
        assert resp.status == 403
        assert len(runner.calls) == before


def test_post_without_ui_or_key_locked_403(server):
    port, runner, head = server
    STATE.lock.acquire("test")
    try:
        status, body = ask(
            port,
            "POST",
            "/api/unmanaged/stop",
            {"host": head, "port": 8000, "container": "model-a-box"},
        )
    finally:
        STATE.lock.release()
    assert status == 403
    assert body["stopped"] is False
    stops = [c for c in runner.calls if c[1].get("op") == "stop_unmanaged"]
    assert stops == []


def test_post_while_loading_409_never_calls_node(server, monkeypatch):
    port, runner, head = server
    monkeypatch.setattr(serve_mod, "load_in_progress", lambda: True)
    before = len(runner.calls)
    status, _ = ask(
        port,
        "POST",
        "/api/unmanaged/stop",
        {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
    )
    assert status == 409
    assert runner.calls[before:] == []


def test_container_from_a_run_record_is_not_stoppable(server, monkeypatch):
    port, runner, head = server
    monkeypatch.setattr(
        "sparksi.recipes.runs.list_runs",
        lambda: [
            {
                "id": "r9",
                "phase": "stopped",
                "port": 8000,
                "nodes": [{"host": head, "container": "model-a-box"}],
            }
        ],
    )
    serve_mod.READS.clear()
    status, body = ask(port, "GET", "/api/unmanaged", None)
    assert status == 200
    deadline = time.monotonic() + 2.0
    while not body.get("checked") and time.monotonic() < deadline:
        time.sleep(0.05)
        status, body = ask(port, "GET", "/api/unmanaged", None)
    assert body["checked"] is True
    items = {item["port"]: item for item in body["items"]}
    assert items[8000]["stoppable"] is False
    assert items[8000]["why"] == unmanaged._OURS
    before = len(runner.calls)
    status, _ = ask(
        port,
        "POST",
        "/api/unmanaged/stop",
        {"host": head, "port": 8000, "container": "model-a-box", "from_ui": True},
    )
    assert status == 409
    stops = [c for c in runner.calls[before:] if c[1].get("op") == "stop_unmanaged"]
    assert stops == []


def test_sparksi_containers_collects_names():
    runs = [
        {
            "id": "r1",
            "nodes": [
                {"host": "h1", "container": " a-box "},
                {"host": "h2", "container": ""},
                {"host": "h3"},
                "not-a-dict",
            ],
        },
        "not-a-dict-run",
        {"id": "r2", "nodes": "not-a-list"},
        {"id": "r3", "nodes": [{"container": 123}]},
    ]
    assert unmanaged.sparksi_containers(runs) == {"a-box"}
    assert unmanaged.sparksi_containers(None) == set()
    assert unmanaged.sparksi_containers("junk") == set()


def test_mark_stoppable_refuses_our_containers():
    items = [
        {"host": "h1", "port": 8000, "container": "a-box"},
        {"host": "h2", "port": 8001, "container": "other-box"},
        {"host": "h3", "port": 8002, "container": None},
    ]
    out = unmanaged.mark_stoppable(items, True, {"a-box"})
    by_port = {item["port"]: item for item in out}
    assert by_port[8000]["stoppable"] is False
    assert by_port[8000]["why"] == unmanaged._OURS
    assert by_port[8001]["stoppable"] is True
    assert by_port[8002]["stoppable"] is False
