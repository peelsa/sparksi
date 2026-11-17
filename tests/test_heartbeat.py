# SPDX-License-Identifier: Apache-2.0

"""Tests for sparksi.heartbeat."""

import time

import pytest

from sparksi import heartbeat
from sparksi.recipes import runs


def beat():
    return {
        "ph": "node-a",
        "v": 1,
        "hn": "node-a",
        "mm": [121.0, 80.0],
        "dk": [900.0, 400.0, 500.0],
        "g": [5, 40, 1500],
        "ls": [[8014, "w"], [22, "w"], [9000, "l"]],
        "ct": [["sparksi-x", [8014]], ["other", [9000]], ["noports", []]],
        "hi": {"8014": [200, ["m1"], 404, []], "9000": [None, [], 200, ["m2"]]},
        "cs": {"sparksi-x": 10.5},
    }


def base():
    return {
        "ok": True,
        "hostname": "node-a",
        "ips": ["192.0.2.10"],
        "mount_archive": "/srv/archive",
        "archive_store": "/srv/store",
        "disk_total_gib": 900.0,
        "disk_free_gib": 400.0,
        "disk_used_gib": 500.0,
        "disk_used_pct": 55.6,
    }


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    monkeypatch.setattr(heartbeat, "BEATS", heartbeat._Store())
    monkeypatch.setattr(heartbeat, "BASES", heartbeat._Store())
    monkeypatch.setattr(heartbeat, "node_key", lambda h: str(h))
    monkeypatch.setattr(runs, "_LOG_READ", {})
    yield
    runs.reset_for_tests()  # no run record (and its port) leaks into later tests
    runs.forget_ready_checks()


def test_parse_valid_and_rejects():
    good = heartbeat.parse(beat())
    assert good is not None and good["ph"] == "node-a" and good["hi"]["8014"][0] == 200
    bad = beat()
    bad["v"] = 2
    assert heartbeat.parse(bad) is None
    bad = beat()
    bad["ph"] = 7
    assert heartbeat.parse(bad) is None
    bad = beat()
    bad["ph"] = "x" * 65
    assert heartbeat.parse(bad) is None
    bad = beat()
    bad["ls"] = [[1, "w"]] * 33 + [[2, "w"]]
    parsed = heartbeat.parse(bad)
    assert len(parsed["ls"]) == 32
    bad = beat()
    bad["ls"] = [[1, 99]]
    assert heartbeat.parse(bad)["ls"] == []
    bad = beat()
    bad["hi"] = {"8014": [200, ["m"] * 5, 404, []]}
    assert "8014" not in heartbeat.parse(bad)["hi"]
    assert heartbeat.parse("nope") is None


def test_answer_probe_shape_and_base():
    got = heartbeat.answer_probe(heartbeat.parse(beat()), {"op": "probe", "ports": [8014, 9000, 9100]}, base())
    assert got["ok"] is True and got["via"] == "heartbeat"
    assert got["hostname"] == "node-a"
    assert got["mem_total_gib"] == 121.0 and got["mem_available_gib"] == 80.0
    assert got["gpu"]["util_pct"] == 5 and got["gpu"]["temp_f"] == 104.0
    assert got["ips"] == ["192.0.2.10"] and got["mount_archive"] == "/srv/archive"
    assert got["listening"] == {"8014": True, "9000": True, "9100": False}
    assert got["http"]["8014"]["status"] == 200
    assert got["http"]["9000"]["status"] == 200 and got["http"]["9000"]["models"] == ["m2"]


def test_answer_probe_none_cases():
    b = heartbeat.parse(beat())
    assert heartbeat.answer_probe(b, {"op": "probe", "paths": ["x"]}, base()) is None
    assert heartbeat.answer_probe(b, {"op": "probe", "zz": 1}, base()) is None
    req = {"op": "probe", "ports": [8014], "http_paths": {"8014": "/x"}}
    assert heartbeat.answer_probe(b, req, base()) is None
    assert heartbeat.answer_probe(b, {"op": "probe", "ports": [22]}, base()) is None  # below the beat's range
    unknown = dict(b, ls=b["ls"] + [[7777, "w"]])
    assert heartbeat.answer_probe(unknown, {"op": "probe", "ports": [7777]}, base()) is None  # no identity
    full = dict(b, ls=[[20000 + i, "w"] for i in range(heartbeat.MAX_LISTEN)])
    assert heartbeat.answer_probe(full, {"op": "probe", "ports": [7777]}, base()) is None  # table may be cut
    no_ct = dict(b)
    no_ct["ct"] = None
    req = {"op": "probe", "container_ports": True}
    assert heartbeat.answer_probe(no_ct, req, base()) is None


class FakeRunner:
    def __init__(self):
        self.calls = []

    def invoke(self, host, req, timeout=60):
        self.calls.append((host, dict(req)))
        return dict(base(), op=req.get("op"))

    def run_argv(self, argv):
        return "argv-ok"


def test_heartbeat_reads_routing():
    r = FakeRunner()
    hb = heartbeat.HeartbeatReads(r)
    heartbeat.BEATS.note("node-a", heartbeat.parse(beat()))
    heartbeat.BASES.note("node-a", base())
    got = hb.invoke("node-a", {"op": "probe"})
    assert r.calls == [] and got["via"] == "heartbeat"
    r2 = FakeRunner()
    hb2 = heartbeat.HeartbeatReads(r2)
    heartbeat.BEATS.note("node-b", heartbeat.parse(beat()))  # a beat but no SSH base yet
    got = hb2.invoke("node-b", {"op": "probe"})
    assert r2.calls and got["ok"] is True and "via" not in got
    assert heartbeat.BASES.get("node-b", 300) == got
    assert hb2.invoke("node-b", {"op": "probe"})["via"] == "heartbeat"
    r3 = FakeRunner()
    hb3 = heartbeat.HeartbeatReads(r3)
    hb3.invoke("node-a", {"op": "stop"})
    assert r3.calls[0][1] == {"op": "stop"}
    assert hb.run_argv(["x"]) == "argv-ok"
    assert hb.unwrap() is r


def test_stale_beat_falls_back(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(heartbeat, "BEATS", heartbeat._Store(clock=lambda: now[0]))
    monkeypatch.setattr(heartbeat, "BASES", heartbeat._Store(clock=lambda: now[0]))
    heartbeat.BEATS.note("node-a", heartbeat.parse(beat()))
    heartbeat.BASES.note("node-a", base())
    now[0] += 9.0
    assert heartbeat.beat_answer("node-a", {"op": "probe"}) is None
    now[0] -= 1.0
    assert heartbeat.beat_answer("node-a", {"op": "probe"})["via"] == "heartbeat"


def test_container_seen():
    heartbeat.BEATS.note("node-a", heartbeat.parse(beat()))
    assert heartbeat.container_seen("node-a", "sparksi-x") == (True, 10.5)
    assert heartbeat.container_seen("node-a", "ghost") == (False, None)
    no_ct = heartbeat.parse(beat())
    no_ct["ct"] = None
    heartbeat.BEATS.note("node-b", no_ct)
    assert heartbeat.container_seen("node-b", "sparksi-x") is None
    assert heartbeat.container_seen("missing", "sparksi-x") is None


def test_inbox_files_a_beat_only_under_its_allowlisted_sender():
    import json

    from sparksi.announce_rx import Inbox

    box = Inbox()
    good = dict(beat(), ph="a")
    assert box.accept("100.64.0.11", json.dumps(good).encode())
    assert heartbeat.BEATS.get("a", 8) is not None
    assert not box.accept("100.64.0.11", json.dumps(dict(good, ph="b")).encode())  # names another node
    assert not box.accept("192.0.2.52", json.dumps(good).encode())  # not on the allowlist
    assert not box.accept("100.64.0.11", json.dumps(dict(good, v=2)).encode())
    assert not box.accept("100.64.0.11", json.dumps(dict(good, pad="x" * 1300)).encode())  # oversize


def test_cached_reads_answer_from_a_beat_and_send_actions_to_ssh():
    from sparksi.serve import CachedReads

    r = FakeRunner()
    heartbeat.BEATS.note("node-a", heartbeat.parse(beat()))
    heartbeat.BASES.note("node-a", base())
    assert CachedReads(r).invoke("node-a", {"op": "probe", "ports": [8014]})["via"] == "heartbeat"
    CachedReads(r).invoke("node-a", {"op": "stop", "run_id": "x"})
    assert [c[1]["op"] for c in r.calls] == ["stop"]


def test_ready_run_skips_ssh_status_while_the_beat_shows_it_running(monkeypatch):
    from sparksi.recipes import runs

    monkeypatch.setattr(runs, "_LOG_READ", {})
    heartbeat.BEATS.note("node-a", heartbeat.parse(beat()))
    run = {"phase": "ready", "nodes": [{"host": "node-a", "container": "sparksi-x", "run_id": "r"}]}
    r = FakeRunner()
    runs._refresh_logs(run, r)  # first pass reads the log tail over SSH
    runs._refresh_logs(run, r)  # then the beat answers
    assert len(r.calls) == 1 and r.calls[0][1]["op"] == "status"
    assert run["nodes"][0]["container_status"] == "running" and run["nodes"][0]["container_gib"] == 10.5
    gone = {"phase": "ready", "nodes": [{"host": "node-a", "container": "sparksi-gone", "run_id": "r"}]}
    runs._refresh_logs(gone, r)
    runs._refresh_logs(gone, r)
    assert len(r.calls) == 3  # not in the beat: always SSH


HEAD = "spark-a"
WORKER = "spark-b"


def _partial_ct():
    return [["c%d" % i, []] for i in range(16)]


def _mk_run():
    return {
        "id": "pair-1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "both",
        "port": 8010,
        "phase": "ready",
        "ok": True,
        "started": time.time(),
        "timeout_s": 1800,
        "health_paths": ["/health"],
        "seen": {"worker": True, "head": True},
        "version": 1,
        "nodes": [
            {
                "role": "worker",
                "host": WORKER,
                "run_id": "pair-1-worker",
                "container": "sparksi-demo",
                "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
                "phase": "started",
            },
            {
                "role": "head",
                "host": HEAD,
                "run_id": "pair-1-head",
                "container": "sparksi-demo",
                "stop_argv": ["docker", "rm", "-f", "sparksi-demo"],
                "phase": "started",
            },
        ],
    }


class Runner:
    def __init__(self):
        self.ops = []

    def invoke(self, host, req, timeout=20):
        self.ops.append(req.get("op"))
        if req.get("op") == "status":
            return {"ok": True, "log_tail": "", "container_status": "running", "phase": "running"}
        if req.get("op") == "probe":
            listening = {"8010": True} if host == HEAD else {}
            http = {"8010": {"status": 200}} if listening else {}
            return {"ok": True, "listening": listening, "http": http, "containers": ["sparksi-demo"]}
        return {"ok": True}


def _note_beat(host, ct, cn, ls, hi):
    raw = dict(
        beat(),
        ph=host,
        ct=ct,
        cn=cn,
        ls=ls,
        hi=hi,
    )
    heartbeat.BEATS.note(host, heartbeat.parse(raw))


def _note_base(host):
    heartbeat.BASES.note(host, base())


def _tick(run, runner):
    runs.forget_ready_checks()
    runs.tick(run, runner)


def test_seventeen_containers_is_unknown_not_gone():
    raw = dict(beat(), ct=_partial_ct(), cn=17)
    parsed = heartbeat.parse(raw)
    assert parsed["cf"] is False
    ans = heartbeat.answer_probe(parsed, {"op": "probe"}, base())
    assert ans["containers"] is None
    heartbeat.BEATS.note("node-a", parsed)
    assert heartbeat.container_seen("node-a", "sparksi-demo") is None


def test_partial_beat_stops_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    runs.reset_for_tests()
    run = _mk_run()
    runs._save(run)
    runner = Runner()

    _note_beat(HEAD, _partial_ct(), 17, [[8010, "w"]], {"8010": [200, [], 200, []]})
    _note_beat(WORKER, _partial_ct(), 17, [], {})
    _note_base(HEAD)
    _note_base(WORKER)

    for _ in range(3):
        _tick(run, runner)

    assert "stop" not in runner.ops
    saved = runs.get_run("pair-1")
    assert saved["phase"] == "ready"


def test_forged_beat_stops_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    runs.reset_for_tests()
    run = _mk_run()
    runs._save(run)
    runner = Runner()

    _note_beat(HEAD, [], 0, [], {})
    _note_beat(WORKER, [], 0, [], {})
    _note_base(HEAD)
    _note_base(WORKER)

    for _ in range(3):
        _tick(run, runner)

    assert "stop" not in runner.ops
    saved = runs.get_run("pair-1")
    assert saved["phase"] == "ready"
    assert saved.get("refused_probes", 0) in (0, None)
    assert "probe" in runner.ops


def test_beat_timeout_never_counts_as_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    runs.reset_for_tests()
    run = _mk_run()
    runs._save(run)
    runner = Runner()

    _note_beat(HEAD, [["sparksi-demo", [8010]]], 1, [[8010, "w"]], {"8010": [None, [], None, []]})
    _note_beat(WORKER, [["sparksi-demo", []]], 1, [], {})
    _note_base(HEAD)
    _note_base(WORKER)

    for _ in range(5):
        _tick(run, runner)

    saved = runs.get_run("pair-1")
    assert saved["phase"] == "ready"
    assert saved["phase"] != "failed"
    assert "stop" not in runner.ops
    assert saved.get("refused_probes", 0) in (0, None)


def test_node_cards_read_through_the_production_runner(monkeypatch):
    """Production leaves STATE.runner unset: the node cards must still get a real runner."""
    from sparksi import serve

    real = FakeRunner()
    seen = {}
    monkeypatch.setattr(serve.STATE, "runner", None)
    monkeypatch.setattr(serve, "active_runner", lambda cat: real)
    monkeypatch.setattr(serve, "_read", lambda name, cat, runner, fn, **kw: fn())
    monkeypatch.setattr(serve, "node_report", lambda cat, runner, force=False: seen.setdefault("r", runner) and {"nodes": []})
    serve._node_read({})
    assert seen["r"].unwrap() is real
    with pytest.raises(TypeError):
        heartbeat.HeartbeatReads(None)
