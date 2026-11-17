# SPDX-License-Identifier: Apache-2.0

"""Per-Spark node facts, status rebuilt from them, and honest load jobs."""

from __future__ import annotations

import json
import logging
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.activity import store as activity_store
from sparksi.health import DEFAULT_HEALTH_TIMEOUT_S, health_paths_for, health_timeout_for
from sparksi.lock import Lock
from sparksi.never import observation_refusal
from sparksi.nodes import (
    node_report,
    parse_docker_ps,
    parse_free_b,
    parse_meminfo,
    parse_nvidia,
    parse_ss_ltn,
    parse_systemctl_show,
    reset_cache_for_tests,
)
from sparksi.serve import STATE, DualServer, Handler
from sparksi.status import assemble_status

GIB = 1024**3


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    from sparksi import jobs

    jobs.reset_for_tests()
    reset_activity()
    reset_cache_for_tests()
    yield
    jobs.reset_for_tests()
    reset_cache_for_tests()
    STATE.runner = None


def _free(total_gib: int, used_gib: int, free_gib: int, avail_gib: int) -> str:
    total, used, free, avail = (n * GIB for n in (total_gib, used_gib, free_gib, avail_gib))
    cache = total - used - free
    return (
        "               total        used        free      shared  buff/cache   available\n"
        f"Mem:     {total} {used} {free} 0 {cache} {avail}\n"
        "Swap: 0 0 0\n"
    )


HEAD_FREE = _free(128, 40, 20, 88)
WORKER_FREE = _free(121, 10, 90, 100)
SS = (
    "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
    "LISTEN 0 0 127.0.0.1:8014 0.0.0.0:*\n"
    "LISTEN 0 0 0.0.0.0:8188 0.0.0.0:*\n"
    "LISTEN 0 0 [::]:22 [::]:*\n"
)
DOCKER = "model-a\texample/vllm:1\t127.0.0.1:8014->8000/tcp\n"
UNITS = (
    "Id=gone.service\nLoadState=not-found\nActiveState=inactive\n"
    "\n"
    "Id=live.service\nLoadState=loaded\nActiveState=active\n"
)
MEMINFO = "MemTotal:       134217728 kB\nMemFree:        20971520 kB\nMemAvailable:   83886080 kB\n"
SPECS = [
    {"id": "head", "name": "head", "role": "head", "key": "head", "ssh": "head.example.invalid"},
    {"id": "worker", "name": "worker", "role": "worker", "key": "worker", "ssh": "worker.example.invalid"},
]
CATALOG = {
    "host": {"name": "head", "uma_gib": 128, "swap": "off"},
    "tenants": {},
    "recipes": {"wide": {"timeout_s": 7200, "members": ["wide-model"]}},
    "seats": {"brain": "gone", "vision": "live"},
    "units_status": ["gone.service", "live.service"],
    "hosts": {
        "head": {"ssh": "head.example.invalid"},
        "worker": {"ssh": "worker.example.invalid"},
    },
    "residents": [
        {
            "id": "gone",
            "kind": "vllm",
            "role": "brain",
            "served_name": "gone",
            "port": 8014,
            "unit": "gone.service",
            "weights": "/weights/gone",
            "listen_host": "head",
        },
        {
            "id": "live",
            "kind": "vllm",
            "role": "vision",
            "served_name": "live",
            "port": 8014,
            "unit": "live.service",
            "weights": "/weights/live",
            "container": "model-a",
            "listen_host": "head",
        },
        {"id": "comfy", "kind": "comfy", "role": "video", "port": 8188, "listen_host": "head"},
    ],
}


class Scripted:
    def __init__(self, *, worker_down: bool = True, sleep_s: float = 0.0):
        self.worker_down = worker_down
        self.sleep_s = sleep_s
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self._active = 0
        self.max_active = 0
        self._lock = threading.Lock()
        self.barrier = threading.Barrier(2)

    def capture(self, host, argv, timeout=8):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        try:
            if argv[0] == "free":
                try:
                    self.barrier.wait(timeout=2)
                except threading.BrokenBarrierError:
                    pass
            if self.sleep_s:
                time.sleep(self.sleep_s)
            self.calls.append((host, tuple(argv)))
            if self.worker_down and host == "worker":
                return {
                    "ok": False,
                    "code": 255,
                    "error": "ssh: connect to worker.example.invalid: Connection timed out",
                    "stdout": "",
                    "stderr": "",
                }
            return self._ok(host, argv)
        finally:
            with self._lock:
                self._active -= 1

    def _ok(self, host, argv):
        prog = argv[0]
        if prog == "free":
            return {"ok": True, "code": 0, "stdout": HEAD_FREE if host == "head" else WORKER_FREE, "stderr": ""}
        if prog == "nvidia-smi":
            text = "12, 47\n" if host == "head" else ""
            return {"ok": bool(text), "code": 0 if text else 127, "stdout": text, "stderr": ""}
        if prog == "ss":
            return {"ok": True, "code": 0, "stdout": SS if host == "head" else "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n", "stderr": ""}
        if prog == "docker":
            return {"ok": True, "code": 0, "stdout": DOCKER if host == "head" else "", "stderr": ""}
        if prog == "systemctl":
            return {"ok": True, "code": 0, "stdout": UNITS if host == "head" else "", "stderr": ""}
        if prog == "test":
            exists = argv[-1] == "/weights/live"
            return {"ok": exists, "code": 0 if exists else 1, "stdout": "", "stderr": ""}
        if prog == "cat":
            return {"ok": True, "code": 0, "stdout": MEMINFO, "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "unexpected"}


def test_parsers_read_free_meminfo_gpu_ports_and_containers():
    mem = parse_free_b(HEAD_FREE)
    assert mem["total_gib"] == 128
    assert mem["used_gib"] == 40
    assert mem["available_gib"] == 88
    info = parse_meminfo(MEMINFO)
    assert info["total_gib"] == 128
    assert info["available_gib"] == 80
    assert info["used_gib"] == 48
    gpu = parse_nvidia("12, 47\n")
    assert gpu == {"util_pct": 12.0, "temp_c": 47.0}
    assert parse_nvidia("") is None
    ports = parse_ss_ltn(SS)
    assert [(row["address"], row["port"]) for row in ports] == [
        ("127.0.0.1", 8014),
        ("0.0.0.0", 8188),
        ("[::]", 22),
    ]
    rows = parse_docker_ps(DOCKER)
    assert rows == [{"name": "model-a", "image": "example/vllm:1", "ports": "127.0.0.1:8014->8000/tcp"}]
    units = parse_systemctl_show(UNITS)
    assert units["gone.service"]["load_state"] == "not-found"
    assert units["live.service"]["active_state"] == "active"


def test_observation_commands_stay_read_only():
    assert observation_refusal(["ss", "-ltn"]) is None
    assert observation_refusal(["free", "-b"]) is None
    assert observation_refusal(["systemctl", "show", "model.service", "-p", "ActiveState"]) is None
    assert observation_refusal(["systemctl", "start", "model.service"]) is not None
    assert observation_refusal(["docker", "ps", "--format", "{{.Names}}"]) is None
    assert observation_refusal(["docker", "stop", "model-a"]) is not None


def test_nodes_parse_live_facts_and_cache(monkeypatch):
    monkeypatch.setattr("sparksi.nodes.node_specs", lambda catalog=None: SPECS)
    clock = {"t": 0.0}
    runner = Scripted()
    first = node_report(CATALOG, runner, specs=SPECS, now=lambda: clock["t"], ttl=5)
    assert first["cached"] is False
    assert runner.max_active >= 2
    head, worker = first["nodes"]
    assert head["reachable"] is True
    assert head["error"] is None
    assert head["mem"]["total_gib"] == 128
    assert head["mem"]["used_gib"] == 40
    assert head["mem"]["available_gib"] == 88
    assert head["gpu"]["util_pct"] == 12
    assert head["gpu"]["temp_c"] == 47
    assert 8188 in head["ports"]
    assert head["containers"][0]["name"] == "model-a"
    assert head["containers"][0]["image"] == "example/vllm:1"
    assert "ssh" not in head
    assert worker["reachable"] is False
    assert "worker.example.invalid" in worker["error"]
    assert worker["mem"] is None
    calls = len(runner.calls)
    second = node_report(CATALOG, runner, specs=SPECS, now=lambda: clock["t"], ttl=5)
    assert second["cached"] is True
    assert len(runner.calls) == calls
    clock["t"] = 6
    third = node_report(CATALOG, runner, specs=SPECS, now=lambda: clock["t"], ttl=5)
    assert third["cached"] is False
    assert len(runner.calls) > calls


def test_meminfo_when_free_is_absent():
    class Runner:
        def capture(self, host, argv, timeout=8):
            if argv[0] == "free":
                return {"ok": False, "code": 127, "error": "not found", "stdout": "", "stderr": ""}
            if argv[0] == "cat":
                return {"ok": True, "code": 0, "stdout": MEMINFO, "stderr": ""}
            if argv[0] == "test":
                return {"ok": False, "code": 1, "stdout": "", "stderr": ""}
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}

    specs = [dict(SPECS[0])]
    report = node_report(CATALOG, Runner(), specs=specs, ttl=0, force=True)
    assert report["nodes"][0]["reachable"] is True
    assert report["nodes"][0]["mem"]["total_gib"] == 128
    assert report["nodes"][0]["mem"]["used_gib"] == 48


def test_status_uses_spark_memory_and_marks_stale_rows(tmp_path):
    report = node_report(CATALOG, Scripted(sleep_s=0), specs=SPECS, ttl=0, force=True)
    status = assemble_status(CATALOG, None, Lock(tmp_path / "lock"), nodes=report)
    assert status["mem"]["mem_total_gib"] == 128
    assert status["mem"]["mem_used_gib"] == 40
    assert status["mem"]["mem_avail_gib"] == 88
    assert status["nodes"][1]["reachable"] is False
    assert "worker.example.invalid" in status["nodes"][1]["error"]
    by_id = {row["id"]: row for row in status["residents"]}
    assert by_id["gone"]["stale"] is True
    assert by_id["gone"]["resident"] is False
    assert by_id["gone"]["unit_missing"] is True
    assert by_id["gone"]["path_missing"] is True
    assert by_id["live"]["stale"] is False
    assert by_id["live"]["resident"] is True
    assert by_id["comfy"]["resident"] is True
    assert status["comfy_live"] is True
    assert "lock" in status and "seats" in status and "floor" in status


def test_health_budget_default_recipe_and_long_start():
    assert health_timeout_for({}, {}) == DEFAULT_HEALTH_TIMEOUT_S == 1800
    assert health_timeout_for(CATALOG, {"id": "wide-model"}) == 7200
    assert health_timeout_for({}, {"id": "folded", "start_timeout_s": 57600}) == 57600
    assert health_paths_for({"port": 8188, "placement": "pair"}) == []
    assert health_paths_for({"placement": "pair", "port": 9000}) == ["/health", "/v1/models"]
    assert health_paths_for({"probe": "/v1/models", "port": 8014, "kind": "vllm"}) == ["/v1/models"]


def test_health_poll_ready_timeout_and_failure():
    from sparksi.jobs import perform_switch

    class Ready:
        def __init__(self):
            self.paths = []

        def invoke(self, host, req, timeout=0):
            paths = req.get("http_paths") or {}
            self.paths.append(paths.get("8014"))
            assert 8188 not in (req.get("ports") or [])
            return {"ok": True, "listening": {"8014": True}, "http": {"8014": {"status": 200}}}

        def run_argv(self, host, argv, timeout=0):
            return {"ok": True}

    ready = Ready()
    plan = {"steps": [{
        "op": "start",
        "host": "head",
        "port": 8014,
        "argv": ["start-model"],
        "health_timeout_s": 30,
        "http_paths": ["/health", "/v1/models"],
    }]}
    assert perform_switch(ready, plan, lambda step: False, pause=3)["ok"] is True
    assert ready.paths[0] == "/health"

    class Down:
        def __init__(self):
            self.invokes = 0

        def invoke(self, host, req, timeout=0):
            self.invokes += 1
            return {"ok": True, "listening": {"8014": False}, "http": {}}

        def run_argv(self, host, argv, timeout=0):
            return {"ok": True}

    clock = {"t": 1000.0}

    def now():
        return clock["t"]

    def snooze(seconds):
        clock["t"] += seconds

    down = Down()
    timed = perform_switch(
        down,
        {"steps": [{
            "op": "start",
            "host": "head",
            "port": 8014,
            "argv": ["start-model"],
            "health_timeout_s": 9,
            "http_paths": ["/health"],
        }]},
        lambda step: False,
        pause=3,
        now=now,
        sleep=snooze,
    )
    assert timed["ok"] is False
    assert timed["phase"] == "timed_out"
    assert "did not answer" in timed["error"]
    assert down.invokes >= 2

    class Sick:
        def invoke(self, host, req, timeout=0):
            return {"ok": True, "listening": {"8014": True}, "http": {"8014": {"status": 500}}}

        def run_argv(self, host, argv, timeout=0):
            return {"ok": False, "error": "launcher exited"}

    failed = perform_switch(
        Sick(),
        {"steps": [{"op": "start", "host": "head", "port": 8014, "argv": ["start-model"], "health_timeout_s": 30}]},
        lambda step: False,
        pause=0,
    )
    assert failed["ok"] is False
    assert failed["phase"] == "failed"
    assert failed["started"] is True
    assert "launcher exited" in failed["error"]


def test_health_failure_after_start_is_not_a_timeout():
    from sparksi.jobs import perform_switch

    class Sick:
        def invoke(self, host, req, timeout=0):
            return {"ok": True, "listening": {"9000": True}, "http": {"9000": {"status": 500}}}

        def run_argv(self, host, argv, timeout=0):
            return {"ok": True}

    result = perform_switch(
        Sick(),
        {"steps": [{
            "op": "start",
            "host": "head",
            "port": 9000,
            "argv": ["start-model"],
            "health_timeout_s": 30,
            "http_paths": ["/health"],
        }]},
        lambda step: False,
        pause=3,
    )
    assert result["phase"] == "failed"
    assert result["ok"] is False


def test_port_8188_is_not_probed():
    from sparksi.jobs import perform_switch

    class Runner:
        def invoke(self, host, req, timeout=0):
            raise AssertionError(req)

        def run_argv(self, host, argv, timeout=0):
            return {"ok": True}

    result = perform_switch(
        Runner(),
        {"steps": [{"op": "start", "host": "head", "port": 8188, "argv": ["echo", "up"]}]},
        lambda step: False,
    )
    assert result["ok"] is True
    assert result["started"] is True


def test_pair_stops_the_side_that_is_still_up():
    from sparksi.jobs import perform_switch

    class Runner:
        def __init__(self):
            self.n = 0
            self.argv = []

        def invoke(self, host, req, timeout=0):
            self.n += 1
            names = ["c-head"] if host == "head" else ["c-worker"]
            if self.n > 2 and host == "worker":
                names = []
            ports = req.get("ports") or []
            return {
                "ok": True,
                "listening": {str(port): False for port in ports},
                "containers": names,
                "http": {},
            }

        def run_argv(self, host, argv, timeout=0):
            self.argv.append((host, list(argv)))
            return {"ok": True}

    runner = Runner()
    result = perform_switch(
        runner,
        {"steps": [{
            "op": "start",
            "host": "head",
            "port": 9000,
            "argv": ["start-pair"],
            "placement": "pair",
            "health_timeout_s": 30,
            "http_paths": [],
            "stop_argv": ["stop-pair"],
            "sides": [
                {"role": "head", "host": "head", "container": "c-head", "port": 9000},
                {"role": "worker", "host": "worker", "container": "c-worker", "port": None},
            ],
        }]},
        lambda step: False,
        pause=0,
    )
    assert result["ok"] is False
    assert result["phase"] == "failed"
    assert result["failed_side"] == "worker"
    assert ("head", ["stop-pair"]) in runner.argv
    assert ("worker", ["stop-pair"]) not in runner.argv


def test_job_persists_and_reloads_unknown_until_health(tmp_path, caplog):
    from sparksi import jobs

    jobs._JOB = {
        "id": "abc",
        "kind": "switch",
        "phase": "waiting_health",
        "ok": None,
        "error": None,
        "started": time.time(),
        "id_row": "live",
        "host": "head",
        "health": {
            "host": "head",
            "port": 9000,
            "http_paths": ["/v1/models"],
            "timeout_s": 1800,
        },
    }
    jobs._persist()
    assert (tmp_path / "job.json").is_file()
    jobs._JOB = None
    unknown = jobs.restore_job(check=False)
    assert unknown["phase"] == "unknown"
    assert unknown["elapsed_s"] >= 0
    assert json.loads((tmp_path / "job.json").read_text())["job"]["phase"] == "unknown"

    class Runner:
        def invoke(self, host, req, timeout=0):
            assert 8188 not in (req.get("ports") or [])
            return {"ok": True, "listening": {"9000": True}, "http": {"9000": {"status": 200}}}

    with caplog.at_level(logging.INFO, logger="sparksi"):
        ready = jobs.restore_job(Runner(), check=True, resume=False)
    assert ready["phase"] == "ready"
    assert ready["ok"] is True
    events = activity_store().events()
    assert any(row["kind"] == "switch" and "phase=ready" in row["message"] for row in events)
    assert any("phase=ready" in rec.message for rec in caplog.records)


def test_switch_failure_is_audited_with_a_short_tail(caplog):
    from sparksi import jobs

    jobs._JOB = {
        "id": "1",
        "kind": "switch",
        "phase": "waiting_health",
        "ok": None,
        "started": time.time(),
        "id_row": "live",
        "host": "head",
    }
    noise = "x" * 500
    with caplog.at_level(logging.INFO, logger="sparksi"):
        jobs._finish(False, "port 9000 did not answer\n" + noise, phase="timed_out")
    event = activity_store().events()[0]
    assert event["kind"] == "switch"
    assert "phase=timed_out" in event["message"]
    assert "error=" in event["message"]
    assert len(event["message"]) < 500
    assert any("timed_out" in rec.message for rec in caplog.records)
    jobs._JOB = {
        "id": "2",
        "kind": "stop",
        "phase": "stopping",
        "ok": None,
        "started": time.time(),
        "id_row": "live",
        "host": "worker",
    }
    jobs._finish(True, phase="ready")
    assert any(row["kind"] == "stop" and "phase=ready" in row["message"] for row in activity_store().events())


def test_post_exception_is_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("sparksi.serve.start_switch", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("post handler blew up")))
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=60)
    STATE.lock = lock
    STATE.runner = None
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        req = Request(
            f"http://127.0.0.1:{port}/api/switch",
            data=json.dumps({"id": "live", "from_ui": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with caplog.at_level(logging.ERROR, logger="sparksi"):
            with pytest.raises(HTTPError) as caught:
                urlopen(req, timeout=10)
        assert caught.value.code == 500
        assert "post handler blew up" in caught.value.read().decode()
        assert any("POST /api/switch failed" in rec.message for rec in caplog.records)
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_api_nodes_and_status_routes(monkeypatch):
    monkeypatch.setattr("sparksi.nodes.node_specs", lambda catalog=None: SPECS)
    STATE.runner = Scripted()
    STATE.lock = Lock()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/nodes", timeout=15) as resp:
            body = json.loads(resp.read().decode())
        assert body["nodes"][0]["mem"]["total_gib"] == 128
        assert body["nodes"][0]["reachable"] is True
        assert "example.invalid" in body["nodes"][1]["error"]
        with urlopen(f"http://127.0.0.1:{port}/api/status", timeout=15) as resp:
            status = json.loads(resp.read().decode())
        assert status["mem"]["mem_total_gib"] == 128
        assert status["comfy_live"] is True
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_cache_for_tests()


def test_page_shows_loading_while_health_is_pending():
    from pathlib import Path

    text = Path("src/sparksi/static/index.html").read_text(encoding="utf-8")
    assert 'phase === "waiting_health"' in text
    assert 'setJobLine("loading  " + elapsed + "s")' in text
    assert "innerHTML" not in text
    assert "localStorage" not in text
    assert "http://" not in text
    assert "https://" not in text
