# SPDX-License-Identifier: Apache-2.0

"""GPU max-clock cap: argv allowlist, parsing, validation, and the API."""

from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

import pytest

from sparksi.gpu_clocks import (
    QUERY_CLOCK,
    QUERY_CSV,
    QUERY_INDEX,
    apply_clock,
    clock_lock_argv,
    parse_clock_report,
    parse_gpu_indices,
    parse_request_mhz,
    validate_max_mhz,
)
from sparksi.lock import Lock
from sparksi.never import (
    launcher_refusal,
    nvidia_smi_is_clock_lock,
    nvidia_smi_is_readonly,
    observation_refusal,
    subprocess_is_readonly,
)
from sparksi.nodes import parse_nvidia
from sparksi.serve import STATE, DualServer, Handler

CLOCK_NO_CAP = """
==============NVSMI LOG==============

Timestamp                                              : Mon Nov 23 08:13:50 2026
Driver Version                                         : 580.178.04

Attached GPUs                                          : 1
GPU 00000000:01:00.0
    Clocks
        Graphics                                       : 2411 MHz
        SM                                             : 2411 MHz
        Memory                                         : N/A
        Video                                          : 2067 MHz
    Applications Clocks
        Graphics                                       : 2418 MHz
        Memory                                         : N/A
    Default Applications Clocks
        Graphics                                       : 2418 MHz
        Memory                                         : N/A
    Max Clocks
        Graphics                                       : 3003 MHz
        SM                                             : 3003 MHz
        Memory                                         : N/A
        Video                                          : 3003 MHz
    Supported Clocks                                   : N/A
"""

CLOCK_WITH_CAP = """
GPU 00000000:01:00.0
    Clocks
        Graphics                                       : 1800 MHz
    Max Clocks
        Graphics                                       : 3003 MHz
    GPU Locked Clocks
        Min                                              : 0 MHz
        Max                                              : 2000 MHz
    Supported Clocks
        Memory                                         : 1111 MHz
            Graphics                                   : 210 MHz
            Graphics                                   : 3003 MHz
"""


def test_clock_lock_allowlist_is_exact() -> None:
    set_cmd = ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000"]
    clear_cmd = ["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"]
    assert nvidia_smi_is_clock_lock(set_cmd)
    assert nvidia_smi_is_clock_lock(clear_cmd)
    assert nvidia_smi_is_clock_lock(["sudo", "-n", "nvidia-smi", "-i", "1", "-lgc", "0,210"])
    assert clock_lock_argv(0, 2000) == set_cmd
    assert clock_lock_argv(0, None) == clear_cmd
    for argv in (
        ["nvidia-smi", "-lgc", "0,2000"],
        ["nvidia-smi", "-rgc"],
        ["sudo", "nvidia-smi", "-i", "0", "-lgc", "0,2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000", "--mode=0"],
        ["sudo", "-n", "-u", "root", "nvidia-smi", "-i", "0", "-rgc"],
        ["sudo", "-n", "nvidia-smi", "-lgc", "0,2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "1,2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,02000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000;id"],
        ["sudo", "-n", "nvidia-smi", "-i", "0;id", "-lgc", "0,2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0$(id)", "-rgc"],
        ["sudo", "-n", "nvidia-smi", "-i", "-1", "-rgc"],
        ["sudo", "-n", "nvidia-smi", "-i", "00", "-rgc"],
        ["sudo", "-n", "/usr/bin/nvidia-smi", "-i", "0", "-rgc"],
        ["/usr/bin/sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-ac", "0,2000"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-pl", "100"],
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-r"],
        ["sudo"],
    ):
        assert not nvidia_smi_is_clock_lock(argv), argv
        assert observation_refusal(argv) is not None, argv
        assert not subprocess_is_readonly(argv), argv
    # Catalog service starts may still use sudo systemctl; that path is unchanged.
    assert launcher_refusal(["sudo", "systemctl", "start", "studio-b"]) is None
    # Clock caps are not a launch command; observation and the clock allowlist handle them.
    assert observation_refusal(set_cmd) is not None
    assert observation_refusal(clear_cmd) is not None


def test_clock_queries_are_readonly() -> None:
    assert observation_refusal(QUERY_CSV) is None
    assert observation_refusal(QUERY_CLOCK) is None
    assert observation_refusal(QUERY_INDEX) is None
    assert nvidia_smi_is_readonly(QUERY_CLOCK[1:])
    assert nvidia_smi_is_readonly(["-q", "-d", "CLOCK"])
    assert not nvidia_smi_is_readonly(["-q", "-d", "CLOCK", "-lgc", "0,2000"])
    assert not nvidia_smi_is_readonly(["-q", "-d", "EVIL"])
    assert subprocess_is_readonly(QUERY_CSV)
    assert subprocess_is_readonly(QUERY_CLOCK)


def test_parse_clock_report_and_csv() -> None:
    open_gpu = parse_clock_report(CLOCK_NO_CAP)
    assert open_gpu["clock_mhz"] == 2411
    assert open_gpu["clock_max_mhz"] == 3003
    assert open_gpu["clock_cap_mhz"] is None
    assert open_gpu["clock_min_mhz"] is None
    assert open_gpu["clock_cap_reported"] is False
    capped = parse_clock_report(CLOCK_WITH_CAP)
    assert capped["clock_cap_reported"] is True
    assert capped["clock_mhz"] == 1800
    assert capped["clock_max_mhz"] == 3003
    assert capped["clock_cap_mhz"] == 2000
    assert capped["clock_cap_min_mhz"] == 0
    assert capped["clock_min_mhz"] == 210
    csv = parse_nvidia("12, 47, 2411, 3003\n")
    assert csv == {"util_pct": 12.0, "temp_c": 47.0, "clock_mhz": 2411.0, "clock_max_mhz": 3003.0}
    assert parse_nvidia("12, 47\n") == {"util_pct": 12.0, "temp_c": 47.0}
    assert parse_gpu_indices("0\n") == [0]
    assert parse_gpu_indices("0\n1\n") == [0, 1]
    assert parse_gpu_indices("0;id\n00\n-1\nindex\n 2 \n") == [2]
    spaced = "Clocks\n" + "Graphics" + " " * 200 + ":" + " " * 200 + "2411 MHz\nMax Clocks\nGraphics:3003\n"
    parsed = parse_clock_report(spaced)
    assert parsed["clock_mhz"] == 2411
    assert parsed["clock_max_mhz"] == 3003


def test_validate_and_request_body() -> None:
    gpu = {"clock_min_mhz": None, "clock_max_mhz": 3003}
    assert validate_max_mhz(None, gpu) is None
    assert validate_max_mhz(2000, gpu) is None
    assert "between 1 and 3003" in (validate_max_mhz(0, gpu) or "")
    assert validate_max_mhz(1, gpu) is None
    assert validate_max_mhz(3003, gpu) is None
    assert "between 1 and 3003" in (validate_max_mhz(3004, gpu) or "")
    assert "maximum clock" in (validate_max_mhz(2000, {}) or "")
    gpu_min = {"clock_min_mhz": 210, "clock_max_mhz": 3003}
    assert "between 210 and 3003" in (validate_max_mhz(100, gpu_min) or "")
    assert parse_request_mhz({}) == (None, None)
    assert parse_request_mhz({"max_mhz": None}) == (None, None)
    assert parse_request_mhz({"max_mhz": ""}) == (None, None)
    assert parse_request_mhz({"max_mhz": 2000}) == (2000, None)
    assert parse_request_mhz({"max_mhz": "2000"}) == (2000, None)
    assert parse_request_mhz({"max_mhz": 2000.5})[0] is None
    assert parse_request_mhz({"max_mhz": True})[0] is None
    assert parse_request_mhz({"max_mhz": "2000;id"})[0] is None
    assert parse_request_mhz({"max_mhz": "0,2000"})[0] is None
    assert parse_request_mhz({"max_mhz": "-lgc"})[0] is None


class ClockRunner:
    def __init__(self, *, fail: str | None = None, cap: int | None = None, max_mhz: int = 3003, indexes: list[int] | None = None, index_text: str | None = None):
        self.fail = fail
        self.cap = cap
        self.max_mhz = max_mhz
        self.indexes = list(indexes) if indexes is not None else [0]
        self.index_text = index_text
        self.applied: list[list[str]] = []

    def capture(self, host, argv, timeout=8):
        if argv == QUERY_CSV:
            current = self.cap if self.cap is not None else 2411
            return {"ok": True, "code": 0, "stdout": f"4, 41, {current}, {self.max_mhz}\n", "stderr": ""}
        if argv == QUERY_CLOCK:
            if self.cap is None:
                text = CLOCK_NO_CAP.replace("3003", str(self.max_mhz)).replace("2411", "2411")
            else:
                text = CLOCK_WITH_CAP.replace("2000", str(self.cap)).replace("3003", str(self.max_mhz)).replace("1800", str(self.cap))
            return {"ok": True, "code": 0, "stdout": text, "stderr": ""}
        if argv == QUERY_INDEX:
            text = self.index_text if self.index_text is not None else "".join(f"{i}\n" for i in self.indexes)
            return {"ok": True, "code": 0, "stdout": text, "stderr": ""}
        return {"ok": False, "error": "unexpected query", "stdout": "", "stderr": "", "code": 1}

    def run_gpu_clock(self, host, gpu_index, max_mhz, timeout=8):
        argv = clock_lock_argv(gpu_index, max_mhz)
        self.applied.append(list(argv))
        if self.fail:
            return {"ok": False, "error": self.fail, "stdout": "", "stderr": self.fail, "code": 1}
        if max_mhz is None:
            self.cap = None
        else:
            self.cap = max_mhz
        return {"ok": True, "code": 0, "stdout": "", "stderr": "", "error": None}


def test_apply_sets_and_clears_and_surfaces_permission_error() -> None:
    runner = ClockRunner()
    set_ok = apply_clock(runner, "head", 2000)
    assert set_ok["ok"] is True
    assert set_ok["gpu"]["clock_cap_mhz"] == 2000
    assert runner.applied == [["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000"]]
    cleared = apply_clock(runner, "head", None)
    assert cleared["ok"] is True
    assert cleared["gpu"]["clock_cap_mhz"] is None
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"]
    denied = ClockRunner(fail="sudo: a password is required")
    out = apply_clock(denied, "head", 2000)
    assert out["ok"] is False
    assert "sudo: a password is required" in out["error"]
    high = apply_clock(ClockRunner(), "head", 9000)
    assert high["ok"] is False
    assert "between 1 and 3003" in high["error"]
    assert high["kind"] == "invalid"


@pytest.fixture
def clock_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    lock = Lock(tmp_path / "lock")
    lock.release()
    STATE.lock = lock
    import sparksi.gpu_clocks as gpu_clocks_mod

    gpu_clocks_mod.reset_for_tests()
    runner = ClockRunner()
    STATE.runner = runner
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    yield f"http://127.0.0.1:{port}", runner, lock
    httpd.shutdown()
    thread.join(timeout=5)
    STATE.runner = None


def _post(url: str, body: dict, origin: str | None = None) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if origin:
        headers["Origin"] = origin
        headers["Host"] = url.split("/")[2]
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=8) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        if hasattr(exc, "code"):
            return exc.code, json.loads(exc.read().decode())  # type: ignore[attr-defined]
        raise


def test_api_sets_clock_and_refuses_when_locked(clock_server) -> None:
    base, runner, lock = clock_server
    url = base + "/api/nodes/a/gpu-clocks"
    status, body = _post(url, {"max_mhz": 2000, "from_ui": True})
    assert status == 200, body
    assert body["ok"] is True
    assert body["gpu"]["clock_cap_mhz"] == 2000
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000"]
    lock.acquire("test")
    status, body = _post(url, {"max_mhz": 1800, "from_ui": True})
    assert status == 403
    assert "Unlock first" in body["error"]
    # A second click while locked still only refuses; nothing new is applied.
    applied = list(runner.applied)
    status, _body = _post(url, {"max_mhz": 1800, "from_ui": True})
    assert status == 403
    assert runner.applied == applied


def test_apply_uses_queried_indexes_and_ignores_injected_index_text() -> None:
    runner = ClockRunner(indexes=[0, 1])
    out = apply_clock(runner, "head", 1800)
    assert out["ok"] is True
    assert runner.applied == [
        ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,1800"],
        ["sudo", "-n", "nvidia-smi", "-i", "1", "-lgc", "0,1800"],
    ]
    poisoned = ClockRunner(index_text="0;rm -rf /\n00\n-1\n")
    bad = apply_clock(poisoned, "head", 1800)
    assert bad["ok"] is False
    assert poisoned.applied == []
    assert "index could not be read" in bad["error"]


def test_api_ignores_client_gpu_index_and_reports_sudo_failure(clock_server) -> None:
    base, runner, lock = clock_server
    url = base + "/api/nodes/a/gpu-clocks"
    status, body = _post(url, {"max_mhz": 1800, "gpu": "0;id", "index": "-1", "from_ui": True})
    assert status == 200, body
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,1800"]
    runner.fail = "sudo: a password is required"
    status, body = _post(url, {"max_mhz": 1700, "from_ui": True})
    assert status == 502
    assert body["ok"] is False
    assert body["error"] == "sudo: a password is required"
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,1700"]

def test_api_sets_and_clears_clock_on_worker(clock_server) -> None:
    base, runner, lock = clock_server
    url = base + "/api/nodes/b/gpu-clocks"
    status, body = _post(url, {"max_mhz": 1800, "from_ui": True})
    assert status == 200, body
    assert body["ok"] is True
    assert body["id"] == "b"
    assert body["role"] == "worker"
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,1800"]
    status, body = _post(url, {"max_mhz": None, "from_ui": True})
    assert status == 200, body
    assert body["ok"] is True
    assert runner.applied[-1] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"]


def test_ssh_runner_builds_sudo_n_argv_and_run_argv_refuses_nvidia_smi(monkeypatch) -> None:
    import subprocess

    import sparksi.remote as remote_mod
    from sparksi.remote import SshRunner

    seen: list[list[str]] = []

    def fake(cmd, *a, **k):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake)
    runner = SshRunner({"hosts": {"head": {"ssh": "spark-x"}}})
    ok = runner.run_gpu_clock("head", 0, 2000)
    assert ok["ok"] is True
    cmd = seen[-1]
    assert cmd[cmd.index("spark-x") + 1 :] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-lgc", "0,2000"]
    cleared = runner.run_gpu_clock("head", 0, None)
    assert cleared["ok"] is True
    assert seen[-1][seen[-1].index("spark-x") + 1 :] == ["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"]

    def fail(cmd, *a, **k):
        seen.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "sudo: a password is required")

    monkeypatch.setattr(remote_mod.subprocess, "run", fail)
    denied = runner.run_gpu_clock("head", 0, 1800)
    assert denied["ok"] is False
    assert denied["error"] == "sudo: a password is required"

    seen.clear()
    monkeypatch.setattr(remote_mod.subprocess, "run", fake)
    bad_index = runner.run_gpu_clock("head", -1, 2000)
    assert bad_index["ok"] is False
    injected = runner.run_gpu_clock("head", 0, "2000;id")  # type: ignore[arg-type]
    assert injected["ok"] is False
    assert seen == []

    refused = runner.run_argv("head", ["nvidia-smi", "-lgc", "0,2000"])
    assert refused["ok"] is False
    assert "nvidia-smi" in (refused.get("error") or "")
    refused_sudo = runner.run_argv("head", ["sudo", "-n", "nvidia-smi", "-i", "0", "-rgc"])
    assert refused_sudo["ok"] is False
    assert "nvidia-smi" in (refused_sudo.get("error") or "")
    assert seen == []


# Real GB10 report shape (driver 580.x): no locked-clocks section, sample blocks carry Max/Min N/A.
CLOCK_GB10 = """
GPU 0000000F:01:00.0
    Clocks
        Graphics                                       : 2437 MHz
        SM                                             : 2437 MHz
        Memory                                         : N/A
        Video                                          : 2132 MHz
    Max Clocks
        Graphics                                       : 3003 MHz
    SM Clock Samples
        Duration                                       : N/A
        Max                                            : N/A
        Min                                            : N/A
    Clock Policy
        Auto Boost                                     : N/A
"""


class Gb10Runner(ClockRunner):
    """A driver that applies -lgc but never prints the lock back."""

    def capture(self, host, argv, timeout=8):
        if argv == QUERY_CLOCK:
            return {"ok": True, "code": 0, "stdout": CLOCK_GB10, "stderr": ""}
        return super().capture(host, argv, timeout)


def test_gb10_report_keeps_cap_unknown_and_memo_shows_what_was_set() -> None:
    import sparksi.gpu_clocks as gpu_clocks_mod

    gpu_clocks_mod.reset_for_tests()
    parsed = parse_clock_report(CLOCK_GB10)
    assert parsed["clock_mhz"] == 2437
    assert parsed["clock_max_mhz"] == 3003
    assert parsed["clock_cap_mhz"] is None
    assert parsed["clock_cap_reported"] is False
    runner = Gb10Runner()
    out = apply_clock(runner, "head", 2000)
    assert out["ok"] is True
    assert out["gpu"]["clock_cap_set"]["mhz"] == 2000
    assert gpu_clocks_mod.with_memo("worker", {"clock_cap_reported": False}).get("clock_cap_set") is None
    cleared = apply_clock(runner, "head", None)
    assert cleared["ok"] is True
    assert cleared["gpu"]["clock_cap_set"]["mhz"] is None
    # A driver that does report the lock wins over the memo.
    assert "clock_cap_set" not in gpu_clocks_mod.with_memo("head", {"clock_cap_reported": True, "clock_cap_mhz": None})
    gpu_clocks_mod.reset_for_tests()


def test_partial_multi_gpu_failure_is_failed_and_remembered() -> None:
    import sparksi.gpu_clocks as gpu_clocks_mod

    gpu_clocks_mod.reset_for_tests()

    class Half(Gb10Runner):
        def run_gpu_clock(self, host, gpu_index, max_mhz, timeout=8):
            if gpu_index == 1:
                self.applied.append(clock_lock_argv(gpu_index, max_mhz))
                return {"ok": False, "error": "sudo: a password is required", "code": 1}
            return super().run_gpu_clock(host, gpu_index, max_mhz, timeout)

    out = apply_clock(Half(indexes=[0, 1]), "head", 1500)
    assert out["ok"] is False
    assert out["kind"] == "failed"
    assert out["error"] == "sudo: a password is required"
    assert out["gpu"]["clock_cap_set"]["mhz"] == 1500
    gpu_clocks_mod.reset_for_tests()


def test_api_needs_confirm_while_a_model_runs(clock_server, monkeypatch) -> None:
    import sparksi.serve as serve_mod

    base, runner, lock = clock_server
    loaded = [{"host": "a", "name": "m", "port": 8000}]
    monkeypatch.setattr(serve_mod, "monitor_loaded", lambda: loaded)
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
    assert status == 409
    assert body["needs_confirm"] is True
    assert "running" in body["error"]
    assert runner.applied == []
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 2000, "from_ui": True, "confirm_running": "yes"})
    assert status == 409, "only a JSON true is an OK"
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 2000, "from_ui": True, "confirm_running": True})
    assert status == 200, body
    # A pair spanning the worker also counts there; a video service does not.
    loaded[:] = [{"host": "a", "span": ["a", "b"], "port": 8000}]
    status, body = _post(base + "/api/nodes/b/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
    assert status == 409
    loaded[:] = [{"host": "b", "video_service": True, "kind": "video_service", "port": 8188}]
    status, body = _post(base + "/api/nodes/b/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
    assert status == 200, body
    # Unknown monitor state also asks.
    monkeypatch.setattr(serve_mod, "monitor_loaded", lambda: None)
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 1900, "from_ui": True})
    assert status == 409
    assert body["needs_confirm"] is True


def test_api_refuses_a_second_save_while_one_runs(clock_server) -> None:
    import sparksi.gpu_clocks as gpu_clocks_mod
    from sparksi.catalog import load_catalog
    from sparksi.nodes import node_specs

    base, runner, lock = clock_server
    key = next(item["key"] for item in node_specs(load_catalog()) if item["id"] == "a")
    assert gpu_clocks_mod.begin_save(key)
    assert not gpu_clocks_mod.begin_save(key)
    try:
        status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
        assert status == 409
        assert "already running" in body["error"]
        assert runner.applied == []
    finally:
        gpu_clocks_mod.end_save(key)
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
    assert status == 200, body
    # The claim is released after each save, success or failure.
    assert gpu_clocks_mod.begin_save(key)
    gpu_clocks_mod.end_save(key)


def test_api_unknown_spark_and_lock_with_token(clock_server) -> None:
    base, runner, lock = clock_server
    status, body = _post(base + "/api/nodes/zz/gpu-clocks", {"max_mhz": 2000, "from_ui": True})
    assert status == 404
    status, body = _post(base + "/api/nodes/a/gpu-clocks", {"max_mhz": "2000;id", "from_ui": True})
    assert status == 400
    assert runner.applied == []


def test_local_runner_run_argv_refuses_nvidia_smi() -> None:
    from sparksi.remote import LocalRunner

    calls = []
    runner = LocalRunner(lambda h, r, t: {"ok": True}, argv_fn=lambda h, r, t: calls.append(r) or {"ok": True})
    out = runner.run_argv("head", ["/usr/bin/nvidia-smi", "-rgc"])
    assert out["ok"] is False
    assert calls == []
