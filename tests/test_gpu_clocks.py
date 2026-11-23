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
    apply_clock,
    clock_lock_argv,
    parse_clock_report,
    parse_request_mhz,
    validate_max_mhz,
)
from sparksi.lock import Lock
from sparksi.never import nvidia_smi_is_clock_lock, nvidia_smi_is_readonly, observation_refusal, subprocess_is_readonly
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
    assert nvidia_smi_is_clock_lock(["nvidia-smi", "-lgc", "0,2000"])
    assert nvidia_smi_is_clock_lock(["nvidia-smi", "--lock-gpu-clocks=0,2000"])
    assert nvidia_smi_is_clock_lock(["/usr/bin/nvidia-smi", "-rgc"])
    assert nvidia_smi_is_clock_lock(["nvidia-smi", "--reset-gpu-clocks"])
    assert clock_lock_argv(2000) == ["nvidia-smi", "-lgc", "0,2000"]
    assert clock_lock_argv(None) == ["nvidia-smi", "-rgc"]
    for argv in (
        ["nvidia-smi", "-lgc", "2000"],
        ["nvidia-smi", "-lgc", "1,2000"],
        ["nvidia-smi", "-lgc", "0,2000", "--mode=0"],
        ["nvidia-smi", "-lgc", "0,02000"],
        ["nvidia-smi", "-lgc", "0,2000;id"],
        ["sudo", "nvidia-smi", "-lgc", "0,2000"],
        ["nvidia-smi", "-ac", "0,2000"],
        ["nvidia-smi", "-pl", "100"],
        ["nvidia-smi", "-r"],
        ["nvidia-smi", "-lgc", "0,2000", "-rgc"],
        ["nvidia-smi"],
    ):
        assert not nvidia_smi_is_clock_lock(argv), argv
        if argv and argv[0].rsplit("/", 1)[-1] == "nvidia-smi":
            assert not nvidia_smi_is_readonly(argv[1:]), argv
            assert observation_refusal(argv) is not None
            assert not subprocess_is_readonly(argv)


def test_clock_queries_are_readonly() -> None:
    assert observation_refusal(QUERY_CSV) is None
    assert observation_refusal(QUERY_CLOCK) is None
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
    capped = parse_clock_report(CLOCK_WITH_CAP)
    assert capped["clock_mhz"] == 1800
    assert capped["clock_max_mhz"] == 3003
    assert capped["clock_cap_mhz"] == 2000
    assert capped["clock_cap_min_mhz"] == 0
    assert capped["clock_min_mhz"] == 210
    csv = parse_nvidia("12, 47, 2411, 3003\n")
    assert csv == {"util_pct": 12.0, "temp_c": 47.0, "clock_mhz": 2411.0, "clock_max_mhz": 3003.0}
    assert parse_nvidia("12, 47\n") == {"util_pct": 12.0, "temp_c": 47.0}


def test_validate_and_request_body() -> None:
    gpu = {"clock_min_mhz": None, "clock_max_mhz": 3003}
    assert validate_max_mhz(None, gpu) is None
    assert validate_max_mhz(2000, gpu) is None
    assert validate_max_mhz(0, gpu) is None
    assert validate_max_mhz(3003, gpu) is None
    assert "between 0 and 3003" in (validate_max_mhz(3004, gpu) or "")
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


class ClockRunner:
    def __init__(self, *, fail: str | None = None, cap: int | None = None, max_mhz: int = 3003):
        self.fail = fail
        self.cap = cap
        self.max_mhz = max_mhz
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
        return {"ok": False, "error": "unexpected query", "stdout": "", "stderr": "", "code": 1}

    def run_gpu_clock(self, host, argv, timeout=8):
        self.applied.append(list(argv))
        if self.fail:
            return {"ok": False, "error": self.fail, "stdout": "", "stderr": self.fail, "code": 1}
        if argv == ["nvidia-smi", "-rgc"]:
            self.cap = None
        elif argv[:2] == ["nvidia-smi", "-lgc"]:
            self.cap = int(argv[2].split(",", 1)[1])
        return {"ok": True, "code": 0, "stdout": "", "stderr": "", "error": None}


def test_apply_sets_and_clears_and_surfaces_permission_error() -> None:
    runner = ClockRunner()
    set_ok = apply_clock(runner, "head", 2000)
    assert set_ok["ok"] is True
    assert set_ok["gpu"]["clock_cap_mhz"] == 2000
    assert runner.applied == [["nvidia-smi", "-lgc", "0,2000"]]
    cleared = apply_clock(runner, "head", None)
    assert cleared["ok"] is True
    assert cleared["gpu"]["clock_cap_mhz"] is None
    assert runner.applied[-1] == ["nvidia-smi", "-rgc"]
    denied = ClockRunner(fail="Unable to set GPU locked clocks \"(gpuClockMin 0, gpuClockMax 2000)\" for GPU 00000000:01:00.0: Insufficient Permissions")
    out = apply_clock(denied, "head", 2000)
    assert out["ok"] is False
    assert "Insufficient Permissions" in out["error"]
    high = apply_clock(ClockRunner(), "head", 9000)
    assert high["ok"] is False
    assert "between 0 and 3003" in high["error"]


@pytest.fixture
def clock_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    lock = Lock(tmp_path / "lock")
    lock.release()
    STATE.lock = lock
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
    assert runner.applied[-1] == ["nvidia-smi", "-lgc", "0,2000"]
    lock.acquire("test")
    status, body = _post(url, {"max_mhz": 1800, "from_ui": True})
    assert status == 403
    assert "Unlock first" in body["error"]
    # A second click while locked still only refuses; nothing new is applied.
    applied = list(runner.applied)
    status, _body = _post(url, {"max_mhz": 1800, "from_ui": True})
    assert status == 403
    assert runner.applied == applied
