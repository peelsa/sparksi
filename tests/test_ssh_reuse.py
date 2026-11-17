# SPDX-License-Identifier: Apache-2.0

"""SSH connection reuse: clients never become masters; one master per Spark, retried at most every 30 s."""

import os
import shutil
import stat
import tempfile
import threading
import time
from pathlib import Path

import pytest

import sparksi.lock
from sparksi import remote
from sparksi.config import CFG


class _FakeThread:
    instances: list["_FakeThread"] = []

    def __init__(self, target=None, args=(), daemon=None, **kwargs):
        self._target = target
        self._args = args
        _FakeThread.instances.append(self)

    def start(self):
        if self._target is not None:
            self._target(*self._args)


@pytest.fixture(autouse=True)
def _reset():
    remote.forget_masters()
    _FakeThread.instances = []
    yield
    remote.forget_masters()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setitem(CFG.controller, "ssh_reuse", True)
    short = Path(tempfile.mkdtemp(prefix="sx", dir="/tmp"))
    monkeypatch.setattr(sparksi.lock, "state_dir", lambda: short)
    monkeypatch.setattr(remote.threading, "Thread", _FakeThread)
    recorded: list[list[str]] = []

    def fake_run(argv, **kwargs):
        recorded.append(list(argv))
        return None

    monkeypatch.setattr(remote.subprocess, "run", fake_run)
    yield short, recorded
    shutil.rmtree(short, ignore_errors=True)


def test_mux_client_opts_and_master_argv(env):
    tmp_path, recorded = env
    argv = remote._ssh_opts("spark-a", 8)
    assert "ControlMaster=no" in argv
    assert f"ControlPath={tmp_path}/ssh/%C" in argv
    assert argv[-2:] == ["--", "spark-a"]
    assert len(recorded) == 1
    master = recorded[0]
    for opt in ("-f", "-N", "ControlMaster=auto", "ControlPersist=120"):
        assert opt in master
    assert master[-2:] == ["--", "spark-a"]
    assert "BatchMode=yes" in master


def test_master_started_once_within_retry_window(env, monkeypatch):
    tmp_path, recorded = env
    remote._ssh_opts("spark-a", 8)
    remote._ssh_opts("spark-a", 8)
    assert len(recorded) == 1
    real_monotonic = remote.time.monotonic
    now = real_monotonic()
    monkeypatch.setattr(remote.time, "monotonic", lambda: now + 31.0)
    remote._ssh_opts("spark-a", 8)
    assert len(recorded) == 2


def test_bad_dir_mode_no_mux(env, monkeypatch):
    tmp_path, recorded = env
    d = tmp_path / "ssh"
    d.mkdir(mode=0o755)
    monkeypatch.setattr(remote.os, "chmod", lambda *a, **k: None)
    argv = remote._ssh_opts("spark-a", 8)
    assert "ControlMaster=no" not in argv
    assert not any(a.startswith("ControlPath=") for a in argv)
    assert recorded == []


def test_ssh_reuse_disabled(env, monkeypatch):
    tmp_path, recorded = env
    monkeypatch.setitem(CFG.controller, "ssh_reuse", False)
    argv = remote._ssh_opts("spark-a", 8)
    assert "ControlMaster=no" not in argv
    assert not any(a.startswith("ControlPath=") for a in argv)
    assert recorded == []


def test_long_path_no_mux(env, monkeypatch):
    tmp_path, recorded = env
    long_parent = tmp_path / ("x" * 70)
    long_parent.mkdir(parents=True)
    monkeypatch.setattr(sparksi.lock, "state_dir", lambda: long_parent)
    argv = remote._ssh_opts("spark-a", 8)
    assert "ControlMaster=no" not in argv
    assert not any(a.startswith("ControlPath=") for a in argv)
    assert recorded == []


def test_master_notices_a_dead_link_fast(env):
    short, recorded = env
    remote._ssh_opts("spark-a", 8)
    master_argv = recorded[0]
    assert "ServerAliveInterval=3" in master_argv
    assert "ServerAliveCountMax=2" in master_argv


def test_owner_check(env, monkeypatch):
    short, recorded = env
    real = remote.os.getuid()
    monkeypatch.setattr(remote.os, "getuid", lambda: real + 1)
    opts = remote._ssh_opts("spark-a", 8)
    assert "ControlMaster=no" not in opts


def test_hung_connection_fails_within_seconds(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> " + str(tmp_path / "calls.log") + "\n"
        'for a in "$@"; do\n'
        '  if [ "$a" = "-N" ] || [ "$a" = "-O" ]; then exit 0; fi\n'
        "done\n"
        "exec sleep 60\n"
    )
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    short = Path(tempfile.mkdtemp(prefix="sx", dir="/tmp"))
    monkeypatch.setattr(sparksi.lock, "state_dir", lambda *a, **k: short)
    monkeypatch.setitem(CFG.controller, "ssh_reuse", True)
    monkeypatch.setattr(remote, "resolve_target", lambda host, catalog=None: "spark-a")
    remote.forget_failures()
    try:
        runner = remote.SshRunner({})
        t0 = time.monotonic()
        out = runner.invoke("spark-a", {"op": "probe"}, timeout=2)
        elapsed = time.monotonic() - t0
        assert elapsed < 4.0
        assert out["ok"] is False

        deadline = time.monotonic() + 2.0
        log = tmp_path / "calls.log"
        while time.monotonic() < deadline:
            if log.exists() and "-O stop" in log.read_text():
                break
            time.sleep(0.05)

        assert "-O stop" in log.read_text()  # the hung master takes no new calls
        t1 = time.monotonic()
        out2 = runner.invoke("spark-a", {"op": "probe"}, timeout=2)
        assert time.monotonic() - t1 < 0.5
        assert out2.get("unreachable") is True
    finally:
        remote.forget_failures()
        shutil.rmtree(short, ignore_errors=True)


def test_a_timed_out_call_does_not_cut_others_on_that_node(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    mark = str(bin_dir / "ssh")
    ssh = bin_dir / "ssh"
    # A fake ssh that behaves like a master: "-O exit" kills every client riding
    # it, "-O stop" lets them finish.
    ssh.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> " + str(tmp_path / "calls.log") + "\n"
        'op=""; prev=""\n'
        'for a in "$@"; do\n'
        '  if [ "$a" = "-N" ]; then exit 0; fi\n'
        '  if [ "$prev" = "-O" ]; then op="$a"; fi\n'
        '  prev="$a"\n'
        "done\n"
        'if [ "$op" = "exit" ]; then pkill -f "' + mark + ' -o BatchMode"; exit 0; fi\n'
        'if [ "$op" = "stop" ]; then exit 0; fi\n'
        "DATA=$(cat)\n"
        'case "$DATA" in\n'
        "  *hang-me*) exec sleep 60 ;;\n"
        "esac\n"
        "sleep 2 &\n"
        "wait $!\n"
        'printf \'%s\\n\' \'{"ok": true, "answer": "fine"}\'\n'
    )
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    short = Path(tempfile.mkdtemp(prefix="sx", dir="/tmp"))
    monkeypatch.setattr(sparksi.lock, "state_dir", lambda *a, **k: short)
    monkeypatch.setitem(CFG.controller, "ssh_reuse", True)
    monkeypatch.setattr(remote, "resolve_target", lambda host, catalog=None: "spark-a")
    remote.forget_failures()
    runner = remote.SshRunner({})

    results = {}

    def run_a():
        t0 = time.monotonic()
        res = runner.invoke("spark-a", {"op": "probe", "ports": [], "why": "ok"}, timeout=8)
        results["a"] = (res, time.monotonic() - t0)

    thread_a = threading.Thread(target=run_a)
    thread_a.start()
    time.sleep(0.3)

    ok_b = runner.invoke("spark-a", {"op": "probe", "why": "hang-me"}, timeout=1)
    assert ok_b["ok"] is False

    thread_a.join(timeout=10)
    assert not thread_a.is_alive()

    res_a, dur_a = results["a"]
    assert res_a["ok"] is True
    assert res_a["answer"] == "fine"
    # A's answer clears the down mark B's timeout set: other reads are not skipped.
    assert remote._recent_failure("spark-a") is None

    log = (tmp_path / "calls.log").read_text()
    assert any("-O stop" in line for line in log.splitlines())
    assert not any("-O exit" in line for line in log.splitlines())

    remote.forget_failures()
    shutil.rmtree(short, ignore_errors=True)
