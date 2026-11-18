# SPDX-License-Identifier: Apache-2.0
"""The controller journal must not print tracebacks when syslog is dead.

A ``/dev/log`` that points at a socket nobody listens on (journald stopped
or restarting) made every record print ``--- Logging error ---``. These
tests point the journal at a temp unix socket instead of the real one.
"""

import logging
import logging.handlers
import os
import socket
import tempfile

import pytest

from sparksi import log as journal


@pytest.fixture
def syslog_path(monkeypatch):
    """A fresh ``sparksi`` logger whose syslog address is a temp socket path."""
    tmp = tempfile.mkdtemp(prefix="sparksi-log-", dir="/tmp")  # short: unix socket paths cap near 108 bytes
    path = os.path.join(tmp, "dev-log")
    monkeypatch.setattr(journal, "_SYSLOG_ADDRESS", path, raising=False)
    real_init = logging.handlers.SysLogHandler.__init__

    def init(self, address=("localhost", logging.handlers.SYSLOG_UDP_PORT), *args, **kwargs):
        real_init(self, path if address == "/dev/log" else address, *args, **kwargs)

    monkeypatch.setattr(logging.handlers.SysLogHandler, "__init__", init)
    monkeypatch.setattr(journal, "_READY", False)
    log = logging.getLogger("sparksi")
    saved, saved_level = log.handlers[:], log.level
    log.handlers = []
    yield path
    for handler in log.handlers:
        handler.close()
    log.handlers = saved
    log.setLevel(saved_level)
    if os.path.exists(path):
        os.unlink(path)
    os.rmdir(tmp)


def _server(path):
    server = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    server.bind(path)
    server.settimeout(2)
    return server


def _syslog_handlers(log):
    return [h for h in log.handlers if isinstance(h, logging.handlers.SysLogHandler)]


@pytest.mark.parametrize("state", ["dead", "missing"])
def test_dead_or_missing_syslog_is_skipped_without_tracebacks(syslog_path, capsys, state):
    if state == "dead":
        _server(syslog_path).close()  # the socket file stays, nobody listens
    log = journal.logger()
    log.info("hello %s", state)
    log.info("again %s", state)
    err = capsys.readouterr().err
    assert "Logging error" not in err
    assert "Traceback" not in err
    assert f"INFO sparksi hello {state}" in err
    assert f"INFO sparksi again {state}" in err
    assert _syslog_handlers(log) == []


def test_live_syslog_gets_records(syslog_path, capsys):
    server = _server(syslog_path)
    try:
        log = journal.logger()
        assert len(_syslog_handlers(log)) == 1
        log.info("to syslog")
        assert b"sparksi INFO to syslog" in server.recv(4096)
    finally:
        server.close()
    assert "INFO sparksi to syslog" in capsys.readouterr().err


def test_syslog_that_dies_later_drops_quietly_and_reconnects(syslog_path, capsys):
    server = _server(syslog_path)
    log = journal.logger()
    log.info("before")
    assert b"before" in server.recv(4096)
    server.close()
    os.unlink(syslog_path)
    for n in range(5):
        log.info("while down %d", n)
    err = capsys.readouterr().err
    assert "Logging error" not in err
    assert "Traceback" not in err
    assert "INFO sparksi while down 4" in err
    restarted = _server(syslog_path)
    try:
        log.info("after restart")
        assert b"sparksi INFO after restart" in restarted.recv(4096)
    finally:
        restarted.close()
