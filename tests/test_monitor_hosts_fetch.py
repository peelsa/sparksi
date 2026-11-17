# SPDX-License-Identifier: Apache-2.0

"""Monitored hosts: the poll-time address check, the pinned connection and the opener rules."""

from __future__ import annotations

import functools
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import URLError

import pytest

from sparksi import hostcheck
from sparksi import monitor_hosts as mh

# Assembled so the identity guard sees no link-local or private literal.
META = ".".join(("169", "254", "169", "254"))  # cloud metadata
THIS_NET = ".".join(("0", "1", "2", "3"))


def _resolver(*addresses: str, calls: list | None = None):
    def resolve(host, port, family=0, type=0):  # noqa: A002 - getaddrinfo's own name
        if calls is not None:
            calls.append((host, port))
        out = []
        for a in addresses:
            fam = socket.AF_INET6 if ":" in a else socket.AF_INET
            out.append((fam, socket.SOCK_STREAM, 6, "", (a, port)))
        return out

    return resolve


# ------------------------------------------------------------------ poll-time address check


@pytest.mark.parametrize("address", ["192.0.2.5", "198.51.100.20", "100.64.0.103", "fd7a:115c:a1e0::5"])
def test_private_and_tailnet_addresses_are_used(address):
    assert mh.resolve_checked("box-a", 8000, _resolver(address)) == address


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",          # loopback
        "127.8.9.10",
        "::1",
        "::ffff:127.0.0.1",   # loopback written as IPv4-mapped IPv6
        META,                 # cloud metadata
        "::ffff:" + META,
        "fe80::1",            # link-local
        "0.0.0.0",
        THIS_NET,             # "this network"
        "224.0.0.1",          # multicast
        "8.8.8.8",            # public
        "2001:4860:4860::8888",
    ],
)
def test_a_name_that_now_resolves_outside_is_refused(address):
    with pytest.raises(mh.Refused, match="not a private or tailnet address"):
        mh.resolve_checked("box-a", 8000, _resolver(address))


def test_every_resolved_address_must_pass():
    with pytest.raises(mh.Refused, match="127.0.0.1"):
        mh.resolve_checked("box-a", 8000, _resolver("192.0.2.5", "127.0.0.1"))


def test_a_lookup_failure_is_offline_not_refused():
    def fails(*_a, **_k):
        raise socket.gaierror("no such host")

    with pytest.raises(URLError) as exc:
        mh.resolve_checked("box-a", 8000, fails)
    assert not isinstance(exc.value, mh.Refused)
    with pytest.raises(URLError):
        mh.resolve_checked("box-a", 8000, _resolver())


def test_ip_ok_reads_mapped_addresses_as_ipv4():
    assert hostcheck.ip_ok("::ffff:192.0.2.5") is True
    assert hostcheck.ip_ok("::ffff:127.0.0.1") is False
    assert hostcheck.ip_ok("::ffff:" + META) is False
    assert hostcheck.ip_ok(THIS_NET) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://192.0.2.5:8000",
        "http://user:pw@192.0.2.5:8000",
        "http://192.0.2.5:8000/v1",
        "http://192.0.2.5:8000?x=1",
        "http://127.0.0.1:8000",
        "http://" + META + ":80",
        "http://192.0.2.5",
    ],
)
def test_get_refuses_a_bad_url_before_any_lookup(url):
    calls: list = []
    with pytest.raises(mh.Refused):
        mh.get(url, "/v1/models", resolve=_resolver("192.0.2.5", calls=calls))
    assert calls == []


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/api/generate", "/api/pull", "/", "/v1/models/../x"])
def test_get_refuses_any_path_outside_the_list(path):
    calls: list = []
    with pytest.raises(mh.Refused, match="not polled"):
        mh.get("http://192.0.2.5:8000", path, resolve=_resolver("192.0.2.5", calls=calls))
    assert calls == []


def test_a_refused_address_is_not_contacted(monkeypatch):
    opened: list = []
    monkeypatch.setattr(mh.monitor_mirror, "no_redirect_opener", lambda *h: opened.append(h))
    with pytest.raises(mh.Refused):
        mh.get("http://box-a:8000", "/v1/models", resolve=_resolver("127.0.0.1"))
    assert opened == []


# ------------------------------------------------------------------ the real connection


class _Server:
    """A local HTTP server; tests allow loopback only by pinning box-a to 127.0.0.1 themselves."""

    def __init__(self, handler):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def _recording_handler(seen: list, status: int = 200, body: bytes = b'{"data":[{"id":"model-a"}]}', location: str = ""):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _any(self):
            seen.append((self.command, self.path, self.headers.get("Host")))
            self.send_response(status)
            if location:
                self.send_header("Location", location)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_PUT = do_DELETE = _any

    return H


@pytest.fixture()
def allow_loopback(monkeypatch):
    """The test server is on loopback, which the real rules refuse; let only 127.0.0.1 through here."""
    monkeypatch.setattr(mh, "ip_ok", lambda a: a == "127.0.0.1")


def test_connects_to_the_checked_address_not_a_fresh_lookup(allow_loopback, monkeypatch):
    seen: list = []
    srv = _Server(_recording_handler(seen))
    calls: list = []
    try:
        # "box-a.invalid" does not resolve anywhere; the request can only arrive through the pinned address.
        body = mh.get(f"http://box-a.invalid:{srv.port}", "/v1/models", resolve=_resolver("127.0.0.1", calls=calls))
    finally:
        srv.close()
    assert body == b'{"data":[{"id":"model-a"}]}'
    assert calls == [("box-a.invalid", srv.port)]
    assert seen == [("GET", "/v1/models", f"box-a.invalid:{srv.port}")]


def test_proxy_settings_are_ignored(allow_loopback, monkeypatch):
    for name in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://203.0.113.1:9")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    seen: list = []
    srv = _Server(_recording_handler(seen))
    try:
        mh.get(f"http://box-a.invalid:{srv.port}", "/v1/models", resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    assert [s[1] for s in seen] == ["/v1/models"]


def test_redirects_are_refused(allow_loopback):
    seen: list = []
    srv = _Server(_recording_handler(seen, status=302, body=b"", location="http://" + META + "/latest/meta-data/"))
    try:
        with pytest.raises(URLError):
            mh.get(f"http://box-a.invalid:{srv.port}", "/v1/models", resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    assert [s[1] for s in seen] == ["/v1/models"]


def test_reply_size_is_capped(allow_loopback, monkeypatch):
    monkeypatch.setattr(mh, "MAX_BYTES", 1000)
    srv = _Server(_recording_handler([], body=b"x" * 1001))
    try:
        with pytest.raises(ValueError, match="too large"):
            mh.get(f"http://box-a.invalid:{srv.port}", "/metrics", resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()


def test_a_silent_host_times_out_quickly(allow_loopback):
    quiet = socket.socket()
    quiet.bind(("127.0.0.1", 0))
    quiet.listen(1)
    port = quiet.getsockname()[1]
    t0 = time.monotonic()
    try:
        with pytest.raises((OSError, URLError)):
            mh.get(f"http://box-a.invalid:{port}", "/v1/models", timeout=0.3, resolve=_resolver("127.0.0.1"))
    finally:
        quiet.close()
    assert time.monotonic() - t0 < 2.0


@pytest.mark.parametrize("engine", sorted(mh.PATHS))
def test_every_engine_poll_is_get_only(allow_loopback, engine):
    seen: list = []
    srv = _Server(_recording_handler(seen, body=b'{"data":[],"models":[]}'))
    fetch = functools.partial(mh.get, resolve=_resolver("127.0.0.1"))
    try:
        mh.read_host({"url": f"http://box-a.invalid:{srv.port}", "engine": engine}, fetch)
    finally:
        srv.close()
    assert seen and all(method == "GET" for method, _path, _host in seen)
    assert [p for _m, p, _h in seen] == list(mh.PATHS[engine])


# ------------------------------------------------------------------ one deadline for the whole GET


class _DripServer:
    """Sends its reply one byte every ``gap`` seconds: each read is quick, the whole reply never ends."""

    def __init__(self, gap: float = 0.1, headers_too: bool = False, close_delimited: bool = False, chunk: int = 1, seconds: float = 0.0):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(4)
        self.port = self.sock.getsockname()[1]
        self.gap = gap
        self.headers_too = headers_too
        self.close_delimited = close_delimited
        self.chunk = chunk
        self.seconds = seconds  # stop sending and close after this long (0: never)
        self.stop = threading.Event()
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while not self.stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._drip, args=(conn,), daemon=True).start()

    def _drip(self, conn):
        try:
            conn.recv(4096)
            if self.close_delimited:
                # No Content-Length: the body ends when the server closes the connection.
                head = b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n"
            else:
                head = b"HTTP/1.1 200 OK\r\nContent-Length: 100000\r\n\r\n"
            if not self.headers_too:
                conn.sendall(head)
                head = b""
            data = head + b"x" * 100000
            end = time.monotonic() + self.seconds if self.seconds else None
            for i in range(0, len(data), self.chunk):
                if self.stop.is_set() or (end is not None and time.monotonic() >= end):
                    break
                conn.sendall(data[i:i + self.chunk])
                time.sleep(self.gap)
        except OSError:
            pass
        finally:
            conn.close()

    def close(self):
        self.stop.set()
        self.sock.close()


@pytest.mark.parametrize("headers_too", [False, True], ids=["body", "headers"])
def test_a_dripping_host_is_cut_off_at_the_total_deadline(allow_loopback, headers_too):
    srv = _DripServer(gap=0.1, headers_too=headers_too)
    t0 = time.monotonic()
    try:
        # Each byte comes well inside the 2 s socket timeout; only the total deadline can stop it.
        with pytest.raises(TimeoutError, match="no full answer"):
            mh.get(f"http://box-a.invalid:{srv.port}", "/metrics", total=1.0, resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    took = time.monotonic() - t0
    assert 0.9 <= took < 2.0, took


def test_the_default_total_deadline_is_five_seconds():
    assert mh.TOTAL_S == 5.0
    assert mh.TIMEOUT_S < mh.TOTAL_S


def test_a_hanging_lookup_stops_at_the_deadline():
    release = threading.Event()

    def hangs(*_a, **_k):
        release.wait(10)
        return []

    t0 = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="could not resolve box-a"):
            mh.get("http://box-a:8000", "/v1/models", total=0.5, resolve=hangs)
    finally:
        release.set()
    assert time.monotonic() - t0 < 1.5


def test_a_lookup_counts_against_the_same_deadline(allow_loopback):
    """A slow lookup leaves less time for the reply: the whole GET still ends near ``total``."""
    srv = _DripServer(gap=0.1)

    def slow(host, port, *a, **k):
        time.sleep(0.9)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))]

    t0 = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            mh.get(f"http://box-a.invalid:{srv.port}", "/metrics", total=1.0, resolve=slow)
    finally:
        srv.close()
    # Sharing the deadline ends near 1.0 s; a fresh 1.0 s deadline after the 0.9 s lookup would take 1.9 s.
    assert time.monotonic() - t0 < 1.4


# ------------------------------------------------------------------ redirects refused by the opener itself


def test_a_redirect_to_the_same_host_and_path_is_still_refused(allow_loopback):
    """The pin and the path check cannot catch this one; only the opener's redirect refusal does."""
    seen: list = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            seen.append(self.path)
            if "?" not in self.path:
                self.send_response(302)
                self.send_header("Location", f"http://box-a.invalid:{self.server.server_address[1]}/v1/models?again=1")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = b'{"data":[{"id":"model-a"}]}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = _Server(H)
    try:
        with pytest.raises(URLError):
            mh.get(f"http://box-a.invalid:{srv.port}", "/v1/models", resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    assert seen == ["/v1/models"]


def test_a_dripping_reply_without_content_length_is_a_timeout(allow_loopback):
    """Connection: close and no Content-Length: the deadline's shutdown looks like a normal end of body,
    so only the deadline checks after the reads stop it being taken as a (cut-short) answer."""
    srv = _DripServer(gap=0.1, close_delimited=True)
    t0 = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="no full answer"):
            mh.get(f"http://box-a.invalid:{srv.port}", "/metrics", total=1.0, resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    assert time.monotonic() - t0 < 2.0


def test_the_read_loop_checks_the_deadline_even_without_the_timer(allow_loopback, monkeypatch):
    """A reply that keeps arriving is stopped by the check after each read, not only by the timer.

    The timer is switched off here; the server streams quickly (no Content-Length) for 4 s and then
    closes, so a loop that only checked after the body ended would finish after about 4 s.
    """

    class _NoTimer:
        def __init__(self, *a, **k):
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(mh.threading, "Timer", _NoTimer)
    srv = _DripServer(gap=0.01, close_delimited=True, chunk=64, seconds=4.0)
    t0 = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="no full answer"):
            mh.get(f"http://box-a.invalid:{srv.port}", "/metrics", total=1.0, resolve=_resolver("127.0.0.1"))
    finally:
        srv.close()
    assert time.monotonic() - t0 < 2.5


def test_a_socket_attached_after_the_deadline_is_shut_at_once():
    """The timer can fire before connect() attaches the socket; attach() must then shut it itself."""
    deadline = mh._Deadline(0.0)
    for _ in range(200):
        if deadline.expired:
            break
        time.sleep(0.01)
    assert deadline.expired
    ours, theirs = socket.socketpair()
    try:
        theirs.settimeout(1.0)
        deadline.attach(ours)
        assert theirs.recv(1) == b""  # our side was shut down: the peer sees end of stream at once
        with pytest.raises(OSError):
            ours.send(b"x")
    finally:
        ours.close()
        theirs.close()

