# SPDX-License-Identifier: Apache-2.0

"""Every request gets an answer, and a page load names its own view."""

from __future__ import annotations

import http.client
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import sparksi.serve as serve_mod
from sparksi.hosts import Allowlist
from sparksi.lock import Lock
from sparksi.serve import _PAGES, STATE, DualServer, Handler, page_html


@pytest.fixture()
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(serve_mod, "HOSTS", Allowlist(lambda: [], ttl=300))
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


def _get(port: int, path: str, method: str = "GET") -> tuple[int, bytes]:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    try:
        conn.request(method, path, headers={"Host": f"127.0.0.1:{port}"})
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _visible_views(text: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'<div id="view-([a-z]+)" class="view">', text)]


def test_each_page_is_served_with_its_own_view_shown() -> None:
    for path in sorted(_PAGES):
        want = "overview" if path in ("/", "/overview") else path.strip("/")
        assert _visible_views(page_html(path)) == [want], path
    assert _visible_views(page_html("/nope")) == ["overview"]


def test_an_unexpected_error_is_a_500_not_a_closed_socket(server, monkeypatch) -> None:
    def boom(path="/"):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(serve_mod, "page_html", boom)
    for method in ("GET", "HEAD"):
        code, body = _get(server, "/models", method)
        assert code == 500, method
        if method == "GET":
            assert json.loads(body) == {"error": "internal error"}
    code, _ = _get(server, "/api/config")
    assert code == 200  # the server keeps answering


def test_a_burst_of_polls_all_get_answers(server) -> None:
    """The page polls about ten endpoints at once; a burst well past the old backlog of 5."""
    assert DualServer.request_queue_size >= 64 and Handler.timeout
    paths = ["/api/config", "/api/activity", "/models", "/network"] * 20
    with ThreadPoolExecutor(max_workers=80) as pool:
        codes = list(pool.map(lambda p: _get(server, p)[0], paths))
    assert codes.count(200) == len(paths), codes
