# SPDX-License-Identifier: Apache-2.0

from sparksi.catalog import load_catalog
from sparksi.lock import Lock
from sparksi.serve import STATE
from sparksi.stop import stop_id


def test_stop_unknown_id() -> None:
    import pytest
    from sparksi.catalog import CatalogError

    with pytest.raises(CatalogError):
        stop_id("not-a-model", load_catalog())


def test_stop_uses_runner(monkeypatch, video_catalog) -> None:
    cat = load_catalog()
    seen = []

    def runner(rid):
        seen.append(rid)
        return True, "ok"

    out = stop_id(video_catalog, cat, runner=runner)
    assert out["stopped"] is True
    assert seen == [video_catalog]


def test_stop_api_requires_unlock(tmp_path, monkeypatch) -> None:
    import json
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from sparksi.serve import DualServer, Handler

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    # Another test can leave a switch job in flight; that answers 409, not 403.
    monkeypatch.setattr("sparksi.serve.load_in_progress", lambda: False)
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        req = Request(
            f"http://127.0.0.1:{httpd.server_address[1]}/api/stop",
            data=json.dumps({"id": "comfy", "from_ui": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(req, timeout=10)
            raise AssertionError("expected 403")
        except HTTPError as e:
            assert e.code == 403
    finally:
        httpd.shutdown()
