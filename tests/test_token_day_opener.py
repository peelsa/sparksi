# SPDX-License-Identifier: Apache-2.0

"""token_day reads metrics_url with the no-proxy, no-redirect opener, never plain urlopen."""

from datetime import datetime

import sparksi.token_day as token_day
import sparksi.monitor_mirror as monitor_mirror


class FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b'{"status":"success","data":{"result":[]}}'


class FakeOpener:
    def __init__(self):
        self.calls = []

    def open(self, req, timeout=None):
        self.calls.append((req, timeout))
        return FakeResp()


def test_query_vector_uses_no_redirect_opener(monkeypatch):
    fake = FakeOpener()
    monkeypatch.setattr(monitor_mirror, "no_redirect_opener", lambda: fake)

    def plain(*a, **k):
        raise AssertionError("plain urlopen used")

    monkeypatch.setattr("urllib.request.urlopen", plain)
    result = token_day.query_vector("up", datetime.now().astimezone())
    assert result == []
    assert len(fake.calls) == 1


def test_health_check_uses_no_redirect_opener(monkeypatch):
    import urllib.error

    seen = []

    class Down:
        def open(self, req, timeout=None):
            seen.append(req)
            raise urllib.error.URLError("down")

    monkeypatch.setattr(monitor_mirror, "no_redirect_opener", lambda: Down())

    def plain(*a, **k):
        raise AssertionError("plain urlopen used")

    monkeypatch.setattr("urllib.request.urlopen", plain)
    report = token_day.build_report(datetime.now().astimezone())
    assert report["ok"] is False
    assert "not answering" in report["error"]
    assert seen and str(seen[0]).endswith("/health")
