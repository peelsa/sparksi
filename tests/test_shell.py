# SPDX-License-Identifier: Apache-2.0

"""History ring, activity log, and the fleet page routes."""

import http.client
import json
import threading
import time
from urllib.error import HTTPError
from urllib.request import urlopen

from sparksi.activity import Activity
from sparksi.history import History, sample_card
from sparksi.lock import Lock
from sparksi.serve import INDEX_HTML, STATE, DualServer, Handler


def test_overview_chart_keeps_a_twenty_minute_window() -> None:
    text = INDEX_HTML.read_text(encoding="utf-8")
    assert "var CHART_WINDOW = 1200;" in text
    assert "covered / 0.75" not in text
    assert "0.75 * span" not in text
    assert "((Number(now) - Number(t)) / CHART_WINDOW)" in text
    assert 'range=1h' in text
    assert "min-height: 280px" in text
    assert "grid-template-columns: max-content minmax(0, 1fr)" in text
    assert "@media (max-width: 800px)" in text


def test_overview_temperature_is_fahrenheit() -> None:
    text = INDEX_HTML.read_text(encoding="utf-8")
    assert "return (Number(celsius) * 9 / 5) + 32;" in text
    assert '["0 F", "50 F", "100 F", "150 F", "200 F"]' in text
    assert "80 F" not in text
    assert "110 F" not in text
    assert "140 F" not in text
    assert "170 F" not in text
    assert "legend-machine" in text and "#overview-chart .legend-swatch" in text
    assert "grid-template-columns: max-content minmax(0, 1fr)" in text
    assert "max-width: 960px" not in text
    assert (0 * 9 / 5) + 32 == 32
    assert (100 * 9 / 5) + 32 == 212


def test_history_keeps_the_first_sample_in_a_bucket(tmp_path) -> None:
    hist = History(tmp_path / "history.json", now=lambda: 1_700_000_011)
    assert hist.observe("a", {"gpu_pct": 10, "cpu_pct": 3}, now=1_700_000_010) is True
    assert hist.observe("a", {"gpu_pct": 99}, now=1_700_000_010.4) is False
    assert hist.observe("a", {"gpu_pct": 11}, now=1_700_000_011) is True
    samples = hist.query("a", "1h", now=1_700_000_011)["samples"]
    assert len(samples) == 2
    assert samples[0]["gpu_pct"] == 10
    assert "gpu_temp_c" not in samples[0]


def test_history_drops_samples_older_than_a_day(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    assert hist.observe("a", {"cpu_pct": 1}, now=1_000) is True
    assert hist.observe("a", {"cpu_pct": 2}, now=1_000 + 24 * 3600 + 120) is True
    samples = hist.query("a", "24h", now=1_000 + 24 * 3600 + 120)["samples"]
    assert [row["cpu_pct"] for row in samples] == [2]


def test_history_query_does_not_invent_a_gap_zero(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    hist.observe("a", {"gpu_pct": 4}, now=10_000)
    hist.observe("a", {"gpu_pct": 8}, now=10_180)
    samples = hist.query("a", "1h", now=10_200)["samples"]
    assert [row["t"] for row in samples] == [10_000, 10_180]
    assert all(row["gpu_pct"] != 0 for row in samples)
    assert len(samples) == 2


def test_stale_host_is_not_written_as_zero(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    card = {
        "hosts": [{"id": "a", "fresh": False, "gpu_pct": 0, "gpu_temp_c": 0}],
        "loaded": [],
    }
    sample_card(card, hist, None, now=50_000)
    assert hist.query("a", "1h", now=50_000)["samples"] == []


def test_activity_caps_at_500(tmp_path) -> None:
    log = Activity(tmp_path / "activity.json", now=lambda: 10)
    for i in range(501):
        log.append("lock", f"Lock on {i}")
    events = log.events()
    assert len(events) == 500
    assert events[0]["message"] == "Lock on 500"
    assert events[-1]["message"] == "Lock on 1"


def test_gpu_temperature_does_not_flap(tmp_path) -> None:
    log = Activity(tmp_path / "activity.json", now=lambda: 10)
    log.note_temp("a", 79)
    log.note_temp("a", 80)
    log.note_temp("a", 81)
    log.note_temp("a", 75)
    log.note_temp("a", 74)
    log.note_temp("a", 80)
    kinds = [row["kind"] for row in reversed(log.events())]
    assert kinds == ["gpu-hot", "gpu-cool", "gpu-hot"]


def test_first_stale_announce_is_not_a_restore(tmp_path) -> None:
    log = Activity(tmp_path / "activity.json", now=lambda: 10)
    log.note_announce("a", False)
    log.note_announce("a", True)
    assert log.events() == []


def test_announce_loss_is_a_transition(tmp_path) -> None:
    log = Activity(tmp_path / "activity.json", now=lambda: 10)
    log.note_announce("a", True)
    log.note_announce("a", True)
    log.note_announce("a", False)
    log.note_announce("a", True)
    kinds = [row["kind"] for row in reversed(log.events())]
    assert kinds == ["announce-lost", "announce-restored"]


def _serve(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    from sparksi.activity import reset_for_tests as reset_activity
    from sparksi.history import reset_for_tests as reset_history

    reset_activity()
    reset_history()
    lk = Lock(tmp_path / "lock")
    lk.acquire(reason="test")
    (tmp_path / "unlock.token").write_text("PLANTED-UNLOCK-TOKEN-DO-NOT-LEAK\n")
    STATE.lock = lk
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, httpd.server_address[1]


def test_shell_routes(tmp_path, monkeypatch) -> None:
    httpd, port = _serve(tmp_path, monkeypatch)
    try:
        History(tmp_path / "history.json").observe("a", {"gpu_pct": 12}, now=time.time())
        from sparksi.history import reset_for_tests

        reset_for_tests()
        planted = "PLANTED-UNLOCK-TOKEN-DO-NOT-LEAK"
        pages = ("/overview", "/tokens", "/activity", "/library", "/archive")
        for legacy in ("/legacy", "/machine/a"):
            with urlopen(f"http://127.0.0.1:{port}{legacy}", timeout=15) as resp:
                assert resp.geturl().rstrip("/").endswith("/overview")
        # The Temps page moved into Overview; the old address points there, section open.
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
        conn.request("GET", "/temps")
        moved = conn.getresponse()
        moved.read()
        conn.close()
        assert moved.status == 302
        assert moved.getheader("Location") == "/overview#temps"
        for path in pages:
            with urlopen(f"http://127.0.0.1:{port}{path}", timeout=15) as resp:
                ctype = resp.headers.get("Content-Type", "")
                body = resp.read().decode()
            assert "html" in ctype
            assert body.lstrip().lower().startswith("<!doctype html>")
            assert "100%" in body
            assert "9 / 5" in body
            assert "stroke-dasharray" in body
            assert planted not in body
        with urlopen(f"http://127.0.0.1:{port}/api/history?machine=a&range=24h", timeout=15) as resp:
            payload = json.loads(resp.read().decode())
        assert payload["machine"] == "a"
        assert payload["samples"][0]["gpu_pct"] == 12
        with urlopen(f"http://127.0.0.1:{port}/api/activity", timeout=15) as resp:
            activity = json.loads(resp.read().decode())
        blob = json.dumps(activity)
        assert "events" in activity
        assert planted not in blob
        assert "unlock_token" not in blob
        try:
            urlopen(f"http://127.0.0.1:{port}/api/history?range=nope", timeout=15)
            raise AssertionError("bad range should be refused")
        except HTTPError as exc:
            assert exc.code == 400
    finally:
        httpd.shutdown()
