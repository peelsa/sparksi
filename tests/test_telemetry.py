# SPDX-License-Identifier: Apache-2.0

"""Durable telemetry stays on disk after the one-hour chart cache drops it."""

import errno
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from sparksi.history import History, sample_card
from sparksi.serve import STATE, Handler
from sparksi.serve import DualServer
from sparksi.lock import Lock


def test_archive_outlives_the_hot_ring_and_a_restart(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("machine-1", {"cpu_pct": 1, "gpu_pct": 4}, now=1_700_000_000, role="head")
    hist.observe("machine-1", {"cpu_pct": 2}, now=1_700_000_000 + 48 * 3600, role="head")
    hot = hist.query("machine-1", "1h", now=1_700_000_000 + 48 * 3600)["samples"]
    assert [row["cpu_pct"] for row in hot] == [2]
    old = hist.query("machine-1", "24h", now=1_700_000_000 + 3600, start=1_700_000_000 - 10, end=1_700_000_000 + 10)
    assert old["samples"] == [{"t": 1_700_000_000, "cpu_pct": 1, "gpu_pct": 4}]
    blob = "\n".join(item.read_text(encoding="utf-8") for item in (tmp_path / "telemetry" / "head").glob("*.jsonl"))
    assert '"role":"head"' in blob
    assert "machine-1" not in blob
    assert "gpu_temp_c" not in blob.splitlines()[0]
    again = History(path)
    restored = again.query("head", "24h", now=1_700_000_000 + 3600, start=1_700_000_000 - 10, end=1_700_000_000 + 10)
    assert restored["samples"][0]["cpu_pct"] == 1
    assert not any("machine-1" in path.name for path in (tmp_path / "telemetry").rglob("*"))


def test_wide_query_downsamples_without_filling_gaps(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    start = 1_800_000_000
    hist.observe("head", {"gpu_pct": 10}, now=start, role="head")
    hist.observe("head", {"gpu_pct": 30}, now=start + 10 * 3600, role="head")
    # A day of one sample every 30s is larger than the raw cap and stays bounded.
    for offset in range(0, 24 * 3600, 60):
        hist.observe("worker", {"gpu_pct": 0 if offset % 600 else 50}, now=start + offset, role="worker")
    wide = hist.query("worker", "24h", now=start + 24 * 3600)
    assert wide["downsampled"] is True
    assert 1 < len(wide["samples"]) < 1440
    assert len(wide["samples"]) <= 1300
    assert all("gpu_pct" in row for row in wide["samples"])
    assert max(row["gpu_pct"] for row in wide["samples"]) > 0
    gap = hist.query("head", "24h", now=start + 24 * 3600, start=start - 5, end=start + 11 * 3600)
    stamps = [row["t"] for row in gap["samples"]]
    assert stamps[0] == start
    assert stamps[-1] == start + 10 * 3600
    assert not any(stamps[i + 1] - stamps[i] < 3600 for i in range(len(stamps) - 1))


def test_partial_line_is_skipped_and_write_errors_are_visible(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    hist.observe("head", {"mem_pct": 25}, now=5_000, role="head")
    folder = next((tmp_path / "telemetry" / "head").glob("*.jsonl"))
    with folder.open("a", encoding="utf-8") as handle:
        handle.write('{"t":')
    rows = hist.query("head", "1h", now=5_100)["samples"]
    assert rows == [{"t": 5_000, "mem_pct": 25}]
    blocked = tmp_path / "blocked" / "history.json"
    (tmp_path / "blocked").mkdir()
    (tmp_path / "blocked" / "telemetry").write_text("not a directory", encoding="utf-8")
    broken = History(blocked)
    assert broken.observe("head", {"cpu_pct": 3}, now=8_000, role="head") is True
    assert broken.query("head", "1h", now=8_000)["samples"][0]["cpu_pct"] == 3
    described = broken.telemetry_index()
    assert described["last_error"]
    assert "home" not in described["last_error"]
    assert described["kept"] == "indefinitely"
    assert described["schema"]["missing"].startswith("a key that is absent")


def test_reads_do_not_write_and_callers_can_read_while_collecting(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    hist.observe("head", {"gpu_pct": 1}, now=9_000, role="head")
    target = next((tmp_path / "telemetry" / "head").glob("*.jsonl"))
    tip = target.with_name(target.stem + ".tip.json")
    before = (target.stat().st_mtime_ns, tip.stat().st_mtime_ns)
    hist.query("head", "1h", now=9_100)
    hist.telemetry_index()
    after = (target.stat().st_mtime_ns, tip.stat().st_mtime_ns)
    assert after == before
    errors = []

    def collect() -> None:
        for i in range(40):
            try:
                hist.observe("worker", {"cpu_pct": i % 7}, now=20_000 + i, role="worker")
            except Exception as exc:  # noqa: BLE001 — the test records a crash
                errors.append(exc)

    thread = threading.Thread(target=collect)
    thread.start()
    for _ in range(20):
        body = hist.query("worker", "1h", now=20_040, start=20_000, end=20_040)
        assert isinstance(body["samples"], list)
        listed = hist.telemetry_index()
        assert "files" in listed
    thread.join()
    assert errors == []
    assert hist.query("worker", "1h", now=20_040)["samples"][-1]["cpu_pct"] == 39 % 7


def test_sample_card_archives_a_fresh_role_only(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    sample_card(
        {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 12, "cpu_pct": None}], "loaded": []},
        hist,
        None,
        now=30_000,
    )
    sample_card(
        {"hosts": [{"id": "head", "fresh": False, "gpu_pct": 0}], "loaded": []},
        hist,
        None,
        now=30_010,
    )
    rows = hist.query("head", "1h", now=30_020)["samples"]
    assert rows == [{"t": 30_000, "gpu_pct": 12}]
    blob = next((tmp_path / "telemetry" / "head").glob("*.jsonl")).read_text(encoding="utf-8")
    assert "cpu_pct" not in blob
    assert '"gpu_pct":12' in blob or '"gpu_pct": 12' in blob


def test_telemetry_route_is_read_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    from sparksi.history import reset_for_tests

    reset_for_tests()
    lock = Lock(tmp_path / "lock")
    lock.acquire(reason="test")
    STATE.lock = lock
    History(tmp_path / "history.json").observe("head", {"gpu_pct": 8}, now=time.time(), role="head")
    reset_for_tests()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        target = next((tmp_path / "telemetry" / "head").glob("*.jsonl"))
        before = target.stat().st_mtime_ns
        with urlopen(f"http://127.0.0.1:{port}/api/telemetry", timeout=15) as resp:
            body = json.loads(resp.read().decode())
        with urlopen(f"http://127.0.0.1:{port}/api/history?machine=head&range=1h", timeout=15) as resp:
            history = json.loads(resp.read().decode())
        assert target.stat().st_mtime_ns == before
        assert body["files"]
        assert body["files"][0]["role"] == "head"
        assert body["files"][0]["path"].startswith("telemetry/head/")
        assert "schema" in body
        assert history["samples"][0]["gpu_pct"] == 8
        try:
            urlopen(f"http://127.0.0.1:{port}/api/history?machine=head&range=1h&start=10", timeout=15)
            raise AssertionError("a half window should be refused")
        except HTTPError as exc:
            assert exc.code == 400
        for raw in ("start=NaN&end=10", "start=inf&end=10", "start=1&end=1e20", "start=1&end=90000"):
            try:
                urlopen(f"http://127.0.0.1:{port}/api/history?machine=head&range=1h&{raw}", timeout=15)
                raise AssertionError(raw)
            except HTTPError as exc:
                assert exc.code == 400
    finally:
        httpd.shutdown()
        reset_for_tests()


def _wipe_hot(path) -> None:
    path.write_text(json.dumps({"machines": {}}), encoding="utf-8")


def test_torn_tail_is_kept_and_the_next_sample_restarts(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"cpu_pct": 1}, now=1_700_000_100, role="head")
    day = next((tmp_path / "telemetry" / "head").glob("*.jsonl"))
    with day.open("ab") as handle:
        handle.write(b'{"t":')
    before = day.read_bytes()
    hist.observe("head", {"cpu_pct": 7}, now=1_700_000_200, role="head")
    blob = day.read_bytes()
    assert before in blob
    assert b'{"t":{' not in blob
    assert blob.count(b"\n") >= 2
    _wipe_hot(path)
    again = History(path)
    rows = again.query("head", "1h", now=1_700_000_300)["samples"]
    assert [row["cpu_pct"] for row in rows] == [1, 7]
    listed = again.telemetry_index()
    assert listed["files"][0]["last_t"] == 1_700_000_200
    assert listed["files"][0]["first_t"] == 1_700_000_100


def test_tip_failure_still_reloads_the_sample(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"gpu_pct": 2}, now=1_700_000_400, role="head")

    def fail_meta(*_args, **_kwargs) -> None:
        raise OSError(errno.EIO, "tip")

    hist.archive._append_meta = fail_meta
    assert hist.observe("head", {"gpu_pct": 4}, now=1_700_000_500, role="head") is True
    assert hist.archive.last_error
    _wipe_hot(path)
    again = History(path)
    rows = again.query("head", "1h", now=1_700_000_600)["samples"]
    assert [row["gpu_pct"] for row in rows] == [2, 4]
    cover = again.telemetry_index()["files"][0]
    assert cover["first_t"] == 1_700_000_400
    assert cover["last_t"] == 1_700_000_500


def test_loaded_interval_is_not_the_samples_that_happened_to_exist(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"gpu_pct": 3}, now=1_700_000_100, role="head")
    hist.observe("head", {"gpu_pct": 7}, now=1_700_000_500, role="head")
    hist.observe("head", {"gpu_pct": 9}, now=1_700_001_000, role="head")
    _wipe_hot(path)
    again = History(path)
    first = again.query("head", "1h", now=1_700_001_100, start=1_700_000_100, end=1_700_000_200)
    second = again.query("head", "1h", now=1_700_001_100, start=1_700_000_900, end=1_700_001_100)
    middle = again.query("head", "1h", now=1_700_001_100, start=1_700_000_400, end=1_700_000_600)
    overlap = again.query("head", "1h", now=1_700_001_100, start=1_700_000_150, end=1_700_000_950)
    gap = again.query("head", "1h", now=1_700_001_100, start=1_700_000_300, end=1_700_000_400)
    assert [row["t"] for row in first["samples"]] == [1_700_000_100]
    assert [row["t"] for row in second["samples"]] == [1_700_001_000]
    assert [row["t"] for row in middle["samples"]] == [1_700_000_500]
    assert [row["t"] for row in overlap["samples"]] == [1_700_000_500]
    assert gap["samples"] == []
    restart = History(path)
    left = restart.query("head", "1h", now=1_700_001_100, start=1_700_000_100, end=1_700_000_200)
    right = restart.query("head", "1h", now=1_700_001_100, start=1_700_000_900, end=1_700_001_100)
    assert [row["gpu_pct"] for row in left["samples"]] == [3]
    assert [row["gpu_pct"] for row in right["samples"]] == [9]


def test_metadata_failure_does_not_invent_a_count(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)

    def fail_meta(*_args, **_kwargs) -> None:
        raise OSError(errno.EIO, "tip")

    hist.archive._append_meta = fail_meta
    assert hist.observe("head", {"gpu_pct": 2}, now=1_700_000_400, role="head") is True
    assert (tmp_path / "telemetry" / "error").is_file()
    _wipe_hot(path)
    again = History(path)
    assert again.observe("head", {"gpu_pct": 4}, now=1_700_000_500, role="head") is True
    assert not (tmp_path / "telemetry" / "error").exists()
    rows = again.query("head", "1h", now=1_700_000_600)["samples"]
    assert [row["gpu_pct"] for row in rows] == [2, 4]
    cover = again.telemetry_index()["files"][0]
    assert cover["samples"] is None
    assert cover["first_t"] == 1_700_000_400
    assert cover["last_t"] == 1_700_000_500
    assert again.telemetry_index()["roles"]["worker"]["last_t"] is None
    assert again.telemetry_index()["roles"]["head"]["last_t"] == 1_700_000_500
    assert again.telemetry_index()["last_error"] == ""
    tip = json.loads(next((tmp_path / "telemetry" / "head").glob("*.tip.json")).read_text(encoding="utf-8"))
    assert tip["samples"] is None
    assert tip["first_t"] is None
    assert tip["last_t"] == 1_700_000_500


def test_torn_offset_tail_is_separated(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"cpu_pct": 1}, now=1_700_000_100, role="head")
    idx = next((tmp_path / "telemetry" / "head").glob("*.idx"))
    with idx.open("ab") as handle:
        handle.write(b"12 ")
    hist.observe("head", {"cpu_pct": 2}, now=1_700_000_100 + 120, role="head")
    blob = idx.read_bytes()
    assert b"12 \n" in blob
    assert b"12 1" not in blob
    tip = next((tmp_path / "telemetry" / "head").glob("*.tip.json"))
    tip.write_bytes(b'{"bytes":')
    hist.observe("head", {"cpu_pct": 3}, now=1_700_000_100 + 240, role="head")
    repaired = json.loads(tip.read_text(encoding="utf-8"))
    assert repaired["samples"] is None
    assert repaired["first_t"] is None
    _wipe_hot(path)
    again = History(path)
    rows = again.query("head", "1h", now=1_700_000_400)["samples"]
    assert [row["cpu_pct"] for row in rows] == [1, 2, 3]


def test_queries_do_not_walk_every_day_and_appends_stay_small(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    boundary = 1_704_067_200  # 2024-01-01T00:00:00Z
    hist.observe("head", {"gpu_pct": 1}, now=boundary - 30, role="head")
    hist.observe("head", {"gpu_pct": 2}, now=boundary + 30, role="head")
    decoy = tmp_path / "telemetry" / "head" / "2020-01-01.jsonl"
    decoy.write_text('{"t":1577836800,"role":"head","gpu_pct":9}\n', encoding="utf-8")
    hist.archive.trace = True
    hist.archive.traced.clear()
    hist.archive.bytes_read = 0
    found = hist.query("head", "1h", now=boundary + 40, start=boundary - 40, end=boundary + 40)
    assert [row["gpu_pct"] for row in found["samples"]] == [1, 2]
    assert not any("2020-01-01" in name for name in hist.archive.traced)
    assert sum(name.startswith("stat:") for name in hist.archive.traced) == 2
    hist.archive.trace = False
    before = hist.archive.bytes_read
    for i in range(20):
        hist.observe("worker", {"cpu_pct": i % 5}, now=boundary + 100 + i, role="worker")
    assert hist.archive.bytes_read - before < 20 * 400
    catalog = tmp_path / "telemetry" / "days"
    with catalog.open("a", encoding="utf-8") as handle:
        for i in range(300):
            handle.write(f"worker/2030-01-{i % 28 + 1:02d}.jsonl\n")
    hist.archive.bytes_read = 0
    page = hist.telemetry_index(limit=2)
    assert len(page["files"]) == 2
    assert all(row["role"] == "head" for row in page["files"])
    assert page["roles"]["head"]["last_t"] == boundary + 30
    assert page["roles"]["worker"]["last_t"] == boundary + 119
    assert hist.archive.bytes_read < 4000
    assert page["files"][0]["samples"] == 1


def test_repeated_archive_queries_keep_the_values(tmp_path) -> None:
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"gpu_pct": 3}, now=1_700_000_100, role="head")
    hist.observe("head", {"gpu_pct": 7}, now=1_700_000_500, role="head")
    hist.observe("head", {"gpu_pct": 9}, now=1_700_001_000, role="head")
    _wipe_hot(path)
    again = History(path)
    window = dict(start=1_700_000_100, end=1_700_001_000, now=1_700_001_100)
    first = again.query("head", "1h", **window)
    again._rows.clear()
    again.archive.bytes_read = 0
    second = again.query("head", "1h", **window)
    subset = again.query("head", "1h", now=1_700_001_100, start=1_700_000_400, end=1_700_000_600)
    assert [row["gpu_pct"] for row in first["samples"]] == [3, 7, 9]
    assert second["samples"] == first["samples"]
    assert again.archive.bytes_read == 0
    assert [row["gpu_pct"] for row in subset["samples"]] == [7]
    again.observe("head", {"gpu_pct": 8}, now=1_700_000_700, role="head")
    again._rows.clear()
    grown = again.query("head", "1h", **window)
    assert [row["gpu_pct"] for row in grown["samples"]] == [3, 7, 8, 9]
    windows, samples = again.cache_size()
    assert windows == 1
    assert samples == 4
    for slot in range(20):
        origin = 1_700_002_000 + slot * 100
        again.observe("head", {"gpu_pct": slot}, now=origin, role="head")
        again.query("head", "1h", now=origin + 10, start=origin, end=origin + 10)
    windows, samples = again.cache_size()
    assert windows <= 2
    assert samples <= 2 * 3
    again._rows.clear()
    again.archive.bytes_read = 0
    evicted = again.query("head", "1h", **window)
    assert [row["gpu_pct"] for row in evicted["samples"]] == [3, 7, 8, 9]
    assert again.archive.bytes_read > 0
    assert again.cache_size()[0] <= 2
    restart = History(path)
    restored = restart.query("head", "1h", **window)
    assert [row["gpu_pct"] for row in restored["samples"]] == [3, 7, 8, 9]
    used = restart.archive.bytes_read
    for _ in range(30):
        polled = restart.query("head", "1h", **window)
        assert [row["gpu_pct"] for row in polled["samples"]] == [3, 7, 8, 9]
    assert restart.archive.bytes_read == used
    assert restart.cache_size()[0] <= 2
    assert restart.archive.trace is False
    assert len(restart.archive.traced) == 0


def test_uncached_repeat_still_reads_the_archive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sparksi.history.CACHE_POINTS", 0)
    path = tmp_path / "history.json"
    hist = History(path)
    hist.observe("head", {"gpu_pct": 4}, now=1_700_000_100, role="head")
    _wipe_hot(path)
    again = History(path)
    first = again.query("head", "1h", now=1_700_000_200, start=1_700_000_100, end=1_700_000_200)
    again._rows.clear()
    second = again.query("head", "1h", now=1_700_000_200, start=1_700_000_100, end=1_700_000_200)
    assert first["samples"] == [{"t": 1_700_000_100, "gpu_pct": 4}]
    assert second["samples"] == first["samples"]
    assert again.cache_size() == (0, 0)


def test_day_registration_survives_a_crash_between_commits(tmp_path, monkeypatch) -> None:
    from sparksi.telemetry import valid_catalog_name

    path = tmp_path / "history.json"
    stamp = 1_700_000_100
    day = time.strftime("%Y-%m-%d", time.gmtime(stamp))
    hist = History(path)

    def fail_marker(self, role: str, day_name: str) -> None:
        raise OSError(errno.EIO, "marker")

    monkeypatch.setattr("sparksi.telemetry.Archive._mark_listed", fail_marker)
    assert hist.observe("head", {"gpu_pct": 2}, now=stamp, role="head") is True
    catalog = tmp_path / "telemetry" / "days"
    assert catalog.read_text(encoding="utf-8").count(f"head/{day}.jsonl\n") == 1
    assert not (tmp_path / "telemetry" / "listed" / "head" / day).exists()
    monkeypatch.undo()
    assert hist.observe("head", {"gpu_pct": 4}, now=stamp + 10, role="head") is True
    assert catalog.read_text(encoding="utf-8").count(f"head/{day}.jsonl\n") == 1
    torn = tmp_path / "telemetry-torn" / "history.json"
    torn.parent.mkdir()
    (torn.parent / "telemetry").mkdir()
    (torn.parent / "telemetry" / "days").write_bytes(f"head/{day}.jsonl".encode("utf-8"))
    fresh = History(torn)
    assert fresh.observe("head", {"gpu_pct": 6}, now=stamp, role="head") is True
    torn_catalog = (torn.parent / "telemetry" / "days").read_text(encoding="utf-8")
    assert torn_catalog.count(f"head/{day}.jsonl\n") == 1
    (torn.parent / "telemetry" / "days").open("ab").write(b"foo/bar\nhead/2024-13-01.jsonl\nnot a day\n")
    monkeypatch.setattr(Path, "iterdir", lambda self: (_ for _ in ()).throw(AssertionError("scan")))
    monkeypatch.setattr(Path, "glob", lambda self, pattern="*": (_ for _ in ()).throw(AssertionError("scan")))
    try:
        page = fresh.telemetry_index()
    finally:
        monkeypatch.undo()
    names = [f"{row['role']}/{row['file']}" for row in page["files"]]
    assert names == [f"head/{day}.jsonl"]
    assert all(valid_catalog_name(name) for name in names)
    again = History(torn)
    assert [row["file"] for row in again.telemetry_index()["files"]] == [f"{day}.jsonl"]


def test_registered_day_is_visible_before_its_first_sample(tmp_path, monkeypatch) -> None:
    from sparksi.telemetry import Archive

    stamp = 1_700_000_100
    day = time.strftime("%Y-%m-%d", time.gmtime(stamp))
    nxt = time.strftime("%Y-%m-%d", time.gmtime(stamp + 86400))
    root = tmp_path / "telemetry"
    Archive(root)._ensure_listed("head", day)
    assert not (root / "head" / f"{day}.jsonl").exists()
    monkeypatch.setattr(Path, "iterdir", lambda self: (_ for _ in ()).throw(AssertionError("scan")))
    monkeypatch.setattr(Path, "glob", lambda self, pattern="*": (_ for _ in ()).throw(AssertionError("scan")))
    try:
        page = Archive(root).discover()
    finally:
        monkeypatch.undo()
    assert [f"{row['role']}/{row['file']}" for row in page["files"]] == [f"head/{day}.jsonl"]
    assert page["files"][0]["first_t"] is None
    assert page["files"][0]["samples"] is None
    assert page["files"][0]["bytes"] is None
    (root / "head").mkdir()
    (root / "head" / f"{day}.jsonl").write_bytes(b"")
    empty = Archive(root).discover()["files"][0]
    assert empty["first_t"] is None
    assert empty["samples"] is None
    assert empty["bytes"] == 0
    opened = Path.open

    def fail_jsonl(self, mode="r", *args, **kwargs):
        if self.name.endswith(".jsonl") and "a" in str(mode):
            raise OSError(errno.EIO, "jsonl")
        return opened(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_jsonl)
    crashed = History(tmp_path / "history.json")
    assert crashed.observe("worker", {"cpu_pct": 1}, now=stamp + 86400, role="worker") is True
    assert crashed.archive.last_error
    monkeypatch.undo()
    catalog = (root / "days").read_text(encoding="utf-8")
    assert catalog.count(f"worker/{nxt}.jsonl\n") == 1
    assert not (root / "worker" / f"{nxt}.jsonl").exists()
    restarted = Archive(root)
    names = [f"{row['role']}/{row['file']}" for row in restarted.discover()["files"]]
    assert names == [f"head/{day}.jsonl", f"worker/{nxt}.jsonl"]
    History(tmp_path / "history.json").observe("head", {"gpu_pct": 3}, now=stamp + 86400 + 10, role="head")
    final = (root / "days").read_text(encoding="utf-8")
    assert final.count(f"head/{day}.jsonl\n") == 1
    assert final.count(f"worker/{nxt}.jsonl\n") == 1


def test_checkpoint_does_not_pass_another_roles_missing_marker(tmp_path, monkeypatch) -> None:
    stamp = 1_700_000_100
    day = time.strftime("%Y-%m-%d", time.gmtime(stamp))
    path = tmp_path / "history.json"
    hist = History(path)
    original = hist.archive._mark_listed
    original_commit = hist.archive._set_catalog_commit

    def refuse_head(self, role: str, day_name: str) -> None:
        if role == "head":
            raise OSError(errno.EIO, "head marker")
        return original(role, day_name)

    def checkpoint(self, offset: int) -> None:
        catalog = self.root / "days"
        if catalog.is_file():
            blob = catalog.read_bytes()
            name = f"head/{day}.jsonl\n".encode()
            marker = self.root / "listed" / "head" / day
            if name in blob and not marker.is_file():
                assert offset <= blob.index(name)
        return original_commit(offset)

    monkeypatch.setattr("sparksi.telemetry.Archive._mark_listed", refuse_head)
    monkeypatch.setattr("sparksi.telemetry.Archive._set_catalog_commit", checkpoint)
    assert hist.observe("head", {"gpu_pct": 2}, now=stamp, role="head") is True
    catalog = tmp_path / "telemetry" / "days"
    assert catalog.read_text(encoding="utf-8").count(f"head/{day}.jsonl\n") == 1
    assert not (tmp_path / "telemetry" / "listed" / "head" / day).exists()
    assert hist.observe("worker", {"cpu_pct": 4}, now=stamp + 5, role="worker") is True
    assert catalog.read_text(encoding="utf-8").count(f"head/{day}.jsonl\n") == 1
    assert f"worker/{day}.jsonl\n" not in catalog.read_text(encoding="utf-8")
    ok = tmp_path / "telemetry" / "days.ok"
    recorded = int(ok.read_text(encoding="utf-8")) if ok.is_file() else 0
    assert recorded <= catalog.read_bytes().index(f"head/{day}.jsonl\n".encode())
    monkeypatch.undo()
    healed = History(path)
    assert healed.observe("worker", {"cpu_pct": 5}, now=stamp + 6, role="worker") is True
    assert healed.observe("head", {"gpu_pct": 3}, now=stamp + 7, role="head") is True
    text = catalog.read_text(encoding="utf-8")
    assert text.count(f"head/{day}.jsonl\n") == 1
    assert text.count(f"worker/{day}.jsonl\n") == 1
    assert (tmp_path / "telemetry" / "listed" / "head" / day).is_file()
    assert (tmp_path / "telemetry" / "listed" / "worker" / day).is_file()
    assert int((tmp_path / "telemetry" / "days.ok").read_text(encoding="utf-8")) == catalog.stat().st_size

    reverse = tmp_path / "reverse" / "history.json"
    reverse.parent.mkdir()
    other = History(reverse)
    marks = {"worker": 0}
    real_mark = other.archive._mark_listed.__func__

    def refuse_first_worker(self, role: str, day_name: str) -> None:
        if role == "worker":
            marks["worker"] += 1
            if marks["worker"] == 1:
                raise OSError(errno.EIO, "worker marker")
        return real_mark(self, role, day_name)

    monkeypatch.setattr("sparksi.telemetry.Archive._mark_listed", refuse_first_worker)
    assert other.observe("worker", {"cpu_pct": 1}, now=stamp, role="worker") is True
    monkeypatch.undo()
    assert other.observe("head", {"gpu_pct": 1}, now=stamp + 1, role="head") is True
    assert other.observe("worker", {"cpu_pct": 2}, now=stamp + 2, role="worker") is True
    blob = (reverse.parent / "telemetry" / "days").read_text(encoding="utf-8")
    wday = time.strftime("%Y-%m-%d", time.gmtime(stamp))
    assert blob.count(f"worker/{wday}.jsonl\n") == 1
    assert blob.count(f"head/{wday}.jsonl\n") == 1


def test_bucket_prefers_one_hot_sample(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    start = 1_900_000_000
    hist.observe("head", {"gpu_pct": 10}, now=start, role="head")
    hist.observe("head", {"gpu_pct": 30}, now=start + 1, role="head")
    hist._rows["head"][-1]["gpu_pct"] = 90
    out = hist.query("head", "24h", now=start + 10, start=start, end=start + 4000)
    assert out["samples"] == [{"t": start + 1, "gpu_pct": 50}]


def test_live_queries_stop_rereading_and_discovery_is_paged(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    origin = 2_000_000_000
    for i in range(5):
        hist.observe("head", {"gpu_pct": i}, now=origin + i, role="head")
    hist.query("head", "1h", now=origin + 10, start=origin, end=origin + 10)
    used = hist.archive.bytes_read
    assert used > 0
    for _ in range(10):
        hist.query("head", "1h", now=origin + 10, start=origin, end=origin + 10)
    assert hist.archive.bytes_read == used
    for day in range(4):
        hist.observe("worker", {"cpu_pct": day}, now=origin + day * 86400, role="worker")
    page = hist.telemetry_index(role="worker", limit=2)
    assert len(page["files"]) == 2
    assert page["next"]
    follow = hist.telemetry_index(role="worker", cursor=page["next"], limit=2)
    assert follow["files"]
    assert follow["files"][0]["path"] != page["files"][0]["path"]
    assert "cpu_pct" not in json.dumps(page["files"])
    with pytest.raises(ValueError):
        hist.query("head", "1h", now=float("inf"))
    with pytest.raises(ValueError):
        hist.query("head", "1h", now=origin, start=float("nan"), end=origin + 10)
    with pytest.raises(ValueError):
        hist.query("head", "24h", now=origin, start=origin, end=origin + 86401)


def test_since_returns_only_newer_samples_and_stays_optional(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    for stamp, gpu in ((100, 1), (101, 2), (102, 3), (130, 4)):
        hist.observe("head", {"gpu_pct": gpu}, now=stamp, role="head")
    hist.observe("worker", {"gpu_pct": 9}, now=130, role="worker")
    full = hist.query("head", "1h", now=200)
    assert [row["t"] for row in full["samples"]] == [100, 101, 102, 130]
    assert "since" not in full
    newer = hist.query("head", "1h", now=200, since=102)
    assert [row["t"] for row in newer["samples"]] == [130]
    assert newer["since"] == 102
    assert newer["start"] == 200 - 3600
    window = hist.query("head", "1h", now=200, start=100, end=130, since=101)
    assert [row["t"] for row in window["samples"]] == [102, 130]
    assert hist.query("head", "1h", now=200, since=500)["samples"] == []
    latest = hist.latest_samples()
    assert latest["head"] == {"t": 130, "gpu_pct": 4}
    assert latest["worker"] == {"t": 130, "gpu_pct": 9}
    quiet = History(tmp_path / "quiet.json", now=lambda: 10_000 + 3_000)
    quiet.observe("head", {"gpu_pct": 1}, now=10_000, role="head")
    quiet.observe("worker", {"gpu_pct": 2}, now=10_000 + 3_000, role="worker")
    fresh = quiet.latest_samples(fresh_secs=1200)
    assert "head" not in fresh
    assert fresh["worker"]["t"] == 10_000 + 3_000
    assert quiet.latest_samples()["head"]["gpu_pct"] == 1
    assert quiet.query("head", "1h", now=10_000 + 3_000)["samples"][0]["gpu_pct"] == 1
    assert len(hist.query("head", "1h", now=200)["samples"]) == 4
    with pytest.raises(ValueError):
        hist.query("head", "1h", now=200, since=float("nan"))
