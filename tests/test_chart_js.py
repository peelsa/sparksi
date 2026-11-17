# SPDX-License-Identifier: Apache-2.0

"""Display math for the overview chart. Storage and the 1 s poll stay raw."""

import json
import subprocess
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

from sparksi.history import BUCKET_SECS, KEEP_SECS, reset_for_tests
from sparksi.lock import Lock
from sparksi.serve import INDEX_HTML, STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


def _math() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    block = html.split("/* chart-math-start */", 1)[1].split("/* chart-math-end */", 1)[0]
    return (
        "function toF(celsius) { return (Number(celsius) * 9 / 5) + 32; }\n"
        + block
    )


def _run(body: str):
    script = _math() + "\n" + body + "\nprocess.stdout.write(JSON.stringify(out));\n"
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_pinned_window_does_not_follow_a_later_refresh() -> None:
    out = _run(
        "var out = {\n"
        "  live: pinnedEnd(5000, true, 1000),\n"
        "  held: [pinnedEnd(5000, false, 3000), pinnedEnd(5600, false, 3000)],\n"
        "  future: pinnedEnd(5000, false, 9000),\n"
        "  floor: pinnedEnd(100000, false, 1000)\n"
        "};"
    )
    assert out["live"] == 5000
    assert out["held"] == [3000, 3000]
    assert out["future"] == 5000
    assert out["floor"] == 100000 - 86400 + 1200


def test_chart_source_keeps_the_fleet_rules() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "preserveAspectRatio" not in html
    assert "#7CC214" not in html
    assert "7cc214" not in html.lower()
    assert "innerHTML" not in html
    assert "localStorage" not in html
    assert "http://" not in html
    assert "https://" not in html
    # The first two Sparks keep their colors; a third and fourth add two more card greens.
    assert 'var SERIES_COLORS = ["#D1F7CB", "#6AAA0C", "#A7DF6E", "#4B9412"];' in html
    assert "vector-effect" in html
    assert "ResizeObserver" in html
    assert "seriesOff" in html
    chunk = html.split("var SERIES_COLORS", 1)[1].split("function tickHistory", 1)[0]
    assert chunk.index('key: "gpu_pct"') < chunk.index('key: "cpu_pct"')
    assert chunk.index('key: "cpu_pct"') < chunk.index('key: "mem_pct"')
    assert chunk.index('key: "mem_pct"') < chunk.index('key: "gpu_temp_c"')
    assert 'dash: "6 4"' in chunk
    assert 'dash: "1.5 3.5"' in chunk
    assert 'dash: "10 3 2 3"' in chunk
    assert "power_w" not in chunk
    assert "tok_s" not in chunk
    assert "var CHART_WINDOW = 1200;" in chunk
    assert "covered / 0.75" not in chunk
    assert "0.75 * span" not in chunk
    assert "((Number(now) - Number(t)) / CHART_WINDOW)" in chunk
    assert "inChartWindow" in chunk
    assert 'range=1h' in html
    assert "> 90" in chunk
    assert 'var fLabels = ["0 F", "50 F", "100 F", "150 F", "200 F"];' in chunk
    assert "Right axis 0 F to 200 F" in chunk
    assert "80 F" not in chunk and "110 F" not in chunk
    assert "140 F" not in chunk and "170 F" not in chunk
    assert "Math.max(0, Math.min(200, value))" in chunk
    assert "Math.max(80," not in html
    assert "tempPercent" in chunk
    assert "return yPct(tempPercent(fahrenheit));" in chunk
    assert KEEP_SECS == 3600
    assert BUCKET_SECS == 1
    serve = (ROOT / "src" / "sparksi" / "serve.py").read_text(encoding="utf-8")
    assert "time.sleep(1)" in serve
    assert "setInterval(tickHistory, 1000)" in html


def test_gaps_and_missing_readings_are_not_zeros() -> None:
    pct = {"key": "gpu_pct", "axis": "pct"}
    out = _run(
        "var pct = " + json.dumps(pct) + ";\n"
        "var out = {\n"
        "  missing: chartSeries([\n"
        "    {t: 1000, gpu_pct: 40},\n"
        "    {t: 1001},\n"
        "    {t: 1002, gpu_pct: 0},\n"
        "    {t: 1100, gpu_pct: 10}\n"
        "  ], pct, 60, 600),\n"
        "  gap90: chartSeries([{t: 1000, gpu_pct: 5}, {t: 1090, gpu_pct: 6}], pct, 60, 600),\n"
        "  gap91: chartSeries([{t: 1000, gpu_pct: 5}, {t: 1091, gpu_pct: 6}], pct, 60, 600),\n"
        "  zero: decimateRun([{t: 5, gpu_pct: 0}], pct, 1),\n"
        "  unsorted: chartSeries([{t: 30, gpu_pct: 2}, {t: 10, gpu_pct: 1}], pct, 60, 600)\n"
        "};"
    )
    assert len(out["missing"]) == 3
    assert out["missing"][0][0]["value"] == 40
    assert out["missing"][1][0]["t"] == 1002
    assert out["missing"][1][0]["value"] == 0
    assert out["missing"][2][0]["value"] == 10
    assert all(pt["t"] != 1001 for run in out["missing"] for pt in run)
    assert len(out["gap90"]) == 1 and len(out["gap90"][0]) == 2
    assert len(out["gap91"]) == 2
    assert out["zero"] == [{"t": 5, "value": 0, "min": 0, "max": 0}]
    assert [pt["t"] for pt in out["unsorted"][0]] == [10, 30]


def test_buckets_average_after_the_gap_split_and_keep_the_newest_sample() -> None:
    pct = {"key": "gpu_pct", "axis": "pct"}
    temp = {"key": "gpu_temp_c", "axis": "f"}
    out = _run(
        "var pct = " + json.dumps(pct) + ";\n"
        "var temp = " + json.dumps(temp) + ";\n"
        "var burst = [];\n"
        "for (var t = 1000; t <= 1009; t++) burst.push({t: t, gpu_pct: t % 2 ? 100 : 0});\n"
        "burst.push({t: 1010, gpu_pct: 40});\n"
        "var partial = [];\n"
        "for (var u = 1000; u <= 1003; u++) partial.push({t: u, gpu_pct: u === 1001 ? 100 : 0});\n"
        "var out = {\n"
        "  bucket: pickBucket(60, 600),\n"
        "  hour: pickBucket(3600, 700),\n"
        "  narrow: pickBucket(3600, 200),\n"
        "  ticks: [pickTickStep(45), pickTickStep(90), pickTickStep(2700)],\n"
        "  mean: decimateRun(burst, pct, 10),\n"
        "  partial: decimateRun(partial, pct, 10),\n"
        "  walls: decimateRun([{t: 1009, gpu_pct: 10}, {t: 1011, gpu_pct: 30}], pct, 10),\n"
        "  clamp: decimateRun([{t: 1000, gpu_pct: 150}, {t: 1001, gpu_pct: -50}, {t: 1010, gpu_pct: 10}], pct, 10),\n"
        "  cold: decimateRun([{t: 0, gpu_temp_c: 0}, {t: 1, gpu_temp_c: 100}], temp, 1),\n"
        "  mild: decimateRun([{t: 2, gpu_temp_c: 40}], temp, 1),\n"
        "  share: [tempPercent(0), tempPercent(50), tempPercent(100), tempPercent(150), tempPercent(200), tempPercent(250), tempPercent(-20), tempPercent(170)],\n"
        "  raw: chartSeries([{t: 10, gpu_pct: 4}, {t: 11, gpu_pct: 8}], pct, 60, 600)\n"
        "};"
    )
    assert out["bucket"] == 1
    assert out["hour"] == 15
    assert out["narrow"] == 30
    assert out["ticks"] == [15, 15, 600]
    assert out["mean"][0] == {"t": 1005, "value": 50, "min": 0, "max": 100}
    assert out["mean"][-1]["t"] == 1010
    assert out["mean"][-1]["value"] == 40
    assert out["partial"] == [{"t": 1003, "value": 0, "min": 0, "max": 100}]
    assert [pt["value"] for pt in out["walls"]] == [10, 30]
    assert out["walls"][0]["t"] == 1005
    assert out["clamp"][0]["value"] == 50
    assert out["clamp"][0]["min"] == 0
    assert out["clamp"][0]["max"] == 100
    assert out["cold"][0] == {"t": 0, "value": 32, "min": 32, "max": 32}
    assert out["cold"][1] == {"t": 1, "value": 200, "min": 200, "max": 200}
    assert out["mild"] == [{"t": 2, "value": 104, "min": 104, "max": 104}]
    assert out["share"] == [0, 25, 50, 75, 100, 100, 0, 85]
    assert out["raw"][0][0]["t"] == 10
    assert out["raw"][0][1]["t"] == 11


def test_window_is_twenty_minutes_and_now_is_on_the_left() -> None:
    minus = "\u2212"
    out = _run(
        "var now = 10000;\n"
        "var left = 46;\n"
        "var width = 400;\n"
        "var ages = timeTickAges(CHART_WINDOW);\n"
        "var out = {\n"
        "  span: [chartSpan(0), chartSpan(45), chartSpan(90), chartSpan(2700), chartSpan(100000)],\n"
        "  start: windowStart(now),\n"
        "  tick: pickTickStep(CHART_WINDOW),\n"
        "  ages: ages,\n"
        "  labels: ages.map(relLabel),\n"
        "  xNow: xAt(now, now, left, width),\n"
        "  x5: xAt(now - 300, now, left, width),\n"
        "  x10: xAt(now - 600, now, left, width),\n"
        "  x15: xAt(now - 900, now, left, width),\n"
        "  x20: xAt(now - 1200, now, left, width),\n"
        "  x30: xAt(now - 1800, now, left, width),\n"
        "  keep: [inChartWindow(now, now), inChartWindow(now - 1200, now), inChartWindow(now + 1, now)],\n"
        "  drop: [inChartWindow(now - 1201, now), inChartWindow(now - 1800, now), inChartWindow(now + 2, now)],\n"
        "  newest: latestRaw(\n"
        "    [{t: now - 1800, gpu_pct: 100}, {t: now - 30, gpu_pct: 40}],\n"
        "    {key: 'gpu_pct', axis: 'pct'},\n"
        "    windowStart(now),\n"
        "    now\n"
        "  )\n"
        "};"
    )
    assert out["span"] == [1200, 1200, 1200, 1200, 1200]
    assert out["start"] == 8800
    assert out["tick"] == 300
    assert out["ages"] == [300, 600, 900, 1200]
    assert out["labels"] == [minus + "5 m", minus + "10 m", minus + "15 m", minus + "20 m"]
    assert out["xNow"] == 46
    assert out["x5"] == 146
    assert out["x10"] == 246
    assert out["x15"] == 346
    assert out["x20"] == 446
    assert out["x30"] == 646
    assert out["xNow"] < out["x20"] < out["x30"]
    assert out["keep"] == [True, True, True]
    assert out["drop"] == [False, False, False]
    assert out["newest"] == 40


def test_latest_reading_ignores_a_missing_field_and_stays_off_zero() -> None:
    pct = {"key": "gpu_pct", "axis": "pct"}
    out = _run(
        "var pct = " + json.dumps(pct) + ";\n"
        "var out = {\n"
        "  missing: latestRaw([{t: 1, gpu_pct: 12}, {t: 2}], pct, 0, 10),\n"
        "  zero: latestRaw([{t: 1, gpu_pct: 0}], pct, 0, 10),\n"
        "  old: latestRaw([{t: 1, gpu_pct: 9}], pct, 5, 10)\n"
        "};"
    )
    assert out["missing"] is None
    assert out["zero"] == 0
    assert out["old"] is None


def test_live_tick_makes_no_history_request(tmp_path, monkeypatch) -> None:
    """A live second reads the monitor sample, not the hour."""
    out = _run(
        "var out = planHistoryTick({\n"
        "  hidden: false,\n"
        "  inflight: false,\n"
        "  seeded: true,\n"
        "  follow: true,\n"
        "  now: 10000,\n"
        "  start: 8800,\n"
        "  end: 10000,\n"
        "  machines: [\n"
        "    {id: 'head', lastT: 10000, newestT: null},\n"
        "    {id: 'worker', lastT: 10000, newestT: null}\n"
        "  ],\n"
        "  cached: []\n"
        "});\n"
    )
    assert out["kind"] == "live"
    assert out["requests"] == []
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "setInterval(tickMonitor, 1000)" in html
    assert "AbortController" in html
    assert "localStorage" not in html
    assert "indexedDB" not in html
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    reset_for_tests()
    end = int(time.time())
    count = 3200

    def rows(origin: int) -> list[dict]:
        return [
            {
                "t": origin + i,
                "cpu_pct": 4,
                "gpu_pct": 12 + (i % 3),
                "gpu_temp_c": 42.2,
                "mem_pct": 25,
                "mem_used_gib": 32.5,
                "power_w": 40,
                "tok_s": 18.4,
            }
            for i in range(count)
        ]

    origin = end - count + 1
    (tmp_path / "history.json").write_text(
        json.dumps({"machines": {"head": rows(origin), "worker": rows(origin)}}),
        encoding="utf-8",
    )
    reset_for_tests()
    STATE.lock = Lock(tmp_path / "lock")
    STATE.monitor_fn = lambda: {
        "ok": True,
        "ts": "12:00:00",
        "hosts": [
            {"id": "head", "fresh": True, "cpu_pct": 4, "gpu_pct": 12, "mem_pct": 25, "gpu_temp_c": 42.2},
            {"id": "worker", "fresh": True, "cpu_pct": 4, "gpu_pct": 12, "mem_pct": 25, "gpu_temp_c": 42.2},
        ],
        "loaded": [],
        "loaded_summary": {"count": 0, "running": 0, "waiting": 0, "mode": "auto"},
    }
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    port = httpd.server_address[1]
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/monitor", timeout=15) as resp:
            monitor_raw = resp.read()
        monitor = json.loads(monitor_raw)
        with urlopen(f"http://127.0.0.1:{port}/api/history?machine=head&range=1h", timeout=15) as resp:
            head_raw = resp.read()
        with urlopen(f"http://127.0.0.1:{port}/api/history?machine=worker&range=1h", timeout=15) as resp:
            worker_raw = resp.read()
        head = json.loads(head_raw)
        since = end - 5
        try:
            urlopen(f"http://127.0.0.1:{port}/api/history?machine=head&range=1h&since=NaN", timeout=15)
            raise AssertionError("a bad since should be refused")
        except HTTPError as exc:
            assert exc.code == 400
        with urlopen(
            f"http://127.0.0.1:{port}/api/history?machine=head&range=1h&since={since}",
            timeout=15,
        ) as resp:
            tail_raw = resp.read()
        tail = json.loads(tail_raw)
    finally:
        httpd.shutdown()
        STATE.monitor_fn = None
        reset_for_tests()
    assert set(monitor["history"]) == {"head", "worker"}
    assert monitor["history"]["head"]["t"] == origin + count - 1
    assert isinstance(monitor["history"]["head"], dict)
    assert "samples" not in monitor["history"]["head"]
    assert len(head["samples"]) >= 3000
    assert "since" not in head
    assert tail["since"] == since
    assert tail["samples"]
    assert all(row["t"] > since for row in tail["samples"])
    assert len(tail["samples"]) < 20
    before = len(head_raw) + len(worker_raw)
    after = len(monitor_raw)
    # The old live poll downloaded both hours every second, on top of the monitor card.
    assert before > 1_000_000
    assert after < before / 50
    assert len(tail_raw) < len(head_raw) / 50


def test_ring_trims_at_twenty_minutes() -> None:
    out = _run(
        "var rows = [];\n"
        "var now = 10000;\n"
        "for (var t = now - 2000; t <= now; t++) appendRing(rows, {t: t, gpu_pct: 1}, now);\n"
        "var n = rows.length;\n"
        "var first = rows[0].t;\n"
        "var last = rows[rows.length - 1].t;\n"
        "appendRing(rows, {t: now, gpu_pct: 7}, now);\n"
        "var afterDup = rows.length;\n"
        "appendRing(rows, {t: now + 1, gpu_pct: 4}, now + 1);\n"
        "var merged = [{t: 8800, gpu_pct: 1}, {t: 10000, gpu_pct: 2}];\n"
        "mergeRing(merged, [{t: 8800, gpu_pct: 9}, {t: 9000, gpu_pct: 3}, {t: 10000, gpu_pct: 8}, {t: 7000, gpu_pct: 1}], now);\n"
        "var out = {\n"
        "  first: first,\n"
        "  last: last,\n"
        "  n: n,\n"
        "  afterDup: afterDup,\n"
        "  stepped: [rows[0].t, rows[rows.length - 1].t, rows.length],\n"
        "  merged: merged.map(function (row) { return [row.t, row.gpu_pct]; })\n"
        "};\n"
    )
    assert out["first"] == 8800
    assert out["last"] == 10000
    assert out["n"] == 1201
    assert out["afterDup"] == 1201
    assert out["stepped"] == [8801, 10001, 1201]
    assert out["merged"] == [[8800, 1], [9000, 3], [10000, 2]]


def test_gap_backfill_uses_since() -> None:
    out = _run(
        "var plan = planHistoryTick({\n"
        "  hidden: false,\n"
        "  inflight: false,\n"
        "  seeded: true,\n"
        "  follow: true,\n"
        "  now: 5000,\n"
        "  start: 3800,\n"
        "  end: 5000,\n"
        "  machines: [\n"
        "    {id: 'head', lastT: 4900, newestT: 5000},\n"
        "    {id: 'worker', lastT: 5000, newestT: 5000}\n"
        "  ],\n"
        "  cached: []\n"
        "});\n"
        "var out = {kind: plan.kind, urls: plan.requests.map(function (item) { return item.url; })};\n"
    )
    assert out["kind"] == "since"
    assert len(out["urls"]) == 1
    assert "machine=head" in out["urls"][0]
    assert "since=4900" in out["urls"][0]
    assert "range=1h" in out["urls"][0]
    assert "start=" not in out["urls"][0]


def test_pan_window_is_fetched_once_then_cached() -> None:
    out = _run(
        "function state(cached, start, end) {\n"
        "  return {\n"
        "    hidden: false,\n"
        "    inflight: false,\n"
        "    seeded: true,\n"
        "    follow: false,\n"
        "    now: 20000,\n"
        "    start: start,\n"
        "    end: end,\n"
        "    machines: [{id: 'head'}, {id: 'worker'}],\n"
        "    cached: cached\n"
        "  };\n"
        "}\n"
        "var first = planHistoryTick(state([], 8000, 9200));\n"
        "var order = rememberWindow([], first.key, 8);\n"
        "var second = planHistoryTick(state(order, 8000, 9200));\n"
        "var other = planHistoryTick(state(order, 7000, 8200));\n"
        "var filled = [];\n"
        "for (var n = 0; n < 9; n++) filled = rememberWindow(filled, 'k' + n, 8);\n"
        "var busy = planHistoryTick({\n"
        "  hidden: false, inflight: true, seeded: true, follow: false,\n"
        "  now: 20000, start: 8000, end: 9200,\n"
        "  machines: [{id: 'head'}], cached: []\n"
        "});\n"
        "var out = {\n"
        "  firstKind: first.kind,\n"
        "  firstN: first.requests.length,\n"
        "  firstUrl: first.requests[0].url,\n"
        "  key: first.key,\n"
        "  secondKind: second.kind,\n"
        "  secondN: second.requests.length,\n"
        "  otherKind: other.kind,\n"
        "  otherN: other.requests.length,\n"
        "  lru: filled,\n"
        "  busy: busy.kind,\n"
        "  busyN: busy.requests.length\n"
        "};\n"
    )
    assert out["firstKind"] == "pan"
    assert out["firstN"] == 2
    assert out["key"] == "8000:9200"
    assert "start=8000" in out["firstUrl"] and "end=9200" in out["firstUrl"]
    assert "since=" not in out["firstUrl"]
    assert out["secondKind"] == "cache"
    assert out["secondN"] == 0
    assert out["otherKind"] == "pan"
    assert out["otherN"] == 2
    assert out["lru"][0] == "k1"
    assert out["lru"][-1] == "k8"
    assert "k0" not in out["lru"]
    assert len(out["lru"]) == 8
    assert out["busy"] == "inflight"
    assert out["busyN"] == 0


def test_hidden_tab_stops_ticking() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    body = html.split("function tickHistory", 1)[1].split("function paintTelemetry", 1)[0]
    assert "document.hidden" in body
    assert "cancelHistory" in body
    out = _run(
        "var hidden = planHistoryTick({\n"
        "  hidden: true,\n"
        "  inflight: false,\n"
        "  seeded: true,\n"
        "  follow: true,\n"
        "  now: 9000,\n"
        "  machines: [{id: 'head', lastT: 1000, newestT: 9000}, {id: 'worker', lastT: 1000, newestT: 9000}],\n"
        "  cached: []\n"
        "});\n"
        "var out = {kind: hidden.kind, n: hidden.requests.length};\n"
    )
    assert out["kind"] == "pause"
    assert out["n"] == 0


def test_incremental_decimation_matches_a_full_rebuild() -> None:
    out = _run(
        "var pct = {key: 'gpu_pct', axis: 'pct'};\n"
        "var temp = {key: 'gpu_temp_c', axis: 'f'};\n"
        "if (pickBucket(CHART_WINDOW, 600) !== 5) throw new Error('bucket');\n"
        "var slot = {fold: null, times: null};\n"
        "var warm = {fold: null, times: null};\n"
        "var rows = [];\n"
        "var now = 10000;\n"
        "var kept = false;\n"
        "var t = 8600;\n"
        "while (t <= now) {\n"
        "  if (t === 9400) { t = 9491; continue; }\n"
        "  var sample = {t: t, gpu_pct: (t % 5) * 20, gpu_temp_c: 40 + (t % 3)};\n"
        "  if (t === 9000) sample = {t: t, gpu_temp_c: 40};\n"
        "  rows.push(sample);\n"
        "  var visible = rows.filter(function (item) { return inChartWindow(item.t, now); });\n"
        "  var before = slot.fold && slot.fold.runs.length ? slot.fold.runs[0].groups[0] : null;\n"
        "  slot = syncSeriesFold(slot, visible, pct, 5);\n"
        "  warm = syncSeriesFold(warm, visible, temp, 5);\n"
        "  var fresh = chartSeries(visible, pct, CHART_WINDOW, 600);\n"
        "  var freshTemp = chartSeries(visible, temp, CHART_WINDOW, 600);\n"
        "  if (JSON.stringify(slot.runs) !== JSON.stringify(fresh)) throw new Error('gpu ' + t);\n"
        "  if (JSON.stringify(warm.runs) !== JSON.stringify(freshTemp)) throw new Error('temp ' + t);\n"
        "  if (before && visible.length > 40 && Number(visible[0].t) === before.rows[0].t) {\n"
        "    if (slot.fold.runs[0].groups[0] !== before) throw new Error('rebuilt ' + t);\n"
        "    kept = true;\n"
        "  }\n"
        "  t += 1;\n"
        "}\n"
        "var later = now + 80;\n"
        "for (var extra = now + 1; extra <= later; extra++) {\n"
        "  rows.push({t: extra, gpu_pct: extra % 50, gpu_temp_c: 41});\n"
        "  var view = rows.filter(function (item) { return inChartWindow(item.t, later); });\n"
        "  slot = syncSeriesFold(slot, view, pct, 5);\n"
        "  var again = chartSeries(view, pct, CHART_WINDOW, 600);\n"
        "  if (JSON.stringify(slot.runs) !== JSON.stringify(again)) throw new Error('trim ' + extra);\n"
        "}\n"
        "var out = {kept: kept, points: slot.runs.length};\n"
    )
    assert out["kept"] is True
    assert out["points"] >= 1


def _live(body: str):
    """Run the chart math plus the live history handlers. The body writes JSON."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    math = html.split("/* chart-math-start */", 1)[1].split("/* chart-math-end */", 1)[0]
    live = html.split("function machineToken", 1)[1].split("function paintTelemetry", 1)[0]
    script = "function toF(celsius) { return (Number(celsius) * 9 / 5) + 32; }\n" + math
    script += "function machineToken" + live + "\n" + body
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


_LIVE_PREAMBLE = r"""
function roleId(role) { return role === "worker" ? "worker" : "head"; }
function sparkRoles() { return ["head", "worker"]; }
function paintFleetChart() {}
var document = { hidden: false };
var CHART_FOLLOW = true;
var CHART_PIN = 0;
var liveRings = {};
var ringCursor = {};
var ringReseed = {};
var seededToken = "head|worker";
var seriesFold = {};
var windowOrder = [];
var windowStore = {};
var panView = null;
var historyBusy = false;
var historyKind = "";
var historyKey = "";
var historyGen = 0;
var historyCtl = null;
var chartHidden = false;
var chartLeftLive = false;
var chartRecover = false;
var fetches = [];
var failFetch = false;
var pending = [];
function armFetch(mode) {
  globalThis.fetch = function (url) {
    fetches.push(url);
    if (mode === "fail") return Promise.reject(new Error("down"));
    if (mode === "hold") {
      return new Promise(function (resolve) {
        pending.push(function (body) {
          resolve({ ok: true, json: function () { return Promise.resolve(body || { samples: [] }); } });
        });
      });
    }
    return Promise.resolve({ ok: true, json: function () { return Promise.resolve({ samples: [] }); } });
  };
}
function wait(ms) { return new Promise(function (resolve) { setTimeout(resolve, ms); }); }
"""


def test_server_clock_ahead_of_the_client_is_kept() -> None:
    out = _run(
        "var client = 2000000000;\n"
        "var rows = [];\n"
        "var t;\n"
        "for (t = client - 10; t <= client; t++) appendRing(rows, {t: t, gpu_pct: 1}, client);\n"
        "appendRing(rows, {t: client + 5, gpu_pct: 8}, client);\n"
        "var out = {\n"
        "  last: rows[rows.length - 1].t,\n"
        "  first: rows[0].t,\n"
        "  edge: followNow(client, [rows]),\n"
        "  visible: inChartWindow(client + 5, followNow(client, [rows]))\n"
        "};\n"
    )
    assert out["last"] == 2000000005
    assert out["first"] >= 2000000005 - 1200
    assert out["edge"] == 2000000005
    assert out["visible"] is True


def test_failed_since_is_retried_from_the_cursor() -> None:
    out = _live(
        _LIVE_PREAMBLE
        + r"""
armFetch("fail");
liveRings.head = [{t: 100, gpu_pct: 1}, {t: 101, gpu_pct: 1}];
ringCursor.head = 101;
Date.now = function () { return 200 * 1000; };
ingestMonitorHistory({ history: { head: { t: 105, gpu_pct: 3 } } });
wait(30).then(function () {
  var first = fetches.slice();
  tickHistory();
  return wait(30).then(function () {
    process.stdout.write(JSON.stringify({
      cursor: ringCursor.head,
      tip: liveRings.head[liveRings.head.length - 1].t,
      kept: liveRings.head.some(function (row) { return row.t === 105; }),
      first: first,
      second: fetches.slice(first.length)
    }));
  });
});
"""
    )
    assert out["cursor"] == 101
    assert out["tip"] == 105
    assert out["kept"] is True
    assert len(out["first"]) == 1
    assert "since=101" in out["first"][0]
    assert "machine=head" in out["first"][0]
    assert len(out["second"]) == 1
    assert "since=101" in out["second"][0]


def test_in_flight_since_still_covers_the_other_machine() -> None:
    out = _live(
        _LIVE_PREAMBLE
        + r"""
armFetch("hold");
liveRings.head = [{t: 100, gpu_pct: 1}, {t: 101, gpu_pct: 1}];
liveRings.worker = [{t: 100, gpu_pct: 1}, {t: 101, gpu_pct: 1}];
ringCursor.head = 101;
ringCursor.worker = 101;
Date.now = function () { return 200 * 1000; };
ingestMonitorHistory({ history: { head: { t: 110, gpu_pct: 2 } } });
ingestMonitorHistory({ history: { worker: { t: 130, gpu_pct: 4 } } });
var during = fetches.slice();
var fill = [];
var t;
for (t = 102; t <= 110; t++) fill.push({ t: t, gpu_pct: 2 });
pending[0]({ samples: fill });
wait(40).then(function () {
  process.stdout.write(JSON.stringify({
    during: during,
    after: fetches.slice(during.length),
    headCursor: ringCursor.head,
    workerCursor: ringCursor.worker,
    workerTip: liveRings.worker[liveRings.worker.length - 1].t
  }));
});
"""
    )
    assert len(out["during"]) == 1
    assert "machine=head" in out["during"][0]
    assert "since=101" in out["during"][0]
    assert out["workerCursor"] == 101
    assert out["workerTip"] == 130
    assert out["headCursor"] == 110
    assert len(out["after"]) == 1
    assert "machine=worker" in out["after"][0]
    assert "since=101" in out["after"][0]


def test_empty_ring_reseed_fetches_twenty_minutes() -> None:
    out = _live(
        _LIVE_PREAMBLE
        + r"""
armFetch("hold");
var client = 2000000000;
Date.now = function () { return client * 1000; };
liveRings.head = [];
ingestMonitorHistory({ history: { head: { t: client + 5, gpu_pct: 6 } } });
process.stdout.write(JSON.stringify({
  url: fetches[0] || "",
  reseed: !!ringReseed.head,
  tip: liveRings.head.length ? liveRings.head[liveRings.head.length - 1].t : null,
  cursor: ringCursor.head == null ? null : ringCursor.head
}));
"""
    )
    assert out["reseed"] is True
    assert out["tip"] == 2000000005
    assert "start=" in out["url"] and "end=" in out["url"]
    assert "since=" not in out["url"]
    assert "machine=head" in out["url"]
    end = 2000000005
    assert f"end={end}" in out["url"]
    assert f"start={end - 1200}" in out["url"]


def test_return_to_live_skips_since_when_the_cursor_matches_the_tip() -> None:
    out = _live(
        _LIVE_PREAMBLE
        + r"""
armFetch("hold");
var client = 2000000000;
Date.now = function () { return client * 1000; };
liveRings.head = [{t: client - 1, gpu_pct: 1}];
ringCursor.head = client - 1;
CHART_FOLLOW = false;
tickHistory();
CHART_FOLLOW = true;
CHART_PIN = 0;
var before = fetches.length;
tickHistory();
var matched = fetches.slice(before);
liveRings.head.push({t: client + 4, gpu_pct: 2});
ringCursor.head = client - 1;
CHART_FOLLOW = false;
tickHistory();
CHART_FOLLOW = true;
before = fetches.length;
tickHistory();
process.stdout.write(JSON.stringify({
  matched: matched,
  hole: fetches.slice(before)
}));
"""
    )
    assert out["matched"] == []
    assert len(out["hole"]) == 1
    assert "since=" + str(2000000000 - 1) in out["hole"][0]
    assert "machine=head" in out["hole"][0]


def test_stale_monitor_sample_does_not_reseed() -> None:
    out = _live(
        _LIVE_PREAMBLE
        + r"""
armFetch("hold");
var client = 2000000000;
Date.now = function () { return client * 1000; };
liveRings.head = [];
ingestMonitorHistory({ history: { head: { t: client - 1800, gpu_pct: 1 } } });
var afterStale = fetches.slice();
tickHistory();
var afterTick = fetches.slice();
ingestMonitorHistory({ history: { head: { t: client - 1800, gpu_pct: 1 } } });
tickHistory();
var afterRepeat = fetches.slice();
ingestMonitorHistory({ history: { head: { t: client, gpu_pct: 4 } } });
process.stdout.write(JSON.stringify({
  afterStale: afterStale,
  afterTick: afterTick,
  afterRepeat: afterRepeat,
  fresh: fetches.slice(afterRepeat.length),
  reseed: !!ringReseed.head,
  tip: liveRings.head.length ? liveRings.head[liveRings.head.length - 1].t : null,
  n: liveRings.head.length
}));
"""
    )
    assert out["afterStale"] == []
    assert out["afterTick"] == []
    assert out["afterRepeat"] == []
    assert out["n"] == 1
    assert out["tip"] == 2000000000
    assert out["reseed"] is True
    assert len(out["fresh"]) == 1
    assert "start=" in out["fresh"][0] and "end=" in out["fresh"][0]
    assert "since=" not in out["fresh"][0]
    assert "machine=head" in out["fresh"][0]
