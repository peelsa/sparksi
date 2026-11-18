# SPDX-License-Identifier: Apache-2.0
"""Leaving a page must not raise page errors or drop writes.

Once a navigation away starts (``beforeunload``), WebKit keeps the old page's
timers running until the new page arrives, but refuses every ordinary request
the old page starts in that window. Each refusal is logged as "Fetch API cannot
load ... due to access control checks." and Playwright reports it as a page
error. A busy machine stretches the window, so the 1 s polls landed in it now
and then. These tests hold the next page's document so the window is long
enough for every poll to fire in it, including past the page's 5 s hold.
"""

import json
import os
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
from pathlib import Path

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src" / "sparksi" / "static" / "index.html"
SINK: list[int] = []
SLOW: list[int] = []
STATUS: list[int] = []


class _SinkHandler(Handler):
    """The real server, plus a blank page and a write sink for the wrapper tests."""

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/__blank":
            body = b"<!doctype html><title>blank</title>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path.startswith("/__status"):
            # A write answered with a status of the caller's choice and no body (204, 205).
            code = int(parse_qs(urlsplit(self.path).query)["code"][0])
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            STATUS.append(code)
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path.startswith("/__slow"):
            # Answers late (``delay`` ms before the headers), then sends the body slowly (``ms``).
            query = parse_qs(urlsplit(self.path).query)
            SLOW.append(len(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
            time.sleep(int(query.get("delay", ["0"])[0]) / 1000)
            steps = max(1, int(query.get("ms", ["0"])[0]) // 100)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(steps * 1024))
            self.end_headers()
            try:
                for _ in range(steps):
                    self.wfile.write(b"x" * 1024)
                    self.wfile.flush()
                    time.sleep(0.1)
            except OSError:
                pass
            return
        if self.path == "/__sink":
            SINK.append(len(self.rfile.read(int(self.headers.get("Content-Length") or 0))))
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_POST()


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def port(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    saved = (CFG.raw, CFG.source)
    CFG.__init__({"nodes": [{"id": "a", "name": "spark-a", "ssh": "spark-a"}]})
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    STATE.lock = lock
    STATE.runner = _Quiet()
    SINK.clear()
    SLOW.clear()
    STATUS.clear()
    httpd = DualServer(("127.0.0.1", 0), _SinkHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()
        CFG.__init__(saved[0], source=saved[1])


def _node() -> Path:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    if node is None or not (node / "playwright").is_dir():
        if os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1":
            raise AssertionError("CI requires the WebKit page-leaving regression")
        pytest.skip("Playwright WebKit is not installed")
    return node


def _run(script: str, port: int, **extra: object) -> dict:
    env = dict(os.environ)
    env["SPARKSI_PORT"] = str(port)
    env.update({key: str(value) for key, value in extra.items()})
    env["NODE_PATH"] = str(_node())
    proc = subprocess.run(["node"], input=script, env=env, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0 and "Executable doesn't exist" in proc.stderr and os.environ.get("SPARKSI_WEBKIT_REQUIRED") != "1":
        pytest.skip("Playwright WebKit is not installed")
    if "__SKIP__" in proc.stdout:
        pytest.skip(proc.stdout.split("__SKIP__", 1)[1].strip())
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.split("__RESULT__", 1)[1].strip())


COMMON = """
const { webkit } = require('playwright');
const base = 'http://127.0.0.1:' + process.env.SPARKSI_PORT;
const holdMs = parseInt(process.env.HOLD_MS || '0', 10);
async function open(holdPath) {
  const browser = await webkit.launch();
  const page = await browser.newPage();
  const errors = [];
  const requests = [];
  const t0 = Date.now();
  page.on('pageerror', (e) => errors.push(String(e)));
  page.on('request', (r) => requests.push({ t: Date.now() - t0, method: r.method(), path: new URL(r.url()).pathname }));
  if (holdPath) await page.route((url) => url.pathname === holdPath, async (route) => {
    if (route.request().resourceType() === 'document') await new Promise((r) => setTimeout(r, holdMs));
    await route.continue().catch(() => {});
  });
  return { browser, page, errors, requests, t0 };
}
function done(result) { console.log('__RESULT__' + JSON.stringify(result)); }
"""

LEAVE = COMMON + """
(async () => {
  const { browser, page, errors } = await open('/models');
  await page.goto(base + '/overview');
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  await page.goto(base + '/models', { timeout: 60000 });
  await page.waitForSelector('#model-rows button.model-row');
  const path = await page.evaluate(() => location.pathname);
  done({ errors, path });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("hold_ms", [2500, 7500])
def test_polls_raise_no_page_error_while_the_page_is_leaving(port, hold_ms) -> None:
    data = _run(LEAVE, port, HOLD_MS=hold_ms)
    assert data["errors"] == []
    assert data["path"] == "/models"


STAY = COMMON + """
(async () => {
  const { browser, page, errors, requests, t0 } = await open(null);
  await page.addInitScript(() => { window.SPARKSI_LEAVING_HOLD_MS = 1500; });
  await page.goto(base + '/overview');
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  await page.waitForTimeout(1200);
  // A beforeunload with no navigation after it (a download link, say): GETs pause, then go on.
  const at = Date.now() - t0;
  await page.evaluate(() => window.dispatchEvent(new Event('beforeunload')));
  await page.waitForTimeout(4000);
  const monitor = requests.filter((r) => r.path === '/api/monitor');
  done({
    errors,
    during: monitor.filter((r) => r.t > at + 100 && r.t < at + 1400).length,
    after: monitor.filter((r) => r.t > at + 1500).length,
  });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_polls_go_on_when_no_navigation_follows_beforeunload(port) -> None:
    """Behaviour, not a repro: GETs wait out the hold, then carry on."""
    data = _run(STAY, port)
    assert data["errors"] == []
    assert data["during"] == 0, data
    assert data["after"] >= 2, data


WRITE = COMMON + """
(async () => {
  const { browser, page, errors, requests, t0 } = await open('/overview');
  await page.goto(base + '/models');
  await page.waitForSelector('#btn-lock:not([hidden])');
  const nav = page.waitForURL('**/overview', { timeout: 60000 });
  await page.evaluate(() => { location.href = '/overview'; });
  await page.waitForTimeout(500);
  // A write started while the page is leaving goes out at once.
  const at = Date.now() - t0;
  await page.evaluate(() => document.getElementById('btn-lock').click());
  await nav;
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  const lock = requests.filter((r) => r.method === 'POST' && r.path === '/api/lock');
  done({ errors, lockPosts: lock.length, sentAfterMs: lock.length ? lock[0].t - at : null });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_write_while_the_page_is_leaving_is_sent_not_dropped(port) -> None:
    data = _run(WRITE, port, HOLD_MS=2500)
    assert data["errors"] == []
    assert data["lockPosts"] == 1, data
    assert data["sentAfterMs"] < 1000, data
    assert STATE.lock.read().locked, "the write never reached the server"


CANCEL = COMMON + """
(async () => {
  const { browser, page, errors, requests, t0 } = await open('/models');
  await page.addInitScript(() => { window.SPARKSI_LEAVING_HOLD_MS = 1500; });
  await page.goto(base + '/overview');
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  // A navigation that is stopped before the next page arrives: the page stays.
  const at = Date.now() - t0;
  await page.evaluate(() => { location.href = '/models'; setTimeout(() => window.stop(), 1000); });
  await page.waitForTimeout(5000);
  const path = await page.evaluate(() => location.pathname);
  const after = requests.filter((r) => r.path === '/api/monitor' && r.t > at + 1600).length;
  done({ errors, path, after });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_a_stopped_navigation_raises_no_page_error_and_polls_go_on(port) -> None:
    data = _run(CANCEL, port, HOLD_MS=4000)
    assert data["errors"] == []
    assert data["path"] == "/overview", data
    assert data["after"] >= 2, data


STOPPED_WRITE = COMMON + """
(async () => {
  const { browser, page, errors, requests, t0 } = await open('/overview');
  await page.addInitScript(() => { window.SPARKSI_LEAVING_HOLD_MS = 1500; });
  await page.goto(base + '/models');
  await page.waitForSelector('#btn-lock:not([hidden])');
  // A navigation that is stopped: the page stays, and requests stay keepalive from here on.
  await page.evaluate(() => { location.href = '/overview'; setTimeout(() => window.stop(), 1000); });
  await page.waitForTimeout(2500);
  const at = Date.now() - t0;
  await page.evaluate(() => document.getElementById('btn-lock').click());
  await page.waitForFunction(() => !document.getElementById('btn-unlock').hidden, null, { timeout: 10000 });
  const path = await page.evaluate(() => location.pathname);
  const lock = requests.filter((r) => r.method === 'POST' && r.path === '/api/lock' && r.t >= at);
  done({ errors, path, lockPosts: lock.length });
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_writes_still_go_after_a_stopped_navigation(port) -> None:
    data = _run(STOPPED_WRITE, port, HOLD_MS=4000)
    assert data["errors"] == []
    assert data["path"] == "/models", data
    assert data["lockPosts"] == 1, data
    assert STATE.lock.read().locked, "the write never reached the server"


def test_the_one_write_with_a_signal_reads_an_abort_from_the_signal() -> None:
    text = PAGE.read_text(encoding="utf-8")
    timed = text.split("function jpostTimed", 1)[1].split("function jput", 1)[0]
    assert "ctrl.signal.aborted ||" in timed


def _wrapper() -> str:
    """The page's own leaving-page fetch wrapper, exactly as shipped."""
    text = PAGE.read_text(encoding="utf-8")
    return text.split("/* leaving-fetch-start */", 1)[1].split("/* leaving-fetch-end */", 1)[0]


BUDGET = """
const pw = require('playwright');
const base = 'http://127.0.0.1:' + process.env.SPARKSI_PORT;
(async () => {
  let browser;
  try { browser = await pw[process.env.ENGINE].launch(); } catch (e) {
    if (process.env.ENGINE !== 'webkit' && /Executable doesn't exist/.test(String(e))) { console.log('__SKIP__ ' + process.env.ENGINE + ' is not installed'); return; }
    throw e;
  }
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.goto(base + '/__blank');
  const out = await page.evaluate(async (code) => {
    window.SPARKSI_LEAVING_HOLD_MS = 300;
    const made = new Function(code + ';return { fetch: fetch, bytes: function () { return typeof keepaliveBytes === "number" ? keepaliveBytes : null; } };')();
    const send = made.fetch;
    const post = (body) => send('/__sink', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body })
      .then((r) => 'ok ' + r.status, (e) => 'refused ' + e.name);
    const ascii = (n) => JSON.stringify({ pad: 'a'.repeat(n) });
    window.dispatchEvent(new Event('beforeunload'));
    const result = {};
    result.multibyte = await post(JSON.stringify({ pad: String.fromCharCode(0x20ac).repeat(30000) }));
    result.big = await post(ascii(70000));
    const first = post(ascii(40000));
    result.inflight = [made.bytes()];
    const rest = [post(ascii(40000)), post(ascii(40000))];
    result.inflight.push(made.bytes());  // the other two would go over: sent as ordinary requests
    result.concurrent = await Promise.all([first].concat(rest));
    await new Promise((r) => setTimeout(r, 600));  // past the hold: sticky keepalive, no pageshow
    result.sequential = [];
    for (let i = 0; i < 5; i++) result.sequential.push(await post(ascii(50000)));
    result.small = await Promise.all([post(ascii(10)), post(ascii(10))]);
    await new Promise((r) => setTimeout(r, 300));
    result.bytesAfter = made.bytes();
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const slow = (query, signal) => send('/__slow?' + query, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: ascii(30000), signal });
    // Aborted before the answer: the browser keeps the request and its bytes until the server answers.
    let ac = new AbortController();
    const early = slow('delay=1500', ac.signal).then(() => 'resolved', (e) => e.name);
    await sleep(100);
    ac.abort();
    result.abortBefore = await early;
    result.bytesAfterAbort = made.bytes();
    result.afterAbort = await post(ascii(56000));
    await sleep(2500);
    result.bytesAfterAnswer = made.bytes();
    // Aborted while the body loads: the caller's read fails at once; the bytes go back when the body ends.
    ac = new AbortController();
    const mid = await slow('ms=1500', ac.signal);
    setTimeout(() => ac.abort(), 200);
    const t = Date.now();
    result.abortMidBody = await mid.text().then(() => 'read', (e) => e.name);
    result.abortMidBodyMs = Date.now() - t;
    result.bytesMidBody = made.bytes();
    await sleep(2000);
    result.bytesEnd = made.bytes();
    // A write with a signal answered 204 or 205: both browsers give those a body that is
    // empty but not null, and a rebuilt Response may not have one.
    result.nullBody = {};
    for (const code of [204, 205]) {
      ac = new AbortController();
      const answer = send('/__status?code=' + code, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: ascii(100), signal: ac.signal })
        .then((r) => r.text().then((t) => 'ok ' + r.status + ' [' + t + ']'), (e) => 'rejected ' + e.name);
      result.nullBody[code] = await Promise.race([answer, sleep(3000).then(() => 'no answer in 3 s')]);
    }
    // A signal aborted before the call: nothing is sent.
    ac = new AbortController();
    ac.abort();
    result.preAborted = await send('/__sink', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: ascii(100), signal: ac.signal })
      .then(() => 'sent', (e) => 'rejected ' + e.name);
    result.bytesAfterPreAborted = made.bytes();
    return result;
  }, process.env.WRAPPER);
  console.log('__RESULT__' + JSON.stringify(Object.assign({ errors }, out)));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_keepalive_writes_stay_inside_the_shared_byte_budget(port, engine) -> None:
    """Writes while leaving are never refused: the 64 KiB keepalive budget is bytes, shared by every request in flight."""
    data = _run(BUDGET, port, ENGINE=engine, WRAPPER=_wrapper())
    assert data["errors"] == []
    assert data["multibyte"] == "ok 200", data
    assert data["big"] == "ok 200", data
    assert data["concurrent"] == ["ok 200"] * 3, data
    assert data["sequential"] == ["ok 200"] * 5, data
    assert data["small"] == ["ok 200"] * 2, data
    assert data["inflight"] == [40010, 40010], data  # only the first 40 kB write counts against the budget
    assert data["bytesAfter"] == 0, data  # every keepalive byte is given back once its response has loaded
    assert sorted(SINK)[-1] >= 90000  # the multi-byte body arrived whole
    assert len(SINK) == 13, SINK  # 1 + 1 + 3 + 5 + 2 writes and the one after the abort; the pre-aborted one never went
    assert data["abortBefore"] == "AbortError", data
    assert data["bytesAfterAbort"] == 30010, data  # still held: the request runs on without its signal
    assert data["afterAbort"] == "ok 200", data  # 30 kB + 56 kB is over the budget: sent as an ordinary request
    assert data["bytesAfterAnswer"] == 0, data
    # The read fails at once. A body stream the page errors reads back as TypeError in both
    # browsers; jpostTimed, the one write that passes a signal, checks signal.aborted first.
    assert data["abortMidBody"] in ("AbortError", "TypeError"), data
    assert data["abortMidBodyMs"] < 1000, data
    assert data["bytesMidBody"] == 30010, data
    assert data["bytesEnd"] == 0, data
    assert SLOW == [30010, 30010], SLOW
    assert data["nullBody"] == {"204": "ok 204 []", "205": "ok 205 []"}, data
    assert STATUS == [204, 205], STATUS
    assert data["preAborted"] == "rejected AbortError", data
    assert data["bytesAfterPreAborted"] == 0, data  # and the sink count above shows nothing was sent
