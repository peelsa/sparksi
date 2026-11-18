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
from pathlib import Path

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


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
    httpd = DualServer(("127.0.0.1", 0), Handler)
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
