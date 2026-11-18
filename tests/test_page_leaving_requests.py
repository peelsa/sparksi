# SPDX-License-Identifier: Apache-2.0
"""Leaving a page must not raise page errors from its polls.

Once a navigation away starts (``beforeunload``), WebKit keeps the old page's
timers running until the new page arrives, but refuses every request the old
page starts in that window. Each refusal is logged as "Fetch API cannot load
... due to access control checks." and Playwright reports it as a page error.
A busy machine stretches the window, so the 1 s polls landed in it now and
then. These tests hold the next page's document so the window is long enough
for every poll to fire in it.
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


def _run(script: str, port: int) -> dict:
    env = dict(os.environ)
    env["SPARKSI_PORT"] = str(port)
    env["NODE_PATH"] = str(_node())
    proc = subprocess.run(["node"], input=script, env=env, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0 and "Executable doesn't exist" in proc.stderr and os.environ.get("SPARKSI_WEBKIT_REQUIRED") != "1":
        pytest.skip("Playwright WebKit is not installed")
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.split("__RESULT__", 1)[1].strip())


LEAVE = """
const { webkit } = require('playwright');
(async () => {
  const base = 'http://127.0.0.1:' + process.env.SPARKSI_PORT;
  const browser = await webkit.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  // Hold the next page's document, so the old page's polls all fire while it is leaving.
  await page.route((url) => url.pathname === '/models', async (route) => {
    if (route.request().resourceType() === 'document') await new Promise((r) => setTimeout(r, 2500));
    await route.continue();
  });
  await page.goto(base + '/overview');
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  await page.goto(base + '/models');
  await page.waitForSelector('#model-rows button.model-row');
  const path = await page.evaluate(() => location.pathname);
  console.log('__RESULT__' + JSON.stringify({ errors, path }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_polls_raise_no_page_error_while_the_page_is_leaving(port) -> None:
    data = _run(LEAVE, port)
    assert data["errors"] == []
    assert data["path"] == "/models"


STAY = """
const { webkit } = require('playwright');
(async () => {
  const base = 'http://127.0.0.1:' + process.env.SPARKSI_PORT;
  const browser = await webkit.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(String(e)));
  await page.addInitScript(() => { window.SPARKSI_LEAVING_HOLD_MS = 1500; });
  const monitor = [];
  page.on('request', (r) => { if (new URL(r.url()).pathname === '/api/monitor') monitor.push(Date.now()); });
  await page.goto(base + '/overview');
  await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 1);
  await page.waitForTimeout(1200);
  // A beforeunload with no navigation after it (a download link, say): polls pause, then go on.
  const t0 = Date.now();
  await page.evaluate(() => window.dispatchEvent(new Event('beforeunload')));
  await page.waitForTimeout(4000);
  const during = monitor.filter((t) => t > t0 + 100 && t < t0 + 1400).length;
  const after = monitor.filter((t) => t > t0 + 1500).length;
  console.log('__RESULT__' + JSON.stringify({ errors, during, after }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_polls_go_on_when_no_navigation_follows_beforeunload(port) -> None:
    data = _run(STAY, port)
    assert data["errors"] == []
    assert data["during"] == 0, data
    assert data["after"] >= 2, data
