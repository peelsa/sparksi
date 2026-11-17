# SPDX-License-Identifier: Apache-2.0

"""N-Spark page regression: ribbon, serving boxes, library, worker, placement."""

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


@pytest.fixture(params=[1, 3, 4])
def site(request, monkeypatch):
    """A site with ``n`` Sparks named spark-a, spark-b, ... in node order."""
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    n = request.param
    saved = (CFG.raw, CFG.source)
    CFG.__init__({"nodes": [{"id": x, "name": "spark-" + x, "ssh": "spark-" + x} for x in "abcd"[:n]]})
    try:
        yield n
    finally:
        CFG.__init__(saved[0], source=saved[1])


@pytest.fixture()
def server(site, monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
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
        yield site, httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()


def test_page_follows_spark_count(server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit n-spark regression")
        pytest.skip("Playwright WebKit is not installed")

    n, port = server

    script = """
    const { webkit } = require('playwright');
    (async () => {
      const N = parseInt(process.env.SPARKSI_N, 10);
      const host = (name) => ({
        id: name, name: name, age: 1, fresh: true,
        mem_used_gib: 32, mem_total_gib: 128, mem_pct: 25,
        cpu_pct: 10, gpu_pct: 20, gpu_temp_c: 45, power_w: 40, zones: []
      });
      const names = [];
      for (let i = 0; i < N; i++) names.push('spark-' + 'abcd'[i]);
      const browser = await webkit.launch();
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror', (e) => errors.push(String(e)));
      await page.route('**/api/monitor*', (route) =>
        route.fulfill({ status: 200, contentType: 'application/json',
          body: JSON.stringify({ ok: true, loaded: [], hosts: names.map(host) }) }));
      await page.goto('http://127.0.0.1:' + process.env.SPARKSI_PORT + '/overview');
      await page.waitForFunction((want) => document.querySelectorAll('#ribbon .mem').length === want, N);
      await page.waitForFunction((want) => document.querySelectorAll('#serving-lines .serve-box').length === want, N);
      const ribbon = await page.$$eval('#ribbon .mem', (els) => els.length);
      const serveBoxes = await page.$$eval('#serving-lines .serve-box', (els) => els.length);
      const libTh = await page.$$eval('#library thead th', (els) => els.length);
      const workerHidden = await page.$eval('#machine-worker', (el) => el.hidden);
      const shots = process.env.SPARKSI_SHOT_DIR || '';
      if (shots) await page.screenshot({ path: shots + '/sparks-' + N + '.png', fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
      const docW = await page.evaluate(() => document.documentElement.scrollWidth);
      if (shots) await page.screenshot({ path: shots + '/sparks-' + N + '-390.png' });
      await page.setViewportSize({ width: 1280, height: 800 });
      await page.goto('http://127.0.0.1:' + process.env.SPARKSI_PORT + '/models');
      await page.waitForSelector('#model-rows button.model-row');
      await page.click('#model-rows button.model-row');
      await page.waitForSelector('#model-placement input');
      const placement = await page.$$eval('#model-placement input', (els) =>
        els.map((el) => ({value: el.value, disabled: el.disabled, text: el.parentElement.textContent})));
      console.log('__RESULT__' + JSON.stringify({ribbon, serveBoxes, libTh, workerHidden, placement, docW, errors}));
      await browser.close();
    })().catch((e) => { console.error(e); process.exit(1); });
    """

    env = dict(os.environ)
    env["SPARKSI_N"] = str(n)
    env["SPARKSI_PORT"] = str(port)
    env["NODE_PATH"] = str(node)
    proc = subprocess.run(["node"], input=script, env=env, capture_output=True, text=True, timeout=120, check=False)
    if proc.returncode != 0 and "Executable doesn't exist" in proc.stderr and not required:
        pytest.skip("Playwright WebKit is not installed")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.split("__RESULT__", 1)[1].strip())

    assert data["errors"] == []
    assert data["docW"] <= 390, data["docW"]
    assert data["ribbon"] == n
    assert data["serveBoxes"] == n
    assert data["libTh"] == 3 + n
    assert data["workerHidden"] == (n == 1)
    roles = ["head", "worker", "node3", "node4"][:n]
    values = [p["value"] for p in data["placement"]]
    assert values[:n] == roles
    assert "both" in values
    if n == 1:
        both = next(p for p in data["placement"] if p["value"] == "both")
        assert both["disabled"]
        assert "needs 2 Sparks" in both["text"]
    if n >= 3:
        assert any("+" in v for v in values)
