# SPDX-License-Identifier: Apache-2.0

"""Shared ribbon on every page; Cluster has no big dials; Model Manager has one Load action."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def ribbon_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
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


def test_ribbon_on_every_page_and_cluster_layout(ribbon_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit ribbon regression")
        pytest.skip("Playwright WebKit is not installed")

    port = ribbon_server
    probe = """
    const { webkit } = require('playwright');
    (async () => {
      const errors = [];
      const browser = await webkit.launch();
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
      const shots = process.env.SPARKSI_SHOT_DIR || '';
      page.on('pageerror', e => errors.push(String(e)));
      const host = (role, gpu) => ({
        id: role, name: role, age: 1, fresh: true,
        mem_used_gib: 32, mem_total_gib: 128, mem_pct: 25,
        cpu_pct: 10, gpu_pct: gpu, gpu_temp_c: 45, power_w: 40, zones: []
      });
      await page.route('**/api/monitor*', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, loaded: [], hosts: [host('spark-a', 12), host('spark-b', 96)] })
      }));
      await page.goto('http://127.0.0.1:PORT/overview');
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      await page.waitForFunction(() => document.querySelector('#ribbon').innerText.includes('96%'));
      const cluster = await page.evaluate(() => {
        const pills = document.querySelectorAll('#ribbon .mem').length;
        const ribbonText = document.querySelector('#ribbon').innerText;
        const gauges = document.querySelectorAll('#ribbon .mem figure').length;
        const temp = Array.from(document.querySelectorAll('#ribbon .mem .temp')).map(e => e.textContent);
        const val = Array.from(document.querySelectorAll('#ribbon .mem .val')).map(e => e.textContent);
        const bigDials = document.querySelectorAll('#view-overview .gauge-group').length;
        const gaugesBox = !!document.getElementById('gauges');
        const chart = !!document.getElementById('fleet-chart');
        const high = document.body.innerText.indexOf('High') >= 0;
        const link = Array.from(document.querySelectorAll('nav.side a')).find(a => a.getAttribute('href').endsWith('/models'));
        const nav = link ? { text: link.textContent.trim(), href: link.getAttribute('href') } : null;
        const firstMem = document.querySelector('#ribbon .mem');
        const firstVal = document.querySelector('#ribbon .mem .val');
        const firstHost = document.querySelector('#ribbon .mem .host');
        const firstState = document.querySelector('#ribbon .mem figure .state');
        const gaugeRect = firstMem ? firstMem.querySelector('figure').getBoundingClientRect() : null;
        const tempRect = firstMem ? firstMem.querySelector('.temp').getBoundingClientRect() : null;
        const valRect = firstVal ? firstVal.getBoundingClientRect() : null;
        const size = {
          gauge: gaugeRect ? Math.round(gaugeRect.width) : null,
          val: firstVal ? getComputedStyle(firstVal).fontSize : null,
          host: firstHost ? getComputedStyle(firstHost).fontSize : null,
          state: firstState ? getComputedStyle(firstState).fontSize : null,
          gap: (tempRect && valRect) ? Math.round(tempRect.left - valRect.right) : null,
          pillPad: firstMem ? getComputedStyle(firstMem).paddingTop : null
        };
        return { pills, ribbonText, gauges, temp, val, bigDials, gaugesBox, chart, high, nav, size };
      });
      if (shots) await page.screenshot({ path: shots + '/built-cluster.png', fullPage: true });
      await page.goto('http://127.0.0.1:PORT/models');
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      await page.waitForSelector('#models-h');
      const mm = await page.evaluate(() => {
        const pills = document.querySelectorAll('#ribbon .mem').length;
        const heading = document.querySelector('#models-h').textContent;
        const loadButtons = Array.from(document.querySelectorAll('#model-setup button')).filter(b => {
          const t = (b.textContent || '').trim();
          return t === 'Load' || t.startsWith('Swap');
        }).length;
        const high = document.body.innerText.indexOf('High') >= 0;
        return { pills, heading, loadButtons, high };
      });
      if (shots) await page.screenshot({ path: shots + '/built-model-manager.png', fullPage: true });
      await page.setViewportSize({ width: 390, height: 844 });
      await page.goto('http://127.0.0.1:PORT/overview');
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      const mobile = await page.evaluate(() => {
        const docW = document.documentElement.scrollWidth;
        const overflow = Array.from(document.querySelectorAll('#ribbon .mem')).filter(e => e.scrollWidth > e.clientWidth + 1).length;
        const valWrapped = Array.from(document.querySelectorAll('#ribbon .mem .val')).filter(e => e.getBoundingClientRect().height > 30).length;
        return { docW, overflow, valWrapped };
      });
      if (shots) await page.locator('header.brandbar').screenshot({ path: shots + '/built-cluster-390.png' });
      await browser.close();
      console.log('__RESULT__' + JSON.stringify({ errors, cluster, mm, size: cluster.size, mobile }));
    })().catch(e => { console.error(e); process.exit(1); });
    """.replace("PORT", str(port))

    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)

    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("__RESULT__")]
    assert lines, proc.stdout
    payload = json.loads(lines[-1][len("__RESULT__"):])

    errors = payload["errors"]
    cluster = payload["cluster"]
    mm = payload["mm"]
    size = payload["size"]
    mobile = payload["mobile"]

    assert errors == []
    assert cluster["pills"] == 2
    assert mm["pills"] == 2
    assert cluster["gauges"] == 4
    for val in cluster["val"]:
        assert val == "32.0 / 128.0 GiB"
    for temp in cluster["temp"]:
        assert temp == "113 F"
    assert "96%" in cluster["ribbonText"]
    assert cluster["bigDials"] == 0
    assert cluster["gaugesBox"] is False
    assert cluster["chart"] is True
    assert cluster["high"] is False
    assert mm["high"] is False
    assert cluster["nav"]["text"] == "Model Manager"
    assert cluster["nav"]["href"].endswith("/models")
    assert mm["heading"] == "Model Manager"
    assert mm["loadButtons"] == 1
    assert size["gauge"] == 55
    assert size["val"] == "16.25px"
    assert size["host"] == "15px"
    assert size["state"] == "11.25px"
    assert size["pillPad"] == "5px"
    assert 0 <= size["gap"] <= 12
    assert mobile["docW"] <= 390
    assert mobile["overflow"] == 0
    assert mobile["valWrapped"] == 0
