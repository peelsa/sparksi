# SPDX-License-Identifier: Apache-2.0

"""The fleet chart's ResizeObserver must not resize its own box inside its callback.

The chart box sits in a grid next to the legend, and a repaint writes the legend's
readouts. A repaint inside the observer callback therefore changed the observed box
in the same frame, and the browser logged ``ResizeObserver loop completed with
undelivered notifications.`` as a page error (the flake in
``test_page_follows_spark_count[4]``, 4 Sparks have the tallest legend). The callback
now repaints on the next frame. Every page error still fails here; nothing is exempt.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest
from test_n_sparks_page import server, site  # noqa: F401  (fixtures: 1 to 4 Sparks, local server)

SCRIPT = """
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
  await page.addInitScript(() => {
    window.__ro = { made: 0, callbacks: [] };
    const Real = window.ResizeObserver;
    window.ResizeObserver = class extends Real {
      constructor(cb) {
        super(cb);
        window.__ro.made += 1;
        window.__ro.callbacks.push([cb, this]);
      }
    };
  });
  await page.route('**/api/monitor*', (route) =>
    route.fulfill({ status: 200, contentType: 'application/json',
      body: JSON.stringify({ ok: true, loaded: [], hosts: names.map(host) }) }));
  await page.goto('http://127.0.0.1:' + process.env.SPARKSI_PORT + '/overview');
  await page.waitForFunction((want) => document.querySelectorAll('#ribbon .mem').length === want, N);
  await page.waitForSelector('#fleet-legend .legend-value');
  await page.waitForTimeout(500);

  // 1. The callback does not repaint inside itself; the repaint lands on a later frame.
  const defer = await page.evaluate(async () => {
    const value = document.querySelector('#fleet-legend .legend-value');
    const [cb, observer] = window.__ro.callbacks[window.__ro.callbacks.length - 1];
    value.textContent = 'SENTINEL';
    cb.call(observer, [], observer);
    cb.call(observer, [], observer);
    const inside = value.textContent;
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return { inside, after: value.textContent, made: window.__ro.made };
  });

  // 2. Many real size changes in a row raise no page error.
  const widths = [1280, 900, 390, 1100, 600, 1280];
  for (let round = 0; round < 6; round++) {
    for (const w of widths) {
      await page.setViewportSize({ width: w, height: 800 });
      await page.waitForTimeout(30);
    }
  }
  await page.waitForTimeout(300);
  console.log('__RESULT__' + JSON.stringify({ errors, defer }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_chart_observer_repaints_on_the_next_frame_and_raises_no_page_error(server) -> None:  # noqa: F811
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit chart resize regression")
        pytest.skip("Playwright WebKit is not installed")

    n, port = server
    env = dict(os.environ)
    env["SPARKSI_N"] = str(n)
    env["SPARKSI_PORT"] = str(port)
    env["NODE_PATH"] = str(node)
    proc = subprocess.run(["node"], input=SCRIPT, env=env, capture_output=True, text=True, timeout=180, check=False)
    if proc.returncode != 0 and "Executable doesn't exist" in proc.stderr and not required:
        pytest.skip("Playwright WebKit is not installed")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.split("__RESULT__", 1)[1].strip())

    assert data["errors"] == [], data["errors"]
    assert data["defer"]["made"] >= 1
    assert data["defer"]["inside"] == "SENTINEL", "the observer callback repainted inside itself"
    assert data["defer"]["after"] != "SENTINEL", "the observer callback never repainted"
