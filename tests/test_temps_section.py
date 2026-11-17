# SPDX-License-Identifier: Apache-2.0

"""Temps lives in a collapsed details on Overview; /temps redirects there."""

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
from webkit_noise import WATCH_JS, real_nav_errors

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def temps_server(monkeypatch, tmp_path):
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


def test_temps_collapsed_section_and_redirect(temps_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit temps-section regression")
        pytest.skip("Playwright WebKit is not installed")

    port = temps_server
    probe = WATCH_JS + """
    const { webkit } = require('playwright');
    (async () => {
      const out = {};
      const browser = await webkit.launch();
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
      const shots = process.env.SPARKSI_SHOT_DIR || '';
      const moving = watchErrors(page, out);
      const host = (role, gpu) => ({
        id: role, name: role, age: 1, fresh: true,
        mem_used_gib: 32, mem_total_gib: 128, mem_pct: 25,
        cpu_pct: 10, gpu_pct: gpu, gpu_temp_c: 45, power_w: 40,
        zones: [
          { label: 'gpu', c: 40 },
          { label: 'nvme', c: 50 }
        ]
      });
      await page.route('**/api/monitor*', route => route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ ok: true, loaded: [], hosts: [host('spark-a', 12), host('spark-b', 96)] })
      }));
      await moving(() => page.goto('http://127.0.0.1:PORT/overview'));
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      const closed = await page.evaluate(() => {
        const section = document.getElementById('temps-section');
        const zones = document.getElementById('zones');
        const next = section ? section.nextElementSibling : null;
        const navTemps = !!Array.from(document.querySelectorAll('nav.side a')).find(a => a.getAttribute('href') === '/temps');
        return {
          exists: !!section,
          open: section ? !!section.open : null,
          zonesChildren: zones ? zones.children.length : null,
          beforeJson: !!(next && next.tagName === 'DETAILS' &&
            next.querySelector('summary') &&
            next.querySelector('summary').textContent.trim() === 'JSON'),
          navTemps: navTemps
        };
      });
      if (shots) await page.screenshot({ path: shots + '/temps-collapsed.png', fullPage: true });
      await page.click('#temps-section > summary');
      await page.waitForFunction(() => {
        const z = document.getElementById('zones');
        return !!z && z.children.length > 0;
      });
      const opened = await page.evaluate(() => {
        const section = document.getElementById('temps-section');
        const zones = document.getElementById('zones');
        return {
          open: !!section.open,
          lines: Array.from(zones.querySelectorAll('p')).map(p => p.textContent)
        };
      });
      if (shots) await page.screenshot({ path: shots + '/temps-expanded.png', fullPage: true });
      await moving(() => page.goto('http://127.0.0.1:PORT/temps'));
      await page.waitForLoadState('load');
      const redirect = await page.evaluate(() => {
        const section = document.getElementById('temps-section');
        return {
          path: location.pathname,
          open: section ? !!section.open : null,
          viewTemps: !!document.getElementById('view-temps')
        };
      });
      await moving(() => page.goto('http://127.0.0.1:PORT/models'));
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      const models = await page.evaluate(() => {
        const section = document.getElementById('temps-section');
        return { hidden: !section || section.hidden || section.offsetParent === null };
      });
      await browser.close();
      console.log('__RESULT__' + JSON.stringify({ errors: out.errors, navErrors: out.navErrors, closed, opened, redirect, models }));
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
    assert real_nav_errors(payload["navErrors"]) == [], payload["navErrors"]
    closed = payload["closed"]
    opened = payload["opened"]
    redirect = payload["redirect"]
    models = payload["models"]

    assert errors == []
    assert closed["exists"] is True
    assert closed["open"] is False
    assert closed["zonesChildren"] == 0
    assert closed["beforeJson"] is True
    assert closed["navTemps"] is False
    assert opened["open"] is True
    assert "spark-a gpu 104 F" in opened["lines"]
    assert "spark-b nvme 122 F" in opened["lines"]
    assert redirect["path"] == "/overview"
    assert redirect["open"] is True
    assert redirect["viewTemps"] is False
    assert models["hidden"] is True
