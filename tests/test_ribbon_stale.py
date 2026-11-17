# SPDX-License-Identifier: Apache-2.0

"""Ribbon pills dim and label themselves when a host report goes stale."""

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


def test_ribbon_stale_pill_state(ribbon_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit ribbon-stale regression")
        pytest.skip("Playwright WebKit is not installed")

    port = ribbon_server
    probe = """
    const { webkit } = require('playwright');
    (async () => {
      const errors = [];
      const browser = await webkit.launch();
      const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
      page.on('pageerror', e => errors.push(String(e)));
      const host = (name, fresh, age, gpu) => ({
        id: name, name: name, age: age, fresh: fresh,
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
        body: JSON.stringify({
          ok: true,
          loaded: [],
          hosts: [
            host('spark-a', true, 1, 12),
            host('spark-b', false, 95, 96)
          ]
        })
      }));
      await page.goto('http://127.0.0.1:PORT/overview');
      await page.waitForFunction(() => document.querySelectorAll('#ribbon .mem').length === 2);
      const pills = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('#ribbon .mem')).map(pill => {
          const age = pill.querySelector('.age');
          const dials = pill.querySelector('.dials');
          return {
            role: pill.getAttribute('data-role'),
            className: pill.className,
            stale: pill.classList.contains('is-stale'),
            ageText: age ? age.textContent : null,
            ageHidden: age ? age.hidden : null,
            borderStyle: getComputedStyle(pill).borderTopStyle,
            dialsOpacity: dials ? getComputedStyle(dials).opacity : null,
            valColor: getComputedStyle(pill.querySelector('.val')).color,
            mutedColor: getComputedStyle(pill.querySelector('.host')).color,
            ariaLabel: pill.getAttribute('aria-label') || ''
          };
        });
      });
      const shots = process.env.SPARKSI_SHOT_DIR || '';
      if (shots) await page.locator('header.brandbar').screenshot({ path: shots + '/ribbon-stale.png' });
      await browser.close();
      console.log('__RESULT__' + JSON.stringify({ errors, pills }));
    })().catch(e => { console.error(e); process.exit(1); });
    """.replace("PORT", str(port))

    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError("webkit probe failed:\nSTDOUT:\n%s\nSTDERR:\n%s" % (proc.stdout, proc.stderr))
    lines = [line for line in proc.stdout.splitlines() if line.startswith("__RESULT__")]
    assert lines, "no result marker in probe output:\n%s" % proc.stdout
    result = json.loads(lines[-1][len("__RESULT__"):])
    assert result["errors"] == []
    pills = {pill["role"]: pill for pill in result["pills"]}
    fresh = pills["head"]
    stale = pills["worker"]
    assert not fresh["stale"]
    assert "is-stale" not in fresh["className"]
    assert fresh["ageHidden"] is True
    assert fresh["borderStyle"] != "dashed"
    assert stale["stale"] is True
    assert "is-stale" in stale["className"]
    assert "Stale" in stale["ageText"]
    assert "95" in stale["ageText"]
    assert stale["ageHidden"] is False
    assert stale["borderStyle"] == "dashed"
    assert stale["dialsOpacity"] == "0.5"
    assert stale["valColor"] == stale["mutedColor"], stale  # numbers in the muted grey
    assert fresh["valColor"] != fresh["mutedColor"], fresh
    assert stale["ariaLabel"].startswith("stale")
    assert not fresh["ariaLabel"].startswith("stale")
