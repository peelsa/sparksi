# SPDX-License-Identifier: Apache-2.0

"""Reloading any route opens that route, in WebKit at 400 px, with and without the script."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.lock import Lock
from sparksi.serve import _PAGES, STATE, DualServer, Handler
from webkit_noise import WATCH_JS, real_nav_errors

ROOT = Path(__file__).resolve().parents[1]

PROBE = WATCH_JS + r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const routes = JSON.parse(process.env.ROUTES);
  const browser = await webkit.launch();
  const out = { errors: [], rows: [] };
  const look = (page) => page.evaluate(() => ({
    views: Array.from(document.querySelectorAll(".view")).filter((v) => !v.hidden && v.offsetParent !== null).map((v) => v.id),
    current: (document.querySelector('.side a[aria-current="page"]') || {}).getAttribute
      ? document.querySelector('.side a[aria-current="page"]').getAttribute("href") : null,
    path: location.pathname,
  }));
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
    const moving = watchErrors(page, out);
    for (const route of routes) {
      await moving(() => page.goto(base + route, { waitUntil: "load" }));
      await page.waitForTimeout(150);
      const first = await look(page);
      await moving(() => page.reload({ waitUntil: "load" }));
      await page.waitForTimeout(150);
      const again = await look(page);
      out.rows.push({ route, first, again });
    }
    // A stalled monitor polled every second keeps one request in flight; once that
    // request is over 3 s old one new request starts, not a pile.
    const slow = await browser.newPage({ viewport: { width: 400, height: 860 } });
    const monitorAt = [];
    await slow.route("**/api/monitor", async (route) => {
      monitorAt.push(Date.now());
      await new Promise((r) => setTimeout(r, 9000));
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ hosts: [] }) }).catch(() => {});
    });
    await slow.goto(base + "/models", { waitUntil: "load" });
    const t0 = Date.now();
    while (!monitorAt.length && Date.now() - t0 < 5000) await slow.waitForTimeout(25);
    await slow.waitForTimeout(2500);
    out.monitorEarly = monitorAt.length;  // 2.5 s after the first: still one
    await slow.waitForTimeout(2500);
    out.monitorGaps = monitorAt.slice(1).map((t, i) => t - monitorAt[i]);
    out.monitorHits = monitorAt.length;  // 5 s after the first: one more
    await slow.close();
    const plain = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 400, height: 860 } });
    const bare = await plain.newPage();
    out.noScript = [];
    for (const route of routes) {
      await bare.goto(base + route, { waitUntil: "load" });
      await bare.reload({ waitUntil: "load" });
      out.noScript.push({ route, views: await bare.evaluate(() => Array.from(document.querySelectorAll(".view")).filter((v) => !v.hidden).map((v) => v.id)) });
    }
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture()
def page_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_reloading_each_route_keeps_that_route(page_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit route reload regression")
        pytest.skip("Playwright WebKit is not installed")
    routes = sorted(_PAGES)
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(page_server), ROUTES=json.dumps(routes))
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=240)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, json.dumps(got, indent=1)
    assert got["errors"] == [], got
    assert real_nav_errors(got.get("navErrors", [])) == [], got
    assert [r["route"] for r in got["rows"]] == routes
    assert got["monitorEarly"] == 1, got  # polled each second, but one in flight
    assert got["monitorHits"] == 2, got  # once that one is stuck past 3 s, one new request
    assert all(2900 <= gap <= 4600 for gap in got["monitorGaps"]), got
    for row in got["rows"]:
        route = row["route"]
        want = "view-overview" if route in ("/", "/overview") else "view-" + route.strip("/")
        for seen in (row["first"], row["again"]):
            assert seen["path"] == route, row
            assert seen["views"] == [want], row
            assert seen["current"] == ("/overview" if route == "/" else route), row
    for row in got["noScript"]:
        route = row["route"]
        want = "view-overview" if route in ("/", "/overview") else "view-" + route.strip("/")
        assert row["views"] == [want], row
