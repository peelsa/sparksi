# SPDX-License-Identifier: Apache-2.0

"""After Unload, the head card stops showing the run within a couple of seconds.

The first run read after the stop still carries the run (the server answers from
before the write); the page reads again soon after instead of waiting for the next
5 s poll.
"""

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
def follow_server(monkeypatch, tmp_path):
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


PROBE = r"""
const { webkit } = require("playwright");
const port = process.env.RECIPE_PORT;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  const running = {runs:[{id:"r1",model_id:"single-a",phase:"ready",port:8013,placement:"head",started:1,nodes:[{host:"spark-a",role:"head"}]}]};
  const stopped = {runs:[{id:"r1",model_id:"single-a",phase:"stopped",port:8013,placement:"head",started:1,nodes:[{host:"spark-a",role:"head"}]}]};
  let stoppedAt = 0;
  let readsAfterStop = 0;
  await page.route("**/api/runs", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    let body = running;
    if (stoppedAt) {
      readsAfterStop += 1;
      if (readsAfterStop > 1) body = stopped;  // the first read after the stop is from before it
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.route("**/api/runs/r1/stop", async (route) => {
    stoppedAt = Date.now();
    await route.fulfill({ status: 200, contentType: "application/json", body: '{"ok":true}' });
  });
  const shows = () => page.evaluate(() => (document.getElementById("model-status") || {}).textContent || "");
  await page.goto("http://127.0.0.1:" + port + "/models");
  await page.getByRole("button", { name: /Single A/ }).click();
  await page.waitForSelector("#model-port");
  await page.waitForFunction(() => /port 8013/.test((document.getElementById("model-status") || {}).textContent || ""));
  const before = await shows();
  await page.evaluate(() => document.getElementById("model-unload").click());
  await page.waitForFunction(() => document.getElementById("model-confirm").open);
  await page.evaluate(() => document.getElementById("model-confirm-ok").click());
  const t0 = Date.now();
  let cleared = null;
  while (Date.now() - t0 < 6000) {
    if (stoppedAt && !/port 8013/.test(await shows())) { cleared = Date.now() - stoppedAt; break; }
    await sleep(50);
  }
  const after = await shows();
  await browser.close();
  console.log(JSON.stringify({ errors, before, after, cleared, readsAfterStop }));
})().catch((err) => { console.error(String(err && err.stack || err)); process.exit(1); });
"""


def test_head_card_catches_up_soon_after_unload(follow_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit header follow-up regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(follow_server)
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == [], got
    assert "port 8013" in got["before"], got
    assert got["cleared"] is not None, got
    assert got["cleared"] < 2500, got  # the 1.5 s follow-up, not the next 5 s poll
    assert "port 8013" not in got["after"], got
