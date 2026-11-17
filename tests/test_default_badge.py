# SPDX-License-Identifier: Apache-2.0

"""After Set as default, the recipe list moves the Default badge at once."""

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
from sparksi.recipes.store import save_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def badge_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "gpu_mem_util": 0.55})
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
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
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  const menu = async (id) => { if (["recipe-cmd-new", "recipe-cmd-save", "recipe-cmd-save-as"].indexOf(id) === -1) await page.click("#recipe-files tr.is-open .recipe-row-more"); await page.click("#" + id); };
  const badges = () => page.$$eval("#recipe-files .recipe-file-row", (els) => els.map((el) => ({
    name: el.querySelector(".recipe-file-name").textContent,
    def: el.querySelector(".recipe-file-default").textContent
  })));
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
  await page.selectOption("#field-ctx", "32768");
  await menu("recipe-cmd-save-as");
  await page.fill("#recipe-name", "long-ctx");
  await page.click("#recipe-name-ok");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "long-ctx");
  /* Hold the list refresh, so the page has only the Set as default reply to go on. */
  await page.route("**/api/models/pair-fast/recipes", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    await new Promise((r) => setTimeout(r, 4000));
    return route.continue();
  });
  await menu("recipe-cmd-default");
  await page.waitForFunction(() => document.getElementById("recipe-note").textContent.indexOf("is now the default") !== -1);
  await page.click("#recipe-files tr.is-open .recipe-row-more");
  const after = await badges();
  const cmds = await page.evaluate(() => ({
    setDefault: document.getElementById("recipe-cmd-default").disabled,
    del: document.getElementById("recipe-cmd-delete").disabled
  }));
  await browser.close();
  console.log(JSON.stringify({ errors, after, cmds }));
})().catch((err) => { console.error(String(err && err.stack || err)); process.exit(1); });
"""


def test_default_badge_moves_after_set_as_default(badge_server) -> None:
    port = badge_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit default-badge regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    after = {row["name"]: row["def"] for row in got["after"]}
    assert after == {"default": "", "long-ctx": "Default"}, got
    assert got["cmds"] == {"setDefault": True, "del": True}, got
