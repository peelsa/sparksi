# SPDX-License-Identifier: Apache-2.0

"""Load default recipe restores the selected model's saved values and does not start anything."""

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
from sparksi.recipes.store import load_recipe, save_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def recipe_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "gpu_mem_util": 0.55})
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


def test_load_default_recipe_restores_the_selected_model(recipe_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit load-default regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const port = process.env.RECIPE_PORT;
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  const methods = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", (dialog) => dialog.accept());
  page.on("request", (req) => {
    const url = req.url();
    if (url.includes("/recipe") || url.includes("/plan") || url.includes("/runs")) methods.push(req.method() + " " + url.split(port)[1]);
  });
  const held = [];
  let hold = "";
  await page.route("**/api/models/*/recipe*", async (route) => {
    const url = route.request().url();
    const query = url.includes("?") ? url.slice(url.indexOf("?")) : "";
    if (hold === "query" && query && url.includes("/pair-fast/")) {
      held.push(route);
      return;
    }
    if (hold === "bare" && !query && url.includes("/pair-fast/recipe")) {
      held.push(route);
      return;
    }
    if (hold === "bad" && !query && url.includes("/pair-fast/recipe")) {
      hold = "";
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ id: "pair-fast", defaults: "bad", engine: "vllm", port: 8099 })
      });
      return;
    }
    await route.continue();
  });
  async function untilHeld(count) {
    for (let i = 0; i < 100; i++) {
      if (held.length >= count) return;
      await page.waitForTimeout(20);
    }
    throw new Error("recipe request was not held");
  }
  function readForm() {
    return page.evaluate(() => ({
      title: document.getElementById("model-title").textContent,
      ctx: document.getElementById("field-ctx").value,
      port: document.getElementById("model-port").value,
      source: document.getElementById("model-default-source").textContent
    }));
  }
  await page.goto("http://127.0.0.1:" + port + "/models", { waitUntil: "networkidle" });
  await page.getByRole("button", { name: /Pair Fast/ }).click();
  await page.waitForSelector("#field-ctx");
  hold = "query";
  await page.selectOption("#field-ctx", "32768");
  await page.fill("#model-port", "8016");
  await untilHeld(1);
  await page.click("#model-load-default");
  await page.waitForFunction(() => document.getElementById("field-ctx").value === "4096" && document.getElementById("model-port").value === "8012");
  const beforeLatePreview = await readForm();
  await held.shift().continue();
  await page.waitForTimeout(400);
  const afterLatePreview = await readForm();
  hold = "bare";
  await page.selectOption("#field-ctx", "32768");
  await page.click("#model-load-default");
  await untilHeld(1);
  await page.getByRole("button", { name: /Single A/ }).click();
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Single A") !== -1 && document.getElementById("model-default-source").textContent.indexOf("engine defaults") !== -1);
  const switched = await readForm();
  await held.shift().continue();
  await page.waitForTimeout(400);
  const afterLateDefaults = await readForm();
  await page.getByRole("button", { name: /Pair Fast/ }).click();
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Pair Fast") !== -1 && document.getElementById("field-ctx").value === "32768");
  const back = await readForm();
  await page.selectOption("#field-ctx", "32768");
  hold = "bad";
  await page.click("#model-load-default");
  await page.waitForFunction(() => document.getElementById("model-default-source").textContent.indexOf("No default") !== -1);
  const failed = await readForm();
  await page.getByRole("button", { name: /Single A/ }).click();
  await page.waitForTimeout(300);
  await page.getByRole("button", { name: /Pair Fast/ }).click();
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Pair Fast") !== -1);
  await page.waitForTimeout(400);
  const roundTrip = await readForm();
  console.log(JSON.stringify({ errors, beforeLatePreview, afterLatePreview, switched, afterLateDefaults, back, failed, roundTrip, methods }));
  await browser.close();
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(recipe_server)
    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if browsers:
        env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["beforeLatePreview"]["ctx"] == "4096"
    assert got["beforeLatePreview"]["port"] == "8012"
    assert got["beforeLatePreview"]["source"] == "from saved recipe"
    assert got["afterLatePreview"] == got["beforeLatePreview"]
    assert got["switched"]["title"] == "Single A"
    assert got["switched"]["ctx"] == "8192"
    assert got["switched"]["port"] == "8014"
    assert "saved recipe" not in got["switched"]["source"]
    assert "engine defaults" in got["switched"]["source"]
    assert got["afterLateDefaults"]["title"] == "Single A"
    assert got["afterLateDefaults"]["ctx"] == "8192"
    assert got["afterLateDefaults"]["port"] == "8014"
    assert got["afterLateDefaults"]["source"] == got["switched"]["source"]
    assert got["back"]["title"].startswith("Pair Fast")
    assert got["back"]["ctx"] == "32768"
    assert got["failed"]["ctx"] == "32768"
    assert got["failed"]["source"] == "No default recipe for this model."
    assert got["roundTrip"]["title"].startswith("Pair Fast")
    assert got["roundTrip"]["ctx"] == "32768"
    assert not any(line.startswith("POST ") or line.startswith("PUT ") for line in got["methods"])
    assert load_recipe("pair-fast")["ctx"] == 4096
    assert load_recipe("pair-fast")["port"] == 8012
