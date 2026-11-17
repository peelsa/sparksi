# SPDX-License-Identifier: Apache-2.0

"""Save As pre-fills a suggested name; a name off the format gets a wording-only hint."""

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
from sparksi.recipes.store import recipe_names, save_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def name_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "gpu_mem_util": 0.55})
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")  # recipe edits keep working while the page is locked
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
  const state = () => page.evaluate(() => ({
    value: document.getElementById("recipe-name").value,
    hint: !document.getElementById("recipe-name-hint").hidden,
    hintText: document.getElementById("recipe-name-hint").textContent,
    current: document.getElementById("recipe-current-name").textContent,
    def: !!document.querySelector("#recipe-files tr.is-open .recipe-default-box:checked"),
    defCount: document.querySelectorAll("#recipe-files .recipe-default-box:checked").length
  }));
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
  const start = await state();
  await page.selectOption("#field-ctx", "32768");
  await menu("recipe-cmd-save-as");
  const suggested = await state();
  if (process.env.SHOT) await page.screenshot({ path: process.env.SHOT, clip: await page.evaluate(() => {
    const r = document.getElementById("recipe-name-row").closest("section, .card, .panel") || document.getElementById("recipe-name-row");
    const b = r.getBoundingClientRect();
    return { x: Math.max(0, b.x - 8), y: Math.max(0, b.y - 8), width: Math.min(1400, b.width + 16), height: b.height + 16 };
  }) });
  await page.fill("#recipe-name", "my fast one");
  await page.fill("#recipe-name", "fast");
  const offFormat = await state();
  if (process.env.SHOT_HINT) await page.screenshot({ path: process.env.SHOT_HINT });
  /* Any valid name saves: the hint never blocks. */
  await page.click("#recipe-name-ok");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "fast");
  const saved = await state();
  /* The next Save As steps around the taken name only when it clashes. */
  await menu("recipe-cmd-save-as");
  const again = await state();
  await page.click("#recipe-name-ok");
  await page.waitForFunction((n) => document.getElementById("recipe-current-name").textContent === n, again.value);
  await menu("recipe-cmd-save-as");
  const clash = await state();
  await page.click("#recipe-name-cancel");
  await page.click("#recipe-files tr.is-open .recipe-default-box");
  await page.waitForFunction(() => document.getElementById("recipe-note").textContent.indexOf("is now the default") !== -1);
  const afterDefault = await state();
  await browser.close();
  console.log(JSON.stringify({ errors, start, suggested, offFormat, saved, again, clash, afterDefault }));
})().catch((err) => { console.error(String(err && err.stack || err)); process.exit(1); });
"""


def test_save_as_prefills_a_suggested_name(name_server) -> None:
    port = name_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-name regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    shots = os.environ.get("SPARKSI_SHOT_DIR", "")
    if shots and Path(shots).is_dir():
        env["SHOT"] = str(Path(shots) / "recipe-save-as-suggested.png")
        env["SHOT_HINT"] = str(Path(shots) / "recipe-save-as-hint.png")
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["start"]["def"] is True and got["start"]["defCount"] == 1, got["start"]
    name = got["suggested"]["value"]
    # Title "Pair Fast NVFP4", 4 sequences, 32k context, fp8 cache, head only: no notes.
    assert name == "pair-fast-nvfp4-c4-32k-fp8", got["suggested"]
    assert got["suggested"]["hint"] is False
    assert got["offFormat"]["hint"] is True
    assert "model-quant-concurrency-kvcache-notes" in got["offFormat"]["hintText"]
    assert "Any name works" in got["offFormat"]["hintText"]
    assert got["saved"]["current"] == "fast"
    assert got["saved"]["def"] is False and got["saved"]["defCount"] == 1
    assert got["again"]["value"] == name
    assert got["clash"]["value"] == name + "-2"
    assert got["afterDefault"]["def"] is True and got["afterDefault"]["defCount"] == 1
    assert got["afterDefault"]["current"] == name
    # Saved names are as typed: nothing is renamed to the format.
    assert set(recipe_names("pair-fast")) == {"default", "fast", name}
