# SPDX-License-Identifier: Apache-2.0

"""The recipe list: a details view of every saved recipe for the model, Explorer style."""

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
from sparksi.recipes.store import recipe_names, save_recipe, using_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]
BASE = {"engine": "vllm", "port": 8012, "gpu_mem_util": 0.55}


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def explorer_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", dict(BASE, placement="head", ctx=4096, max_num_seqs=4, kv_cache_dtype="fp8"))
    with using_recipe("pair-fast", "pair-fast-nvfp4-c8-64k-fp8"):
        save_recipe("pair-fast", dict(BASE, placement="head", ctx=65536, max_num_seqs=8, kv_cache_dtype="fp8"))
    with using_recipe("pair-fast", "fast"):
        save_recipe("pair-fast", dict(BASE, placement="head", ctx=32768, max_num_seqs=2, speculative={"method": "eagle", "model": "", "tokens": 3}))
    with using_recipe("pair-fast", "scratch"):
        save_recipe("pair-fast", dict(BASE, placement="head", ctx=16384))
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
  const shots = process.env.SHOTS || "";
  const browser = await webkit.launch();
  const errors = [];
  const dialogs = [];
  const posts = [];
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { dialogs.push(d.message()); await d.accept(); });
  page.on("request", (req) => { if (req.method() !== "GET") posts.push(req.method() + " " + new URL(req.url()).pathname); });
  const names = () => page.$$eval("#recipe-files .recipe-file-row", (els) => els.map((el) => el.getAttribute("data-name")));
  const note = () => page.textContent("#recipe-note");
  const waitNote = (t) => page.waitForFunction((x) => document.getElementById("recipe-note").textContent.indexOf(x) !== -1, t);
  const shot = async (name, clip) => { if (shots) await page.screenshot({ path: shots + "/" + name, clip, fullPage: true }); };
  const barClip = () => page.evaluate(() => {
    const b = document.getElementById("recipe-bar").getBoundingClientRect();
    return { x: Math.max(0, b.x - 12), y: Math.max(0, b.y + window.scrollY - 60), width: Math.min(1280, b.width + 24), height: Math.min(760, b.height + 200) };
  });
  const out = { errors, dialogs, posts };
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
  await page.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 4);
  out.locked = await page.evaluate(() => /locked/.test(document.getElementById("lock-word").className));
  out.heads = await page.$$eval("#recipe-table thead th", (els) => els.map((el) => el.textContent.trim()));
  out.first = await names();
  out.cells = await page.$$eval("#recipe-files tr[data-name='fast'] td", (els) => els.map((el) => el.textContent.trim()));
  out.summary = await page.textContent("#recipe-summary");
  /* Sort by clicking a column, then again to reverse. */
  await page.click(".recipe-sort-head[data-key='concurrency']");
  out.byConc = await names();
  await page.click(".recipe-sort-head[data-key='concurrency']");
  out.byConcDesc = await names();
  out.ariaSort = await page.getAttribute("#recipe-table th[data-key='concurrency']", "aria-sort");
  await page.click(".recipe-sort-head[data-key='name']");
  out.byName = await names();
  /* Keyboard: focus a row, arrow down, Enter opens. */
  await page.focus("#recipe-files tr[data-name='default']");
  await page.keyboard.press("ArrowDown");
  out.focused = await page.evaluate(() => document.activeElement.getAttribute("data-name"));
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "fast");
  out.openRow = await page.$eval("#recipe-files tr.is-open", (el) => el.getAttribute("data-name"));
  /* The row menu opens below its row and never covers the help text above the list. */
  await page.click("#recipe-files tr[data-name='scratch'] .recipe-row-more");
  out.menu = await page.evaluate(() => {
    const m = document.getElementById("recipe-menu").getBoundingClientRect();
    const k = document.getElementById("recipe-keys").getBoundingClientRect();
    const s = document.getElementById("recipe-summary").getBoundingClientRect();
    const ids = ["recipe-cmd-open", "recipe-cmd-rename", "recipe-cmd-copy", "recipe-cmd-default", "recipe-cmd-delete"];
    return {
      shown: !document.getElementById("recipe-menu").hidden,
      clearOfHelp: m.top >= k.bottom && m.top >= s.bottom,
      focus: document.activeElement.id,
      disabled: ids.filter((id) => document.getElementById(id).disabled)
    };
  });
  await shot("recipe-explorer-menu.png", await barClip());
  await page.keyboard.press("Escape");
  out.menuAfterEsc = await page.evaluate(() => document.getElementById("recipe-menu").hidden);
  /* F2 renames in place; an off-format name gets the wording-only hint; Esc cancels. */
  await page.focus("#recipe-files tr[data-name='scratch']");
  await page.keyboard.press("F2");
  await page.waitForSelector("#recipe-rename");
  await page.fill("#recipe-rename", "my scratch");
  out.hintShown = await page.evaluate(() => !document.getElementById("recipe-rename-hint").hidden);
  await page.press("#recipe-rename", "Enter");
  out.badNameNote = await note();
  await page.fill("#recipe-rename", "scratch2");
  await shot("recipe-explorer-rename.png", await barClip());
  await page.press("#recipe-rename", "Escape");
  out.afterEsc = await names();
  await page.focus("#recipe-files tr[data-name='scratch']");
  await page.keyboard.press("F2");
  await page.fill("#recipe-rename", "pair-fast-nvfp4-c4-16k");
  out.hintOnPolicyName = await page.evaluate(() => !document.getElementById("recipe-rename-hint").hidden);
  await page.press("#recipe-rename", "Enter");
  await waitNote("Renamed to pair-fast-nvfp4-c4-16k");
  await page.waitForFunction(() => !!document.querySelector("#recipe-files tr[data-name='pair-fast-nvfp4-c4-16k']"));
  /* The Default box moves: exactly one is checked, and a checked one cannot be cleared. */
  await page.click("#recipe-files tr[data-name='pair-fast-nvfp4-c8-64k-fp8'] .recipe-default-box");
  await waitNote("is now the default");
  await page.waitForFunction(() => document.querySelector("#recipe-files tr[data-name='pair-fast-nvfp4-c8-64k-fp8'] .recipe-default-box").checked);
  out.checked = await page.$$eval("#recipe-files .recipe-default-box:checked", (els) => els.map((el) => el.closest("tr").getAttribute("data-name")));
  await page.click("#recipe-files tr[data-name='pair-fast-nvfp4-c8-64k-fp8'] .recipe-default-box");
  out.stillChecked = await page.$$eval("#recipe-files .recipe-default-box:checked", (els) => els.length);
  /* Delete on a row that is not open: confirm, and the open recipe stays open. */
  await page.focus("#recipe-files tr[data-name='pair-fast-nvfp4-c4-16k']");
  await page.keyboard.press("Delete");
  await waitNote("Deleted pair-fast-nvfp4-c4-16k");
  await page.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 3);
  out.afterDelete = await names();
  out.stillOpen = await page.textContent("#recipe-current-name");
  /* The default cannot be deleted from the menu. */
  await page.click("#recipe-files tr[data-name='pair-fast-nvfp4-c8-64k-fp8'] .recipe-row-more");
  out.defaultMenuDisabled = await page.evaluate(() => ["recipe-cmd-default", "recipe-cmd-delete"].map((id) => document.getElementById(id).disabled));
  await page.keyboard.press("Escape");
  /* Unsaved changes show on the open row. */
  await page.selectOption("#field-ctx", "8192");
  await page.waitForFunction(() => !document.getElementById("recipe-dirty").hidden);
  out.openTag = await page.textContent("#recipe-files tr.is-open .recipe-file-open");
  await shot("recipe-explorer-desktop.png", await barClip());
  /* A model with no saved recipes shows the empty state. */
  await page.click("button:has-text('Single A')");
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Single A") !== -1);
  await page.waitForFunction(() => !document.getElementById("recipe-empty").hidden || document.querySelectorAll("#recipe-files .recipe-file-row").length > 0);
  out.empty = await page.evaluate(() => ({ shown: !document.getElementById("recipe-empty").hidden, table: !document.getElementById("recipe-table").hidden, text: document.getElementById("recipe-empty").textContent }));
  await shot("recipe-explorer-empty.png", await barClip());
  await page.click("#recipe-empty-save");
  out.emptySave = await page.evaluate(() => ({ row: !document.getElementById("recipe-name-row").hidden, value: document.getElementById("recipe-name").value }));
  /* Phone: two-line rows and a Sort menu instead of headers. */
  const phone = await browser.newPage({ viewport: { width: 390, height: 844 } });
  phone.on("pageerror", (err) => errors.push(String(err)));
  await phone.goto(base + "/models", { waitUntil: "networkidle" });
  await phone.click("button:has-text('Pair Fast')");
  await phone.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 3);
  await phone.selectOption("#recipe-sort", "name:asc");
  out.phone = await phone.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    headHidden: getComputedStyle(document.querySelector("#recipe-table thead")).display === "none",
    sortShown: getComputedStyle(document.querySelector(".recipe-sort-row")).display !== "none",
    lines: Array.prototype.map.call(document.querySelectorAll("#recipe-files .recipe-file-line"), (el) => el.textContent),
    order: Array.prototype.map.call(document.querySelectorAll("#recipe-files .recipe-file-row"), (el) => el.getAttribute("data-name"))
  }));
  await phone.click("#recipe-files tr.is-open .recipe-row-more");
  out.phoneMenuRight = await phone.evaluate(() => document.getElementById("recipe-menu").getBoundingClientRect().right <= window.innerWidth);
  if (shots) {
    const b = await phone.evaluate(() => { const r = document.getElementById("recipe-bar").getBoundingClientRect(); return { y: r.y + window.scrollY, h: r.height }; });
    await phone.screenshot({ path: shots + "/recipe-explorer-phone.png", fullPage: true, clip: { x: 0, y: Math.max(0, b.y - 50), width: 390, height: Math.min(700, b.h + 60) } });
  }
  await browser.close();
  console.log(JSON.stringify(out));
})().catch((err) => { console.error(String(err && err.stack || err)); process.exit(1); });
"""


def test_recipe_list_works_like_a_details_view(explorer_server) -> None:
    port = explorer_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-list regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    shots = os.environ.get("SPARKSI_SHOT_DIR", "")
    if shots and Path(shots).is_dir():
        env["SHOTS"] = shots
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=240)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["locked"] is True
    assert got["heads"] == ["Default \u25b2", "Name", "Concurrency", "Context / KV", "Nodes", "Spec", "Modified", ""]
    assert got["first"] == ["default", "fast", "pair-fast-nvfp4-c8-64k-fp8", "scratch"]
    assert got["cells"][1].startswith("fast")
    assert got["cells"][2:6] == ["2", "32k", "1 (head)", "eagle"]
    assert "nvfp4" in got["summary"] and "4 recipes" in got["summary"] and "default default" in got["summary"]
    assert got["byConc"] == ["fast", "default", "pair-fast-nvfp4-c8-64k-fp8", "scratch"]
    assert got["byConcDesc"] == ["pair-fast-nvfp4-c8-64k-fp8", "default", "fast", "scratch"]
    assert got["ariaSort"] == "descending"
    assert got["byName"] == ["default", "fast", "pair-fast-nvfp4-c8-64k-fp8", "scratch"]
    assert got["focused"] == "fast"
    assert got["openRow"] == "fast"
    menu = got["menu"]
    assert menu["shown"] is True and menu["clearOfHelp"] is True, menu
    assert menu["focus"] == "recipe-cmd-open"
    assert menu["disabled"] == ["recipe-cmd-copy"], menu
    assert got["menuAfterEsc"] is True
    assert got["hintShown"] is True
    assert "Use letters, digits" in got["badNameNote"]
    assert "scratch" in got["afterEsc"] and "scratch2" not in got["afterEsc"]
    assert got["hintOnPolicyName"] is False
    assert got["checked"] == ["pair-fast-nvfp4-c8-64k-fp8"]
    assert got["stillChecked"] == 1
    assert got["afterDelete"] == ["default", "fast", "pair-fast-nvfp4-c8-64k-fp8"]
    assert got["stillOpen"] == "fast"
    assert got["defaultMenuDisabled"] == [True, True]
    assert got["openTag"] == "Open \u2022 unsaved"
    assert any("Delete recipe pair-fast-nvfp4-c4-16k" in d for d in got["dialogs"])
    assert got["empty"]["shown"] is True and got["empty"]["table"] is False, got["empty"]
    assert "No saved recipes" in got["empty"]["text"]
    assert got["emptySave"]["row"] is True and got["emptySave"]["value"], got["emptySave"]
    phone = got["phone"]
    assert phone["overflow"] is False and phone["headHidden"] is True and phone["sortShown"] is True, phone
    assert phone["order"] == ["default", "fast", "pair-fast-nvfp4-c8-64k-fp8"]
    assert "c2 \u00b7 32k \u00b7 1 node \u00b7 spec" in phone["lines"][1], phone
    assert got["phoneMenuRight"] is True
    assert not any("/api/runs" in p for p in got["posts"])
    # The rename and delete reached the files; nothing else was renamed.
    assert sorted(recipe_names("pair-fast")) == ["default", "fast", "pair-fast-nvfp4-c8-64k-fp8"]
