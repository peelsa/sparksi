# SPDX-License-Identifier: Apache-2.0

"""The recipe File menu saves, opens, renames and deletes named recipes in the browser."""

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
def bar_server(monkeypatch, tmp_path):
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
        yield httpd.server_address[1], tmp_path
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()


def test_recipe_file_menu_round_trip(bar_server) -> None:
    port, tmp_path = bar_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-bar regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const methods = [];
  const dialogs = [];
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { dialogs.push(d.message()); await d.accept(); });
  page.on("request", (req) => {
    const url = new URL(req.url());
    const p = url.pathname;
    if (p.includes("/recipe") || p.includes("/plan") || p.includes("/runs")) {
      methods.push(req.method() + " " + p + url.search);
    }
  });
  const menu = async (id) => {
    if (["recipe-cmd-new", "recipe-cmd-save", "recipe-cmd-save-as"].indexOf(id) === -1) await page.click("#recipe-files tr.is-open .recipe-row-more");
    await page.click("#" + id);
  };
  const waitName = async (text) => {
    await page.waitForFunction((t) => document.getElementById("recipe-current-name").textContent === t, text);
  };
  const waitText = async (sel, text) => {
    await page.waitForFunction(
      ([s, t]) => document.getElementById(s) && document.getElementById(s).textContent.indexOf(t) !== -1,
      [sel, text]
    );
  };
  globalThis.__state = async () => page.evaluate(() => ({
    name: document.getElementById("recipe-current-name").textContent,
    note: document.getElementById("recipe-note").textContent,
    ctx: document.getElementById("field-ctx") && document.getElementById("field-ctx").value
  })).then((x) => Object.assign(x, { methods: methods.filter((m) => m.indexOf("/api/runs") === -1).slice(-8) }));
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForSelector("#field-ctx");
  await page.waitForFunction(
    () => document.getElementById("recipe-current-name") && document.getElementById("recipe-current-name").textContent === "default"
  );
  await page.waitForFunction(() => /locked/.test(document.getElementById("lock-word").className));
  const layout = await page.evaluate(() => ({
    afterTitle: document.getElementById("model-title").nextElementSibling.id,
    inDetails: !!document.getElementById("recipe-bar").closest("details"),
    above: document.getElementById("recipe-bar").getBoundingClientRect().top < document.getElementById("model-fields").getBoundingClientRect().top,
    loadDisabled: document.getElementById("model-load").disabled,
    oldSave: !!document.getElementById("model-save"),
    oldDry: !!document.getElementById("recipe-dry"),
    loads: document.querySelectorAll("#model-load").length,
    drys: document.querySelectorAll("#model-dry").length,
    dryBesideMenu: document.getElementById("model-dry").parentElement.className === "recipe-head"
  }));
  await page.selectOption("#field-ctx", "32768");
  await page.waitForFunction(() => !document.getElementById("recipe-dirty").hidden);
  await menu("recipe-cmd-save-as");
  await page.fill("#recipe-name", "long-ctx");
  await page.click("#recipe-name-ok");
  await waitName("long-ctx");
  await page.waitForFunction(() => document.getElementById("recipe-dirty").hidden);
  await page.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 2);
  const rows = await page.$$eval("#recipe-files .recipe-file-row", (els) => els.map((el) => ({
    name: el.querySelector(".recipe-file-name").textContent,
    def: el.querySelector(".recipe-file-default").textContent,
    time: el.querySelector(".recipe-file-time").textContent
  })));
  await page.click("#recipe-files .recipe-file-row[data-name='default']");
  await page.waitForFunction(() => document.getElementById("field-ctx").value === "4096");
  await page.click("#recipe-files .recipe-file-row[data-name='long-ctx']");
  await page.waitForFunction(() => document.getElementById("field-ctx").value === "32768");
  await menu("recipe-cmd-default");
  await waitText("recipe-note", "is now the default");
  await menu("recipe-cmd-rename");
  await page.fill("#recipe-rename", "long");
  await page.press("#recipe-rename", "Enter");
  await waitName("long");
  await menu("recipe-cmd-save-as");
  await page.fill("#recipe-name", "default");
  await page.click("#recipe-name-ok");
  await waitText("recipe-note", "already exists");
  const overwriteNote = await page.textContent("#recipe-note");
  await page.click("#recipe-name-ok");
  await waitName("default");
  await menu("recipe-cmd-delete");
  await waitText("recipe-note", "Deleted default");
  await menu("recipe-cmd-new");
  await page.waitForFunction(
    () => document.getElementById("recipe-current-name").textContent === "Untitled" && !document.getElementById("recipe-dirty").hidden
  );
  await page.click("#model-dry");
  await page.waitForFunction(() => {
    const t = document.getElementById("model-dry-note").textContent;
    return t.length > 0 && t !== "Checking…";
  });
  const phonePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  phonePage.on("pageerror", (err) => errors.push(String(err)));
  await phonePage.goto(base + "/models", { waitUntil: "networkidle" });
  await phonePage.click("button:has-text('Pair Fast')");
  await phonePage.waitForFunction(
    () => document.getElementById("recipe-current-name") && document.getElementById("recipe-current-name").textContent === "long"
  );
  await phonePage.click("#recipe-files tr.is-open .recipe-row-more");
  const phone = await phonePage.evaluate(() => {
    const rect = (id) => document.getElementById(id).getBoundingClientRect();
    return {
      menuRight: rect("recipe-menu").right,
      menuShown: !document.getElementById("recipe-menu").hidden,
      line: document.querySelector("#recipe-files tr.is-open .recipe-file-line").textContent,
      width: window.innerWidth,
      pageOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
      visible: ["recipe-cmd-new", "recipe-cmd-save-as", "model-dry", "model-load", "recipe-sort"].every((id) => {
        const r = rect(id);
        return r.width > 0 && r.height > 0;
      })
    };
  });
  await browser.close();
  console.log(JSON.stringify({ errors, layout, rows, overwriteNote, dialogs, phone, methods }));
})().catch(async (err) => {
  let state = null;
  try { state = globalThis.__state ? await globalThis.__state() : null; } catch (e) { state = String(e); }
  console.error(String(err && err.stack || err) + "\nstate: " + JSON.stringify(state));
  process.exit(1);
});
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
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
    layout = got["layout"]
    assert layout["afterTitle"] == "recipe-bar"
    assert layout["inDetails"] is False
    assert layout["above"] is True
    assert layout["loadDisabled"] is True
    assert layout["oldSave"] is False
    assert layout["oldDry"] is False
    assert layout["loads"] == 1
    assert layout["drys"] == 1
    assert layout["dryBesideMenu"] is True
    by_name = {row["name"]: row for row in got["rows"]}
    assert by_name["default"]["def"] == "Default"
    assert by_name["default"]["time"]
    assert by_name["long-ctx"]["time"]
    assert "already exists" in got["overwriteNote"]
    assert any("Delete recipe default" in d for d in got["dialogs"])
    phone = got["phone"]
    assert phone["menuRight"] <= phone["width"]
    assert phone["pageOverflow"] is False
    assert phone["visible"] is True
    assert phone["menuShown"] is True
    assert "c" in phone["line"] and "k" in phone["line"], phone
    methods = got["methods"]
    assert "POST /api/models/pair-fast/recipes" in methods
    assert "POST /api/models/pair-fast/recipes/long-ctx/default" in methods
    assert "POST /api/models/pair-fast/recipes/long-ctx/rename" in methods
    assert "DELETE /api/models/pair-fast/recipes/default" in methods
    assert any(m.startswith("POST /api/models/pair-fast/plan") for m in methods)
    assert not any(m.startswith("POST /api/runs") for m in methods)
    assert not any(m == "PUT /api/models/pair-fast/recipe" for m in methods)
    assert load_recipe("pair-fast")["ctx"] == 32768
    assert not (tmp_path / "recipes" / "named" / "pair-fast" / "default.toml").exists()


def test_switching_models_keeps_the_recipe_name_of_a_draft(bar_server) -> None:
    port, _tmp_path = bar_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-bar regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { await d.accept(); });
  const shown = () => page.evaluate(() => ({
    name: document.getElementById("recipe-current-name").textContent,
    dirty: !document.getElementById("recipe-dirty").hidden,
    ctx: document.getElementById("field-ctx") && document.getElementById("field-ctx").value,
    title: document.getElementById("model-title").textContent
  }));
  const waitName = async (text) => {
    await page.waitForFunction((t) => document.getElementById("recipe-current-name").textContent === t, text);
  };
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForSelector("#field-ctx");
  await waitName("default");
  await page.selectOption("#field-ctx", "32768");
  await page.click("#recipe-cmd-save-as");
  await page.fill("#recipe-name", "long-ctx");
  await page.click("#recipe-name-ok");
  await waitName("long-ctx");
  await page.selectOption("#field-ctx", "16384");
  await page.waitForFunction(() => !document.getElementById("recipe-dirty").hidden);
  await page.click("button:has-text('Pair EXL3')");
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Pair EXL3") !== -1);
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent !== "long-ctx");
  await page.waitForTimeout(500);
  const away = await shown();
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(
    () => document.getElementById("model-title").textContent.indexOf("Pair Fast") !== -1 && document.getElementById("field-ctx").value === "16384"
  );
  const atSwitch = await shown();
  await page.waitForFunction(() => document.getElementById("field-ctx").value === "16384");
  const back = await shown();
  await page.waitForTimeout(6000);
  const later = await shown();
  await browser.close();
  console.log(JSON.stringify({ errors, away, atSwitch, back, later }));
})().catch((err) => {
  console.error(String(err && err.stack || err));
  process.exit(1);
});
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
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
    assert got["away"]["name"] != "long-ctx"
    assert got["away"]["dirty"] is False
    # When the draft is back in the form, the name already matches it.
    assert got["atSwitch"]["name"] == "long-ctx"
    assert got["atSwitch"]["dirty"] is True
    assert got["back"]["name"] == "long-ctx"
    assert got["back"]["dirty"] is True
    assert got["back"]["ctx"] == "16384"
    # The 5 s models tick must not reset the name or the draft.
    assert got["later"]["name"] == "long-ctx"
    assert got["later"]["dirty"] is True
    assert got["later"]["ctx"] == "16384"



SAVE_PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const mode = process.env.SAVE_MODE;
  const browser = await webkit.launch();
  const errors = [];
  const calls = [];
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { await d.accept(); });
  page.on("response", async (r) => {
    const u = new URL(r.url());
    if (r.request().method() !== "GET" && u.pathname.indexOf("/recipes") !== -1) {
      calls.push({ method: r.request().method(), path: u.pathname, status: r.status() });
    }
  });
  const note = () => page.evaluate(() => document.getElementById("recipe-note").textContent);
  const state = () => page.evaluate(() => ({
    name: document.getElementById("recipe-current-name").textContent,
    dirty: !document.getElementById("recipe-dirty").hidden,
    note: document.getElementById("recipe-note").textContent,
    row: !document.getElementById("recipe-name-row").hidden
  }));
  const menu = async (id) => {
    if (["recipe-cmd-new", "recipe-cmd-save", "recipe-cmd-save-as"].indexOf(id) === -1) await page.click("#recipe-files tr.is-open .recipe-row-more");
    await page.click("#" + id);
  };
  const settle = async () => {
    await page.waitForFunction(() => {
      const t = document.getElementById("recipe-note").textContent;
      return t.length > 0 && t.indexOf("…") === -1;
    });
    return state();
  };
  const out = { errors, calls };
  try {
    await page.goto(base + "/models", { waitUntil: "networkidle" });
    await page.click("button:has-text('Pair Fast')");
    await page.waitForSelector("#field-ctx");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
    await page.selectOption("#field-ctx", "16384");
    if (mode === "plain") {
      await menu("recipe-cmd-save");
      out.saved = await settle();
    } else if (mode === "pinned") {
      /* Save is greyed out on the original pinned recipe and says why. */
      const saveState = () => page.evaluate(() => {
        const b = document.getElementById("recipe-cmd-save");
        const why = document.getElementById("recipe-save-why");
        return { disabled: b.disabled, aria: b.getAttribute("aria-disabled"), title: b.title, why: why.hidden ? "" : why.textContent };
      });
      out.locked = await saveState();
      out.lockedState = await state();
      await menu("recipe-cmd-save-as");
      await page.fill("#recipe-name", "edited");
      await page.click("#recipe-name-ok");
      out.savedAs = await settle();
      await page.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 2);
      out.copy = await saveState();
      await page.selectOption("#field-ctx", "8192");
      await page.evaluate(() => { document.getElementById("recipe-note").textContent = ""; });
      await menu("recipe-cmd-save");
      out.saved = await settle();
      await page.waitForTimeout(300);
      out.rows = await page.$$eval("#recipe-files .recipe-file-row", (els) => els.map((el) => el.getAttribute("data-name")));
    } else {
      await page.route("**/api/models/pair-fast/recipes/default", (route) => {
        if (route.request().method() === "PUT") {
          return route.fulfill({ status: 403, contentType: "application/json", body: JSON.stringify({ error: "cross-origin write refused" }) });
        }
        return route.continue();
      });
      await menu("recipe-cmd-save");
      out.refused = await settle();
      await page.route("**/api/models/pair-fast/recipes", (route) => {
        if (route.request().method() === "POST") return route.abort();
        return route.continue();
      });
      await menu("recipe-cmd-save-as");
      await page.fill("#recipe-name", "offline");
      await page.click("#recipe-name-ok");
      out.offline = await settle();
      await page.unroute("**/api/models/pair-fast/recipes");
      await page.route("**/api/models/pair-fast/recipes/default/rename", (route) => route.fulfill({ status: 500, contentType: "text/html", body: "<h1>oops</h1>" }));
      await page.click("#recipe-name-cancel");
      await menu("recipe-cmd-rename");
      await page.fill("#recipe-rename", "renamed");
      await page.press("#recipe-rename", "Enter");
      out.badReply = await settle();
      await page.press("#recipe-rename", "Escape");
      await page.unroute("**/api/models/pair-fast/recipes/default/rename");
      await menu("recipe-cmd-save-as");
      await page.fill("#recipe-name", "spare");
      await page.click("#recipe-name-ok");
      out.spare = await settle();
      await page.route("**/api/models/pair-fast/recipes/spare", (route) => {
        if (route.request().method() === "DELETE") {
          return route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ error: "set another recipe as the default first" }) });
        }
        return route.continue();
      });
      await page.evaluate(() => { document.getElementById("recipe-note").textContent = ""; });
      await menu("recipe-cmd-delete");
      out.deleteRefused = await settle();
    }
  } catch (err) {
    out.crash = String(err && err.stack || err);
    try { out.at = await state(); } catch (e) { out.at = String(e); }
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def _save_probe(port: int, mode: str) -> dict:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-bar regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    env["SAVE_MODE"] = mode
    proc = subprocess.run(["node"], input=SAVE_PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_file_save_writes_the_default_recipe_at_phone_width(bar_server) -> None:
    port, _tmp_path = bar_server
    got = _save_probe(port, "plain")
    assert "crash" not in got, got
    assert got["errors"] == [], got
    assert got["saved"]["note"].startswith("Saved default at "), got
    assert got["saved"]["dirty"] is False, got
    assert {"method": "PUT", "path": "/api/models/pair-fast/recipes/default", "status": 200} in got["calls"], got
    assert load_recipe("pair-fast")["ctx"] == 16384


def test_save_on_a_pinned_recipe_is_greyed_out_and_its_copy_saves_in_place(bar_server) -> None:
    """The staging path: a pinned imported default, Save as a copy, then Save on the copy.

    Save is disabled on the original with "Pinned recipe: use Save As". Regression: Save on the
    Save as copy used to make yet another new row, because the copy inherited the pinned lock."""
    from sparksi.recipes.store import named_recipe_path, recipe_path, using_recipe

    port, _tmp_path = bar_server
    pinned = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8012,
        "ctx": 4096,
        "imported_from": "container:example at 2026-10-31T00:00:00Z",
        "roles": {"head": {"runtime": "docker", "image": "nvcr.io/nvidia/vllm:26.05-py3", "args": ["vllm", "serve", "/models/x", "--port", "8012", "--max-model-len", "4096"]}},
    }
    save_recipe("pair-fast", pinned)
    original = recipe_path("pair-fast").read_bytes()
    got = _save_probe(port, "pinned")
    assert "crash" not in got, got
    assert got["errors"] == [], got
    assert got["locked"] == {"disabled": True, "aria": "true", "title": "Pinned recipe: use Save As", "why": "Pinned recipe: use Save As"}, got
    assert got["lockedState"]["row"] is False, got
    assert got["savedAs"]["note"].startswith("Saved as edited at "), got
    assert got["copy"] == {"disabled": False, "aria": "false", "title": "", "why": ""}, got
    assert got["saved"]["note"].startswith("Saved edited at "), got
    assert got["saved"]["row"] is False, got
    assert sorted(got["rows"]) == ["default", "edited"], got
    puts = [c for c in got["calls"] if c["method"] == "PUT"]
    assert puts == [{"method": "PUT", "path": "/api/models/pair-fast/recipes/edited", "status": 200}], got
    assert recipe_path("pair-fast").read_bytes() == original
    assert named_recipe_path("pair-fast", "edited").is_file()
    with using_recipe("pair-fast", "edited"):
        saved = load_recipe("pair-fast")
    assert saved["kind"] == "pinned" and saved["copied_from"] == "default"
    assert "8192" in json.dumps(saved) and "16384" not in json.dumps(saved)


def test_every_recipe_write_failure_is_shown_in_the_bar(bar_server) -> None:
    port, _tmp_path = bar_server
    got = _save_probe(port, "failures")
    assert "crash" not in got, json.dumps(got, indent=1)
    assert got["errors"] == [], got
    assert got["refused"]["note"] == "Not saved. cross-origin write refused", got
    assert got["offline"]["note"].startswith("Not saved. The server did not answer"), got
    assert got["badReply"]["note"] == "Not renamed. The server answered 500.", got
    assert got["spare"]["note"].startswith("Saved as spare at "), got
    assert got["deleteRefused"]["note"] == "Not deleted. set another recipe as the default first", got


LIB_PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  const out = { errors };
  try {
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    out.navHref = await page.getAttribute("nav.side a:has-text('Library')", "href");
    await page.click("#menu-btn");  // pages live in the Menu dropdown
    await page.click("nav.side a:has-text('Library')");
    await page.waitForFunction(() => document.getElementById("job-line").textContent === "No copy running.");
    await page.waitForFunction(() => document.querySelectorAll("#library tbody tr").length > 0);
    out.library = await page.evaluate(() => {
      const s = document.querySelector("#view-library .scroll");
      return {
        path: location.pathname,
        visible: !document.getElementById("view-library").hidden,
        pauseHidden: document.getElementById("btn-job-pause").hidden,
        cancelHidden: document.getElementById("btn-job-cancel").hidden,
        scrollW: s.scrollWidth,
        clientW: s.clientWidth,
        docW: document.documentElement.scrollWidth
      };
    });
    await page.route("**/api/models/pair-fast/plan", (route) => route.fulfill({
      status: 400,
      contentType: "application/json",
      body: JSON.stringify({ ok: false, verdict: "REFUSE", conflicts: ["could not reach the head"] })
    }));
    await page.goto(base + "/models", { waitUntil: "networkidle" });
    await page.click("button:has-text('Pair Fast')");
    await page.waitForSelector("#field-ctx");
    await page.click("#model-dry");
    await page.waitForFunction(() => document.getElementById("model-dry-note").textContent.indexOf("Not ready") === 0);
    out.dry = await page.textContent("#model-dry-note");
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def test_library_link_idle_job_buttons_phone_width_and_unreachable_note(bar_server) -> None:
    from sparksi.jobs import reset_for_tests as reset_jobs

    reset_jobs()
    port, _tmp_path = bar_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit library regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(["node"], input=LIB_PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, got
    assert got["errors"] == [], got
    assert got["navHref"] == "/library", got
    lib = got["library"]
    assert lib["path"] == "/library" and lib["visible"], got
    assert lib["pauseHidden"] is True and lib["cancelHidden"] is True, got
    assert lib["scrollW"] <= lib["clientW"] + 1, got
    assert lib["docW"] <= 400, got
    assert got["dry"] == (
        "Not ready: could not reach the head. Check that the machine is on and answers over SSH, "
        "then try again. Nothing was started."
    ), got


LATE_LIST_PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { await d.accept(); });
  // Hold one recipe-list reply: read it from the server now, hand it to the page later.
  let hold = false;
  let release = null;
  let held = 0;
  await page.route("**/api/models/pair-fast/recipes", async (route) => {
    if (route.request().method() !== "GET" || !hold) return route.continue();
    hold = false;
    held += 1;
    const response = await route.fetch();
    await new Promise((resolve) => { release = resolve; });
    await route.fulfill({ response });
  });
  const menu = async (id) => {
    if (["recipe-cmd-new", "recipe-cmd-save", "recipe-cmd-save-as"].indexOf(id) === -1) await page.click("#recipe-files tr.is-open .recipe-row-more");
    await page.click("#" + id);
  };
  const name = () => page.evaluate(() => document.getElementById("recipe-current-name").textContent);
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
  await page.selectOption("#field-ctx", "32768");
  await menu("recipe-cmd-save-as");
  await page.fill("#recipe-name", "long-ctx");
  await page.click("#recipe-name-ok");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "long-ctx");
  await page.waitForLoadState("networkidle");
  // Set as default: its list refresh is the reply held back.
  hold = true;
  await menu("recipe-cmd-default");
  await page.waitForFunction(() => document.getElementById("recipe-note").textContent.indexOf("is now the default") !== -1);
  const t0 = Date.now();
  while (!release && Date.now() - t0 < 5000) await page.waitForTimeout(25);
  // Rename while that older list is still out.
  await menu("recipe-cmd-rename");
  await page.fill("#recipe-rename", "long");
  await page.press("#recipe-rename", "Enter");
  await page.waitForFunction(() => document.getElementById("recipe-note").textContent.indexOf("Renamed to long") !== -1);
  await page.waitForTimeout(400);
  const beforeRelease = await name();
  if (release) release();
  await page.waitForTimeout(800);
  const afterRelease = await name();
  await browser.close();
  console.log(JSON.stringify({ errors, held, beforeRelease, afterRelease }));
})().catch((err) => {
  console.error(String(err && err.stack || err));
  process.exit(1);
});
"""


def test_a_late_recipe_list_does_not_undo_a_rename(bar_server) -> None:
    """A list reply that started before a rename lands after it: the bar keeps the new name."""
    port, _tmp_path = bar_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit recipe-bar regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(["node"], input=LATE_LIST_PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == [], got
    assert got["held"] == 1, got
    assert got["beforeRelease"] == "long", got
    assert got["afterRelease"] == "long", got
