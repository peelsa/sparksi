# SPDX-License-Identifier: Apache-2.0

"""Shared ribbon on every page; Cluster has no big dials; Model Manager has one Load action."""

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
def menu_server(monkeypatch, tmp_path):
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


def test_site_menu_probe(menu_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit site-menu regression")
        pytest.skip("Playwright WebKit is not installed")

    port = menu_server

    probe = r"""
const { webkit } = require("playwright");
const fs = require("fs");
const path = require("path");

(async () => {
  const errors = [];
  const port = process.env.RECIPE_PORT;
  const shotDir = process.env.SPARKSI_SHOT_DIR;
  if (shotDir) fs.mkdirSync(shotDir, { recursive: true });
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  page.on("pageerror", (e) => errors.push("pageerror: " + String(e)));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push("console: " + m.text());
  });

  await page.goto("http://127.0.0.1:" + port + "/overview", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#menu-btn");

  const closed = await page.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const btn = document.getElementById("menu-btn");
    return {
      hidden: menu.hasAttribute("hidden"),
      visible: menu.getClientRects().length > 0,
      expanded: btn.getAttribute("aria-expanded"),
      sideWidth: Math.round(document.getElementById("main").getBoundingClientRect().left),
    };
  });
  if (shotDir) await page.screenshot({ path: path.join(shotDir, "menu-closed.png"), fullPage: false });

  await page.click("#menu-btn");
  const opened = await page.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const btn = document.getElementById("menu-btn");
    const current = menu.querySelector('a[aria-current="page"]');
    const links = Array.from(menu.querySelectorAll("a")).filter(
      (a) => a.getClientRects().length > 0
    );
    return {
      expanded: btn.getAttribute("aria-expanded"),
      visible: menu.getClientRects().length > 0,
      focusedHref: document.activeElement.getAttribute("href"),
      current: current ? current.getAttribute("href") : null,
      currentWeight: current ? getComputedStyle(current).fontWeight : "",
      labels: links.map((a) => a.textContent.trim()),
    };
  });
  if (shotDir) await page.screenshot({ path: path.join(shotDir, "menu-open.png"), fullPage: false });

  await page.keyboard.press("ArrowDown");
  const afterArrow = await page.evaluate(() => document.activeElement.getAttribute("href"));

  await page.keyboard.press("Escape");
  const esc = await page.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const btn = document.getElementById("menu-btn");
    return {
      expanded: btn.getAttribute("aria-expanded"),
      visible: menu.getClientRects().length > 0,
      focusIsButton: document.activeElement.id === "menu-btn",
    };
  });

  await page.click("#menu-btn");
  await page.mouse.click(1300, 700);
  const outside = await page.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const btn = document.getElementById("menu-btn");
    return {
      expanded: btn.getAttribute("aria-expanded"),
      visible: menu.getClientRects().length > 0,
    };
  });

  await page.click("#menu-btn");
  await page.click('#site-menu a[href="/library"]');
  await page.waitForURL("**/library");
  const nav = await page.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const btn = document.getElementById("menu-btn");
    const current = menu.querySelector('a[aria-current="page"]');
    return {
      path: location.pathname,
      expanded: btn.getAttribute("aria-expanded"),
      visible: menu.getClientRects().length > 0,
      current: current ? current.getAttribute("href") : null,
    };
  });

  const mobilePage = await browser.newPage({ viewport: { width: 390, height: 844 } });
  await mobilePage.goto("http://127.0.0.1:" + port + "/overview", { waitUntil: "domcontentloaded" });
  await mobilePage.waitForSelector("#menu-btn");
  await mobilePage.click("#menu-btn");
  const mobile = await mobilePage.evaluate(() => {
    const menu = document.getElementById("site-menu");
    const rect = menu.getBoundingClientRect();
    return {
      visible: menu.getClientRects().length > 0,
      docW: document.documentElement.scrollWidth,
      menuRight: Math.round(rect.right),
      menuLeft: Math.round(rect.left),
    };
  });

  await browser.close();
  console.log("__RESULT__" + JSON.stringify({ errors, closed, opened, afterArrow, esc, outside, nav, mobile }));
})().catch((e) => {
  console.log("__RESULT__" + JSON.stringify({ fatal: String(e) }));
  process.exit(1);
});
"""

    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    proc = subprocess.run(
        ["node"],
        input=probe,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=180,
    )
    if proc.returncode != 0 and "Executable doesn't exist" in proc.stderr and not required:
        pytest.skip("Playwright WebKit is not installed")
    if proc.returncode != 0:
        raise AssertionError(
            "site-menu probe failed\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr)
        )

    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result_line = line
    assert result_line is not None, "probe produced no __RESULT__ line:\n%s" % proc.stdout
    data = json.loads(result_line[len("__RESULT__"):])

    errors = data["errors"]
    assert errors == []

    closed = data["closed"]
    assert closed["hidden"] is True
    assert closed["visible"] is False
    assert closed["expanded"] == "false"
    assert closed["sideWidth"] < 80

    opened = data["opened"]
    assert opened["expanded"] == "true"
    assert opened["visible"] is True
    assert opened["focusedHref"] == "/overview"
    assert opened["current"] == "/overview"
    assert int(opened["currentWeight"]) >= 700
    for label in (
        "Overview",
        "Model Manager",
        "Library",
        "Network",
        "Options",
        "Token totals",
        "Activity",
    ):
        assert label in opened["labels"]

    assert data["afterArrow"] == "/models"

    esc = data["esc"]
    assert esc["expanded"] == "false"
    assert esc["visible"] is False
    assert esc["focusIsButton"] is True

    outside = data["outside"]
    assert outside["expanded"] == "false"
    assert outside["visible"] is False

    nav = data["nav"]
    assert nav["path"] == "/library"
    assert nav["visible"] is False
    assert nav["expanded"] == "false"
    assert nav["current"] == "/library"

    mobile = data["mobile"]
    assert mobile["visible"] is True
    assert mobile["docW"] <= 390
    assert mobile["menuLeft"] >= 0
    assert mobile["menuRight"] <= 390
