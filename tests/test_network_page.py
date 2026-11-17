# SPDX-License-Identifier: Apache-2.0

"""Network page: Discover lists machines, roles and endpoints, with an Add to Sparksi control for each, in WebKit at 400 px."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

SCAN = {
    "scanned_at": "2026-01-01T12:00:00Z", "took_s": 2.4,
    "hosts": [
        {"name": "ctl", "dns": "ctl.example.ts.net", "ips": ["100.64.0.1"], "sources": ["tailscale"], "self": True,
         "role": "controller", "why": "this machine", "endpoints": [], "proposal": ""},
        {"name": "spark-one", "dns": "spark-one.example.ts.net", "ips": ["100.64.0.11", "192.0.2.11"], "sources": ["lan", "tailscale"],
         "self": False, "role": "spark", "why": "name looks like a DGX Spark",
         "endpoints": [{"port": 8014, "engine": "vLLM", "models": ["example/model-a"], "url": "http://100.64.0.11:8014"}],
         "proposal": "[[nodes]]\nid = \"spark-one\"\nname = \"spark-one\"\nssh = \"spark-one\"\ntailscale = \"100.64.0.11\""},
        {"name": "a-very-long-machine-name-that-should-wrap-on-a-phone.example.ts.net", "dns": "", "ips": ["100.64.0.20"],
         "sources": ["tailscale"], "self": False, "role": "inference", "why": "serves an inference API",
         "endpoints": [{"port": 11434, "engine": "Ollama", "models": ["small:7b", "tiny:1b"], "url": "http://100.64.0.20:11434"}],
         "proposal": "# A monitored inference host (Network page: Add to Sparksi).\n[[endpoints]]\nname = \"x\"\nurl = \"http://100.64.0.20:11434\"\nengine = \"Ollama\""},
    ],
}

PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const browser = await webkit.launch();
  const out = { errors: [], posts: [] };
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => out.errors.push(String(err)));
  let posted = false;
  out.pollsAfterPost = 0;
  await page.route("**/api/discover", (route) => {
    const req = route.request();
    if (req.method() === "POST") {
      out.posts.push(req.headers()["content-type"] || "");
      posted = true;
      return route.fulfill({
        status: 202,
        contentType: "application/json",
        body: JSON.stringify({ running: true, started: true, scanned_at: null, hosts: [], took_s: null }),
      });
    }
    if (!posted) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ scanned_at: null, hosts: [], took_s: null, running: false }),
      });
    }
    out.pollsAfterPost += 1;
    if (out.pollsAfterPost === 1) {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ scanned_at: null, hosts: [], took_s: null, running: true }),
      });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(Object.assign({}, scan, { running: false })),
    });
  });
  try {
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.getElementById("discover-note").textContent !== "");
    out.before = await page.textContent("#discover-note");
    out.navOn = await page.evaluate(() => document.querySelector('.side a[aria-current="page"]').textContent);
    out.visible = await page.evaluate(() => !document.getElementById("view-network").hidden && document.getElementById("view-overview").hidden);
    await page.click("#btn-discover");
    await page.waitForFunction(() => document.querySelectorAll(".net-host").length === 3);
    out.note = await page.textContent("#discover-note");
    out.cards = await page.evaluate(() => Array.from(document.querySelectorAll(".net-host")).map((s) => ({
      title: s.querySelector("h3").textContent,
      role: s.querySelector(".net-role").textContent,
      items: Array.from(s.querySelectorAll("li")).map((li) => li.textContent),
      add: !!s.querySelector(".net-add"),
    })));
    out.addRoles = await page.evaluate(() => Array.from(document.querySelectorAll(".net-add-row select")).map((sel) => sel.value));
    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);
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


def test_network_page_lists_machines_with_add_controls(page_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit network page regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(page_server), SCAN=json.dumps(SCAN))
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, json.dumps(got, indent=1)
    assert got["errors"] == [], got
    assert got["before"] == "Not scanned yet. Press Discover to look at this network.", got
    assert got["navOn"] == "Network" and got["visible"] is True, got
    assert got["posts"] == ["application/json"], got
    assert got["pollsAfterPost"] >= 2, got  # 202 at once, then the page polls GET until the scan is done
    assert got["note"].startswith("3 machines · scanned ") and got["note"].endswith(" · took 2.4 s."), got
    cards = got["cards"]
    assert [c["role"] for c in cards] == ["This controller", "DGX Spark", "Inference host"], cards
    assert cards[1]["items"] == ["vLLM · http://100.64.0.11:8014 · example/model-a"], cards
    assert cards[2]["items"] == ["Ollama · http://100.64.0.20:11434 · small:7b, tiny:1b"], cards
    assert [c["add"] for c in cards] == [False, True, True], cards
    assert got["addRoles"] == ["head", "monitor"], got  # a Spark defaults to head, an inference host to monitor
    assert got["docW"] <= 400, got
