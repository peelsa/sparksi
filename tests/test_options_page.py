# SPDX-License-Identifier: Apache-2.0

"""Options page: storage path/name/machine, monitor mirror masking, request log, metrics database, in WebKit at 400 px."""

import json
import os
import subprocess
import threading
import tomllib
from pathlib import Path

import pytest

from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
const { webkit } = require("playwright");
(async () => {
  const out = { dialogs: [], pageErrors: [], posts: 0 };
  const browser = await webkit.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
    page.on("pageerror", (err) => out.pageErrors.push(String(err)));
    page.on("dialog", (d) => { out.dialogs.push(d.message()); d.accept(); });
    page.on("request", (req) => {
      if (req.method() === "POST" && new URL(req.url()).pathname === "/api/options") out.posts += 1;
    });
    const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
    await page.goto(base + "/options", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("form.opt-row").length === 6);
    out.labels = await page.$$eval("form.opt-row label", (els) => els.map((e) => e.textContent.trim()));
    out.navCurrent = await page.$eval(".side a[aria-current=page]", (e) => e.getAttribute("href"));
    out.mirrorValue = await page.$eval("#opt-controller-monitor_upstream", (e) => e.value);
    out.bodyText = await page.evaluate(() => document.body.innerText);
    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);

    await page.fill("#opt-archive-root", "/etc/storage");
    await page.click("#opt-archive-root ~ .opt-save");
    await page.waitForTimeout(200);
    out.badPath = { note: await page.textContent("#opt-note"), posts: out.posts };

    await page.fill("#opt-controller-metrics_url", "http://192.0.2.5:8428");
    await page.click("#opt-controller-metrics_url ~ .opt-save");
    await page.waitForTimeout(200);
    out.badUrl = { note: await page.textContent("#opt-note"), posts: out.posts };

    await page.selectOption("#opt-controller-access_log", "all");
    await page.click("#opt-controller-access_log ~ .opt-save");
    await page.waitForFunction(() => document.querySelector("#opt-note").textContent.startsWith("Saved"));
    out.saved = {
      note: await page.textContent("#opt-note"),
      posts: out.posts,
      restart: await page.$eval("#opt-controller-access_log ~ .opt-restart", (e) => e.textContent.trim()).catch(() => null),
      value: await page.$eval("#opt-controller-access_log", (e) => e.value),
    };
    console.log(JSON.stringify(out));
  } finally {
    await browser.close();
  }
})().catch((err) => { console.error(err); process.exit(1); });
"""


@pytest.fixture()
def options_server(monkeypatch, tmp_path):
    cfg = tmp_path / "sparksi.toml"
    text = (ROOT / "config" / "sparksi.example.toml").read_text()
    lines = []
    for line in text.splitlines(keepends=True):
        if line.startswith("monitor_upstream = "):
            line = 'monitor_upstream = "http://user:secret@127.0.0.1:7500"\n'
        lines.append(line)
    cfg.write_text("".join(lines))
    monkeypatch.setenv("SPARKSI_CONFIG", str(cfg))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
    STATE.lock.acquire("test")
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], tmp_path
    finally:
        httpd.shutdown()
        STATE.runner = None


def _run(probe: str, port: int) -> dict:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit options page regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_options_page(options_server):
    port, tmp_path = options_server
    out = _run(PROBE, port)

    assert out["pageErrors"] == []
    assert out["dialogs"] == []
    assert out["labels"] == [
        "Storage path",
        "Storage name",
        "Storage machine",
        "Monitor mirror",
        "Request log",
        "Metrics database",
    ]
    assert out["navCurrent"] == "/options"
    assert out["docW"] <= 400

    assert "secret" not in out["mirrorValue"]
    assert "secret" not in out["bodyText"]
    assert out["mirrorValue"] == "http://***@127.0.0.1:7500"

    assert out["badPath"]["note"].startswith("Not saved: Storage path must be an absolute folder")
    assert out["badPath"]["posts"] == 0
    assert out["badUrl"]["note"] == "Not saved: Metrics database must be http://host:port on loopback or the tailnet."
    assert out["badUrl"]["posts"] == 0

    saved = out["saved"]
    assert saved["note"].startswith("Saved Request log. Backup: sparksi.toml.bak-")
    assert saved["note"].endswith("Restart Sparksi to use the new setting.")
    assert saved["posts"] == 1
    assert saved["restart"] == "Restart Sparksi to use the new setting."
    assert saved["value"] == "all"

    cfg = tomllib.loads((tmp_path / "sparksi.toml").read_text())
    assert cfg["controller"]["access_log"] == "all"
    assert STATE.lock.read().locked is True
