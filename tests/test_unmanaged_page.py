# SPDX-License-Identifier: Apache-2.0

"""Overview page: unmanaged models section, stop guard, unlock flow, escaping in WebKit at 400 px."""

import json
import os
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

ITEMS = [
    {
        "host": "spark-a",
        "host_name": "Spark A",
        "role": "head",
        "port": 8000,
        "label": "org/model-a",
        "container": "model-a-box",
        "why": "",
        "stoppable": True,
    },
    {
        "host": "spark-b",
        "host_name": "Spark B",
        "role": "worker",
        "port": 8001,
        "label": "proxy-thing",
        "container": None,
        "why": "not a container Sparksi can stop; stop it on the machine",
        "stoppable": False,
    },
    {
        "host": "spark-b",
        "host_name": "Spark B",
        "role": "worker",
        "port": 8002,
        "label": "<img src=x onerror=alert(1)>",
        "container": "odd-box",
        "why": "",
        "stoppable": True,
    },
]

_PROBE = """
const { webkit } = require("playwright");
(async () => {
  const out = { dialogs: [], pageErrors: [], posts: [] };
  const browser = await webkit.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
    page.on("pageerror", (err) => out.pageErrors.push(String(err)));
    let firstDialog = true;
    page.on("dialog", (d) => {
      out.dialogs.push(d.message());
      if (firstDialog) { firstDialog = false; d.dismiss(); } else { d.accept(); }
    });
    const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#unmanaged .unmanaged-row").length === 3);
    const rows = await page.$$eval("#unmanaged .unmanaged-row", (els) =>
      els.map((el) => ({
        name: el.querySelector(".name") ? el.querySelector(".name").textContent : null,
        num: el.querySelector(".num") ? el.querySelector(".num").textContent : null,
        muted: el.querySelector(".muted") ? el.querySelector(".muted").textContent : null,
        button: el.querySelector("button.halt") ? el.querySelector("button.halt").textContent : null,
        note: el.querySelector(".note") ? el.querySelector(".note").textContent : null,
      }))
    );
    const imgCount = await page.$$eval("#unmanaged img", (els) => els.length);
    const hidden = await page.$eval("#unmanaged", (el) => el.hasAttribute("hidden"));
    const docW = await page.evaluate(() => document.documentElement.scrollWidth);
    out.rows = rows;
    out.imgCount = imgCount;
    out.hidden = hidden;
    out.docW = docW;
    await page.click("#unmanaged .unmanaged-row >> nth=0 >> button.halt");
    await page.waitForTimeout(200);
    out.note1 = await page.$eval("#unmanaged-note", (el) => el.textContent);
    out.dialogsAfterLocked = out.dialogs.length;
    await page.route("**/api/unmanaged/stop", async (route) => {
      out.posts.push(route.request().postDataJSON());
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ ok: true, stopped: true, container: "model-a-box", port: 8000 }) });
    });
    await page.evaluate(async () => {
      await fetch("/api/unlock", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ from_ui: true }) });
    });
    await page.waitForTimeout(100);
    await page.reload({ waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#unmanaged .unmanaged-row").length === 3);
    await page.click("#unmanaged .unmanaged-row >> nth=0 >> button.halt");
    await page.waitForTimeout(200);
    out.postsAfterDismiss = out.posts.length;
    await page.click("#unmanaged .unmanaged-row >> nth=0 >> button.halt");
    await page.waitForFunction(() => document.querySelector("#unmanaged-note").textContent.startsWith("Stopped"));
    out.note2 = await page.$eval("#unmanaged-note", (el) => el.textContent);
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(out));
})().catch((err) => { console.error(err); process.exit(1); });
"""


@pytest.fixture()
def page_server(monkeypatch, tmp_path):
    import sparksi.serve as serve_mod

    cfg = tmp_path / "sparksi.toml"
    cfg.write_text((ROOT / "config" / "sparksi.example.toml").read_text())
    monkeypatch.setenv("SPARKSI_CONFIG", str(cfg))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
    STATE.lock.acquire("test")
    monkeypatch.setattr(STATE, "monitor_fn", lambda: {"hosts": [], "loaded": []})
    monkeypatch.setattr(serve_mod, "_unmanaged_items", lambda runner, *, cached, cat: (ITEMS, True))
    monkeypatch.setitem(CFG.controller, "stop_unmanaged", True)
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], tmp_path
    finally:
        httpd.shutdown()
        STATE.runner = None
        STATE.monitor_fn = None


def _run(probe: str, port: int) -> dict:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit unmanaged models regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_api_lists_items(page_server):
    port, _tmp = page_server
    with urlopen(f"http://127.0.0.1:{port}/api/unmanaged") as resp:
        data = json.loads(resp.read())
    assert data["items"] == ITEMS
    assert data["enabled"] is True
    assert data["checked"] is True


def test_page_shows_and_guards_stop(page_server):
    port, _tmp = page_server
    out = _run(_PROBE, port)
    assert out["pageErrors"] == []
    assert out["imgCount"] == 0
    assert out["hidden"] is False
    assert out["docW"] <= 400
    assert out["rows"] == [
        {
            "name": "org/model-a",
            "num": " · Spark A · port 8000",
            "muted": " · container model-a-box",
            "button": "Stop",
            "note": None,
        },
        {
            "name": "proxy-thing",
            "num": " · Spark B · port 8001",
            "muted": None,
            "button": None,
            "note": "not a container Sparksi can stop; stop it on the machine",
        },
        {
            "name": "<img src=x onerror=alert(1)>",
            "num": " · Spark B · port 8002",
            "muted": " · container odd-box",
            "button": "Stop",
            "note": None,
        },
    ]
    assert out["note1"] == "Unlock first, then Stop."
    assert out["dialogsAfterLocked"] == 0
    assert out["dialogs"] == [
        "Stop org/model-a on Spark A (port 8000)?\n\nSparksi did not start it. This stops the container model-a-box on that machine.",
        "Stop org/model-a on Spark A (port 8000)?\n\nSparksi did not start it. This stops the container model-a-box on that machine.",
    ]
    assert out["postsAfterDismiss"] == 0
    assert out["posts"] == [{"host": "spark-a", "port": 8000, "container": "model-a-box", "from_ui": True}]
    assert out["note2"] == "Stopped org/model-a."
