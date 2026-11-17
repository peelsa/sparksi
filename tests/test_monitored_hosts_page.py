# SPDX-License-Identifier: Apache-2.0

"""Overview page: monitored host cards, statuses, model lines, escaping, layout in WebKit at 400 px."""

import json
import os
import subprocess
import threading
from pathlib import Path
from urllib.request import urlopen

import pytest

from sparksi import monitor_hosts
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

HOSTS = [
    {
        "id": "box-a",
        "name": "Box A",
        "url": "http://192.0.2.5:8000",
        "engine": "vLLM",
        "address": "192.0.2.5:8000",
        "state": "reachable",
        "last_seen": "2026-01-02T03:04:05Z",
        "age_s": 1,
        "error": "",
        "served": ["org/model-a"],
        "available": [],
        "loaded": [],
        "health": "ok",
        "metrics": {"kv_cache_pct": 12.5, "running": 2},
    },
    {
        "id": "box-b",
        "name": "Box B",
        "url": "http://192.0.2.6:11434",
        "engine": "Ollama",
        "address": "192.0.2.6:11434",
        "state": "offline",
        "last_seen": "2026-01-02T03:04:05Z",
        "age_s": 90,
        "error": "connection refused",
        "served": [],
        "available": [],
        "loaded": [],
        "health": "",
        "metrics": {},
    },
    {
        "id": "box-c",
        "name": "<img src=x onerror=alert(1)>",
        "url": "http://192.0.2.7:8080",
        "engine": "llama.cpp",
        "address": "192.0.2.7:8080",
        "state": "reachable",
        "last_seen": "2026-01-02T03:04:05Z",
        "age_s": 2,
        "error": "",
        "served": [],
        "available": [],
        "loaded": [],
        "health": "loading",
        "metrics": {},
    },
]

PROBE = """
const { webkit } = require("playwright");
(async () => {
  const out = { dialogs: [], pageErrors: [] };
  const browser = await webkit.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
    page.on("pageerror", (err) => out.pageErrors.push(String(err)));
    page.on("dialog", (d) => { out.dialogs.push(d.message()); d.accept(); });
    const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#monitored-hosts .host-card").length === 3);
    out.cards = await page.$$eval("#monitored-hosts .host-card", (els) =>
      els.map((e) => ({
        id: e.dataset.id,
        h2: e.querySelector("h2").textContent,
        status: e.querySelector("h2 .status").textContent,
        statusClass: e.querySelector("h2 .status").className,
        where: e.querySelector(".host-where").textContent,
        lines: Array.from(e.querySelectorAll(".host-models")).map((x) => x.textContent),
        note: e.querySelector(".note") ? e.querySelector(".note").textContent : null,
        cls: e.className,
      }))
    );
    out.imgCount = await page.evaluate(() => document.querySelectorAll("#monitored-hosts img").length);
    out.hidden = await page.$eval("#monitored-hosts", (e) => e.classList.contains("hidden"));
    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);
    out.order = await page.evaluate(() =>
      Array.from(document.querySelector("#view-overview").children)
        .map((e) => e.id)
        .filter(Boolean)
    );
    console.log(JSON.stringify(out));
  } finally {
    await browser.close();
  }
})().catch((err) => { console.error(err); process.exit(1); });
"""


@pytest.fixture()
def page_server(monkeypatch, tmp_path):
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
    monkeypatch.setattr(monitor_hosts, "view", lambda: HOSTS)
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
            raise AssertionError("CI requires the WebKit monitored hosts regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_monitor_api_lists_monitored_hosts(page_server):
    port, _tmp_path = page_server
    with urlopen(f"http://127.0.0.1:{port}/api/monitor", timeout=10) as resp:
        data = json.loads(resp.read())
    assert data["monitored"] == HOSTS


def test_overview_shows_monitored_host_cards(page_server):
    port, _tmp_path = page_server
    out = _run(PROBE, port)

    assert out["pageErrors"] == []
    assert out["dialogs"] == []
    assert out["imgCount"] == 0
    assert out["hidden"] is False
    assert out["docW"] <= 400
    assert "gauges" not in out["order"]
    assert out["order"].index("overview-chart") < out["order"].index("monitored-hosts")

    cards = out["cards"]
    assert [c["id"] for c in cards] == ["box-a", "box-b", "box-c"]

    a, b, c = cards
    assert a["status"] == "Reachable"
    assert a["statusClass"] == "status ok"
    assert a["where"] == "192.0.2.5:8000 · vLLM"
    assert a["lines"] == ["Serves org/model-a", "KV cache 12.5% in use", "Requests 2 running"]
    assert a["note"] is None
    assert "is-offline" not in a["cls"]

    assert b["status"].startswith("Offline · last seen ")
    assert b["status"].endswith(" · connection refused")
    assert b["statusClass"] == "status muted"
    assert b["where"] == "192.0.2.6:11434 · Ollama"
    assert b["lines"] == []
    assert "is-offline" in b["cls"]

    assert c["h2"].startswith("<img src=x onerror=alert(1)>")
    assert c["status"] == "Reachable · loading"
    assert c["note"] == "No models served"

    for card in cards:
        text = card["status"].lower()
        assert "red" not in text
        assert "amber" not in text
        assert "yellow" not in text
