# SPDX-License-Identifier: Apache-2.0

"""Overview paints the serving state from GET /api/runs. No model is started."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler
from sparksi.serving_state import reconcile
from tests.test_serving_state import _hosts, _split_pair, _stopped

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def serving_page(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_cache_for_tests()
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    STATE.lock = lock
    STATE.runner = _Quiet()
    rows = reconcile([_split_pair()], [_stopped()], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    STATE.runs_fn = lambda: {"runs": [_stopped()], "serving": rows}
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], rows
    finally:
        httpd.shutdown()
        STATE.runs_fn = None
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()


def test_overview_shows_the_split_pair(serving_page) -> None:
    port, rows = serving_page
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit serving-state regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const port = process.env.SERVING_PORT;
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  const writes = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (req) => {
    const url = req.url();
    if (req.method() !== "GET" && (url.includes("/api/runs") || url.includes("/api/models"))) {
      writes.push(req.method() + " " + url);
    }
  });
  await page.goto("http://127.0.0.1:" + port + "/overview", { waitUntil: "domcontentloaded" });
  const box = page.locator(".serving");
  await box.getByText("worker up").waitFor({ timeout: 15000 });
  const text = await box.innerText();
  const cls = await box.getAttribute("class");
  await browser.close();
  process.stdout.write(JSON.stringify({ text: text, cls: cls, errors: errors, writes: writes }));
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = dict(os.environ)
    env["NODE_PATH"] = str(node)
    env["SERVING_PORT"] = str(port)
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["writes"] == []
    assert "is-shared" not in (got["cls"] or "")
    assert "worker up" in got["text"]
    assert "not confirmed" in got["text"]
    assert "example-engine" in got["text"]
    assert "Nothing running" not in got["text"]
    assert "Ready" not in got["text"]
    assert "retired" not in got["text"].lower()
    assert rows[0]["state"] == "split"
