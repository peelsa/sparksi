# SPDX-License-Identifier: Apache-2.0

"""Overview: context size and KV cache use on the serving card heading, with no change in card size.

The real page runs in WebKit against GET /api/runs with a ready pair run. The server
read is replaced by fixed answers; nothing is started and no model server is asked."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi import serving_info
from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler
from sparksi.serving_state import reconcile
from tests.test_serving_state import BETA_SERVES, _alpha_failed, _beta_run, _shared_pair

ROOT = Path(__file__).resolve().parents[1]
INFO = {"ctx": 850000, "kv_pct": 13.6, "kv_tokens": 1233779, "running": 1, "engine": "vllm"}


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


class _Answers:
    """Stands in for serving_info.CACHE: the test flips what the 'server' reported."""

    def __init__(self):
        self.info: dict = {}
        self.urls: list[str] = []

    def get(self, url, key=""):
        self.urls.append(url)
        return dict(self.info)


@pytest.fixture()
def kv_page(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_cache_for_tests()
    STATE.lock = Lock(tmp_path / "lock")
    STATE.runner = _Quiet()
    answers = _Answers()
    monkeypatch.setattr(serving_info, "CACHE", answers)
    live = _shared_pair()
    hosts = [{"id": CFG.head.id, "fresh": True, "age": 0.2}, {"id": CFG.worker.id, "fresh": True, "age": 0.2}]
    runs = [_alpha_failed(), _beta_run(health={"host": CFG.head.name, "port": 8014, "verdict": "ready"})]
    rows = reconcile([live], runs, hosts=hosts, head_id=CFG.head.id, worker_id=CFG.worker.id, served_names=BETA_SERVES)
    STATE.runs_fn = lambda: {"runs": runs, "serving": rows}
    STATE.monitor_fn = lambda: {"hosts": hosts, "loaded": [live]}
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], answers
    finally:
        httpd.shutdown()
        STATE.runs_fn = None
        STATE.monitor_fn = None
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()


PROBE = r"""
const { webkit } = require("playwright");
const port = process.env.SERVING_PORT;
(async () => {
  const browser = await webkit.launch();
  const out = { errors: [], writes: [], views: {} };
  try {
    for (const width of [1400, 390]) {
      const page = await browser.newPage({ viewport: { width: width, height: 900 } });
      page.on("pageerror", (err) => out.errors.push(String(err)));
      page.on("request", (req) => {
        if (req.method() !== "GET" && req.url().includes("/api/")) out.writes.push(req.method() + " " + req.url());
      });
      const measure = async () => {
        await page.goto("http://127.0.0.1:" + port + "/overview", { waitUntil: "domcontentloaded" });
        await page.locator(".serving").getByText("Ready · port 8014").first().waitFor({ timeout: 15000 });
        await page.waitForTimeout(1200);
        return page.$$eval(".serving .serve-box", (els) => els.map((el) => {
          const r = el.getBoundingClientRect();
          const kv = el.querySelector(".serve-kv");
          const head = el.querySelector("h2");
          const style = kv ? getComputedStyle(kv) : null;
          return {
            role: el.getAttribute("data-role"), w: Math.round(r.width), h: Math.round(r.height),
            kv: kv ? kv.textContent : null, title: kv ? kv.getAttribute("title") : null,
            color: style ? style.color : null, headH: Math.round(head.getBoundingClientRect().height),
            spill: el.scrollWidth > el.clientWidth,
          };
        }));
      };
      await fetch("http://127.0.0.1:" + process.env.CONTROL_PORT + "/info");
      const withInfo = await measure();
      await fetch("http://127.0.0.1:" + process.env.CONTROL_PORT + "/none");
      const without = await measure();
      out.views[String(width)] = { withInfo, without };
      await page.close();
    }
  } finally {
    await browser.close();
  }
  process.stdout.write(JSON.stringify(out));
})().catch((err) => { console.error(err); process.exit(1); });
"""


def _control(answers):
    """A tiny local switch the probe calls between page loads: /info or /none."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Switch(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            answers.info = dict(INFO) if self.path == "/info" else {}
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

        def log_message(self, *_a):
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Switch)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_card_heading_shows_context_and_kv_without_growing(kv_page) -> None:
    port, answers = kv_page
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    if node is None or not (node / "playwright").is_dir():
        if os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1":
            raise AssertionError("CI requires the WebKit context and KV cache regression")
        pytest.skip("Playwright WebKit is not installed")
    switch = _control(answers)
    try:
        env = dict(os.environ)
        env.update(NODE_PATH=str(node), SERVING_PORT=str(port), CONTROL_PORT=str(switch.server_address[1]))
        proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=180)
    finally:
        switch.shutdown()
        switch.server_close()
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["errors"] == [] and got["writes"] == []
    assert answers.urls and all(url.endswith(":8014") for url in answers.urls)
    for width, view in got["views"].items():
        with_info, without = view["withInfo"], view["without"]
        assert [c["role"] for c in with_info] == ["head", "worker"], (width, with_info)
        for card in with_info:
            # Both sides of the pair name the run's facts; the title explains the shared cache.
            assert card["kv"] == "850k ctx · 14% used", (width, card)
            assert "shared by all requests" in card["title"] and "not one conversation" in card["title"], card
            assert "1,233,779 tokens" in card["title"] and "850,000 tokens" in card["title"], card
            assert card["spill"] is False, (width, card)
        for card in without:
            assert card["kv"] == "" and card["title"] is None, (width, card)
        for a, b in zip(with_info, without, strict=True):
            assert (a["w"], a["h"], a["headH"]) == (b["w"], b["h"], b["headH"]), (width, a, b)
        # Muted text from the approved palette, the same colour as the clock beside it.
        assert len({c["color"] for c in with_info}) == 1
