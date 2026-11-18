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


SHARED_LIBRARY = [
    {
        "id": "alpha",
        "title": "Alpha harness",
        "port": 8014,
        "placement": "pair",
        "identity": {"served": ["shared-served"], "served_prefix": "shared", "container": "alpha-harness"},
        "serving": False,
        "live": False,
        "on": {},
        # The server's answer for the newest alpha run: its containers are gone.
        "run_live": {"id": "alpha-old", "live": False, "up": False, "sides": {"head": "gone", "worker": "stopped"}},
    },
    {
        "id": "beta",
        "title": "Beta recipe",
        "port": 8014,
        "placement": "pair",
        "identity": {"served": ["shared-served"], "container": "beta-harness"},
        "serving": True,
        "live": False,
        "on": {},
        "run_live": {"id": "beta-run", "live": True, "up": True, "sides": {"head": "up", "worker": "up"}},
    },
]


@pytest.fixture()
def shared_page(monkeypatch, tmp_path):
    from tests.test_serving_state import BETA_SERVES, _alpha_failed, _beta_run, _shared_pair

    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_cache_for_tests()
    lock = Lock(tmp_path / "lock")
    STATE.lock = lock
    STATE.runner = _Quiet()
    live = _shared_pair()
    hosts = [{"id": CFG.head.id, "fresh": True, "age": 0.2}, {"id": CFG.worker.id, "fresh": True, "age": 0.2}]
    runs = [_alpha_failed(), _beta_run()]
    rows = reconcile([live], runs, hosts=hosts, head_id=CFG.head.id, worker_id=CFG.worker.id, served_names=BETA_SERVES)
    STATE.runs_fn = lambda: {"runs": runs, "serving": rows}
    STATE.monitor_fn = lambda: {"hosts": hosts, "loaded": [live]}
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1], rows
    finally:
        httpd.shutdown()
        STATE.runs_fn = None
        STATE.monitor_fn = None
        STATE.runner = None
        reset_runs()
        reset_cache_for_tests()


def test_overview_names_the_run_whose_containers_answer(shared_page) -> None:
    """Two rows serve one id. The cards name the owning run, not an older failed row."""
    port, rows = shared_page
    assert [row["run_id"] for row in rows] == ["beta-run", "beta-run"]
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    if node is None or not (node / "playwright").is_dir():
        if os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1":
            raise AssertionError("CI requires the WebKit shared served id regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const port = process.env.SERVING_PORT;
const library = JSON.parse(process.env.SHARED_LIBRARY);
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  const writes = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("request", (req) => {
    if (req.method() !== "GET" && req.url().includes("/api/")) writes.push(req.method() + " " + req.url());
  });
  await page.route("**/api/fleet", (route) => route.fulfill({
    status: 200, contentType: "application/json",
    body: JSON.stringify({ library: library, machines: [], hosts: [], storage: [] })
  }));
  await page.route("**/api/models", async (route) => {
    const real = await route.fetch();
    const body = await real.json();
    body.models = (body.models || []).concat(library.map((row) => ({
      id: row.id, title: row.title, placement: row.placement, port: row.port, identity: row.identity
    })));
    await route.fulfill({ response: real, json: body });
  });
  await page.goto("http://127.0.0.1:" + port + "/overview", { waitUntil: "domcontentloaded" });
  const box = page.locator(".serving");
  await box.getByText("Beta recipe").first().waitFor({ timeout: 15000 });
  await page.waitForTimeout(1500);
  const text = await box.innerText();
  const cls = await box.getAttribute("class");
  await page.goto("http://127.0.0.1:" + port + "/models#alpha", { waitUntil: "domcontentloaded" });
  await page.locator("#model-unload").waitFor({ state: "attached", timeout: 15000 });
  await page.waitForTimeout(1500);
  const alphaUnload = await page.locator("#model-unload").isDisabled();
  const alphaPick = await page.locator(".model-row.is-on .name").innerText();
  await page.goto("about:blank");  // a hash change alone keeps the earlier pick
  await page.goto("http://127.0.0.1:" + port + "/models#beta", { waitUntil: "domcontentloaded" });
  await page.locator("#model-unload").waitFor({ state: "attached", timeout: 15000 });
  await page.waitForTimeout(1500);
  const betaUnload = await page.locator("#model-unload").isDisabled();
  const betaPick = await page.locator(".model-row.is-on .name").innerText();
  await browser.close();
  process.stdout.write(JSON.stringify({ text, cls, alphaUnload, alphaPick, betaUnload, betaPick, errors, writes }));
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = dict(os.environ)
    env["NODE_PATH"] = str(node)
    env["SERVING_PORT"] = str(port)
    env["SHARED_LIBRARY"] = json.dumps(SHARED_LIBRARY)
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=120)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["writes"] == []
    assert got["text"].count("Beta recipe") == 2, got["text"]
    assert "Alpha harness" not in got["text"]
    assert "Ready · port 8014" in got["text"]
    assert "is-shared" in (got["cls"] or "")
    assert got["alphaPick"] == "Alpha harness" and got["alphaUnload"] is True, got
    assert got["betaPick"] == "Beta recipe" and got["betaUnload"] is False, got
