# SPDX-License-Identifier: Apache-2.0

"""Model confirm dialog: Load / Unload / Restart verbs hit the right endpoints exactly once."""

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
from sparksi.recipes.store import save_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def recipe_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "gpu_mem_util": 0.55})
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


def test_model_confirm_dialog_verbs(recipe_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit model-confirm regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const port = process.env.RECIPE_PORT;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  const errors = [];
  const posts = [];
  const nativeDialogs = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", (d) => { nativeDialogs.push(d.message()); d.dismiss(); });
  page.on("request", (req) => {
    const m = req.method();
    if (m === "POST" || m === "PUT" || m === "DELETE") {
      posts.push(m + " " + req.url().split(port)[1]);
    }
  });
  await page.route("**/api/runs", async (route) => {
    const req = route.request();
    if (req.method() !== "GET") return route.continue();
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({runs:[{id:"r1",model_id:"single-a",phase:"ready",port:8013,placement:"head",started:1,nodes:[{host:"spark-a",role:"head"}]}]}),
    });
  });
  let stops = 0;
  await page.route("**/api/runs/r1/stop", async (route) => {
    stops += 1;
    await route.fulfill({ status: 200, contentType: "application/json", body: '{"ok":true}' });
  });
  let lockedNow = false;
  await page.route("**/api/status", async (route) => {
    const resp = await route.fetch();
    const j = await resp.json();
    if (lockedNow) j.lock = { locked: true, locked_for_actions: true, on: "ON" };
    await route.fulfill({ response: resp, json: j });
  });
  let holdStopped = false;
  await page.route("**/api/runs/r1", async (route) => {
    if (route.request().method() !== "GET") return route.continue();
    const phase = holdStopped ? "running" : "stopped";
    await route.fulfill({ status: 200, contentType: "application/json", body: '{"run":{"id":"r1","phase":"' + phase + '"}}' });
  });
  const runsSeen = page.waitForResponse((r) => r.url().endsWith("/api/runs") && r.request().method() === "GET");
  await page.goto("http://127.0.0.1:" + port + "/models");
  await runsSeen;
  await page.getByRole("button", { name: /Single A/ }).click();
  await page.waitForSelector("#model-port");
  await page.waitForResponse((r) => r.url().endsWith("/api/runs") && r.request().method() === "GET");
  await sleep(200);
  const open = () => page.evaluate(() => document.getElementById("model-confirm").open);
  const click = (id) => page.evaluate((i) => document.getElementById(i).click(), id);
  const txt = (sel) => page.$eval(sel, (el) => el.textContent);
  const out = {};
  const dialogOpen = async () => page.waitForFunction(() => document.getElementById("model-confirm").open);

  // 1 Load -> Cancel
  await click("model-load");
  await dialogOpen();
  out.s1_open = await open();
  out.s1_ok = await txt("#model-confirm-ok");
  out.s1_model = await txt("#model-confirm-model");
  out.s1_port = await txt("#model-confirm-port");
  out.s1_port_field = await page.$eval("#model-port", (el) => el.value);
  const postsBeforeCancel = posts.length;
  await click("model-confirm-cancel");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  out.s1_closed = !(await open());
  out.s1_no_post = posts.length === postsBeforeCancel;

  // 2 Load -> Confirm
  await click("model-load");
  await dialogOpen();
  const planAsked = page.waitForRequest((r) => r.method() === "POST" && r.url().indexOf("/api/models/single-a/plan") !== -1, { timeout: 10000 }).catch(() => null);
  await click("model-confirm-ok");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  await planAsked;
  out.s2_plan_post = posts.filter((p) => p.startsWith("POST /api/models/single-a/plan")).length > 0;

  // 3 Unload -> Escape
  await click("model-unload");
  await dialogOpen();
  out.s3_ok = await txt("#model-confirm-ok");
  out.s3_machines = await txt("#model-confirm-machines");
  out.s3_port = await txt("#model-confirm-port");
  out.s3_note = await txt("#model-confirm-note");
  const stopsBeforeEscape = stops;
  await page.keyboard.press("Escape");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  out.s3_closed = !(await open());
  out.s3_no_stop = stops === stopsBeforeEscape;

  // 4 Unload -> Confirm
  await click("model-unload");
  await dialogOpen();
  const stopAsked = page.waitForRequest((r) => r.method() === "POST" && r.url().endsWith("/api/runs/r1/stop"), { timeout: 10000 }).catch(() => null);
  await click("model-confirm-ok");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  await stopAsked;
  await sleep(100);
  out.s4_stops = stops;
  out.s4_library_stop = posts.filter((p) => p.indexOf("/api/library/stop") !== -1).length;

  // 5 Restart with changes -> Confirm
  await click("model-restart");
  await dialogOpen();
  out.s5_ok = await txt("#model-confirm-ok");
  const restartStop = page.waitForRequest((r) => r.method() === "POST" && r.url().endsWith("/api/runs/r1/stop"), { timeout: 10000 }).catch(() => null);
  await click("model-confirm-ok");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  await restartStop;
  await sleep(1500);
  out.s5_stops_after = stops;
  out.s5_reopened = await open();

  // 8 Restart re-checks the port
  await click("model-restart");
  await dialogOpen();
  const prevPort = await page.$eval("#model-port", (el) => el.value);
  const stopsBefore8 = stops;
  await page.evaluate(() => { document.getElementById("model-port").value = "8188"; });
  await click("model-confirm-ok");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  await sleep(800);
  out.s8_stops_delta = stops - stopsBefore8;
  out.s8_note = await txt("#model-load-note");
  await page.evaluate((v) => { document.getElementById("model-port").value = v; }, prevPort);

  // 9 Restart loads the model captured on Confirm
  holdStopped = true;
  await click("model-restart");
  await dialogOpen();
  const restartStop9 = page.waitForRequest((r) => r.method() === "POST" && r.url().endsWith("/api/runs/r1/stop"), { timeout: 10000 }).catch(() => null);
  await click("model-confirm-ok");
  await page.waitForFunction(() => !document.getElementById("model-confirm").open);
  await restartStop9;
  await page.evaluate(() => { const b = Array.from(document.querySelectorAll("button")).find((x) => /Pair Fast/.test(x.textContent)); b.click(); });
  await sleep(300);
  // Only the real load plan: previews of the newly selected model also POST /plan with preview set.
  const planSeen = page.waitForRequest((r) => r.method() === "POST" && r.url().indexOf("/plan") !== -1 && (r.postData() || "").indexOf('"preview":true') === -1, { timeout: 10000 }).catch(() => null);
  holdStopped = false;
  const plan = await planSeen;
  out.s9_plan_url = plan ? plan.url().split(port)[1] : "";
  if (plan) await plan.response();
  await sleep(300);  // let the plan's note settle before the next step

  // 7 Unload blocked while locked
  // Back to the model with the live run, so Unload takes the run path.
  await page.evaluate(() => { const b = Array.from(document.querySelectorAll("button")).find((x) => /Single A/.test(x.textContent)); b.click(); });
  await page.waitForFunction(() => document.getElementById("model-title").textContent.indexOf("Single A") !== -1);
  await sleep(300);
  const statusSeen = () => page.waitForResponse((r) => r.url().endsWith("/api/status") && r.request().method() === "GET", { timeout: 25000 }).catch(() => null);  // the page reads the lock every 20 s
  await click("model-unload");
  await dialogOpen();
  const stopsBefore7 = stops;
  lockedNow = true;
  out.s7_status_seen = !!(await statusSeen());
  await sleep(200);
  await click("model-confirm-ok");
  await sleep(500);
  out.s7_stops_delta = stops - stopsBefore7;
  out.s7_note = await txt("#model-load-note");
  out.s7_machines = await txt("#model-confirm-machines");

  out.s6_errors = errors;
  out.s6_native = nativeDialogs;
  console.log("__RESULT__" + JSON.stringify(out));
  await browser.close();
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(recipe_server)
    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if browsers:
        env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    raw = proc.stdout
    marker = "__RESULT__"
    got = json.loads(raw[raw.index(marker) + len(marker) :].strip().splitlines()[-1])
    assert got["s1_open"] is True
    assert got["s1_ok"] == "Load"
    assert "Single A" in got["s1_model"]
    assert got["s1_port"] == got["s1_port_field"]
    assert got["s1_closed"] is True
    assert got["s1_no_post"] is True
    assert got["s2_plan_post"] is True
    assert got["s3_ok"] == "Unload"
    assert "spark-a" in got["s3_machines"]
    assert got["s3_port"] == "8013"
    assert got["s3_closed"] is True
    assert got["s3_no_stop"] is True
    assert got["s3_note"] == ""  # the run path, not the outside-run path
    assert got["s4_stops"] == 1
    assert got["s4_library_stop"] == 0
    assert got["s5_ok"] == "Restart"
    assert got["s5_stops_after"] == 2
    assert got["s5_reopened"] is False
    assert got["s6_errors"] == []
    assert got["s6_native"] == []
    assert got["s7_status_seen"] is True
    assert got["s7_stops_delta"] == 0
    assert "Unlock first" in got["s7_note"]
    assert "spark-a" in got["s7_machines"]
    assert got["s8_stops_delta"] == 0
    assert "8188" in got["s8_note"]
    assert "/api/models/single-a/plan" in got["s9_plan_url"]
    assert "pair-fast" not in got["s9_plan_url"]
