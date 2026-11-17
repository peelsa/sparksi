# SPDX-License-Identifier: Apache-2.0

"""Archive page sync check and Overview SSH fallback, in WebKit."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.jobs import reset_for_tests as reset_jobs
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def page_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_jobs()
    reset_cache_for_tests()
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
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
        reset_jobs()
        reset_cache_for_tests()


PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const stages = [];
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  const sync = {
    hosts: ["spark-a", "spark-b"],
    rows: [
      { id: "flat-x", title: "Flat X", gib: 124, archive_state: "archived", hosts: {
        "spark-a": { state: "differs", note: "3 of 120 shards missing or a different size", can_copy: true, refill: true },
        "spark-b": { state: "unchecked", note: "could not reach spark-b", can_copy: false, refill: false } } },
      { id: "tar-y", title: "Tar Y", gib: 8, archive_state: "archived", hosts: {
        "spark-a": { state: "missing", note: "not on this Spark", can_copy: true, refill: false },
        "spark-b": { state: "in_sync", note: "all 4 shards match", can_copy: false, refill: false } } }
    ],
    checked_at: "2026-11-14T10:00:00Z"
  };
  await page.route("**/api/sync", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(sync) }));
  let refuse = false;
  await page.route("**/api/library/stage", (route) => {
    stages.push(JSON.parse(route.request().postData() || "{}"));
    if (refuse) return route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ verdict: "REFUSE", reasons: ["spark-a has no archive mount"] }) });
    return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ verdict: "STARTED" }) });
  });
  await page.route("**/api/nodes", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ nodes: [
    { role: "head", reachable: true, mem: { used_gib: 40.2, total_gib: 121.7 }, gpu: { util_pct: 3, temp_c: 40 } },
    { role: "worker", reachable: false, mem: null, gpu: null }
  ] }) }));
  const out = { errors, stages };
  try {
    await page.goto(base + "/archive", { waitUntil: "networkidle" });
    out.before = await page.evaluate(() => ({ table: !document.getElementById("sync-table").hidden, note: document.getElementById("sync-note").textContent }));
    await page.click("#btn-sync-check");
    await page.waitForFunction(() => document.getElementById("sync-note").textContent.indexOf("checked") !== -1);
    out.after = await page.evaluate(() => {
      const t = document.getElementById("sync-table");
      return {
        note: document.getElementById("sync-note").textContent,
        heads: Array.from(t.querySelectorAll("thead th")).map((th) => th.textContent),
        cells: Array.from(t.querySelectorAll("tbody tr")).map((tr) => Array.from(tr.querySelectorAll("td")).map((td) => td.textContent)),
        buttons: Array.from(t.querySelectorAll("tbody button")).map((b) => b.textContent),
        docW: document.documentElement.scrollWidth
      };
    });
    await page.click("#sync-table button:has-text('Copy missing shards')");
    await page.waitForFunction(() => document.getElementById("archive-msg").textContent.indexOf("started") !== -1);
    out.started = await page.textContent("#archive-msg");
    refuse = true;
    await page.click("#sync-table button:text-is('Copy')");
    await page.waitForFunction(() => document.getElementById("archive-msg").textContent.indexOf("Not copied") === 0);
    out.refused = await page.textContent("#archive-msg");
    out.copyEnabled = await page.evaluate(() => !document.querySelector("#sync-table tbody tr:nth-child(2) button").disabled);
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.getElementById("status-head").textContent.indexOf("SSH") !== -1, null, { timeout: 15000 });
    out.overview = await page.evaluate(() => ({
      head: document.getElementById("status-head").textContent,
      headClass: document.getElementById("status-head").className,
      worker: document.getElementById("status-worker").textContent,
      stats: document.getElementById("stats-head").textContent,
      disk: document.getElementById("disk-head").textContent,
      decode: document.getElementById("tile-decode").textContent
    }));
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def test_sync_check_copy_and_ssh_fallback_in_webkit(page_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit sync page regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["PAGE_PORT"] = str(page_server)
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, json.dumps(got, indent=1)
    assert got["errors"] == [], got
    assert got["before"]["table"] is False and got["before"]["note"] == "", got
    after = got["after"]
    assert after["note"].startswith("2 copies differ or are missing · 1 not checked · checked "), got
    assert after["heads"] == ["model", "spark-a", "spark-b"], got
    assert after["cells"][0][0] == "Flat X  124 GiB"
    assert after["cells"][0][1].startswith("Differs3 of 120 shards"), got
    assert after["cells"][0][2] == "Not checkedcould not reach spark-b", got
    assert after["cells"][1][2] == "In syncall 4 shards match", got
    assert after["buttons"] == ["Copy missing shards", "Copy"], got
    assert after["docW"] <= 400, got
    assert got["stages"][0] == {"id": "flat-x", "hosts": ["spark-a"], "refill": True, "from_ui": True}, got
    assert got["stages"][1]["refill"] is False and got["stages"][1]["id"] == "tar-y", got
    assert got["started"] == "Copy to spark-a started. Progress shows above.", got
    assert got["refused"] == "Not copied: spark-a has no archive mount.", got
    assert got["copyEnabled"] is True, got
    ov = got["overview"]
    assert ov["decode"] == "Idle", ov  # the head answered SSH, so Idle has evidence
    assert ov["head"] == "Reachable over SSH · announcer quiet" and "ok" in ov["headClass"], got
    assert "SSH" not in ov["worker"], got
    assert "40.2/121.7 G" in ov["stats"] and "104 F" in ov["stats"] and "3%" in ov["stats"], got
    assert ov["disk"] == "from SSH", got


EVIDENCE_PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const browser = await webkit.launch();
  const errors = [];
  const plans = [];
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.route("**/api/nodes", (route) => route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify({ nodes: [
    { role: "head", reachable: false, mem: null, gpu: null, error: "timed out" },
    { role: "worker", reachable: false, mem: null, gpu: null, error: "timed out" }
  ] }) }));
  await page.route("**/api/models/*/plan", (route) => {
    plans.push(JSON.parse(route.request().postData() || "{}"));
    return route.fulfill({ status: 400, contentType: "application/json", body: JSON.stringify({ ok: false, verdict: "REFUSE",
      conflicts: ["could not reach the head", "IPC_LOCK is only allowed for a two-machine recipe when both Sparks have their high-speed network configured."] }) });
  });
  const out = { errors, plans };
  try {
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll(".serve-box .serve-model").length >= 2
      && Array.from(document.querySelectorAll(".serve-box .serve-model")).every((m) => m.textContent !== "Loading…"), null, { timeout: 20000 });
    await page.waitForTimeout(1500);
    out.serving = await page.evaluate(() => Array.from(document.querySelectorAll(".serve-box")).map((box) => ({
      model: box.querySelector(".serve-model").textContent,
      meta: box.querySelector(".serve-meta").textContent
    })));
    out.anyNothing = await page.evaluate(() => document.body.innerText.indexOf("Nothing running") !== -1);
    out.decode = await page.evaluate(() => document.getElementById("tile-decode").textContent);
    await page.goto(base + "/models", { waitUntil: "networkidle" });
    await page.click("button:has-text('Pair Fast')");
    await page.waitForSelector("#field-ctx");
    await page.waitForTimeout(1000);
    out.radios = await page.evaluate(() => Array.from(document.querySelectorAll("#model-placement input")).map((i) => i.value + (i.checked ? "*" : "") + (i.disabled ? "-" : "")));
    out.selected = await page.evaluate(() => document.getElementById("model-placement").getAttribute("data-selected"));
    await page.evaluate(() => {
      const box = document.getElementById("model-placement");
      Array.from(box.querySelectorAll("input")).forEach((i) => { i.checked = false; i.disabled = true; });
    });
    await page.click("#model-dry");
    await page.waitForFunction(() => document.getElementById("model-dry-note").textContent.indexOf("Not ready") === 0);
    out.dry = await page.textContent("#model-dry-note");
    out.dryLines = await page.evaluate(() => Array.from(document.querySelectorAll("#model-dry-note .note-line")).map((n) => n.textContent));
    // Load is disabled here, so stand in a leftover Load message by hand.
    await page.evaluate(() => { document.getElementById("model-load-note").textContent = "Not loaded: could not reach both Sparks."; });
    out.loadBefore = await page.textContent("#model-load-note");
    await page.click("button.model-row:has-text('Single A')");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent !== "");
    await page.waitForTimeout(1000);
    out.afterSwitch = await page.evaluate(() => ({ dry: document.getElementById("model-dry-note").textContent, load: document.getElementById("model-load-note").textContent }));
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def test_no_nothing_running_without_evidence_and_one_line_per_reason(page_server) -> None:
    from sparksi.recipes.store import save_recipe

    role = {"runtime": "docker", "image": "nvcr.io/nvidia/vllm:26.05-py3", "args": ["vllm", "serve", "/models/x", "--port", "8012"]}
    save_recipe("pair-fast", {
        "kind": "pinned", "engine": "vllm", "placement": "both", "port": 8012,
        "imported_from": "container:example at 2026-10-31T00:00:00Z", "roles": {"head": role, "worker": dict(role)},
    })
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit evidence regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["PAGE_PORT"] = str(page_server)
    proc = subprocess.run(["node"], input=EVIDENCE_PROBE, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, json.dumps(got, indent=1)
    assert got["errors"] == [], got
    assert got["serving"] == [
        {"model": "No report yet", "meta": "Can't reach · no announcer or SSH answer"},
        {"model": "No report yet", "meta": "Can't reach · no announcer or SSH answer"},
    ], got
    assert got["anyNothing"] is False, got
    assert got["decode"] == "No report yet", got
    # With every placement disabled, Dry run checks the recipe's own placement, not a "head" fallback.
    assert got["selected"] == "both", json.dumps({k: got[k] for k in ("radios", "selected", "plans")})
    assert got["plans"][-1]["placement"] == got["selected"], got
    # Switching model clears the previous model's plan, Dry run and Load messages.
    assert got["loadBefore"], got
    assert got["afterSwitch"] == {"dry": "", "load": ""}, got
    assert got["dryLines"] == [
        "Not ready. Nothing was started.",
        "· could not reach the head. Check that the machine is on and answers over SSH, then try again.",
        "· IPC_LOCK is only allowed for a two-machine recipe when both Sparks have their high-speed network configured.",
    ], got
