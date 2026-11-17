# SPDX-License-Identifier: Apache-2.0

"""Monitor polling: one request in flight, a new one once it is stuck past 3 s, newest reply wins.

A slow reply still moves the gauges when nothing newer has landed, and a late
reply never overwrites a reply from a newer request.
"""

import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_overlapping_monitor_replies_still_move_the_gauges() -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit monitor-overlap regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const http = require("http");
const fs = require("fs");
const path = require("path");
const staticDir = process.env.STATIC_DIR;
const waiting = [];
function host(role, gpu) {
  return { id: role, name: role, age: 1, fresh: true, mem_used_gib: 32, mem_total_gib: 128, mem_pct: 30, cpu_pct: 10, gpu_pct: gpu, gpu_temp_c: 45, power_w: 40, zones: [] };
}
function send(res, body) {
  res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}
const server = http.createServer((req, res) => {
  const url = (req.url || "/").split("?")[0];
  if (url === "/api/monitor") {
    const seq = waiting.length + 1;
    const gpu = seq === 1 ? 40 : seq === 2 ? 60 : 80;
    waiting.push({ res, gpu, at: Date.now() });
    return;
  }
  const stubs = {
    "/api/config": { nodes: [{ role: "head", id: "head", name: "head" }, { role: "worker", id: "worker", name: "worker" }], head: { id: "head", name: "head" }, worker: { id: "worker", name: "worker" }, controller: { name: "controller" }, archive: { id: "archive", label: "Archive" } },
    "/api/status": { lock: { locked: true, on: "ON", reason: "held" } },
    "/api/runs": { runs: [] },
    "/api/models": { models: [], services: [], reserved_ports: [] },
    "/api/job": { job: null },
    "/api/activity": { events: [] },
    "/api/fleet": { library: [], storage: [], machines: [], hosts: [] },
    "/api/fleet-power": { watts: 40, estimate: true },
    "/api/tokens": { models: {} },
    "/api/token-day": { ok: true, days: [] },
    "/api/token-life": { models: {} },
    "/api/nodes": { nodes: [] },
    "/api/telemetry": { kept: "indefinitely", files: [], roles: {}, last_error: "", next: null },
    "/api/history": { machine: "head", range: "1h", now: 1700000000, samples: [] }
  };
  if (stubs[url]) { send(res, stubs[url]); return; }
  const file = (url === "/" || url === "/overview") ? "index.html" : url.replace(/^\//, "");
  const full = path.join(staticDir, file);
  if (!full.startsWith(staticDir) || !fs.existsSync(full)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200);
  res.end(fs.readFileSync(full));
});
function gaugeText(page) {
  return page.evaluate(() => Array.prototype.map.call(document.querySelectorAll("#ribbon .state"), function (node) { return node.textContent; }).join(" "));
}
server.listen(0, "127.0.0.1", async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.goto("http://127.0.0.1:" + server.address().port + "/overview", { waitUntil: "domcontentloaded" });
  const waitFor = async (n, ms) => {
    const t = Date.now();
    while (waiting.length < n) {
      if (Date.now() - t > ms) throw new Error("monitor request " + n + " did not start");
      await page.waitForTimeout(25);
    }
  };
  const gauges = (want, not) => page.waitForFunction(([w, n]) => {
    const text = Array.prototype.map.call(document.querySelectorAll("#ribbon .state"), function (node) { return node.textContent; }).join(" ");
    return text.indexOf(w) >= 0 && (!n || text.indexOf(n) < 0);
  }, [want, not]);
  await waitFor(1, 8000);
  await page.waitForTimeout(2000);
  const inFlightEarly = waiting.length;  // polled every second, still one request
  await waitFor(2, 4000);
  const firstGap = waiting[1].at - waiting[0].at;
  // 1 answers while 2 is pending: nothing newer has landed, so the gauges move.
  send(waiting[0].res, { ok: true, hosts: [host("head", 40), host("worker", 40)], loaded: [] });
  await gauges("40%", "60%");
  const mid = await gaugeText(page);
  // 2 is stuck; past 3 s a third request starts. 3 answers first, then the late 2.
  await waitFor(3, 6000);
  const secondGap = waiting[2].at - waiting[1].at;
  send(waiting[2].res, { ok: true, hosts: [host("head", 80), host("worker", 80)], loaded: [] });
  await gauges("80%");
  send(waiting[1].res, { ok: true, hosts: [host("head", 60), host("worker", 60)], loaded: [] });
  await page.waitForTimeout(1500);
  const later = await gaugeText(page);
  for (const item of waiting.slice(3)) send(item.res, { ok: true, hosts: [host("head", 80), host("worker", 80)], loaded: [] });
  await browser.close();
  server.close();
  process.stdout.write(JSON.stringify({ errors, mid, later, inFlightEarly, firstGap, secondGap }));
});
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["STATIC_DIR"] = str(ROOT / "src" / "sparksi" / "static")
    browsers = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "")
    if browsers:
        env["PLAYWRIGHT_BROWSERS_PATH"] = browsers
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == []
    assert got["inFlightEarly"] == 1, got  # one in flight normally
    assert 2900 <= got["firstGap"] <= 4600, got  # a new request once one is stuck past 3 s
    assert 2900 <= got["secondGap"] <= 4600, got
    assert "40%" in got["mid"] and "60%" not in got["mid"], got  # an older reply lands when nothing newer has
    assert "80%" in got["later"] and "60%" not in got["later"], got  # a stale reply never overwrites a newer one
