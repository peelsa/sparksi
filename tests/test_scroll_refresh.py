# SPDX-License-Identifier: Apache-2.0

"""Overview refreshes must not move the scroll position.

WebKit clamps scrollY when a refresh empties a node, then leaves it there.
These tests run Playwright's WebKit engine. They are not physical iPhone Safari
and do not claim the mobile URL bar. In CI the WebKit run is required.
"""

import json
import os
import subprocess
from pathlib import Path

from sparksi.serve import INDEX_HTML

ROOT = Path(__file__).resolve().parents[1]


def test_refresh_keeps_gauge_nodes_and_chart_pin() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    gauge = html.split("function ribbonPill", 1)[1].split("function paintOverview", 1)[0]
    assert "clear(" not in gauge
    assert "removeChild" not in gauge
    serve = html.split("function paintServing", 1)[1].split("function gaugeArc", 1)[0]
    assert "clear(" not in serve
    assert "CHART_FOLLOW" in html
    assert "pinnedEnd" in html
    math_block = html.split("/* chart-math-start */", 1)[1].split("/* chart-math-end */", 1)[0]
    script = math_block + """
var first = pinnedEnd(10000, false, 4000);
var second = pinnedEnd(10100, false, 4000);
process.stdout.write(JSON.stringify({ first: first, second: second, live: pinnedEnd(10100, true, 4000) }));
"""
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["first"] == got["second"] == 4000
    assert got["live"] == 10100


def _playwright_node() -> Path | None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    candidate = Path(node_path) if node_path else None
    if candidate is not None and (candidate / "playwright").is_dir():
        return candidate
    return None


def test_webkit_scroll_holds_across_refreshes() -> None:
    node = _playwright_node()
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None:
        if required:
            raise AssertionError("CI requires the WebKit scroll regression (set SPARKSI_PLAYWRIGHT_NODE)")
        import pytest

        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const http = require("http");
const fs = require("fs");
const path = require("path");
const staticDir = process.env.STATIC_DIR;
const gpuCycle = [80, 96, 40];
let tick = 0;
let monitorN = 0;
let gpuPhase = null;
let lastMonitor = null;
let holdHistory = false;
const held = [];
function releaseHeld() {
  const queued = held.splice(0).reverse();
  queued.forEach((item) => {
    item.res.writeHead(200, { "Content-Type": "application/json", "Cache-Control": "no-store" });
    item.res.end(JSON.stringify(item.body));
  });
}
function historyBody() {
  const now = Math.floor(Date.now() / 1000);
  const samples = [];
  for (let t = now - 1200; t <= now; t += 20) {
    samples.push({ t, gpu_pct: 20 + (t % 30), cpu_pct: 8, mem_pct: 40, gpu_temp_c: 42 });
  }
  return { machine: "head", range: "1h", now, samples };
}
function payload(url) {
  const pathName = url.split("?")[0];
  tick += 1;
  if (pathName === "/control/delay") { holdHistory = true; return { ok: true }; }
  if (pathName === "/control/gpu-phase") {
    gpuPhase = Number(new URL(url, "http://127.0.0.1").searchParams.get("n"));
    return { ok: true, floor: monitorN };
  }
  if (pathName === "/api/config") return { nodes: [{ role: "head", id: "head", name: "head" }, { role: "worker", id: "worker", name: "worker" }], head: { id: "head", name: "head" }, worker: { id: "worker", name: "worker" }, controller: { name: "controller" }, archive: { id: "archive", label: "Archive" } };
  if (pathName === "/api/status") return { lock: { locked: true, on: "ON", reason: "held" } };
  if (pathName === "/api/monitor") {
    const gpu = gpuPhase === null ? gpuCycle[monitorN % gpuCycle.length] : gpuCycle[gpuPhase];
    monitorN += 1;
    lastMonitor = { gpu: gpu, seq: monitorN };
    const host = (role, used) => ({ id: role, name: role, age: monitorN % 5, fresh: true, mem_used_gib: used, mem_total_gib: 128, mem_pct: 30, cpu_pct: 10, gpu_pct: gpu, gpu_temp_c: 45, power_w: 40, zones: [] });
    return { ok: true, hosts: [host("head", 32), host("worker", 90)], loaded: [] };
  }
  if (pathName === "/api/token-day") return { ok: true, days: [{ total: { generation: 120, prompt: 340 }, lines: [] }] };
  if (pathName === "/api/token-life" || pathName === "/api/tokens") return { models: {} };
  if (pathName === "/api/fleet") return { library: [], storage: [], machines: [], hosts: [] };
  if (pathName === "/api/job") return { job: null };
  if (pathName === "/api/activity") return { events: [] };
  if (pathName === "/api/fleet-power") return { watts: 40, estimate: true };
  if (pathName === "/api/history") return historyBody();
  if (pathName === "/api/models") return { models: [{ id: "example", title: "Example Model" }], services: [], reserved_ports: [] };
  if (pathName === "/api/runs") {
    const port = tick % 2 ? 8014 : 8015;
    return { runs: [{ id: "run-1", model_id: "example", phase: "ready", placement: "both", port, started: 1700000000, nodes: [{ role: "head", phase: "ready" }, { role: "worker", phase: "ready" }] }] };
  }
  if (pathName === "/api/nodes") return { nodes: [] };
  if (pathName === "/api/telemetry") return { kept: "indefinitely", files: [], roles: { head: { first_t: 1700000000, last_t: 1700001000, samples: 2 }, worker: { first_t: null, last_t: 1700002000, samples: null } }, last_error: "", next: null };
  return null;
}
const server = http.createServer((req, res) => {
  const url = req.url || "/";
  const body = payload(url);
  if (body && url.split("?")[0] === "/api/history" && holdHistory) {
    held.push({ res, body });
    if (held.length >= 2) releaseHeld();
    return;
  }
  if (body) {
    const headers = { "Content-Type": "application/json", "Cache-Control": "no-store" };
    if (url.split("?")[0] === "/api/monitor" && lastMonitor) {
      headers["X-Gpu"] = String(lastMonitor.gpu);
      headers["X-Seq"] = String(lastMonitor.seq);
    }
    res.writeHead(200, headers);
    res.end(JSON.stringify(body));
    return;
  }
  const rel = url.split("?")[0];
  const file = (rel === "/" || rel === "/overview") ? "index.html" : rel.replace(/^\//, "");
  const full = path.join(staticDir, file);
  if (!full.startsWith(staticDir) || !fs.existsSync(full)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200);
  res.end(fs.readFileSync(full));
});
function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }
async function boot(browser, width, height, mobile) {
  const touch = mobile !== false && width < 900;
  const page = await browser.newPage({ viewport: { width, height }, isMobile: touch, hasTouch: touch, deviceScaleFactor: touch ? 2 : 1 });
  const errors = [];
  page.on("pageerror", (err) => errors.push(String(err)));
  await page.clock.install();
  await page.goto("http://127.0.0.1:" + server.address().port + "/overview", { waitUntil: "domcontentloaded" });
  await page.waitForSelector("#ribbon figure");
  await page.clock.runFor(1500);
  await sleep(200);
  return { page, errors };
}
async function at(page, fraction) {
  return page.evaluate((part) => {
    const max = Math.max(0, document.documentElement.scrollHeight - document.documentElement.clientHeight);
    const y = Math.round(max * part);
    window.scrollTo(0, y);
    return window.scrollY;
  }, fraction);
}
async function stable(page, ms) {
  const before = await page.evaluate(() => ({ y: window.scrollY, doc: document.documentElement.scrollHeight, range: (document.getElementById("chart-range") || {}).textContent || "" }));
  await page.clock.runFor(ms);
  await sleep(250);
  const after = await page.evaluate(() => ({ y: window.scrollY, doc: document.documentElement.scrollHeight, range: (document.getElementById("chart-range") || {}).textContent || "", high: document.body.innerText.indexOf("High") >= 0 }));
  return { before, after, dy: Math.round((after.y - before.y) * 100) / 100 };
}
server.listen(0, "127.0.0.1", async () => {
  const browser = await webkit.launch();
  const out = { engine: "playwright-webkit", cases: [], gestures: {} };
  const views = [[320, 700], [390, 844], [844, 390], [1440, 900]];
  for (const [width, height] of views) {
    const { page, errors } = await boot(browser, width, height, true);
    const rows = [];
    for (const part of [0, 0.5, 1]) {
      const y = await at(page, part);
      const trace = await stable(page, 4000);
      rows.push({ part, y, dy: trace.dy, doc: trace.after.doc });
      if (Math.abs(trace.dy) >= 1) throw new Error("jump " + width + " " + JSON.stringify(trace));
    }
    out.cases.push({ width, height, rows, errors });
    await page.close();
  }
  const { page, errors } = await boot(browser, 390, 844, false);
  await at(page, 0);
  await page.mouse.wheel(0, 600);
  let scrolled = 0;
  for (let i = 0; i < 12; i++) {
    await sleep(80);
    const y = await page.evaluate(() => window.scrollY);
    if (y === scrolled && y > 0) break;
    scrolled = y;
  }
  if (!(scrolled > 40)) throw new Error("wheel did not scroll " + scrolled);
  const held = await stable(page, 3000);
  if (Math.abs(held.dy) >= 1) throw new Error("wheel position jumped " + JSON.stringify(held));
  // The page is shorter without the big dials, so bring the chart on screen before dragging it.
  await page.locator("#fleet-chart").scrollIntoViewIfNeeded();
  await sleep(100);
  const chart = await page.locator("#fleet-chart").boundingBox();
  await page.mouse.move(chart.x + chart.width * 0.7, chart.y + 40);
  await page.mouse.down();
  await page.mouse.move(chart.x + chart.width * 0.3, chart.y + 40, { steps: 8 });
  await page.mouse.up();
  await sleep(200);
  const pinned = await page.locator("#chart-range").innerText();
  if (pinned.indexOf("Live") === 0) throw new Error("drag did not leave live: " + pinned);
  const afterDrag = await stable(page, 5000);
  const pinnedAfter = await page.locator("#chart-range").innerText();
  if (pinnedAfter !== pinned) throw new Error("delayed refresh yanked the chart " + pinned + " -> " + pinnedAfter);
  if (Math.abs(afterDrag.dy) >= 1) throw new Error("pan refresh jumped " + JSON.stringify(afterDrag));
  await page.locator("#fleet-chart svg").focus();
  await page.keyboard.press("ArrowLeft");
  await sleep(150);
  const keyed = await page.locator("#chart-range").innerText();
  if (keyed === pinned) throw new Error("keyboard did not pan");
  const afterKey = await stable(page, 3000);
  const keyedAfter = await page.locator("#chart-range").innerText();
  if (keyedAfter !== keyed) throw new Error("keyboard pin moved " + keyed + " -> " + keyedAfter);
  if (Math.abs(afterKey.dy) >= 1) throw new Error("keyboard refresh jumped");
  await page.evaluate(() => fetch("/control/delay"));
  await page.locator("#chart-older").click();
  await page.clock.runFor(2000);
  await sleep(400);
  const duringDelay = await page.locator("#chart-range").innerText();
  if (duringDelay.indexOf("Live") === 0) throw new Error("out-of-order history returned to live");
  await page.locator("#chart-live").click();
  await sleep(200);
  const live = await page.locator("#chart-range").innerText();
  if (live.indexOf("Live") !== 0) throw new Error("Live button failed: " + live);
  out.gestures = { scrolled, pinned, keyed, duringDelay, live, errors };
  const cycle = [];
  const y0 = await page.evaluate(() => window.scrollY);
  const gpus = [80, 96, 40];
  for (let phase = 0; phase < 3; phase++) {
    const mark = await page.evaluate(async (n) => {
      const res = await fetch("/control/gpu-phase?n=" + n);
      const body = await res.json();
      return body.floor;
    }, phase);
    for (let rep = 0; rep < 2; rep++) {
      const want = String(gpus[phase]);
      const pending = page.waitForResponse((resp) => {
        const pathName = resp.url().split("?")[0];
        return pathName.endsWith("/api/monitor") && resp.headers()["x-gpu"] === want && Number(resp.headers()["x-seq"]) > mark;
      });
      await page.clock.runFor(1100);
      await pending;
      await page.evaluate(() => Promise.resolve());
      const row = await page.evaluate(() => ({
        y: window.scrollY,
        text: Array.prototype.map.call(document.querySelectorAll("#ribbon .state"), function (node) { return node.textContent; }).join(" ")
      }));
      if (row.y !== y0) throw new Error("High cycle moved scroll " + JSON.stringify(row));
      if (row.text.indexOf(want + "%") < 0) throw new Error("gauge missed " + want + " " + row.text);
      cycle.push(row);
    }
  }
  const flags = cycle.map((row) => row.text.indexOf("High") >= 0);
  const below = cycle[1] && cycle[1].text.indexOf("80%") >= 0 && flags[1] === false;
  const above = cycle[3] && cycle[3].text.indexOf("96%") >= 0 && flags[3] === false;
  const belowAgain = cycle[5] && cycle[5].text.indexOf("40%") >= 0 && flags[5] === false;
  if (!(below && above && belowAgain)) throw new Error("High did not cross down " + JSON.stringify(cycle));
  if (cycle.some((row) => row.y !== y0)) throw new Error("High cycle moved scroll " + JSON.stringify(cycle));
  out.high = { flags, y0 };
  const headBtn = page.locator("button.legend-machine-name").first();
  await headBtn.click();
  const pinnedFocus = await page.evaluate(() => {
    const buttons = Array.prototype.slice.call(document.querySelectorAll("button.legend-machine-name"));
    const worker = document.querySelector('g[data-machine="worker"]');
    return {
      pressed: buttons.map((node) => node.getAttribute("aria-pressed")),
      focus: buttons.map((node) => node.className.indexOf("is-focus") >= 0),
      opacity: worker ? worker.style.opacity : ""
    };
  });
  if (pinnedFocus.pressed[0] !== "true" || pinnedFocus.focus[0] !== true) throw new Error("legend pin missed " + JSON.stringify(pinnedFocus));
  if (pinnedFocus.opacity !== "0.25") throw new Error("other machine stayed bright " + JSON.stringify(pinnedFocus));
  await page.clock.runFor(1100);
  await sleep(150);
  const afterRefresh = await headBtn.getAttribute("aria-pressed");
  if (afterRefresh !== "true") throw new Error("refresh cleared the legend pin");
  await headBtn.focus();
  await page.keyboard.press("Enter");
  const cleared = await headBtn.getAttribute("aria-pressed");
  if (cleared !== "false") throw new Error("keyboard did not clear the legend pin");
  out.legend = { pinned: pinnedFocus.pressed, afterRefresh, cleared };
  await page.close();
  await browser.close();
  server.close();
  process.stdout.write(JSON.stringify(out));
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
            import pytest

            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["engine"] == "playwright-webkit"
    # The High word is gone: crossing 90% changes the dial, never adds a word.
    assert got["high"]["flags"] == [False] * 6
    assert got["gestures"]["scrolled"] > 40
    assert not got["gestures"]["pinned"].startswith("Live")
    assert got["gestures"]["live"].startswith("Live")
    for case in got["cases"]:
        assert case["errors"] == []
        for row in case["rows"]:
            assert abs(row["dy"]) < 1, row
