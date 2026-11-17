# SPDX-License-Identifier: Apache-2.0

"""Network add-by-address row: labels, inputs and button line up, and password managers are told to skip it."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

PROBE = r"""
(async () => {
  const { webkit } = require("playwright");
  const port = process.env.PAGE_PORT;
  const widths = [1440, 800, 390];
  const errors = [];
  const rows = {};
  const browser = await webkit.launch();
  try {
    for (const w of widths) {
      const context = await browser.newContext({ viewport: { width: w, height: 900 } });
      const page = await context.newPage();
      page.on("pageerror", (e) => errors.push(String(e)));
      await page.goto(`http://127.0.0.1:${port}/network`, { waitUntil: "networkidle" });
      await page.waitForSelector("#net-manual-add");
      const measure = () => page.evaluate(() => {
        const ids = ["net-manual-address", "net-manual-port", "net-manual-name", "net-manual-add"];
        const out = {};
        for (const id of ids) {
          const el = document.getElementById(id);
          const r = el.getBoundingClientRect();
          const item = {
            top: Math.round(r.top),
            bottom: Math.round(r.bottom),
            left: Math.round(r.left),
            right: Math.round(r.right),
          };
          const label = el.closest("label");
          if (label) {
            const text = label.querySelector(".net-field-label");
            if (text) item.labelTop = Math.round(text.getBoundingClientRect().top);
          }
          out[id] = item;
        }
        return out;
      });
      const before = await measure();
      const injected = await page.evaluate(() => {
        const address = document.getElementById("net-manual-address");
        const s = document.createElement("span");
        s.style.cssText = "display:inline-block;width:20px;height:24px";
        address.after(s);
      });
      const injectedRow = injected === undefined ? await measure() : null;
      const attrs = await page.evaluate(() => {
        const ids = ["net-manual-address", "net-manual-port", "net-manual-name"];
        const names = ["autocomplete", "data-1p-ignore", "data-lpignore", "data-bwignore", "data-form-type"];
        const out = {};
        for (const id of ids) {
          const el = document.getElementById(id);
          const o = {};
          for (const n of names) {
            const v = el.getAttribute(n);
            o[n] = v === null ? null : v;
          }
          out[id] = o;
        }
        return out;
      });
      const docW = await page.evaluate(() => document.documentElement.scrollWidth);
      const shotDir = process.env.SPARKSI_SHOT_DIR;
      if (shotDir && w === 1440) {
        await page.evaluate(() => document.getElementById("net-manual").scrollIntoView());
        const fs = require("fs");
        const path = require("path");
        fs.mkdirSync(shotDir, { recursive: true });
        await page.locator("#view-network").screenshot({ path: path.join(shotDir, "network-add-row.png") });
      }
      rows[String(w)] = { before, injected: injectedRow, attrs, docW };
      await context.close();
    }
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify({ errors, rows }));
})().catch((e) => { console.error(e); process.exit(1); });
"""


@pytest.fixture()
def add_row_server(monkeypatch, tmp_path):
    cfg = tmp_path / "sparksi.toml"
    cfg.write_bytes((ROOT / "config" / "sparksi.example.toml").read_bytes())
    monkeypatch.setenv("SPARKSI_CONFIG", str(cfg))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
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
            raise AssertionError("CI requires the WebKit network add-row regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_network_add_row(add_row_server):
    port, _tmp = add_row_server
    data = _run(PROBE, port)
    assert data["errors"] == []
    rows = data["rows"]
    ids = ["net-manual-address", "net-manual-port", "net-manual-name", "net-manual-add"]
    inputs = ids[:3]
    expected_attrs = {
        "autocomplete": "off",
        "data-1p-ignore": "",
        "data-lpignore": "true",
        "data-bwignore": "",
        "data-form-type": "other",
    }
    for width in ("1440", "800", "390"):
        row = rows[width]
        w = int(width)
        assert row["docW"] <= w
        for phase in ("before", "injected"):
            m = row[phase]
            if w in (1440, 800):
                tops = {m[i]["top"] for i in ids}
                bottoms = {m[i]["bottom"] for i in ids}
                assert len(tops) == 1
                assert len(bottoms) == 1
                label_tops = {m[i]["labelTop"] for i in inputs}
                assert len(label_tops) == 1
                for prev, cur in zip(ids, ids[1:]):
                    assert m[cur]["left"] >= m[prev]["right"]
            else:
                seq = [m[i] for i in ids]
                for prev, cur in zip(seq, seq[1:]):
                    assert cur["top"] >= prev["bottom"]
                lefts = {e["left"] for e in seq}
                rights = {e["right"] for e in seq}
                assert len(lefts) == 1
                assert len(rights) == 1
        assert row["injected"] == row["before"]
        for i in inputs:
            assert row["attrs"][i] == expected_attrs
