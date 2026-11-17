# SPDX-License-Identifier: Apache-2.0

"""Network page setup: roles saved to sparksi.toml (backup kept), Remove, add by address or port, name limits, slow saves, the 4-machine limit and the lock bar, in WebKit at 400 px."""

import json
import os
import subprocess
import threading
import tomllib
from pathlib import Path

import pytest

from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

SCAN = {
    "scanned_at": "2026-01-01T12:00:00Z", "took_s": 2.4,
    "hosts": [
        {"name": "ctl", "dns": "ctl.example.ts.net", "ips": ["100.64.0.1"], "sources": ["tailscale"], "self": True,
         "role": "controller", "why": "this machine", "endpoints": [], "proposal": ""},
        {"name": "spark-one", "dns": "spark-one.example.ts.net", "ips": ["100.64.0.11", "192.0.2.11"], "sources": ["lan", "tailscale"],
         "self": False, "role": "spark", "why": "name looks like a DGX Spark",
         "endpoints": [{"port": 8014, "engine": "vLLM", "models": ["example/model-a"], "url": "http://100.64.0.11:8014"}],
         "proposal": "[[nodes]]\nid = \"spark-one\"\nname = \"spark-one\"\nssh = \"spark-one\"\ntailscale = \"100.64.0.11\""},
        {"name": "a-very-long-machine-name-that-should-wrap-on-a-phone.example.ts.net", "dns": "", "ips": ["100.64.0.20"],
         "sources": ["tailscale"], "self": False, "role": "inference", "why": "serves an inference API",
         "endpoints": [{"port": 11434, "engine": "Ollama", "models": ["small:7b", "tiny:1b"], "url": "http://100.64.0.20:11434"}],
         "proposal": "# A monitored inference host (Network page: Add to Sparksi).\n[[endpoints]]\nname = \"x\"\nurl = \"http://100.64.0.20:11434\"\nengine = \"Ollama\""},
    ],
}

PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const browser = await webkit.launch();
  const out = { errors: [], dialogs: [] };
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => out.errors.push(String(err)));
  page.on("dialog", (d) => { out.dialogs.push(d.message()); d.accept(); });
  await page.route("**/api/discover", (route) => {
    const req = route.request();
    if (req.method() === "POST") {
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) });
    }
    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({ scanned_at: null, hosts: [], took_s: null, running: false }),
    });
  });
  const rowText = () => Array.from(document.querySelectorAll("#net-layout-list .net-row"))
    .map((r) => r.innerText.replace(/\s+/g, " ").trim());
  try {
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#net-layout-list .net-row").length >= 3);
    out.rows1 = await page.evaluate(rowText);

    await page.evaluate(() => {
      const row = Array.from(document.querySelectorAll("#net-layout-list .net-row"))
        .find((r) => r.querySelector("strong").textContent === "Head Spark");
      row.querySelector("select.net-role-pick").value = "worker";
      row.querySelector("select.net-role-pick").dispatchEvent(new Event("change", { bubbles: true }));
    });
    await page.waitForFunction(() => document.getElementById("net-layout-note").textContent.startsWith("Saved:"));
    out.note1 = await page.textContent("#net-layout-note");
    out.rows2 = await page.evaluate(rowText);

    await page.click("#btn-discover");
    await page.waitForFunction(() => document.querySelectorAll(".net-host").length === 3);
    await page.evaluate(() => {
      const host = document.querySelectorAll(".net-host")[2];
      host.querySelector(".net-add").click();
    });
    await page.waitForFunction(() => {
      return Array.from(document.querySelectorAll("#net-layout-list .net-row strong"))
        .some((s) => s.textContent === "Monitored");
    });
    out.rows3 = await page.evaluate(rowText);

    await page.evaluate(() => {
      const row = Array.from(document.querySelectorAll("#net-layout-list .net-row"))
        .find((r) => r.querySelector("strong").textContent === "Monitored");
      row.querySelector("button.net-remove").click();
    });
    await page.waitForFunction(() => {
      return !Array.from(document.querySelectorAll("#net-layout-list .net-row strong"))
        .some((s) => s.textContent === "Monitored");
    });
    out.rows4 = await page.evaluate(rowText);

    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


@pytest.fixture()
def network_server(monkeypatch, tmp_path):
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


def _run(probe: str, port: int, scan: dict = SCAN) -> dict:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit network setup regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port), SCAN=json.dumps(scan))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_network_setup_page(network_server):
    port, tmp_path = network_server
    out = _run(PROBE, port)

    assert not out.get("crash"), out
    assert out["errors"] == [], out

    rows1 = out["rows1"]
    assert any("Head Spark" in r and "spark-a" in r for r in rows1)
    assert any("Worker Spark" in r and "spark-b" in r for r in rows1)

    rows2 = out["rows2"]
    assert any("Head Spark" in r and "spark-b" in r for r in rows2)

    assert "head and worker swapped" in out["note1"]
    assert "Backup: sparksi.toml.bak-" in out["note1"]

    rows3 = out["rows3"]
    assert any("Monitored" in r and "Ollama" in r for r in rows3)

    rows4 = out["rows4"]
    assert not any("Monitored" in r for r in rows4)

    assert len(out["dialogs"]) == 3
    for msg in out["dialogs"]:
        assert "Nothing starts or stops" in msg

    assert out["docW"] <= 400

    cfg = tomllib.loads((tmp_path / "sparksi.toml").read_text())
    assert [n["id"] for n in cfg["nodes"]] == ["b", "a"]
    assert "endpoints" not in cfg

    backups = list(tmp_path.glob("sparksi.toml.bak-*"))
    assert backups


def test_network_locked_rows_disabled(network_server):
    port, _tmp_path = network_server
    STATE.lock.acquire("test")

    probe = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const browser = await webkit.launch();
  const out = { errors: [] };
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => out.errors.push(String(err)));
  try {
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#net-layout-list .net-row").length >= 3);
    out.picks = await page.evaluate(() => Array.from(
      document.querySelectorAll("#net-layout-list .net-role-pick")
    ).map((s) => ({ disabled: s.disabled, title: s.title })));
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""
    out = _run(probe, port)

    assert not out.get("crash")
    assert out["errors"] == []
    assert out["picks"]
    for pick in out["picks"]:
        assert pick["disabled"] is True
        assert pick["title"] == "Unlock first (Unlock, above) to change the setup."


SCAN_LIMIT = {
    "scanned_at": "2026-01-02T09:00:00Z", "took_s": 1.2,
    "hosts": [
        {"name": "ctl", "dns": "ctl.example.ts.net", "ips": ["100.64.0.1"], "sources": ["tailscale"], "self": True,
         "role": "controller", "why": "this machine", "endpoints": [], "proposal": ""},
        {"name": "", "dns": "", "ips": ["192.0.2.1"], "role": "machine", "why": "answers on the LAN", "sources": ["lan"],
         "endpoints": [{"port": 80, "engine": "open port", "url": "http://192.0.2.1:80", "models": []}],
         "self": False, "proposal": ""},
        {"name": "box-two", "dns": "box-two.example.ts.net", "ips": ["100.64.0.30"], "role": "machine", "why": "on the tailnet",
         "sources": ["tailscale"], "endpoints": [], "self": False, "proposal": ""},
    ],
}

PROBE_LIMIT = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const browser = await webkit.launch();
  const out = { errors: [], dialogs: [] };
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (err) => out.errors.push(String(err)));
  page.on("dialog", (d) => { out.dialogs.push({type: d.type(), message: d.message()}); d.type() === "prompt" ? d.accept("8000") : d.accept(); });
  await page.route("**/api/discover", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) }));
  try {
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#net-layout-list .net-row").length >= 3);

    await page.click("#btn-discover");
    await page.waitForFunction(() => document.querySelectorAll(".net-host").length === 3);

    await page.evaluate(() => {
      document.querySelectorAll(".net-host")[1].querySelector(".net-add").click();
    });
    await page.waitForFunction(() => document.getElementById("net-layout-note").textContent.startsWith("Saved:"));
    out.note1 = await page.textContent("#net-layout-note");

    await page.fill("#net-manual-address", "198.51.100.4");
    await page.fill("#net-manual-port", "8000");
    await page.fill("#net-manual-name", "typed-box");
    await page.click("#net-manual-add");
    await page.waitForFunction(() => /^Saved: now monitoring typed-box/.test(document.getElementById("net-layout-note").textContent));
    out.note2 = await page.textContent("#net-layout-note");

    await page.evaluate(() => {
      document.querySelectorAll(".net-host")[2].querySelector(".net-add").click();
    });
    await page.waitForFunction(() => document.getElementById("net-layout-note").textContent.startsWith("Not saved:"));
    out.note3 = await page.textContent("#net-layout-note");

    out.monitoredRows = await page.evaluate(() => Array.from(
      document.querySelectorAll("#net-layout-list .net-row")
    ).filter((r) => r.querySelector("strong").textContent === "Monitored").length);
    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def test_network_machine_limit_and_manual_add(network_server):
    """An IP-only machine with nothing answering, then one typed by address, fill the 4; a fifth is refused."""
    port, tmp_path = network_server
    out = _run(PROBE_LIMIT, port, SCAN_LIMIT)

    assert not out.get("crash"), out
    assert out["errors"] == [], out

    dialogs = out["dialogs"]
    assert dialogs, out
    assert dialogs[0]["type"] == "prompt", out
    assert "192.0.2.1" in dialogs[0]["message"], out
    assert "offline" in dialogs[0]["message"], out
    for msg in dialogs:
        assert "Add  to" not in msg["message"], out
    assert any("Add 192.0.2.1 to sparksi.toml" in m["message"] and
               "Nothing answers on this port right now; it will show as offline until it does." in m["message"]
               for m in dialogs), out
    assert any("Add typed-box to sparksi.toml" in m["message"] for m in dialogs), out

    assert "nothing answers on this port right now" in out["note1"], out
    assert "up to 2 monitored hosts" in out["note3"], out
    assert "remove a monitored host first" in out["note3"], out

    assert out["monitoredRows"] == 2, out
    assert out["docW"] <= 400, out

    cfg = tomllib.loads((tmp_path / "sparksi.toml").read_text())
    assert [e["url"] for e in cfg["endpoints"]] == ["http://192.0.2.1:8000", "http://198.51.100.4:8000"], out
    assert [e["name"] for e in cfg["endpoints"]] == ["192.0.2.1", "typed-box"], out


PROBE_FORM = r"""
const { webkit } = require("playwright");
(async () => {
  const out = { dialogs: [], pageerrors: [], crash: null };
  const browser = await webkit.launch();
  try {
    const page = await browser.newPage({
      viewport: { width: 400, height: 860 },
    });
    page.on("pageerror", (err) => out.pageerrors.push(String(err)));
    page.on("dialog", async (dialog) => {
      out.dialogs.push(dialog.message());
      await dialog.accept();
    });
    const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    async function submit(addr, port, name) {
      await page.fill("#net-manual-address", addr);
      await page.fill("#net-manual-port", port);
      await page.fill("#net-manual-name", name);
      const before = out.dialogs.length;
      await page.click("#net-manual-add");
      return before;
    }
    const note = () => page.textContent("#net-layout-note");

    let before = await submit("8.8.8.8", "8000", "");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.bad1 = {
      note: await note(),
      dialogsAdded: out.dialogs.length - before,
    };

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    before = await submit("bad host", "8000", "");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.bad2 = {
      note: await note(),
      dialogsAdded: out.dialogs.length - before,
    };

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    before = await submit("198.51.100.4", "0", "");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.bad3 = {
      note: await note(),
      dialogsAdded: out.dialogs.length - before,
    };

    before = await submit("198.51.100.4", "8000", "one");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Saved:")
    );
    out.afterAdd = {
      address: await page.inputValue("#net-manual-address"),
      port: await page.inputValue("#net-manual-port"),
      name: await page.inputValue("#net-manual-name"),
    };

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    await submit("198.51.100.5", "8000", "one");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.dup = await note();

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    await submit("198.51.100.6", "8000", "two");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Saved:")
    );

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    before = await submit("8.8.8.8", "8000", "three");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.fullBad = {
      note: await note(),
      dialogsAdded: out.dialogs.length - before,
    };

    await page.evaluate(() => {
      document.getElementById("net-layout-note").textContent = "";
    });
    await submit("198.51.100.7", "8000", "three");
    await page.waitForFunction(
      () => document.getElementById("net-layout-note").textContent.startsWith("Not saved:")
    );
    out.fullOk = await note();
  } catch (err) {
    out.crash = String(err);
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(out));
})();
"""


def test_network_setup_form_flow(network_server):
    """The add-by-address form checks input before any confirm (also when full), refuses a taken name, and clears after a save."""
    port, tmp_path = network_server
    out = _run(PROBE_FORM, port)

    assert out["crash"] is None, out
    assert out["pageerrors"] == [], out

    assert "host name or a private or tailnet IP" in out["bad1"]["note"]
    assert out["bad1"]["dialogsAdded"] == 0
    assert "host name or a private or tailnet IP" in out["bad2"]["note"]
    assert out["bad2"]["dialogsAdded"] == 0

    assert "1 to 65535" in out["bad3"]["note"]
    assert out["bad3"]["dialogsAdded"] == 0

    assert out["afterAdd"] == {"address": "", "port": "8000", "name": ""}

    assert out["dup"] == (
        "Not saved: a machine named one is already in the setup; pick another name."
    )

    assert "private or tailnet" in out["fullBad"]["note"]
    assert out["fullBad"]["dialogsAdded"] == 0

    assert "up to 2 monitored hosts" in out["fullOk"]

    with open(tmp_path / "sparksi.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    names = [ep["name"] for ep in cfg["endpoints"]]
    assert names == ["one", "two"]


SCAN_STALL = {
    "hosts": [
        {
            "name": "spark-b",
            "dns": "spark-b.example.ts.net",
            "ips": ["100.64.0.70"],
            "role": "spark",
            "why": "name looks like a DGX Spark",
            "sources": ["tailscale"],
            "endpoints": [],
            "self": False,
            "proposal": "",
        },
        {
            "name": "box-three",
            "dns": "box-three.example.ts.net",
            "ips": ["100.64.0.33"],
            "role": "inference",
            "why": "serves an inference API",
            "sources": ["tailscale"],
            "endpoints": [
                {
                    "port": 8000,
                    "engine": "vLLM",
                    "models": ["example/m"],
                    "url": "http://100.64.0.33:8000",
                }
            ],
            "self": False,
            "proposal": "",
        },
    ],
    "scanned_at": "2026-01-03T09:00:00Z",
    "took_s": 1.0,
}

PROBE_STALL = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let stall = true;
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.route("**/api/network/machine", async (route) => {
      if (stall) return new Promise(() => {});
      return route.continue();
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");
    await page.click("#btn-discover");
    await page.waitForFunction(() => document.querySelectorAll(".net-host").length === 2);

    await page.fill("#net-manual-address", "100.64.0.40");
    await page.fill("#net-manual-port", "8000");
    await page.fill("#net-manual-name", "spark-a");
    await page.click("#net-manual-add");
    await page.waitForTimeout(200);
    out.dupTyped = {
      note: await page.textContent("#net-layout-note"),
      dialogs: out.dialogs.length,
    };

    await page.click(".net-host >> nth=0 >> .net-add");
    await page.waitForTimeout(200);
    out.dupDiscover = {
      note: await page.textContent("#net-layout-note"),
      dialogs: out.dialogs.length,
    };

    const boxThreeAdd = page.locator(".net-host").nth(1).locator(".net-add");
    const t0 = Date.now();
    await boxThreeAdd.click();
    await page.waitForTimeout(100);
    out.during = {
      note: (await page.textContent("#net-layout-note")).trim(),
      disabled: await boxThreeAdd.isDisabled(),
    };
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Not saved:"),
      null,
      { timeout: 10000 }
    );
    out.afterStall = {
      note: (await page.textContent("#net-layout-note")).trim(),
      ms: Date.now() - t0,
      addDisabled: await page.$$eval(".net-add", (els) => els.some((e) => e.disabled)),
      formDisabled: await page.$eval("#net-manual-add", (e) => e.disabled),
    };

    stall = false;
    await boxThreeAdd.click();
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Saved:"),
      null,
      { timeout: 10000 }
    );
    out.saved = (await page.textContent("#net-layout-note")).trim();
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

def test_network_stall_and_retry(network_server):
    """Taken names are refused before any confirm; a save with no reply re-reads the layout, says the change is not in it, re-enables everything, and a retry works."""
    port, tmp_path = network_server
    res = _run(PROBE_STALL, port, SCAN_STALL)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["dupTyped"]["note"].strip() == (
        "Not saved: a machine named spark-a is already in the setup; pick another name."
    )
    assert res["dupTyped"]["dialogs"] == 0
    assert res["dupDiscover"]["note"].strip() == (
        "Not saved: a machine named spark-b is already in the setup; change its role under This setup."
    )
    assert res["dupDiscover"]["dialogs"] == 0
    assert res["during"]["note"] == "Saving…"
    assert res["during"]["disabled"] is True
    assert res["afterStall"]["note"] == "Not saved: no answer from Sparksi, and the change is not in the setup. Try again."
    assert 1000 <= res["afterStall"]["ms"] < 8000
    assert res["afterStall"]["addDisabled"] is False
    assert res["afterStall"]["formDisabled"] is False
    assert "now monitoring box-three" in res["saved"]
    assert len(res["dialogs"]) == 2
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    assert [e["name"] for e in cfg.get("endpoints", [])] == ["box-three"]


PROBE_NAME_LIMITS = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");
    await page.click("#btn-discover");
    await page.waitForFunction(() => document.querySelectorAll(".net-host").length === 1);

    await page.click(".net-host >> nth=0 >> .net-add");
    await page.waitForTimeout(200);
    out.longDiscover = {
      note: (await page.textContent("#net-layout-note")).trim(),
      dialogs: out.dialogs.length,
    };

    const longName = "x".repeat(101);
    await page.fill("#net-manual-address", "100.64.0.30");
    await page.fill("#net-manual-port", "8000");
    await page.fill("#net-manual-name", longName);
    await page.click("#net-manual-add");
    await page.waitForTimeout(200);
    out.long = {
      note: (await page.textContent("#net-layout-note")).trim(),
      dialogs: out.dialogs.length,
    };

    // Typing drops some control characters, so set the value the way a paste or script could.
    await page.$eval("#net-manual-name", (e) => { e.value = "box\u0007four"; });
    await page.click("#net-manual-add");
    await page.waitForTimeout(200);
    out.ctrl = {
      note: (await page.textContent("#net-layout-note")).trim(),
      dialogs: out.dialogs.length,
    };

    // Typing drops some control characters, so set the value the way a paste or script could.
    await page.$eval("#net-manual-name", (e) => { e.value = "box\u2028four"; });
    await page.click("#net-manual-add");
    await page.waitForTimeout(200);
    out.lsep = {
      note: (await page.textContent("#net-layout-note")).trim(),
      dialogs: out.dialogs.length,
    };

    await page.fill("#net-manual-name", "box-four");
    await page.click("#net-manual-add");
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Saved:"),
      null,
      { timeout: 10000 }
    );
    out.saved = (await page.textContent("#net-layout-note")).trim();
    out.savedDialogs = out.dialogs.length;
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

PROBE_SLOW_SAVE = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.route("**/api/network/machine", async (route) => {
      const resp = await route.fetch();
      await new Promise((r) => setTimeout(r, 4000));
      await route.fulfill({ response: resp }).catch(() => {});
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    await page.fill("#net-manual-address", "100.64.0.50");
    await page.fill("#net-manual-port", "8000");
    await page.fill("#net-manual-name", "box-four");
    await page.click("#net-manual-add");
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Saved"),
      null,
      { timeout: 10000 }
    );
    out.note = (await page.textContent("#net-layout-note")).trim();
    out.dialogCount = out.dialogs.length;
    out.addDisabled = await page.$$eval(".net-add", (els) => els.some((e) => e.disabled));
    out.formDisabled = await page.$eval("#net-manual-add", (e) => e.disabled);
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

PROBE_LOCK_BAR = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    out.initial = {
      lockText: (await page.textContent("#net-lock-text")).trim(),
      lockVisible: await page.isVisible("#net-btn-lock"),
      unlockVisible: await page.isVisible("#net-btn-unlock"),
      lockMsg: (await page.textContent("#net-lock-msg")).trim(),
    };

    await page.click("#net-btn-lock");
    await page.waitForFunction(
      () => document.querySelector("#net-lock-text").textContent.trim().startsWith("Locked"),
      null,
      { timeout: 10000 }
    );
    out.locked = {
      lockText: (await page.textContent("#net-lock-text")).trim(),
      unlockVisible: await page.isVisible("#net-btn-unlock"),
      lockVisible: await page.isVisible("#net-btn-lock"),
      formDisabled: await page.$eval("#net-manual-add", (e) => e.disabled),
      rolePicks: await page.$$eval("select.net-role-pick", (els) => els.length),
      rolePicksDisabled: await page.$$eval("select.net-role-pick", (els) =>
        els.every((e) => e.disabled)
      ),
      status: await page.evaluate(async () => {
        const r = await fetch("/api/status");
        const j = await r.json();
        return j.lock ? j.lock.on : null;
      }),
    };

    await page.click("#net-btn-unlock");
    await page.waitForFunction(
      () => document.querySelector("#net-lock-text").textContent.trim().startsWith("Unlocked"),
      null,
      { timeout: 10000 }
    );
    out.unlocked = {
      lockText: (await page.textContent("#net-lock-text")).trim(),
      lockVisible: await page.isVisible("#net-btn-lock"),
      unlockVisible: await page.isVisible("#net-btn-unlock"),
      formDisabled: await page.$eval("#net-manual-add", (e) => e.disabled),
      rolePicksEnabled: await page.$$eval("select.net-role-pick", (els) =>
        els.length > 0 && els.every((e) => !e.disabled)
      ),
      status: await page.evaluate(async () => {
        const r = await fetch("/api/status");
        const j = await r.json();
        return j.lock ? j.lock.on : null;
      }),
    };
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

SCAN_NAME_LIMITS = {
    "hosts": [
        {
            "name": "h" * 101,
            "dns": "box-long.example.ts.net",
            "ips": ["100.64.0.70"],
            "role": "inference",
            "why": "serves an inference API",
            "sources": ["tailscale"],
            "endpoints": [{"port": 8000, "engine": "vLLM", "models": ["example/model-a"], "url": "http://100.64.0.70:8000"}],
            "self": False,
            "proposal": "",
        }
    ]
}

SCAN_SLOW_SAVE = {
    "hosts": [
        {
            "name": "spark-c",
            "dns": "spark-c.example.ts.net",
            "ips": ["100.64.0.14"],
            "role": "spark",
            "why": "name looks like a DGX Spark",
            "sources": ["tailscale"],
            "endpoints": [],
            "self": False,
            "proposal": "",
        }
    ]
}


def test_network_name_limits_before_confirm(network_server):
    """Bad names are refused before any confirm dialog; a good name still saves."""
    port, tmp_path = network_server
    res = _run(PROBE_NAME_LIMITS, port, SCAN_NAME_LIMITS)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["longDiscover"]["note"] == "Not saved: a name must be 1 to 100 characters."
    assert res["longDiscover"]["dialogs"] == 0
    assert res["long"]["note"] == "Not saved: a name must be 1 to 100 characters."
    assert res["long"]["dialogs"] == 0
    assert res["ctrl"]["note"] == (
        "Not saved: a name must not contain control or line-break characters."
    )
    assert res["ctrl"]["dialogs"] == 0
    assert res["lsep"]["note"] == (
        "Not saved: a name must not contain control or line-break characters."
    )
    assert res["lsep"]["dialogs"] == 0
    assert "box-four" in res["saved"]
    assert res["savedDialogs"] == 1
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    names = [e["name"] for e in cfg.get("endpoints", [])]
    assert names == ["box-four"]


def test_network_slow_save_is_reported_saved(network_server):
    """A save whose reply is slower than the timeout still lands: the page says so via the layout."""
    port, tmp_path = network_server
    res = _run(PROBE_SLOW_SAVE, port, SCAN_SLOW_SAVE)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["note"] == (
        "Saved (the reply was slow). Restart Sparksi to use the new setup."
    )
    assert res["dialogCount"] == 1
    assert res["addDisabled"] is False
    assert res["formDisabled"] is False
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    names = [e["name"] for e in cfg.get("endpoints", [])]
    assert names == ["box-four"]


def test_network_lock_bar(network_server):
    """The lock bar reflects server state; locking disables editing, unlocking restores it."""
    port, _tmp_path = network_server
    res = _run(PROBE_LOCK_BAR, port, {})
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    init = res["initial"]
    assert init["lockText"].startswith("Unlocked"), init
    assert init["lockVisible"] is True
    assert init["unlockVisible"] is False
    locked = res["locked"]
    assert locked["lockText"].startswith("Locked"), locked
    assert locked["unlockVisible"] is True
    assert locked["lockVisible"] is False
    assert locked["formDisabled"] is True
    assert locked["rolePicks"] > 0
    assert locked["rolePicksDisabled"] is True
    assert locked["status"] == "ON"
    unlocked = res["unlocked"]
    assert unlocked["lockText"].startswith("Unlocked")
    assert unlocked["lockVisible"] is True
    assert unlocked["unlockVisible"] is False
    assert unlocked["formDisabled"] is False
    assert unlocked["rolePicksEnabled"] is True
    assert unlocked["status"] != "ON"


SCAN_TIMEOUT_NO_PICTURE = {
    "hosts": [
        {
            "name": "box-six", "dns": "box-six.example.ts.net", "ips": ["100.64.0.61"], "role": "inference", "why": "serves an inference API",
            "sources": ["tailscale"], "self": False, "proposal": "",
            "endpoints": [{"port": 8000, "engine": "vLLM", "models": ["example/model-a"], "url": "http://100.64.0.61:8000"}],
        }
    ]
}

SCAN_TIMEOUT_OUTSIDE = {"hosts": []}

SCAN_TIMEOUT_LANDED = {"hosts": []}

SCAN_NAME_TRIM = {
    "hosts": [
        {
            "name": "", "dns": "", "ips": ["  ", "x" * 101], "role": "inference", "why": "serves an inference API",
            "sources": ["tailscale"], "self": False, "proposal": "",
            "endpoints": [{"port": 8000, "engine": "vLLM", "models": ["example/model-b"], "url": "http://100.64.0.67:8000"}],
        }
    ]
}

PROBE_TIMEOUT_NO_PICTURE = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/network/layout", async (route) => {
      if (out.layoutHits === undefined) out.layoutHits = 0;
      out.layoutHits += 1;
      if (out.layoutHits === 1) {
        await route.fulfill({ status: 500, contentType: "application/json", body: "{}" });
      } else {
        await route.continue();
      }
    });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.route("**/api/network/machine", async (route) => {
      await route.fetch();
      await new Promise(() => {});
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#btn-discover:not([disabled])");
    await page.click("#btn-discover");
    await page.waitForSelector(".net-host");
    out.layoutHitsBeforeSave = out.layoutHits;
    await page.click(".net-host .net-add");
    await page.waitForFunction(
      () => {
        const t = document.querySelector("#net-layout-note").textContent.trim();
        return t.startsWith("Not saved:") || t.startsWith("Saved");
      },
      null,
      { timeout: 15000 }
    );
    out.note = (await page.textContent("#net-layout-note")).trim();
    out.dialogCount = out.dialogs.length;
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

PROBE_TIMEOUT_OUTSIDE = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.route("**/api/network/machine", async (route) => {
      await route.fetch({
        postData: JSON.stringify({
          action: "add",
          role: "monitor",
          machine: { address: "100.64.0.62", port: 8001, name: "box-seven" },
          from_ui: true,
        }),
      });
      await new Promise(() => {});
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    await page.fill("#net-manual-address", "100.64.0.63");
    await page.fill("#net-manual-port", "8002");
    await page.fill("#net-manual-name", "box-eight");
    await page.click("#net-manual-add");
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Not saved:"),
      null,
      { timeout: 15000 }
    );
    out.note = (await page.textContent("#net-layout-note")).trim();
    out.dialogCount = out.dialogs.length;
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

PROBE_TIMEOUT_LANDED = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/network/machine", async (route) => {
      const resp = await route.fetch();
      await new Promise((r) => setTimeout(r, 4000));
      await route.fulfill({ response: resp }).catch(() => {});
    });
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    const waitSaved = async () => {
      await page.waitForFunction(
        () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Saved ("),
        null,
        { timeout: 15000 }
      );
      return (await page.textContent("#net-layout-note")).trim();
    };

    await page.fill("#net-manual-address", "100.64.0.64");
    await page.fill("#net-manual-port", "8003");
    await page.fill("#net-manual-name", "box-nine");
    await page.click("#net-manual-add");
    out.noteA = await waitSaved();

    await page.$$eval("#net-layout-list .net-row", (rows) => {
      const row = rows.find((r) => {
        const s = r.querySelector("strong");
        return s && s.textContent.trim() === "Head Spark";
      });
      const sel = row.querySelector("select.net-role-pick");
      sel.value = "worker";
      sel.dispatchEvent(new Event("change", { bubbles: true }));
    });
    out.noteB = await waitSaved();

    await page.$$eval("#net-layout-list .net-row", (rows) => {
      const row = rows.find((r) => {
        const s = r.querySelector("strong");
        return s && s.textContent.trim() === "Monitored";
      });
      row.querySelector("button.net-remove").click();
    });
    out.noteC = await waitSaved();
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''

PROBE_NAME_TRIM = r'''const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const scan = JSON.parse(process.env.SCAN);
  const out = { errors: [], dialogs: [] };
  let browser;
  try {
    browser = await webkit.launch();
    const ctx = await browser.newContext({ viewport: { width: 400, height: 860 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => out.errors.push(String(e)));
    page.on("dialog", async (d) => {
      out.dialogs.push(d.message());
      await d.accept();
    });
    await page.addInitScript(() => { window.SPARKSI_NET_SAVE_TIMEOUT_MS = 1500; });
    await page.route("**/api/discover", (route) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(scan) })
    );
    await page.goto(base + "/network", { waitUntil: "networkidle" });
    await page.waitForSelector("#net-manual-add:not([disabled])");

    const waitNote = async (prefix) => {
      await page.waitForFunction(
        (p) => document.querySelector("#net-layout-note").textContent.trim().startsWith(p),
        prefix,
        { timeout: 15000 }
      );
      return (await page.textContent("#net-layout-note")).trim();
    };

    await page.fill("#net-manual-address", "100.64.0.65");
    await page.fill("#net-manual-port", "8004");
    await page.$eval("#net-manual-name", (e) => {
      e.value = "\u001c\u001d\u001e\u001fbox-ten\u0085";
    });
    await page.click("#net-manual-add");
    out.noteA = await waitNote("Saved:");
    out.dialogsAfterA = out.dialogs.length;

    await page.fill("#net-manual-address", "100.64.0.66");
    await page.fill("#net-manual-port", "8005");
    await page.$eval("#net-manual-name", (e) => {
      e.value = "\ufeffbox-eleven";
    });
    await page.click("#net-manual-add");
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Not saved:"),
      null,
      { timeout: 15000 }
    );
    out.noteB = (await page.textContent("#net-layout-note")).trim();
    out.dialogsAfterB = out.dialogs.length;

    await page.waitForSelector("#btn-discover:not([disabled])");
    await page.click("#btn-discover");
    await page.waitForSelector(".net-host");
    await page.click(".net-host .net-add");
    await page.waitForFunction(
      () => document.querySelector("#net-layout-note").textContent.trim().startsWith("Not saved:"),
      null,
      { timeout: 15000 }
    );
    out.noteC = (await page.textContent("#net-layout-note")).trim();
    out.dialogsAfterC = out.dialogs.length;
  } catch (e) {
    out.crash = String(e);
  } finally {
    if (browser) await browser.close().catch(() => {});
  }
  console.log(JSON.stringify(out));
})();'''


def test_network_timeout_without_before_picture(network_server):
    """The layout never loaded, so there is no before-picture: after a timeout the page cannot tell, and says reload to check."""
    port, tmp_path = network_server
    res = _run(PROBE_TIMEOUT_NO_PICTURE, port, SCAN_TIMEOUT_NO_PICTURE)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["layoutHitsBeforeSave"] == 1, res
    assert res["dialogCount"] == 1
    assert res["note"] == "Not saved: no answer from Sparksi; reload to check."
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    names = [e["name"] for e in cfg.get("endpoints", [])]
    assert names == ["box-six"]


def test_network_timeout_with_outside_change(network_server):
    """An unrelated change lands while the page's own save gets no reply: not reported as saved."""
    port, tmp_path = network_server
    res = _run(PROBE_TIMEOUT_OUTSIDE, port, SCAN_TIMEOUT_OUTSIDE)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["note"] == (
        "Not saved: no answer from Sparksi, and the change is not in the setup. Try again."
    )
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    names = [e["name"] for e in cfg.get("endpoints", [])]
    assert names == ["box-seven"]
    assert res["dialogCount"] == 1


def test_network_timeout_change_landed(network_server):
    """Slow replies past the timeout: each change still lands and is reported as such."""
    port, tmp_path = network_server
    res = _run(PROBE_TIMEOUT_LANDED, port, SCAN_TIMEOUT_LANDED)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    slow_note = "Saved (the reply was slow). Restart Sparksi to use the new setup."
    assert res["noteA"] == slow_note
    assert res["noteB"] == slow_note
    assert res["noteC"] == slow_note
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    node_ids = [n["id"] for n in cfg["nodes"]]
    assert node_ids == ["b", "a"]
    assert cfg.get("endpoints", []) == []


def test_network_name_trim_matches_python(network_server):
    """Name cleaning matches the Python side: trimmed saves, control names and bad scans do not."""
    port, tmp_path = network_server
    res = _run(PROBE_NAME_TRIM, port, SCAN_NAME_TRIM)
    assert "crash" not in res, res.get("crash")
    assert res["errors"] == [], res
    assert res["noteA"].startswith("Saved:")
    assert res["dialogsAfterA"] == 1
    assert res["noteB"] == (
        "Not saved: a name must not contain control or line-break characters."
    )
    assert res["dialogsAfterB"] == 1
    assert res["noteC"] == "Not saved: a name must be 1 to 100 characters."
    assert res["dialogsAfterC"] == 1
    with open(tmp_path / "sparksi.toml", "rb") as f:
        cfg = tomllib.load(f)
    names = [e["name"] for e in cfg.get("endpoints", [])]
    assert names == ["box-ten"]
