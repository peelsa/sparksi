# SPDX-License-Identifier: Apache-2.0

"""Overview: the emergency stop button for servers started outside Sparksi, in WebKit at 400 px.

The lock stays on for the whole run: the emergency stop works with it, behind the
confirmation that names the model, Spark and port."""

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.config import CFG
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]

ITEMS = [
    {"host": "s1", "host_name": "spark-1", "role": "head", "port": 8000, "label": "org/model-a",
     "container": "model-a-box", "why": "", "stoppable": True, "emergency": False},
    {"host": "s2", "host_name": "spark-2", "role": "worker", "port": 8020, "label": "llama",
     "container": None, "why": "not a container Sparksi can stop; stop it on the machine", "stoppable": False, "emergency": True},
    {"host": "s3", "host_name": "spark-3", "role": "spark-3", "port": 8030, "label": "boxed",
     "container": "boxed-server", "why": "", "stoppable": False, "emergency": True},
    {"host": "s4", "host_name": "spark-4", "role": "spark-4", "port": 8040, "label": "keepme",
     "container": None, "why": "protected on this controller; stop it on the machine", "stoppable": False, "emergency": False},
]

_PROBE = """
const { webkit } = require("playwright");
(async () => {
  const out = { pageErrors: [], posts: [], dialogs: [] };
  const browser = await webkit.launch();
  try {
    const page = await browser.newPage({ viewport: { width: 400, height: 900 } });
    page.on("pageerror", (err) => out.pageErrors.push(String(err)));
    page.on("dialog", (d) => { out.dialogs.push(d.message()); d.dismiss(); });
    let reply = { stopped: true, port_closed: true, gpu_checked: true, gpu_freed_mib: 5000, signal: "TERM" };
    let status = 200;
    let delay = 0;
    await page.route("**/api/outside/stop", async (route) => {
      out.posts.push(route.request().postDataJSON());
      if (delay) await new Promise((r) => setTimeout(r, delay));
      await route.fulfill({ status: status, contentType: "application/json", body: JSON.stringify(reply) });
    });
    const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
    await page.goto(base + "/overview", { waitUntil: "networkidle" });
    await page.waitForFunction(() => document.querySelectorAll("#unmanaged .unmanaged-row").length === 4);
    out.rows = await page.$$eval("#unmanaged .unmanaged-row", (els) =>
      els.map((el) => ({
        button: el.querySelector("button") ? el.querySelector("button").textContent : null,
        cls: el.querySelector("button") ? el.querySelector("button").className : null,
        note: el.querySelector(".note") ? el.querySelector(".note").textContent : null,
        h: Math.round(el.getBoundingClientRect().height),
      }))
    );
    out.docW = await page.evaluate(() => document.documentElement.scrollWidth);
    // The same rows with the old note in place of the button, measured on a copy.
    out.oldH = await page.$$eval("#unmanaged .unmanaged-row", (els) =>
      els.map((el) => {
        const copy = el.cloneNode(true);
        const btn = copy.querySelector("button");
        if (btn) {
          const note = document.createElement("span");
          note.className = "note";
          note.textContent = "not a container Sparksi can stop; stop it on the machine";
          btn.replaceWith(note);
        }
        el.parentNode.appendChild(copy);
        const h = Math.round(copy.getBoundingClientRect().height);
        copy.remove();
        return h;
      })
    );
    const rowSel = (n) => "#unmanaged .unmanaged-row >> nth=" + n + " >> button";
    const facts = () => page.evaluate(() => ({
      open: document.getElementById("model-confirm").open,
      title: document.getElementById("model-confirm-title").textContent,
      model: document.getElementById("model-confirm-model").textContent,
      machines: document.getElementById("model-confirm-machines").textContent,
      port: document.getElementById("model-confirm-port").textContent,
      note: document.getElementById("model-confirm-note").textContent,
      ok: document.getElementById("model-confirm-ok").textContent,
      okCls: document.getElementById("model-confirm-ok").className,
    }));
    await page.click(rowSel(1));
    out.dialog1 = await facts();
    await page.click("#model-confirm-cancel");
    await page.waitForTimeout(150);
    out.postsAfterCancel = out.posts.length;
    await page.click(rowSel(1));
    await page.click("#model-confirm-ok");
    await page.waitForFunction(() => document.querySelector("#unmanaged-note").textContent.startsWith("Stopped"));
    out.note1 = await page.$eval("#unmanaged-note", (el) => el.textContent);
    reply = { stopped: false, error: "that server belongs to another user; nothing was stopped", command: "sudo kill -s TERM -- 4242" };
    status = 502;
    delay = 800;
    await page.click(rowSel(2));
    out.dialog2 = await facts();
    await page.click("#model-confirm-ok");
    // A second click while that stop is in flight opens nothing and sends nothing.
    await page.waitForTimeout(100);
    await page.click(rowSel(1));
    out.inFlight = { open: await page.evaluate(() => document.getElementById("model-confirm").open), posts: out.posts.length };
    await page.waitForFunction(() => document.querySelector("#unmanaged-note").textContent.startsWith("Not stopped"));
    out.note2 = await page.$eval("#unmanaged-note", (el) => el.textContent);
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify(out));
})().catch((err) => { console.error(err); process.exit(1); });
"""


@pytest.fixture()
def page_server(monkeypatch, tmp_path):
    import sparksi.serve as serve_mod

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
    monkeypatch.setattr(serve_mod, "_unmanaged_items", lambda runner, *, cached, cat: (ITEMS, True))
    monkeypatch.setitem(CFG.controller, "stop_unmanaged", True)
    httpd = DualServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        STATE.lock.release()
        STATE.runner = None
        STATE.monitor_fn = None


def _run(probe: str, port: int) -> dict:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit emergency stop regression")
        pytest.skip("Playwright WebKit is not installed")
    env = os.environ.copy()
    env.update(NODE_PATH=str(node), PAGE_PORT=str(port))
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    return json.loads(proc.stdout)


def test_emergency_stop_button_confirms_and_posts_with_the_lock_on(page_server):
    out = _run(_PROBE, page_server)
    assert out["pageErrors"] == []
    assert out["dialogs"] == []
    assert out["docW"] <= 400
    assert [(r["button"], r["cls"], r["note"]) for r in out["rows"]] == [
        ("Stop", "halt", None),
        ("Emergency stop", "halt", None),
        ("Emergency stop", "halt", None),
        (None, None, "protected on this controller; stop it on the machine"),
    ]
    # One button per row: no taller than the same row showing the note it replaces.
    for row, old_h in zip(out["rows"][1:3], out["oldH"][1:3]):
        assert row["h"] <= old_h
    assert out["dialog1"] == {
        "open": True,
        "title": "Emergency stop llama?",
        "model": "llama",
        "machines": "spark-2",
        "port": "8020",
        "note": "Sparksi did not start it. This ends the process listening on port 8020 and its child processes: SIGTERM, then SIGKILL if it does not exit. It works with the lock on.",
        "ok": "Emergency stop",
        "okCls": "halt",
    }
    assert out["postsAfterCancel"] == 0
    assert out["posts"][0] == {
        "host": "s2", "port": 8020, "container": None, "from_ui": True,
        "confirm": {"model": "llama", "spark": "spark-2", "port": 8020},
    }
    assert out["note1"] == "Stopped llama on spark-2. Port 8020 closed. GPU freed 5000 MiB."
    assert out["dialog2"]["note"] == "Sparksi did not start it. This stops the container boxed-server with docker stop. It works with the lock on."
    assert out["posts"][1]["container"] == "boxed-server"
    assert out["posts"][1]["confirm"] == {"model": "boxed", "spark": "spark-3", "port": 8030}
    assert out["note2"] == (
        "Not stopped: that server belongs to another user; nothing was stopped."
        " Run on spark-3: sudo kill -s TERM -- 4242"
    )
    assert out["inFlight"] == {"open": False, "posts": 2}
    assert len(out["posts"]) == 2
    assert STATE.lock.read().locked is True
