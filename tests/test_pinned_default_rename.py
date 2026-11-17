# SPDX-License-Identifier: Apache-2.0

"""A pinned default: its list row reads the launch command, it can be renamed in place, and Save asks for a name."""

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
from sparksi.recipes.store import default_recipe_name, named_dir, recipe_path, save_recipe, using_recipe
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]
HUB = "/root/.cache/huggingface/hub/models--example--Pair-Fast-exl3-4bpw-instruct/snapshots/abc"
NEW_NAME = "pair-fast-instruct-exl3-4bpw-c4-850k-fp8-2node-spec"
# A start.sh recipe: every setting is in the environment the script reads.
PINNED = {
    "kind": "pinned",
    "engine": "other",
    "placement": "both",
    "port": 8014,
    "imported_from": "container:example at 2026-11-01T00:00:00Z",
    "roles": {
        "head": {
            "runtime": "docker",
            "image": "example/start-sh:1",
            "args": ["/start.sh"],
            "env": {
                "MODEL_DIR": HUB,
                "QUANTIZATION": "exl3",
                "MAX_NUM_SEQS": "4",
                "MAX_MODEL_LEN": "850000",
                "KV_CACHE_DTYPE": "fp8",
                "TP": "2",
                "SPEC_METHOD": "dflash",
            },
        }
    },
}


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def pinned_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    save_recipe("pair-fast", PINNED)
    with using_recipe("pair-fast", "plain"):
        save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 16384, "gpu_mem_util": 0.55})
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")  # recipe edits keep working while the page is locked
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


PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.RECIPE_PORT;
  const newName = process.env.NEW_NAME;
  const browser = await webkit.launch();
  const errors = [];
  const writes = [];
  const reads = [];
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("pageerror", (err) => errors.push(String(err)));
  page.on("dialog", async (d) => { await d.accept(); });
  page.on("request", (req) => {
    const u = new URL(req.url());
    if (req.method() !== "GET") writes.push(req.method() + " " + u.pathname);
    else if (u.pathname.indexOf("/api/models/pair-fast/recipe") === 0) reads.push(u.pathname + u.search);
  });
  const names = () => page.$$eval("#recipe-files .recipe-file-row", (els) => els.map((el) => el.getAttribute("data-name")));
  const waitNote = (t) => page.waitForFunction((x) => document.getElementById("recipe-note").textContent.indexOf(x) !== -1, t);
  const ids = ["recipe-cmd-open", "recipe-cmd-rename", "recipe-cmd-copy", "recipe-cmd-default", "recipe-cmd-delete"];
  const out = { errors, writes };
  await page.goto(base + "/models", { waitUntil: "networkidle" });
  await page.click("button:has-text('Pair Fast')");
  await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
  await page.waitForFunction(() => document.querySelectorAll("#recipe-files .recipe-file-row").length === 2);
  out.cells = await page.$$eval("#recipe-files tr[data-name='default'] td", (els) => els.map((el) => el.textContent.trim()));
  /* Rename is allowed on the pinned default; Delete and Set as default are not. */
  await page.click("#recipe-files tr[data-name='default'] .recipe-row-more");
  out.disabled = await page.evaluate((list) => list.filter((id) => document.getElementById(id).disabled), ids);
  await page.keyboard.press("Escape");
  /* Save on a pinned recipe asks for a name, prefilled from the launch command. */
  await page.click("#recipe-cmd-save");
  await page.waitForFunction(() => !document.getElementById("recipe-name-row").hidden);
  out.saveAsk = await page.evaluate(() => ({ note: document.getElementById("recipe-note").textContent, value: document.getElementById("recipe-name").value }));
  await page.click("#recipe-name-cancel");
  /* F2 renames the default in place. */
  await page.focus("#recipe-files tr[data-name='default']");
  await page.keyboard.press("F2");
  await page.waitForSelector("#recipe-rename");
  await page.fill("#recipe-rename", newName);
  await page.press("#recipe-rename", "Enter");
  await waitNote("Renamed to " + newName);
  await page.waitForFunction((n) => !!document.querySelector("#recipe-files tr[data-name='" + n + "']"), newName);
  out.afterRename = await names();
  out.current = await page.textContent("#recipe-current-name");
  out.checked = await page.$$eval("#recipe-files .recipe-default-box:checked", (els) => els.map((el) => el.closest("tr").getAttribute("data-name")));
  /* Load default recipe reads the same file under its new name and starts nothing. */
  const before = writes.length;
  reads.length = 0;
  await page.click("#model-load-default");
  await page.waitForFunction(() => document.getElementById("model-default-source").textContent.length > 0);
  await page.waitForTimeout(300);
  out.loadReads = reads.slice();
  out.loadWrites = writes.slice(before);
  out.afterLoad = await page.textContent("#recipe-current-name");
  out.source = await page.textContent("#model-default-source");
  /* A pinned recipe is never the target of a replace: rename, then Save as, onto its name. */
  const postsBefore = writes.length;
  await page.focus("#recipe-files tr[data-name='plain']");
  await page.keyboard.press("F2");
  await page.waitForSelector("#recipe-rename");
  await page.fill("#recipe-rename", newName);
  await page.press("#recipe-rename", "Enter");
  await waitNote("is not replaced");
  out.renameOnto = await page.textContent("#recipe-note");
  await page.press("#recipe-rename", "Escape");
  await page.click("#recipe-cmd-save-as");
  await page.fill("#recipe-name", newName);
  await page.click("#recipe-name-ok");
  await waitNote("is not replaced");
  out.saveAsOnto = await page.textContent("#recipe-note");
  await page.click("#recipe-name-cancel");
  out.ontoWrites = writes.slice(postsBefore);
  /* The server's answer is handled the same way when the list was stale. */
  await page.route("**/api/models/pair-fast/recipes/plain/rename", (route) => route.fulfill({
    status: 409, contentType: "application/json",
    body: JSON.stringify({ error: "late runs a pinned launch command", pinned: true, save_as: true })
  }));
  await page.focus("#recipe-files tr[data-name='plain']");
  await page.keyboard.press("F2");
  await page.waitForSelector("#recipe-rename");
  await page.fill("#recipe-rename", "late");
  await page.press("#recipe-rename", "Enter");
  await waitNote("late runs a pinned launch command, so it is not replaced");
  out.lateNote = await page.textContent("#recipe-note");
  out.finalNames = await names();
  await browser.close();
  console.log(JSON.stringify(out));
})().catch((err) => { console.error(String(err && err.stack || err)); process.exit(1); });
"""


def test_pinned_default_rename_and_save(pinned_server) -> None:
    port = pinned_server
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit pinned-default regression")
        pytest.skip("Playwright WebKit is not installed")
    original = recipe_path("pair-fast").read_bytes()
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(port)
    env["NEW_NAME"] = NEW_NAME
    proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env, timeout=240)
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert got["errors"] == [], got
    # Concurrency, Context / KV, Nodes and Spec come from the start.sh environment.
    assert got["cells"][2:6] == ["4", "850k \u00b7 fp8", "2 (pair)", "dflash"], got["cells"]
    assert got["disabled"] == ["recipe-cmd-open", "recipe-cmd-default", "recipe-cmd-delete"], got
    assert got["saveAsk"]["note"].startswith("default runs a pinned launch command"), got
    assert got["saveAsk"]["value"] == NEW_NAME, got
    assert got["afterRename"] == [NEW_NAME, "plain"], got
    assert got["current"] == NEW_NAME and got["checked"] == [NEW_NAME], got
    assert got["afterLoad"] == NEW_NAME, got
    assert got["loadWrites"] == [], got
    assert not any(w.startswith("PUT") for w in got["writes"]), got
    assert [w for w in got["writes"] if w.startswith("POST")] == [
        "POST /api/models/pair-fast/recipes/default/rename",
        "POST /api/models/pair-fast/recipes/plain/rename",  # the routed stale-list reply only
    ], got
    assert got["ontoWrites"] == [], got
    assert got["renameOnto"] == NEW_NAME + " runs a pinned launch command, so it is not replaced. Pick another name.", got
    assert got["saveAsOnto"] == got["renameOnto"], got
    assert got["lateNote"].startswith("late runs a pinned launch command, so it is not replaced"), got
    assert got["finalNames"] == [NEW_NAME, "plain"], got
    assert (named_dir("pair-fast") / "plain.toml").is_file()
    assert all("/api/runs" not in w for w in got["writes"]), got
    assert default_recipe_name("pair-fast") == NEW_NAME
    assert recipe_path("pair-fast").read_bytes() == original
