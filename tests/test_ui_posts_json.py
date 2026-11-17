# SPDX-License-Identifier: Apache-2.0

"""The dashboard's own writes all send JSON, so the same-origin JSON guard never breaks a button."""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

import pytest

from sparksi.serve import INDEX_HTML
from sparksi.activity import reset_for_tests as reset_activity
from sparksi.jobs import reset_for_tests as reset_jobs
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "src" / "sparksi" / "static" / "index.html"


def test_every_write_fetch_in_the_page_sends_json() -> None:
    text = PAGE.read_text(encoding="utf-8")
    assert "XMLHttpRequest" not in text and "sendBeacon" not in text
    assert not re.search(r"<form[^>]*method\s*=\s*[\"']?post", text, re.I)
    calls = [m.start() for m in re.finditer(r"\bfetch\(", text)]
    assert calls, "no fetch calls found"
    writes = 0
    for start in calls:
        chunk = text[start:start + 300]
        method = re.search(r"method:\s*\"([A-Z]+)\"", chunk)
        assert method, chunk
        if method.group(1) == "GET":
            continue
        writes += 1
        assert '"Content-Type": "application/json"' in chunk, chunk
    assert writes == 5  # jpost, jpostRead, jpostTimed, jput, jdel: every POST, PUT and DELETE goes through one of them


SEEN: list[dict] = []


class Recording(Handler):
    def _note(self) -> None:
        SEEN.append({"method": self.command, "path": self.path, "type": self.headers.get("Content-Type"), "origin": self.headers.get("Origin")})

    def do_POST(self) -> None:  # noqa: N802
        self._note()
        super().do_POST()

    def do_PUT(self) -> None:  # noqa: N802
        self._note()
        super().do_PUT()

    def do_DELETE(self) -> None:  # noqa: N802
        self._note()
        super().do_DELETE()


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


PROBE = r"""
const { webkit } = require("playwright");
(async () => {
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  const browser = await webkit.launch();
  const out = { errors: [], refused: [] };
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  page.on("pageerror", (err) => out.errors.push(String(err)));
  page.on("dialog", (d) => d.accept());
  page.on("response", (r) => { if (r.status() === 415 || r.status() === 403 && r.request().method() !== "GET") out.refused.push(r.status() + " " + r.request().method() + " " + new URL(r.url()).pathname); });
  const menu = async (id) => { if (["recipe-cmd-new", "recipe-cmd-save", "recipe-cmd-save-as"].indexOf(id) === -1) await page.click("#recipe-files tr.is-open .recipe-row-more"); await page.click("#" + id); };
  try {
    await page.goto(base + "/models", { waitUntil: "networkidle" });
    await page.click("button:has-text('Pair Fast')");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "default");
    await page.click("#btn-unlock");
    await page.waitForFunction(() => !/locked/.test(document.getElementById("lock-word").className));
    await page.click("#btn-lock");
    await page.waitForFunction(() => /locked/.test(document.getElementById("lock-word").className));
    await page.selectOption("#field-ctx", "32768");
    await menu("recipe-cmd-save");
    await page.waitForFunction(() => document.getElementById("recipe-dirty").hidden);
    await menu("recipe-cmd-save-as");
    await page.fill("#recipe-name", "json-check");
    await page.click("#recipe-name-ok");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "json-check");
    await menu("recipe-cmd-default");
    await page.waitForTimeout(500);
    await page.click("#model-dry");
    await page.waitForFunction(() => /Ready|Not ready/.test(document.getElementById("model-dry-note").textContent));
    await menu("recipe-cmd-save-as");
    await page.fill("#recipe-name", "json-scratch");
    await page.click("#recipe-name-ok");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "json-scratch");
    await menu("recipe-cmd-rename");
    await page.fill("#recipe-rename", "json-check-2");
    await page.press("#recipe-rename", "Enter");
    await page.waitForFunction(() => document.getElementById("recipe-current-name").textContent === "json-check-2");
    await menu("recipe-cmd-delete");
    await page.waitForTimeout(800);
  } catch (err) {
    out.crash = String(err && err.stack || err);
  }
  await browser.close();
  console.log(JSON.stringify(out));
})();
"""


def test_dashboard_buttons_send_json_to_the_guarded_server(monkeypatch, tmp_path) -> None:
    from sparksi.recipes.store import save_recipe

    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit JSON-writes regression")
        pytest.skip("Playwright WebKit is not installed")
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_jobs()
    reset_cache_for_tests()
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": "nvcr.io/nvidia/vllm:26.05-py3"})
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    STATE.lock = lock
    STATE.runner = _Quiet()
    SEEN.clear()
    httpd = DualServer(("127.0.0.1", 0), Recording)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["PAGE_PORT"] = str(httpd.server_address[1])
    try:
        proc = subprocess.run(["node"], input=PROBE, text=True, capture_output=True, check=False, env=env)
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_runs()
        reset_jobs()
        reset_cache_for_tests()
    if proc.returncode != 0:
        if "Executable doesn't exist" in (proc.stderr or "") and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    got = json.loads(proc.stdout)
    assert "crash" not in got, json.dumps({"got": got, "seen": SEEN}, indent=1)
    assert got["errors"] == [] and got["refused"] == [], got
    kinds = {(s["method"], re.sub(r"/recipes/[^/?]+", "/recipes/<name>", s["path"].split("?")[0])) for s in SEEN}
    for want in (
        ("POST", "/api/unlock"),
        ("POST", "/api/lock"),
        ("PUT", "/api/models/pair-fast/recipes/<name>"),
        ("POST", "/api/models/pair-fast/recipes"),
        ("POST", "/api/models/pair-fast/recipes/<name>/default"),
        ("POST", "/api/models/pair-fast/recipes/<name>/rename"),
        ("POST", "/api/models/pair-fast/plan"),
        ("DELETE", "/api/models/pair-fast/recipes/<name>"),
    ):
        assert want in kinds, (want, sorted(kinds))
    # What the server received: every write carried JSON.
    assert all((s["type"] or "").split(";")[0].strip() == "application/json" for s in SEEN), SEEN


def test_a_plan_preview_does_not_mark_cached_gets_stale():
    lines = INDEX_HTML.read_text(encoding="utf-8").splitlines()

    start = next(i for i, ln in enumerate(lines) if "var inflight = {};" in ln)
    jpost_at = next(i for i, ln in enumerate(lines) if "function jpost(url, body) {" in ln)
    end = next(i for i in range(jpost_at + 1, len(lines)) if lines[i] == "  }")

    script_lines = lines[start : end + 1]

    harness = """
var gets = [];
var posts = [];
var fetch = function (url, opts) {
  var method = (opts && opts.method) || "GET";
  if (method === "GET") {
    gets.push(url);
    return new Promise(function () {});
  }
  posts.push(url);
  return Promise.resolve({ ok: true, status: 200, json: function () { return Promise.resolve({}); } });
};
jget("/api/x");
jget("/api/x");
jpostRead("/api/models/m/plan", { preview: true });
var afterPreview = gets.length;
jget("/api/x");
var sharedAfterPreview = gets.length;
jpost("/api/models/m/recipes", {});
jget("/api/x");
setTimeout(function () {
  process.stdout.write(JSON.stringify({ gets: gets.length, posts: posts.length, afterPreview: afterPreview, shared: sharedAfterPreview }));
}, 20);
"""

    script = "\n".join(script_lines) + "\n" + harness
    out = subprocess.run(
        ["node"],
        input=script,
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(out.stdout)
    assert data["shared"] == 1, data  # a preview POST left the in-flight GET shared

    assert data["gets"] == 2
    assert data["posts"] == 2
