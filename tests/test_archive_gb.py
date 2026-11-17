# SPDX-License-Identifier: Apache-2.0

"""Archive GB summary: statvfs bytes in the agent, passed through storage, shown as GB."""

import json
import os
import subprocess
import threading
import types
from pathlib import Path

import pytest

from sparksi import agent
from sparksi.activity import reset_for_tests as reset_activity
from sparksi.library import storage
from sparksi.lock import Lock
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


class _Quiet:
    def invoke(self, host, req, timeout=20):
        return {"ok": True, "paths": {}, "listening": {}, "http": {}, "containers": []}


@pytest.fixture()
def gb_server(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CATALOG", str(ROOT / "config" / "catalog.example.yaml"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
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


class FakeStatvfs(types.SimpleNamespace):
    pass


def test_store_bytes_values(monkeypatch):
    st = FakeStatvfs(f_blocks=1000, f_bfree=300, f_bavail=250, f_frsize=4096)
    monkeypatch.setattr(os, "statvfs", lambda root: st)
    result = agent._store_bytes("/mnt/archive")
    assert result["total_bytes"] == 1000 * 4096
    assert result["used_bytes"] == (1000 - 300) * 4096
    assert result["free_bytes"] == 300 * 4096
    assert result["avail_bytes"] == 250 * 4096


def test_store_bytes_oserror(monkeypatch):
    def boom(root):
        raise OSError("no such mount")

    monkeypatch.setattr(os, "statvfs", boom)
    assert agent._store_bytes("/mnt/archive") == {}


def test_storage_passes_byte_fields():
    probes = {"spark-a": {"ok": True, "disk_total_gib": 3755.0, "disk_used_gib": 1377.0, "disk_free_gib": 2187.0, "disk_used_pct": 39.0, "mount_archive": True, "archive_store": {"path": "/mnt/archive", "device": "/dev/sda1", "fstype": "ext4", "total_gib": 3666.0, "used_gib": 1485.0, "free_gib": 2181.0, "used_pct": 41.0, "total_bytes": 4000000000000, "used_bytes": 1600000000000, "free_bytes": 2400000000000, "avail_bytes": 2200000000000}}, "spark-b": {"ok": True, "disk_total_gib": 3755.0, "disk_free_gib": 2395.0, "mount_archive": True}}
    rows = storage({}, probes)
    arch = [r for r in rows if r["kind"] == "archive"][0]
    assert arch["total_bytes"] == 4000000000000
    assert arch["used_bytes"] == 1600000000000
    assert arch["free_bytes"] == 2400000000000
    assert arch["avail_bytes"] == 2200000000000


def test_archive_gb_on_page_and_tile(gb_server) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit archive GB regression")
        pytest.skip("Playwright WebKit is not installed")
    probe = r"""
const { webkit } = require("playwright");
const port = process.env.RECIPE_PORT;
const archive = { id: "archive", kind: "archive", label: "Archive (RAID 1 DAS)", host: "spark-a", ok: true,
  total_gib: 3725.3, used_gib: 1490.1, free_gib: 2048.9, used_pct: 42.1,
  total_bytes: 4000000000000, used_bytes: 1600000000000, free_bytes: 2400000000000, avail_bytes: 2200000000000 };
let storageRows = [archive];
(async () => {
  const browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  const errors = [];
  const shots = process.env.SPARKSI_SHOT_DIR || "";
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.route("**/api/fleet*", async (route) => {
    try {
      const resp = await route.fetch();
      const body = await resp.json();
      body.storage = storageRows;
      await route.fulfill({ response: resp, body: JSON.stringify(body) });
    } catch (e) {
      // A reload or close can cancel an in-flight poll; that request is moot.
    }
  });
  await page.goto("http://127.0.0.1:" + port + "/archive", { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => /GB/.test((document.getElementById("archive-gb") || {}).textContent || ""), null, { timeout: 30000 });
  const read = () => page.evaluate(() => {
    const out = {};
    document.querySelectorAll("#archive-gb dd").forEach((dd) => {
      out[dd.getAttribute("data-key")] = { label: dd.previousElementSibling.textContent, value: dd.textContent };
    });
    out.tile = document.getElementById("tile-archive").textContent;
    out.count = document.querySelectorAll("#archive-gb dd").length;
    return out;
  });
  const mounted = await read();
  if (shots) await page.locator("#view-archive").screenshot({ path: shots + "/terra-gb.png" });
  storageRows = [Object.assign({}, archive, { ok: false, error: "archive not mounted on spark-a" })];
  await page.reload({ waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => /n\/a/.test((document.getElementById("archive-gb") || {}).textContent || ""), null, { timeout: 30000 });
  const unmounted = await read();
  await browser.close();
  console.log("__RESULT__" + JSON.stringify({ errors, mounted, unmounted }));
})().catch((e) => { console.error(e); process.exit(1); });
"""
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["RECIPE_PORT"] = str(gb_server)
    proc = subprocess.run(["node"], input=probe, text=True, capture_output=True, check=False, env=env, timeout=180)
    if proc.returncode != 0:
        missing = "Executable doesn't exist" in (proc.stderr or "")
        if missing and not required:
            pytest.skip("Playwright WebKit browser is not installed")
        raise AssertionError(proc.stderr or proc.stdout)
    raw = proc.stdout
    got = json.loads(raw[raw.index("__RESULT__") + len("__RESULT__"):].strip().splitlines()[0])
    assert got["errors"] == []
    m = got["mounted"]
    assert m["count"] == 4
    assert [m[k]["label"] for k in ("total", "used", "free", "avail")] == ["Total", "Used", "Free", "Available"]
    assert [m[k]["value"] for k in ("total", "used", "free", "avail")] == ["4000.0 GB", "1600.0 GB", "2400.0 GB", "2200.0 GB"]
    assert m["tile"] == "2200.0 GB"
    u = got["unmounted"]
    assert [u[k]["value"] for k in ("total", "used", "free", "avail")] == ["n/a"] * 4
    assert u["tile"] == "n/a"
