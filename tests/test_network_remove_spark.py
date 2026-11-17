# SPDX-License-Identifier: Apache-2.0

"""Network page: only the last Spark has Remove, the head never does, and Remove asks first."""

import tomllib

import pytest

from tests.test_network_setup_page import _run, network_server  # noqa: F401

PROBE = r"""
const { webkit } = require("playwright");
const out = { errors: [], dialogs: [] };
let browser = null;
(async () => {
  browser = await webkit.launch();
  const page = await browser.newPage({ viewport: { width: 400, height: 860 } });
  page.on("pageerror", (e) => out.errors.push(String(e)));
  page.on("dialog", (d) => { out.dialogs.push(d.message()); d.accept(); });
  const base = "http://127.0.0.1:" + process.env.PAGE_PORT;
  await page.goto(base + "/network");
  await page.waitForFunction(
    () => document.querySelectorAll("#net-layout-list .net-row").length >= 3
  );
  const rows = () =>
    Array.from(document.querySelectorAll("#net-layout-list .net-row")).map((row) => ({
      label: row.querySelector("strong").textContent,
      remove: !!row.querySelector("button.net-remove"),
      text: row.textContent,
    }));
  out.before = await page.evaluate(rows);
  const target = process.env.TARGET;
  const handle = await page.evaluateHandle((t) => {
    for (const row of document.querySelectorAll("#net-layout-list .net-row")) {
      if (row.querySelector("strong").textContent === t) return row.querySelector("button.net-remove");
    }
    return null;
  }, target);
  await handle.asElement().click();
  await page.waitForFunction(() => /saved/i.test(document.querySelector("#net-layout-note").textContent || ""));
  out.after = await page.evaluate(rows);
  out.note = await page.evaluate(() => document.querySelector("#net-layout-note").textContent);
  await browser.close();
  console.log(JSON.stringify(out));
})().catch(async (err) => {
  out.crash = String(err);
  console.log(JSON.stringify(out));
  if (browser) await browser.close();
  process.exit(0);
});
"""


def _rows(rows):
    return {r["label"]: r for r in rows}


@pytest.mark.parametrize("n", [2, 3])
def test_network_remove_spark(network_server, n, monkeypatch, tmp_path):  # noqa: F811
    port, tmp = network_server
    if n == 3:
        p = tmp_path / "sparksi.toml"
        text = p.read_text()
        text += (
            '\n[[nodes]]\nid = "c"\nname = "spark-c"\n'
            'ssh = "spark-c"\ntailscale = "100.64.0.13"\n'
        )
        p.write_text(text)
    target = "Spark 3" if n == 3 else "Worker Spark"
    monkeypatch.setenv("TARGET", target)
    out = _run(PROBE, port)
    assert "crash" not in out, out
    assert out["errors"] == []
    assert len(out["dialogs"]) == 1
    assert "Remove" in out["dialogs"][0]
    assert "Recipes saved for this Spark" in out["dialogs"][0]

    before = _rows(out["before"])
    after = _rows(out["after"])
    assert before["Head Spark"]["remove"] is False
    assert before[target]["remove"] is True
    if n == 3:
        assert before["Worker Spark"]["remove"] is False
    # The worker row stays (empty) on one Spark; a third Spark's row goes away.
    gone = "spark-c" if n == 3 else "spark-b"
    assert not any(gone in r["text"] for r in out["after"])
    if n == 3:
        assert target not in after
    else:
        assert after[target]["remove"] is False
    assert "Saved: removed " + gone in out["note"]

    data = tomllib.loads((tmp_path / "sparksi.toml").read_text())
    assert len(data["nodes"]) == n - 1
