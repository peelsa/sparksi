# SPDX-License-Identifier: Apache-2.0

"""Overview Spark cards are no taller or wider than they were before the GPU clock cap."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "sparksi" / "static"
INDEX = STATIC / "index.html"

# Pre-PR #27 serving-box rules (peelsa/sparksi@4496a37). Shared card geometry only.
BEFORE_CARD_CSS = """
  #view-overview {
    container-type: inline-size;
    container-name: overview;
    min-width: 0;
  }
  .serving {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: var(--s3);
    align-items: stretch;
    margin: 0 0 var(--s4);
    min-width: 0;
  }
  @container overview (max-width: 720px) {
    .serving { grid-template-columns: minmax(0, 1fr); }
  }
  .serve-box {
    min-width: 0;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--s3);
  }
  .serve-box h2 { margin: 0 0 var(--s2); font-size: 16px; line-height: 1.2; }
  .serve-box p { margin: 0; min-width: 0; font-weight: 600; overflow-wrap: anywhere; }
  .serve-box .serve-meta { color: var(--muted); font-weight: 500; }
  .serve-endpoint {
    display: block;
    margin-top: 4px;
    color: var(--zest);
    font-weight: 600;
    overflow-wrap: anywhere;
    background: none;
    border: 0;
    padding: 0;
    text-align: left;
    cursor: pointer;
  }
  .serve-copy { min-height: 44px; }
  .serve-endpoint[hidden], .serve-copy[hidden] { display: none; }
"""

# PR #27 Overview box (dd72dd25): the extra block we are moving off this page.
BOX_CARD_CSS = BEFORE_CARD_CSS + """
  .gpu-clock {
    margin-top: var(--s3);
    padding-top: var(--s3);
    border-top: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    gap: var(--s2);
  }
  .gpu-clock .gpu-clock-read,
  .gpu-clock .gpu-clock-range,
  .gpu-clock .gpu-clock-msg {
    flex: 1 0 100%;
    margin: 0;
    font-weight: 500;
    color: var(--muted);
  }
  .gpu-clock .gpu-clock-read { color: var(--text); font-weight: 600; }
  .gpu-clock label {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
    flex: 1 1 8rem;
  }
  .gpu-clock input {
    min-height: 44px;
    width: 100%;
    max-width: 10rem;
    padding: var(--s2);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius);
    background: transparent;
    color: var(--text);
  }
"""


def _shared_css() -> str:
    html = INDEX.read_text(encoding="utf-8")
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    # Fonts, tokens, and typography only — not later layout rules.
    font_end = style.find("  :root {")
    root_end = style.find("\n  }", font_end) + 4
    body_start = style.find("  body {")
    body_end = style.find("\n  }", body_start) + 4
    return style[:root_end] + "\n" + style[body_start:body_end]


def _after_card_css() -> str:
    html = INDEX.read_text(encoding="utf-8")
    style = html.split("<style>", 1)[1].split("</style>", 1)[0]
    overview = style[style.find("  #view-overview {") : style.find("  .serving {")]
    start = style.find("  .serving {")
    end = style.find("  .serve-link {")
    # Include the Overview container query that stacks the boxes on a narrow page.
    query_start = style.find("  @container overview (max-width: 720px)")
    query = ""
    if query_start != -1:
        query = style[query_start : style.find("  }", query_start) + 3]
    return overview + style[start:end] + "\n" + query + """
  .serve-endpoint {
    display: block;
    margin-top: 4px;
    color: var(--zest);
    font-weight: 600;
    overflow-wrap: anywhere;
    background: none;
    border: 0;
    padding: 0;
    text-align: left;
    cursor: pointer;
  }
  .serve-copy { min-height: 44px; }
  .serve-endpoint[hidden], .serve-copy[hidden] { display: none; }
"""


def _box_markup(kind: str, *, clock: str, cap: str, model: bool) -> str:
    if kind == "before":
        heading = "<h2>Head</h2>"
        extra = ""
    elif kind == "after":
        clock_span = f'<span class="serve-clock">{clock}</span>' if clock else '<span class="serve-clock"></span>'
        heading = f'<h2><span class="serve-title">Head</span>{clock_span}</h2>'
        extra = ""
    else:
        heading = "<h2>Head</h2>"
        extra = f"""
        <div class="gpu-clock">
          <p class="gpu-clock-read">Clock {clock} · Cap {cap}</p>
          <p class="gpu-clock-range">Allowed 1–3003 MHz</p>
          <label><span>Max clock (MHz)</span><input type="text" value="" placeholder="No cap"/></label>
          <button type="button">Save</button>
          <p class="note gpu-clock-msg"></p>
        </div>
        """
    if model:
        body = """
        <div class="serve-body">
          <div>
            <p class="serve-model">single-a</p>
            <p class="serve-meta">Ready · port 8013</p>
            <a class="serve-endpoint" href="http://127.0.0.1:8013">http://127.0.0.1:8013</a>
            <button type="button" class="serve-copy">Copy</button>
          </div>
        </div>
        """
    else:
        body = """
        <div class="serve-body">
          <div>
            <p class="serve-model">Nothing running</p>
            <p class="serve-meta"></p>
            <a class="serve-endpoint" hidden></a>
            <button type="button" class="serve-copy" hidden>Copy</button>
          </div>
        </div>
        """
    worker_heading = heading.replace("Head", "Worker")
    if kind == "after":
        worker_heading = worker_heading.replace("2411 MHz", "2405 MHz")
    return f"""
    <div id="view-overview">
    <div class="serving">
      <section class="serve-box" data-role="head">{heading}{body}{extra}</section>
      <section class="serve-box" data-role="worker">{worker_heading}{body}{extra}</section>
    </div>
    </div>
    """


def write_kind_page(path: Path, kind: str, css: str, title: str) -> None:
    shared = _shared_css()
    idle = _box_markup(kind, clock="2411 MHz", cap="none", model=False)
    busy = _box_markup(kind, clock="2411 MHz", cap="none", model=True)
    path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
{shared}
  body {{ padding: 16px; }}
  .case {{ margin: 0 0 32px; }}
  .case h1 {{ font-size: 14px; font-weight: 600; margin: 0 0 8px; color: var(--muted); }}
{css}
</style>
</head>
<body>
<section class="case" id="case-idle"><h1>{title} · idle</h1>
<div id="idle">{idle}</div></section>
<section class="case" id="case-busy"><h1>{title} · serving</h1>
<div id="busy">{busy}</div></section>
</body>
</html>
""",
        encoding="utf-8",
    )


def write_measure_pages(dest: Path) -> None:
    write_kind_page(dest / "before.html", "before", BEFORE_CARD_CSS, "Before PR #27")
    write_kind_page(dest / "after.html", "after", _after_card_css(), "After")
    write_kind_page(dest / "box.html", "box", BOX_CARD_CSS, "PR #27 box on Overview")


PROBE = r"""
const { webkit, chromium } = require("playwright");
const fs = require("fs");
const path = require("path");

(async () => {
  const engine = process.env.SPARKSI_FOOTPRINT_ENGINE || "webkit";
  const exe = process.env.SPARKSI_CHROME_PATH || "";
  const shotDir = process.env.SPARKSI_SHOT_DIR || "";
  const base = process.env.SPARKSI_MEASURE_URL;
  const browser = engine === "chrome"
    ? await chromium.launch({ executablePath: exe, args: ["--no-sandbox", "--disable-gpu"] })
    : await webkit.launch();
  const out = { widths: {}, errors: [] };
  try {
    for (const width of [1280, 390]) {
      const height = width === 1280 ? 900 : 844;
      const pack = {};
      const page = await browser.newPage({ viewport: { width, height } });
      page.on("pageerror", (e) => out.errors.push(String(e)));
      for (const kind of ["before", "after", "box"]) {
        await page.goto(base + kind + ".html", { waitUntil: "networkidle" });
        await page.evaluate(() => document.fonts && document.fonts.ready);
        const measured = await page.evaluate(() => {
          function box(sel) {
            const el = document.querySelector(sel);
            const r = el.getBoundingClientRect();
            return { w: Math.round(r.width), h: Math.round(r.height) };
          }
          function cards(root) {
            return Array.from(document.querySelectorAll(root + " .serve-box")).map((el) => {
              const r = el.getBoundingClientRect();
              return { role: el.getAttribute("data-role"), w: Math.round(r.width), h: Math.round(r.height) };
            });
          }
          return {
            idle: { grid: box("#idle .serving"), cards: cards("#idle") },
            busy: { grid: box("#busy .serving"), cards: cards("#busy") }
          };
        });
        pack[kind + "-idle"] = measured.idle;
        pack[kind + "-busy"] = measured.busy;
        if (shotDir) {
          fs.mkdirSync(shotDir, { recursive: true });
          for (const state of ["idle", "busy"]) {
            const handle = await page.$("#case-" + state);
            if (handle) await handle.screenshot({ path: path.join(shotDir, kind + "-" + state + "-" + width + ".png") });
          }
        }
      }
      out.widths[String(width)] = pack;
      await page.close();
    }
  } finally {
    await browser.close();
  }
  console.log("__RESULT__" + JSON.stringify(out));
})();
"""


@pytest.fixture()
def measure_server(tmp_path):
    dest = tmp_path / "static"
    dest.mkdir()
    write_measure_pages(dest)
    # Fonts for Inter, so typography matches the dashboard.
    (dest / "fonts").symlink_to(STATIC / "fonts", target_is_directory=True)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(dest), **k)

        def log_message(self, *_a) -> None:  # noqa: D102
            return

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def _run_probe(url: str, *, shot_dir: Path | None, engine: str, chrome: str, node: Path) -> dict:
    env = os.environ.copy()
    env["NODE_PATH"] = str(node)
    env["SPARKSI_MEASURE_URL"] = url
    env["SPARKSI_FOOTPRINT_ENGINE"] = engine
    if chrome:
        env["SPARKSI_CHROME_PATH"] = chrome
    if shot_dir:
        env["SPARKSI_SHOT_DIR"] = str(shot_dir)
    proc = subprocess.run(
        ["node"],
        input=PROBE,
        text=True,
        capture_output=True,
        check=False,
        env=env,
        timeout=180,
    )
    return proc


def test_overview_cards_match_pre_pr27_footprint(measure_server, tmp_path) -> None:
    node_path = os.environ.get("SPARKSI_PLAYWRIGHT_NODE", "")
    node = Path(node_path) if node_path else None
    required = os.environ.get("SPARKSI_WEBKIT_REQUIRED") == "1"
    if node is None or not (node / "playwright").is_dir():
        if required:
            raise AssertionError("CI requires the WebKit Overview footprint check")
        pytest.skip("Playwright WebKit is not installed")

    shot = tmp_path / "shots"
    proc = _run_probe(measure_server, shot_dir=shot, engine="webkit", chrome="", node=node)
    if proc.returncode != 0 and "Executable doesn't exist" in (proc.stderr or "") and not required:
        pytest.skip("Playwright WebKit browser is not installed")
    if proc.returncode != 0:
        raise AssertionError("footprint probe failed\nstdout:\n%s\nstderr:\n%s" % (proc.stdout, proc.stderr))
    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith("__RESULT__"):
            result_line = line
    assert result_line is not None, "probe produced no __RESULT__ line:\n%s" % proc.stdout
    data = json.loads(result_line[len("__RESULT__") :])
    assert data["errors"] == []
    for width, pack in data["widths"].items():
        for state in ("idle", "busy"):
            before = pack[f"before-{state}"]
            after = pack[f"after-{state}"]
            assert before["grid"]["w"] == after["grid"]["w"], (width, state, before, after)
            assert after["grid"]["h"] <= before["grid"]["h"], (width, state, before, after)
            for left, right in zip(before["cards"], after["cards"], strict=True):
                assert right["w"] <= left["w"], (width, state, left, right)
                assert right["h"] <= left["h"], (width, state, left, right)
            boxed = pack[f"box-{state}"]
            # The Overview field we removed had made the card taller.
            assert boxed["grid"]["h"] > after["grid"]["h"], (width, state, boxed, after)
