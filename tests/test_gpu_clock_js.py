# SPDX-License-Identifier: Apache-2.0

"""Overview Spark boxes have a per-Spark GPU max-clock field and Save."""

from __future__ import annotations

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _run(extra: str, *, roles: list[str] | None = None) -> object:
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* gpu-clock-start */", 1)[1].split("/* gpu-clock-end */", 1)[0]
    role_list = roles or ["head", "worker"]
    role_js = json.dumps(role_list)
    preamble = r"""
var nodeFacts = [
  { role: "head", id: "a", name: "spark-a", reachable: true, gpu: { util_pct: 4, temp_c: 40, clock_mhz: 2411, clock_max_mhz: 3003, clock_cap_mhz: null, clock_cap_reported: true } },
  { role: "worker", id: "b", name: "spark-b", reachable: true, gpu: { util_pct: 1, temp_c: 38, clock_mhz: 2405, clock_max_mhz: 3003, clock_cap_mhz: 2000 } },
  { role: "node3", id: "c", name: "spark-c", reachable: true, gpu: { clock_mhz: 2000, clock_max_mhz: 3003, clock_min_mhz: 210, clock_cap_mhz: null } },
  { role: "node4", id: "d", name: "spark-d", reachable: false, gpu: null }
];
var gpuClockDraft = {};
var gpuClockSaving = {};
var gpuClockConfirmPending = null;
var posted = [];
function sparkRoles() { return ROLES; }
function roleId(role) { return role === "worker" ? "b" : (role === "node3" ? "c" : (role === "node4" ? "d" : "a")); }
function roleName(role) { return "spark-" + roleId(role); }
function roleWord(role) {
  if (role === "worker") return "Worker";
  if (role === "node3") return "Spark 3";
  if (role === "node4") return "Spark 4";
  return "Head";
}
function lockOn() { return !!globalThis.lockIsOn; }
function trimNum(value) { return String(Math.round(Number(value))); }
function setText(node, text) {
  if (typeof node === "string") node = el(node);
  if (node) node.textContent = text;
}
function serveEntries(role) { return globalThis.runningRoles && globalThis.runningRoles[role] ? [{}] : []; }
function runningOn() { return []; }
function makeEl(tag) {
  var kids = [];
  var attrs = {};
  var node = {
    tag: tag,
    children: kids,
    childNodes: kids,
    textContent: "",
    className: "",
    type: "",
    hidden: false,
    disabled: false,
    title: "",
    value: "",
    inputMode: "",
    autocomplete: "",
    placeholder: "",
    parent: null,
    _listeners: {},
    setAttribute: function (k, v) { attrs[k] = String(v); },
    getAttribute: function (k) { return attrs[k] || null; },
    appendChild: function (child) { child.parent = node; kids.push(child); return child; },
    addEventListener: function (name, fn) { (node._listeners[name] || (node._listeners[name] = [])).push(fn); },
    querySelector: function (sel) { return query(node, sel); },
    click: function () { (node._listeners.click || []).forEach(function (fn) { fn(); }); }
  };
  return node;
}
function all(node, out) {
  out.push(node);
  (node.children || []).forEach(function (c) { all(c, out); });
  return out;
}
function query(root, sel) {
  var nodes = all(root, []);
  var roleMatch = sel.match(/\[data-role="([^"]+)"\]/);
  var wantRole = roleMatch ? roleMatch[1] : null;
  var clsMatch = sel.match(/^\.([^\[\s]+)/);
  var wantClass = clsMatch ? clsMatch[1] : null;
  for (var i = 0; i < nodes.length; i++) {
    var n = nodes[i];
    if (wantRole && n.getAttribute("data-role") !== wantRole) continue;
    if (wantClass) {
      if ((" " + (n.className || "") + " ").indexOf(" " + wantClass + " ") >= 0) return n;
      continue;
    }
    if (wantRole) return n;
    if (sel === "input" && n.tag === "input") return n;
    if (sel === "button" && n.tag === "button") return n;
  }
  return null;
}
var serving = makeEl("div");
serving.id = "serving-lines";
var confirmNote = makeEl("p");
confirmNote.id = "gpu-clock-confirm-note";
var ids = { "serving-lines": serving, "gpu-clock-confirm-note": confirmNote, "gpu-clock-confirm": null };
function el(id) { return ids[id] || null; }
function ensureServing() {
  if (serving.querySelector && serving.children.some(function (c) { return c.className === "gpu-clock"; })) return serving;
  sparkRoles().forEach(function (role) {
    serving.appendChild(gpuClockForm(role));
  });
  return serving;
}
function jpost(url, body) {
  posted.push({ url: url, body: body });
  return Promise.resolve(globalThis.nextReply || { ok: true, status: 200, body: { ok: true, gpu: { clock_mhz: 2000, clock_max_mhz: 3003, clock_cap_mhz: body.max_mhz } } });
}
var document = { createElement: function (tag) { return makeEl(tag); }, activeElement: null };
var window = { confirm: function () { return true; } };
""".replace("ROLES", role_js)
    driver = "\n" + extra + "\n"
    proc = subprocess.run(["node"], input=preamble + script + driver, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_each_spark_gets_a_clock_box_and_save() -> None:
    got = _run(
        """
        paintGpuClocks();
        var boxes = serving.children.map(function (box) {
          var input = box.querySelector("input");
          var read = box.querySelector(".gpu-clock-read");
          var btn = box.querySelector("button");
          return {
            role: box.getAttribute("data-role"),
            read: read && read.textContent,
            range: box.querySelector(".gpu-clock-range") && box.querySelector(".gpu-clock-range").textContent,
            value: input && input.value,
            label: input && input.getAttribute("aria-label"),
            save: btn && btn.textContent,
            disabled: btn && btn.disabled
          };
        });
        process.stdout.write(JSON.stringify({ boxes: boxes }));
        """,
        roles=["head", "worker"],
    )
    assert [row["role"] for row in got["boxes"]] == ["head", "worker"]
    head, worker = got["boxes"]
    assert head["read"] == "Clock 2411 MHz · Cap none"
    assert head["value"] == ""
    assert head["save"] == "Save"
    assert head["disabled"] is False
    assert "1–3003" in head["range"]
    assert worker["read"] == "Clock 2405 MHz · Cap 2000 MHz"
    assert worker["value"] == "2000"
    assert "Head" in head["label"]
    assert "Worker" in worker["label"]


def test_four_sparks_and_unlock_first() -> None:
    got = _run(
        """
        globalThis.lockIsOn = true;
        paintGpuClocks();
        var boxes = serving.children.map(function (box) {
          return {
            role: box.getAttribute("data-role"),
            msg: box.querySelector(".gpu-clock-msg") && box.querySelector(".gpu-clock-msg").textContent,
            disabled: box.querySelector("button") && box.querySelector("button").disabled
          };
        });
        process.stdout.write(JSON.stringify({ boxes: boxes, count: serving.children.length }));
        """,
        roles=["head", "worker", "node3", "node4"],
    )
    assert got["count"] == 4
    assert [row["role"] for row in got["boxes"]] == ["head", "worker", "node3", "node4"]
    assert all(row["disabled"] is True for row in got["boxes"][:3])
    assert all("Unlock first" in (row["msg"] or "") for row in got["boxes"][:3])


def test_bad_input_and_in_flight_second_click_do_not_post_twice() -> None:
    got = _run(
        """
        paintGpuClocks();
        var head = gpuClockBox("head");
        var input = head.querySelector("input");
        input.value = "abc";
        saveGpuClock("head");
        var firstMsg = head.querySelector(".gpu-clock-msg").textContent;
        var firstPosts = posted.length;
        input.value = "2000";
        gpuClockSaving.head = true;
        saveGpuClock("head");
        process.stdout.write(JSON.stringify({ firstMsg: firstMsg, firstPosts: firstPosts, secondPosts: posted.length }));
        """
    )
    assert "whole number" in got["firstMsg"]
    assert got["firstPosts"] == 0
    assert got["secondPosts"] == 0


def test_running_model_asks_before_save() -> None:
    got = _run(
        """
        globalThis.runningRoles = { head: true };
        var notes = [];
        confirmGpuClock = function (note, proceed) { notes.push(note); };
        paintGpuClocks();
        var head = gpuClockBox("head");
        head.querySelector("input").value = "2000";
        saveGpuClock("head");
        process.stdout.write(JSON.stringify({ notes: notes, posts: posted.length }));
        """
    )
    assert got["posts"] == 0
    assert len(got["notes"]) == 1
    assert "running" in got["notes"][0].lower()
    assert "2000" in got["notes"][0]


def test_unreported_cap_shows_what_sparksi_set_not_none() -> None:
    got = _run(
        """
        nodeFacts[0].gpu = { clock_mhz: 2411, clock_max_mhz: 3003, clock_cap_reported: false };
        nodeFacts[1].gpu = { clock_mhz: 1990, clock_max_mhz: 3003, clock_cap_reported: false, clock_cap_set: { mhz: 2000, at: 1 } };
        paintGpuClocks();
        var head = gpuClockBox("head");
        var worker = gpuClockBox("worker");
        process.stdout.write(JSON.stringify({
          head: head.querySelector(".gpu-clock-read").textContent,
          worker: worker.querySelector(".gpu-clock-read").textContent,
          workerValue: worker.querySelector("input").value
        }));
        """
    )
    assert got["head"] == "Clock 2411 MHz · Cap not reported by the GPU"
    assert got["worker"] == "Clock 1990 MHz · Cap 2000 MHz, set from Sparksi"
    assert got["workerValue"] == "2000"


def test_zero_is_refused_on_the_page() -> None:
    got = _run(
        """
        paintGpuClocks();
        var head = gpuClockBox("head");
        head.querySelector("input").value = "0";
        saveGpuClock("head");
        process.stdout.write(JSON.stringify({ msg: head.querySelector(".gpu-clock-msg").textContent, posts: posted.length }));
        """
    )
    assert got["posts"] == 0
    assert "between 1 and 3003" in got["msg"]


def test_confirmed_save_sends_the_ok_and_server_ask_reprompts() -> None:
    got = _run(
        """
        var notes = [];
        confirmGpuClock = function (note, proceed) { notes.push(note); proceed(); };
        globalThis.runningRoles = { head: true };
        paintGpuClocks();
        var head = gpuClockBox("head");
        head.querySelector("input").value = "2000";
        saveGpuClock("head");
        var afterLocal = posted.slice();
        globalThis.runningRoles = {};
        var replies = [
          { ok: false, status: 409, body: { ok: false, needs_confirm: true, error: "A model is running on this Spark." } },
          { ok: true, status: 200, body: { ok: true, gpu: { clock_mhz: 1800, clock_max_mhz: 3003, clock_cap_mhz: 1800 } } }
        ];
        jpost = function (url, body) { posted.push({ url: url, body: body }); return Promise.resolve(replies.shift()); };
        var worker = gpuClockBox("worker");
        worker.querySelector("input").value = "1800";
        saveGpuClock("worker");
        setTimeout(function () {
          process.stdout.write(JSON.stringify({
            afterLocal: afterLocal,
            posted: posted,
            notes: notes,
            msg: worker.querySelector(".gpu-clock-msg").textContent
          }));
        }, 20);
        """
    )
    assert got["afterLocal"][0]["body"]["confirm_running"] is True
    worker_posts = [p for p in got["posted"] if p["url"].endswith("/b/gpu-clocks")]
    assert len(worker_posts) == 2
    assert "confirm_running" not in worker_posts[0]["body"]
    assert worker_posts[1]["body"]["confirm_running"] is True
    assert len(got["notes"]) == 2
    assert "Cap set to 1800" in got["msg"]
