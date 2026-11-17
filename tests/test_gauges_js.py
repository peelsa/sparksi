# SPDX-License-Identifier: Apache-2.0

"""Overview ribbon pills stay in place, and serving text stays on the role boxes."""

import json
import subprocess

from sparksi.serve import INDEX_HTML


def test_two_pills_and_one_serving_line() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    chunk = html.split("function gaugeArc", 1)[1].split("function paintOverview", 1)[0]
    serving = html.split("function servingPlace", 1)[1].split("function phaseWord", 1)[0]
    preamble = r"""
function gaugeArc""" + chunk + r"""
function servingPlace""" + serving + r"""
var SVGNS = "htt" + "p://www.w3.org/2000/svg";
function trimNum(value) { return String(Math.round(Number(value))); }
function setText(node, text) { if (node) node.textContent = text == null ? "" : String(text); }
function toF(c) { return c * 9 / 5 + 32; }
function roleWord(role) { return role === "both" ? "Both" : (role === "worker" ? "Worker" : "Head"); }
function roleName(role) { return role === "head" ? "spark-a" : "spark-b"; }
function sparkRoles() { return ["head", "worker"]; }
function machineByRole(role) {
  return { stats: { gpu_pct: role === "worker" ? 96 : 12, mem_used_gib: 115.2, mem_total_gib: 128 }, report: "fresh" };
}
function rawHost() { return { gpu_temp_c: 40 }; }
function matches(node, sel) {
  sel = String(sel || "");
  if (sel.indexOf(" ") !== -1) return false;
  if (sel.charAt(0) !== "." && sel.charAt(0) !== "[") return node.tag === sel;
  var parts = [];
  var rest = sel;
  while (rest) {
    if (rest.charAt(0) === ".") {
      var dot = rest.indexOf("[");
      if (dot === -1) { parts.push(["class", rest.slice(1)]); break; }
      parts.push(["class", rest.slice(1, dot)]);
      rest = rest.slice(dot);
      continue;
    }
    if (rest.charAt(0) === "[") {
      var end = rest.indexOf("]");
      var body = rest.slice(1, end);
      var eq = body.indexOf("=");
      parts.push(["attr", body.slice(0, eq), body.slice(eq + 2, -1)]);
      rest = rest.slice(end + 1);
      continue;
    }
    break;
  }
  for (var i = 0; i < parts.length; i++) {
    var part = parts[i];
    if (part[0] === "class" && String(node.className || "").split(/\s+/).indexOf(part[1]) === -1) return false;
    if (part[0] === "attr" && String((node.attrs || {})[part[1]] || "") !== part[2]) return false;
  }
  return parts.length > 0;
}
function walk(node, sel, out) {
  (node.children || []).forEach(function (child) {
    if (matches(child, sel)) out.push(child);
    walk(child, sel, out);
  });
}
function makeEl(tag) {
  var kids = [];
  return {
    tag: tag, children: kids, childNodes: kids, textContent: "", className: "", hidden: false,
    attrs: {},
    setAttribute: function (k, v) { this.attrs[k] = String(v); },
    getAttribute: function (k) { return this.attrs[k]; },
    appendChild: function (child) { kids.push(child); return child; },
    querySelector: function (sel) {
      var found = [];
      walk(this, sel, found);
      return found[0] || null;
    },
    querySelectorAll: function (sel) {
      var found = [];
      walk(this, sel, found);
      return found;
    }
  };
}
var ribbon = makeEl("div");
var servingBox = makeEl("div");
var removed = 0;
var origRemove = null;
function el(id) { return id === "ribbon" ? ribbon : (id === "serving-lines" ? servingBox : null); }
function clear(node) { if (!node) return; removed += (node.children || []).length; node.children.length = 0; node.childNodes.length = 0; }
var document = {
  createElement: function (tag) { return makeEl(tag); },
  createElementNS: function (ns, tag) { var n = makeEl(tag); n.ns = ns; return n; }
};
var modelCards = [{ id: "glm", title: "Example Flash" }];
var modelRuns = [{ model_id: "glm", phase: "ready", placement: "both", port: 8014, nodes: [{ role: "head" }, { role: "worker" }] }];
function newestRun(runs, id) { return runs[0]; }
function paintServing() {
  var box = el("serving-lines");
  clear(box);
  var seen = {};
  (modelRuns || []).forEach(function (run) {
    if (!run || seen[run.model_id]) return;
    var text = servingText(newestRun(modelRuns, run.model_id));
    if (!text) return;
    seen[run.model_id] = true;
    var line = document.createElement("p");
    line.textContent = text;
    box.appendChild(line);
  });
}
paintRibbon();
var first = ribbon.querySelectorAll('[data-metric="gpu"]').slice();
var before = removed;
paintRibbon();
var second = ribbon.querySelectorAll('[data-metric="gpu"]');
var pills = ribbon.querySelectorAll(".mem");
var labels = [];
pills.forEach(function (pill) {
  ["gpu", "mem"].forEach(function (metric) {
    var fig = pill.querySelector('[data-metric="' + metric + '"]');
    var name = fig.querySelector(".gauge-name");
    labels.push({ visible: name.textContent, aria: fig.attrs["aria-label"], state: fig.querySelector(".state").textContent });
  });
});
var vals = pills.map(function (pill) { return pill.querySelector(".val").textContent; });
var temps = pills.map(function (pill) { return pill.querySelector(".temp").textContent; });
paintServing();
process.stdout.write(JSON.stringify({
  pills: pills.length,
  n: ribbon.querySelectorAll(".gauge").length,
  same: first[0] === second[0] && first[1] === second[1],
  removedAfter: removed - before,
  labels: labels,
  vals: vals,
  temps: temps,
  pillAria: pills[0].attrs["aria-label"],
  serving: servingBox.children.map(function (p) { return p.textContent; })
}));
"""
    proc = subprocess.run(["node"], input=preamble, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    got = json.loads(proc.stdout)
    assert got["pills"] == 2
    assert got["n"] == 4
    assert got["same"] is True
    assert got["removedAfter"] == 0
    assert [item["visible"] for item in got["labels"]] == ["GPU", "Mem"] * 2
    assert got["labels"][0]["aria"].startswith("Head GPU")
    assert got["labels"][2]["aria"].startswith("Worker GPU")
    assert got["labels"][0]["state"] == "12%"
    assert got["labels"][2]["state"] == "96%"
    assert all("High" not in item["state"] for item in got["labels"])
    assert got["vals"] == ["115.2 / 128.0 GiB"] * 2
    assert got["temps"] == ["104 F"] * 2
    assert got["pillAria"].startswith("spark-a GPU 12%")
    assert got["serving"] == ["Example Flash · Both · Ready · port 8014"]
    assert "mem-strip" in html
    assert "gauge-groups" in html
    assert "container-type: inline-size" in html
    assert "overflow-wrap: anywhere" in html
    assert "max-width: 960px" not in html
    assert "http://" not in html
    assert "· High" not in html
    assert 'id="gauges"' not in html
    assert "clear(box)" not in html.split("function paintRibbon", 1)[1].split("function paintOverview", 1)[0]
    assert "Last start failed" not in html.split("function paintOverview", 1)[1].split("function toF", 1)[0]
