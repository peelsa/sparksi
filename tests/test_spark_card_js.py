# SPDX-License-Identifier: Apache-2.0

"""Spark cards list a controllable service as status only. Start and Stop stay in Services."""

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _cards() -> list:
    html = INDEX_HTML.read_text(encoding="utf-8")
    script = html.split("/* spark-card-start */", 1)[1].split("/* spark-card-end */", 1)[0]
    preamble = r"""
var sparkSig = "";
var serviceCards = [
  { id: "studio", title: "Studio", port: 8188, host: "head", running: false, controllable: true, protected: true },
  { id: "studio-live", title: "Studio live", port: 8189, host: "head", running: true, controllable: true, protected: true },
  { id: "comfy", title: "ComfyUI", port: 8190, host: "head", running: true, controllable: false, protected: true }
];
var modelRuns = [{
  id: "run1",
  phase: "ready",
  port: 8010,
  model_id: "demo",
  placement: "both",
  nodes: [{ role: "head" }, { role: "worker" }]
}];
var nodeFacts = [
  { role: "head", id: "head", name: "head", reachable: true, mem: { used_gib: 10, total_gib: 120 }, gpu: { util_pct: 4, temp_c: 40 } },
  { role: "worker", id: "worker", name: "worker", reachable: true, mem: { used_gib: 8, total_gib: 120 }, gpu: { util_pct: 1, temp_c: 38 } }
];
var lastMonitor = { loaded: [] };
function loadedOn() { return false; }
function libraryIdFor() { return ""; }
function phaseWord(phase) { return phase || ""; }
function roleId(role) { return role; }
function roleName(role) { return role; }
function sparkRoles() { return ["head", "worker"]; }
function roleLabel(role) { return role === "worker" ? "Worker" : "Head"; }
function lockOn() { return false; }
function trimNum(value) { return String(value); }
function toF(value) { return value; }
function stopLiveItem() {}
function serviceAction() { throw new Error("service control rendered on a node card"); }
function makeEl(tag) {
  var kids = [];
  return {
    tag: tag,
    children: kids,
    childNodes: kids,
    textContent: "",
    className: "",
    type: "",
    disabled: false,
    title: "",
    appendChild: function (child) { kids.push(child); return child; },
    addEventListener: function () {}
  };
}
var spark = makeEl("aside");
function el(id) { return id === "model-status" ? spark : null; }
function clear(node) { node.children.length = 0; }
var document = { createElement: function (tag) { return makeEl(tag); } };
"""
    driver = r"""
paintSparkLive();
function flat(node) {
  return {
    tag: node.tag,
    text: node.textContent || "",
    cls: node.className || "",
    kids: (node.children || []).map(flat)
  };
}
process.stdout.write(JSON.stringify(flat(spark)));
"""
    proc = subprocess.run(["node"], input=preamble + script + driver, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)["kids"]


def _walk(node):
    yield node
    for child in node.get("kids") or []:
        yield from _walk(child)


def _buttons(node) -> list[str]:
    return [item["text"] for item in _walk(node) if item.get("tag") == "button"]


def _row_with(card, text: str) -> dict:
    for node in _walk(card):
        if node.get("tag") == "div" and any(child.get("text") == text for child in node.get("kids") or []):
            return node
    raise AssertionError(text)


def test_node_cards_have_no_service_controls() -> None:
    cards = _cards()
    assert [card["kids"][0]["text"] for card in cards] == ["Head", "Worker"]
    head, worker = cards
    assert not any(item.get("text") == "Studio · Off" for item in _walk(head))
    assert not any(item.get("text") == "Studio live · On" for item in _walk(head))
    assert "Start" not in _buttons(head)
    assert "Stop" not in _buttons(head)
    protected = _row_with(head, "ComfyUI · port 8190 · up")
    assert _buttons(protected) == []
    assert any(child.get("text") == "Protected" for child in protected["kids"])
    model = _row_with(head, "demo · port 8010 · ready")
    assert _buttons(model) == []
    assert _buttons(worker) == []
    assert not any(item.get("text", "").endswith("· On") or item.get("text", "").endswith("· Off") for item in _walk(worker))
    page = INDEX_HTML.read_text(encoding="utf-8")
    services = page.split("function paintServices()", 1)[1].split("function serviceAction", 1)[0]
    assert 'start.textContent = "Start"' in services
    assert 'stop.textContent = "Stop"' in services
    card = page.split("/* spark-card-start */", 1)[1].split("/* spark-card-end */", 1)[0]
    assert "serviceAction" not in card
    assert "Start" not in card
