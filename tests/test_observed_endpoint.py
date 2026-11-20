"""A serving card with no owning run still shows the vetted client URL with Copy."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parents[1] / "src" / "sparksi" / "static" / "index.html"
URL = "http://100.64.0.5:8014/v1"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _block() -> str:
    html = PAGE.read_text(encoding="utf-8")
    script = html.split("/* serving-state-start */", 1)[1].split("/* serving-state-end */", 1)[0]
    word = re.search(r"  function roleWord\(role\) \{.*?\n  \}\n", html, re.S)
    assert word is not None
    return script + word.group(0)


STUBS = """
var ROLES = %(roles)s;
var INFO = %(info)s;
function sparkRoles() { return ROLES.slice(); }
function roleId(role) { return (INFO[role] || {}).id || role; }
function rawRoleName(role) { return (INFO[role] || {}).name || ""; }
"""

TWO = {"head": {"id": "a", "name": "spark-a"}, "worker": {"id": "b", "name": "spark-b"}}
FOUR = dict(TWO, node3={"id": "c", "name": "spark-c"}, node4={"id": "d", "name": "spark-d"})


def _paint(rows: list[dict], loaded: list[dict] | None, info: dict = TWO) -> dict:
    roles = list(info)
    script = STUBS % {"roles": json.dumps(roles), "info": json.dumps(info)} + _block()
    if loaded is not None:
        script += "var lastMonitor = " + json.dumps({"loaded": loaded}) + ";\n"
    script += "servingRows = " + json.dumps(rows) + ";\n"
    script += "var out = {};\nROLES.forEach(function (r) { out[r] = observedServing(r); });\n"
    script += "process.stdout.write(JSON.stringify(out));"
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _row(role: str, port: int = 8014, state: str = "answering") -> dict:
    return {"role": role, "state": state, "text": f"Answering · port {port}", "title": "example-served", "port": port, "run_id": ""}


def _pair(**extra) -> dict:
    mod = {
        "name": "example-served", "host": "a", "port": 8014, "served": "example-served", "span": ["a", "b"],
        "container": "example-engine", "container_source": "engine", "head_up": True, "worker_up": True,
        "worker_host": "b", "endpoint": URL,
    }
    mod.update(extra)
    return mod


def test_external_pair_head_shows_url_and_worker_points_at_head():
    out = _paint([_row("head"), _row("worker")], [_pair()])
    assert out["head"][0]["endpoint"] == {"label": URL, "href": URL}
    assert out["worker"][0]["endpoint"] == {"label": "Head endpoint · " + URL, "href": URL}
    assert out["head"][0]["status"] == "Answering · port 8014"


def test_span_by_name_also_counts():
    out = _paint([_row("head"), _row("worker")], [_pair(host="spark-a", span=["spark-a", "spark-b"])])
    assert out["head"][0]["endpoint"]["href"] == URL
    assert out["worker"][0]["endpoint"]["label"] == "Head endpoint · " + URL


def test_single_on_each_spark_keeps_its_own_url():
    other = "http://100.64.0.6:8014/v1"
    loaded = [
        {"name": "x", "host": "a", "port": 8014, "endpoint": URL},
        {"name": "y", "host": "b", "port": 8014, "endpoint": other},
    ]
    out = _paint([_row("head"), _row("worker")], loaded)
    assert out["head"][0]["endpoint"]["href"] == URL
    assert out["worker"][0]["endpoint"] == {"label": other, "href": other}


def test_own_single_wins_over_a_pair_on_the_same_port():
    mine = "http://100.64.0.6:8014/v1"
    loaded = [_pair(), {"name": "y", "host": "b", "port": 8014, "endpoint": mine}]
    out = _paint([_row("worker")], loaded)
    assert out["worker"][0]["endpoint"]["label"] == mine


@pytest.mark.parametrize(
    "endpoint",
    ["", "http://127.0.0.1:8014/v1", "http://localhost:8014/v1", "http://[::1]:8014/v1",
     "http://0.0.0.0:8014/v1", "javascript:alert(1)", 'http://x"y:8014/v1'],
)
def test_no_url_unless_vetted(endpoint):
    out = _paint([_row("head"), _row("worker")], [_pair(endpoint=endpoint)])
    assert out["head"][0]["endpoint"] == {}
    assert out["worker"][0]["endpoint"] == {}


def test_wrong_port_down_or_unreported_shows_no_url():
    assert _paint([_row("head", port=8020)], [_pair()])["head"][0]["endpoint"] == {}
    assert _paint([_row("head")], [_pair(state="DOWN")])["head"][0]["endpoint"] == {}
    assert _paint([_row("head")], None)["head"][0]["endpoint"] == {}
    assert _paint([_row("head", port=0)], [_pair()])["head"][0]["endpoint"] == {}


@pytest.mark.parametrize("state", ["split", "stale", "loading"])
def test_unsettled_states_show_no_url(state):
    out = _paint([_row("head", state=state), _row("worker", state=state)], [_pair()])
    assert out["head"][0]["endpoint"] == {}
    assert out["worker"][0]["endpoint"] == {}


def test_observed_identity_still_shows_url():
    out = _paint([_row("head", state="observed")], [_pair()])
    assert out["head"][0]["endpoint"]["href"] == URL


def test_four_sparks():
    url3 = "http://100.64.0.7:8030/v1"
    loaded = [
        _pair(),
        {"name": "z", "host": "c", "port": 8030, "endpoint": url3},
        {"name": "w", "host": "c", "port": 8040, "span": ["c", "d"], "endpoint": "http://100.64.0.7:8040/v1"},
    ]
    rows = [_row("head"), _row("worker"), _row("node3", port=8030), _row("node4", port=8040), _row("node4", port=8014)]
    out = _paint(rows, loaded, FOUR)
    assert out["head"][0]["endpoint"]["href"] == URL
    assert out["node3"][0]["endpoint"] == {"label": url3, "href": url3}
    assert out["node4"][0]["endpoint"]["label"] == "Spark 3 endpoint · http://100.64.0.7:8040/v1"
    # node4 is not part of the head pair on 8014.
    assert out["node4"][1]["endpoint"] == {}


def test_lead_off_site_shows_no_url_on_the_other_spark():
    out = _paint([_row("worker")], [_pair(host="z", span=["z", "b"])])
    assert out["worker"][0]["endpoint"] == {}


def test_card_shows_link_and_copy_for_an_observed_url():
    html = PAGE.read_text(encoding="utf-8")
    fill = re.search(r"  function fillServeBody\(body, lines\) \{.*?\n  \}\n", html, re.S)
    assert fill is not None
    dom = r"""
function Node(tag) { this.tagName = tag; this.children = []; this.attrs = {}; this.hidden = false; this.textContent = ""; this.className = ""; }
Node.prototype.appendChild = function (c) { this.children.push(c); return c; };
Node.prototype.setAttribute = function (k, v) { this.attrs[k] = String(v); };
Node.prototype.getAttribute = function (k) { return k in this.attrs ? this.attrs[k] : null; };
Node.prototype.removeAttribute = function (k) { delete this.attrs[k]; };
Node.prototype.addEventListener = function () {};
Node.prototype.querySelector = function (sel) {
  var cls = sel.replace(/^\./, ""), found = null;
  (function walk(n) { n.children.forEach(function (c) { if (!found && c.className === cls) found = c; walk(c); }); })(this);
  return found;
};
var document = { createElement: function (t) { return new Node(t); } };
function setText(n, t) { n.textContent = String(t); }
"""
    lines = [{"title": "example-served", "status": "Answering · port 8014", "endpoint": {"label": URL, "href": URL}}]
    script = dom + fill.group(0) + "var body = new Node('div');\nfillServeBody(body, " + json.dumps(lines) + ");\n"
    script += "var b = body.children[0]; var a = b.querySelector('.serve-endpoint'); var c = b.querySelector('.serve-copy');\n"
    script += "process.stdout.write(JSON.stringify({text: a.textContent, href: a.getAttribute('href'), hidden: a.hidden, copy: !c.hidden}));"
    proc = subprocess.run(["node"], input=script, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout) == {"text": URL, "href": URL, "hidden": False, "copy": True}
