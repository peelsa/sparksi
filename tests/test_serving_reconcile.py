# SPDX-License-Identifier: Apache-2.0

"""Serving rows follow the monitor snapshot and the catalog identity, not a stand-in."""

import json
import subprocess

from sparksi.config import CFG
from sparksi.monitor.__main__ import snapshot
from sparksi.monitor.peer import STALE_SECS
from sparksi.serve import INDEX_HTML


class _Peer:
    def __init__(self, data, age=0.4):
        self._data = data
        self._age = age

    def snapshot(self):
        return self._data, self._age


def _hw(**over):
    base = {
        "mem": (119.0, 40.0),
        "cpu_pct": 1.0,
        "cpu_mhz": 1800.0,
        "gpu": [1.0, 40.0, 800.0],
        "zones": [],
        "http": {},
        "containers": [],
        "engines": [],
        "comfy": {},
        "truth": "",
    }
    base.update(over)
    return base


def _js(extra: str, cards=None) -> dict:
    html = INDEX_HTML.read_text(encoding="utf-8")
    identity = "function isVideoRow" + html.split("function isVideoRow", 1)[1].split("function stopLoaded", 1)[0]
    loaded = "function loadedOn" + html.split("function loadedOn", 1)[1].split('el("btn-stop-live")', 1)[0]
    roles = "function roleName" + html.split("function roleName", 1)[1].split("function machineByRole", 1)[0]
    receipt = "function acceptMonitor" + html.split("function acceptMonitor", 1)[1].split("function tickMonitor", 1)[0]
    draft = html.split("/* model-draft-start */", 1)[1].split("/* model-draft-end */", 1)[0]
    status = html.split("/* overview-status-start */", 1)[1].split("/* overview-status-end */", 1)[0]
    serve = "function serveStatus" + html.split("function serveStatus", 1)[1].split("function chainGraphic", 1)[0]
    endpoint = "function endpointFor" + html.split("function endpointFor", 1)[1].split("function serveTitle", 1)[0]
    words = html.split("function roleWord", 1)[1].split("function placementBadge", 1)[0]
    words = "var LOAD_PHASES = " + html.split("var LOAD_PHASES = ", 1)[1].split(";", 1)[0] + ";\nfunction roleWord" + words
    progress = html.split("function paintProgress", 1)[1].split("/* spark-card-start */", 1)[0]
    progress = "function paintProgress" + progress
    picker = html.split("function selectedCard", 1)[1].split("function placementBadge", 1)[0]
    picker = "function selectedCard" + picker
    run_for = "function runFor" + html.split("function runFor", 1)[1].split("/* overview-status-start */", 1)[0]
    opener = html.split("function openRecipe", 1)[1].split("function clear", 1)[0]
    opener = "function openRecipe" + opener + "function clear" + html.split("function clear", 1)[1].split("function jget", 1)[0]
    preamble = """
var modelRuns = [];
var modelCards = [];
var libraryRows = [];
var modelPick = "";
var lastMonitor = null;
var monitorFailed = false;
var progressSig = "";
var progressPhase = "";
var progressCard = "";
var progressRun = "";
var progressArmed = false;
var HEAD_ID = %s, WORKER_ID = %s, HEAD_NAME = "", WORKER_NAME = "";
var NODES = [{role: "head", id: HEAD_ID, name: "head"}, {role: "worker", id: WORKER_ID, name: "worker"}];
function rawHost(role) {
  var id = roleId(role);
  var name = roleName(role);
  var found = null;
  ((lastMonitor && lastMonitor.hosts) || []).forEach(function (host) {
    if (host && (host.id === id || host.id === name || host.id === role)) found = host;
  });
  return found;
}
function makeEl(tag) {
  var node = { tag: tag, children: [], parent: null, textContent: "", className: "", title: "", hidden: false, open: false };
  node.appendChild = function (child) { child.parent = node; node.children.push(child); return child; };
  node.removeChild = function (child) {
    node.children = node.children.filter(function (item) { return item !== child; });
    child.parent = null;
  };
  Object.defineProperty(node, "firstChild", { get: function () { return node.children[0] || null; } });
  Object.defineProperty(node, "childNodes", { get: function () { return node.children; } });
  node.closest = function (sel) {
    var here = node;
    while (here) { if (here.tag === sel) return here; here = here.parent; }
    return null;
  };
  return node;
}
var byId = {};
var document = { createElement: function (tag) { return makeEl(tag); }, getElementById: function (id) { return byId[id] || null; } };
function el(id) { return document.getElementById(id); }
""" % (json.dumps(CFG.head.id), json.dumps(CFG.worker.id))
    proc = subprocess.run(
        ["node"],
        input=preamble + identity + loaded + roles + receipt + opener + words + picker + run_for + draft + status + serve + endpoint + progress + extra,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _accept(snap: dict) -> str:
    body = {"hosts": snap["hosts"], "loaded": snap["loaded"]}
    return "acceptMonitor(" + json.dumps(body) + ");"


def test_span_flags_and_static_container_do_not_confirm_the_wrong_role() -> None:
    down = _hw(
        http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}},
        engines=[{"name": "current-head", "port": 8014, "tp": 2}],
    )
    worker = _hw(containers=[{"name": "current-worker", "status": "Exited"}])
    split = snapshot(_Peer(down), _Peer(worker), {})
    row = split["loaded"][0]
    assert row["worker_up"] is False
    assert row["container_source"] == "engine"
    assert row["container"] == "current-head"

    unknown = snapshot(_Peer(down), _Peer(worker, age=STALE_SECS + 5), {})
    unknown_row = unknown["loaded"][0]
    assert unknown_row["worker_up"] is None

    stale_truth = """
models:
  - port: 8022
    display: Example TP=2
    served_name: stale-served
    container: stale-box
"""
    headless = snapshot(
        _Peer(_hw(truth=stale_truth)),
        _Peer(_hw(containers=[{"name": "stale-box", "status": "Up"}])),
        {},
    )
    headless_row = next(item for item in headless["loaded"] if item.get("port") == 8022)
    assert headless_row["head_up"] is False
    assert headless_row["worker_up"] is True
    assert headless_row["container_source"] == "truth"

    reused = snapshot(
        _Peer(_hw(
            truth="""
models:
  - port: 8014
    display: Stale
    served_name: stale-served
    container: stale-box
""",
            http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}},
            announce_from=CFG.head.tailscale,
        )),
        _Peer(_hw()),
        {},
    )
    reused_row = next(item for item in reused["loaded"] if item.get("port") == 8014)
    assert reused_row["served"] == "current-served"
    assert reused_row["container"] == "stale-box"
    assert reused_row["container_source"] == "truth"
    assert reused_row["endpoint"] == ""

    got = _js(
        """
        libraryRows = [
          { id: "current", title: "Current", identity: { served: ["current-served"], container: "current-head" } },
          { id: "stale", title: "Stale", identity: { served: ["stale-served"], container: "stale-box" } }
        ];
        var runs = [{
          id: "joint", model_id: "current", phase: "ready", placement: "both", port: 8014, started: 5,
          nodes: [
            { role: "head", phase: "started", container: "current-head" },
            { role: "worker", phase: "started", container: "current-worker" }
          ]
        }];
        %s
        var downWorker = serveEntries("worker").map(function (run) { return run.id; });
        var downChain = sharedRun(runs);
        %s
        var unknownWorker = roleStanding(runs[0], "worker");
        var unknownText = serveStatus(runs[0], "worker");
        var unknownChain = sharedRun(runs);
        %s
        var headlessHead = roleStanding({
          id: "half", model_id: "stale", phase: "ready", placement: "both", port: 8022,
          nodes: [{ role: "head", phase: "started" }, { role: "worker", phase: "started", container: "stale-box" }]
        }, "head");
        var headlessWorker = serveStatus({
          id: "half", model_id: "stale", phase: "ready", placement: "both", port: 8022,
          nodes: [{ role: "head", phase: "started" }, { role: "worker", phase: "started", container: "stale-box" }]
        }, "worker");
        %s
        process.stdout.write(JSON.stringify({
          downWorker: downWorker,
          downChain: downChain,
          unknownWorker: unknownWorker,
          unknownText: unknownText,
          unknownChain: unknownChain,
          reused: libraryIdFor(lastMonitor.loaded[0]),
          endpoint: lastMonitor.loaded[0].endpoint,
          headlessHead: headlessHead,
          headlessWorker: headlessWorker
        }));
        """
        % (_accept(split), _accept(unknown), _accept(headless), _accept(reused))
    )
    assert got["downWorker"] == []
    assert got["downChain"] is None
    assert got["unknownWorker"] == "stale"
    assert got["unknownText"] == "Not confirmed"
    assert got["unknownChain"] is None
    assert got["reused"] == "current"
    assert got["endpoint"] == reused_row["endpoint"]
    assert got["headlessHead"] == ""
    assert "Ready" in got["headlessWorker"]


def test_independent_observations_do_not_chain_and_a_real_span_can() -> None:
    both = snapshot(
        _Peer(_hw(http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}})),
        _Peer(_hw(http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}})),
        {},
    )
    assert len(both["loaded"]) == 2
    assert all(not item.get("span") for item in both["loaded"])
    paired = snapshot(
        _Peer(_hw(
            truth="""
models:
  - port: 8014
    display: Example TP=2
    served_name: listed-served
    container: listed-head
""",
            http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}},
            engines=[{"name": "current-head", "port": 8014, "tp": 2}],
        )),
        _Peer(_hw(containers=[{"name": "current-worker", "status": "Up"}])),
        {},
    )
    span = paired["loaded"][0]
    assert span["head_up"] is True and span["worker_up"] is True
    assert span["container"] == "current-head"
    assert span["worker_name"] == "current-worker"
    assert span["container_source"] == "engine"
    got = _js(
        """
        libraryRows = [{ id: "current", identity: { served: ["current-served"], container: "current-head" } }];
        var old = {
          id: "old-joint", model_id: "current", phase: "ready", placement: "both", port: 8014, started: 1,
          nodes: [
            { role: "head", phase: "started", container: "old-head" },
            { role: "worker", phase: "started", container: "old-worker" }
          ]
        };
        var live = {
          id: "live-joint", model_id: "current", phase: "ready", placement: "both", port: 8014, started: 2,
          nodes: [
            { role: "head", phase: "started", container: "current-head" },
            { role: "worker", phase: "started", container: "current-worker" }
          ]
        };
        %s
        var independent = sharedRun([old]);
        var independentStatus = serveStatus(old, "head");
        %s
        process.stdout.write(JSON.stringify({
          independent: independent,
          independentStatus: independentStatus,
          mismatched: sharedRun([old]),
          matched: sharedRun([live]) && sharedRun([live]).id,
          ready: serveStatus(live, "head")
        }));
        """
        % (_accept(both), _accept(paired))
    )
    assert got["independent"] is None
    assert got["independentStatus"] == "Ready · port 8014"
    assert got["mismatched"] is None
    assert got["matched"] == "live-joint"
    assert got["ready"] == "Ready · port 8014"


def test_failed_refresh_and_receipt_age_drop_confirmation_only() -> None:
    snap = snapshot(
        _Peer(_hw(http={8014: {"served": "current-served", "metrics": {"num_requests_running": 0}}})),
        _Peer(_hw()),
        {},
    )
    got = _js(
        """
        libraryRows = [{ id: "current", identity: { served: ["current-served"] } }];
        var run = {
          id: "one", model_id: "current", phase: "ready", placement: "head", port: 8014,
          nodes: [{ role: "head", phase: "started" }]
        };
        %s
        var fresh = serveStatus(run, "head");
        var kept = (lastMonitor.loaded || []).length;
        failMonitor();
        var failed = serveStatus(run, "head");
        var still = (lastMonitor.loaded || []).length;
        monitorFailed = false;
        lastMonitor.received_at = Date.now() - 11000;
        var expired = serveStatus(run, "head");
        process.stdout.write(JSON.stringify({ fresh: fresh, failed: failed, expired: expired, kept: kept, still: still }));
        """
        % _accept(snap)
    )
    assert got["fresh"] == "Ready · port 8014"
    assert got["failed"] == "Not confirmed"
    assert got["expired"] == "Not confirmed"
    assert got["kept"] == got["still"] == 1


def test_loopback_producer_does_not_publish_the_announce_origin(monkeypatch) -> None:
    from sparksi import announce

    # 127.0.0.1:8014. The address word is little-endian; 8014 is 1F4E.
    monkeypatch.setattr(
        announce,
        "_listen_rows",
        lambda: ["0: 0100007F:1F4E 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 0 10 0"],
    )
    monkeypatch.setattr(announce, "_http_ports", lambda _engines: [8014])
    monkeypatch.setenv("SPARKSI_ADVERTISE", CFG.head.tailscale)
    seen = []

    class Response:
        def read(self, _n):
            return b'{"data":[{"id":"current-served"}]}'

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(url, timeout=0):
        seen.append(str(url))
        if str(url).endswith("/metrics"):
            raise OSError("no metrics")
        return Response()

    monkeypatch.setattr(announce.urllib.request, "urlopen", opener)
    produced = announce._http([])
    assert produced["8014"][2] == "127.0.0.1"
    assert len(produced["8014"]) == 3
    assert all(CFG.head.tailscale not in url for url in seen)
    peer = announce.to_peer({
        "h": CFG.head.id,
        "mem": [10.0, 4.0],
        "cpu": [1.0, 1800.0],
        "gpu": [1.0, 40.0, 800.0],
        "http": produced,
    })
    assert peer["http"][8014]["bind"] == "loopback"
    assert "listen" not in peer["http"][8014]
    peer["announce_from"] = CFG.head.tailscale
    snap = snapshot(_Peer(peer), _Peer(_hw()), {})
    row = snap["loaded"][0]
    assert row["bind"] == "loopback"
    assert row["endpoint"] == ""
    assert CFG.head.tailscale not in row["endpoint"]
    got = _js(
        """
        libraryRows = [{ id: "current", identity: { served: ["current-served"] } }];
        var run = {
          id: "one", model_id: "current", phase: "ready", placement: "head", port: 8014,
          nodes: [{ role: "head", phase: "started" }]
        };
        %s
        process.stdout.write(JSON.stringify(endpointFor(run, "head")));
        """
        % _accept(snap)
    )
    assert got["href"] == ""
    assert got["label"] == ""


def test_wildcard_configured_route_is_verified_before_publish(monkeypatch) -> None:
    from sparksi import announce

    monkeypatch.setattr(
        announce,
        "_listen_rows",
        lambda: ["0: 00000000:1F4E 00000000:0000 0A 00000000:00000000 00:00000000 00000000 0 0 0 10 0"],
    )
    monkeypatch.setattr(announce, "_http_ports", lambda _engines: [8014])
    monkeypatch.setenv("SPARKSI_ADVERTISE", CFG.head.tailscale)
    state = {"body": b'{"data":[{"id":"current-served"}]}', "fail": False, "calls": []}

    class Response:
        def __init__(self, body, url):
            self._body = body
            self._url = url

        def geturl(self):
            return self._url

        def read(self, _n):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def opener(url, timeout=0):
        text = str(url)
        state["calls"].append((text, timeout))
        if text.endswith("/metrics"):
            raise OSError("no metrics")
        if CFG.head.tailscale in text:
            if state["fail"]:
                raise OSError("unreachable")
            return Response(state["body"], text)
        if "127.0.0.1:8014" in text:
            return Response(b'{"data":[{"id":"current-served"}]}', text)
        raise OSError(text)

    monkeypatch.setattr(announce.urllib.request, "urlopen", opener)
    monkeypatch.setattr(announce, "_open_advertised", lambda url: opener(url, announce.ADVERTISE_TIMEOUT))

    def produce():
        produced = announce._http([])
        peer = announce.to_peer({
            "h": CFG.head.id,
            "mem": [10.0, 4.0],
            "cpu": [1.0, 1800.0],
            "gpu": [1.0, 40.0, 800.0],
            "http": produced,
        })
        peer["announce_from"] = "203.0.113.77"
        return produced, peer, snapshot(_Peer(peer), _Peer(_hw()), {})

    produced, peer, snap = produce()
    assert produced["8014"][2] == "0.0.0.0"
    assert produced["8014"][3] == CFG.head.tailscale
    assert peer["http"][8014]["bind"] == "wildcard"
    assert peer["http"][8014]["route"] == CFG.head.tailscale
    assert "listen" not in peer["http"][8014]
    assert snap["loaded"][0]["endpoint"] == f"http://{CFG.head.tailscale}:8014/v1"
    assert "203.0.113.77" not in snap["loaded"][0]["endpoint"]
    advertised = [item for item in state["calls"] if CFG.head.tailscale in item[0]]
    assert advertised
    assert all(timeout == announce.ADVERTISE_TIMEOUT for _url, timeout in advertised)
    assert announce.ADVERTISE_TIMEOUT <= 0.4
    got = _js(
        """
        libraryRows = [{ id: "current", identity: { served: ["current-served"] } }];
        var run = {
          id: "one", model_id: "current", phase: "ready", placement: "head", port: 8014,
          nodes: [{ role: "head", phase: "started" }]
        };
        %s
        var live = endpointFor(run, "head");
        lastMonitor.hosts[0].age = 30;
        var stale = endpointFor(run, "head");
        process.stdout.write(JSON.stringify({ live: live, stale: stale }));
        """
        % _accept(snap)
    )
    assert got["live"]["href"] == snap["loaded"][0]["endpoint"]
    assert got["stale"]["href"] == ""

    state["body"] = b'{"data":[{"id":"other-served"}]}'
    _produced, peer, snap = produce()
    assert "route" not in peer["http"][8014]
    assert snap["loaded"][0]["endpoint"] == ""

    state["fail"] = True
    _produced, peer, snap = produce()
    assert "route" not in peer["http"][8014]
    assert snap["loaded"][0]["bind"] == "wildcard"
    assert snap["loaded"][0]["endpoint"] == ""

    monkeypatch.delenv("SPARKSI_ADVERTISE", raising=False)
    state["fail"] = False
    before = len(state["calls"])
    _produced, peer, snap = produce()
    assert len(_produced["8014"]) == 3
    assert snap["loaded"][0]["endpoint"] == ""
    assert not any(CFG.head.tailscale in url for url, _timeout in state["calls"][before:])

    monkeypatch.setenv("SPARKSI_ADVERTISE", CFG.head.tailscale)
    state["fail"] = False
    state["body"] = b'{"data":[{"id":"current-served"}]}'
    _produced, peer, good = produce()
    peer["http"][8014]["route"] = "203.0.113.50"
    forged = snapshot(_Peer(peer), _Peer(_hw()), {})
    assert forged["loaded"][0]["endpoint"] == ""
    assert good["loaded"][0]["endpoint"].endswith(":8014/v1")


def test_progress_opens_only_on_a_new_action_or_error() -> None:
    got = _js(
        """
        var details = makeEl("details");
        var box = makeEl("div");
        details.appendChild(box);
        byId["model-progress"] = box;
        modelPick = "current";
        modelCards = [{ id: "current" }];
        modelRuns = [{ id: "old", model_id: "current", phase: "failed", nodes: [{ role: "head", phase: "failed" }] }];
        paintProgress();
        var historical = details.open;
        details.open = false;
        modelRuns[0].error = "later line";
        paintProgress();
        var refreshed = details.open;
        modelRuns = [{ id: "new", model_id: "current", phase: "starting", nodes: [{ role: "head", phase: "starting" }] }];
        paintProgress();
        var started = details.open;
        details.open = false;
        modelRuns[0].nodes[0].log_tail = "more";
        paintProgress();
        var held = details.open;
        modelPick = "ready-one";
        modelCards = [{ id: "ready-one" }, { id: "failed-one" }];
        modelRuns = [
          { id: "ready-run", model_id: "ready-one", phase: "ready", nodes: [{ role: "head", phase: "ready" }] },
          { id: "failed-run", model_id: "failed-one", phase: "failed", nodes: [{ role: "head", phase: "failed" }] }
        ];
        paintProgress();
        var sawReady = details.open;
        modelPick = "failed-one";
        paintProgress();
        var switched = details.open;
        process.stdout.write(JSON.stringify({
          historical: historical, refreshed: refreshed, started: started, held: held,
          sawReady: sawReady, switched: switched
        }));
        """
    )
    assert got["historical"] is False
    assert got["refreshed"] is False
    assert got["started"] is True
    assert got["held"] is False
    assert got["sawReady"] is False
    assert got["switched"] is False
