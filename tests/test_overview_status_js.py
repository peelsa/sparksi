# SPDX-License-Identifier: Apache-2.0

"""Overview cards reuse the Models list status, and the lock line has no raw ttl."""

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _run(extra: str) -> object:
    html = INDEX_HTML.read_text(encoding="utf-8")
    draft = html.split("/* model-draft-start */", 1)[1].split("/* model-draft-end */", 1)[0]
    status = html.split("/* overview-status-start */", 1)[1].split("/* overview-status-end */", 1)[0]
    serve = "function serveStatus" + html.split("function serveStatus", 1)[1].split("function chainGraphic", 1)[0]
    endpoint = "function endpointFor" + html.split("function endpointFor", 1)[1].split("function serveTitle", 1)[0]
    identity = html.split("function isVideoRow", 1)[1]
    identity = "function isVideoRow" + identity.split("function stopLoaded", 1)[0]
    loaded = html.split("function loadedOn", 1)[1].split("el(\"btn-stop-live\")", 1)[0]
    loaded = "function loadedOn" + loaded
    roles = html.split("function roleName", 1)[1].split("function machineByRole", 1)[0]
    roles = "function roleName" + roles
    receipt = html.split("function acceptMonitor", 1)[1].split("function tickMonitor", 1)[0]
    receipt = "function acceptMonitor" + receipt
    lock = html.split("/* lock-line-start */", 1)[1].split("/* lock-line-end */", 1)[0]
    preamble = r"""
var modelRuns = [];
var modelCards = [];
var libraryRows = [];
var lastMonitor = { loaded: [] };
var monitorFailed = false;
var HEAD_ID = "head", WORKER_ID = "worker", HEAD_NAME = "", WORKER_NAME = "";
function runFor(id) { return newestRun(modelRuns, id); }
function rawHost(role) {
  var found = null;
  ((lastMonitor && lastMonitor.hosts) || []).forEach(function (host) {
    if (host && (host.id === role || host.name === role)) found = host;
  });
  return found;
}
function roleLabel(role) { return role === "worker" ? "Worker" : "Head"; }
function roleWord(role) {
  if (role === "worker") return "Worker";
  if (role === "both") return "Both";
  return "Head";
}
function makeEl(tag) {
  var kids = [];
  return {
    tag: tag,
    children: kids,
    textContent: "",
    className: "",
    title: "",
    appendChild: function (child) { kids.push(child); return child; }
  };
}
var document = { createElement: function (tag) { return makeEl(tag); } };
"""
    proc = subprocess.run(
        ["node"],
        input=preamble + identity + loaded + roles + receipt + draft + status + serve + endpoint + lock + extra,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_overview_cards_reuse_model_status() -> None:
    got = _run(
        r"""
        libraryRows = [
          { id: "flash", title: "Example Flash", identity: { served: ["flash-served"] } },
          { id: "solo", title: "Example Solo", identity: { served: ["solo-served"] } }
        ];
        modelCards = [
          { id: "flash", title: "Example Flash" },
          { id: "solo", title: "Example Solo" }
        ];
        modelRuns = [
          {
            model_id: "flash",
            phase: "waiting_health",
            placement: "both",
            port: 8014,
            started: 20,
            nodes: [{ role: "worker" }, { role: "head" }]
          },
          {
            model_id: "solo",
            phase: "failed",
            placement: "head",
            port: 8010,
            started: 10,
            error: "weights missing\nmore",
            nodes: [{ role: "head" }]
          }
        ];
        var head = makeEl("a");
        var worker = makeEl("a");
        paintOverviewRuns(head, "head");
        paintOverviewRuns(worker, "worker");
        modelRuns = [{
          model_id: "flash",
          phase: "ready",
          placement: "both",
          port: 8014,
          started: 30,
          nodes: [{ role: "worker" }, { role: "head" }]
        }];
        var ready = makeEl("a");
        paintOverviewRuns(ready, "head");
        function texts(node) { return node.children.map(function (child) { return child.textContent; }); }
        var unseen = { head: texts(head), worker: texts(worker), ready: texts(ready) };
        acceptMonitor({ hosts: [{ id: "head", fresh: true, age: 1 }, { id: "worker", fresh: true, age: 1 }], loaded: [
          { host: "head", port: 8014, served: "flash-served", state: "UP" },
          { host: "worker", port: 8014, served: "flash-served", state: "UP" }
        ]});
        modelRuns = [{
          model_id: "flash", phase: "ready", placement: "both", port: 8014, started: 30,
          nodes: [{ role: "head", phase: "ready" }, { role: "worker", phase: "ready" }]
        }];
        var seen = makeEl("a");
        paintOverviewRuns(seen, "head");
        process.stdout.write(JSON.stringify({
          unseen: unseen,
          seen: texts(seen),
          failedClass: head.children.filter(function (child) { return child.className === "is-last-fail"; }).length
        }));
        """
    )
    assert got["unseen"]["head"] == ["Example Flash · Not confirmed"]
    assert "Ready" not in "".join(got["unseen"]["head"] + got["unseen"]["worker"] + got["unseen"]["ready"])
    assert not any(line.startswith("Example Solo") for line in got["unseen"]["head"] + got["unseen"]["worker"])
    assert got["seen"] == ["Example Flash · Ready · port 8014"]
    assert got["failedClass"] == 0


def test_shared_run_is_one_record_not_a_matching_name() -> None:
    got = _run(
        r"""
        libraryRows = [{ id: "flash", identity: { served: ["flash-served"] } }];
        acceptMonitor({ hosts: [{ id: "head", fresh: true, age: 1 }, { id: "worker", fresh: true, age: 1 }], loaded: [
          { host: "head", port: 8014, served: "flash-served", state: "UP" },
          { host: "worker", port: 8014, served: "flash-served", state: "UP" }
        ]});
        var linked = sharedRun([{
          id: "run-1", model_id: "flash", phase: "ready", placement: "both", port: 8014,
          nodes: [{ role: "head", phase: "ready" }, { role: "worker", phase: "ready" }]
        }]);
        var unconfirmed = (acceptMonitor({ hosts: [], loaded: [] }), sharedRun([{
          id: "run-1", model_id: "flash", phase: "ready", placement: "both", port: 8014,
          nodes: [{ role: "head", phase: "ready" }, { role: "worker", phase: "ready" }]
        }]));
        acceptMonitor({ hosts: [{ id: "head", fresh: true, age: 1 }, { id: "worker", fresh: true, age: 1 }], loaded: [
          { host: "head", port: 8014, served: "flash-served", state: "UP" },
          { host: "worker", port: 8014, served: "flash-served", state: "UP" }
        ]});
        var names = sharedRun([
          { id: "run-1", model_id: "flash", phase: "ready", placement: "head", nodes: [{ role: "head" }] },
          { id: "run-2", model_id: "flash", phase: "ready", placement: "worker", nodes: [{ role: "worker" }] }
        ]);
        var stopped = sharedRun([{
          id: "run-9", model_id: "flash", phase: "stopped", placement: "both",
          nodes: [{ role: "head", phase: "stopped" }, { role: "worker", phase: "stopped" }]
        }]);
        process.stdout.write(JSON.stringify({
          linked: linked && linked.id,
          unconfirmed: unconfirmed,
          names: names,
          stopped: stopped
        }));
        """
    )
    assert got["linked"] is None
    assert got["unconfirmed"] is None
    assert got["names"] is None
    assert got["stopped"] is None


def test_each_role_keeps_its_own_current_run() -> None:
    got = _run(
        r"""
        var runs = [
          { id: "old-both", model_id: "m", started: 1, phase: "failed", nodes: [
            { role: "head", phase: "failed" }, { role: "worker", phase: "timed_out" }
          ]},
          { id: "head-now", model_id: "m", started: 2, phase: "ready", nodes: [{ role: "head", phase: "ready" }] },
          { id: "worker-now", model_id: "m", started: 3, phase: "ready", nodes: [{ role: "worker", phase: "ready" }] },
          { id: "half", model_id: "other", started: 4, phase: "ready", nodes: [
            { role: "head", phase: "stopped" }, { role: "worker", phase: "ready" }
          ]}
        ];
        libraryRows = [
          { id: "m", identity: { served: ["m-served"] } },
          { id: "other", identity: { served: ["other-served"] } }
        ];
        modelRuns = runs;
        var before = serveStatus(runs[1], "head");
        acceptMonitor({ hosts: [{ id: "head", fresh: true, age: 1 }, { id: "worker", fresh: true, age: 1 }], loaded: [
          { host: "head", port: 8014, served: "m-served", state: "UP" },
          { host: "worker", served: "m-served", state: "UP" },
          { host: "worker", served: "other-served", state: "UP" }
        ]});
        runs[1].port = 8014;
        process.stdout.write(JSON.stringify({
          before: before,
          head: serveEntries("head").map(function (run) { return run.id; }),
          worker: serveEntries("worker").map(function (run) { return run.id; }),
          chain: sharedRun(runs),
          none: sharedRun([]),
          stoppedNode: sharedRun([runs[3]])
        }));
        """
    )
    assert got["before"] == "Not confirmed"
    assert got["head"] == ["head-now"]
    assert "old-both" not in got["worker"]
    assert got["chain"] is None
    assert got["none"] is None
    assert got["stoppedNode"] is None


def test_failed_parent_does_not_keep_stale_children_current() -> None:
    got = _run(
        r"""
        var runs = [
          { id: "old-a", model_id: "wide", phase: "failed", placement: "both", port: 8014, started: 1, nodes: [
            { role: "head", phase: "started" }, { role: "worker", phase: "started" }
          ]},
          { id: "old-b", model_id: "other", phase: "failed", placement: "both", port: 8015, started: 2, nodes: [
            { role: "head", phase: "started" }, { role: "worker", phase: "started" }
          ]},
          { id: "live", model_id: "solo", phase: "ready", placement: "head", port: 8014, started: 3, nodes: [
            { role: "head", phase: "started" }
          ]},
          { id: "split", model_id: "split", phase: "failed", placement: "both", port: 8020, started: 4, nodes: [
            { role: "head", phase: "failed" }, { role: "worker", phase: "started" }
          ]}
        ];
        libraryRows = [
          { id: "wide", identity: { served: ["wide-served"] } },
          { id: "other", identity: { served: ["other-served"] } },
          { id: "solo", identity: { served: ["solo-served"] } },
          { id: "split", identity: { served: ["split-served"] } }
        ];
        modelRuns = runs;
        var plain = {
          head: serveEntries("head").map(function (run) { return run.id; }),
          worker: serveEntries("worker").map(function (run) { return run.id; }),
          status: serveStatus(runs[2], "head")
        };
        acceptMonitor({ hosts: [{ id: "head", fresh: true, age: 1 }, { id: "worker", fresh: true, age: 1 }], loaded: [
          { host: "head", port: 8014, served: "solo-served", state: "UP", listen: "203.0.113.10" },
          { host: "worker", port: 8020, served: "split-served", state: "UP", listen: "203.0.113.20" },
          { host: "head", port: 8014, served: "wide-served", state: "UP", listen: "203.0.113.30" }
        ]});
        var observed = {
          head: serveEntries("head").map(function (run) { return run.id; }),
          worker: serveEntries("worker").map(function (run) { return run.id; }),
          chain: sharedRun(runs),
          status: serveStatus(runs[3], "worker"),
          solo: endpointFor(runs[2], "head").href,
          wide: endpointFor(runs[0], "head").href,
          workerLabel: endpointFor(runs[3], "worker").label
        };
        lastMonitor.hosts[1].fresh = false;
        var stale = serveEntries("worker").map(function (run) { return run.id; });
        process.stdout.write(JSON.stringify({ plain: plain, observed: observed, stale: stale }));
        """
    )
    assert got["plain"]["status"] == "Not confirmed"
    assert "old-a" not in got["plain"]["head"]
    assert got["plain"]["worker"] == []
    assert got["observed"]["worker"] == ["split"]
    assert "old-b" not in got["observed"]["head"]
    assert got["observed"]["chain"] is None
    assert got["observed"]["status"] == "Answering · port 8020"
    assert got["observed"]["solo"] == ""
    assert got["observed"]["wide"] == ""
    assert got["observed"]["workerLabel"] == "Uses the head endpoint"
    assert got["stale"] == []


def test_lock_line_drops_the_raw_ttl() -> None:
    got = _run(
        r"""
        process.stdout.write(JSON.stringify({
          line: lockHeadline(false, "unlock ttl=900s", "3m 48s left unlocked"),
          locked: lockHeadline(true, "unlock ttl=900s", "3m 48s left unlocked"),
          other: lockHeadline(false, "holder busy", "1m 02s left unlocked")
        }));
        """
    )
    assert got["line"] == "Unlocked · 3m 48s left unlocked"
    assert "unlock ttl" not in got["line"]
    assert "900s" not in got["line"]
    assert got["locked"] == "Locked"
    assert got["other"] == "Unlocked · holder busy · 1m 02s left unlocked"
    page = INDEX_HTML.read_text(encoding="utf-8")
    assert 'id="btn-lock"' in page
    assert 'id="btn-lock"' in page.split('id="view-models"', 1)[1].split('id="view-library"', 1)[0]
    assert "Legacy view" not in page
    assert "Model maximum" not in page
    assert "Head and worker share this run" in page
    assert "unlock ttl=" not in page.split("function lockHeadline", 1)[1].split("function lockRemain", 1)[0]


def test_restart_waits_for_both_sides() -> None:
    stopped = _run(
        r"""
        process.stdout.write(JSON.stringify({
          one: runIsStopped({ phase: "stopped", nodes: [{ role: "head", phase: "stopped" }] }),
          both: runIsStopped({ phase: "stopped", placement: "both", nodes: [
            { role: "worker", phase: "stopped" },
            { role: "head", phase: "started" }
          ] }),
          done: runIsStopped({ phase: "stopped", placement: "both", nodes: [
            { role: "worker", phase: "stopped" },
            { role: "head", phase: "stopped" }
          ] }),
          early: runIsStopped({ phase: "stopping", nodes: [{ role: "head", phase: "stopped" }] })
        }));
        """
    )
    assert stopped["one"] is True
    assert stopped["both"] is False
    assert stopped["done"] is True
    assert stopped["early"] is False
    page = INDEX_HTML.read_text(encoding="utf-8")
    assert "Stopping… then loading" in page
    assert "waitUntilStopped" in page


def test_unconfirmed_overview_line_names_the_machine_and_port() -> None:
    got = _run(
        r"""
        HEAD_NAME = "spark-one"; WORKER_NAME = "spark-two";
        var seenOn = { head: [{ name: "flash-served", port: 8014, health: "ready" }], worker: [] };
        function runningOn(role) { return seenOn[role] || []; }
        var run = { model_id: "flash", phase: "ready", placement: "both", port: 8014, started: 30,
                    nodes: [{ role: "head" }, { role: "worker" }] };
        process.stdout.write(JSON.stringify({
          head: whereStatus(run, "head", serveStatus(run, "head")),
          worker: whereStatus(run, "worker", serveStatus(run, "worker")),
          ready: whereStatus(run, "head", "Ready · port 8014")
        }));
        """
    )
    assert got["head"] == "spark-one · port 8014 · ready · not confirmed"
    assert got["worker"] == "spark-two · port 8014 · not confirmed"
    assert got["ready"] == "Ready · port 8014"  # a confirmed line is unchanged
