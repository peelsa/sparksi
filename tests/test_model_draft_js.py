# SPDX-License-Identifier: Apache-2.0

"""In-memory recipe drafts and the newest-run status words."""

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _script() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.split("/* model-draft-start */", 1)[1].split("/* model-draft-end */", 1)[0]


def _run(body: str) -> object:
    proc = subprocess.run(["node"], input=_script() + body, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_draft_survives_until_load_default_or_save() -> None:
    got = _run(
        """
        var store = {};
        rememberDraft(store, "left", { engine: "vllm", placement: "head", port: "8014", fields: { ctx: 8192 }, edited: true });
        rememberDraft(store, "right", { engine: "llamacpp", placement: "worker", port: "8015", fields: { ctx: 4096 }, edited: false });
        var back = recallDraft(store, "left");
        forgetDraft(store, "left");
        process.stdout.write(JSON.stringify({
          back: back,
          after: recallDraft(store, "left"),
          right: recallDraft(store, "right")
        }));
        """
    )
    assert got["back"]["engine"] == "vllm"
    assert got["back"]["port"] == "8014"
    assert got["back"]["fields"]["ctx"] == 8192
    assert got["back"]["edited"] is True
    assert got["after"] is None
    assert got["right"]["placement"] == "worker"


def test_status_uses_the_newest_run() -> None:
    got = _run(
        """
        var runs = [
          { model_id: "demo", phase: "failed", started: 100, error: "boom" },
          { model_id: "demo", phase: "ready", started: 200 },
          { model_id: "demo", phase: "stopped", started: 300 }
        ];
        var newest = newestRun(runs, "demo");
        var failed = newestRun([{ model_id: "demo", phase: "failed", started: 50, error: "weights missing\\nmore" }], "demo");
        var timed = newestRun([{ model_id: "demo", phase: "timed_out", started: 3600 }], "demo");
        process.stdout.write(JSON.stringify({
          newest: statusForRun(newest),
          failed: statusForRun(failed),
          timed: statusForRun(timed)
        }));
        """
    )
    assert got["newest"]["text"] == "Not running"
    assert got["newest"]["failed"] is False
    assert got["failed"]["text"].startswith("Last start failed: weights missing")
    assert got["failed"]["failed"] is True
    assert got["timed"]["text"].startswith("Last start timed out")
    page = INDEX_HTML.read_text(encoding="utf-8")
    assert "Stop on both Sparks" not in page
    assert 'id="model-load"' in page
    assert 'id="model-unload"' in page
    assert "Not running" in page
    assert "Load default recipe" in page
    kept = _run(
        """
        process.stdout.write(JSON.stringify({
          held: keepEntered("8014", "8013", "saved"),
          blank: keepEntered("", "8014", "saved"),
          reset: keepEntered("8014", "8010", "defaults"),
          draft: keepEntered("8013", "8014", "draft")
        }));
        """
    )
    assert kept["held"] == "8014"
    assert kept["blank"] == "8014"
    assert kept["reset"] == "8010"
    assert kept["draft"] == "8014"
