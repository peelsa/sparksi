# SPDX-License-Identifier: Apache-2.0

import time

from sparksi.activity import reset_for_tests as reset_activity, store as activity_store
from sparksi.config import CFG
from sparksi.recipes.runs import reset_for_tests as reset_runs, tick

HEAD = CFG.head.name


def test_ready_is_logged_once_per_load(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    reset_runs()
    reset_activity()

    class Runner:
        def invoke(self, host, req, timeout=20):
            return {
                "ok": True,
                "phase": "running",
                "container_status": "running",
                "listening": {"8010": True},
                "http": {"8010": {"status": 200, "path": "/health"}},
            }

    run = {
        "id": "ready1",
        "model_id": "demo",
        "engine": "vllm",
        "placement": "head",
        "port": 8010,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 600,
        "health_paths": ["/health"],
        "nodes": [
            {
                "role": "head",
                "host": HEAD,
                "run_id": "ready1-head",
                "container": "sparksi-demo",
                "stop_argv": [],
                "phase": "started",
            }
        ],
    }

    for _ in range(5):
        run = tick(run, Runner())

    assert run["phase"] == "ready"
    ready_events = [row for row in activity_store().events() if "phase=ready" in row["message"]]
    assert len(ready_events) == 1

    reset_runs()
    reset_activity()
