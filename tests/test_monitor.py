# SPDX-License-Identifier: Apache-2.0

"""The monitor card is a snapshot dict. Hardware collection stays in dgx-monitor."""

from sparksi.serve import STATE, Handler, DualServer, bind_listeners
from sparksi.lock import Lock
import json
import threading
from urllib.request import urlopen


def test_monitor_route_uses_injected_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.monitor_fn = lambda: {
        "ok": True,
        "ts": "12:00:00",
        "hosts": [
            {"id": "a", "fresh": True, "mem_used_gib": 80, "mem_total_gib": 119, "mem_pct": 67},
            {"id": "b", "fresh": False, "unreachable": True, "age": 30},
        ],
        "loaded": [
            {
                "name": "Pair-EXL3",
                "state": "SPLIT",
                "span": ["a", "b"],
                "worker_name": "pair-exl3-worker",
                "worker_host": "b",
                "worker_up": False,
                "port": 8014,
                "tok_s": 0,
                "kv_pct": 1,
                "running": 0,
                "waiting": 0,
            }
        ],
        "loaded_summary": {"count": 1, "running": 0, "waiting": 0, "mode": "auto"},
    }
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/api/monitor", timeout=5) as resp:
            body = json.loads(resp.read().decode())
    finally:
        httpd.shutdown()
        STATE.monitor_fn = None
    assert body["hosts"][1]["unreachable"] is True
    assert body["loaded"][0]["state"] == "SPLIT"
    assert body["loaded"][0]["span"] == ["a", "b"]


def test_tailscale_bind_failure_stays_pending():
    def factory(host, port):
        if host != "127.0.0.1":
            raise OSError("Cannot assign requested address")
        return (host, port)

    servers, pending = bind_listeners(7500, ["127.0.0.1", "100.64.0.1"], factory)
    assert servers == [("127.0.0.1", 7500)]
    assert pending == ["100.64.0.1"]
