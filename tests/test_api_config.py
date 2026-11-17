# SPDX-License-Identifier: Apache-2.0

"""GET /api/config is display-only, and the dashboard does not embed node names."""

import copy
import json
import threading
from urllib.request import urlopen

from sparksi.catalog import load_catalog
from sparksi.config import CFG, Config
from sparksi.launch import plan_stage
from sparksi.serve import INDEX_HTML, STATE, DualServer, Handler, page_html
from sparksi.lock import Lock

HTML = INDEX_HTML.read_text(encoding="utf-8")

_BANNED_KEYS = {
    "ssh",
    "tailscale",
    "bind",
    "bind_deny",
    "bind_allow",
    "path",
    "paths",
    "home",
    "token",
    "catalog",
    "metrics_url",
    "root",
    "extra_blocked",
    "announce_addr",
    "state_dir",
    "legacy_archive_id",
    "legacy_mount_key",
    "legacy_row_keys",
    "uma_gib",
}


def _keys(obj):
    if isinstance(obj, dict):
        for key, val in obj.items():
            yield str(key)
            yield from _keys(val)
    elif isinstance(obj, list):
        for item in obj:
            yield from _keys(item)


def test_page_has_no_hardcoded_node_names() -> None:
    text = page_html()
    assert text == HTML
    for name in ("spark-a", "spark-b", "node-a", "node-b", "__HEAD_ID__", "__HEAD_NAME__", "__WORKER_ID__", "__WORKER_NAME__", "__ARCHIVE_LABEL__"):
        assert name not in text
    assert "/api/config" in text
    assert "Load on head" not in text
    assert "Load on worker" not in text
    assert 'id="btn-load-head"' in text
    assert 'id="btn-load-worker"' in text
    assert 'id="btn-load-pair"' in text
    assert ">Load</button>" in text
    assert "mount_archive" in text
    assert "archive_state" in text


def test_public_config_allowlist() -> None:
    pub = CFG.public()
    blob = json.dumps(pub)
    assert pub["head"] == {"id": "a", "name": "spark-a"}
    assert pub["worker"] == {"id": "b", "name": "spark-b"}
    assert pub["nodes"][0]["role"] == "head"
    assert pub["controller"] == {"name": "controller"}
    assert pub["archive"]["id"] == "archive"
    assert pub["archive"]["label"]
    assert set(_keys(pub)) <= {
        "nodes", "role", "id", "name",
        "head", "worker", "controller", "archive", "label",
    }
    for banned in ("100.64.0.", "192.0.2.", "0.0.0.0", "/home/", "/mnt/", "ssh", "tailscale"):
        assert banned not in blob
    assert _BANNED_KEYS.isdisjoint(set(_keys(pub)))


def test_public_config_drops_private_fields() -> None:
    raw = {
        "controller": {
            "name": "controller",
            "bind": ["127.0.0.1", "100.64.0.1"],
            "bind_deny": ["0.0.0.0", "192.0.2.11"],
            "announce_addr": "100.64.0.1",
            "catalog": "/home/USER/.config/sparksi/catalog.yaml",
            "metrics_url": "http://192.0.2.50:8428",
            "state_dir": "/home/USER/state",
        },
        "nodes": [
            {
                "id": "a",
                "name": "spark-a",
                "ssh": "ssh-target-marker",
                "tailscale": "100.64.0.11",
                "bind_deny": ["192.0.2.11"],
                "uma_gib": 121,
            },
            {
                "id": "b",
                "name": "spark-b",
                "ssh": "node-b",
                "tailscale": "100.64.0.12",
                "bind_deny": ["192.0.2.12"],
                "uma_gib": 121,
            },
        ],
        "archive": {
            "id": "archive",
            "node": "a",
            "root": "/mnt/archive",
            "label": "Archive",
            "extra_blocked": ["/mnt/archive-raw"],
        },
        "paths": {
            "home": "/home/USER",
            "hf_hub": "/home/USER/.cache/huggingface/hub",
            "model_truth": "/home/USER/ops/MODEL_TRUTH.yaml",
        },
        "compat": {
            "legacy_archive_id": "old-archive-id",
            "legacy_mount_key": "old_mount_key",
            "legacy_row_keys": {"old_tar_key": "archive_tar"},
        },
    }
    pub = Config(raw).public()
    blob = json.dumps(pub)
    assert pub["head"]["name"] == "spark-a"
    assert pub["archive"]["label"] == "Archive"
    for secret in (
        "ssh-target-marker",
        "node-b",
        "100.64.0.11",
        "100.64.0.12",
        "100.64.0.1",
        "192.0.2.11",
        "192.0.2.12",
        "192.0.2.50",
        "127.0.0.1",
        "0.0.0.0",
        "/home/USER",
        "/mnt/archive",
        "old-archive-id",
        "old_mount_key",
        "old_tar_key",
        "unlock",
    ):
        assert secret not in blob
    assert _BANNED_KEYS.isdisjoint(set(_keys(pub)))


def test_api_config_route_hides_private_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    saved = copy.deepcopy(CFG.raw)
    source = CFG.source
    planted = "PLANTED-SECRET-PATH"
    try:
        poisoned = copy.deepcopy(saved)
        poisoned["paths"]["home"] = f"/home/USER/{planted}"
        poisoned["nodes"][0]["ssh"] = "ssh-target-marker"
        poisoned["nodes"][0]["tailscale"] = "100.64.0.11"
        poisoned["controller"]["bind_deny"] = ["0.0.0.0", "192.0.2.11"]
        CFG.__init__(poisoned, source=source)
        STATE.lock = Lock(tmp_path / "lock")
        httpd = DualServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        port = httpd.server_address[1]
        try:
            with urlopen(f"http://127.0.0.1:{port}/api/config", timeout=15) as resp:
                assert resp.status == 200
                body = json.loads(resp.read().decode())
            blob = json.dumps(body)
            assert body["head"]["name"] == "spark-a"
            assert body["archive"]["label"]
            for secret in (planted, "ssh-target-marker", "100.64.0.11", "192.0.2.11", "0.0.0.0", "/home/"):
                assert secret not in blob
            with urlopen(f"http://127.0.0.1:{port}/", timeout=15) as resp:
                page = resp.read().decode()
            assert "spark-a" not in page
            assert "/api/config" in page
        finally:
            httpd.shutdown()
    finally:
        CFG.__init__(saved, source=source)


def test_legacy_row_key_still_renames(tmp_path) -> None:
    saved = copy.deepcopy(CFG.raw)
    source = CFG.source
    try:
        raw = copy.deepcopy(saved)
        raw["compat"] = {"legacy_row_keys": {"old_tar_key": "archive_tar"}}
        CFG.__init__(raw, source=source)
        path = tmp_path / "catalog.yaml"
        path.write_text(
            "host: {uma_gib: 121}\n"
            "tenants: {}\n"
            "residents: []\n"
            "recipes: {}\n"
            "library:\n"
            "  - id: row\n"
            "    old_tar_key: /mnt/archive/models/x\n"
        )
        row = load_catalog(path)["library"][0]
        assert row["archive_tar"] == "/mnt/archive/models/x"
        assert "old_tar_key" not in row
    finally:
        CFG.__init__(saved, source=source)


def test_stage_reads_archive_tar_without_legacy_tar_key() -> None:
    row = {
        "id": "only-new",
        "archive_tar": "/mnt/archive/models/releases/only-new/file.tar",
        "sha256": "abc",
        "source_path": "/home/USER/services/only-new",
        "bytes": 1000,
        "on": {"spark-a": "absent"},
        "local": {"spark-a": "/home/USER/services/only-new"},
        "placement": "single",
    }
    catalog = {"hosts": {"spark-a": {}}}
    probes = {
        "spark-a": {
            "ok": True,
            "mount_archive": True,
            "disk_free_gib": 1000,
            "disk_total_gib": 2000,
            "listening": {},
        }
    }
    got = plan_stage(catalog, row, "spark-a", probes, lock_on=False)
    assert got["verdict"] == "READY", got["reasons"]
    assert got["request"]["op"] == "stage_tar"
    assert got["request"]["tar"] == row["archive_tar"]
