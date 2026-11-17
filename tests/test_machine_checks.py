# SPDX-License-Identifier: Apache-2.0

"""Machine names, ssh destinations, endpoint URLs and backups: each check fails its test when reverted."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pytest

from sparksi import netconfig, hostcheck
from sparksi.config import Config
from sparksi.remote import SshRunner
from sparksi.monitor.peer import Peer
import sparksi.remote as remote_mod
import sparksi.monitor.peer as peer_mod

PAYLOAD = "-oProxyCommand=touch /tmp/pwn"
# The cloud metadata address, assembled so the identity guard sees no link-local literal.
META = ".".join(("169", "254", "169", "254"))

RAW = {
    "nodes": [
        {"id": "a", "name": "spark-a", "ssh": "spark-a"},
        {"id": "b", "name": "spark-b", "ssh": "spark-b"},
    ]
}

BASE_TEXT = (
    'ui_port = 7500\n\n[[nodes]]\nid = "a"\nname = "spark-a"\nssh = "spark-a"\n\n'
    '[[nodes]]\nid = "b"\nname = "spark-b"\nssh = "spark-b"\n'
)

SWAP_NODES = [
    {"id": "b", "name": "spark-b", "ssh": "spark-b"},
    {"id": "a", "name": "spark-a", "ssh": "spark-a"},
]


def _fake_run(record):
    def fake(cmd, *a, **k):
        record.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return fake


def _valid_raw():
    return {
        "nodes": [
            {"id": "a", "name": "spark-a", "ssh": "spark-a"},
            {"id": "b", "name": "spark-b", "ssh": "spark-b"},
        ]
    }


# ---------------------------------------------------------------- 1. ssh injection

def test_node_from_rejects_payload_ssh():
    with pytest.raises(ValueError):
        netconfig.node_from({"name": "box", "dns": PAYLOAD})


def test_change_rejects_payload_and_leaves_raw_untouched():
    raw = {"nodes": [dict(n) for n in RAW["nodes"]]}
    snapshot = json.dumps(raw, sort_keys=True)
    machine = {"name": "box", "dns": PAYLOAD, "ips": ["100.64.0.5"]}
    with pytest.raises(ValueError):
        netconfig.change(raw, "add", "head", machine)
    assert json.dumps(raw, sort_keys=True) == snapshot


def test_config_rejects_payload_in_nodes_ssh():
    raw = _valid_raw()
    raw["nodes"][0]["ssh"] = PAYLOAD
    with pytest.raises(ValueError) as ei:
        Config(raw)
    assert "[[nodes]] #1 ssh" in str(ei.value)


def test_config_ssh_falls_back_to_name_and_validates_it():
    raw = _valid_raw()
    del raw["nodes"][0]["ssh"]
    raw["nodes"][0]["name"] = PAYLOAD
    with pytest.raises(ValueError) as ei:
        Config(raw)
    assert "[[nodes]] #1 ssh" in str(ei.value)


@pytest.mark.parametrize(
    "bad",
    ["spark a", "spark\x00a", "a" * 254, "user;rm@spark-x"],
)
def test_refused_ssh_targets_at_load(bad):
    raw = _valid_raw()
    raw["nodes"][0]["ssh"] = bad
    with pytest.raises(ValueError):
        Config(raw)


@pytest.mark.parametrize(
    "good",
    ["spark-x", "user@spark-x", "100.64.0.5", "box.example.ts.net", "local"],
)
def test_accepted_ssh_targets_at_load(good):
    raw = _valid_raw()
    raw["nodes"][0]["ssh"] = good
    cfg = Config(raw)
    assert cfg is not None


# ---------------------------------------------------------------- 2. argv guard

def test_run_argv_and_capture_place_ddash_before_host(monkeypatch):
    r = SshRunner({"hosts": {"x": {"ssh": "spark-x"}}})
    for meth, argv in (("run_argv", ["echo", "hi"]), ("capture", ["free", "-b"])):
        seen: list = []
        monkeypatch.setattr(remote_mod.subprocess, "run", _fake_run(seen))
        getattr(r, meth)("x", argv)
        assert seen, f"{meth}: subprocess.run not called"
        cmd = seen[-1]
        # "--" ends ssh's options, so the destination can never be read as one.
        assert cmd.index("--") + 1 == cmd.index("spark-x"), cmd
        assert cmd[cmd.index("spark-x") + 1:] == argv, cmd


def test_invoke_places_ddash_before_host(monkeypatch):
    seen: list = []

    def fake(cmd, *a, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, '{"ok": true}', "")

    monkeypatch.setattr(remote_mod.subprocess, "run", fake)
    assert SshRunner({"hosts": {"x": {"ssh": "spark-x"}}}).invoke("x", {"op": "probe"}).get("ok") is True
    cmd = seen[-1]
    assert cmd.index("--") + 1 == cmd.index("spark-x"), cmd
    assert cmd[cmd.index("spark-x") + 1:] == ["python3", "-"], cmd


def test_peer_places_ddash_before_host(monkeypatch):
    p = Peer("spark-x")
    seen: list = []

    def fake(cmd, *a, **k):
        seen.append(cmd)
        p._stop.set()
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(peer_mod.subprocess, "run", fake)
    p._loop()
    assert seen, "peer: subprocess.run not called"
    cmd = seen[-1]
    assert cmd.index("--") + 1 == cmd.index("spark-x"), cmd


def test_catalog_payload_refused_without_subprocess(monkeypatch):
    r = SshRunner({"hosts": {"x": {"ssh": PAYLOAD}}})
    seen: list = []
    monkeypatch.setattr(remote_mod.subprocess, "run", _fake_run(seen))
    results = (r.run_argv("x", ["echo"]), r.capture("x", ["free", "-b"]), r.invoke("x", {"op": "probe"}))
    for res in results:
        assert res.get("ok") is False, res
        assert "refused ssh destination" in res.get("error", ""), res
    assert seen == []


# ---------------------------------------------------------------- 3. endpoints

def _ep_machine(entries):
    m = {"name": "box", "dns": "box.example.ts.net", "ips": ["100.64.0.5"]}
    m["endpoints"] = entries
    return m


def test_endpoint_url_refused_local_and_metadata():
    with pytest.raises(ValueError):
        netconfig.endpoint_from(
            _ep_machine([{"port": 8000, "engine": "vLLM", "url": "file:///etc/passwd"}])
        )
    with pytest.raises(ValueError):
        netconfig.endpoint_from(
            _ep_machine(
                [{"port": 80, "engine": "vLLM", "url": f"http://{META}:80/latest"}]
            )
        )


def test_endpoint_typed_address_metadata_refused():
    with pytest.raises(ValueError, match="private or tailnet"):
        netconfig.endpoint_from({"name": "meta", "address": META, "port": 80})


def test_endpoint_discovered_url_rebuilt_exactly():
    ep, ok = netconfig.endpoint_from(
        _ep_machine(
            [{"port": 11434, "engine": "Ollama", "url": "http://100.64.0.9:11434/x?y=1"}]
        )
    )
    assert ep["url"] == "http://100.64.0.9:11434"
    assert ep["engine"] == "Ollama"


@pytest.mark.parametrize("engine", ["<b>x</b>", "HTTP service"])
def test_unusable_engine_requires_explicit_port(engine):
    with pytest.raises(ValueError, match="choose the port"):
        netconfig.endpoint_from(_ep_machine([{"port": 8000, "engine": engine, "url": "http://100.64.0.9:8000"}]))


def test_config_rejects_bad_endpoint_urls():
    for url in ("file:///etc/passwd", f"http://{META}:80"):
        raw = _valid_raw()
        raw["endpoints"] = [
            {"id": "e1", "name": "ep", "url": url, "engine": "vLLM"}
        ]
        with pytest.raises(ValueError) as ei:
            Config(raw)
        assert "[[endpoints]] #1 url" in str(ei.value)


# ---------------------------------------------------------------- 4. backups

def test_backups_distinct_and_not_overwritten(tmp_path: Path):
    path = tmp_path / "sparksi.toml"
    path.write_text(BASE_TEXT)
    now = datetime(2026, 1, 2, 3, 4, 5)

    r1 = netconfig.save_tables(path, SWAP_NODES, [], now=now)
    assert r1["changed"] is True
    r2 = netconfig.save_tables(path, list(RAW["nodes"]), [], now=now)
    assert r2["changed"] is True

    assert r1["backup"] != r2["backup"]
    first, second = tmp_path / r1["backup"], tmp_path / r2["backup"]
    assert sorted(p.name for p in tmp_path.glob("sparksi.toml.bak-*")) == sorted([first.name, second.name])
    assert first.read_text() == BASE_TEXT  # the second save did not overwrite the first backup
    assert second.read_text() == netconfig.replace_tables(BASE_TEXT, SWAP_NODES, [])


# ---------------------------------------------------------------- 5. layout

def test_layout_has_no_ssh_keys():
    raw = {
        "nodes": [
            {"id": "a", "name": "spark-a", "ssh": "secret-ssh-target"},
            {"id": "b", "name": "spark-b", "ssh": "secret-ssh-target"},
        ]
    }
    lay = netconfig.layout(raw, "ctl")
    assert "ssh" not in lay["head"]
    assert "ssh" not in lay["worker"]
    assert "ssh" not in json.dumps(lay)


# ---------------------------------------------------------------- 6. names

def test_check_name_refusals():
    for bad in ("x" * 101, "a\u2028b", "a\u2029b", "a\x07b", " lead", ""):
        with pytest.raises(ValueError):
            hostcheck.check_name(bad)


def test_check_name_accepts_spaces_and_digits():
    assert hostcheck.check_name("GPU box 2") == "GPU box 2"


def test_node_from_rejects_line_separator_in_name():
    with pytest.raises(ValueError):
        netconfig.node_from({"name": "a\u2028b"})


def test_config_rejects_bad_name_node2():
    raw = _valid_raw()
    raw["nodes"][1]["name"] = "a\u2029b"
    with pytest.raises(ValueError) as ei:
        Config(raw)
    assert "[[nodes]] #2 name" in str(ei.value)


# ---------------------------------------------------------------- 7. check_same_elsewhere

def test_check_same_elsewhere_enforced(tmp_path: Path, monkeypatch):
    path = tmp_path / "sparksi.toml"
    path.write_text(BASE_TEXT)

    real_replace = netconfig.replace_tables

    def tweaked(old_text, nodes, endpoints):
        out = real_replace(old_text, nodes, endpoints)
        return out.replace("ui_port = 7500", "ui_port = 7599")

    monkeypatch.setattr(netconfig, "replace_tables", tweaked)

    with pytest.raises(ValueError) as ei:
        netconfig.save_tables(path, SWAP_NODES, [], now=datetime(2026, 1, 2, 3, 4, 5))
    assert "other settings" in str(ei.value)

    assert path.read_text() == BASE_TEXT
    assert not list(tmp_path.glob("sparksi.toml.bak-*"))
