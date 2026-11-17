# SPDX-License-Identifier: Apache-2.0

"""group_machines joins public fleet data and never copies addresses."""

import json

from sparksi.groups import group_machines

NODES = [
    {"role": "head", "id": "a", "name": "head"},
    {"role": "worker", "id": "b", "name": "worker"},
]
MODELS = [
    {"id": "pair-exl3", "title": "Pair EXL3", "port": 8014, "placement": "pair"},
    {
        "id": "single-a",
        "title": "Single A",
        "port": 8015,
        "placement": "single",
        "seats": {"head": {"port": 8015}},
    },
    {
        "id": "comfy",
        "title": "ComfyUI video",
        "port": 8188,
        "placement": "single",
        "seats": {"head": {"port": 8188}},
    },
]
LIBRARY = [
    {"id": "pair-exl3", "gib": 42, "on": {"head": "live", "worker": "live"}, "local": {"head": "planted-path"}},
    {"id": "single-a", "gib": 18, "on": {"head": "local", "worker": "absent"}},
    {"id": "comfy", "gib": 40, "on": {"head": "live", "worker": "absent"}},
]


def _live(head_fresh=True, worker=None, loaded=None):
    worker = {"id": "b", "fresh": True, "age": 1} if worker is None else worker
    hosts = [
        {
            "id": "a",
            "fresh": head_fresh,
            "age": 1,
            "mem_used_gib": 80.4,
            "mem_total_gib": 119,
            "cpu_pct": 22,
            "gpu_pct": 41,
            "gpu_temp_f": 98,
            "disk_used_gib": 800,
            "disk_total_gib": 3700,
            "mount_archive": True,
            "ips": {"lan": "192.0.2.11", "tailscale": "100.64.0.11"},
        }
    ]
    if worker is not False:
        hosts.append(worker)
    return {"hosts": hosts, "loaded": loaded or []}


def _grouped(**kwargs):
    return group_machines(
        NODES,
        MODELS,
        live=kwargs.get("live", _live()),
        library=kwargs.get("library", LIBRARY),
        controller=kwargs.get("controller", {"name": "controller"}),
        extras=kwargs.get("extras", [{"name": "extra"}]),
    )


def _by_role(out):
    return {row["role"] if row["role"] != "extra" else row["name"]: row for row in out["machines"]}


def test_healthy_groups_name_machines_and_keeps_controller_quiet():
    out = _grouped(live=_live(loaded=[
        {"name": "Pair EXL3", "host": "a", "port": 8014, "state": "idle", "span": ["a", "b"]},
        {"name": "Single A", "host": "a", "port": 8015, "state": "UP"},
        {"name": "ComfyUI video", "host": "a", "port": 8188, "state": "idle"},
    ]))
    rows = _by_role(out)
    assert [row["role"] for row in out["machines"]] == ["head", "worker", "controller", "extra"]
    assert rows["head"]["status"] == "fresh 1s"
    assert rows["worker"]["status"] == "fresh 1s"
    assert rows["head"]["stats"]["mem_used_gib"] == 80.4
    assert rows["head"]["archive_mounted"] is True
    assert rows["head"]["models"][1]["presence"] == "local"
    assert rows["controller"]["quiet"] is True
    assert rows["controller"]["status"] == "no report yet"
    assert rows["controller"]["stats"] is None
    assert rows["controller"]["models"] == []
    assert rows["extra"]["quiet"] is True
    assert rows["extra"]["status"] == "no report yet"


def test_stale_host_says_last_heard_and_dims_last_numbers():
    out = _grouped(live=_live(worker={
        "id": "b",
        "fresh": False,
        "age": 12,
        "mem_used_gib": 54,
        "cpu_pct": 11,
    }))
    worker = _by_role(out)["worker"]
    assert worker["report"] == "stale"
    assert worker["status"] == "last heard 12s ago"
    assert worker["dimmed"] is True
    assert worker["stats"]["mem_used_gib"] == 54
    assert _by_role(out)["head"]["status"] == "fresh 1s"
    assert _by_role(out)["head"]["dimmed"] is False


def test_missing_report_says_no_report_yet():
    out = _grouped(live=_live(worker=False))
    worker = _by_role(out)["worker"]
    assert worker["report"] == "missing"
    assert worker["status"] == "no report yet"
    assert worker["stats"] is None
    assert worker["models"] == []
    assert worker["quiet"] is False


def test_model_down_is_a_word_on_that_machine():
    out = _grouped(live=_live(loaded=[
        {"name": "ComfyUI video", "host": "a", "port": 8188, "state": "DOWN"},
    ]))
    head = _by_role(out)["head"]
    comfy = next(row for row in head["models"] if row["id"] == "comfy")
    assert comfy["state"] == "down"
    assert "down" not in {row["state"] for row in _by_role(out)["worker"]["models"]}


def test_pair_model_is_on_both_machines_with_one_load():
    out = _grouped(live=_live(loaded=[
        {"name": "Pair EXL3", "host": "a", "port": 8014, "state": "GEN", "tok_s": 42, "span": ["a", "b"]},
    ]))
    rows = _by_role(out)
    for role in ("head", "worker"):
        pair = next(row for row in rows[role]["models"] if row["id"] == "pair-exl3")
        assert pair["pair"] is True
        assert pair["state"] == "generating"
        assert pair["port"] == 8014
    pair_loads = [row for row in out["loads"] if row["pair"]]
    assert pair_loads == [{
        "id": "pair-exl3",
        "name": "Pair EXL3",
        "port": 8014,
        "pair": True,
        "on": ["head", "worker"],
    }]
    single = next(row for row in out["loads"] if row["id"] == "single-a")
    assert single["on"] == ["head"]
    assert single["pair"] is False


def test_models_keep_distinct_ports():
    out = _grouped()
    ports = {row["id"]: row["port"] for row in _by_role(out)["head"]["models"]}
    assert ports["pair-exl3"] == 8014
    assert ports["single-a"] == 8015
    assert ports["comfy"] == 8188
    assert len(set(ports.values())) == 3


SHARED_PORT = [
    {
        "id": "glm-a",
        "title": "GLM A",
        "port": 8014,
        "placement": "pair",
        "identity": {"served": ["glm-a-id"]},
    },
    {
        "id": "glm-b",
        "title": "GLM B",
        "port": 8014,
        "placement": "pair",
        "identity": {"served": ["glm-b-id"]},
    },
    {
        "id": "qwen",
        "title": "Qwen",
        "port": 8014,
        "placement": "single",
        "seats": {"head": {"port": 8014}},
        "identity": {"served_prefix": "Qwen"},
    },
    {
        "id": "underdog",
        "title": "Underdog",
        "port": 8014,
        "placement": "single",
        "seats": {"head": {"port": 8014}},
        "identity": {"served": ["underdog-id"]},
    },
]


def _states(out, role):
    return {row["id"]: row["state"] for row in _by_role(out)[role]["models"]}


def test_shared_port_marks_only_the_served_model_live():
    out = group_machines(NODES, SHARED_PORT, live=_live(loaded=[{
        "name": "GLM A…",
        "served": "glm-a-id",
        "host": "a",
        "port": 8014,
        "state": "GEN",
        "tok_s": 20,
        "span": ["a", "b"],
    }]))
    head = _states(out, "head")
    worker = _states(out, "worker")
    assert head["glm-a"] == "generating"
    assert worker["glm-a"] == "generating"
    assert head["glm-b"] == "idle"
    assert worker["glm-b"] == "idle"
    assert head["qwen"] == "idle"
    assert head["underdog"] == "idle"
    live_ids = {row["id"] for row in _by_role(out)["head"]["models"] if row["state"] == "generating"}
    assert live_ids == {"glm-a"}
    assert all(row.get("catalog") is not False for row in _by_role(out)["head"]["models"])


def test_served_prefix_matches_without_using_the_port():
    out = group_machines(NODES, SHARED_PORT, live=_live(loaded=[{
        "name": "Qwen…",
        "served": "Qwen3-extra",
        "host": "a",
        "port": 8014,
        "state": "GEN",
        "tok_s": 20,
    }]))
    head = _states(out, "head")
    assert head == {"glm-a": "idle", "glm-b": "idle", "qwen": "generating", "underdog": "idle"}


def test_unlisted_loaded_model_is_read_only_on_its_machine():
    out = group_machines(NODES, SHARED_PORT, live=_live(loaded=[{
        "name": "Eval…",
        "served": "eval-candidate",
        "host": "b",
        "port": 8020,
        "state": "BUSY",
    }]))
    head_ids = [row["id"] for row in _by_role(out)["head"]["models"]]
    assert "unlisted:8020:eval-candidate" not in head_ids
    worker_models = _by_role(out)["worker"]["models"]
    extra = next(row for row in worker_models if row["id"] == "unlisted:8020:eval-candidate")
    assert extra["name"] == "eval-candidate"
    assert extra["port"] == 8020
    assert extra["state"] == "busy"
    assert extra["catalog"] is False
    assert extra["note"] == "Not in catalog"
    assert extra["pair"] is False
    assert all(row["id"] != extra["id"] for row in out["loads"])


def test_model_without_served_identity_still_matches_by_port():
    models = [
        {"id": "plain", "title": "Plain", "port": 8015, "seats": {"head": {"port": 8015}}},
        {
            "id": "named",
            "title": "Named",
            "port": 8015,
            "seats": {"head": {"port": 8015}},
            "identities": {"served": ["named-id"]},
        },
    ]
    out = group_machines(NODES, models, live=_live(loaded=[{
        "name": "Plain…",
        "served": "something-else",
        "host": "a",
        "port": 8015,
        "state": "GEN",
        "tok_s": 7,
    }]))
    states = _states(out, "head")
    assert states["plain"] == "generating"
    assert states["named"] == "idle"
    assert all(row.get("catalog") is not False for row in _by_role(out)["head"]["models"])


def test_grouped_output_has_no_addresses():
    out = _grouped()
    blob = json.dumps(out)
    assert "192.0.2.11" not in blob
    assert "100.64.0.11" not in blob
    assert "planted-path" not in blob
    assert "ips" not in blob
    assert "tailscale" not in blob
    assert "lan" not in blob
