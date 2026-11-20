# SPDX-License-Identifier: Apache-2.0

"""An engine pair started outside Sparksi: same container name on both Sparks.

The shape: no TP, no -head / -worker suffix, the port only in ``--port``, the
head answering on its own tailnet address and the worker headless.
"""

from __future__ import annotations

from sparksi import agent, announce
from sparksi.config import CFG
from sparksi.monitor import models
from sparksi.monitor.__main__ import _loaded_row
from sparksi.monitor.models import discover_loaded
from sparksi.serving_state import reconcile

NAME = "example-engine"
SERVED = "example-served"


def _head(port: int = 8014, tp=None) -> dict:
    return {
        "fresh": True,
        "http": {port: {"served": SERVED, "metrics": {"num_requests_running": 0}, "bind": "specific"}},
        "engines": [{"name": NAME, "status": "Up", "port": port, "tp": tp}],
        "containers": [{"name": NAME, "status": "Up"}],
        "truth": "",
    }


def _worker(port: int = 8014, fresh: bool = True, http=None, name: str = NAME) -> dict:
    return {
        "fresh": fresh,
        "http": http or {},
        "engines": [{"name": name, "status": "Up", "port": port, "tp": None}],
        "containers": [{"name": name, "status": "Up"}],
    }


def _hosts(worker_fresh: bool = True) -> list[dict]:
    return [{"id": CFG.head.id, "fresh": True}, {"id": CFG.worker.id, "fresh": worker_fresh}]


def test_same_name_same_port_on_both_sparks_is_one_pair() -> None:
    raw = discover_loaded(_head(), _worker())
    assert len(raw) == 1, raw
    rec = raw[0]
    assert rec["span"] == [CFG.head.id, CFG.worker.id]
    assert rec["worker_host"] == CFG.worker.id
    assert rec["worker_up"] is True
    assert rec["worker_name"] == NAME
    assert rec["container"] == NAME
    assert rec["container_source"] == "engine"
    row = _loaded_row(rec, 0, {})
    rows = reconcile([row], [], hosts=_hosts(), head_id=CFG.head.id, worker_id=CFG.worker.id)
    by_role = {item["role"]: item for item in rows}
    assert by_role["head"]["state"] == "answering"
    assert by_role["worker"]["state"] == "answering"
    assert by_role["worker"]["text"] == "Answering · port 8014"
    assert all(item["run_id"] == "" for item in rows)
    assert all(item["reason"] == "This process is up. No active run owns it." for item in rows)


def test_worker_that_answers_http_on_the_port_is_its_own_model() -> None:
    answering = {8014: {"served": "other-served", "metrics": None}}
    raw = discover_loaded(_head(), _worker(http=answering))
    assert len(raw) == 2
    assert not any(rec.get("span") for rec in raw)
    assert {rec["host"] for rec in raw} == {CFG.head.id, CFG.worker.id}


def test_stale_peer_that_last_answered_http_is_not_a_twin() -> None:
    answering = {8014: {"served": "other-served", "metrics": None}}
    raw = discover_loaded(_head(), _worker(fresh=False, http=answering))
    assert len(raw) == 1
    assert not raw[0].get("span")


def test_stale_peer_keeps_the_pair_with_worker_unknown() -> None:
    raw = discover_loaded(_head(), _worker(fresh=False))
    assert len(raw) == 1
    rec = raw[0]
    assert rec["span"] == [CFG.head.id, CFG.worker.id]
    assert rec["worker_up"] is None
    rows = reconcile([_loaded_row(rec, 0, {})], [], hosts=_hosts(worker_fresh=False),
                     head_id=CFG.head.id, worker_id=CFG.worker.id)
    # Same as a suffixed pair with a stale peer: the worker side is not confirmed.
    assert [item["state"] for item in rows] == ["split", "split"]
    assert rows[1]["text"] == "Split · worker not confirmed · port 8014"
    assert all(item["run_id"] == "" for item in rows)


def test_mismatched_ports_do_not_pair() -> None:
    raw = discover_loaded(_head(port=8014), _worker(port=8015))
    assert len(raw) == 1
    assert not raw[0].get("span")
    assert raw[0].get("worker_host") is None


def test_different_names_do_not_pair() -> None:
    raw = discover_loaded(_head(), _worker(name="something-else"))
    assert not raw[0].get("span")


def test_two_single_spark_sparksi_runs_with_one_name_do_not_pair() -> None:
    """Sparksi names a single-Spark run sparksi-<model>[-<port>]: the same on both Sparks."""
    head = _head()
    head["engines"][0]["name"] = "sparksi-example-8014"
    raw = discover_loaded(head, _worker(name="sparksi-example-8014"))
    assert len(raw) == 1
    assert not raw[0].get("span")


def test_peer_without_engines_does_not_pair() -> None:
    worker = _worker()
    worker["engines"] = []
    raw = discover_loaded(_head(), worker)
    assert not raw[0].get("span")


def test_headless_head_does_not_pair_by_name_alone() -> None:
    head = _head()
    head["http"] = {}
    assert discover_loaded(head, _worker()) == []


def test_one_spark_site_never_spans(monkeypatch) -> None:
    monkeypatch.setattr(models, "PEER_HOST", "")
    raw = discover_loaded(_head(), _worker())
    assert len(raw) == 1
    assert not raw[0].get("span")


def test_third_and_fourth_spark_stay_single_rows() -> None:
    extra = {
        "fresh": True,
        "http": {8014: {"served": SERVED, "metrics": None}},
        "engines": [{"name": NAME, "status": "Up", "port": 8014, "tp": None}],
        "containers": [{"name": NAME, "status": "Up"}],
    }
    raw = discover_loaded(_head(), _worker(), extras=[("node3", extra), ("node4", dict(extra))])
    spans = [rec for rec in raw if rec.get("span")]
    assert len(spans) == 1
    assert spans[0]["span"] == [CFG.head.id, CFG.worker.id]
    singles = [rec for rec in raw if rec.get("host") in ("node3", "node4")]
    assert len(singles) == 2
    assert not any(rec.get("span") for rec in singles)


def test_suffixed_pair_still_pairs_as_before() -> None:
    head = _head()
    head["engines"] = [{"name": "example-engine-head", "status": "Up", "port": 8014, "tp": None}]
    worker = _worker(name="example-engine-worker")
    worker["engines"] = []
    raw = discover_loaded(head, worker)
    assert raw[0]["span"] == [CFG.head.id, CFG.worker.id]
    assert raw[0]["worker_name"] == "example-engine-worker"


# ---------- tensor-parallel size from a container's Env and Cmd ----------


def _blob(env: list[str], cmd: list[str]) -> str:
    return "\n".join([*env, "---CMD---", *cmd]) + "\n"


def test_tp_flag_forms() -> None:
    assert announce._port_from_blob(_blob([], ["serve", "--port", "8014", "--tensor-parallel-size", "2"])) == (8014, 2)
    assert announce._port_from_blob(_blob([], ["serve", "--port=8014", "--tensor-parallel-size=2"])) == (8014, 2)
    assert announce._port_from_blob(_blob([], ["--tp=2", "--port", "8014"])) == (8014, 2)
    assert announce._port_from_blob(_blob([], ["--tp", "2"])) == (0, 2)
    assert announce._port_from_blob(_blob(["TP=2"], [])) == (0, 2)
    assert announce._port_from_blob(_blob(["TENSOR_PARALLEL_SIZE=2"], [])) == (0, 2)


def test_world_size_counts_only_without_a_tp_setting() -> None:
    assert announce._port_from_blob(_blob(["WORLD_SIZE=2"], ["--port", "8014"])) == (8014, 2)
    assert announce._port_from_blob(_blob(["WORLD_SIZE=4", "TP=2"], [])) == (0, 2)
    assert announce._port_from_blob(_blob(["WORLD_SIZE=x"], [])) == (0, 0)


def test_outside_launch_shape_reports_port_without_tp() -> None:
    blob = _blob(
        ["TF_EXAMPLE_WORLD=2", "HOME=/state/home"],
        ["serve", "/model", "--name", SERVED, "--host", "100.64.0.11", "--port", "8014", "--context", "1048576"],
    )
    assert announce._port_from_blob(blob) == (8014, 0)


# ---------- agent probe: a server bound only to the node's own tailnet address ----------


def _probe_with(monkeypatch, open_on: set[tuple[str, int]], tailnet, off_limits=(8188,)) -> tuple[dict, list]:
    dialed: list = []

    def port_open(port, host="127.0.0.1"):
        dialed.append((host, int(port)))
        return (host, int(port)) in open_on

    def http(port, path="/v1/models", host="127.0.0.1"):
        dialed.append((host, int(port), path))
        return {"path": path, "status": 200, "models": [SERVED]}

    monkeypatch.setattr(agent, "_port_open", port_open)
    monkeypatch.setattr(agent, "_http_identity", http)
    monkeypatch.setattr(agent, "_running_containers", lambda: [NAME])
    monkeypatch.setattr(agent, "_ips", lambda: {"lan": "192.0.2.10", "tailscale": tailnet, "direct": ["198.18.0.10"]})
    monkeypatch.setattr(agent, "_gpu", lambda: {})
    monkeypatch.setattr(agent, "_meminfo", lambda: {})
    monkeypatch.setitem(agent._CFG, "off_limits_ports", list(off_limits))
    return dialed, []


def test_probe_finds_a_server_bound_to_its_own_tailnet_address(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("100.64.0.11", 8014)}, "100.64.0.11")
    out = agent._probe({"op": "probe", "ports": [8014, 8015], "paths": [], "http_paths": {"8014": "/v1/models"}})
    assert out["listening"] == {"8014": True, "8015": False}
    assert out["http"]["8014"]["models"] == [SERVED]
    assert ("100.64.0.11", 8014, "/v1/models") in dialed
    assert out["ips"]["tailscale"] == "100.64.0.11"
    hosts = {item[0] for item in dialed}
    assert hosts <= {"127.0.0.1", "100.64.0.11"}


def test_probe_never_tries_an_off_limits_port_on_the_tailnet(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("100.64.0.11", 8188)}, "100.64.0.11")
    out = agent._probe({"op": "probe", "ports": [8188], "paths": []})
    assert out["listening"] == {"8188": False}
    assert ("100.64.0.11", 8188) not in dialed
    assert out["http"] == {}


def test_probe_without_a_tailnet_address_stays_on_loopback(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("100.64.0.11", 8014)}, None)
    out = agent._probe({"op": "probe", "ports": [8014], "paths": []})
    assert out["listening"] == {"8014": False}
    assert {item[0] for item in dialed} == {"127.0.0.1"}


def test_probe_ignores_a_non_tailnet_100_address(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("100.1.2.3", 8014)}, "100.1.2.3")
    out = agent._probe({"op": "probe", "ports": [8014], "paths": []})
    assert out["listening"] == {"8014": False}
    assert {item[0] for item in dialed} == {"127.0.0.1"}


def test_probe_asked_for_a_tailnet_address_dials_only_that(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("100.64.0.11", 8014)}, "100.64.0.12")
    out = agent._probe({"op": "probe", "ports": [8014], "paths": [], "listen_host": "100.64.0.11"})
    assert out["listening"] == {"8014": True}
    assert {item[0] for item in dialed} == {"100.64.0.11"}


def test_probe_prefers_loopback_when_it_answers(monkeypatch) -> None:
    dialed, _ = _probe_with(monkeypatch, {("127.0.0.1", 8014), ("100.64.0.11", 8014)}, "100.64.0.11")
    out = agent._probe({"op": "probe", "ports": [8014], "paths": []})
    assert out["listening"] == {"8014": True}
    assert {item[0] for item in dialed} == {"127.0.0.1"}
