# SPDX-License-Identifier: Apache-2.0

import re

import pytest

from sparksi.config import CFG, Config, SingleSparkError, MAX_NODES
from sparksi.models import ModelCard, derive_placements


def make_raw(n):
    return {
        "nodes": [
            {"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}"}
            for i in range(1, n + 1)
        ]
    }


def make_card(n, weights_gib, hosts=None, engines=("vllm",)):
    if hosts is None:
        hosts = [f"spark-{i}" for i in range(1, n + 1)]
    return ModelCard(
        id="m",
        title="M",
        format="bf16",
        weights_gib=weights_gib,
        engines=list(engines),
        files={h: "/w" for h in hosts},
        presence={h: True for h in hosts},
    )


ROLES = ("head", "worker", "node3", "node4")


def budgets(n):
    return {r: 111 for r in ROLES[:n]}


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_config_shapes(n):
    cfg = Config(make_raw(n))
    assert cfg.roles == ROLES[:n]
    assert n <= MAX_NODES
    assert cfg.has_worker == (n > 1)
    pub = cfg.public()
    assert len(pub["nodes"]) == n
    assert [nd["role"] for nd in pub["nodes"]] == list(cfg.roles)
    assert ("worker" in pub) == (n > 1)


@pytest.mark.parametrize("n,msg", [(0, "at least one [[nodes]]"), (5, "1 to 4 Sparks")])
def test_config_count_errors(n, msg):
    with pytest.raises(ValueError, match=re.escape(msg)):
        Config(make_raw(n))


def test_duplicate_id():
    raw = make_raw(2)
    raw["nodes"][1]["id"] = "s1"
    with pytest.raises(ValueError, match="used by an earlier Spark"):
        Config(raw)


def test_single_node_worker_error():
    cfg = Config(make_raw(1))
    with pytest.raises(SingleSparkError):
        cfg.worker


@pytest.fixture
def site():
    saved = (CFG.raw, CFG.source)

    def _site(n):
        CFG.__init__(make_raw(n))
        return CFG

    yield _site

    CFG.__init__(saved[0], source=saved[1])


def test_derive_one_node(site):
    site(1)
    out = derive_placements(make_card(1, 40), budgets(1))
    assert out["both"]["allowed"] is False
    assert out["both"]["reason"] == "needs 2 Sparks; this site has 1 Spark"


def test_derive_two_nodes(site):
    site(2)
    out = derive_placements(make_card(2, 40), budgets(2))
    assert set(out) == {"head", "worker", "both"}


def test_derive_three_nodes_big_model(site):
    site(3)
    card = make_card(3, 124)
    out = derive_placements(card, budgets(3))
    assert out["worker+node3"]["allowed"] is True
    assert out["head"]["allowed"] is False


def test_derive_four_nodes(site):
    site(4)
    big = derive_placements(make_card(4, 124), budgets(4))
    assert "node3+node4" in big
    small = derive_placements(
        make_card(4, 40, hosts=["spark-4"]), budgets(4)
    )
    assert small["node4"]["allowed"] is True


def test_render_pair_on_other_sparks(site):
    from sparksi.recipes.render import render

    site(3)
    card = make_card(3, 124)
    one = render(card, engine="vllm", placement="node3", port=8014, fields={})
    assert [(n["role"], n["host"]) for n in one["nodes"]] == [("node3", "spark-3")]
    pair = render(card, engine="vllm", placement="worker+node3", port=8014, fields={})
    assert pair["placement"] == "worker+node3"
    # The second Spark of the pair launches first; the first one leads and serves the port.
    assert pair["order"] == ["node3", "worker"]
    assert {n["role"]: n["host"] for n in pair["nodes"]} == {"node3": "spark-3", "worker": "spark-2"}


def test_render_refuses_missing_sparks(site):
    from sparksi.recipes.render import render

    site(1)
    with pytest.raises(Exception, match="needs 2 Sparks; this site has 1 Spark"):
        render(make_card(1, 124), engine="vllm", placement="both", port=8014, fields={})


def test_pinned_pair_on_one_spark(site):
    from sparksi.recipes.pinned import validate_pinned

    site(1)
    role = {"image": "nvcr.io/nvidia/vllm:latest", "env": {}, "mounts": [], "network": "host",
            "container": "pinned-x", "args": ["vllm", "serve", "/w", "--port", "8014"]}
    recipe = {"kind": "pinned", "engine": "vllm", "placement": "both", "port": 8014,
              "rendezvous_ports": [], "roles": {"head": dict(role), "worker": dict(role)}}
    assert any("needs 2 Sparks; this site has 1 Spark" in c for c in validate_pinned(recipe))
    recipe.update(placement="head", roles={"head": dict(role)})
    assert not any("this site has 1 Spark" in c for c in validate_pinned(recipe))


def test_run_lead_role():
    from sparksi.recipes.runs import _lead_role, _multi, _side_stopped_note

    assert _lead_role({"placement": "both"}) == "head"
    assert _lead_role({"placement": "worker"}) == "head"
    assert _lead_role({"placement": "node3"}) == "node3"
    assert _lead_role({"placement": "worker+node4"}) == "worker"
    assert _multi({"placement": "both"}) and _multi({"placement": "node3+node4"})
    assert not _multi({"placement": "head"}) and not _multi({"placement": "node4"})
    assert _side_stopped_note("node3") == "Spark 3 side stopped"


def test_network_change_any_count():
    from sparksi.netconfig import change, layout

    raw = make_raw(2)
    nodes, _eps, note = change(raw, "remove", "", {"name": "spark-2"})
    assert [n["id"] for n in nodes] == ["s1"] and "1 Spark left" in note
    one = {"nodes": nodes}
    assert "worker" not in layout(one, "ctl")
    with pytest.raises(ValueError, match="head runs Sparksi"):
        change(one, "remove", "", {"name": "spark-1"})
    four = make_raw(4)
    shown = layout(four, "ctl")
    assert [n["role"] for n in shown["nodes"]] == list(ROLES)
    nodes, _eps, _note = change(four, "remove", "", {"name": "spark-4"})
    assert [n["id"] for n in nodes] == ["s1", "s2", "s3"]


def test_serving_rows_follow_the_spark_count():
    from sparksi.serving_state import reconcile

    assert [r.get("role") for r in reconcile([], [], head_id="s1", worker_id="")] == ["head"]
    assert [r.get("role") for r in reconcile([], [], head_id="s1", worker_id="s2")] == ["head", "worker"]
    rows = reconcile([], [], head_id="s1", worker_id="s2", extra_ids={"node3": "s3", "node4": "s4"})
    assert [r.get("role") for r in rows] == ["head", "worker", "node3", "node4"]


def test_remove_middle_spark_is_refused():
    from sparksi.netconfig import change
    raw = make_raw(4)
    with pytest.raises(ValueError, match="not the last Spark; remove Spark 4 first"):
        change(raw, "remove", "", {"name": "spark-2"})
    nodes, _eps, _note = change(raw, "remove", "", {"name": "spark-4"})
    assert [n["id"] for n in nodes] == ["s1", "s2", "s3"]


def test_row_hosts_follow_the_placement(site):
    from sparksi.models import _row_hosts
    site(3)
    assert _row_hosts({"placement": "pair"}) == ["spark-1", "spark-2"]
    assert _row_hosts({"placement": "worker+node3"}) == ["spark-2", "spark-3"]
    assert _row_hosts({"placement": "pair", "local": {"spark-3": "/w", "spark-1": "/w"}}) == ["spark-3", "spark-1"]
    site(1)
    assert _row_hosts({"placement": "pair"}) == ["spark-1"]


def test_pair_reason_names_the_sparks(site):
    site(3)
    card = make_card(3, 124, hosts=["spark-1", "spark-2"])
    out = derive_placements(card, budgets(3))
    assert out["worker+node3"]["reason"] == "weights are not on Spark 3"
    assert out["both"]["allowed"] is True
