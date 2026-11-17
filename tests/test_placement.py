# SPDX-License-Identifier: Apache-2.0

import sparksi.placement as P

ONE = ("head",)
TWO = ("head", "worker")
THREE = ("head", "worker", "node3")
FOUR = ("head", "worker", "node3", "node4")


def test_roles_of():
    assert P.roles_of("both", TWO) == ("head", "worker")
    assert P.roles_of("pair", TWO) == ()
    assert P.roles_of("head", TWO) == ("head",)
    assert P.roles_of("node3", THREE) == ("node3",)
    assert P.roles_of("worker+node3", THREE) == ("worker", "node3")
    assert P.roles_of("node3+worker", THREE) == ("worker", "node3")
    assert P.roles_of("both", ONE) == ()
    assert P.roles_of("node4", TWO) == ()
    assert P.roles_of("head+head", TWO) == ()
    assert P.roles_of("", TWO) == ()
    assert P.roles_of("head", ()) == ()


def test_placement_for_round_trip():
    assert P.placement_for(("head", "worker"), TWO) == "both"
    assert P.placement_for(("head",), TWO) == "head"
    assert P.placement_for(("worker", "node3"), THREE) == "worker+node3"
    assert P.placement_for((), TWO) == ""
    assert P.placement_for(("node4",), TWO) == ""
    for roles in (TWO, THREE, FOUR):
        for placement in P.all_placements(roles, max_size=len(roles)):
            assert P.roles_of(placement, roles) and P.placement_for(P.roles_of(placement, roles), roles) == placement


def test_multis():
    assert P.multis(TWO) == ["both"]
    assert len(P.multis(THREE)) == 3
    assert P.multis(THREE)[0] == "both"
    assert len(P.multis(FOUR, max_size=2)) == 6
    assert len(P.multis(FOUR, max_size=4)) == 11
    assert P.multis(FOUR, max_size=4)[:3] == ["both", "head+node3", "head+node4"]


def test_leader():
    assert P.leader("both", TWO) == "head"
    assert P.leader("worker+node3", THREE) == "worker"
    assert P.leader("node3", THREE) == "node3"
    assert P.leader("nope", TWO) == ""


def test_engine_view_legacy():
    assert P.engine_view("head", TWO) == ("head", {"head": "head"})
    assert P.engine_view("worker", TWO) == ("worker", {"worker": "worker"})
    assert P.engine_view("both", TWO) == ("both", {"head": "head", "worker": "worker"})


def test_engine_view_new():
    assert P.engine_view("node3", THREE) == ("head", {"head": "node3"})
    assert P.engine_view("worker+node3", THREE) == ("both", {"head": "worker", "worker": "node3"})
    assert P.engine_view("node4", TWO) == ("", {})
    assert P.engine_view("head+worker+node3", THREE) == ("", {})


def test_needs_text():
    assert P.needs_text("both", ONE) == "needs 2 Sparks; this site has 1 Spark"
    assert P.needs_text("worker", ONE) == "needs the worker Spark; this site has 1 Spark"
    assert P.needs_text("head", ONE) == ""
    assert P.needs_text("both", TWO) == ""
