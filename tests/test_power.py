# SPDX-License-Identifier: Apache-2.0

"""Fleet power estimate and old announce packets."""

from sparksi.announce import to_peer
from sparksi.power import fleet_power, node_watts


def test_three_field_gpu_packet_has_no_watts() -> None:
    peer = to_peer({"h": "a", "mem": [1, 1], "cpu": [10, 1000], "gpu": [4, 40, 800]})
    assert peer["gpu"] == [4, 40, 800]
    assert peer["power_w"] is None


def test_four_field_gpu_packet_records_watts() -> None:
    peer = to_peer({"h": "a", "mem": [1, 1], "cpu": [10, 1000], "gpu": [4, 40, 800, 90.5]})
    assert peer["gpu"] == [4, 40, 800]
    assert peer["power_w"] == 90.5


def test_node_estimate_matches_the_sparkdash_formula() -> None:
    assert node_watts(100, 0) == 100 + 5.2 + 23
    assert node_watts(100, 100) == 100 + 65 + 23
    assert node_watts(200, 100) == 240
    assert node_watts(None, 50) is None
    assert node_watts(80, None) is None


def test_stale_peer_drops_out_and_marks_partial() -> None:
    hosts = [
        {"id": "a", "fresh": True, "power_w": 100, "cpu_pct": 0},
        {"id": "b", "fresh": False, "power_w": 100, "cpu_pct": 0},
    ]
    body = fleet_power(hosts, ["a", "b"])
    assert body["partial"] is True
    assert body["watts"] == round(100 + 5.2 + 23, 1)
    assert body["machines"][1]["watts"] is None


def test_full_coverage_is_not_partial() -> None:
    hosts = [
        {"id": "a", "fresh": True, "power_w": 10, "cpu_pct": 0},
        {"id": "b", "fresh": True, "power_w": 10, "cpu_pct": 0},
    ]
    body = fleet_power(hosts, ["a", "b"])
    assert body["partial"] is False
    assert body["estimate"] is True
    assert body["watts"] == round(2 * (10 + 5.2 + 23), 1)
