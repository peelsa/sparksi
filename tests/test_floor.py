# SPDX-License-Identifier: Apache-2.0

from sparksi.dryrun import evaluate
from sparksi.probe import tenant_floor


def _tenants_cat():
    return {
        "host": {"uma": True, "uma_gib": 121, "swap": "off", "dry_run_margin_gib": 4.0},
        "recipes": {
            "coresident-3": {
                "estimate_gib": 103,
                "unit": "declared_vram_gib",
                "source": "model_truth",
                "confidence": "declared",
                "members": ["brain-a"],
            }
        },
        "tenants": {
            "os": {"always": True, "measure": "catalog_min", "min_gib": 6.0},
            "docker": {"always": True, "measure": "rss", "comm": ["dockerd"]},
            "agent": {"always": True, "measure": "rss", "user": "USER-agent", "cmdline_substr": ["agent"]},
            "infra": {"always": True, "measure": "docker_stats", "name_substr": ["infra"]},
            "comfy": {"always": False, "measure": "rss", "port": 8188, "cmdline_substr": ["comfy"]},
        },
    }


def test_missing_agent_and_infra_floor_incomplete_uncertain() -> None:
    cat = _tenants_cat()

    def rss_fn(*, comms=None, cmdline_substr=None, user=None):
        # docker engine present; agent missing
        if comms and "dockerd" in comms:
            return 0.2, [{"pid": 1, "comm": "dockerd", "rss_kib": 200000}], None
        return 0.0, [], None

    def docker_fn(substr):
        return 0.0, [], None  # no infra containers

    def listen_fn(port, host):
        return False

    floor = tenant_floor(cat, rss_fn=rss_fn, docker_fn=docker_fn, listen_fn=listen_fn)
    assert floor["complete"] is False
    reasons = " ".join(floor["reasons"])
    assert "agent" in reasons
    assert "infra" in reasons
    os_part = next(p for p in floor["parts"] if p["name"] == "os")
    assert os_part["confidence"] == "catalog_min"
    assert os_part["unit"] == "catalog_min_gib"
    assert "not measured" in (os_part["detail"] or "")
    comfy = next(p for p in floor["parts"] if p["name"] == "comfy")
    assert comfy["gib"] == 0.0
    assert comfy["confidence"] == "absent"

    mem = {"mem_total_gib": 121, "mem_avail_gib": 80.0, "mem_free_gib": 70.0}
    r = evaluate(cat, "coresident-3", floor=floor, mem=mem)
    assert r["verdict"] == "UNCERTAIN"
    assert r["avail_not_used_for_verdict"] is True


def test_complete_floor_with_hits() -> None:
    cat = _tenants_cat()

    def rss_fn(*, comms=None, cmdline_substr=None, user=None):
        if comms and "dockerd" in comms:
            return 0.2, [{"pid": 1, "comm": "dockerd"}], None
        if user == "USER-agent":
            return 1.5, [{"pid": 2, "comm": "agent"}], None
        return 0.0, [], None

    def docker_fn(substr):
        return 0.3, [{"name": "infra-relay", "gib": 0.3}], None

    def listen_fn(port, host):
        return False

    floor = tenant_floor(cat, rss_fn=rss_fn, docker_fn=docker_fn, listen_fn=listen_fn)
    assert floor["complete"] is True
    assert floor["total_gib"] == 8.0  # 6 + 0.2 + 1.5 + 0.3
