# SPDX-License-Identifier: Apache-2.0

from sparksi.catalog import CatalogError
from sparksi.dryrun import evaluate
import pytest


def _cat(**recipe_extra):
    rec = {
        "estimate_gib": 103,
        "source": "model_truth",
        "confidence": "declared",
        "members": ["a"],
    }
    rec.update(recipe_extra)
    return {
        "host": {"uma": True, "uma_gib": 121, "swap": "off", "dry_run_margin_gib": 4.0},
        "recipes": {
            "coresident-3": rec,
            "video": {
                "aliases": ["comfy-h3"],
                "estimate_gib": None,
                "source": "none",
                "confidence": "unknown",
                "members": ["comfy"],
                "notes": "no ceiling",
            },
        },
    }


def test_free_only_is_uncertain() -> None:
    floor = {"complete": False, "total_gib": None, "reasons": ["not measured"], "parts": []}
    mem = {"mem_total_gib": 121, "mem_avail_gib": 80.0, "mem_free_gib": 70.0}
    r = evaluate(_cat(), "coresident-3", floor=floor, mem=mem)
    assert r["verdict"] == "UNCERTAIN"
    assert r["avail_not_used_for_verdict"] is True


def test_overflow_is_no() -> None:
    floor = {"complete": True, "total_gib": 20.0, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121, "mem_avail_gib": 100.0, "mem_free_gib": 90.0}
    r = evaluate(_cat(estimate_gib=110), "coresident-3", floor=floor, mem=mem)
    # 20 + 110 + 4 = 134 > 121
    assert r["verdict"] == "NO"
    assert r["needed_gib"] == 134.0


def test_complete_fit_is_ok() -> None:
    floor = {"complete": True, "total_gib": 10.0, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121, "mem_avail_gib": 1.0, "mem_free_gib": 0.5}
    r = evaluate(_cat(), "coresident-3", floor=floor, mem=mem)
    # 10 + 103 + 4 = 117 <= 121
    assert r["verdict"] == "OK"
    assert r["mem_avail_gib"] == 1.0


def test_video_unknown_is_uncertain() -> None:
    floor = {"complete": True, "total_gib": 8.0, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121, "mem_avail_gib": 90.0, "mem_free_gib": 80.0}
    r = evaluate(_cat(), "video", floor=floor, mem=mem)
    assert r["verdict"] == "UNCERTAIN"
    assert r["recipe"]["estimate_gib"] is None


def test_unknown_set() -> None:
    with pytest.raises(CatalogError):
        evaluate(_cat(), "nope", floor={"complete": True, "total_gib": 1}, mem={"mem_total_gib": 121})


def test_non_uma_refuses_add() -> None:
    cat = _cat()
    cat["host"] = {"uma": False, "uma_gib": 121, "dry_run_margin_gib": 4.0}
    floor = {"complete": True, "total_gib": 10.0, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121, "mem_avail_gib": 80.0, "mem_free_gib": 70.0}
    r = evaluate(cat, "coresident-3", floor=floor, mem=mem)
    assert r["verdict"] == "UNCERTAIN"
    assert r["needed_gib"] is None
    assert r["comparable"] is False


def _uma_cat():
    return {
        "host": {"uma": True, "uma_gib": 121, "swap": "off", "dry_run_margin_gib": 4.0},
        "residents": [
            {
                "id": "counsel-a",
                "kind": "vllm",
                "role": "counsel",
                "gpu_memory_utilization": 0.28,
            },
            {
                "id": "sandbox-big",
                "kind": "vllm",
                "role": "sandbox",
                "gpu_memory_utilization": 0.72,
            },
        ],
        "recipes": {
            "coresident-3": {
                "estimate_gib": 103,
                "source": "model_truth",
                "confidence": "declared",
                "members": ["a"],
            },
            "sandbox-big": {
                "estimate_gib": 72,
                "source": "catalog",
                "confidence": "declared",
                "members": ["sandbox-big"],
            },
        },
    }


def test_empty_counsel_a_fits() -> None:
    floor = {"complete": True, "total_gib": 8.4, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121.689, "mem_avail_gib": 114.0, "mem_free_gib": 90.0}
    r = evaluate(_uma_cat(), "counsel-a", floor=floor, mem=mem, residents=[])
    # 8.4 + 0 live + 0.28*121 + 4 = 46.28
    assert r["verdict"] == "OK"
    assert r["needed_gib"] < 50


def test_empty_big_plus_counsel_a_does_not_stack_to_121() -> None:
    floor = {"complete": True, "total_gib": 8.4, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121.689, "mem_avail_gib": 114.0, "mem_free_gib": 90.0}
    r = evaluate(_uma_cat(), "counsel-a,sandbox-big", floor=floor, mem=mem, residents=[])
    # starts: 33.88 + 72 = 105.88; +8.4 +4 = 118.28 <= 121.689
    assert r["recipe"]["start_cost_gib"] == 105.88
    assert r["verdict"] == "OK"
    assert r["needed_gib"] < 121.689
    assert r["needed_gib"] != 133.405


def test_live_gpu_plus_big_start() -> None:
    floor = {"complete": True, "total_gib": 8.4, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121.689, "mem_avail_gib": 15.0, "mem_free_gib": 4.0}
    live = [{"id": "counsel-a", "resident": True, "gpu_gib": 30.0, "role": "counsel"}]
    r = evaluate(_uma_cat(), "sandbox-big", floor=floor, mem=mem, residents=live)
    # 8.4 + 30 + 72 + 4 = 114.4
    assert r["verdict"] == "OK"
    assert r["recipe"]["live_gpu_gib"] == 30.0
    assert abs(r["needed_gib"] - 114.4) < 0.01


def test_avail_still_ignored() -> None:
    floor = {"complete": True, "total_gib": 8.4, "reasons": [], "parts": []}
    mem = {"mem_total_gib": 121.689, "mem_avail_gib": 1.0, "mem_free_gib": 0.1}
    r = evaluate(_uma_cat(), "counsel-a", floor=floor, mem=mem, residents=[])
    assert r["verdict"] == "OK"
    assert r["avail_not_used_for_verdict"] is True
