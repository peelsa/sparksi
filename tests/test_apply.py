# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from sparksi.apply import build_plan, commit as apply_commit, is_protected
from sparksi.catalog import load_catalog
from sparksi.lock import Lock
from sparksi.wanted import resolve_spec


def _residents(catalog, live: set[str]) -> list[dict]:
    out = []
    for row in catalog["residents"]:
        r = dict(row)
        r["resident"] = r["id"] in live
        if r.get("role") == "video":
            r["seat"] = "video"
        out.append(r)
    return out


def _dry(verdict="OK"):
    return {
        "verdict": verdict,
        "recipe": {
            "estimate_gib": 103 if verdict != "UNCERTAIN" else None,
            "unit": "declared_vram_gib",
            "source": "model_truth",
            "confidence": "declared" if verdict == "OK" else "unknown",
        },
        "floor": {
            "complete": verdict == "OK",
            "total_gib": 10.0 if verdict == "OK" else None,
            "parts": [
                {
                    "name": "os",
                    "gib": 6.0,
                    "unit": "catalog_min_gib",
                    "instrument": "catalog_min",
                    "confidence": "catalog_min",
                }
            ],
        },
        "margin_gib": 4.0,
        "needed_gib": 117.0 if verdict == "OK" else None,
        "mem_total_gib": 121.0,
        "reasons": ["fixture"],
    }


def test_video_plus_live_video_service_is_noop(video_catalog) -> None:
    cat = load_catalog()
    wanted = resolve_spec(cat, "video")
    plan = build_plan(cat, wanted, _residents(cat, {video_catalog}), _dry("UNCERTAIN"), lock_on=True)
    assert plan["verdict"] == "NOOP"
    assert plan["victims"] == []
    assert plan["starts"] == []


def test_trio_plus_live_video_service_is_refuse(video_catalog) -> None:
    cat = load_catalog()
    wanted = resolve_spec(cat, "coresident-3")
    plan = build_plan(cat, wanted, _residents(cat, {video_catalog}), _dry("NO"), lock_on=True)
    assert plan["verdict"] == "REFUSE"
    assert plan["victims"] == []
    assert "will not start the LLM stack" in " ".join(plan["reasons"])


def test_add_keeps_live_counsel_a() -> None:
    cat = load_catalog()
    wanted = resolve_spec(cat, "brain-a,vision-a")
    plan = build_plan(cat, wanted, _residents(cat, {"counsel-a"}), _dry("OK"), lock_on=False)
    assert plan["verdict"] == "READY"
    assert plan["victims"] == []
    start_ids = {s["id"] for s in plan["starts"]}
    assert start_ids == {"brain-a", "vision-a"}
    keep_ids = {k["id"] for k in plan.get("keep") or []}
    assert "counsel-a" in keep_ids


def test_commit_lock_on_no_mutation(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    cat = load_catalog()
    wanted = resolve_spec(cat, "coresident-3")
    plan = build_plan(cat, wanted, _residents(cat, set()), _dry("OK"), lock_on=True)
    # empty live, no comfy: READY
    assert plan["verdict"] == "READY"
    lk = Lock(tmp_path / "lock")
    lk.acquire()
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 0

    out = apply_commit(cat, plan, _residents(cat, set()), lk, runner=runner, wanted=wanted)
    assert out["verdict"] == "REFUSE"
    assert out.get("executed") is False
    assert calls == []


def test_commit_video_service_up_trio_never_stops(tmp_path: Path, monkeypatch, video_catalog) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    cat = load_catalog()
    wanted = resolve_spec(cat, "trio")
    live = _residents(cat, {video_catalog})
    plan = build_plan(cat, wanted, live, _dry("OK"), lock_on=False)
    assert plan["verdict"] == "REFUSE"
    lk = Lock(tmp_path / "lock")
    lk.release(ttl_s=60)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 0

    out = apply_commit(cat, plan, live, lk, runner=runner, wanted=wanted)
    assert out.get("executed") is False
    assert calls == []
    joined = " ".join(" ".join(c) for c in calls)
    assert "stop" not in joined
    assert "video-mode" not in joined


def test_protected_helper() -> None:
    assert is_protected({"id": "comfy", "port": 8188, "kind": "comfy"})
    assert is_protected({"id": "video-service", "port": 8188, "kind": "video_service"})
    assert is_protected({"id": "comfy", "port": 9999, "kind": "vllm"})
    assert is_protected({"id": "x", "port": 9999, "kind": "comfy"})
    assert is_protected({"id": "x", "port": 9999, "kind": "video_service"})
    assert is_protected({"id": "brain-a", "port": 8188})
    assert not is_protected({"id": "brain-a", "port": 8007, "kind": "vllm"})


def test_ready_start_does_not_stop_comfy(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    cat = load_catalog()
    wanted = resolve_spec(cat, "coresident-3")
    live = _residents(cat, set())
    plan = build_plan(cat, wanted, live, _dry("OK"), lock_on=False)
    assert plan["verdict"] == "READY"
    lk = Lock(tmp_path / "lock")
    lk.release(ttl_s=60)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 0

    out = apply_commit(cat, plan, live, lk, runner=runner, wanted=wanted)
    assert out.get("executed") is True
    assert calls
    assert calls[0] == [
        "sudo",
        "-n",
        "/bin/systemctl",
        "restart",
        "--no-block",
        "vllm-coresident.service",
    ]
    assert not any("stop" in a for a in calls for a in [a])
    assert not any("comfy" in " ".join(a).lower() for a in calls)
    assert not any("video-mode" in " ".join(a) for a in calls)


def test_counsel_a_only_starts_leaf_not_full_stack(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    cat = load_catalog()
    wanted = resolve_spec(cat, "counsel-a")
    live = _residents(cat, set())
    plan = build_plan(cat, wanted, live, _dry("OK"), lock_on=False)
    assert plan["verdict"] == "READY"
    assert {s["id"] for s in plan["starts"]} == {"counsel-a"}
    lk = Lock(tmp_path / "lock")
    lk.release(ttl_s=60)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 0

    out = apply_commit(cat, plan, live, lk, runner=runner, wanted=wanted)
    assert out.get("executed") is True
    assert calls == [
        [
            "sudo",
            "-n",
            "/bin/systemctl",
            "start",
            "--no-block",
            "vllm-counsel-a.service",
        ]
    ]
    assert not any("vllm-coresident" in " ".join(a) for a in calls)
