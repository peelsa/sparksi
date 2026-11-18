# SPDX-License-Identifier: Apache-2.0

"""Two-Spark library, copy/delete gates, and the remote agent."""

import hashlib
import json
import shutil
import tarfile

import pytest
from pathlib import Path

from sparksi.agent import main as agent_main
from sparksi.catalog import load_catalog
from sparksi.launch import plan_drop, plan_run, plan_stage, plan_stop, plan_switch
from sparksi.library import fleet, row_by_id
from sparksi.remote import LocalRunner
from sparksi.serve import allowed_hosts

ROOT = Path(__file__).resolve().parents[1]


def _probe(host, req, paths_exist=None, listening=None, releases=None, free_gib=2000, mount=True,
           http=None, containers=None):
    paths_exist = paths_exist or {}
    listening = listening or {}
    paths = {}
    for p in req.get("paths") or []:
        paths[p] = {"exists": bool(paths_exist.get(p)), "nonempty": bool(paths_exist.get(p))}
    out = {
        "ok": True,
        "disk_free_gib": free_gib,
        "disk_total_gib": 3700,
        "mount_archive": mount,
        "listening": {str(k): bool(v) for k, v in listening.items()},
        "http": http or {},
        "containers": containers,
        "paths": paths,
        "gpu": {"util_pct": 3, "temp_f": 100},
        "mem_total_gib": 121,
        "mem_available_gib": 20,
    }
    if req.get("scan_releases"):
        out["releases"] = releases or []
    if req.get("scan_flat"):
        out["flat"] = {"ok": True, "full": 120, "index": True}
    return out


def _runner(**kw):
    def invoke(host, req, timeout=0):
        return _probe(host, req, **kw)
    return LocalRunner(invoke)


def test_fleet_merges_release_and_keeps_pair_fast():
    cat = load_catalog()
    releases = [
        {
            "slug": "release-a-nvfp4",
            "title": "Release-A",
            "bytes": 24713185280,
            "sha256": "abc",
            "tar": "/mnt/archive/models/releases/release-a/file.tar",
            "source": "/home/USER/services/vllm-release-a/models",
        }
    ]
    data = fleet(cat, _runner(releases=releases))
    ids = {r["id"] for r in data["library"]}
    assert "pair-fast" in ids
    assert "release-a-nvfp4" in ids
    assert "pair-exl3" in ids
    orn = row_by_id(data["library"], "release-a-nvfp4")
    assert orn["launcher"] == "no"
    assert orn["archive_state"] == "archived"
    assert len(data["hosts"]) == 2


def test_tensorfold_monitor_row_stops_tensorfold_not_instanttensor():
    import json
    import subprocess

    from sparksi.monitor.__main__ import snapshot

    class Peer:
        def __init__(self, data):
            self._data = data

        def snapshot(self):
            return self._data, 0.4

    head = {
        "mem": (119.0, 40.0),
        "cpu_pct": 1.0,
        "cpu_mhz": 1800.0,
        "gpu": [1.0, 40.0, 800.0],
        "zones": [],
        "http": {8015: {"served": "Pair-EXL3", "metrics": {"num_requests_running": 0}}},
        "containers": [{"name": "pair-tf", "status": "Up"}],
        "engines": [{"name": "pair-tf", "port": 8015, "tp": 2}],
        "comfy": {},
        "truth": "",
    }
    worker = dict(head)
    worker["http"] = {}
    worker["engines"] = []
    row = snapshot(Peer(head), Peer(worker), {})["loaded"][0]
    assert row["container"] == "pair-tf"
    html = (ROOT / "src" / "sparksi" / "static" / "index.html").read_text()
    start = html.index("function libraryIdFor")
    end = html.index("function stopLoaded", start)
    rows_js = "var libraryRows = " + json.dumps(load_catalog()["library"]) + ";\n"
    script = rows_js + html[start:end]
    ident = subprocess.check_output(
        ["node", "-e", script + "\nprocess.stdout.write(libraryIdFor(" + json.dumps(row) + ") || '')"],
        text=True,
    )
    cat = load_catalog()
    chosen = row_by_id(cat["library"], ident)
    stopped = plan_stop(chosen, lock_on=False)
    assert stopped["verdict"] == "READY", stopped["reasons"]
    assert stopped["argv"][-1].endswith("pair-tf-harness/stop.sh")
    bare = dict(row)
    bare["container"] = None
    instant = subprocess.check_output(
        ["node", "-e", script + "\nprocess.stdout.write(libraryIdFor(" + json.dumps(bare) + ") || '')"],
        text=True,
    )
    assert instant == "pair-exl3"


def test_pair_tensorfold_sits_beside_instanttensor():
    cat = load_catalog()
    assert "pair-tf" in cat["exclusive_groups"]["big-pair"]
    assert "pair-exl3" in cat["exclusive_groups"]["big-pair"]
    row = row_by_id(cat["library"], "pair-tf")
    assert row["port"] == 8014
    assert row["placement"] == "pair"
    assert row["listen_host"] == "spark-a"
    assert row["exclusive"] == "big-pair"
    weight = "/home/USER/.cache/huggingface/hub/models--example--pair-tf-4bpw"
    assert row["local"]["spark-a"] == weight
    assert row["local"]["spark-b"] == weight
    assert row["archive_tar"] is None
    assert "archive_flat" not in row
    root = "/home/USER/projects/pair-tf-harness"
    assert row["launch"]["start"] == [f"{root}/start.sh"]
    assert row["launch"]["stop"] == [f"{root}/stop.sh"]
    instant = row_by_id(cat["library"], "pair-exl3")
    assert instant["port"] == 8014
    assert "start.sh" in instant["launch"]["start"][0]
    data = fleet(cat, _runner())
    live = row_by_id(data["library"], "pair-tf")
    assert live["on"]["spark-a"] == "absent"
    assert live["on"]["spark-b"] == "absent"
    refused = plan_run(cat, live, data["probes"], lock_on=False)
    assert refused["verdict"] == "REFUSE"
    assert any("absent" in r for r in refused["reasons"])
    loaded = plan_switch(cat, live, data["library"], lock_on=False, host="pair")
    assert loaded["verdict"] == "READY", loaded["reasons"]
    assert [s["op"] for s in loaded["steps"]] == ["start"]
    assert loaded["steps"][0]["argv"][0].endswith("pair-tf-harness/start.sh")


def test_stage_refuses_without_archive_and_when_pair_is_live():
    cat = load_catalog()
    data = fleet(cat, _runner())
    flash = row_by_id(data["library"], "pair-fast")
    refused = plan_stage(cat, flash, "spark-a", data["probes"], lock_on=False)
    assert refused["verdict"] == "REFUSE"
    assert any("archive" in r for r in refused["reasons"])

    live = fleet(cat, _runner(listening={"8888": True}))
    orn = row_by_id(live["library"], "release-a-nvfp4") if False else None
    # release-a is only present when the fake scan returns it
    data = fleet(
        cat,
        _runner(
            listening={"8888": True},
            releases=[
                {
                    "slug": "release-a-nvfp4",
                    "title": "Release-A",
                    "bytes": 1000,
                    "sha256": "abc",
                    "tar": "/mnt/archive/models/releases/release-a/file.tar",
                    "source": "/home/USER/services/vllm-release-a/models",
                }
            ],
        ),
    )
    orn = row_by_id(data["library"], "release-a-nvfp4")
    got = plan_stage(cat, orn, "spark-a", data["probes"], lock_on=False)
    assert got["verdict"] == "REFUSE"
    assert any("pair" in r for r in got["reasons"])


def test_stage_refuses_under_20_percent_free():
    cat = load_catalog()
    data = fleet(
        cat,
        _runner(
            free_gib=10,
            releases=[
                {
                    "slug": "release-a-nvfp4",
                    "title": "Release-A",
                    "bytes": 80 * 1024**3,
                    "sha256": "abc",
                    "tar": "/mnt/archive/models/releases/release-a/file.tar",
                    "source": "/home/USER/services/vllm-release-a/models",
                }
            ],
        ),
    )
    orn = row_by_id(data["library"], "release-a-nvfp4")
    got = plan_stage(cat, orn, "spark-b", data["probes"], lock_on=False)
    assert got["verdict"] == "REFUSE"
    assert any("20%" in r for r in got["reasons"])


def test_stage_ready_for_archived_release():
    cat = load_catalog()
    data = fleet(
        cat,
        _runner(
            releases=[
                {
                    "slug": "release-a-nvfp4",
                    "title": "Release-A",
                    "bytes": 1000,
                    "sha256": "abc",
                    "tar": "/mnt/archive/models/releases/release-a/file.tar",
                    "source": "/home/USER/services/vllm-release-a/models",
                }
            ],
        ),
    )
    orn = row_by_id(data["library"], "release-a-nvfp4")
    got = plan_stage(cat, orn, "spark-a", data["probes"], lock_on=False)
    assert got["verdict"] == "READY", got["reasons"]
    assert got["request"]["op"] == "stage_tar"
    assert "docker" not in json.dumps(got["request"])


def test_drop_refuses_pair_fast_and_live_weights():
    cat = load_catalog()
    flash_path = "/home/USER/.cache/huggingface/hub/models--example--pair-fast-nvfp4"
    data = fleet(cat, _runner(paths_exist={flash_path: True}))
    flash = row_by_id(data["library"], "pair-fast")
    got = plan_drop(flash, "spark-a", data["probes"], lock_on=False, confirm=flash_path)
    assert got["verdict"] == "REFUSE"
    assert any("archive" in r for r in got["reasons"])

    data = fleet(
        cat,
        _runner(
            listening={"8888": True},
            paths_exist={flash_path: True},
            http={"8888": {"path": "/v1/models", "status": 200, "models": ["pair-fast-model"]}},
            containers=["vllm-pf"],
        ),
    )
    flash = row_by_id(data["library"], "pair-fast")
    assert flash["live"] is True


def test_run_refuses_missing_launcher_comfy_and_missing_weights():
    cat = load_catalog()
    data = fleet(
        cat,
        _runner(
            releases=[
                {
                    "slug": "release-a-nvfp4",
                    "title": "Release-A",
                    "bytes": 1000,
                    "sha256": "abc",
                    "tar": "/mnt/archive/models/releases/release-a/file.tar",
                    "source": "/home/USER/services/vllm-release-a/models",
                }
            ],
        ),
    )
    orn = row_by_id(data["library"], "release-a-nvfp4")
    got = plan_run(cat, orn, data["probes"], lock_on=False)
    assert got["verdict"] == "REFUSE"
    assert got["argv"] is None
    assert any("launcher" in r for r in got["reasons"])

    flash_path = "/home/USER/.cache/huggingface/hub/models--example--pair-fast-nvfp4"
    glm_path = "/home/USER/.cache/huggingface/hub/models--example--pair-exl3-4bpw"
    data = fleet(cat, _runner(listening={"8188": True}, paths_exist={flash_path: True, glm_path: True}))
    flash = row_by_id(data["library"], "pair-fast")
    # pair local paths are the same string on both hosts; one path key covers both
    blocked = plan_run(cat, flash, data["probes"], lock_on=False)
    assert blocked["verdict"] == "REFUSE"
    assert any("protected video service" in r for r in blocked["reasons"])

    data = fleet(cat, _runner(paths_exist={}))
    flash = row_by_id(data["library"], "pair-fast")
    missing = plan_run(cat, flash, data["probes"], lock_on=False)
    assert missing["verdict"] == "REFUSE"
    assert any("absent" in r for r in missing["reasons"])


def test_run_ready_when_weights_are_local():
    cat = load_catalog()
    flash_path = "/home/USER/.cache/huggingface/hub/models--example--pair-fast-nvfp4"
    data = fleet(cat, _runner(paths_exist={flash_path: True}))
    flash = row_by_id(data["library"], "pair-fast")
    got = plan_run(cat, flash, data["probes"], lock_on=False)
    assert got["verdict"] == "READY", got["reasons"]
    assert got["argv"][0].endswith("start.sh")
    assert "--launch" in got["argv"]
    assert not any("docker" == a for a in got["argv"])


def test_agent_stage_and_drop_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_ARCHIVE_ROOT", str(tmp_path / "archive_tar"))
    tar_dir = tmp_path / "archive_tar" / "models" / "releases" / "toy"
    tar_dir.mkdir(parents=True)
    payload = tmp_path / "weight.bin"
    payload.write_bytes(b"weights-go-here")
    tar_path = tar_dir / "toy.tar"
    with tarfile.open(tar_path, "w") as tf:
        tf.add(payload, arcname="weight.bin")
    digest = hashlib.sha256(tar_path.read_bytes()).hexdigest()
    dest = tmp_path / "a" / "b" / "c" / "models"
    staged = agent_main(
        {
            "op": "stage_tar",
            "tar": str(tar_path),
            "sha256": digest,
            "dest": str(dest),
            "bytes": tar_path.stat().st_size,
            "min_free_ratio": 0.01,
            "refuse_if_listening": [],
        }
    )
    assert staged["ok"] is True, staged
    assert (dest / "weight.bin").read_bytes() == b"weights-go-here"
    bad = agent_main(
        {
            "op": "stage_tar",
            "tar": str(tmp_path / "not-on-archive.tar"),
            "sha256": digest,
            "dest": str(dest),
            "bytes": 10,
        }
    )
    assert bad["ok"] is False
    hub = agent_main({"op": "drop", "path": "/home/USER/.cache/huggingface/hub", "confirm": "/home/USER/.cache/huggingface/hub"})
    assert hub["ok"] is False
    dropped = agent_main(
        {
            "op": "drop",
            "path": str(dest),
            "confirm": str(dest),
            "tar": str(tar_path),
            "sha256": digest,
            "refuse_if_listening": [],
        }
    )
    assert dropped["ok"] is True, dropped
    assert not dest.exists()


def test_stop_video_uses_stop_video():
    from sparksi.launch import plan_stop

    row = {
        "id": "video-seat",
        "listen_host": "spark-a",
        "seats": {
            "spark-a": {
                "port": 8188,
                "start": ["/home/USER/scripts/video-mode.sh", "video"],
                "stop": ["/home/USER/scripts/video-mode.sh", "stop-video"],
            }
        },
        "launch": {
            "host": "spark-a",
            "start": ["/home/USER/scripts/video-mode.sh", "video"],
            "stop": ["/home/USER/scripts/video-mode.sh", "stop-video"],
        },
    }
    plan = plan_stop(row, lock_on=False)
    assert plan["verdict"] == "READY", plan["reasons"]
    assert plan["argv"][-1] == "stop-video"
    blocked = plan_stop(
        {**row, "launch": {"stop": ["/home/USER/scripts/video-mode.sh", "trio"]}, "seats": {}},
        lock_on=False,
    )
    assert blocked["verdict"] == "REFUSE"


def test_video_load_starts_video_and_still_refuses_trio():
    from sparksi.jobs import perform_switch
    from sparksi.launch import plan_switch
    from sparksi.never import launcher_refusal
    from sparksi.remote import SshRunner

    hosts = {
        "hosts": {
            "spark-a": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
            "spark-b": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4},
        }
    }
    big_row = {
        "id": "single-a",
        "title": "Single A",
        "placement": "single",
        "live": True,
        "estimate_gib": 90,
        "listen_host": "spark-a",
        "on": {"spark-a": "live"},
        "seats": {"spark-a": {"port": 8014, "start": ["start-big"], "stop": ["stop-big"]}},
    }
    h3 = {
        "id": "video-seat",
        "title": "ComfyUI video",
        "placement": "single",
        "live": False,
        "start_note": "ComfyUI starts empty. The video model loads on the first job.",
        "estimate_gib": 40,
        "listen_host": "spark-a",
        "on": {"spark-a": "local"},
        "seats": {
            "spark-a": {
                "port": 8188,
                "start": ["/home/USER/scripts/video-mode.sh", "video"],
                "stop": ["/home/USER/scripts/video-mode.sh", "stop-video"],
            }
        },
    }
    on_head = plan_switch(hosts, h3, [big_row, h3], lock_on=False, host="spark-a")
    assert on_head["verdict"] == "READY"
    assert on_head["steps"][-1]["op"] == "note"
    assert on_head["steps"][-2]["argv"] == ["/home/USER/scripts/video-mode.sh", "video"]
    assert on_head["steps"][0]["op"] == "stop"
    on_worker = plan_switch(hosts, h3, [big_row, h3], lock_on=False, host="spark-b")
    assert on_worker["reasons"] == ["no launcher on b"]
    assert "video-mode.sh" not in json.dumps(on_worker)
    assert launcher_refusal(["/home/USER/scripts/video-mode.sh", "stop-video"]) is None
    assert launcher_refusal(["/home/USER/scripts/video-mode.sh", "video"]) is None
    runner = SshRunner({})
    denied = runner.run_argv("spark-a", ["/home/USER/scripts/video-mode.sh", "trio"])
    assert denied["ok"] is False
    assert "trio" in denied["error"]

    calls = []

    class Runner:
        def run_argv(self, host, argv, timeout=0):
            calls.append(list(argv))
            return {"ok": True}

        def invoke(self, host, req, timeout=0):
            raise AssertionError(req)

    started = perform_switch(
        Runner(),
        {"steps": [{"op": "start", "host": "spark-a", "argv": ["/home/USER/scripts/video-mode.sh", "video"]}]},
        lambda step: False,
    )
    assert started["ok"] is True
    assert calls == [["/home/USER/scripts/video-mode.sh", "video"]]
    blocked = perform_switch(
        Runner(),
        {"steps": [{"op": "start", "host": "spark-a", "argv": ["/home/USER/scripts/video-mode.sh", "trio"]}]},
        lambda step: False,
    )
    assert blocked["ok"] is False
    assert blocked["started"] is False
    assert "trio" in blocked["error"]


def test_accepted_load_locks_before_start(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    from sparksi.jobs import start_stop, start_switch
    from sparksi.lock import Lock

    Lock(tmp_path / "lock").release(ttl_s=60)
    big_row = {
        "id": "single-a",
        "title": "Single A",
        "placement": "single",
        "estimate_gib": 90,
        "listen_host": "spark-a",
        "on": {"spark-a": "local"},
        "seats": {"spark-a": {"port": 8014, "start": ["start-big"], "stop": ["stop-big"]}},
        "launch": {"host": "spark-a", "stop": ["stop-big"]},
    }
    hosts = {"hosts": {"spark-a": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}}}

    class Runner:
        def invoke(self, host, req, timeout=0):
            ports = req.get("ports") or []
            return {"ok": True, "listening": {str(p): True for p in ports}}

        def run_argv(self, host, argv, timeout=0):
            return {"ok": True}

    monkeypatch.setattr("sparksi.jobs.fleet", lambda catalog, runner=None: {"library": [big_row], "probes": {}})
    started = start_switch(hosts, "single-a", lock_on=False, host="spark-a", runner=Runner())
    assert started["verdict"] == "STARTED"
    assert started["lock"]["locked"] is True
    assert started["lock"]["reason"] == "load started"
    assert Lock(tmp_path / "lock").read().locked is True
    monkeypatch.setattr(
        "sparksi.jobs.current",
        lambda: {"kind": "switch", "phase": "start", "id_row": "single-a"},
    )
    refused = start_stop(hosts, "single-a", lock_on=False, runner=Runner())
    assert refused["verdict"] == "REFUSE"
    assert "does not interrupt a load" in refused["reasons"][0]


def test_finished_switch_locks(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    from sparksi.jobs import relock_after_switch
    from sparksi.lock import Lock

    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=60)
    assert lock.read().locked is False
    relock_after_switch()
    again = Lock(tmp_path / "lock").read()
    assert again.locked is True
    assert again.reason == "switch finished"


def test_switch_stops_before_start_and_refuses_a_live_port():
    from sparksi.jobs import perform_switch
    from sparksi.launch import plan_switch

    catalog = {
        "exclusive_groups": {"big-pair": ["pair-exl3", "pair-fast", "video-seat"]},
    }
    glm = {
        "id": "pair-exl3",
        "title": "GLM",
        "live": True,
        "exclusive": "big-pair",
        "port": 8014,
        "placement": "pair",
        "on": {"spark-a": "live", "spark-b": "live"},
        "launch": {"host": "spark-a", "start": ["start-exl3"], "stop": ["stop-exl3"]},
    }
    flash = {
        "id": "pair-fast",
        "title": "Pair-Fast",
        "live": False,
        "exclusive": "big-pair",
        "port": 8888,
        "placement": "pair",
        "on": {"spark-a": "local", "spark-b": "local"},
        "launch": {"host": "spark-a", "start": ["start-fast", "--launch"], "stop": ["stop-fast"]},
    }
    video = {
        "id": "video-seat",
        "title": "ComfyUI video",
        "live": False,
        "exclusive": "big-pair",
        "port": 8188,
        "launch": {"host": "spark-a", "start": ["/home/USER/scripts/video-mode.sh", "video"], "stop": ["/home/USER/scripts/video-mode.sh", "stop-video"]},
    }
    plan = plan_switch(catalog, flash, [glm, flash, video], lock_on=False, host="pair")
    assert plan["verdict"] == "READY"
    assert [s["op"] for s in plan["steps"]] == ["stop", "start"]
    assert "pair" in plan["preview"][-1]

    bare = {
        "id": "release-a",
        "title": "Release-A",
        "live": False,
        "placement": "single",
        "on": {"spark-a": "absent"},
        "launch": None,
        "seats": {},
    }
    refused = plan_switch(catalog, bare, [glm, bare], lock_on=False, host="spark-a")
    assert refused["verdict"] == "REFUSE"
    assert refused["steps"] == []
    assert refused["reasons"] == ["no launcher on a"]

    class Runner:
        def __init__(self, clear: bool):
            self.clear = clear
            self.argv = []
            self.up = {8014: True, 8888: False}

        def invoke(self, host, req, timeout=0):
            ports = req.get("ports") or []
            listening = {str(p): self.up.get(int(p), False) for p in ports}
            http = {}
            for port, up in listening.items():
                if up:
                    http[port] = {"status": 200}
            return {"ok": True, "listening": listening, "http": http}

        def run_argv(self, host, argv, timeout=0):
            self.argv.append(list(argv))
            if self.clear and argv == ["stop-exl3"]:
                self.up[8014] = False
            if argv[:1] == ["start-fast"]:
                self.up[8888] = True
            return {"ok": True}

    stuck = Runner(False)
    blocked = perform_switch(stuck, plan, lambda step: False, attempts=1, pause=0)
    assert blocked["ok"] is False
    assert blocked["started"] is False
    assert stuck.argv == [["stop-exl3"]]

    moved = Runner(True)
    done = perform_switch(moved, plan, lambda step: False, attempts=2, pause=0)
    assert done["ok"] is True
    assert moved.argv == [["stop-exl3"], ["start-fast", "--launch"]]

    big_row = {
        "id": "single-a",
        "title": "Single A",
        "placement": "single",
        "estimate_gib": 90,
        "listen_host": "spark-a",
        "on": {"spark-a": "local", "spark-b": "absent"},
        "seats": {"spark-a": {"port": 8014, "start": ["start-big"], "stop": ["stop-big"]}},
    }
    h3 = {
        "id": "video-seat",
        "title": "ComfyUI video",
        "placement": "single",
        "live": True,
        "estimate_gib": 40,
        "listen_host": "spark-a",
        "on": {"spark-a": "live"},
        "seats": {"spark-a": {"port": 8188, "start": ["video-mode.sh", "video"], "stop": ["video-mode.sh", "stop-video"]}},
    }
    hosts = {"hosts": {"spark-a": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}, "spark-b": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}}}
    on_head = plan_switch(hosts, big_row, [big_row, h3], lock_on=False, host="spark-a")
    assert on_head["verdict"] == "READY"
    assert [s["text"] for s in on_head["steps"]] == ["on a, stop ComfyUI video", "on a, start Single A"]
    assert "b" not in " ".join(on_head["preview"])
    on_worker = plan_switch(hosts, h3, [big_row, h3], lock_on=False, host="spark-b")
    assert on_worker["verdict"] == "REFUSE"
    assert on_worker["reasons"] == ["no launcher on b"]
    assert on_worker["steps"] == []
    assert "video-mode.sh" not in json.dumps(on_worker)
    q_worker = plan_switch(hosts, big_row, [big_row, h3], lock_on=False, host="spark-b")
    assert q_worker["reasons"] == ["no launcher on b"]


def test_worker_studio_waits_until_the_pair_is_finished():
    from sparksi.launch import plan_switch

    hosts = {"hosts": {"spark-a": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}, "spark-b": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}}}
    studio = {
        "id": "studio-b",
        "title": "Studio B",
        "placement": "single",
        "estimate_gib": 8,
        "listen_host": "spark-b",
        "on": {"spark-b": "local"},
        "seats": {"spark-b": {"port": 8765, "start": ["sudo", "systemctl", "start", "studio-b"], "stop": ["sudo", "systemctl", "stop", "studio-b"]}},
    }
    pair = {"id": "pair-exl3", "title": "GLM", "placement": "pair", "live": True, "on": {"spark-a": "live", "spark-b": "live"}}
    qwen = {"id": "single-a", "title": "Single A", "placement": "single", "live": True, "on": {"spark-a": "live"}}
    blocked = plan_switch(hosts, studio, [studio, pair, qwen], lock_on=False, host="spark-b")
    assert blocked["verdict"] == "REFUSE"
    assert blocked["reasons"] == ["b is in a pair; stop that model first"]
    assert blocked["steps"] == []
    clear = plan_switch(hosts, studio, [studio, qwen], lock_on=False, host="spark-b")
    assert clear["verdict"] == "READY"
    assert clear["steps"][-1]["argv"] == ["sudo", "systemctl", "start", "studio-b"]
    assert clear["steps"][-1]["host"] == "spark-b"
    on_head = plan_switch(hosts, studio, [studio, qwen], lock_on=False, host="spark-a")
    assert on_head["reasons"] == ["no launcher on a"]
    # Port 8014 makes the GLM row look live while Qwen is the model actually loaded.
    qwen_loaded = plan_switch(
        hosts, studio, [studio, pair, qwen], lock_on=False, host="spark-b",
        loaded=[{"host": "a", "served": "single-a"}],
    )
    assert qwen_loaded["verdict"] == "READY"
    glm_loaded = plan_switch(
        hosts, studio, [studio, pair], lock_on=False, host="spark-b",
        loaded=[{"host": "a", "span": ["a", "b"], "served": "Pair-EXL3"}],
    )
    assert glm_loaded["verdict"] == "REFUSE"
    assert glm_loaded["reasons"] == ["b is in a pair; stop that model first"]


def test_svg_b_load_is_worker_only():
    from sparksi.launch import plan_switch

    hosts = {"hosts": {"spark-a": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}, "spark-b": {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}}}
    svg = {
        "id": "svg-b",
        "title": "SVG B 4B",
        "placement": "single",
        "estimate_gib": 17,
        "listen_host": "spark-b",
        "on": {"spark-b": "local"},
        "seats": {"spark-b": {"port": 8195, "start": ["/home/USER/services/svg-b/start.sh"], "stop": ["/home/USER/services/svg-b/stop.sh"]}},
    }
    pair = {"id": "pair-exl3", "title": "GLM", "placement": "pair", "live": True, "on": {"spark-a": "live", "spark-b": "live"}}
    qwen = {"id": "single-a", "title": "Single A", "placement": "single", "live": True, "on": {"spark-a": "live"}}
    h3 = {"id": "video-seat", "title": "ComfyUI video", "placement": "single", "live": True, "estimate_gib": 40, "listen_host": "spark-a", "on": {"spark-a": "live"}, "seats": {"spark-a": {"port": 8188, "start": ["video-mode.sh", "video"], "stop": ["video-mode.sh", "stop-video"]}}}
    blocked = plan_switch(hosts, svg, [svg, pair, qwen, h3], lock_on=False, host="spark-b")
    assert blocked["verdict"] == "REFUSE"
    assert blocked["reasons"] == ["b is in a pair; stop that model first"]
    clear = plan_switch(hosts, svg, [svg, qwen, h3], lock_on=False, host="spark-b")
    assert clear["verdict"] == "READY"
    assert clear["steps"][-1]["argv"] == ["/home/USER/services/svg-b/start.sh"]
    assert clear["steps"][-1]["host"] == "spark-b"
    assert "video-mode.sh" not in json.dumps(clear)
    on_head = plan_switch(hosts, svg, [svg, qwen, h3], lock_on=False, host="spark-a")
    assert on_head["reasons"] == ["no launcher on a"]
    assert on_head["steps"] == []


@pytest.mark.skipif(shutil.which("rsync") is None, reason="needs rsync on PATH")
def test_shard_copy_skips_full_files(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_ARCHIVE_ROOT", str(tmp_path / "archive_tar"))
    src = tmp_path / "archive_tar" / "models" / "glm"
    src.mkdir(parents=True)
    (src / "model-00001-of-00002.safetensors").write_bytes(b"a" * 20)
    (src / "model-00002-of-00002.safetensors").write_bytes(b"b" * 20)
    dest = tmp_path / "a" / "b" / "c" / "snap"
    dest.mkdir(parents=True)
    (dest / "model-00001-of-00002.safetensors").write_bytes(b"a" * 20)
    listed = agent_main(
        {
            "op": "list_shards",
            "src": str(src),
            "dest": str(dest),
            "shards": 2,
            "min_bytes": 10,
        }
    )
    assert listed["ok"] is True, listed
    assert listed["done"] == ["model-00001-of-00002.safetensors"]
    assert listed["pending"] == ["model-00002-of-00002.safetensors"]
    copied = agent_main(
        {
            "op": "stage_shard",
            "src": str(src),
            "dest": str(dest),
            "name": "model-00002-of-00002.safetensors",
            "min_free_ratio": 0.0,
            "refuse_if_listening": [],
        }
    )
    assert copied["ok"] is True, copied
    assert (dest / "model-00002-of-00002.safetensors").read_bytes() == b"b" * 20


def test_stage_does_not_use_gpu_lock(tmp_path, monkeypatch):
    import json
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from sparksi.lock import Lock
    from sparksi.serve import STATE, DualServer, Handler

    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    STATE.lock = Lock(tmp_path / "lock")
    assert STATE.lock.read().locked
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        def post(path, payload):
            req = Request(
                f"http://127.0.0.1:{port}{path}",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode())
        try:
            post("/api/library/run", {"id": "pair-exl3", "from_ui": True})
            raise AssertionError("run should be locked")
        except Exception as exc:
            assert getattr(exc, "code", None) == 403
        try:
            post("/api/library/stage", {"id": "pair-exl3", "from_ui": True})
            raise AssertionError("stage without a host should be a 400")
        except HTTPError as exc:
            assert exc.code == 400
            assert json.loads(exc.read().decode()).get("error") == "missing host"
    finally:
        httpd.shutdown()


def test_controller_bind(monkeypatch):
    monkeypatch.setenv("SPARKSI_CONTROLLER_MODE", "controller")
    hosts = allowed_hosts(load_catalog())
    assert hosts == ["127.0.0.1", "100.64.0.1"]


UNDERDOG = "/home/USER/services/single-gguf/models/single-gguf-q2.gguf"
GLM = "/home/USER/.cache/huggingface/hub/models--example--pair-exl3-4bpw"
GLM_TF = "/home/USER/.cache/huggingface/hub/models--example--pair-tf-4bpw"
QWEN = "/home/USER/services/vllm-single-a"


def _ground_truth_2026_10_19(**extra):
    """Only llama-server gguf answers :8014 on a; GLM/Qwen weights are on disk; no containers."""
    kw = dict(
        listening={"8014": True},
        paths_exist={UNDERDOG: True, GLM: True, GLM_TF: True, QWEN: True},
        http={"8014": {"path": "/v1/models", "status": 200, "models": ["single-gguf-model"]}},
        containers=[],
    )
    kw.update(extra)
    return kw


def test_shared_8014_is_live_only_for_the_model_that_answers():
    cat = load_catalog()
    data = fleet(cat, _runner(**_ground_truth_2026_10_19()))
    rows = {r["id"]: r for r in data["library"]}
    assert rows["single-gguf"]["live"] is True
    assert rows["single-gguf"]["on"]["spark-a"] == "live"
    for rid in ("pair-exl3", "pair-tf", "single-a", "pair-fast"):
        assert rows[rid]["live"] is False, rid
        assert "live" not in rows[rid]["on"].values(), rid
    assert rows["single-a"]["on"]["spark-a"] == "local"
    assert rows["single-a"]["port_holder"]["spark-a"] == "single-gguf-model"


def test_bound_port_without_http_answer_is_not_live():
    cat = load_catalog()
    data = fleet(cat, _runner(**_ground_truth_2026_10_19(http={})))
    under = row_by_id(data["library"], "single-gguf")
    assert under["live"] is False
    assert under["port_holder"]["spark-a"] == "port bound, no HTTP answer"


def test_pair_needs_its_own_container_and_worker_worker():
    cat = load_catalog()
    http = {"8014": {"path": "/v1/models", "status": 200, "models": ["Pair-EXL3"]}}

    def invoke(host, req, timeout=0):
        cont = ["pair-tf"] if host == "spark-a" else []
        return _probe(host, req, listening={"8014": True} if host == "spark-a" else {},
                      paths_exist={GLM: True, GLM_TF: True}, http=http if host == "spark-a" else {},
                      containers=cont)

    rows = {r["id"]: r for r in fleet(cat, LocalRunner(invoke))["library"]}
    assert rows["pair-exl3"]["live"] is False  # pair-exl3-head is not running
    assert rows["pair-tf"]["on"]["spark-a"] == "live"
    assert rows["pair-tf"]["on"]["spark-b"] == "local"  # worker container not up on b


def test_comfy_live_needs_system_stats():
    cat = load_catalog()
    comfy_path = "/home/USER/services/video"
    data = fleet(cat, _runner(listening={"8188": True}, paths_exist={comfy_path: True},
                              http={"8188": {"path": "/system_stats", "status": 404, "models": []}}))
    assert row_by_id(data["library"], "video-seat")["live"] is False
    data = fleet(cat, _runner(listening={"8188": True}, paths_exist={comfy_path: True},
                              http={"8188": {"path": "/system_stats", "status": 200, "models": [], "comfy": True}}))
    assert row_by_id(data["library"], "video-seat")["live"] is True


def test_probe_asks_comfy_system_stats():
    cat = load_catalog()
    seen = {}

    def invoke(host, req, timeout=0):
        seen[host] = req.get("http_paths")
        return _probe(host, req)

    fleet(cat, LocalRunner(invoke))
    assert seen["spark-a"]["8188"] == "/system_stats"


def test_switch_refuses_port_held_by_unknown_server():
    cat = load_catalog()
    http = {"8014": {"path": "/v1/models", "status": 200, "models": ["mystery-model"]}}
    data = fleet(cat, _runner(**_ground_truth_2026_10_19(http=http)))
    qwen = row_by_id(data["library"], "single-a")
    got = plan_switch(cat, qwen, data["library"], lock_on=False, host="spark-a")
    assert got["verdict"] == "REFUSE"
    assert any("mystery-model" in r for r in got["reasons"])
    # gguf answering is a known row, so the plan stops it first instead
    data = fleet(cat, _runner(**_ground_truth_2026_10_19()))
    qwen = row_by_id(data["library"], "single-a")
    got = plan_switch(cat, qwen, data["library"], lock_on=False, host="spark-a")
    assert got["steps"][0]["op"] == "stop" and got["steps"][0]["id"] == "single-gguf"
    assert not any(s.get("id", "").startswith("glm") for s in got["steps"] if s["op"] == "stop")


def test_fleet_reports_machine_ips():
    cat = load_catalog()

    def invoke(host, req, timeout=0):
        out = _probe(host, req)
        out["ips"] = {"lan": "192.0.2.65" if host == "spark-a" else "192.0.2.67",
                      "tailscale": "100.64.0.11" if host == "spark-a" else "100.64.0.12", "direct": []}
        return out

    data = fleet(cat, LocalRunner(invoke))
    by = {m["id"]: m for m in data["machines"]}
    assert by["spark-a"]["lan"] == "192.0.2.65"
    assert by["spark-b"]["tailscale"] == "100.64.0.12"
    assert "controller" in by
    card = [h for h in data["hosts"] if h["id"] == "spark-a"][0]
    assert card["ips"]["tailscale"] == "100.64.0.11"


def test_unload_of_a_model_that_is_not_running_runs_no_stop(monkeypatch):
    """Unload on a card with nothing running is a 409 no-op: the stop command never runs."""
    from sparksi import jobs

    calls = []
    row = {"id": "m", "title": "M", "serving": False, "launch": {"host": "spark-a", "stop": ["stop-m"]}}
    monkeypatch.setattr("sparksi.jobs.fleet", lambda *a, **k: {"library": [row], "probes": {}})
    monkeypatch.setattr("sparksi.jobs.load_in_progress", lambda: False)

    class Runner:
        def run_argv(self, *a, **k):
            calls.append(a)
            return {"ok": True}

    out = jobs.start_stop({}, "m", lock_on=False, runner=Runner())
    assert out["verdict"] == "REFUSE"
    assert out["not_running"] is True and out["stopped"] is False
    assert "not running" in out["reasons"][0]
    assert calls == []


def test_serving_needs_the_rows_own_container():
    """Another harness serving the same id on the same port does not make this row running."""
    from sparksi.library import annotate

    row = {"id": "m", "placement": "single", "port": 8014, "listen_host": "spark-a",
           "identity": {"served": ["M"], "container": "m-head"}}
    probe = {"listening": {"8014": True}, "http": {"8014": {"status": 200, "models": ["M"]}},
             "containers": ["m-head"], "paths": {}}
    assert annotate([row], {"spark-a": probe})[0]["serving"] is True
    other = dict(probe, containers=["other-head"])
    assert annotate([row], {"spark-a": other})[0]["serving"] is False


def _pair_row():
    return {"id": "p", "placement": "pair", "port": 8014, "listen_host": "spark-a",
            "identity": {"served": ["P"], "container": "p-head", "worker_container": "p-worker"}}


def _down(containers):
    return {"listening": {"8014": False}, "http": {}, "paths": {}, "containers": containers}


def test_a_half_up_pair_counts_as_running_so_its_stop_is_allowed(monkeypatch):
    """Head down, worker still up: Unload must still be able to stop the worker."""
    from sparksi import jobs
    from sparksi.library import annotate

    row = _pair_row()
    half = annotate([row], {"spark-a": _down([]), "spark-b": _down(["p-worker"])})[0]
    assert half["serving"] is True
    loading = annotate([row], {"spark-a": _down(["p-head"]), "spark-b": _down([])})[0]
    assert loading["serving"] is True
    gone = annotate([row], {"spark-a": _down([]), "spark-b": _down([])})[0]
    assert gone["serving"] is False
    # The worker's name on the head host is not the worker.
    wrong_host = annotate([row], {"spark-a": _down(["p-worker"]), "spark-b": _down([])})[0]
    assert wrong_host["serving"] is False

    launch = dict(half, title="P", launch={"host": "spark-a", "stop": ["stop-p"]})
    monkeypatch.setattr("sparksi.jobs.fleet", lambda *a, **k: {"library": [launch], "probes": {}})
    monkeypatch.setattr("sparksi.jobs.load_in_progress", lambda: False)
    started = []
    monkeypatch.setattr("sparksi.jobs._start_argv",
                        lambda catalog, kind, plan, runner: started.append((kind, plan)) or {"started": True})
    out = jobs.start_stop({}, "p", lock_on=False, runner=None)
    assert not out.get("not_running"), out
    assert [kind for kind, _plan in started] == ["stop"]


def test_an_unlisted_container_is_not_running_and_unknown_lists_do_not_block_a_stop(monkeypatch):
    """A container missing from the running list is down; no list at all is unknown (None)."""
    from sparksi import jobs
    from sparksi.library import annotate

    row = {"id": "m", "placement": "single", "port": 8014, "listen_host": "spark-a",
           "identity": {"served": ["M"], "container": "m-head"}}
    answering = {"listening": {"8014": True}, "http": {"8014": {"status": 200, "models": ["M"]}}, "paths": {}}
    assert annotate([row], {"spark-a": dict(answering, containers=["m-other"])})[0]["serving"] is False
    assert annotate([row], {"spark-a": _down(["m-other"])})[0]["serving"] is False
    # Docker unreadable: the served-name match still counts, and a silent port is unknown.
    assert annotate([row], {"spark-a": dict(answering, containers=None)})[0]["serving"] is True
    unknown = annotate([row], {"spark-a": _down(None)})[0]
    assert unknown["serving"] is None

    launch = dict(unknown, title="M", launch={"host": "spark-a", "stop": ["stop-m"]})
    monkeypatch.setattr("sparksi.jobs.fleet", lambda *a, **k: {"library": [launch], "probes": {}})
    monkeypatch.setattr("sparksi.jobs.load_in_progress", lambda: False)
    started = []
    monkeypatch.setattr("sparksi.jobs._start_argv",
                        lambda catalog, kind, plan, runner: started.append((kind, plan)) or {"started": True})
    out = jobs.start_stop({}, "m", lock_on=False, runner=None)
    assert not out.get("not_running"), out
    assert [kind for kind, _plan in started] == ["stop"]
