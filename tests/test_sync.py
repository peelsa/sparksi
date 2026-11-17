# SPDX-License-Identifier: Apache-2.0

"""Sync with storage: a read-only compare of each Spark against the archive, and the refill copy."""

from __future__ import annotations

import sparksi.launch as launch
import sparksi.library as library
from sparksi.launch import plan_stage

HOSTS = ["spark-a", "spark-b"]


class Runner:
    def __init__(self, pending: list[str]) -> None:
        self.pending = pending
        self.calls: list[tuple[str, dict]] = []

    def invoke(self, host, req, timeout=30):
        self.calls.append((host, dict(req)))
        if req.get("op") == "list_shards":
            done = [n for n in ("s1", "s2", "s3") if n not in self.pending]
            return {"ok": True, "total": 3, "done": done, "pending": list(self.pending)}
        return {"ok": False, "error": "unexpected"}


def _rows(flat_on: str = "local") -> list[dict]:
    local = {"spark-a": "/srv/models/a/{id}", "spark-b": "/srv/models/b/{id}"}

    def row(rid: str, on: dict, archive_state: str, **extra) -> dict:
        return {"id": rid, "title": rid.upper(), "gib": 10, "on": on, "archive_state": archive_state,
                "local": {h: p.format(id=rid) for h, p in local.items()}, **extra}

    return [
        row("flat-x", {"spark-a": flat_on, "spark-b": "absent"}, "archived", archive_flat="/archive/flat-x", shards=3),
        row("tar-y", {"spark-a": "absent", "spark-b": "absent"}, "archived", archive_tar="/archive/tar-y.tar"),
        row("only-z", {"spark-a": "local", "spark-b": "absent"}, "none"),
        row("none-w", {"spark-a": "absent", "spark-b": "absent"}, "none"),
    ]


def _report(monkeypatch, runner: Runner, flat_on: str = "local") -> dict:
    snap = {
        "library": _rows(flat_on),
        "probes": {"spark-a": {"ok": True, "mount_archive": True}, "spark-b": {"ok": False, "error": "timed out"}},
    }
    monkeypatch.setattr(library, "fleet", lambda catalog, runner=None: snap)
    monkeypatch.setattr(library, "host_ids", lambda catalog: list(HOSTS))
    report = library.sync_report({}, runner)
    return {row["id"]: row for row in report["rows"]}


def test_sync_report_names_each_difference(monkeypatch) -> None:
    runner = Runner(pending=["s3"])
    rows = _report(monkeypatch, runner)
    flat_a = rows["flat-x"]["hosts"]["spark-a"]
    assert flat_a["state"] == "differs" and flat_a["refill"] is True and flat_a["can_copy"] is True
    assert "1 of 3 shards" in flat_a["note"]
    asks = [req for host, req in runner.calls if req.get("op") == "list_shards"]
    assert asks and all(req["exact"] is True for req in asks)
    assert asks[0]["dest"] == "/srv/models/a/flat-x" and asks[0]["src"] == "/archive/flat-x"
    assert rows["flat-x"]["hosts"]["spark-b"]["state"] == "unchecked"
    assert rows["tar-y"]["hosts"]["spark-a"] == {"state": "missing", "note": "not on this Spark", "can_copy": True, "refill": False}
    assert rows["only-z"]["hosts"]["spark-a"]["state"] == "not_in_storage"
    assert rows["only-z"]["hosts"]["spark-a"]["can_copy"] is False
    assert "none-w" not in rows
    # Only list_shards was asked: the check copies and deletes nothing.
    assert {req["op"] for _, req in runner.calls} == {"list_shards"}


def test_sync_report_in_sync_and_in_use(monkeypatch) -> None:
    rows = _report(monkeypatch, Runner(pending=[]))
    assert rows["flat-x"]["hosts"]["spark-a"]["state"] == "in_sync"
    rows = _report(monkeypatch, Runner(pending=["s1"]), flat_on="live")
    cell = rows["flat-x"]["hosts"]["spark-a"]
    assert cell["state"] == "differs" and cell["can_copy"] is False and "in use" in cell["note"]


def test_sync_report_without_the_archive_mount_offers_no_copy(monkeypatch) -> None:
    snap = {"library": _rows(), "probes": {"spark-a": {"ok": True}, "spark-b": {"ok": True}}}
    monkeypatch.setattr(library, "fleet", lambda catalog, runner=None: snap)
    monkeypatch.setattr(library, "host_ids", lambda catalog: list(HOSTS))
    rows = {row["id"]: row for row in library.sync_report({}, Runner(pending=[]))["rows"]}
    cell = rows["tar-y"]["hosts"]["spark-a"]
    assert cell["state"] == "missing" and cell["can_copy"] is False and "not mounted" in cell["note"]


def test_plan_stage_refill_only_for_a_flat_row_that_is_not_in_use(monkeypatch) -> None:
    monkeypatch.setattr(launch, "pair_live", lambda probes: False)
    catalog = {"hosts": {"spark-a": {}}}
    probes = {"spark-a": {"ok": True, "mount_archive": True, "disk_free_gib": 1000, "disk_total_gib": 2000, "flat": {"ok": True}}}
    row = {"id": "flat-x", "on": {"spark-a": "local"}, "bytes": 0, "local": {"spark-a": "/srv/models/a/flat-x"},
           "archive_flat": "/archive/flat-x", "shards": 3}
    plain = plan_stage(catalog, row, "spark-a", probes, lock_on=False)
    assert plain["verdict"] == "REFUSE" and "already present on spark-a" in plain["reasons"]
    refill = plan_stage(catalog, row, "spark-a", probes, lock_on=False, refill=True)
    assert refill["verdict"] == "READY", refill
    assert refill["request"]["exact"] is True and refill["request"]["op"] == "stage_flat"
    live = plan_stage(catalog, {**row, "on": {"spark-a": "live"}, "live": True}, "spark-a", probes, lock_on=False, refill=True)
    assert live["verdict"] == "REFUSE"
    tar = plan_stage(catalog, {**row, "archive_flat": None, "archive_tar": "/archive/t.tar", "sha256": "x", "source_path": "/srv/x"},
                     "spark-a", probes, lock_on=False, refill=True)
    assert "already present on spark-a" in tar["reasons"]


def test_agent_list_shards_exact_compares_sizes(monkeypatch, tmp_path) -> None:
    from pathlib import Path

    import sparksi.agent as agent

    src = tmp_path / "archive" / "flat-x"
    dest = tmp_path / "models" / "flat-x"
    src.mkdir(parents=True)
    dest.mkdir(parents=True)
    names = [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]
    for name in names:
        (src / name).write_bytes(b"x" * 64)
    (dest / names[0]).write_bytes(b"x" * 64)
    (dest / names[1]).write_bytes(b"x" * 32)
    monkeypatch.setattr(agent, "archive_root", lambda: str(tmp_path / "archive"))
    monkeypatch.setattr(agent, "_weight_path", lambda raw: Path(raw))
    out = agent._list_shards({"src": str(src), "dest": str(dest), "shards": 3, "exact": True})
    assert out["ok"] and out["exact"] is True
    assert out["done"] == [names[0]]
    assert out["pending"] == [names[1], names[2]]
    loose = agent._list_shards({"src": str(src), "dest": str(dest), "shards": 3, "min_bytes": 16})
    assert loose["done"] == [names[0], names[1]] and loose["exact"] is False


# --- Refill copies replace wrong files (rsync is faked: the command list is what is checked) ---

SHARD = "model-00001-of-00001.safetensors"


def _agent_dirs(monkeypatch, tmp_path):
    from pathlib import Path

    import sparksi.agent as agent

    src = tmp_path / "archive" / "flat-x"
    dest = tmp_path / "models" / "flat-x"
    src.mkdir(parents=True)
    dest.mkdir(parents=True)
    monkeypatch.setattr(agent, "archive_root", lambda: str(tmp_path / "archive"))
    monkeypatch.setattr(agent, "_weight_path", lambda raw: Path(raw))
    monkeypatch.setattr(agent, "_free_ok", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_refuse_listening", lambda ports: None)
    runs: list[list[str]] = []

    class Done:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(argv, **kwargs):
        runs.append(list(argv))
        if "--append-verify" in argv and Path(argv[-1]).is_file() and Path(argv[-1]).stat().st_size >= Path(argv[-2]).stat().st_size:
            return Done()  # what real rsync does: skip a target that is not smaller, exit 0
        Path(argv[-1]).write_bytes(Path(argv[-2]).read_bytes())
        return Done()

    monkeypatch.setattr(agent.subprocess, "run", fake_run)
    return agent, src, dest, runs


def test_refill_replaces_a_shard_bigger_on_the_spark(monkeypatch, tmp_path) -> None:
    agent, src, dest, runs = _agent_dirs(monkeypatch, tmp_path)
    (src / SHARD).write_bytes(b"g" * 64)
    (dest / SHARD).write_bytes(b"w" * 96)
    req = {"src": str(src), "dest": str(dest), "name": SHARD}
    # Without exact (a normal copy) the bigger file is left alone, as before.
    loose = agent._stage_shard(dict(req))
    assert loose["mode"] == "resume" and "--append-verify" in runs[-1]
    assert (dest / SHARD).read_bytes() == b"w" * 96
    fixed = agent._stage_shard(dict(req, exact=True))
    assert fixed["ok"] and fixed["mode"] == "replace", fixed
    assert runs[-1] == ["rsync", "-a", str(src / SHARD), str(dest / SHARD)]
    assert (dest / SHARD).read_bytes() == b"g" * 64


def test_refill_same_size_shard_is_not_skipped_by_append(monkeypatch, tmp_path) -> None:
    agent, src, dest, runs = _agent_dirs(monkeypatch, tmp_path)
    (src / SHARD).write_bytes(b"g" * 64)
    (dest / SHARD).write_bytes(b"w" * 64)
    out = agent._stage_shard({"src": str(src), "dest": str(dest), "name": SHARD, "exact": True})
    # Plain rsync -a: its size and time check re-sends a same-size shard whose time differs.
    assert out["mode"] == "replace" and "--append-verify" not in runs[-1], runs
    assert (dest / SHARD).read_bytes() == b"g" * 64


def test_refill_resumes_a_partial_shard(monkeypatch, tmp_path) -> None:
    agent, src, dest, runs = _agent_dirs(monkeypatch, tmp_path)
    (src / SHARD).write_bytes(b"g" * 64)
    (dest / SHARD).write_bytes(b"g" * 10)
    out = agent._stage_shard({"src": str(src), "dest": str(dest), "name": SHARD, "exact": True})
    assert out["mode"] == "resume" and runs[-1][:4] == ["rsync", "-a", "--partial", "--append-verify"]
    missing = agent._stage_shard({"src": str(src), "dest": str(dest), "name": SHARD.replace("00001-of", "00002-of"), "exact": True})
    assert missing["ok"] is False


def test_refill_index_and_config_compared_and_replaced(monkeypatch, tmp_path) -> None:
    agent, src, dest, runs = _agent_dirs(monkeypatch, tmp_path)
    (src / SHARD).write_bytes(b"g" * 64)
    (dest / SHARD).write_bytes(b"g" * 64)
    (src / "config.json").write_text('{"a": 1}')
    (dest / "config.json").write_text('{"a": 12}')
    (src / "model.safetensors.index.json").write_text('{"x": 1}')
    (dest / "model.safetensors.index.json").write_text('{"y": 2}')  # same size, different content
    listing = agent._list_shards({"src": str(src), "dest": str(dest), "shards": 1, "exact": True})
    assert listing["pending"] == [] and listing["files_differ"] == ["config.json"], listing
    assert agent._list_shards({"src": str(src), "dest": str(dest), "shards": 1})["files_differ"] == []
    for name in ("config.json", "model.safetensors.index.json"):
        out = agent._stage_shard({"src": str(src), "dest": str(dest), "name": name, "exact": True, "min_free_ratio": 0.0})
        assert out["mode"] == "replace" and runs[-1] == ["rsync", "-a", "--checksum", str(src / name), str(dest / name)]
        assert (dest / name).read_text() == (src / name).read_text()


def test_sync_check_reports_index_or_config_that_differs(monkeypatch) -> None:
    class FilesRunner(Runner):
        def invoke(self, host, req, timeout=30):
            self.calls.append((host, dict(req)))
            return {"ok": True, "total": 3, "done": ["s1", "s2", "s3"], "pending": [], "files_differ": ["config.json"]}

    rows = _report(monkeypatch, FilesRunner([]))
    cell = rows["flat-x"]["hosts"]["spark-a"]
    assert cell["state"] == "differs" and cell["note"] == "config.json differ" and cell["can_copy"] is True, cell


class CopyRunner:
    def __init__(self, after: dict) -> None:
        self.after = after
        self.calls: list[dict] = []
        self.lists = 0

    def invoke(self, host, req, timeout=30):
        self.calls.append(dict(req))
        if req["op"] == "list_shards":
            self.lists += 1
            if self.lists == 1:
                return {"ok": True, "total": 2, "done": ["s1"], "pending": ["s2"], "files_differ": ["config.json"]}
            return dict({"ok": True, "total": 2, "done": ["s1", "s2"], "pending": [], "files_differ": []}, **self.after)
        if req["op"] == "stage_shard":
            return {"ok": True, "name": req["name"]} if req["name"] != "model.safetensors.index.json" else {"ok": False, "error": "missing shard model.safetensors.index.json"}
        return {"ok": False, "error": "unexpected"}


def _copy(monkeypatch, runner, exact=True):
    import sparksi.jobs as jobs

    finished: list[tuple] = []
    monkeypatch.setattr(jobs, "_set", lambda **kw: None)
    monkeypatch.setattr(jobs, "_wait_if_paused", lambda: True)
    monkeypatch.setattr(jobs, "_finish", lambda ok, error=None, **kw: finished.append((ok, error)))
    plan = {"host": "spark-a", "request": {"op": "stage_flat", "src": "/archive/flat-x", "dest": "/srv/models/a/flat-x", "shards": 2, "exact": exact}}
    return jobs._copy_shards(runner, plan), finished


def test_refill_passes_exact_and_rechecks(monkeypatch) -> None:
    runner = CopyRunner({})
    ok, finished = _copy(monkeypatch, runner)
    assert ok is True and finished == [], finished
    stages = [c for c in runner.calls if c["op"] == "stage_shard"]
    assert [c["name"] for c in stages] == ["s2", "model.safetensors.index.json", "config.json"]
    assert all(c["exact"] is True for c in stages), stages
    assert [c.get("exact") for c in runner.calls if c["op"] == "list_shards"] == [True, True]


def test_refill_fails_when_the_recheck_still_differs(monkeypatch) -> None:
    runner = CopyRunner({"pending": ["s2"], "done": ["s1"], "files_differ": ["config.json"]})
    ok, finished = _copy(monkeypatch, runner)
    assert ok is False
    assert finished == [(False, "2 file(s) still differ from storage after the copy: s2, config.json")], finished


def test_normal_copy_does_not_recheck(monkeypatch) -> None:
    runner = CopyRunner({"pending": ["s2"]})
    ok, _ = _copy(monkeypatch, runner, exact=False)
    assert ok is True and runner.lists == 1
    assert all(c["exact"] is False for c in runner.calls if c["op"] == "stage_shard")
