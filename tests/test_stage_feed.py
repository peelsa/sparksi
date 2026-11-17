# SPDX-License-Identifier: Apache-2.0

"""The stage feed is a loopback reader. It does not write to the live state."""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from sparksi.history import History, sample_card
from sparksi.lock import state_dir
from sparksi.serve import LISTENERS, STATE, collector_card, serve
from sparksi.stage_feed import (
    FRESH_S,
    Feed,
    _STOP_POLL,
    _get_json,
    isolated_state,
    load_run_snapshot,
    loopback_upstream,
    production_writer_roots,
    run,
    scrub_env,
)


def test_stage_refuses_the_live_state_and_a_remote_upstream(tmp_path) -> None:
    try:
        isolated_state(state_dir())
        raise AssertionError("live state should be refused")
    except SystemExit as exc:
        assert "separate" in str(exc)
    isolated_state(tmp_path / "stage")
    try:
        loopback_upstream("http://example.com")
        raise AssertionError("remote upstream should be refused")
    except SystemExit as exc:
        assert "loopback" in str(exc)


def test_monitor_hook_does_not_feed_the_collector() -> None:
    STATE.monitor_fn = lambda: {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 1}], "loaded": []}
    STATE.observe_fn = None
    try:
        assert collector_card().get("hosts", [{}])[0].get("gpu_pct") != 1
        STATE.observe_fn = lambda: {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 9}], "loaded": []}
        assert collector_card()["hosts"][0]["gpu_pct"] == 9
    finally:
        STATE.monitor_fn = None
        STATE.observe_fn = None


def test_stage_copies_monitor_and_runs_with_get_only(tmp_path) -> None:
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            seen.append(self.command + " " + self.path)
            if self.path.startswith("/api/monitor"):
                body = {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 11, "cpu_pct": 3}], "loaded": []}
            elif self.path.startswith("/api/runs"):
                body = {"runs": [{"id": "run-1", "model_id": "example", "phase": "ready", "placement": "head", "port": 8014, "nodes": [{"role": "head", "phase": "ready"}]}]}
            else:
                self.send_error(404)
                return
            raw = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:  # noqa: N802
            seen.append("POST " + self.path)
            self.send_error(405)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=upstream.serve_forever, daemon=True).start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    (tmp_path / "runs.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "id": "run-1",
                        "model_id": "example",
                        "phase": "ready",
                        "placement": "head",
                        "port": 8014,
                        "host": "hidden",
                        "nodes": [{"role": "head", "phase": "ready", "host": "hidden", "log_tail": "secret"}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    live = state_dir()
    live_before = (live / "history.json").exists()
    holder: dict = {}
    import sparksi.serve as serve_mod

    calls: list[str] = []
    original_restore = serve_mod.restore_job

    def record_restore(*_args, **_kwargs) -> None:
        calls.append("restore")

    serve_mod.restore_job = record_restore

    def serve() -> None:
        try:
            run(upstream_url, tmp_path / "stage", port=7612, runs_snapshot=tmp_path / "runs.json")
        except Exception as exc:  # noqa: BLE001 — surfaced below
            holder["error"] = exc

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 20
        samples = []
        while time.time() < deadline:
            if holder.get("error"):
                raise holder["error"]
            try:
                with urlopen("http://127.0.0.1:7612/api/monitor", timeout=2) as response:
                    monitor = json.loads(response.read().decode())
                with urlopen("http://127.0.0.1:7612/api/runs", timeout=2) as response:
                    runs = json.loads(response.read().decode())
                with urlopen("http://127.0.0.1:7612/api/history?machine=head&range=1h", timeout=2) as response:
                    history = json.loads(response.read().decode())
            except OSError:
                time.sleep(0.2)
                continue
            samples = history.get("samples") or []
            if monitor.get("hosts") and runs.get("runs") and samples:
                break
            time.sleep(0.3)
        assert monitor["hosts"][0]["gpu_pct"] == 11
        assert runs["stale"] is True
        assert runs["source"] == "snapshot"
        assert runs["runs"][0]["id"] == "run-1"
        assert "host" not in runs["runs"][0]
        assert "log_tail" not in json.dumps(runs["runs"][0])
        assert samples[-1]["gpu_pct"] == 11
        (tmp_path / "stage" / "job.json").write_text('{"resume": true, "action": "start"}', encoding="utf-8")
        denied = Request(
            "http://127.0.0.1:7612/api/unlock",
            data=b'{"from_ui":true}',
            method="POST",
        )
        try:
            urlopen(denied, timeout=2)
            raise AssertionError("unlock should be refused")
        except HTTPError as exc:
            assert exc.code == 403
        assert calls == []
        assert seen
        assert all(item.startswith("GET /api/monitor") for item in seen)
        assert all("/api/runs" not in item for item in seen)
        assert (tmp_path / "stage" / "history.json").is_file()
        assert (live / "history.json").exists() is live_before
    finally:
        serve_mod.restore_job = original_restore
        _STOP_POLL.set()
        for server in list(LISTENERS):
            server.shutdown()
        upstream.shutdown()
        STATE.monitor_fn = None
        STATE.observe_fn = None
        STATE.runs_fn = None
        STATE.readonly = False
        os.environ.pop("SPARKSI_STATE_DIR", None)
        os.environ.pop("SPARKSI_LOCK", None)
        from sparksi.history import reset_for_tests

        reset_for_tests()
        thread.join(timeout=5)


def test_reused_and_linked_state_is_refused_and_writers_stay_inside(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("SPARKSI_STATE_DIR", raising=False)
    monkeypatch.delenv("DGX_MEM_STATE_DIR", raising=False)
    monkeypatch.delenv("SPARKSI_LOCK", raising=False)
    monkeypatch.delenv("DGX_MEM_LOCK", raising=False)
    durable = Path.home() / ".local" / "state" / "sparksi"
    durable.mkdir(parents=True, exist_ok=True)
    (durable / "job.json").write_text('{"resume": true}', encoding="utf-8")
    runtime = Path(os.environ["XDG_RUNTIME_DIR"]) / "sparksi"
    before = production_writer_roots()
    assert any(item == durable.resolve() for item in before)
    assert any(item == runtime.resolve() for item in before)
    for index, target in enumerate((durable, runtime, state_dir(), durable / "telemetry")):
        link = tmp_path / f"link-{index}"
        link.symlink_to(target, target_is_directory=True)
        with pytest.raises(SystemExit, match="separate"):
            isolated_state(link)
    used = tmp_path / "used"
    used.mkdir()
    (used / "job.json").write_text('{"resume": true}', encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        isolated_state(used)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert isolated_state(empty) == empty.resolve()
    stage = isolated_state(tmp_path / "stage")
    scrub_env(stage)
    from sparksi.activity import activity_path
    from sparksi.history import history_path
    from sparksi.jobs import job_path
    from sparksi.ledger import ledger_path as counter_ledger
    from sparksi.llama_day import ledger_path as llama_ledger
    from sparksi.lock import default_lock_path
    from sparksi.recipes.runs import runs_dir
    from sparksi.tokens import ledger_path as token_ledger

    writers = {
        "history": history_path(),
        "activity": activity_path(),
        "job": job_path(),
        "ledger": counter_ledger(),
        "llama": llama_ledger(),
        "tokens": token_ledger(),
        "lock": default_lock_path(),
        "runs": runs_dir(),
        "telemetry": state_dir() / "telemetry",
        "state": state_dir(),
    }
    stage_root = stage.resolve()
    for name, path in writers.items():
        resolved = path.expanduser().resolve()
        assert resolved == stage_root or stage_root in resolved.parents, name
        for live in before:
            assert resolved != live, name
            assert live not in resolved.parents, name
            assert resolved not in live.parents, name


def test_readonly_serve_does_not_resume_a_copied_job(tmp_path, monkeypatch) -> None:
    calls: list[str] = []
    state = tmp_path / "copied"
    state.mkdir()
    (state / "job.json").write_text('{"resume": true, "action": "start"}', encoding="utf-8")
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(state))
    monkeypatch.setenv("SPARKSI_LOCK", str(state / "lock"))

    class Runner:
        def __getattr__(self, name: str):
            def _call(*_args, **_kwargs) -> None:
                calls.append(name)

            return _call

    STATE.runner = Runner()
    monkeypatch.setattr("sparksi.serve.restore_job", lambda *_a, **_k: calls.append("restore"))
    monkeypatch.setattr("sparksi.serve.poll_once", lambda: calls.append("poll"))
    monkeypatch.setattr("sparksi.serve.list_models", lambda *_a, **_k: calls.append("warm"))
    monkeypatch.setattr("sparksi.lock.Lock.release", lambda *_a, **_k: calls.append("unlock"))
    monkeypatch.setattr("sparksi.nodes.SshRunner", lambda *_a, **_k: calls.append("ssh"))
    monkeypatch.setattr("sparksi.library.SshRunner", lambda *_a, **_k: calls.append("ssh"))
    monkeypatch.setattr("sparksi.serve.runs_index", lambda *_a, **_k: calls.append("adopt"))

    def _serve() -> None:
        serve(port=7623, hosts=["127.0.0.1"], announce=False, readonly=True)

    thread = threading.Thread(target=_serve, daemon=True)
    thread.start()
    try:
        deadline = time.time() + 15
        ready = False
        while time.time() < deadline:
            try:
                with urlopen("http://127.0.0.1:7623/api/job", timeout=1) as response:
                    ready = response.status == 200
                    break
            except (OSError, HTTPError):
                time.sleep(0.05)
        assert ready, thread.is_alive()
        for method, path in (
            ("POST", "/api/unlock"),
            ("POST", "/api/lock"),
            ("POST", "/api/runs/stop"),
            ("PUT", "/api/models/example/recipe"),
            ("DELETE", "/api/runs/example"),
        ):
            request = Request(
                f"http://127.0.0.1:7623{path}",
                data=b'{"from_ui":true}',
                method=method,
            )
            try:
                urlopen(request, timeout=2)
                raise AssertionError(method + " " + path)
            except HTTPError as exc:
                assert exc.code == 403, path
        for path in ("/api/nodes", "/api/fleet", "/api/runs", "/api/monitor"):
            with urlopen(f"http://127.0.0.1:7623{path}", timeout=2) as response:
                body = json.loads(response.read().decode())
            if path == "/api/runs":
                assert body["stale"] is True
                assert body["runs"] == []
            if path == "/api/nodes":
                assert body["nodes"] == []
        time.sleep(0.3)
        assert calls == []
        assert (state / "job.json").read_text(encoding="utf-8").startswith("{")
    finally:
        _STOP_POLL.set()
        for server in list(LISTENERS):
            server.shutdown()
        thread.join(timeout=5)
        LISTENERS.clear()
        STATE.runner = None
        STATE.readonly = False
        from sparksi.history import reset_for_tests

        reset_for_tests()


def test_stale_poll_is_not_archived_as_fresh(tmp_path) -> None:
    hist = History(tmp_path / "history.json")
    feed = Feed()
    card = {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 11}], "loaded": []}
    feed.note_success(card, {"runs": []}, now=10)
    sample_card(feed.current_card(now=11), hist, None, now=1_700_000_000)
    sample_card(feed.current_card(now=10 + FRESH_S + 1), hist, None, now=1_700_000_040)
    feed.note_failure()
    sample_card(feed.current_card(now=12), hist, None, now=1_700_000_050)
    sample_card(feed.current_card(now=10 + FRESH_S + 5), hist, None, now=1_700_000_080)
    rows = hist.query("head", "1h", now=1_700_000_100)["samples"]
    assert [row["t"] for row in rows] == [1_700_000_000]
    assert rows[0]["gpu_pct"] == 11
    feed.note_success(
        {"hosts": [{"id": "head", "fresh": True, "gpu_pct": 40}], "loaded": []},
        {"runs": []},
        now=30,
    )
    sample_card(feed.current_card(now=31), hist, None, now=1_700_000_090)
    again = hist.query("head", "1h", now=1_700_000_100)["samples"]
    assert [row["gpu_pct"] for row in again] == [11, 40]


def test_feed_refuses_redirects_and_proxies() -> None:
    hits: list[str] = []

    class Target(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.path)
            raw = b'{"hosts":[],"loaded":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    class Bounce(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{target.server_address[1]}/api/monitor")
            self.end_headers()

    class Proxy(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            hits.append("proxy")
            self.send_error(500)

    target = ThreadingHTTPServer(("127.0.0.1", 0), Target)
    bounce = ThreadingHTTPServer(("127.0.0.1", 0), Bounce)
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), Proxy)
    for server in (target, bounce, proxy):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    previous = {key: os.environ.get(key) for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "all_proxy")}
    os.environ["http_proxy"] = f"http://127.0.0.1:{proxy.server_address[1]}"
    os.environ["HTTP_PROXY"] = os.environ["http_proxy"]
    os.environ["https_proxy"] = os.environ["http_proxy"]
    os.environ["all_proxy"] = os.environ["http_proxy"]
    try:
        with pytest.raises(URLError):
            _get_json(f"http://127.0.0.1:{bounce.server_address[1]}", "/api/monitor")
        assert hits == []
        direct = ThreadingHTTPServer(("127.0.0.1", 0), Target)
        threading.Thread(target=direct.serve_forever, daemon=True).start()
        try:
            body = _get_json(f"http://127.0.0.1:{direct.server_address[1]}", "/api/monitor")
        finally:
            direct.shutdown()
        assert body == {"hosts": [], "loaded": []}
        assert "proxy" not in hits
    finally:
        for server in (target, bounce, proxy):
            server.shutdown()
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_run_snapshot_is_local_and_the_live_runs_path_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError):
        _get_json("http://127.0.0.1:9", "/api/runs")
    durable = Path.home() / ".local" / "state" / "sparksi"
    durable.mkdir(parents=True, exist_ok=True)
    planted = durable / "job.json"
    planted.write_text("{}", encoding="utf-8")
    link = tmp_path / "linked-job"
    link.symlink_to(planted)
    roots = production_writer_roots()
    with pytest.raises(SystemExit, match="live controller"):
        load_run_snapshot(link, roots)
    with pytest.raises(SystemExit, match="live controller"):
        load_run_snapshot(planted, roots)
    snapshot = tmp_path / "runs.json"
    snapshot.write_text(
        json.dumps({"runs": [{"id": "run-1", "phase": "ready", "token": "secret", "nodes": [{"role": "head", "phase": "ready", "container": "box"}]}]}),
        encoding="utf-8",
    )
    loaded = load_run_snapshot(snapshot, roots)
    assert loaded["stale"] is True
    assert loaded["source"] == "snapshot"
    assert loaded["runs"] == [{"id": "run-1", "phase": "ready", "nodes": [{"role": "head", "phase": "ready"}]}]
    feed = Feed()
    feed.runs = loaded
    feed.note_failure()
    assert feed.current_runs()["runs"][0]["id"] == "run-1"
    assert feed.current_runs()["stale"] is True


def test_monitor_card_does_not_adopt_or_write(tmp_path, monkeypatch) -> None:
    def refuse(*_args, **_kwargs):
        raise AssertionError("adopt")

    monkeypatch.setattr("sparksi.recipes.adopt.adopt_running", refuse)

    class Peer:
        def snapshot(self):
            return (
                {
                    "mem": (128.0, 64.0),
                    "gpu": [10, 40, 1000],
                    "cpu_pct": 3,
                    "cpu_mhz": 1000,
                    "zones": [],
                    "http": {},
                    "containers": [],
                    "engines": [],
                    "comfy": {},
                    "truth": "",
                },
                1,
            )

    from sparksi.dash import monitor_card

    home = Path.home() / ".local" / "state"
    before = sorted(path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file())
    card = monitor_card(Peer(), Peer())
    after = sorted(path.relative_to(home).as_posix() for path in home.rglob("*") if path.is_file())
    assert card["hosts"]
    assert after == before
    assert not list(tmp_path.iterdir())
