# SPDX-License-Identifier: Apache-2.0

"""Golden argv, validation, placement, the recipe API, and pair stop."""

from __future__ import annotations

import json
import threading
import time
from urllib.request import Request, urlopen

import pytest

from sparksi.activity import reset_for_tests as reset_activity
from sparksi.agent import main as agent_main
from sparksi.lock import Lock
from sparksi.models import ModelCard, build_cards, derive_placements
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.engines.custom import build as build_custom
from sparksi.recipes.engines.llamacpp import build as build_llamacpp
from sparksi.recipes.engines.sglang import build as build_sglang
from sparksi.recipes.engines.trtllm import EXPERIMENTAL_BOTH
from sparksi.recipes.engines.trtllm import build as build_trtllm
from sparksi.recipes.engines.vllm import build as build_vllm
from sparksi.recipes.runs import reset_for_tests as reset_runs
from sparksi.recipes.runs import tick
from sparksi.recipes.spec import Spec
from sparksi.recipes.store import load_recipe, save_recipe
from sparksi.recipes.validate import check_port, memory_conflict, validate_fields
from sparksi.serve import STATE, DualServer, Handler

OFF_LIMITS_PORT = 8188  # the video-service protect preset port (on in tests via conftest)

HEAD = "spark-a"
WORKER = "spark-b"
IMAGE = "nvcr.io/nvidia/vllm:26.05-py3"
SGLANG = "nvcr.io/nvidia/sglang:26.02-py3"
TRT = "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"
ALLOWED = [IMAGE, SGLANG, TRT]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "recipes"))
    monkeypatch.setenv("SPARKSI_CARDS_DIR", str(tmp_path / "cards"))
    reset_runs()
    reset_activity()
    reset_cache_for_tests()
    yield
    reset_runs()
    reset_cache_for_tests()
    STATE.runner = None


def _spec(**over) -> Spec:
    base = dict(
        model_id="demo",
        placement="head",
        port=8010,
        paths={"head": "/models/demo", "worker": "/models/demo"},
        hosts={"head": HEAD, "worker": WORKER},
        binds={"head": "100.64.0.11", "worker": "100.64.0.12"},
        fabric_ip={"head": "192.0.2.11", "worker": "192.0.2.12"},
        fabric_iface={"head": "enp1s0f1np1", "worker": "enp1s0f1np1"},
        fabric_hcas={"head": ["roce0", "roce1"], "worker": ["roce0", "roce1"]},
        image=IMAGE,
        hf_cache="/home/USER/.cache/huggingface",
        ctx=8192,
        hints={"mtp": False},
        scripts={},
    )
    base.update(over)
    return Spec(**base)


def _vllm_single() -> list[str]:
    return [
        "docker", "run", "-d",
        "--name", "sparksi-demo-8010",
        "--gpus", "all",
        "--ipc", "host",
        "--network", "host",
        "--ulimit", "memlock=-1",
        "--shm-size", "16g",
        "-e", "HF_HUB_OFFLINE=1",
        "-v", "/home/USER/.cache/huggingface:/root/.cache/huggingface",
        IMAGE,
        "vllm", "serve", "/models/demo",
        "--served-model-name", "demo",
        "--host", "100.64.0.11",
        "--port", "8010",
        "--gpu-memory-utilization", "0.80",
        "--max-model-len", "8192",
        "--max-num-seqs", "4",
        "--kv-cache-dtype", "fp8",
        "--enable-chunked-prefill",
    ]


def test_vllm_single_argv_is_golden() -> None:
    built = build_vllm(_spec())
    assert built.experimental is False
    assert built.nodes[0].argv == _vllm_single()
    assert built.nodes[0].stop_argv == ["docker", "rm", "-f", "sparksi-demo-8010"]
    assert "0.0.0.0" not in built.nodes[0].argv


def test_vllm_kv_bytes_replace_utilization_and_prefix_follows_the_card() -> None:
    built = build_vllm(_spec(kv_cache_memory_bytes=4096, prefix_cache=True))
    argv = built.nodes[0].argv
    assert "--kv-cache-memory-bytes" in argv
    assert "4096" in argv
    assert "--gpu-memory-utilization" not in argv
    assert "--enable-prefix-caching" in argv
    refused = build_vllm(_spec(prefix_cache=True, forbid_prefix=True))
    assert "--enable-prefix-caching" not in refused.nodes[0].argv


def test_vllm_two_node_argv_uses_fabric_and_starts_the_worker_first() -> None:
    built = build_vllm(_spec(placement="both"))
    assert [node.role for node in built.nodes] == ["worker", "head"]
    assert built.order == ["worker", "head"]
    worker, head = built.nodes
    assert worker.rank == 1 and head.rank == 0
    for node in built.nodes:
        argv = node.argv
        assert argv[argv.index("--tensor-parallel-size") + 1] == "2"
        assert argv[argv.index("--nnodes") + 1] == "2"
        assert argv[argv.index("--master-addr") + 1] == "192.0.2.11"
        assert argv[argv.index("--master-port") + 1] == "29501"
        assert "NCCL_SOCKET_IFNAME=enp1s0f1np1" in argv
        assert "GLOO_SOCKET_IFNAME=enp1s0f1np1" in argv
        assert "TP_SOCKET_IFNAME=enp1s0f1np1" in argv
        assert "UCX_NET_DEVICES=enp1s0f1np1" in argv
        assert "NCCL_IB_HCA=roce0,roce1" in argv
        assert "NCCL_IB_DISABLE=0" in argv
        assert "NCCL_IGNORE_CPU_AFFINITY=1" in argv
        assert "0.0.0.0" not in argv
        assert isinstance(argv, list)
    assert worker.argv[worker.argv.index("--node-rank") + 1] == "1"
    assert head.argv[head.argv.index("--node-rank") + 1] == "0"
    assert worker.argv[-1] == "--headless"
    assert "--headless" not in head.argv


def test_shell_metacharacters_stay_one_argv_element() -> None:
    path = "/models/demo;rm -rf /"
    built = build_vllm(_spec(paths={"head": path, "worker": path}))
    argv = built.nodes[0].argv
    assert isinstance(argv, list)
    assert path in argv
    assert argv.count(path) == 1
    assert "sh" not in argv
    assert "-c" not in argv


def test_sglang_single_and_two_node() -> None:
    single = build_sglang(_spec(image=SGLANG))
    argv = single.nodes[0].argv
    assert argv[argv.index(SGLANG) + 1 : argv.index(SGLANG) + 4] == ["python", "-m", "sglang.launch_server"]
    assert "--model-path" in argv and "--mem-fraction-static" in argv
    assert argv[argv.index("--mem-fraction-static") + 1] == "0.80"
    assert argv[argv.index("--context-length") + 1] == "8192"
    both = build_sglang(_spec(image=SGLANG, placement="both"))
    assert [node.role for node in both.nodes] == ["worker", "head"]
    worker = both.nodes[0].argv
    assert worker[worker.index("--tp") + 1] == "2"
    assert worker[worker.index("--dist-init-addr") + 1] == "192.0.2.11:50000"
    assert worker[worker.index("--node-rank") + 1] == "1"
    assert "NCCL_IB_HCA=roce0,roce1" in worker
    assert both.experimental is False


def test_trtllm_single_is_complete_and_two_node_is_experimental() -> None:
    single = build_trtllm(_spec(image=TRT))
    argv = single.nodes[0].argv
    assert single.experimental is False
    assert argv[argv.index(TRT) + 1] == "trtllm-serve"
    assert "--backend" in argv and argv[argv.index("--backend") + 1] == "pytorch"
    assert argv[argv.index("--kv_cache_free_gpu_memory_fraction") + 1] == "0.80"
    assert "--max_seq_len" in argv
    both = build_trtllm(_spec(image=TRT, placement="both"))
    assert both.experimental is True
    assert "sshd" in both.experimental_reason
    assert both.experimental_reason == EXPERIMENTAL_BOTH
    head = next(node for node in both.nodes if node.role == "head").argv
    assert "mpirun" in head
    assert "trtllm-llmapi-launch" in head
    assert "trtllm-serve" in head
    assert head[head.index("--tp_size") + 1] == "2"
    assert "192.0.2.11:1,192.0.2.12:1" in head
    assert "OMPI_MCA_btl_tcp_if_include=enp1s0f1np1" in head
    worker = next(node for node in both.nodes if node.role == "worker").argv
    assert worker[-2:] == ["sleep", "infinity"]
    assert [node.role for node in both.nodes] == ["worker", "head"]


def test_llamacpp_single_and_mtp_and_never_two_node() -> None:
    argv = build_llamacpp(_spec(paths={"head": "/models/demo.gguf"})).nodes[0].argv
    assert argv[:4] == ["llama-server", "-m", "/models/demo.gguf", "--host"]
    assert argv[argv.index("--host") + 1] == "100.64.0.11"
    assert "-ngl" in argv and argv[argv.index("-ngl") + 1] == "99"
    assert argv[argv.index("-ctk") + 1] == "q8_0"
    assert "--jinja" in argv
    assert "--spec-type" not in argv
    mtp = build_llamacpp(_spec(hints={"mtp": True})).nodes[0].argv
    assert mtp[mtp.index("--spec-type") + 1] == "draft-mtp"
    with pytest.raises(ValueError):
        build_llamacpp(_spec(placement="both"))


def test_custom_env_is_only_port_and_host() -> None:
    spec = _spec(scripts={HEAD: {"start": ["/home/USER/services/tool/start.sh"], "stop": ["/home/USER/services/tool/stop.sh"]}})
    built = build_custom(spec)
    assert built.nodes[0].argv == ["/home/USER/services/tool/start.sh"]
    assert built.nodes[0].env == {"PORT": "8010", "HOST": "100.64.0.11"}
    assert built.nodes[0].stop_argv == ["/home/USER/services/tool/stop.sh"]
    assert built.nodes[0].container == ""


def test_port_skips_reserved_listening_and_other_recipes(tmp_path, monkeypatch) -> None:
    from sparksi.recipes.service import _choose, _reserved_ports
    from sparksi.recipes.store import save_recipe
    from sparksi.catalog import load_catalog

    catalog = load_catalog()
    reserved = _reserved_ports(catalog)
    assert 8011 in reserved
    assert 8188 in reserved
    card = build_cards(catalog, presence={row["id"]: {host: True for host in (row.get("local") or {})} for row in catalog["library"]}, probe=False)
    single = next(item for item in card if item.id == "single-gguf")
    listening = {HEAD: {8010}, WORKER: {8012}}
    _engine, placement, port, conflicts = _choose(single, {"engine": "llamacpp"}, listening, set(), catalog)
    assert conflicts == []
    assert placement == "head"
    assert port == 8014
    assert port != 8011
    listening[HEAD].add(8014)
    busy = _choose(single, {"engine": "llamacpp"}, listening, set(), catalog)
    assert busy[2] == 8013
    listening[HEAD].discard(8014)
    listening[HEAD].add(8015)
    taken = _choose(single, {"engine": "llamacpp", "port": 8015}, listening, {"port"}, catalog)
    listening[HEAD].discard(8015)
    assert any("already listening on head" in item for item in taken[3])
    assert any("8011" in item and "reserved" in item for item in _choose(single, {"engine": "llamacpp", "port": 8011}, listening, {"port"}, catalog)[3])
    save_recipe("pair-fast", {"engine": "vllm", "placement": "both", "port": 8013})
    _engine, _placement, port, conflicts = _choose(single, {"engine": "llamacpp"}, listening, set(), catalog)
    assert port == 8014
    assert conflicts == []
    shared = _choose(single, {"engine": "llamacpp", "port": 8013}, listening, {"port"}, catalog)
    assert not any("another recipe" in item for item in shared[3])
    assert shared[3] == []
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "own"))
    save_recipe("single-gguf", {"engine": "llamacpp", "placement": "head", "port": 8014})
    saved = load_recipe("single-gguf")
    _engine, _placement, port, conflicts = _choose(single, saved, listening, set(), catalog)
    assert port == 8014
    assert conflicts == []


def test_speculative_and_parsers_are_optional_argv() -> None:
    plain = build_vllm(_spec())
    assert "--speculative-config" not in plain.nodes[0].argv
    assert "--tool-call-parser" not in plain.nodes[0].argv
    built = build_vllm(_spec(speculative=True, tool_parser="hermes", reasoning_parser="qwen3"))
    argv = built.nodes[0].argv
    assert argv[argv.index("--speculative-config") + 1] == '{"method":"mtp","num_speculative_tokens":1}'
    assert argv[argv.index("--tool-call-parser") + 1] == "hermes"
    assert argv[argv.index("--reasoning-parser") + 1] == "qwen3"
    sglang = build_sglang(_spec(image=SGLANG, speculative=True, tool_parser="hermes")).nodes[0].argv
    assert sglang[sglang.index("--speculative-algorithm") + 1] == "NEXTN"
    assert sglang[sglang.index("--tool-call-parser") + 1] == "hermes"
    off = build_llamacpp(_spec(hints={"mtp": True}, speculative=False)).nodes[0].argv
    assert "--spec-type" not in off
    cleaned, bad = validate_fields("vllm", {"tool_parser": "nope", "speculative": "maybe"}, allowed_images=ALLOWED)
    assert cleaned.get("tool_parser") is None
    assert any("tool_parser" in item for item in bad)
    assert any("speculative" in item for item in bad)


def test_validation_rejects_ports_flags_and_budget() -> None:
    assert any("8188" in item for item in check_port(8188))
    assert any("outside" in item for item in check_port(8009))
    assert any("outside" in item for item in check_port(8100))
    assert check_port(8010) == []
    assert check_port(OFF_LIMITS_PORT)[0].startswith("port 8188")
    conflict = check_port(8012, {"spark-a": {8012}}, ["spark-a"])
    assert any("already listening" in item for item in conflict)
    cleaned, bad = validate_fields("vllm", {"extra_flags": ["--not-a-real-flag"]}, allowed_images=ALLOWED)
    assert cleaned.get("extra_flags") is None
    assert any("unknown flag" in item for item in bad)
    cleaned, bad = validate_fields("vllm", {"extra_flags": [";rm -rf /"]}, allowed_images=ALLOWED)
    assert any("unknown flag" in item for item in bad)
    _, bad = validate_fields("vllm", {"gpu_mem_util": 0.99, "ctx": 10, "image": "nvcr.io/nvidia/vllm:latest", "nope": 1}, allowed_images=ALLOWED)
    assert any("gpu_mem_util" in item for item in bad)
    assert any("ctx" in item for item in bad)
    assert any("pinned" in item or "allowed" in item for item in bad)
    assert any("unknown field" in item for item in bad)
    over = memory_conflict(124, 8, "head", {"head": 111, "worker": 111})
    assert over and "over budget" in over[0]
    assert memory_conflict(40, 8, "head", {"head": 111, "worker": 111}) == []


def test_placement_reasons() -> None:
    wide = ModelCard(
        id="wide", title="Wide", format="nvfp4", weights_gib=124, engines=["vllm", "sglang", "trtllm"],
        files={HEAD: "/w", WORKER: "/w"}, presence={HEAD: True, WORKER: True},
    )
    got = derive_placements(wide, {"head": 111, "worker": 111}, wide.presence)
    assert got["head"]["allowed"] is False
    assert "132" in got["head"]["reason"] and "111" in got["head"]["reason"]
    assert got["worker"]["allowed"] is False
    assert got["both"]["allowed"] is True
    assert got["both"]["reason"] == ""

    small = ModelCard(
        id="small", title="Small", format="gguf", weights_gib=8, engines=["llamacpp"],
        files={HEAD: "/g"}, presence={HEAD: True},
    )
    got = derive_placements(small, {"head": 111, "worker": 111}, small.presence)
    assert got["head"]["allowed"] is True
    assert got["worker"]["reason"] == "weights are not on the worker"
    assert got["both"]["allowed"] is False
    assert got["both"]["reason"] == "llamacpp runs on one Spark"

    fits = ModelCard(
        id="fits", title="Fits", format="bf16", weights_gib=40, engines=["vllm"],
        files={HEAD: "/a"}, presence={HEAD: True},
    )
    got = derive_placements(fits, {"head": 111, "worker": 111}, fits.presence)
    assert got["both"]["reason"] == "this model fits on one Spark"


def test_example_catalog_placements_with_injected_presence() -> None:
    from sparksi.catalog import load_catalog

    catalog = load_catalog()
    presence = {}
    for row in catalog["library"]:
        presence[row["id"]] = {host: True for host in (row.get("local") or {})}
    cards = {card.id: card for card in build_cards(catalog, presence=presence, probe=False)}
    assert cards["pair-fast"].placements["both"]["allowed"] is True
    assert cards["pair-fast"].placements["head"]["allowed"] is False
    assert "llamacpp" in cards["single-gguf"].engines
    assert cards["single-gguf"].placements["both"]["reason"] == "llamacpp runs on one Spark"
    assert cards["single-gguf"].placements["head"]["allowed"] is True
    assert cards["video-seat"].engines == ["custom"]


def _free() -> str:
    gib = 1024**3
    total, used, free, avail = 121 * gib, 20 * gib, 80 * gib, 90 * gib
    cache = total - used - free
    return (
        "               total        used        free      shared  buff/cache   available\n"
        f"Mem:     {total} {used} {free} 0 {cache} {avail}\n"
        "Swap: 0 0 0\n"
    )


class Fake:
    def __init__(self) -> None:
        self.started: list[tuple[str, list[str]]] = []
        self.stopped: list[tuple[str, list[str]]] = []
        self.listen: dict[str, set[int]] = {HEAD: set(), WORKER: set()}
        self.missing_image = False
        self.health = True

    def capture(self, host, argv, timeout=8):
        prog = argv[0]
        if prog == "free":
            return {"ok": True, "code": 0, "stdout": _free(), "stderr": ""}
        if prog == "nvidia-smi":
            return {"ok": True, "code": 0, "stdout": "1, 40\n", "stderr": ""}
        if prog == "ss":
            lines = ["State Recv-Q Send-Q Local Address:Port Peer Address:Port"]
            for port in sorted(self.listen.get(host) or []):
                lines.append(f"LISTEN 0 128 0.0.0.0:{port} 0.0.0.0:*")
            return {"ok": True, "code": 0, "stdout": "\n".join(lines) + "\n", "stderr": ""}
        if prog == "docker" and list(argv[:3]) == ["docker", "image", "inspect"]:
            if self.missing_image:
                return {"ok": False, "code": 1, "stdout": "", "stderr": "missing", "error": "missing"}
            return {"ok": True, "code": 0, "stdout": "present\n", "stderr": ""}
        if prog == "docker":
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
        if prog == "systemctl":
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
        if prog == "test":
            return {"ok": False, "code": 1, "stdout": "", "stderr": ""}
        if prog == "cat":
            return {"ok": True, "code": 0, "stdout": "MemTotal: 126877696 kB\nMemAvailable: 94371840 kB\n", "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "unexpected"}

    def invoke(self, host, req, timeout=60):
        op = req.get("op")
        if op == "probe":
            paths = {path: {"exists": True, "nonempty": True} for path in (req.get("paths") or [])}
            asked = [int(port) for port in (req.get("ports") or [])]
            identity = req.get("http_paths") or {}
            if OFF_LIMITS_PORT in asked:
                assert str(OFF_LIMITS_PORT) in {str(key) for key in identity}
            listening = {}
            http = {}
            for port in req.get("ports") or []:
                if self.health and host == HEAD:
                    listening[str(port)] = True
                    http[str(port)] = {"status": 200, "path": "/health"}
            return {"ok": True, "paths": paths, "listening": listening, "http": http, "containers": [f"sparksi-x-{port}" for port in (req.get("ports") or [])]}
        if op == "run":
            self.started.append((host, list(req.get("argv") or [])))
            return {"ok": True, "phase": "running", "run_id": req.get("run_id")}
        if op == "stop":
            self.stopped.append((host, list(req.get("argv") or [])))
            return {"ok": True, "phase": "stopped"}
        if op == "status":
            return {"ok": True, "phase": "running", "log_tail": "model ready"}
        return {"ok": False, "error": f"unexpected {op}"}


def _serve(lock: Lock):
    STATE.lock = lock
    STATE.runner = Fake()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, httpd.server_address[1]


def _json(port: int, path: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict]:
    data = None if body is None else json.dumps(body).encode()
    req = Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=20) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as exc:
        if hasattr(exc, "code"):
            raw = exc.read().decode()
            return exc.code, json.loads(raw or "{}")
        raise


def test_api_cards_recipe_plan_run_and_stop(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    httpd, port = _serve(lock)
    try:
        code, cards = _json(port, "/api/models")
        assert code == 200
        by_id = {row["id"]: row for row in cards["models"]}
        assert by_id["pair-fast"]["placements"]["both"]["allowed"] is True
        assert by_id["pair-fast"]["presence"][HEAD] is True
        assert by_id["single-gguf"]["placements"]["both"]["reason"] == "llamacpp runs on one Spark"

        code, recipe = _json(port, "/api/models/pair-fast/recipe?placement=both&engine=vllm")
        assert code == 200, recipe
        assert recipe["port"] == 8010
        assert recipe["experimental"] is False
        preview = recipe["argv_preview"]
        assert preview["worker"][preview["worker"].index("--master-addr") + 1] == "192.0.2.11"
        assert "0.0.0.0" not in preview["head"]
        assert preview["head"][preview["head"].index("--host") + 1] == "100.64.0.11"
        assert recipe["order"] == ["worker", "head"]

        code, refused = _json(port, "/api/models/pair-fast/plan", "POST", {"placement": "both", "engine": "vllm", "port": 8188})
        assert code == 400
        assert any("8188" in item for item in refused["conflicts"])
        assert refused["recipe"] is None

        code, refused = _json(port, "/api/models/pair-fast/plan", "POST", {"placement": "head", "engine": "vllm", "port": 8010})
        assert code == 400
        assert any("cannot hold" in item or "over budget" in item for item in refused["conflicts"])

        code, refused = _json(port, "/api/models/pair-fast/plan", "POST", {
            "placement": "both", "engine": "vllm", "port": 8010, "extra_flags": ["--not-a-real-flag"],
        })
        assert code == 400
        assert any("unknown flag" in item for item in refused["conflicts"])

        STATE.runner.listen[HEAD].add(8012)
        reset_cache_for_tests()
        code, refused = _json(port, "/api/models/single-gguf/plan", "POST", {"placement": "head", "engine": "llamacpp", "port": 8012})
        assert code == 400
        assert any("already listening" in item for item in refused["conflicts"])
        STATE.runner.listen[HEAD].clear()
        reset_cache_for_tests()

        code, missing = _json(port, "/api/models/pair-fast/plan", "POST", {"placement": "both", "engine": "vllm", "port": 8010})
        STATE.runner.missing_image = True
        reset_cache_for_tests()
        code, missing = _json(port, "/api/models/pair-fast/plan", "POST", {"placement": "both", "engine": "vllm", "port": 8010})
        assert code == 400
        assert any("missing" in item for item in missing["conflicts"])
        STATE.runner.missing_image = False
        reset_cache_for_tests()

        code, started = _json(port, "/api/runs", "POST", {"id": "pair-fast", "placement": "both", "engine": "vllm", "port": 8010})
        assert code == 200, started
        run_id = started["run"]["id"]
        assert STATE.runner.started[0][0] == WORKER
        assert STATE.runner.started[1][0] == HEAD
        assert "--headless" in STATE.runner.started[0][1]
        assert "0.0.0.0" not in STATE.runner.started[1][1]
        deadline = time.time() + 3
        phase = started["run"]["phase"]
        while phase not in ("ready", "failed", "timed_out") and time.time() < deadline:
            _, body = _json(port, f"/api/runs/{run_id}")
            phase = body["run"]["phase"]
            time.sleep(0.05)
        _, body = _json(port, f"/api/runs/{run_id}")
        assert body["run"]["phase"] == "ready"
        assert body["run"]["nodes"][0]["log_tail"] == "model ready"
        code, stopped = _json(port, f"/api/runs/{run_id}/stop", "POST", {})
        assert code == 200, stopped
        assert stopped["stopped"] is True
        assert STATE.runner.stopped

        lock.acquire("test")
        code, denied = _json(port, "/api/runs", "POST", {"id": "pair-fast", "placement": "both", "engine": "vllm", "port": 8011})
        assert code == 403
        lock.release(ttl_s=120)

        code, saved = _json(port, "/api/models/pair-fast/recipe", "PUT", {
            "engine": "vllm", "placement": "both", "port": 8014, "ctx": 4096, "gpu_mem_util": 0.8,
        })
        assert code == 200, saved
        assert load_recipe("pair-fast")["port"] == 8014
        assert load_recipe("pair-fast")["ctx"] == 4096
        text = (tmp_path / "recipes" / "pair-fast.toml").read_text(encoding="utf-8")
        assert 'engine = "vllm"' in text
        assert not (tmp_path / "sparksi.toml").exists()
        assert not (tmp_path / "catalog.yaml").exists()
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_stop_refuses_8188_and_never_touch(tmp_path) -> None:
    from sparksi.recipes.runs import _save

    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    fake = Fake()
    STATE.lock = lock
    STATE.runner = fake
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        _save({
            "id": "badport1",
            "model_id": "pair-fast",
            "port": 8188,
            "phase": "ready",
            "nodes": [{"role": "head", "host": HEAD, "run_id": "badport1-head", "stop_argv": ["docker", "rm", "-f", "x"]}],
        })
        code, body = _json(port, "/api/runs/badport1/stop", "POST", {})
        assert code == 400
        assert "8188" in body["error"]
        assert fake.stopped == []
        _save({
            "id": "comfyrun1",
            "model_id": "comfy",
            "port": 8010,
            "phase": "ready",
            "nodes": [{"role": "head", "host": HEAD, "run_id": "comfyrun1-head", "stop_argv": ["/home/USER/scripts/video-mode.sh", "stop-video"]}],
        })
        code, body = _json(port, "/api/runs/comfyrun1/stop", "POST", {})
        assert code == 400
        assert fake.stopped == []
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_pair_stop_records_the_side_that_died() -> None:
    class Pair:
        def __init__(self) -> None:
            self.worker_up = True
            self.stopped: list[tuple[str, list[str]]] = []

        def invoke(self, host, req, timeout=20):
            if req.get("op") == "probe":
                assert 8188 not in (req.get("ports") or [])
                containers = ["sparksi-wide-8010"]
                if host == WORKER and not self.worker_up:
                    containers = []
                listening, http = {}, {}
                for port in req.get("ports") or []:
                    if host == HEAD:
                        listening[str(port)] = True
                        http[str(port)] = {"status": None}
                return {"ok": True, "listening": listening, "http": http, "containers": containers}
            if req.get("op") == "status":
                return {"ok": True, "phase": "running", "log_tail": "booting"}
            if req.get("op") == "stop":
                self.stopped.append((host, list(req.get("argv") or [])))
                return {"ok": True}
            return {"ok": False, "error": "unexpected"}

    runner = Pair()
    stop = ["docker", "rm", "-f", "sparksi-wide-8010"]
    run = {
        "id": "pair1",
        "model_id": "wide",
        "engine": "vllm",
        "placement": "both",
        "port": 8010,
        "phase": "waiting_health",
        "started": time.time(),
        "timeout_s": 30,
        "health_paths": ["/health", "/v1/models"],
        "seen": {},
        "nodes": [
            {"role": "worker", "host": WORKER, "run_id": "pair1-worker", "container": "sparksi-wide-8010", "stop_argv": stop, "phase": "started"},
            {"role": "head", "host": HEAD, "run_id": "pair1-head", "container": "sparksi-wide-8010", "stop_argv": stop, "phase": "started"},
        ],
    }
    first = tick(run, runner)
    assert first["phase"] == "waiting_health"
    assert first["seen"].get("worker") is True
    runner.worker_up = False
    second = tick(first, runner)
    assert second["phase"] == "failed"
    assert second["failed_side"] == "worker"
    assert any(host == HEAD for host, _argv in runner.stopped)
    assert all(argv == stop for _host, argv in runner.stopped)


def test_agent_execs_argv_literally_and_refuses_8188(tmp_path) -> None:
    refused = agent_main({"op": "run", "run_id": "nope", "argv": ["llama-server", "--port", "8188"]})
    assert refused["ok"] is False
    assert "8188" in refused["error"]
    wild = agent_main({"op": "run", "run_id": "wild", "argv": ["llama-server", "--host", "0.0.0.0", "--port", "8010"]})
    assert wild["ok"] is False
    assert "0.0.0.0" in wild["error"]
    marker = tmp_path / "not-a-shell"
    missed = agent_main({"op": "run", "run_id": "meta", "argv": [f"echo hi; touch {marker}"]})
    assert missed["ok"] is False
    assert "refusing executable" in missed["error"]
    assert not marker.exists()
    binary = tmp_path / "llama-server"
    binary.symlink_to("/bin/sleep")
    started = agent_main({"op": "run", "run_id": "stay", "argv": [str(binary), "30"]})
    assert started["ok"] is True, started
    try:
        status = agent_main({"op": "status", "run_id": "stay"})
        assert status["phase"] == "running"
        assert status["pid"]
        stopped = agent_main({"op": "stop", "run_id": "stay", "argv": []})
        assert stopped["ok"] is True
    finally:
        agent_main({"op": "stop", "run_id": "stay", "argv": []})


def test_recipe_executable_allowlist() -> None:
    from sparksi.agent import _refuse_recipe_argv

    assert _refuse_recipe_argv(["docker", "run", "-d", "--name", "sparksi-demo-8010", "example.invalid/vllm:1", "vllm", "serve"]) is None
    assert _refuse_recipe_argv(["docker", "stop", "sparksi-demo-8010"]) is None
    assert _refuse_recipe_argv(["docker", "rm", "-f", "sparksi-demo-8010"]) is None
    assert _refuse_recipe_argv(["docker", "logs", "--tail", "80", "sparksi-demo-8010"]) is None
    assert _refuse_recipe_argv(["docker", "inspect", "sparksi-demo-8010"]) is None
    assert _refuse_recipe_argv(["docker", "image", "inspect", "example.invalid/vllm:1"]) is None
    assert _refuse_recipe_argv(["docker", "run", "example.invalid/trt:1", "trtllm-serve", "/models/demo"]) is None
    assert _refuse_recipe_argv(["docker", "run", "example.invalid/trt:1", "mpirun", "-n", "2", "trtllm-serve"]) is None
    assert _refuse_recipe_argv(["llama-server", "-m", "/models/demo.gguf", "--host", "100.64.0.11", "--port", "8010"]) is None
    assert _refuse_recipe_argv(["/opt/llama/llama-server", "-m", "/models/demo.gguf"]) is None

    for index, (argv, name) in enumerate((
        (["sleep", "30"], "sleep"),
        (["python3", "-c", "print(1)"], "python3"),
        (["bash", "-c", "true"], "bash"),
        (["trtllm-serve", "/models/demo"], "trtllm-serve"),
        (["mpirun", "-n", "2", "trtllm-serve"], "mpirun"),
        (["/usr/bin/mpirun", "-n", "2"], "mpirun"),
        (["bin/llama-server", "-m", "/models/demo.gguf"], "bin/llama-server"),
        (["docker", "pull", "example.invalid/vllm:1"], "pull"),
        (["docker", "exec", "sparksi-demo-8010", "sh"], "exec"),
        (["docker", "image", "pull", "example.invalid/vllm:1"], "pull"),
        (["docker"], "docker"),
    )):
        refused = agent_main({"op": "run", "run_id": f"allow{index}", "argv": argv})
        assert refused["ok"] is False, argv
        assert refused["error"] == f"refusing executable {name}", refused
        stopped = agent_main({"op": "stop", "run_id": f"allow{index}", "argv": argv})
        assert stopped["ok"] is False
        assert stopped["error"] == f"refusing executable {name}"

    announced = agent_main({
        "op": "run",
        "run_id": "announce1",
        "argv": ["llama-server", "-m", "/models/demo-dgx-announce-weights"],
    })
    assert announced["ok"] is False
    assert announced["error"] == "refusing dgx-announce"
    named = agent_main({
        "op": "run",
        "run_id": "announce2",
        "argv": ["/home/USER/.local/bin/dgx-announce"],
    })
    assert named["ok"] is False
    assert named["error"] == "refusing forbidden command"


def test_unresolved_recipe_does_not_run_on_the_controller(monkeypatch) -> None:
    from sparksi.remote import SshRunner

    called = []
    monkeypatch.setattr("sparksi.remote.agent_main", lambda req: called.append(req) or {"ok": True, "phase": "running"})
    runner = SshRunner({"hosts": {}})
    missing = runner.invoke("spark-a", {"op": "run", "run_id": "gone", "argv": ["llama-server", "-m", "/models/demo"]}, timeout=5)
    assert missing["ok"] is False
    assert missing["error"] == "could not resolve spark-a"
    assert called == []
    stopped = runner.invoke("spark-a", {"op": "stop", "run_id": "gone", "argv": ["docker", "rm", "-f", "sparksi-demo-8010"]}, timeout=5)
    assert stopped["ok"] is False
    assert "could not resolve" in stopped["error"]
    assert called == []
    blank = SshRunner({"hosts": {"spark-a": {}}})
    unresolved = blank.invoke("spark-a", {"op": "run", "run_id": "blank", "argv": ["llama-server", "-m", "/models/demo"]}, timeout=5)
    assert unresolved["ok"] is False
    assert unresolved["error"] == "could not resolve spark-a"
    assert called == []

    local = SshRunner({"hosts": {"spark-a": {"ssh": "local"}}})
    allowed = local.invoke("spark-a", {"op": "status", "run_id": "stay"}, timeout=5)
    assert called and called[0]["op"] == "status"
    assert allowed.get("ok") is True


def test_api_models_builds_an_ssh_runner_when_unset(monkeypatch, tmp_path) -> None:
    from sparksi.catalog import load_catalog
    import sparksi.serve as serve_mod

    calls = []

    class Recording:
        def __init__(self, catalog):
            calls.append(("init", catalog))

        def capture(self, host, argv, timeout=8):
            calls.append(("capture", host, tuple(argv)))
            prog = argv[0] if argv else ""
            if prog == "free":
                return {"ok": True, "code": 0, "stdout": _free(), "stderr": ""}
            if prog == "ss":
                return {
                    "ok": True,
                    "code": 0,
                    "stdout": "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n",
                    "stderr": "",
                }
            if prog == "nvidia-smi":
                return {"ok": True, "code": 0, "stdout": "0, 40\n", "stderr": ""}
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}

        def invoke(self, host, req, timeout=60):
            calls.append(("invoke", host, req.get("op"), list(req.get("paths") or [])))
            if req.get("op") == "probe":
                paths = {path: {"exists": True, "nonempty": True} for path in (req.get("paths") or [])}
                return {"ok": True, "paths": paths, "listening": {}, "http": {}}
            return {"ok": False, "error": f"unexpected {req.get('op')}"}

    monkeypatch.setattr("sparksi.serve.SshRunner", Recording)
    serve_mod._PROD_RUNNER = None
    reset_cache_for_tests()
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    STATE.lock = lock
    STATE.runner = None
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    try:
        code, cards = _json(port, "/api/models")
        assert code == 200, cards
        assert any(item[0] == "init" for item in calls)
        probed = {item[1] for item in calls if item[0] == "invoke" and item[2] == "probe"}
        assert probed == {HEAD, WORKER}
        by_id = {row["id"]: row for row in cards["models"]}
        assert by_id["single-gguf"]["presence"][HEAD] is True
        assert by_id["single-gguf"]["placements"]["head"]["allowed"] is True
        assert by_id["single-gguf"]["placements"]["head"]["reason"] == ""
        assert by_id["pair-fast"]["placements"]["both"]["allowed"] is True
        assert "not checked" not in json.dumps(by_id["single-gguf"]["placements"])
        before = len(calls)
        code, again = _json(port, "/api/models")
        assert code == 200, again
        assert len(calls) == before
        catalog = load_catalog()
        assert calls[0][1]["library"][0]["id"] == catalog["library"][0]["id"]
    finally:
        httpd.shutdown()
        STATE.runner = None
        serve_mod._PROD_RUNNER = None
        reset_cache_for_tests()


def test_dark_spark_is_unreachable_not_a_crash() -> None:
    from sparksi.catalog import load_catalog
    from sparksi.recipes.service import list_model_cards

    class Down:
        def capture(self, host, argv, timeout=8):
            raise OSError("timed out")

        def invoke(self, host, req, timeout=60):
            raise OSError("timed out")

    reset_cache_for_tests()
    body = list_model_cards(load_catalog(), Down())
    by_id = {row["id"]: row for row in body["models"]}
    assert by_id["single-gguf"]["placements"]["head"]["reason"] == "could not reach the head"
    assert by_id["single-gguf"]["placements"]["head"]["allowed"] is False
    assert by_id["pair-fast"]["placements"]["head"]["reason"] == "could not reach the head"
    assert by_id["pair-fast"]["placements"]["worker"]["reason"] == "could not reach the worker"
    assert by_id["pair-fast"]["placements"]["both"]["reason"] == "could not reach both Sparks"


def test_model_probes_both_sparks_together() -> None:
    from sparksi.catalog import load_catalog
    from sparksi.recipes.service import list_model_cards

    gate = threading.Barrier(2)

    class Slow:
        def capture(self, host, argv, timeout=8):
            prog = argv[0] if argv else ""
            if prog == "free":
                return {"ok": True, "code": 0, "stdout": _free(), "stderr": ""}
            if prog == "ss":
                return {
                    "ok": True,
                    "code": 0,
                    "stdout": "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n",
                    "stderr": "",
                }
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}

        def invoke(self, host, req, timeout=60):
            if req.get("op") == "probe":
                gate.wait(timeout=2)
                paths = {path: {"exists": True, "nonempty": True} for path in (req.get("paths") or [])}
                return {"ok": True, "paths": paths}
            return {"ok": False, "error": "unexpected"}

    reset_cache_for_tests()
    body = list_model_cards(load_catalog(), Slow())
    by_id = {row["id"]: row for row in body["models"]}
    assert by_id["single-gguf"]["placements"]["head"]["allowed"] is True


def test_default_recipe_source_priority() -> None:
    from sparksi.recipes.defaults import SOURCE_CATALOG, SOURCE_ENGINE, SOURCE_LAUNCH, SOURCE_SAVED, derive_defaults

    roomy = {"head": 111.0, "worker": 111.0}
    umas = {"head": 121.0, "worker": 121.0}
    bare = ModelCard(id="demo", title="Demo", format="fp8", weights_gib=40, engines=["vllm"])
    one = derive_defaults(bare, "vllm", "head", saved={}, row={}, override={}, budgets=roomy, umas=umas, image=IMAGE)
    pair = derive_defaults(bare, "vllm", "both", saved={}, row={}, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert one["source"]["ctx"] == SOURCE_ENGINE
    assert one["source"]["kv_cache_dtype"] == SOURCE_ENGINE
    assert one["source"]["gpu_mem_util"] == SOURCE_ENGINE
    assert one["values"]["ctx"] == 8192
    assert one["values"]["kv_cache_dtype"] == "fp8"
    assert one["values"]["gpu_mem_util"] == 0.80
    assert one["values"]["max_num_seqs"] == 4
    assert pair["values"]["gpu_mem_util"] == 0.85
    assert one["summary"] == "Engine defaults: 8K context, fp8 KV, 0.80 memory"
    assert pair["summary"] == "Engine defaults: 8K context, fp8 KV, 0.85 memory"

    tight = ModelCard(id="demo", title="Demo", format="fp8", weights_gib=24, engines=["vllm"])
    low = derive_defaults(
        tight, "vllm", "head", saved={}, row={}, override={},
        budgets={"head": 30.0, "worker": 111.0}, umas=umas, image=IMAGE,
    )
    assert low["values"]["ctx"] == 4096
    assert low["values"]["gpu_mem_util"] == 0.30
    assert low["source"]["ctx"] == SOURCE_ENGINE

    card = ModelCard(id="demo", title="Demo", format="fp8", weights_gib=40, engines=["vllm"], ctx=16384, max_ctx=131072)
    row = {
        "id": "demo",
        "ctx": 16384,
        "max_ctx": 131072,
        "kv_cache_dtype": "bf16",
        "tool_parser": "hermes",
        "gpu_memory_utilization": 0.72,
    }
    known = derive_defaults(card, "vllm", "head", saved={}, row=row, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert known["max_ctx"] == 131072
    assert known["values"]["ctx"] == 16384
    assert known["source"]["ctx"] == SOURCE_CATALOG
    assert known["source"]["kv_cache_dtype"] == SOURCE_CATALOG
    assert known["source"]["tool_parser"] == SOURCE_CATALOG
    assert known["values"]["gpu_mem_util"] == 0.72
    assert known["source"]["gpu_mem_util"] == SOURCE_CATALOG
    assert known["source"]["max_num_seqs"] == SOURCE_ENGINE
    assert known["origin"] == SOURCE_CATALOG
    assert known["summary"] == "Catalog: 16K context, bf16 KV, 0.72 memory"

    row["launch"] = {"ctx": 32768, "gpu_memory_utilization": 0.90, "image": IMAGE}
    launched = derive_defaults(card, "vllm", "head", saved={}, row=row, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert launched["values"]["ctx"] == 32768
    assert launched["source"]["ctx"] == SOURCE_LAUNCH
    assert launched["values"]["gpu_mem_util"] == 0.90
    assert launched["source"]["image"] == SOURCE_LAUNCH
    assert launched["source"]["kv_cache_dtype"] == SOURCE_CATALOG
    assert launched["origin"] == SOURCE_LAUNCH
    assert launched["summary"] == "Launch config: 32K context, bf16 KV, 0.90 memory"

    row["launch"]["start"] = ["vllm", "serve", "/models/demo", "--max-model-len", "65536", "--kv-cache-dtype", "fp8"]
    from_argv = derive_defaults(card, "vllm", "head", saved={}, row=row, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert from_argv["values"]["ctx"] == 65536
    assert from_argv["source"]["ctx"] == SOURCE_LAUNCH
    assert from_argv["values"]["kv_cache_dtype"] == "fp8"
    assert from_argv["values"]["gpu_mem_util"] == 0.90

    saved = {"ctx": 32768, "gpu_mem_util": 0.85, "kv_cache_dtype": "fp8"}
    stored = derive_defaults(card, "vllm", "both", saved=saved, row=row, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert stored["values"]["ctx"] == 32768
    assert stored["source"]["ctx"] == SOURCE_SAVED
    assert stored["source"]["gpu_mem_util"] == SOURCE_SAVED
    assert stored["source"]["kv_cache_dtype"] == SOURCE_SAVED
    assert stored["source"]["tool_parser"] == SOURCE_CATALOG
    assert stored["values"]["gpu_mem_util"] == 0.85
    assert stored["origin"] == SOURCE_SAVED
    assert stored["summary"] == "Saved recipe: 32K context, fp8 KV, 0.85 memory"
    moved = derive_defaults(card, "vllm", "head", saved=saved, row=row, override={}, budgets=roomy, umas=umas, image=IMAGE)
    assert moved["values"]["gpu_mem_util"] == 0.85


def test_ctx_validation_respects_model_maximum() -> None:
    cleaned, bad = validate_fields("vllm", {"ctx": 10}, allowed_images=ALLOWED)
    assert cleaned.get("ctx") is None
    assert any("ctx must be from 256 to 2097152" in item for item in bad)
    cleaned, bad = validate_fields("vllm", {"ctx": 5000}, allowed_images=ALLOWED, max_ctx=8192)
    assert bad == []
    assert cleaned["ctx"] == 5000
    cleaned, bad = validate_fields("vllm", {"ctx": 9000}, allowed_images=ALLOWED, max_ctx=8192)
    assert cleaned.get("ctx") is None
    assert any("model maximum 8192" in item for item in bad)
    cleaned, bad = validate_fields("vllm", {"ctx": 8192}, allowed_images=ALLOWED, max_ctx=8192)
    assert cleaned["ctx"] == 8192
    cleaned, bad = validate_fields("vllm", {"ctx": 300000}, allowed_images=ALLOWED)
    assert bad == []
    assert cleaned["ctx"] == 300000
    _, bad = validate_fields("vllm", {"ctx": 3000000}, allowed_images=ALLOWED)
    assert any("2097152" in item for item in bad)
    cleaned, bad = validate_fields("vllm", {"ctx": 270000}, allowed_images=ALLOWED, max_ctx=300000)
    assert bad == []
    assert cleaned["ctx"] == 270000
    _, bad = validate_fields("vllm", {"ctx": 300001}, allowed_images=ALLOWED, max_ctx=300000)
    assert any("model maximum 300000" in item for item in bad)
    _, bad = validate_fields("vllm", {"ctx": "nope"}, allowed_images=ALLOWED, max_ctx=8192)
    assert any("integer" in item for item in bad)


def test_plan_refuses_ctx_above_the_model_maximum() -> None:
    from sparksi.config import CFG
    from sparksi.recipes.service import plan_model

    host = CFG.head.name
    catalog = {
        "library": [{
            "id": "demo",
            "title": "Demo",
            "format": "fp8",
            "gib": 8,
            "max_ctx": 8192,
            "engines": ["vllm"],
            "local": {host: "/models/demo"},
        }],
        "hosts": {host: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4}},
    }
    code, high = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8010, "ctx": 9000})
    assert code == 400
    assert any("model maximum" in item for item in high["conflicts"])
    _code, held = plan_model(catalog, None, "demo", {"engine": "vllm", "placement": "head", "port": 8010, "ctx": 5000})
    assert not any("ctx" in item for item in held["conflicts"])
    assert (held.get("draft") or held.get("recipe") or {})["fields"]["ctx"] == 5000
    assert held["defaults"]["max_ctx"] == 8192


def test_recipe_preview_exposes_defaults(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    httpd, port = _serve(lock)
    try:
        code, both = _json(port, "/api/models/pair-fast/recipe?placement=both&engine=vllm")
        assert code == 200, both
        assert both["defaults"]["origin"] == "engine defaults"
        assert both["defaults"]["values"]["ctx"] == 8192
        assert both["defaults"]["values"]["kv_cache_dtype"] == "fp8"
        assert both["defaults"]["values"]["gpu_mem_util"] == 0.85
        assert both["defaults"]["source"]["gpu_mem_util"] == "engine defaults"
        assert both["defaults"]["summary"] == "Engine defaults: 8K context, fp8 KV, 0.85 memory"
        preview = both["argv_preview"]["head"]
        assert preview[preview.index("--gpu-memory-utilization") + 1] == "0.85"
        assert preview[preview.index("--max-model-len") + 1] == "8192"

        _code, head = _json(port, "/api/models/pair-fast/recipe?placement=head&engine=vllm")
        assert head["defaults"]["values"]["gpu_mem_util"] != both["defaults"]["values"]["gpu_mem_util"]

        code, custom = _json(port, "/api/models/pair-fast/recipe?placement=both&engine=vllm&ctx=5000")
        assert code == 200, custom
        assert custom["fields"]["ctx"] == 5000
        argv = custom["argv_preview"]["head"]
        assert argv[argv.index("--max-model-len") + 1] == "5000"
        assert custom["defaults"]["values"]["ctx"] == 8192

        code, saved = _json(port, "/api/models/pair-fast/recipe", "PUT", {
            "engine": "vllm", "placement": "both", "port": 8014, "ctx": 32768,
            "gpu_mem_util": 0.85, "kv_cache_dtype": "fp8",
        })
        assert code == 200, saved
        code, again = _json(port, "/api/models/pair-fast/recipe?placement=both&engine=vllm")
        assert code == 200, again
        assert again["defaults"]["origin"] == "saved recipe"
        assert again["defaults"]["values"]["ctx"] == 32768
        assert again["defaults"]["source"]["ctx"] == "saved recipe"
        assert again["defaults"]["summary"] == "Saved recipe: 32K context, fp8 KV, 0.85 memory"
        assert again["fields"]["ctx"] == 32768
        argv = again["argv_preview"]["head"]
        assert argv[argv.index("--max-model-len") + 1] == "32768"
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_saved_recipe_round_trip(tmp_path) -> None:
    save_recipe("demo", {"engine": "vllm", "port": 8019, "prefix_cache": False, "extra_flags": ["--trust-remote-code"]})
    loaded = load_recipe("demo")
    assert loaded["port"] == 8019
    assert loaded["prefix_cache"] is False
    assert loaded["extra_flags"] == ["--trust-remote-code"]


def test_overview_does_not_poll_model_cards() -> None:
    text = open("src/sparksi/static/index.html", encoding="utf-8").read()
    refresh = text.split("function refresh()")[1].split("function renderAll()")[0]
    assert "/api/models" not in refresh


_UI_FORM_BODY = {
    "engine": "vllm", "placement": "head", "timeout_s": 1800, "port": 8010,
    "ctx": 8192, "max_num_seqs": 4, "gpu_mem_util": 0.4, "kv_cache_dtype": "fp8",
    "speculative_method": "", "speculative_model": "", "speculative_tokens": 0,
    "tool_parser": "", "reasoning_parser": "",
    "image": IMAGE, "repo": "", "revision": "",
    "evict_page_cache": False, "extra_flags": [], "from_ui": True,
}


def _put_raw(port: int, body: dict, headers: dict[str, str]) -> int:
    req = Request(f"http://127.0.0.1:{port}/api/models/pair-fast/recipe", data=json.dumps(body).encode(), method="PUT")
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urlopen(req, timeout=20) as resp:
            return resp.status
    except Exception as exc:
        if hasattr(exc, "code"):
            return exc.code
        raise


def _no_unknown_field(payload: dict) -> bool:
    lines = list(payload.get("conflicts") or []) + [str(payload.get("error") or "")]
    return not any("unknown field" in str(line) for line in lines)


def test_recipe_save_is_not_behind_the_model_lock(tmp_path, monkeypatch) -> None:
    import sparksi.serve as serve_mod

    monkeypatch.setattr(serve_mod, "load_in_progress", lambda: False)  # no load left over from other tests
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    httpd, port = _serve(lock)
    try:
        body = dict(_UI_FORM_BODY, placement="both")
        code, resp = _json(port, "/api/models/pair-fast/recipe", "PUT", body)
        assert code == 200, resp
        saved = load_recipe("pair-fast")
        assert saved["ctx"] == 8192
        assert saved.get("evict_page_cache") in (False, None)
        assert lock.read().locked is True

        code, resp = _json(port, "/api/runs", "POST", {"id": "pair-fast", "placement": "both", "engine": "vllm", "port": 8010, "from_ui": True})
        assert code == 403, resp
        assert resp["error"] == "Unlock first."
        code, resp = _json(port, "/api/runs/someid/stop", "POST", {"from_ui": True})
        assert code == 403, resp
        code, resp = _json(port, "/api/stop", "POST", {"id": "pair-fast", "from_ui": True})
        assert code == 403, resp
        code, resp = _json(port, "/api/models/video-h3/service", "POST", {"action": "start", "from_ui": True})
        assert code == 403, resp
        assert not STATE.runner.started

        assert _put_raw(port, body, {"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
        assert _put_raw(port, body, {"Content-Type": "text/plain"}) == 415
        assert _put_raw(port, body, {"Content-Type": "application/json", "Origin": f"http://127.0.0.1:{port}"}) == 200
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_ui_form_body_has_no_unknown_fields(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.release(ttl_s=120)
    httpd, port = _serve(lock)
    try:
        code, plan = _json(port, "/api/models/pair-fast/plan", "POST", _UI_FORM_BODY)
        assert _no_unknown_field(plan), plan
        code, saved = _json(port, "/api/models/pair-fast/recipe", "PUT", _UI_FORM_BODY)
        assert code == 200, saved
        assert _no_unknown_field(saved), saved

        variant = dict(_UI_FORM_BODY, evict_page_cache=True, placement="both")
        code, plan = _json(port, "/api/models/pair-fast/plan", "POST", variant)
        assert _no_unknown_field(plan), plan
        assert plan["recipe"] is not None, plan
        assert plan["recipe"]["evict_page_cache"] is True
        code, saved = _json(port, "/api/models/pair-fast/recipe", "PUT", variant)
        assert code == 200, saved
        assert load_recipe("pair-fast")["evict_page_cache"] is True

        code, started = _json(port, "/api/runs", "POST", dict(variant, id="pair-fast"))
        assert code == 200, started
        assert _no_unknown_field(started), started
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_engine_fields_drop_control_raw_speculative_and_recipe_keys() -> None:
    from sparksi.recipes.service import _engine_fields

    fields = _engine_fields(_UI_FORM_BODY)
    for key in ("from_ui", "speculative_method", "speculative_model", "speculative_tokens", "speculative", "evict_page_cache", "repo", "revision"):
        assert key not in fields
    assert fields["ctx"] == 8192
    on = _engine_fields(dict(_UI_FORM_BODY, speculative_method="mtp", speculative_tokens=2))
    assert on["speculative"] == {"method": "mtp", "model": "", "tokens": 2}
    for engine in ("vllm", "trtllm", "custom"):
        _, conflicts = validate_fields(engine, _engine_fields(dict(_UI_FORM_BODY, engine=engine)), allowed_images=ALLOWED)
        assert not any(c.startswith("unknown field speculative") for c in conflicts), (engine, conflicts)


def _max_model_len(plan: dict) -> str | None:
    for node in (plan.get("draft") or {}).get("nodes") or []:
        argv = node.get("argv") or []
        if "--max-model-len" in argv:
            return argv[argv.index("--max-model-len") + 1]
    return None


def test_named_recipes_round_trip_while_locked(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
    httpd, port = _serve(lock)
    try:
        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert code == 200, (code, body)
        assert [r["name"] for r in body["recipes"]] == ["default"]
        assert body["default"] == "default"

        payload = {
            "name": "long",
            "engine": "vllm",
            "placement": "head",
            "port": 8012,
            "ctx": 32768,
            "image": IMAGE,
            "from_ui": True,
            "recipe_name": "default",
        }
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", payload)
        assert code == 200, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert [r["name"] for r in body["recipes"]] == ["default", "long"]

        code, body = _json(port, "/api/models/pair-fast/recipes/long", "GET", None)
        assert code == 200, (code, body)
        assert body["recipe"]["ctx"] == 32768
        assert load_recipe("pair-fast")["ctx"] == 4096

        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", payload)
        assert code == 409, (code, body)
        assert body["exists"] is True
        overwrite = dict(payload, ctx=16384, overwrite=True)
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", overwrite)
        assert code == 200, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", dict(payload, name="bad name!"))
        assert code == 400, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes/nope", "PUT", {"ctx": 1})
        assert code == 404, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes/long", "PUT", {"ctx": 24576})
        assert code == 200, (code, body)
        code, body = _json(port, "/api/models/pair-fast/recipes/long", "GET", None)
        assert body["recipe"]["ctx"] == 24576

        code, body = _json(port, "/api/models/pair-fast/recipe", "GET", None)
        assert code == 200 and body["ctx"] == 4096, (code, body)
        code, body = _json(port, "/api/models/pair-fast/recipe?recipe_name=long", "GET", None)
        assert code == 200 and body["ctx"] == 24576, (code, body)
        code, body = _json(port, "/api/models/pair-fast/plan", "POST", {"recipe_name": "long", "from_ui": True})
        assert code == 200, (code, body)
        assert _max_model_len(body) == "24576"
        code, body = _json(port, "/api/models/pair-fast/plan", "POST", {"recipe_name": "bad name!", "from_ui": True})
        assert code == 400, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes/long/default", "POST", {})
        assert code == 200, (code, body)
        assert body["default"] == "long"
        assert load_recipe("pair-fast")["ctx"] == 24576
        code, body = _json(port, "/api/models/pair-fast/recipes/default", "GET", None)
        assert body["recipe"]["ctx"] == 4096

        code, body = _json(port, "/api/models/pair-fast/recipes/long", "DELETE", {})
        assert code == 409, (code, body)
        code, body = _json(port, "/api/models/pair-fast/recipes/default", "DELETE", {})
        assert code == 200, (code, body)
        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert [r["name"] for r in body["recipes"]] == ["long"]
        code, body = _json(port, "/api/models/pair-fast/recipes/long", "DELETE", {})
        assert code == 409, (code, body)

        code, body = _json(
            port,
            "/api/runs",
            "POST",
            {"id": "pair-fast", "from_ui": True, "recipe_name": "long"},
        )
        assert code == 403, (code, body)
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_pinned_default_survives_named_recipes(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    pinned = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-10-31T00:00:00Z",
        "roles": {"head": {"runtime": "docker", "image": IMAGE, "args": ["vllm", "serve", "/models/x", "--port", "8014"]}},
    }
    path = save_recipe("pair-fast", pinned)
    original = path.read_bytes()
    httpd, port = _serve(lock)
    try:
        from sparksi.recipes.store import named_dir

        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert code == 200, (code, body)
        assert [(r["name"], r["kind"]) for r in body["recipes"]] == [("default", "pinned")]

        d = named_dir("pair-fast")
        d.mkdir(parents=True)
        (d / "plain.toml").write_text('engine = "vllm"\nplacement = "head"\nport = 8012\nctx = 8192\n')

        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert [r["name"] for r in body["recipes"]] == ["default", "plain"]

        code, body = _json(port, "/api/models/pair-fast/recipes/plain/default", "POST", {})
        assert code == 200, (code, body)

        code, body = _json(port, "/api/models/pair-fast/recipes/default", "DELETE", {})
        assert code == 409, (code, body)
        assert "pinned" in body["error"]

        code, body = _json(port, "/api/models/pair-fast/recipes/default/default", "POST", {})
        assert code == 200, (code, body)
        assert path.read_bytes() == original
        assert load_recipe("pair-fast")["kind"] == "pinned"
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_recipe_names_and_paths_unit(tmp_path) -> None:
    from sparksi.recipes.store import recipe_path, recipes_dir, using_recipe, valid_recipe_name

    assert valid_recipe_name("fast-1")
    assert valid_recipe_name("A_b")
    assert not valid_recipe_name("../x")
    assert not valid_recipe_name("")
    assert not valid_recipe_name("-x")
    assert not valid_recipe_name("a" * 81)
    assert valid_recipe_name("a" * 80)
    assert valid_recipe_name("glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node")
    assert not valid_recipe_name(".hidden")
    assert not valid_recipe_name("a/b")
    assert not valid_recipe_name("a b")
    with using_recipe("pair-fast", "fast"):
        assert recipe_path("pair-fast") == recipes_dir() / "named" / "pair-fast" / "fast.toml"
        assert recipe_path("other") == recipes_dir() / "other.toml"
    assert recipe_path("pair-fast") == recipes_dir() / "pair-fast.toml"


def test_named_recipe_writes_keep_the_same_origin_json_guard(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
    httpd, port = _serve(lock)
    try:
        body = {"name": "evil", "engine": "vllm", "placement": "head", "port": 8012, "ctx": 8192, "image": IMAGE}
        base = "/api/models/pair-fast/recipes"
        assert _raw(port, base, "POST", body, {"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
        assert _raw(port, base, "POST", body, {"Content-Type": "text/plain"}) == 415
        assert _raw(port, base + "/default", "DELETE", {}, {"Content-Type": "text/plain"}) == 415
        assert _raw(port, base + "/default/default", "POST", {}, {"Content-Type": "application/json", "Origin": "http://evil.example"}) == 403
        code, listed = _json(port, base, "GET", None)
        assert [r["name"] for r in listed["recipes"]] == ["default"]
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_save_as_from_a_pinned_recipe_keeps_its_roles_and_rolls_back_on_refusal(tmp_path) -> None:
    from sparksi.recipes.store import named_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    pinned = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-10-31T00:00:00Z",
        "roles": {"head": {"runtime": "docker", "image": IMAGE, "args": ["vllm", "serve", "/models/x", "--port", "8014"]}},
    }
    original = save_recipe("pair-fast", pinned).read_bytes()
    httpd, port = _serve(lock)
    try:
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", {"name": "tuned", "kind": "pinned", "from_ui": True, "recipe_name": "default"})
        copy = named_dir("pair-fast") / "tuned.toml"
        assert code == 200, body
        assert body["recipe"]["kind"] == "pinned"
        assert body["recipe"]["roles"]["head"]["args"][:3] == ["vllm", "serve", "/models/x"]
        assert copy.is_file()
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", {"name": "broken", "kind": "pinned", "port": 1, "from_ui": True})
        assert code == 400, body
        assert not (named_dir("pair-fast") / "broken.toml").exists()
        assert (tmp_path / "recipes" / "pair-fast.toml").read_bytes() == original
    finally:
        httpd.shutdown()
        STATE.runner = None


def _raw(port: int, path: str, method: str, body: dict, headers: dict[str, str]) -> int:
    req = Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode(), method=method)
    for key, value in headers.items():
        req.add_header(key, value)
    try:
        with urlopen(req, timeout=20) as resp:
            return resp.status
    except Exception as exc:
        if hasattr(exc, "code"):
            return exc.code
        raise


def _save_as(port: int, name: str, ctx: int, **extra) -> tuple[int, dict]:
    body = {"name": name, "engine": "vllm", "placement": "head", "port": 8012, "image": IMAGE, "ctx": ctx, "from_ui": True}
    body.update(extra)
    return _json(port, "/api/models/pair-fast/recipes", "POST", body)


def _names(port: int) -> list[str]:
    code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
    assert code == 200, body
    return [item["name"] for item in body["recipes"]]


def test_rename_named_recipes(tmp_path) -> None:
    from sparksi.recipes.store import recipes_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
    httpd, port = _serve(lock)
    base = "/api/models/pair-fast/recipes"
    try:
        assert _save_as(port, "a", 8192)[0] == 200
        assert _save_as(port, "b", 16384)[0] == 200
        code, body = _json(port, base, "GET", None)
        assert all(item["modified"].endswith("Z") for item in body["recipes"]), body

        code, body = _json(port, base + "/a/rename", "POST", {"new_name": "fast"})
        assert code == 200, body
        assert [item["name"] for item in body["recipes"]] == ["default", "b", "fast"]
        assert _json(port, base + "/fast", "GET", None)[1]["recipe"]["ctx"] == 8192

        code, body = _json(port, base + "/fast/rename", "POST", {"new_name": "b"})
        assert code == 409 and body["exists"] is True, body
        code, body = _json(port, base + "/fast/rename", "POST", {"new_name": "b", "overwrite": True})
        assert code == 200, body
        assert _names(port) == ["default", "b"]
        assert _json(port, base + "/b", "GET", None)[1]["recipe"]["ctx"] == 8192

        assert _json(port, base + "/b/rename", "POST", {"new_name": "default"})[0] == 400
        assert _json(port, base + "/b/rename", "POST", {"new_name": "bad name!"})[0] == 400
        assert _json(port, base + "/nope/rename", "POST", {"new_name": "x"})[0] == 404

        assert _json(port, base + "/b/default", "POST", {})[0] == 200
        code, body = _json(port, base + "/b/rename", "POST", {"new_name": "main"})
        assert code == 200 and body["default"] == "main", body
        assert load_recipe("pair-fast")["ctx"] == 8192
        assert (recipes_dir() / "pair-fast.toml").is_file()
        assert _names(port) == ["main", "default"]
        code, body = _json(port, base + "/default/rename", "POST", {"new_name": "main", "overwrite": True})
        assert code == 409, body

        code, body = _json(port, "/api/models/pair-fast/recipe?recipe_new=1", "GET", None)
        assert code == 200 and body["port"] != 8012, body  # model defaults, not the saved 8012
        code, body = _save_as(port, "fresh", 12288, recipe_new=True, recipe_name="main")
        assert code == 200, body
        assert _json(port, base + "/fresh", "GET", None)[1]["recipe"]["ctx"] == 12288
        assert load_recipe("pair-fast")["ctx"] == 8192
    finally:
        httpd.shutdown()
        STATE.runner = None


PINNED_DEFAULT = {
    "kind": "pinned",
    "engine": "vllm",
    "placement": "head",
    "port": 8014,
    "imported_from": "container:example at 2026-10-31T00:00:00Z",
    "roles": {
        "head": {
            "runtime": "docker",
            "image": IMAGE,
            "args": [
                "vllm", "serve", "/models/x", "--port", "8014", "--served-model-name", "model-x",
                "--max-model-len", "32768", "--max-num-seqs", "2", "--kv-cache-dtype", "fp8",
            ],
        }
    },
}


def test_renaming_the_pinned_default_keeps_it_the_default(tmp_path) -> None:
    """The default changes its name in index.json only; the launch command stays at <id>.toml."""
    from sparksi.recipes.store import named_dir, recipe_path

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", PINNED_DEFAULT)
    original = recipe_path("pair-fast").read_bytes()
    httpd, port = _serve(lock)
    base = "/api/models/pair-fast/recipes"
    try:
        code, body = _json(port, base + "/default/rename", "POST", {"new_name": "model-x-c2-32k-fp8"})
        assert code == 200, body
        assert body["default"] == "model-x-c2-32k-fp8", body
        assert [item["name"] for item in body["recipes"]] == ["model-x-c2-32k-fp8"]
        assert recipe_path("pair-fast").read_bytes() == original
        assert json.loads((named_dir("pair-fast") / "index.json").read_text()) == {"default": "model-x-c2-32k-fp8"}
        assert not (named_dir("pair-fast") / "default.toml").exists()
        # Load default recipe reads <id>.toml with no recipe name: the same command.
        code, plain = _json(port, "/api/models/pair-fast/recipe", "GET", None)
        assert code == 200 and plain.get("kind") == "pinned", plain
        code, named = _json(port, base + "/model-x-c2-32k-fp8", "GET", None)
        assert code == 200 and named["default"] is True and named["recipe"]["kind"] == "pinned", named
        # The old name is gone and stays reserved.
        assert _json(port, base + "/default", "GET", None)[0] == 404
        assert _json(port, base + "/model-x-c2-32k-fp8/rename", "POST", {"new_name": "default"})[0] == 400
        # The renamed default is still the model's only pinned command, so it is not deleted.
        assert _json(port, base + "/model-x-c2-32k-fp8", "DELETE", {})[0] == 409
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_list_columns_are_read_from_the_pinned_command(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", PINNED_DEFAULT)
    httpd, port = _serve(lock)
    try:
        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert code == 200, body
        row = body["recipes"][0]
        assert (row["ctx"], row["max_num_seqs"], row["kv_cache_dtype"]) == (32768, 2, "fp8"), row
        assert row["model_hint"] == "model-x" and row["quant_hint"] == "x", row
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_save_never_rewrites_a_pinned_command(tmp_path) -> None:
    """File > Save on any pinned recipe answers with Save as; Save as never overwrites one either."""
    from sparksi.recipes.store import named_dir, recipe_path

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", PINNED_DEFAULT)
    original = recipe_path("pair-fast").read_bytes()
    httpd, port = _serve(lock)
    base = "/api/models/pair-fast/recipes"
    fields = {"engine": "vllm", "placement": "head", "port": 8014, "ctx": 8192, "from_ui": True}
    try:
        code, body = _json(port, base + "/default", "PUT", fields)
        assert code == 409 and body["save_as"] is True and "pinned launch command" in body["error"], body
        assert recipe_path("pair-fast").read_bytes() == original
        code, body = _json(port, base + "/default/rename", "POST", {"new_name": "main"})
        assert code == 200, body
        code, body = _json(port, base + "/main", "PUT", fields)
        assert code == 409 and body["save_as"] is True, body
        code, body = _json(port, base, "POST", {**fields, "name": "main", "overwrite": True})
        assert code == 409 and body["save_as"] is True, body
        assert recipe_path("pair-fast").read_bytes() == original
        # Save as to a new name keeps the edit, as a second pinned recipe.
        code, body = _json(port, base, "POST", {**fields, "name": "edited", "recipe_name": "main"})
        assert code == 200, body
        assert "8192" in (named_dir("pair-fast") / "edited.toml").read_text()
        assert _json(port, base + "/edited", "PUT", fields)[0] == 409
        # With two pinned recipes the one that is not the default can go.
        assert _json(port, base + "/edited", "DELETE", {})[0] == 200
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_a_pinned_recipe_that_is_not_the_default_can_be_renamed(tmp_path) -> None:
    from sparksi.recipes.store import named_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", PINNED_DEFAULT)
    directory = named_dir("pair-fast")
    directory.mkdir(parents=True)
    (directory / "plain.toml").write_text('engine = "vllm"\nplacement = "head"\nport = 8012\nctx = 8192\n')
    httpd, port = _serve(lock)
    try:
        assert _json(port, "/api/models/pair-fast/recipes/plain/default", "POST", {})[0] == 200
        code, body = _json(port, "/api/models/pair-fast/recipes/default/rename", "POST", {"new_name": "imported"})
        assert code == 200, body
        assert _names(port) == ["plain", "imported"]
        assert body["default"] == "plain", body
        # The only pinned command stays, even when it is not the default.
        code, body = _json(port, "/api/models/pair-fast/recipes/imported", "DELETE", {})
        assert code == 409 and "only pinned" in body["error"], body
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_recipe_list_survives_a_junk_pinned_file(tmp_path) -> None:
    """GET recipes stays 200 when a pinned file's args or env have the wrong shape."""
    from sparksi.recipes.store import named_dir, recipes_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    recipes_dir().mkdir(parents=True, exist_ok=True)
    (recipes_dir() / "pair-fast.toml").write_text(
        'kind = "pinned"\nengine = "vllm"\nplacement = "head"\nport = 8014\n\n'
        '[roles.head]\nargs = "vllm serve /models/x --max-model-len 4096"\nenv = ["TP=2"]\n'
    )
    directory = named_dir("pair-fast")
    directory.mkdir(parents=True)
    (directory / "odd.toml").write_text('kind = "pinned"\nengine = "vllm"\n\n[roles]\nhead = "vllm serve"\n')
    (directory / "odder.toml").write_text('kind = "pinned"\nengine = "vllm"\n\n[roles.head]\nargs = 42\n\n[roles.head.env]\nTP = 2\n')
    httpd, port = _serve(lock)
    try:
        code, body = _json(port, "/api/models/pair-fast/recipes", "GET", None)
        assert code == 200, body
        rows = {item["name"]: item for item in body["recipes"]}
        assert set(rows) == {"default", "odd", "odder"}, rows
        assert rows["default"]["ctx"] is None and rows["default"]["model_hint"] == "", rows
        assert rows["odder"]["tp"] == 2, rows
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_rename_never_overwrites_a_pinned_recipe(tmp_path) -> None:
    """Renaming a generated recipe onto a pinned name is refused; the pinned command stays."""
    from sparksi.recipes.store import named_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
    directory = named_dir("pair-fast")
    directory.mkdir(parents=True)
    from sparksi.recipes.store import using_recipe

    with using_recipe("pair-fast", "imported"):
        save_recipe("pair-fast", PINNED_DEFAULT)
    pinned_bytes = (directory / "imported.toml").read_bytes()
    httpd, port = _serve(lock)
    base = "/api/models/pair-fast/recipes"
    try:
        assert _save_as(port, "plain", 8192)[0] == 200
        code, body = _json(port, base + "/plain/rename", "POST", {"new_name": "imported", "overwrite": True})
        assert code == 409 and body["pinned"] is True and body["save_as"] is True, body
        assert "imported runs a pinned launch command" in body["error"], body
        assert (directory / "imported.toml").read_bytes() == pinned_bytes
        assert (directory / "plain.toml").is_file()
        # Without overwrite it is the usual "already exists".
        code, body = _json(port, base + "/plain/rename", "POST", {"new_name": "imported"})
        assert code == 409 and body["exists"] is True, body
        assert sorted(_names(port)) == ["default", "imported", "plain"]
    finally:
        httpd.shutdown()
        STATE.runner = None


def test_the_older_recipe_put_never_rewrites_a_pinned_default(tmp_path) -> None:
    """PUT /api/models/<id>/recipe refuses a pinned default, so no route overwrites a pinned command."""
    from sparksi.recipes.store import recipe_path

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", PINNED_DEFAULT)
    original = recipe_path("pair-fast").read_bytes()
    httpd, port = _serve(lock)
    try:
        for body in (
            {"engine": "vllm", "placement": "head", "port": 8014, "ctx": 8192},
            {"kind": "generated", "engine": "vllm", "placement": "head", "port": 8012, "ctx": 8192, "image": IMAGE},
            {"kind": "pinned", "extra_args": ["--max-num-seqs 9"]},
        ):
            code, reply = _json(port, "/api/models/pair-fast/recipe", "PUT", body)
            assert code == 409 and reply["save_as"] is True, reply
        assert recipe_path("pair-fast").read_bytes() == original
        # A generated default still saves through the same route.
        save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
        code, reply = _json(port, "/api/models/pair-fast/recipe", "PUT", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 8192, "image": IMAGE})
        assert code == 200, reply
        assert load_recipe("pair-fast")["ctx"] == 8192
    finally:
        httpd.shutdown()
        STATE.runner = None


SAFARI_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 26_1 like Mac OS X) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/26.1 Mobile/15E148 Safari/604.1"
)


def _as_browser(port: int, path: str, method: str, body: dict, host: str, origin: str | None) -> tuple[int, dict]:
    """Send exactly what a browser on the tailnet sends: its own Host and Origin."""
    import http.client

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=20)
    headers = {"Host": host, "Content-Type": "application/json", "User-Agent": SAFARI_UA, "Accept": "*/*"}
    if origin is not None:
        headers["Origin"] = origin
    try:
        conn.request(method, path, body=json.dumps(body), headers=headers)
        resp = conn.getresponse()
        raw = resp.read().decode()
        return resp.status, json.loads(raw or "{}")
    finally:
        conn.close()


def test_recipe_writes_from_a_tailnet_address_are_accepted_and_logged(tmp_path, caplog, monkeypatch) -> None:
    """File > Save, Save as, Rename, Set as default and Delete from a browser on another machine."""
    import logging

    import sparksi.serve as serve_mod
    from sparksi.hosts import Allowlist

    # This machine's own tailnet names, as the startup scan would find them.
    monkeypatch.setattr(serve_mod, "HOSTS", Allowlist(lambda: ["100.64.0.7", "box.example.ts.net", "fd7a:115c:a1e0::7"]))

    from sparksi.recipes.store import named_dir, recipe_path

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 4096, "image": IMAGE})
    httpd, port = _serve(lock)
    host = "100.64.0.7:7601"
    origin = "http://100.64.0.7:7601"
    base = "/api/models/pair-fast/recipes"
    fields = {"engine": "vllm", "placement": "head", "port": 8012, "image": IMAGE, "from_ui": True}
    caplog.set_level(logging.INFO)
    try:
        code, body = _as_browser(port, base + "/default", "PUT", {**fields, "ctx": 8192}, host, origin)
        assert code == 200, body
        assert load_recipe("pair-fast")["ctx"] == 8192
        code, body = _as_browser(port, base, "POST", {**fields, "name": "edited", "ctx": 16384}, host, origin)
        assert code == 200, body
        assert (named_dir("pair-fast") / "edited.toml").is_file()
        code, body = _as_browser(port, base + "/edited", "PUT", {**fields, "ctx": 2048}, host, origin)
        assert code == 200, body
        assert "2048" in (named_dir("pair-fast") / "edited.toml").read_text()
        code, body = _as_browser(port, base + "/edited/rename", "POST", {"new_name": "kept"}, host, origin)
        assert code == 200, body
        code, body = _as_browser(port, base + "/kept/default", "POST", {}, host, origin)
        assert code == 200, body
        code, body = _as_browser(port, base + "/default/default", "POST", {}, host, origin)
        assert code == 200, body
        code, body = _as_browser(port, base + "/kept", "DELETE", {}, host, origin)
        assert code == 200, body
        assert not (named_dir("pair-fast") / "kept.toml").exists()
        assert recipe_path("pair-fast").is_file()

        # The same page reached by name, with the spellings browsers and proxies use.
        for h, o in (
            ("Box.Example.ts.net:7601", "http://box.example.ts.net:7601"),
            ("box.example.ts.net.:7601", "http://box.example.ts.net:7601"),
            ("box.example.ts.net", "http://box.example.ts.net:80"),
            ("[fd7a:115c:a1e0::7]:7601", "http://[FD7A:115C:A1E0::7]:7601"),
        ):
            code, body = _as_browser(port, base + "/default", "PUT", {**fields, "ctx": 4096}, h, o)
            assert code == 200, (h, o, body)
        # A write with no Origin (curl, the CLI) is still same-origin by definition.
        code, body = _as_browser(port, base + "/default", "PUT", {**fields, "ctx": 4096}, host, None)
        assert code == 200, body

        for o in ("null", "http://evil.example", "http://100.64.0.7:7602", "file://", "ftp://100.64.0.7:7601"):
            code, body = _as_browser(port, base + "/default", "PUT", {**fields, "ctx": 1024}, host, o)
            assert code == 403, (o, body)
            assert "cross-origin" in body["error"]
        assert load_recipe("pair-fast")["ctx"] == 4096
    finally:
        httpd.shutdown()
        STATE.runner = None
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("recipe write")]
    assert any("PUT /api/models/pair-fast/recipes/default -> 200" in m and "host=100.64.0.7:7601" in m and "iPhone" in m for m in lines), lines
    assert any("-> 403" in m and "origin=null" in m for m in lines), lines


def test_plan_does_not_call_an_image_missing_when_ssh_gave_no_answer(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    httpd, port = _serve(lock)
    body = {"placement": "both", "engine": "vllm", "port": 8010}
    try:
        STATE.runner.capture_image = STATE.runner.capture

        def flaky(host, argv, timeout=8):
            if list(argv[:3]) == ["docker", "image", "inspect"]:
                return {"ok": False, "code": 255, "stdout": "", "stderr": "", "error": "ssh: connect timed out"}
            return STATE.runner.capture_image(host, argv, timeout)

        STATE.runner.capture = flaky
        reset_cache_for_tests()
        code, plan = _json(port, "/api/models/pair-fast/plan", "POST", body)
        assert code == 400, plan
        assert not any("is missing" in item for item in plan["conflicts"]), plan["conflicts"]
        assert any(item.startswith("could not check image") and "no answer over SSH" in item for item in plan["conflicts"]), plan["conflicts"]

        def down(host, argv, timeout=8):
            raise OSError("timed out")

        STATE.runner.capture = down
        reset_cache_for_tests()
        code, plan = _json(port, "/api/models/pair-fast/plan", "POST", body)
        assert code == 400, plan
        assert not any("is missing" in item for item in plan["conflicts"]), plan["conflicts"]
        assert any(item.startswith(("could not reach", "could not check image")) for item in plan["conflicts"]), plan["conflicts"]
    finally:
        httpd.shutdown()
        STATE.runner = None
        reset_cache_for_tests()


def test_save_as_keeps_ipc_lock_and_dry_run_refuses_it_in_plain_words(tmp_path) -> None:
    """A head-only recipe with the high-speed network flags saves; Dry run and Load refuse it."""
    from sparksi.recipes.pinned import FABRIC_ONLY
    from sparksi.recipes.store import named_dir

    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    pinned = {
        "kind": "pinned",
        "engine": "vllm",
        "placement": "head",
        "port": 8014,
        "imported_from": "container:example at 2026-10-31T00:00:00Z",
        "roles": {"head": {
            "runtime": "docker", "image": IMAGE, "cap_add": ["IPC_LOCK"], "security_opt": ["label=disable"],
            "args": ["vllm", "serve", "/models/x", "--port", "8014"],
        }},
    }
    save_recipe("pair-fast", pinned)
    httpd, port = _serve(lock)
    try:
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", {"name": "tuned", "kind": "pinned", "from_ui": True, "recipe_name": "default"})
        assert code == 200, body
        assert body["not_loadable"] == FABRIC_ONLY
        saved = (named_dir("pair-fast") / "tuned.toml").read_text()
        assert "IPC_LOCK" in saved
        # Save on a pinned recipe never rewrites its command; the page asks for a name instead.
        code, body = _json(port, "/api/models/pair-fast/recipes/tuned", "PUT", {"kind": "pinned", "port": 8014, "from_ui": True})
        assert code == 409 and body["save_as"] is True, body
        assert (named_dir("pair-fast") / "tuned.toml").read_text() == saved
        code, plan = _json(port, "/api/models/pair-fast/plan", "POST", {"recipe_name": "tuned"})
        assert code == 400, plan
        assert FABRIC_ONLY in plan["conflicts"], plan["conflicts"]
        assert not any(item.startswith("refusing --cap-add") for item in plan["conflicts"]), plan["conflicts"]
        code, run = _json(port, "/api/runs", "POST", {"id": "pair-fast", "recipe_name": "tuned", "from_ui": True})
        assert code != 200, run
        assert STATE.runner.started == []
        # Any other added capability is still refused when saving.
        bad = {**pinned, "roles": {"head": {**pinned["roles"]["head"], "cap_add": ["SYS_ADMIN"]}}}
        save_recipe("pair-fast", bad)
        code, body = _json(port, "/api/models/pair-fast/recipes", "POST", {"name": "worse", "kind": "pinned", "from_ui": True, "recipe_name": "default"})
        assert code == 400 and "SYS_ADMIN" in json.dumps(body), body
        assert not (named_dir("pair-fast") / "worse.toml").exists()
    finally:
        httpd.shutdown()
        STATE.runner = None


# The body the Models page sends when a field of a pinned recipe changes (captured
# from the page; env trimmed). A preview must answer 200 even when the plan refuses.
UI_PINNED_PREVIEW = {
    "preview": True, "engine": "other", "placement": "both", "timeout_s": 1800, "port": 8188,
    "ctx": 850000, "max_num_seqs": 5, "gpu_mem_util": 0.8, "kv_cache_dtype": "fp8",
    "speculative_method": "", "speculative_model": "", "speculative_tokens": 0,
    "kind": "pinned", "extra_args": [], "env": ["PORT=8188", "TP=2"],
}


def test_pinned_preview_answers_200_with_the_refusal(tmp_path) -> None:
    lock = Lock(tmp_path / "lock")
    lock.acquire("test")
    httpd, port = _serve(lock)
    try:
        code, preview = _json(port, "/api/models/pair-fast/plan", "POST", dict(UI_PINNED_PREVIEW))
        assert code == 200, preview
        assert preview["verdict"] == "REFUSE" and preview["conflicts"], preview
        assert not any("preview" in str(item) for item in preview["conflicts"])  # the flag is not a recipe field
        dry = {k: v for k, v in UI_PINNED_PREVIEW.items() if k != "preview"}
        code, refused = _json(port, "/api/models/pair-fast/plan", "POST", dry)
        assert code == 400 and refused["verdict"] == "REFUSE"  # Dry run and Load keep 400
        assert refused["conflicts"] == preview["conflicts"]
    finally:
        httpd.shutdown()


def test_page_marks_the_pinned_preview() -> None:
    from sparksi.serve import page_html

    text = page_html()
    assert 'jpostRead("/api/models/" + encodeURIComponent(card.id) + "/plan", Object.assign({ preview: true }, body))' in text


def test_recipe_list_carries_the_list_columns() -> None:
    from sparksi.recipes.named import list_named
    from sparksi.recipes.store import using_recipe

    save_recipe("pair-fast", {"engine": "vllm", "placement": "both", "port": 8012, "ctx": 131072, "max_num_seqs": 4, "kv_cache_dtype": "fp8"})
    with using_recipe("pair-fast", "spec-one"):
        save_recipe("pair-fast", {"engine": "vllm", "placement": "head", "port": 8012, "ctx": 32768, "speculative": {"method": "eagle", "model": "", "tokens": 3}})
    code, body = list_named("pair-fast")
    assert code == 200
    rows = {row["name"]: row for row in body["recipes"]}
    assert rows["default"]["ctx"] == 131072
    assert rows["default"]["max_num_seqs"] == 4
    assert rows["default"]["kv_cache_dtype"] == "fp8"
    assert rows["default"]["speculative"] == ""
    assert rows["spec-one"]["ctx"] == 32768
    assert rows["spec-one"]["max_num_seqs"] is None
    assert rows["spec-one"]["speculative"] == "eagle"
