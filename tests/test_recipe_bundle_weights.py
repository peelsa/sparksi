# SPDX-License-Identifier: Apache-2.0

"""Bundling when the model's weights root is the whole service folder: a single small
mounted file (a patch) is a recipe input and is copied; mounted folders (model, draft,
cache) and large files stay where they are."""

from __future__ import annotations

import copy
import hashlib
import json
import os

import pytest

from sparksi import agent
from sparksi.config import CFG
from sparksi.recipes import bundle as rb
from sparksi.recipes import folder as rf
from sparksi.recipes.store import folder_path, load_recipe, save_recipe, using_recipe

MODEL = "demo"
IMAGE = "example.invalid/vllm:1"
BUILD = "sha256:" + "e" * 64
PATCHES = {"modelopt.py": b"# patched quantization\n" * 40, "attention.py": b"# attention\n" * 30, "draft_model.py": b"# draft\n" * 20}
SITE = "/usr/lib/python3/site-packages/app"


def _raw(n: int, roots: list[str]) -> dict:
    nodes = [{"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}", "mount_roots": roots} for i in range(1, n + 1)]
    return {"nodes": nodes, "paths": {"home": "/home/USER"}}


@pytest.fixture()
def service(tmp_path, monkeypatch, request):
    """A service folder that is also the weights root: model/, draft/, cache/ and patch files."""
    saved = copy.copy(CFG.__dict__)
    root = tmp_path / "spark"
    svc = root / "services" / "example"
    CFG.__init__(_raw(getattr(request, "param", 2), [str(root)]))
    request.addfinalizer(lambda: (CFG.__dict__.clear(), CFG.__dict__.update(saved)))
    for sub, name in (("model", "weights.safetensors"), ("draft", "draft.safetensors"), ("cache", "kernel.bin")):
        (svc / sub).mkdir(parents=True)
        (svc / sub / name).write_bytes(b"w" * 256)
    for name, data in PATCHES.items():
        (svc / name).write_bytes(data)
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "config" / "recipes"))
    monkeypatch.setitem(agent._CFG, "mount_roots", [str(root)])
    for node in CFG.nodes:
        monkeypatch.setattr(node, "recipe_root", str(tmp_path / "copies" / node.id))
    monkeypatch.setitem(agent._CFG, "recipe_roots", [str(tmp_path / "copies" / node.id) for node in CFG.nodes])
    return svc


class _Sparks:
    """Every Spark reads the same local folder through the real agent. Runs nothing."""

    def __init__(self):
        self.reads: list[dict] = []

    def invoke(self, host, req, timeout=20):
        if req.get("op") == "files_read":
            self.reads.append(dict(req))
        if req.get("op") in ("files_read", "file_sha256", "files_check"):
            return agent.main(dict(req))
        raise AssertionError(f"nothing may run: {req.get('op')}")

    def capture(self, host, argv, timeout=8):
        assert list(argv[:3]) == ["docker", "image", "inspect"]
        return {"ok": True, "code": 0, "stdout": json.dumps([{"Id": BUILD}]), "stderr": ""}


def _role(svc) -> dict:
    return {
        "runtime": "docker", "image": IMAGE, "env": {}, "network": "host", "gpus": "all", "container": "sparksi-example",
        "args": ["serve", "/models"], "devices": [],
        "mounts": [
            f"{svc}/model:/models",
            f"{svc}/draft:/draft",
            f"{svc}/cache:/root/.cache",
            *(f"{svc}/{name}:{SITE}/{name}" for name in PATCHES),
        ],
    }


def _save(svc, placement: str) -> tuple[str, ...]:
    roles = ("head", "worker") if placement == "both" else (placement,)
    save_recipe(MODEL, {
        "kind": "pinned", "engine": "vllm", "placement": placement, "port": 8014,
        "imported_from": "compose:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
        "roles": {role: _role(svc) for role in roles},
    })
    return roles


def _catalog(svc) -> dict:
    return {"library": [{"id": MODEL, "title": "Example", "format": "fp8", "gib": 8, "local": {node.name: str(svc) for node in CFG.nodes}}]}


CASES = [(1, "head"), (2, "head"), (2, "worker"), (2, "both"), (3, "both"), (4, "both")]


@pytest.mark.parametrize(("service", "placement"), CASES, indirect=["service"])
def test_patch_files_under_the_weights_root_are_copied_and_folders_stay(service, placement):
    roles = _save(service, placement)
    runner = _Sparks()
    plan = rb.plan_bundle(_catalog(service), runner, MODEL)
    view = plan.view()
    assert view["ok"], view["problems"]
    assert sorted(f["path"] for f in view["files"]) == sorted(f"files{SITE}/{name}" for name in PATCHES)
    assert sum(f["size"] for f in view["files"]) == sum(len(data) for data in PATCHES.values())
    kept = sorted(view["kept"])
    assert kept == sorted(f"{role}: {service}/{sub}:{target}" for role in roles for sub, target in (("model", "/models"), ("draft", "/draft"), ("cache", "/root/.cache")))
    for role in roles:
        mounts = plan.recipe["roles"][role]["mounts"]
        assert mounts[:3] == [f"{service}/model:/models", f"{service}/draft:/draft", f"{service}/cache:/root/.cache"]
        assert mounts[3:] == [f"@files{SITE}/{name}:{SITE}/{name}" for name in PATCHES]
        assert plan.recipe["roles"][role]["image_id"] == BUILD
    # Folders under the weights root are only looked at, never read.
    folder_reads = [req for req in runner.reads if req["paths"][0].rsplit("/", 1)[-1] in ("model", "draft", "cache")]
    assert folder_reads and all(req.get("files_only") is True for req in folder_reads)
    answer = agent.main({"op": "files_read", "paths": [str(service / "model")], "files_only": True, "file_max": 1 << 20})
    assert answer["files"] == {} and answer["skipped"] == {str(service / "model"): "dir"}
    ok, said = rb.write_bundle(plan, stamp="20261122")
    assert ok, said
    root = folder_path(MODEL, plan.name)
    found = rf.scan(root)
    assert found.problems == [] and found.version == view["version"]
    assert sorted(e.path for e in found.entries) == sorted(f"files{SITE}/{name}" for name in PATCHES)
    assert {e.path: e.sha256 for e in found.entries} == {f"files{SITE}/{n}": hashlib.sha256(d).hexdigest() for n, d in PATCHES.items()}
    with using_recipe(MODEL, plan.name):
        resolved = rf.resolve(load_recipe(MODEL), MODEL, plan.name, root)
    assert resolved.conflicts == []
    states = rf.check_sparks(_Sparks(), {"version": resolved.version, "nodes": resolved.nodes, "manifest": [
        {"path": e.path, "sha256": e.sha256, "size": e.size, "executable": e.executable} for e in resolved.entries]})
    assert sorted(row["host"] for row in states.values()) == sorted(CFG.node_by_role(role).name for role in roles)
    assert {row["state"] for row in states.values()} == {"missing"}  # "not copied yet"; the bundle copied nothing to a Spark
    assert not os.path.exists(CFG.node_by_role(roles[0]).recipe_root)


def test_a_large_file_a_link_or_a_kept_file_under_the_weights_root_stays(service, monkeypatch):
    monkeypatch.setattr(rb, "MAX_FILE_BYTES", 500)  # modelopt.py (920 bytes) is now over the limit
    os.symlink(service / "attention.py", service / "linked.py")
    role = _role(service)
    role["mounts"].append(f"{service}/linked.py:/opt/linked.py")
    save_recipe(MODEL, {
        "kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014, "imported_from": "", "port_note": "",
        "rendezvous_ports": [], "roles": {"head": role},
    })
    plan = rb.plan_bundle(_catalog(service), _Sparks(), MODEL, keep=[f"{service}/draft_model.py"])
    view = plan.view()
    assert view["ok"], view["problems"]
    assert [f["path"] for f in view["files"]] == [f"files{SITE}/attention.py"]
    kept = " ".join(view["kept"])
    for name in ("modelopt.py", "draft_model.py", "linked.py", "model:", "draft:", "cache:"):
        assert f"{service}/{name}" in kept


def test_outside_the_weights_root_a_large_file_is_still_refused(service, monkeypatch):
    monkeypatch.setattr(rb, "MAX_FILE_BYTES", 500)
    save_recipe(MODEL, {
        "kind": "pinned", "engine": "vllm", "placement": "head", "port": 8014, "imported_from": "", "port_note": "",
        "rendezvous_ports": [], "roles": {"head": _role(service)},
    })
    catalog = {"library": [{"id": MODEL, "title": "Example", "local": {CFG.head.name: str(service / "model")}}]}
    plan = rb.plan_bundle(catalog, _Sparks(), MODEL)
    assert any("modelopt.py" in p and "--keep" in p for p in plan.problems)
