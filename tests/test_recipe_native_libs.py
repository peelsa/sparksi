# SPDX-License-Identifier: Apache-2.0

"""A native recipe records its binary and the libraries it loads from its own build
tree, checks them all before a start, and copies none of them. A folder recipe with
no files still has a version and shows each Spark's copy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct

import pytest

from sparksi import agent
from sparksi.agent import main as agent_main
from sparksi.config import CFG
from sparksi.recipes import folder as rf
from sparksi.recipes import named
from sparksi.recipes.bundle import plan_bundle, write_bundle
from sparksi.recipes.pins import pin_conflicts
from sparksi.recipes.service import plan_model
from sparksi.recipes.store import folder_path, load_recipe, save_recipe, using_recipe

MODEL = "demo"
ROLES = ("head", "worker", "node3", "node4")


def _elf(needed: list[str], runpath: str = "", rpath: str = "", payload: bytes = b"") -> bytes:
    """A minimal 64-bit little-endian ELF with a dynamic section; enough for the agent's reader."""
    strings = b"\0"
    offsets = []
    for name in [*needed, runpath, rpath]:
        offsets.append(len(strings))
        strings += name.encode() + b"\0"
    strtab = 64 + 2 * 56
    dyn_off = strtab + len(strings)
    dyn_off += (-dyn_off) % 8
    entries = [(1, offsets[i]) for i in range(len(needed))]
    if runpath:
        entries.append((29, offsets[len(needed)]))
    if rpath:
        entries.append((15, offsets[len(needed) + 1]))
    entries += [(5, strtab), (10, len(strings)), (0, 0)]
    dynamic = b"".join(struct.pack("<qQ", tag, value) for tag, value in entries)
    size = dyn_off + len(dynamic) + len(payload)
    head = bytearray(64)
    head[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<Q", head, 0x20, 64)
    struct.pack_into("<HH", head, 0x36, 56, 2)
    phdrs = struct.pack("<IIQQQQQQ", 1, 5, 0, 0, 0, size, size, 8) + struct.pack(
        "<IIQQQQQQ", 2, 6, dyn_off, dyn_off, dyn_off, len(dynamic), len(dynamic), 8
    )
    body = bytes(head) + phdrs + strings
    body += b"\0" * (dyn_off - len(body))
    return body + dynamic + payload


def _raw(n: int, roots: list[str]) -> dict:
    nodes = [
        {"id": f"s{i}", "name": f"spark-{i}", "ssh": f"spark-{i}", "tailscale": f"100.64.0.{10 + i}", "mount_roots": roots}
        for i in range(1, n + 1)
    ]
    return {"nodes": nodes, "paths": {"home": "/home/USER"}}


@pytest.fixture()
def tree(tmp_path, monkeypatch, request):
    """A build tree: bin/server -> libone (bin, via $ORIGIN) -> libtwo (lib, via RPATH); libc is a system library.

    The site has ``request.param`` Sparks (2 when not set).
    """
    saved = copy.copy(CFG.__dict__)
    CFG.__init__(_raw(getattr(request, "param", 2), [str(tmp_path / "spark"), str(tmp_path / "system")]))
    request.addfinalizer(lambda: (CFG.__dict__.clear(), CFG.__dict__.update(saved)))
    build = tmp_path / "spark" / "build"
    (build / "bin").mkdir(parents=True)
    (build / "lib").mkdir()
    system = tmp_path / "system"
    system.mkdir()
    (system / "libc.so.6").write_bytes(_elf([]))
    (build / "bin" / "server").write_bytes(_elf(["libone.so.0", "libc.so.6"], runpath="$ORIGIN"))
    (build / "bin" / "libone.so.0.1").write_bytes(_elf(["libtwo.so", "libc.so.6"], rpath="${ORIGIN}/../lib"))
    os.symlink("libone.so.0.1", build / "bin" / "libone.so.0")
    (build / "lib" / "libtwo.so").write_bytes(_elf([], payload=b"two"))
    monkeypatch.setitem(agent._CFG, "mount_roots", [str(tmp_path / "spark")])  # the Spark side: libc is outside
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SPARKSI_RECIPES_DIR", str(tmp_path / "config" / "recipes"))
    for node in CFG.nodes:
        monkeypatch.setattr(node, "recipe_root", str(tmp_path / "copies" / node.id))
    monkeypatch.setitem(agent._CFG, "recipe_roots", [str(tmp_path / "copies" / node.id) for node in CFG.nodes])
    return build, system


def _sha(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


class _Sparks:
    """Every Spark reads the same local tree through the real agent code. Runs nothing."""

    def __init__(self):
        self.ops: list[tuple[str, dict]] = []

    def invoke(self, host, req, timeout=20):
        self.ops.append((host, dict(req)))
        op = req.get("op")
        if op in ("file_sha256", "files_check", "files_read", "files_stage"):
            return agent_main(dict(req))
        if op == "probe":
            return {"ok": True, "paths": {p: {"exists": True} for p in req.get("paths") or []}, "listening": {}, "http": {}, "containers": []}
        raise AssertionError(f"nothing may run: {op}")

    def capture(self, host, argv, timeout=8):
        raise AssertionError("no image here")

    def run_argv(self, host, argv, timeout=3600):
        raise AssertionError("nothing may run")


def test_the_agent_lists_libraries_from_the_build_tree_only(tree):
    build, system = tree
    out = agent_main({"op": "file_sha256", "path": str(build / "bin" / "server"), "libs": True, "ld_library_path": str(system)})
    assert out["ok"] and out["sha256"] == _sha(build / "bin" / "server")
    assert out["libs"] == {
        str(build / "bin" / "libone.so.0"): _sha(build / "bin" / "libone.so.0.1"),
        str(build / "lib" / "libtwo.so"): _sha(build / "lib" / "libtwo.so"),
    }
    # A script, or a file outside the mount roots, has no libraries to record.
    script = build / "bin" / "run.sh"
    script.write_text("#!/bin/sh\n")
    assert agent_main({"op": "file_sha256", "path": str(script), "libs": True})["libs"] == {}
    assert agent_main({"op": "file_sha256", "path": str(system / "libc.so.6"), "libs": True})["ok"] is False
    assert agent_main({"op": "file_sha256", "paths": [str(system / "libc.so.6")]})["ok"] is False


def _native(build, system) -> dict:
    return {
        "runtime": "native", "binary": str(build / "bin" / "server"), "args": ["--port", "8014"],
        "env": {"LD_LIBRARY_PATH": str(system)}, "image": "", "mounts": [], "network": "host", "container": "",
    }


def _bundled(build, system, placement="head", roles=("head",)) -> dict:
    save_recipe(MODEL, {
        "kind": "pinned", "engine": "llamacpp", "placement": placement, "port": 8014,
        "imported_from": "script:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
        "roles": {role: _native(build, system) for role in roles},
    })
    plan = plan_bundle({"library": [{"id": MODEL, "title": "Example"}]}, _Sparks(), MODEL)
    assert plan.view()["ok"], plan.problems
    return plan


def test_bundle_records_binary_and_libraries_and_copies_nothing(tree):
    build, system = tree
    plan = _bundled(build, system)
    view = plan.view()
    head = plan.recipe["roles"]["head"]
    assert view["files"] == []  # nothing is copied into the folder
    assert head["binary_sha256"] == _sha(build / "bin" / "server")
    assert sorted(head["library_sha256"]) == [str(build / "bin" / "libone.so.0"), str(build / "lib" / "libtwo.so")]
    assert "libc.so.6" not in json.dumps(head["library_sha256"])
    assert any("recorded 2 libraries" in note for note in view["notes"])


@pytest.mark.parametrize("tree", [1, 2, 3, 4], indirect=True)
def test_a_changed_library_refuses_with_one_sentence_on_each_role(tree):
    build, system = tree
    n = len(CFG.nodes)
    roles = ROLES[:n]
    plan = _bundled(build, system, placement="both" if n > 1 else "head", roles=roles)
    sparks = _Sparks()
    assert pin_conflicts(sparks, plan.recipe) == []
    checks = [req for _host, req in sparks.ops if req.get("op") == "file_sha256"]
    assert len(checks) == n and all(len(req["paths"]) == 3 for req in checks)  # one call per role: binary + 2 libraries
    (build / "lib" / "libtwo.so").write_bytes(_elf([], payload=b"rebuilt"))
    lib = str(build / "lib" / "libtwo.so")
    got = pin_conflicts(_Sparks(), plan.recipe)
    assert len(got) == n
    assert got == [f"{lib} on spark-{i} does not match the SHA-256 this recipe records, so it will not start" for i in range(1, n + 1)]
    os.remove(build / "bin" / "libone.so.0")
    got = pin_conflicts(_Sparks(), plan.recipe)
    assert any(c.endswith("so it will not start") and "libone.so.0 is missing on" in c for c in got)


def test_library_pins_must_be_absolute_paths_to_hashes(tree):
    build, system = tree
    role = _native(build, system)
    for bad in ({"relative/lib.so": "a" * 64}, {"/x/lib.so": "nothex"}, ["/x/lib.so"], {"/x/../lib.so": "a" * 64}):
        recipe = {"roles": {"head": {**role, "library_sha256": bad}}}
        assert pin_conflicts(_Sparks(), recipe) == ["the head role's library_sha256 must map absolute paths to 64 hex characters"]


@pytest.mark.parametrize(("tree", "placement"), [(1, "head"), (2, "head"), (2, "worker"), (2, "both"), (4, "both")], indirect=["tree"])
def test_an_empty_folder_recipe_has_a_version_and_shows_each_spark(tree, placement):
    build, system = tree
    roles = ("head", "worker") if placement == "both" else (placement,)
    plan = _bundled(build, system, placement=placement, roles=roles)
    ok, said = write_bundle(plan, stamp="20261122")
    assert ok, said
    root = folder_path(MODEL, plan.name)
    assert sorted(os.listdir(root)) == ["recipe.toml"]
    with using_recipe(MODEL, plan.name):  # library paths are quoted TOML keys and read back
        assert load_recipe(MODEL)["roles"][roles[0]]["library_sha256"] == plan.recipe["roles"][roles[0]]["library_sha256"]
    found = rf.scan(root)
    assert found.entries == [] and found.problems == [] and len(found.version) == rf.VERSION_CHARS
    _code, listed = named.list_named(MODEL)
    row = next(item for item in listed["recipes"] if item["name"] == plan.name)
    assert row["folder"] is True and row["version"] == found.version
    catalog = {"library": [{"id": MODEL, "title": "Example", "format": "gguf", "gib": 8, "engines": ["llamacpp"],
                            "local": {node.name: str(build) for node in CFG.nodes}, "port": 8014}],
               "hosts": {node.name: {"uma_gib": 121, "floor_gib": 6, "margin_gib": 4} for node in CFG.nodes},
               "exclusive_groups": {}, "never_touch": {}}
    sparks = _Sparks()
    with using_recipe(MODEL, plan.name):
        _code, planned = plan_model(catalog, sparks, MODEL, {})
    files = planned["files"]
    assert files["version"] == found.version and files["count"] == 0 and files["folder"] == str(root)
    assert planned["ok"], planned["conflicts"]
    want = sorted(CFG.node_by_role(role).name for role in roles)
    assert sorted(row["host"] for row in files["sparks"].values()) == want
    assert {row["state"] for row in files["sparks"].values()} == {"missing"}  # the page reads "not copied yet"
    assert not any(req.get("op") == "files_stage" for _host, req in sparks.ops)  # a plan copies nothing
    # A Load makes the (empty) copy on each Spark it uses; the page then reads "copy matches".
    why, states = rf.copy_to_sparks(_Sparks(), files)
    assert why == "" and {row["state"] for row in states.values()} == {"match"}
    assert {row["state"] for row in rf.check_sparks(_Sparks(), files).values()} == {"match"}
    # Settings count toward the version, so a recorded hash that changes makes a new version.
    before = found.version
    text = (root / "recipe.toml").read_text()
    (root / "recipe.toml").write_text(text.replace(_sha(build / "bin" / "server"), "0" * 64))
    assert rf.scan(root).version != before


def test_a_linked_recipe_toml_is_a_problem(tmp_path):
    (tmp_path / "real.toml").write_text('kind = "pinned"\n')
    os.symlink(tmp_path / "real.toml", tmp_path / "recipe.toml")
    assert rf.scan(tmp_path).problems == ["recipe.toml in the recipe folder must be a regular file"]


def test_the_folder_files_version_still_ignores_nothing(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.py").write_text("x\n")
    first = rf.scan(tmp_path).version
    (tmp_path / "recipe.toml").write_text('kind = "pinned"\n')
    assert rf.scan(tmp_path).version != first
    assert copy.copy(rf.scan(tmp_path).version) == rf.scan(tmp_path).version


def test_a_malformed_elf_is_read_as_no_libraries_not_a_crash(tmp_path):
    good = _elf(["libx.so"], runpath="$ORIGIN")
    bad = {"empty": b"", "short": good[:70], "elf32": b"\x7fELF\x01\x01" + good[6:], "big_endian": b"\x7fELF\x02\x02" + good[6:]}
    for name, at in (("phoff", 0x20), ("dyn_offset", 64 + 56 + 8), ("load_vaddr", 64 + 8)):
        data = bytearray(good)
        struct.pack_into("<Q", data, at, (1 << 64) - 1)
        bad[name] = bytes(data)
    for name, data in bad.items():
        path = tmp_path / name
        path.write_bytes(data)
        assert agent._elf_dynamic(str(path)) is None, name


def test_a_library_in_the_tree_that_is_malformed_is_still_recorded(tree):
    build, system = tree
    (build / "lib" / "libtwo.so").write_bytes(b"\x7fELF\x02\x01" + b"\xff" * 80)
    out = agent_main({"op": "file_sha256", "path": str(build / "bin" / "server"), "libs": True})
    assert str(build / "lib" / "libtwo.so") in out["libs"]


def test_a_library_found_through_its_loaders_rpath_is_recorded(tree):
    """libone has no RPATH or RUNPATH; the loader then searches the binary's RPATH."""
    build, system = tree
    (build / "bin" / "server").write_bytes(_elf(["libone.so.0"], rpath="$ORIGIN:$ORIGIN/../lib"))
    (build / "bin" / "libone.so.0.1").write_bytes(_elf(["libtwo.so"]))
    out = agent_main({"op": "file_sha256", "path": str(build / "bin" / "server"), "libs": True})
    assert sorted(out["libs"]) == [str(build / "bin" / "libone.so.0"), str(build / "lib" / "libtwo.so")]
    # A RUNPATH on the binary is not inherited: libone then cannot find libtwo.
    (build / "bin" / "server").write_bytes(_elf(["libone.so.0"], runpath="$ORIGIN:$ORIGIN/../lib"))
    out = agent_main({"op": "file_sha256", "path": str(build / "bin" / "server"), "libs": True})
    assert sorted(out["libs"]) == [str(build / "bin" / "libone.so.0")]


def test_the_library_cap_is_one_number_and_a_full_check_is_never_cut_short(tree, monkeypatch):
    from sparksi.recipes import pins

    build, system = tree
    assert pins.LIBS_MAX == agent.LIBS_MAX
    monkeypatch.setattr(agent, "LIBS_MAX", 2)
    monkeypatch.setattr(pins, "LIBS_MAX", 2)
    (build / "lib" / "libthree.so").write_bytes(_elf([]))
    (build / "bin" / "server").write_bytes(_elf(["libone.so.0", "libtwo.so", "libthree.so"], runpath="$ORIGIN:$ORIGIN/../lib"))
    out = agent_main({"op": "file_sha256", "path": str(build / "bin" / "server"), "libs": True})
    assert len(out["libs"]) == 2  # never more than the cap
    role = {**_native(build, system), "binary_sha256": _sha(build / "bin" / "server"), "library_sha256": out["libs"]}
    assert pin_conflicts(_Sparks(), {"roles": {"head": role}}) == []  # binary + a full set of libraries fit one check
    too_many = {**out["libs"], str(build / "lib" / "libthree.so"): _sha(build / "lib" / "libthree.so")}
    got = pin_conflicts(_Sparks(), {"roles": {"head": {**role, "library_sha256": too_many}}})
    assert got == ["the head role's library_sha256 must map absolute paths to 64 hex characters"]
    paths = [str(build / "bin" / "server"), *too_many]
    assert agent_main({"op": "file_sha256", "paths": paths})["ok"] is False


def test_an_older_recipe_with_only_a_binary_hash_gets_its_libraries_when_bundled(tree):
    build, system = tree
    role = {**_native(build, system), "binary_sha256": _sha(build / "bin" / "server")}
    save_recipe(MODEL, {
        "kind": "pinned", "engine": "llamacpp", "placement": "head", "port": 8014,
        "imported_from": "script:example at 2026-11-01T00:00:00Z", "port_note": "", "rendezvous_ports": [],
        "roles": {"head": role},
    })
    catalog = {"library": [{"id": MODEL, "title": "Example"}]}
    plan = plan_bundle(catalog, _Sparks(), MODEL)
    assert plan.view()["ok"], plan.problems
    assert sorted(plan.recipe["roles"]["head"]["library_sha256"]) == [str(build / "bin" / "libone.so.0"), str(build / "lib" / "libtwo.so")]
    # A binary rebuilt since its hash was recorded: refuse rather than pin the new build's libraries.
    (build / "bin" / "server").write_bytes(_elf(["libone.so.0"], runpath="$ORIGIN", payload=b"rebuilt"))
    plan = plan_bundle(catalog, _Sparks(), MODEL)
    assert not plan.view()["ok"]
    assert plan.problems == [f"{build / 'bin' / 'server'} on spark-1 does not match the SHA-256 this recipe records, so its libraries were not recorded"]
