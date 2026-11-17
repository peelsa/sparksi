# SPDX-License-Identifier: Apache-2.0

"""Tests for the Network page's config edits (netconfig)."""

from __future__ import annotations

import copy
import os
import tomllib
from datetime import datetime
from pathlib import Path

import pytest

from sparksi.netconfig import (
    MAX_MACHINES,
    change,
    check_same_elsewhere,
    layout,
    render_tables,
    replace_tables,
    save_tables,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config" / "sparksi.example.toml"


def load_example() -> tuple[dict, str]:
    text = EXAMPLE.read_text(encoding="utf-8")
    return tomllib.loads(text), text


def section_text(text: str, header: str) -> str:
    """Slice the text between one header and the next header line."""
    lines = text.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("[")]
    begin = next(i for i in starts if lines[i].strip().startswith(header))
    after = [i for i in starts if i > begin]
    end = after[0] if after else len(lines)
    return "".join(lines[begin:end])


def example_nodes(data: dict) -> list[dict]:
    return copy.deepcopy(data["nodes"])


def example_endpoints(data: dict) -> list[dict]:
    return copy.deepcopy(data.get("endpoints", []))


def write_site(tmp_path: Path, text: str, mode: int | None = None) -> Path:
    path = tmp_path / "sparksi.toml"
    path.write_text(text, encoding="utf-8")
    if mode is not None:
        os.chmod(path, mode)
    return path


# 1. head/worker swap through save_tables


def test_save_swap_only_reorders_nodes(tmp_path: Path) -> None:
    data, text = load_example()
    path = write_site(tmp_path, text, mode=0o600)
    nodes = example_nodes(data)
    nodes[0], nodes[1] = nodes[1], nodes[0]

    result = save_tables(
        path,
        nodes,
        example_endpoints(data),
        now=datetime(2026, 1, 2, 3, 4, 5),
    )

    assert result == {"changed": True, "backup": "sparksi.toml.bak-20260102-030405-000000"}
    new_text = path.read_text(encoding="utf-8")
    new_data = tomllib.loads(new_text)
    old_no_nodes = {k: v for k, v in data.items() if k != "nodes"}
    new_no_nodes = {k: v for k, v in new_data.items() if k != "nodes"}
    assert new_no_nodes == old_no_nodes
    assert new_data["nodes"] == nodes

    backup = path.with_name("sparksi.toml.bak-20260102-030405-000000")
    assert backup.is_file()
    assert backup.read_text(encoding="utf-8") == text
    assert path.stat().st_mode & 0o777 == 0o600


def test_save_swap_keeps_other_sections_byte_identical(tmp_path: Path) -> None:
    data, text = load_example()
    path = write_site(tmp_path, text)
    nodes = example_nodes(data)
    nodes.reverse()

    save_tables(path, nodes, example_endpoints(data))

    new_text = path.read_text(encoding="utf-8")
    for header in ("[engines]", "[controller]"):
        assert section_text(new_text, header) == section_text(text, header)
    old_lines = text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    kept = [
        line
        for i, line in enumerate(old_lines)
        if not _in_nodes_tables(old_lines, i)
    ]
    assert kept == [
        line
        for i, line in enumerate(new_lines)
        if not _in_nodes_tables(new_lines, i)
    ]


def _in_nodes_tables(lines: list[str], index: int) -> bool:
    from sparksi import netconfig

    sections = netconfig._sections(lines)
    for name, start, end in sections:
        if netconfig._is_target(name) and start <= index < end:
            return True
    return False


# 2. idempotent save


def test_save_same_tables_is_noop(tmp_path: Path) -> None:
    data, text = load_example()
    path = write_site(tmp_path, text)

    first = save_tables(path, example_nodes(data), example_endpoints(data))
    assert first == {"changed": False, "backup": None}

    backups = list(tmp_path.glob("*.bak-*"))
    assert backups == []


# 3. add / remove an endpoint


def test_save_add_then_remove_endpoint(tmp_path: Path) -> None:
    data, text = load_example()
    path = write_site(tmp_path, text)
    endpoint = {
        "id": "gpu-box",
        "name": "gpu-box",
        "url": "http://100.64.0.9:11434",
        "engine": "Ollama",
    }

    save_tables(path, example_nodes(data), example_endpoints(data) + [endpoint])
    new_data = tomllib.loads(path.read_text(encoding="utf-8"))
    assert endpoint in new_data["endpoints"]
    assert new_data["endpoints"][-1] == endpoint

    save_tables(path, example_nodes(data), [])
    again = tomllib.loads(path.read_text(encoding="utf-8"))
    assert again.get("endpoints", []) == []
    assert not any(
        line.lstrip().startswith("[[endpoints]]")
        for line in path.read_text(encoding="utf-8").splitlines()
    )


# 4. replace_tables vs headers inside multi-line arrays


def test_replace_tables_ignores_header_like_array_lines(tmp_path: Path) -> None:
    text = (
        "service_commands = [\n"
        '  ["/srv/example/a.sh"],\n'
        "]\n"
        "\n"
        "[[nodes]]\n"
        'id = "a"\n'
        'name = "spark-a"\n'
        'ssh = "spark-a"\n'
        "\n"
        "[[nodes]]\n"
        'id = "b"\n'
        'name = "spark-b"\n'
        'ssh = "spark-b"\n'
    )
    path = write_site(tmp_path, text)
    data = tomllib.loads(text)
    nodes = copy.deepcopy(data["nodes"])
    nodes.reverse()

    new_text = replace_tables(text, nodes, [])
    assert tomllib.loads(new_text)["service_commands"] == [
        ["/srv/example/a.sh"]
    ]
    save_tables(path, nodes, [])
    assert tomllib.loads(path.read_text(encoding="utf-8"))["service_commands"] == [
        ["/srv/example/a.sh"]
    ]


# 5. check_same_elsewhere


def test_check_same_elsewhere_raises_on_other_changes() -> None:
    data, text = load_example()
    nodes = example_nodes(data)
    nodes.reverse()
    good = replace_tables(text, nodes, example_endpoints(data))
    check_same_elsewhere(text, good, nodes, example_endpoints(data))

    assert "ui_port = 7500" in good
    broken = good.replace("ui_port = 7500", "ui_port = 7599", 1)
    with pytest.raises(ValueError):
        check_same_elsewhere(text, broken, nodes, example_endpoints(data))


# 6. write failure leaves the original intact


def test_save_write_failure_keeps_original(tmp_path: Path, monkeypatch) -> None:
    from sparksi import netconfig

    data, text = load_example()
    path = write_site(tmp_path, text)
    nodes = example_nodes(data)
    nodes.reverse()

    def boom(src, dst):
        raise OSError("disk on fire")

    monkeypatch.setattr(netconfig.os, "replace", boom)
    with pytest.raises(OSError):
        save_tables(path, nodes, example_endpoints(data))

    assert path.read_text(encoding="utf-8") == text
    assert list(tmp_path.glob("*.tmp")) == []


# 7. change()


def base_raw() -> dict:
    return {
        "nodes": [
            {
                "id": "a",
                "name": "spark-a",
                "ssh": "spark-a",
                "tailscale": "100.64.0.1",
            },
            {
                "id": "b",
                "name": "spark-b",
                "ssh": "spark-b",
                "tailscale": "100.64.0.2",
            },
        ],
        "endpoints": [],
    }


def machine(name: str, ip: str, endpoints: list | None = None) -> dict:
    return {
        "name": name,
        "dns": f"{name}.example",
        "ips": [ip],
        "endpoints": endpoints or [],
    }


def test_change_swap(tmp_path: Path) -> None:
    raw = base_raw()
    nodes, endpoints, note = change(raw, "role", "head", {"id": "b"})
    assert note == "head and worker swapped"
    assert [n["id"] for n in nodes] == ["b", "a"]
    assert endpoints == []


def test_change_role_already_held() -> None:
    raw = base_raw()
    nodes, endpoints, note = change(raw, "role", "head", {"id": "a"})
    assert note == "no change"
    assert [n["id"] for n in nodes] == ["a", "b"]


def test_change_new_machine_as_head_replaces() -> None:
    raw = base_raw()
    nodes, _, note = change(raw, "add", "head", machine("gpu-box", "100.64.0.9"))
    assert "replaces" in note
    assert nodes[0]["id"] == "gpu-box"
    assert nodes[0]["name"] == "gpu-box"
    assert nodes[1]["id"] == "b"


def test_change_monitor_appends_endpoint() -> None:
    raw = base_raw()
    m = machine(
        "gpu-box",
        "100.64.0.9",
        [{"url": "http://100.64.0.9:11434", "engine": "Ollama"}],
    )
    nodes, endpoints, note = change(raw, "add", "monitor", m)
    assert note == "now monitoring gpu-box at http://100.64.0.9:11434"
    assert endpoints == [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    assert nodes == raw["nodes"]


def test_change_monitor_open_port_only_raises() -> None:
    raw = base_raw()
    m = machine(
        "gpu-box",
        "100.64.0.9",
        [{"url": "http://100.64.0.9:80", "engine": "open port"}],
    )
    with pytest.raises(ValueError):
        change(raw, "add", "monitor", m)


def test_change_monitor_limit() -> None:
    raw = base_raw()
    raw["endpoints"] = [
        {
            "id": f"box-{k}",
            "name": f"box-{k}",
            "url": f"http://100.64.0.{k}:11434",
            "engine": "Ollama",
        }
        for k in (3, 4)
    ]
    m = machine(
        "gpu-box",
        "100.64.0.9",
        [{"url": "http://100.64.0.9:11434", "engine": "Ollama"}],
    )
    with pytest.raises(ValueError):
        change(raw, "add", "monitor", m)
    assert len(raw["endpoints"]) == 2


def test_change_remove_node_raises() -> None:
    raw = base_raw()
    with pytest.raises(ValueError):
        change(raw, "remove", "head", {"id": "a"})


def test_change_remove_endpoint() -> None:
    raw = base_raw()
    raw["endpoints"] = [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    nodes, endpoints, note = change(raw, "remove", "monitor", {"id": "gpu-box"})
    assert endpoints == []
    assert nodes == raw["nodes"]
    assert "removed" in note


def test_change_node_to_monitor_raises() -> None:
    raw = base_raw()
    with pytest.raises(ValueError):
        change(raw, "role", "monitor", {"id": "a"})


def test_change_endpoint_becomes_worker() -> None:
    raw = base_raw()
    raw["endpoints"] = [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    nodes, endpoints, note = change(raw, "role", "worker", {"id": "gpu-box"})
    assert endpoints == []
    assert "replaces" in note
    assert [n["id"] for n in nodes] == ["a", "gpu-box"]


def test_change_unknown_action_and_role() -> None:
    raw = base_raw()
    with pytest.raises(ValueError):
        change(raw, "shuffle", "head", {"id": "a"})
    with pytest.raises(ValueError):
        change(raw, "role", "queen", {"id": "a"})
    with pytest.raises(ValueError):
        change(raw, "add", "queen", machine("gpu-box", "100.64.0.9"))


def test_change_does_not_mutate_raw() -> None:
    raw = base_raw()
    raw["endpoints"] = [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    snapshot = copy.deepcopy(raw)

    change(raw, "role", "head", {"id": "b"})
    assert raw == snapshot

    change(raw, "add", "monitor", dict(machine("other-box", "100.64.0.19"), port=8000))
    assert raw == snapshot

    change(raw, "remove", "monitor", {"id": "gpu-box"})
    assert raw == snapshot


# 8. layout()


def test_layout_shape() -> None:
    raw = base_raw()
    raw["endpoints"] = [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    view = layout(raw, "sparksi-core")

    assert view["controller"] == {"name": "sparksi-core"}
    assert view["head"] == {
        "id": "a",
        "name": "spark-a",
        "tailscale": "100.64.0.1",
    }
    assert view["worker"] == {
        "id": "b",
        "name": "spark-b",
        "tailscale": "100.64.0.2",
    }
    assert view["monitored"] == [
        {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
    ]
    assert view["limit"] == MAX_MACHINES == 4
    assert view["count"] == 3


def test_long_lists_stay_one_item_per_line():
    roots = [f"/srv/models/example-model-{i}" for i in range(4)]
    text = render_tables([{"id": "a", "name": "spark-a", "mount_roots": roots}, {"id": "b", "name": "spark-b", "tags": ["x"]}], [])
    assert 'mount_roots = [\n  "/srv/models/example-model-0",\n' in text
    assert '  "/srv/models/example-model-3",\n]' in text
    assert 'tags = ["x"]' in text
    assert tomllib.loads(text)["nodes"][0]["mount_roots"] == roots


def test_unchanged_entries_keep_their_comments_and_swap_back_is_byte_identical():
    text = EXAMPLE.read_text()
    raw = tomllib.loads(text)
    assert replace_tables(text, raw["nodes"], raw.get("endpoints", [])) == text
    swapped = replace_tables(text, raw["nodes"][::-1], [])
    assert tomllib.loads(swapped)["nodes"] == raw["nodes"][::-1]
    assert "# display name, catalog host key" in swapped  # the head's inline comments moved with it
    assert replace_tables(swapped, raw["nodes"], []) == text
