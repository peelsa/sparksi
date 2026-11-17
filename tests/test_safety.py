# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from sparksi.never import (
    Refused,
    assert_bind_allowed,
    assert_serve_port,
    refuse_mutator,
    subprocess_is_readonly,
)
from sparksi.serve import allowed_hosts


SRC = Path(__file__).resolve().parents[1] / "src" / "sparksi"


def test_probe_has_no_mutators() -> None:
    text = (SRC / "probe.py").read_text()
    for needle in (
        "systemctl start",
        "systemctl stop",
        "systemctl restart",
        "docker start",
        "docker stop",
        "docker rm",
        "video-mode",
        "tmux",
    ):
        assert needle not in text, needle


def test_readonly_filter() -> None:
    assert subprocess_is_readonly(["systemctl", "is-active", "vllm-coresident"])
    assert subprocess_is_readonly(["systemctl", "show", "x"])
    assert not subprocess_is_readonly(["systemctl", "start", "vllm-coresident"])
    assert not subprocess_is_readonly(["systemctl", "stop", "vllm-counsel-a"])
    assert subprocess_is_readonly(["docker", "ps"])
    assert subprocess_is_readonly(["docker", "stats", "--no-stream"])
    assert not subprocess_is_readonly(["docker", "rm", "-f", "vllm-brain-a"])
    assert not subprocess_is_readonly(["docker", "compose", "down"])
    assert not subprocess_is_readonly(["tmux", "list-sessions"])
    assert not subprocess_is_readonly(["/home/USER/scripts/video-mode.sh", "status"])
    assert subprocess_is_readonly(["nvidia-smi", "--query-compute-apps=pid"])
    assert subprocess_is_readonly(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    assert not subprocess_is_readonly(["nvidia-smi", "--gpu-reset"])
    assert not subprocess_is_readonly(["nvidia-smi", "-r"])
    assert not subprocess_is_readonly(["nvidia-smi", "--persistence-mode", "1"])


def test_bind_rejects_lan_and_wildcard() -> None:
    with pytest.raises(Refused):
        assert_bind_allowed(["0.0.0.0"])
    with pytest.raises(Refused):
        assert_bind_allowed(["192.0.2.11"])
    with pytest.raises(Refused):
        assert_bind_allowed(["198.51.100.11"])
    assert_bind_allowed(["127.0.0.1", "100.64.0.11"])


def test_allowed_hosts_from_catalog() -> None:
    hosts = allowed_hosts(
        {
            "host": {
                "bind_allow": ["127.0.0.1", "100.64.0.11"],
                "bind_deny": ["0.0.0.0", "192.0.2.11"],
            }
        }
    )
    assert hosts == ["127.0.0.1", "100.64.0.11"]
    with pytest.raises(Refused):
        allowed_hosts({"host": {"bind_allow": ["0.0.0.0"], "bind_deny": ["0.0.0.0"]}})


def test_mutator_message() -> None:
    msg = refuse_mutator("swap")
    assert "not implemented" in msg
    assert "swap" in msg


def test_serve_port_denied() -> None:
    with pytest.raises(Refused):
        assert_serve_port(8000)
    with pytest.raises(Refused):
        assert_serve_port(8001)
    with pytest.raises(Refused):
        assert_serve_port(8188)
    assert_serve_port(7500) is None
