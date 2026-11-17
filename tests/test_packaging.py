# SPDX-License-Identifier: Apache-2.0

"""Unit rendering and the announcer copy script. Neither touches a live service."""

import os
import stat
import subprocess
from pathlib import Path

from sparksi.cli import main

ROOT = Path(__file__).resolve().parents[1]


def test_install_unit_requires_print(capsys) -> None:
    assert main(["install-unit"]) == 2
    err = capsys.readouterr().err
    assert "--print" in err
    assert "enable" in err


def test_install_unit_prints_controller(capsys) -> None:
    assert main(["install-unit", "--print", "--unit", "controller"]) == 0
    out = capsys.readouterr().out
    assert "WorkingDirectory=/home/USER/src/sparksi" in out
    assert "ExecStart=/home/USER/src/sparksi/.venv/bin/sparksi serve --port 7500" in out
    assert "Environment=SPARKSI_CONFIG=%h/.config/sparksi/sparksi.toml" in out
    assert "WantedBy=default.target" in out
    assert "@PREFIX@" not in out
    exec_lines = [line for line in out.splitlines() if line.startswith("ExecStart=")]
    assert exec_lines
    assert all("0.0.0.0" not in line for line in exec_lines)
    assert "127.0.0.1" in out
    assert "100.64.0.1" in out


def test_install_unit_prints_announcer(capsys) -> None:
    assert main([
        "install-unit", "--print", "--unit", "announce",
        "--node-id", "a", "--controller", "100.64.0.12:7501",
    ]) == 0
    out = capsys.readouterr().out
    from sparksi.config import CFG

    assert "Environment=SPARKSI_NODE_ID=a" in out
    assert "Environment=SPARKSI_CONTROLLER=100.64.0.12:7501" in out
    assert f"Environment=SPARKSI_ADVERTISE={CFG.head.tailscale}" in out
    assert CFG.head.tailscale
    assert "Environment=SPARKSI_ADVERTISE=127.0.0.1" not in out
    assert "ExecStart=/usr/bin/python3 %h/.local/bin/sparksi-announce" in out
    assert "@NODE_ID@" not in out
    assert "@CONTROLLER@" not in out
    assert "@ADVERTISE@" not in out
    assert "/home/" not in out
    assert main([
        "install-unit", "--print", "--unit", "announce",
        "--node-id", "b", "--controller", "192.0.2.1:7501",
    ]) == 0
    worker = capsys.readouterr().out
    assert f"Environment=SPARKSI_ADVERTISE={CFG.worker.tailscale}" in worker
    assert CFG.worker.tailscale != CFG.head.tailscale


def test_announce_unit_uses_the_node_address_and_refuses_loopback(capsys, monkeypatch) -> None:
    from sparksi.config import CFG

    monkeypatch.setattr(CFG.head, "tailscale", "")
    assert main([
        "install-unit", "--print", "--unit", "announce",
        "--node-id", "a", "--controller", "192.0.2.1:7501",
    ]) == 0
    out = capsys.readouterr().out
    assert "Environment=SPARKSI_ADVERTISE=\n" in out or "Environment=SPARKSI_ADVERTISE=" in out
    advertise = [line for line in out.splitlines() if line.startswith("Environment=SPARKSI_ADVERTISE=")]
    assert advertise == ["Environment=SPARKSI_ADVERTISE="]

    monkeypatch.setattr(CFG.head, "tailscale", "127.0.0.1")
    assert main([
        "install-unit", "--print", "--unit", "announce",
        "--node-id", "a", "--controller", "192.0.2.1:7501",
    ]) == 2
    err = capsys.readouterr().err
    assert "refusing" in err


def test_announce_unit_refuses_backslash_and_quotes(capsys, monkeypatch) -> None:
    from sparksi.config import CFG

    for bad in ("192.0.2.10\\", '192.0.2.10"', "192.0.2.10'", "192.0.2.10\\n"):
        monkeypatch.setattr(CFG.head, "tailscale", bad)
        assert main([
            "install-unit", "--print", "--unit", "announce",
            "--node-id", "a", "--controller", "192.0.2.1:7501",
        ]) == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "ExecStart" not in captured.out
        assert "advertised address" in captured.err


def test_install_unit_refuses_wildcard_controller(capsys) -> None:
    assert main([
        "install-unit", "--print", "--unit", "announce",
        "--controller", "0.0.0.0:7501",
    ]) == 2
    err = capsys.readouterr().err
    assert "refusing" in err


def _fake_ssh(tmp_path: Path) -> tuple[Path, Path, Path]:
    remote = tmp_path / "remote-home"
    remote.mkdir()
    log = tmp_path / "ssh.log"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ssh = bin_dir / "ssh"
    ssh.write_text(
        """#!/bin/bash
set -euo pipefail
printf '%s\\n' "$*" >> "$SSH_LOG"
host=""
# Like ssh: options, then "--", then the host; everything after the host is the command.
seen_dd=0
skip=0
cmd=()
for a in "$@"; do
  if [[ -n "$host" ]]; then
    cmd+=("$a")
    continue
  fi
  if [[ "$skip" -eq 1 ]]; then
    skip=0
    continue
  fi
  if [[ "$seen_dd" -eq 0 && "$a" == -o ]]; then
    skip=1
    continue
  fi
  if [[ "$seen_dd" -eq 0 && "$a" == -- ]]; then
    seen_dd=1
    continue
  fi
  if [[ "$seen_dd" -eq 0 && "$a" == -* ]]; then
    echo "fake ssh: option before host: $a" >&2
    exit 255
  fi
  host="$a"
done
if [[ "$seen_dd" -ne 1 ]]; then
  echo "fake ssh: no -- before the host" >&2
  exit 255
fi
echo "HOST=$host" >> "$SSH_LOG"
if [[ "${cmd[0]:-}" == systemctl ]]; then
  if [[ "${cmd[2]:-}" == list-unit-files ]]; then
    echo "LIST" >> "$SSH_LOG"
    printf '%s' "${FAKE_UNITS:-}"
    exit 0
  fi
  echo "RESTART" >> "$SSH_LOG"
  printf '%s\\n' "${cmd[@]}" >> "$SSH_LOG"
  exit 0
fi
export HOME="$FAKE_REMOTE_HOME"
exec "${cmd[@]}"
"""
    )
    ssh.chmod(ssh.stat().st_mode | stat.S_IEXEC)
    return remote, log, bin_dir


def _run_install(
    bin_dir: Path, log: Path, remote: Path, *args: str, units: str = ""
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["FAKE_UNITS"] = units
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["SSH_LOG"] = str(log)
    env["FAKE_REMOTE_HOME"] = str(remote)
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "install-announce.sh"), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_install_announce_backs_up_and_does_not_restart(tmp_path: Path) -> None:
    remote, log, bin_dir = _fake_ssh(tmp_path)
    dest = remote / ".local" / "bin" / "sparksi-announce"
    dest.parent.mkdir(parents=True)
    dest.write_text("previous copy\n")
    proc = _run_install(bin_dir, log, remote, "--host", "node-a")
    assert proc.returncode == 0, proc.stderr
    script = (ROOT / "src" / "sparksi" / "announce.py").read_text(encoding="utf-8")
    assert dest.read_text(encoding="utf-8") == script
    assert dest.with_name("sparksi-announce.bak").read_text(encoding="utf-8") == "previous copy\n"
    recorded = log.read_text(encoding="utf-8").splitlines()
    assert "HOST=node-a" in recorded
    assert any("BatchMode=yes" in line for line in recorded)
    assert "RESTART" not in recorded
    assert not any(line == "systemctl" for line in recorded)
    # The deployed bytes are the repo file. This script does not write a node id
    # or a controller address into it.
    assert dest.read_bytes() == (ROOT / "src" / "sparksi" / "announce.py").read_bytes()


def test_install_announce_restart_is_opt_in(tmp_path: Path) -> None:
    remote, log, bin_dir = _fake_ssh(tmp_path)
    proc = _run_install(bin_dir, log, remote, "--host", "node-b", "--restart")
    assert proc.returncode == 0, proc.stderr
    dest = remote / ".local" / "bin" / "sparksi-announce"
    assert dest.is_file()
    assert not dest.with_suffix(dest.suffix + ".bak").exists()
    recorded = log.read_text(encoding="utf-8").splitlines()
    assert "RESTART" in recorded
    assert "systemctl" in recorded
    assert "--user" in recorded
    assert "restart" in recorded
    assert "sparksi-announce.service" in recorded
    assert "enable" not in recorded


def test_install_announce_requires_host(tmp_path: Path) -> None:
    _remote, log, bin_dir = _fake_ssh(tmp_path)
    proc = _run_install(bin_dir, log, _remote)
    assert proc.returncode == 2
    assert "--host" in proc.stderr
    assert not log.exists() or log.read_text(encoding="utf-8") == ""


def test_install_announce_restart_finds_the_installed_dgx_unit(tmp_path: Path) -> None:
    remote, log, bin_dir = _fake_ssh(tmp_path)
    units = "dgx-announce.service enabled enabled\n"
    proc = _run_install(bin_dir, log, remote, "--host", "node-b", "--restart", units=units)
    assert proc.returncode == 0, proc.stderr
    recorded = log.read_text(encoding="utf-8").splitlines()
    assert "LIST" in recorded
    assert "dgx-announce.service" in recorded
    assert "sparksi-announce.service" not in recorded
    assert "restarted dgx-announce.service on node-b" in proc.stdout


def test_install_announce_restart_takes_a_named_unit(tmp_path: Path) -> None:
    remote, log, bin_dir = _fake_ssh(tmp_path)
    proc = _run_install(bin_dir, log, remote, "--host", "node-b", "--restart", "--unit", "dgx-announce.service")
    assert proc.returncode == 0, proc.stderr
    recorded = log.read_text(encoding="utf-8").splitlines()
    assert "LIST" not in recorded
    assert "dgx-announce.service" in recorded
    bad = _run_install(bin_dir, log, remote, "--host", "node-b", "--restart", "--unit", "x;reboot")
    assert bad.returncode == 2
