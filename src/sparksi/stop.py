# SPDX-License-Identifier: Apache-2.0

"""Explicit stop of a live resident. Not apply. Does not start anything."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from sparksi.catalog import CatalogError, load_catalog, resident_by_id
from sparksi import compat, protect
from sparksi.config import CFG

VIDEO_STATE = Path(compat.video_service_root(CFG.path("video_service_root"))) / "state"
VIDEO_PID = compat.video_pid_file(VIDEO_STATE)
VIDEO_MODE = VIDEO_STATE / "mode"


def _systemctl_stop(unit: str) -> tuple[int, str]:
    argv = ["sudo", "-n", "/bin/systemctl", "stop", unit]
    try:
        p = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return 1, str(e)
    if p.returncode == 0:
        return 0, ""
    return 1, (p.stderr or p.stdout or "").strip() or f"systemctl stop {unit} failed"


def _stop_video_service() -> tuple[bool, str]:
    pid = None
    if VIDEO_PID.is_file():
        try:
            pid = int(VIDEO_PID.read_text().strip())
        except ValueError:
            pid = None
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        time.sleep(2)
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            VIDEO_PID.unlink(missing_ok=True)
        except OSError:
            pass
    if VIDEO_MODE.parent.is_dir():
        try:
            VIDEO_MODE.write_text("idle\n")
        except OSError:
            pass
    return True, "stopped the video service (video ended)"


def stop_id(
    rid: str,
    catalog: dict[str, Any] | None = None,
    *,
    runner: Callable[[str], tuple[bool, str]] | None = None,
) -> dict[str, Any]:
    cat = catalog or load_catalog()
    row = resident_by_id(cat, rid)
    if not row:
        raise CatalogError(f"unknown resident {rid!r}")
    if runner:
        ok, msg = runner(rid)
        return {"id": rid, "stopped": ok, "detail": msg}
    if protect.is_video_kind(row.get("kind")) or protect.is_video_id(rid) or protect.is_off_limits(row.get("port")):
        ok, msg = _stop_video_service()
        return {"id": rid, "stopped": ok, "detail": msg}
    unit = row.get("unit")
    if not unit:
        raise CatalogError(f"no unit to stop for {rid!r}")
    rc, err = _systemctl_stop(str(unit))
    if rc != 0:
        return {"id": rid, "stopped": False, "detail": err}
    return {"id": rid, "stopped": True, "detail": f"stopped {unit}"}
