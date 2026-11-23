# SPDX-License-Identifier: Apache-2.0

"""Per-Spark GPU max-clock cap: query, validate, and apply via nvidia-smi.

The cap is a live nvidia-smi lock, not a setting that survives reboot.
``nvidia-smi -lgc`` and ``-rgc`` require root according to the tool itself;
this module never prefixes sudo. A permission error is returned as-is.
Range limits come from the GPU's own nvidia-smi report. When the GPU does
not report a minimum, 0 is used because that is the lower bound nvidia-smi
documents for ``-lgc MIN,MAX``.
"""

from __future__ import annotations

import re
from typing import Any

from sparksi.never import nvidia_smi_is_clock_lock

QUERY_CSV = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.max.graphics",
    "--format=csv,noheader,nounits",
]
QUERY_CLOCK = ["nvidia-smi", "-q", "--display=CLOCK"]

_MHZ = re.compile(r"(-?\d+)\s*(?:MHz)?\s*$", re.I)
_SECTION = re.compile(r"^([A-Za-z][A-Za-z0-9 ]+?)\s*(?::\s*(.*))?$")
_FIELD = re.compile(r"^(Graphics|SM|Memory|Video|Min|Max)\s*:\s*(.*)$", re.I)


def clock_lock_argv(max_mhz: int | None) -> list[str]:
    """Exact argv for a max cap (``-lgc 0,N``) or a clear (``-rgc``)."""
    if max_mhz is None:
        return ["nvidia-smi", "-rgc"]
    if not isinstance(max_mhz, int) or isinstance(max_mhz, bool) or max_mhz < 0:
        raise ValueError("max clock must be a non-negative integer MHz, or empty to clear")
    argv = ["nvidia-smi", "-lgc", f"0,{max_mhz}"]
    if not nvidia_smi_is_clock_lock(argv):
        raise ValueError("max clock is not a valid lock argument")
    return argv


def _mhz(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
        return None
    match = _MHZ.search(text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def parse_clock_report(text: str) -> dict[str, int | None]:
    """Read current, max, applied cap, and reported min from ``nvidia-smi -q --display=CLOCK``."""
    current: int | None = None
    max_mhz: int | None = None
    cap: int | None = None
    cap_min: int | None = None
    reported_min: int | None = None
    supported: list[int] = []
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        field = _FIELD.match(line)
        heading = _SECTION.match(line) if not field else None
        if heading and heading.group(1) not in {"Graphics", "SM", "Memory", "Video", "Min", "Max"}:
            section = heading.group(1).strip()
            trailing = heading.group(2)
            if section.lower() == "supported clocks" and trailing:
                # "Supported Clocks : N/A"
                if _mhz(trailing) is None:
                    continue
            continue
        if not field:
            continue
        name = field.group(1).lower()
        value = _mhz(field.group(2))
        if value is None:
            continue
        key = section.lower()
        if name == "graphics":
            if key == "clocks":
                current = value
            elif key == "max clocks":
                max_mhz = value
            elif key == "supported clocks":
                supported.append(value)
        elif key in {"gpu locked clocks", "locked clocks"}:
            if name == "max":
                cap = value
            elif name == "min":
                cap_min = value
    if supported:
        reported_min = min(supported)
        if max_mhz is None:
            max_mhz = max(supported)
    return {
        "clock_mhz": current,
        "clock_max_mhz": max_mhz,
        "clock_min_mhz": reported_min,
        "clock_cap_mhz": cap,
        "clock_cap_min_mhz": cap_min,
    }


def validation_bounds(gpu: dict[str, Any] | None) -> tuple[int, int | None]:
    """Lower and upper MHz accepted for a cap. Upper is None when the GPU did not report a max."""
    gpu = gpu or {}
    reported_min = gpu.get("clock_min_mhz")
    try:
        low = int(reported_min) if reported_min is not None else 0
    except (TypeError, ValueError):
        low = 0
    if low < 0:
        low = 0
    reported_max = gpu.get("clock_max_mhz")
    try:
        high = int(reported_max) if reported_max is not None else None
    except (TypeError, ValueError):
        high = None
    return low, high


def validate_max_mhz(max_mhz: int | None, gpu: dict[str, Any] | None) -> str | None:
    """None if the value may be applied; otherwise a one-sentence error."""
    if max_mhz is None:
        return None
    if not isinstance(max_mhz, int) or isinstance(max_mhz, bool):
        return "Enter a whole number of MHz, or leave the box empty to clear the cap."
    low, high = validation_bounds(gpu)
    if high is None:
        return "The GPU did not report a maximum clock, so the cap cannot be checked or set."
    if max_mhz < low or max_mhz > high:
        return f"Enter a clock between {low} and {high} MHz, or leave the box empty to clear the cap."
    return None


def parse_request_mhz(body: dict[str, Any]) -> tuple[int | None, str | None]:
    """Read ``max_mhz`` from a POST body. Empty, missing, or null clears the cap."""
    if "max_mhz" not in body:
        return None, None
    raw = body.get("max_mhz")
    if raw is None or raw == "":
        return None, None
    if isinstance(raw, bool) or isinstance(raw, float):
        return None, "Enter a whole number of MHz, or leave the box empty to clear the cap."
    if isinstance(raw, int):
        if raw < 0:
            return None, "Enter a whole number of MHz, or leave the box empty to clear the cap."
        return raw, None
    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            return None, None
        if text.isdigit():
            return int(text), None
    return None, "Enter a whole number of MHz, or leave the box empty to clear the cap."


def read_clocks(runner: Any, host: str, timeout: int = 8) -> dict[str, Any]:
    """Query live clocks on ``host``. Missing fields stay None."""
    gpu: dict[str, Any] = {}
    capture = getattr(runner, "capture", None)
    if capture is None:
        return {"ok": False, "error": "runner cannot query GPU clocks", "gpu": gpu}
    csv = capture(host, list(QUERY_CSV), timeout=timeout)
    if isinstance(csv, dict) and (csv.get("stdout") or "").strip():
        from sparksi.nodes import parse_nvidia

        parsed = parse_nvidia(csv.get("stdout") or "")
        if parsed:
            gpu.update(parsed)
    report = capture(host, list(QUERY_CLOCK), timeout=timeout)
    report_err = None
    if isinstance(report, dict):
        if (report.get("stdout") or "").strip():
            parsed = parse_clock_report(report.get("stdout") or "")
            for key, value in parsed.items():
                if value is not None:
                    gpu[key] = value
        if not report.get("ok") and not (report.get("stdout") or "").strip():
            report_err = str(report.get("error") or report.get("stderr") or "clock query failed")
    if not gpu and report_err:
        return {"ok": False, "error": report_err, "gpu": gpu}
    if not gpu and isinstance(csv, dict) and not csv.get("ok"):
        return {"ok": False, "error": str(csv.get("error") or csv.get("stderr") or "clock query failed"), "gpu": gpu}
    return {"ok": True, "error": None, "gpu": gpu}


def apply_clock(runner: Any, host: str, max_mhz: int | None, timeout: int = 8) -> dict[str, Any]:
    """Set or clear the cap, then read the clocks back. Never uses sudo."""
    before = read_clocks(runner, host, timeout=timeout)
    gpu = dict(before.get("gpu") or {})
    err = validate_max_mhz(max_mhz, gpu)
    if err:
        return {"ok": False, "error": err, "gpu": gpu}
    try:
        argv = clock_lock_argv(max_mhz)
    except ValueError as exc:
        return {"ok": False, "error": str(exc), "gpu": gpu}
    run = getattr(runner, "run_gpu_clock", None)
    if run is None:
        return {"ok": False, "error": "runner cannot set GPU clocks", "gpu": gpu}
    result = run(host, argv, timeout=timeout)
    if not isinstance(result, dict):
        return {"ok": False, "error": "runner returned a non-object", "gpu": gpu}
    if not result.get("ok"):
        message = str(result.get("error") or result.get("stderr") or result.get("stdout") or "nvidia-smi failed").strip()
        after = read_clocks(runner, host, timeout=timeout)
        return {
            "ok": False,
            "error": message or "nvidia-smi failed",
            "gpu": dict(after.get("gpu") or gpu),
        }
    after = read_clocks(runner, host, timeout=timeout)
    out_gpu = dict(after.get("gpu") or gpu)
    if max_mhz is None:
        # A driver that does not report locked clocks still treated a successful -rgc as cleared.
        if "clock_cap_mhz" not in out_gpu:
            out_gpu["clock_cap_mhz"] = None
    return {"ok": True, "error": None, "gpu": out_gpu}
