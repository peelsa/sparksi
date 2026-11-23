# SPDX-License-Identifier: Apache-2.0

"""Per-Spark GPU max-clock cap: query, validate, and apply via nvidia-smi.

The cap is a live nvidia-smi lock, not a setting that survives reboot.
Setting or clearing it uses only ``sudo -n nvidia-smi -i <gpu> -lgc 0,<MHz>``
or ``sudo -n nvidia-smi -i <gpu> -rgc`` (non-interactive, never a prompt).
No other command is run with sudo, and sudoers is never edited. If sudo -n
fails, the error is returned as written. Range limits come from the GPU's
own nvidia-smi report. When the GPU does not report a minimum, 1 MHz is the
lowest cap, so a typed 0 is never applied as a 0 MHz ceiling. The ``-lgc``
minimum is always 0.
The GPU index is read from nvidia-smi, never from the browser.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from sparksi.never import clock_lock_int_text, nvidia_smi_is_clock_lock

QUERY_CSV = [
    "nvidia-smi",
    "--query-gpu=utilization.gpu,temperature.gpu,clocks.current.graphics,clocks.max.graphics",
    "--format=csv,noheader,nounits",
]
QUERY_CLOCK = ["nvidia-smi", "-q", "--display=CLOCK"]
QUERY_INDEX = ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]

_FIELD_NAMES = frozenset({"graphics", "sm", "memory", "video", "min", "max"})
_LOCK_SECTIONS = frozenset({"gpu locked clocks", "locked clocks"})

# Lowest cap accepted when the GPU reports no supported-clock minimum. 0 is refused so a
# box typed as "0" is never read as "no cap" and applied as a 0 MHz ceiling.
MIN_CAP_FLOOR_MHZ = 1

# What Sparksi itself last applied per Spark, in this dashboard process only. Shown when the
# driver does not report a locked clock. Never written to disk; a Spark reboot drops the real
# cap, which is why the page labels this as "set from Sparksi" rather than as a GPU reading.
_MEMO_LOCK = threading.Lock()
_LAST_SET: dict[str, dict[str, Any]] = {}
_SAVING: set[str] = set()


def clock_lock_argv(gpu_index: int, max_mhz: int | None) -> list[str]:
    """Exact argv: ``sudo -n nvidia-smi -i <gpu> -lgc 0,N`` or ``... -rgc``."""
    index = clock_lock_int_text(gpu_index)
    if max_mhz is None:
        argv = ["sudo", "-n", "nvidia-smi", "-i", index, "-rgc"]
    else:
        argv = ["sudo", "-n", "nvidia-smi", "-i", index, "-lgc", "0," + clock_lock_int_text(max_mhz)]
    if not nvidia_smi_is_clock_lock(argv):
        raise ValueError("clock cap command is not a valid lock argument")
    return argv


def parse_gpu_indices(text: str) -> list[int]:
    """Accepted GPU indexes from ``nvidia-smi --query-gpu=index``. Junk lines are dropped."""
    from sparksi.never import CLOCK_LOCK_INT

    out: list[int] = []
    for raw in text.splitlines():
        token = raw.strip()
        if CLOCK_LOCK_INT.fullmatch(token) is None:
            continue
        value = int(token)
        if value not in out:
            out.append(value)
    return out


def _split_label(line: str) -> tuple[str, str | None]:
    """Split a report line at the first colon. No regex."""
    if ":" not in line:
        return line.strip(), None
    left, right = line.split(":", 1)
    return left.strip(), right.strip()


def _is_heading(label: str) -> bool:
    return bool(label) and label[0].isalpha() and all(ch.isalnum() or ch == " " for ch in label)


def _mhz(raw: str) -> int | None:
    text = (raw or "").strip()
    if not text or text.upper() in {"N/A", "[N/A]", "[NOT SUPPORTED]", "NOT SUPPORTED"}:
        return None
    if text.lower().endswith("mhz"):
        text = text[:-3].rstrip()
    if text.startswith("-"):
        digits = text[1:]
        return -int(digits) if digits.isdigit() else None
    return int(text) if text.isdigit() else None


def parse_clock_report(text: str) -> dict[str, int | None]:
    """Read current, max, applied cap, and reported min from ``nvidia-smi -q --display=CLOCK``."""
    current: int | None = None
    max_mhz: int | None = None
    cap: int | None = None
    cap_min: int | None = None
    reported_min: int | None = None
    supported: list[int] = []
    cap_reported = False
    section = ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        label, trailing = _split_label(line)
        name = label.lower() if label.lower() in _FIELD_NAMES else ""
        if not name:
            if _is_heading(label):
                section = label
                if section.lower() in _LOCK_SECTIONS:
                    cap_reported = True
                if section.lower() == "supported clocks" and trailing:
                    # "Supported Clocks : N/A"
                    if _mhz(trailing) is None:
                        continue
            continue
        value = _mhz(trailing or "")
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
        elif key in _LOCK_SECTIONS:
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
        # Some drivers (GB10 on 580.x among them) print no locked-clocks section at all,
        # so "no cap reported" is not the same as "no cap".
        "clock_cap_reported": cap_reported,
    }


def validation_bounds(gpu: dict[str, Any] | None) -> tuple[int, int | None]:
    """Lower and upper MHz accepted for a cap. Upper is None when the GPU did not report a max."""
    gpu = gpu or {}
    reported_min = gpu.get("clock_min_mhz")
    try:
        low = int(reported_min) if reported_min is not None else 0
    except (TypeError, ValueError):
        low = 0
    if low < MIN_CAP_FLOOR_MHZ:
        low = MIN_CAP_FLOOR_MHZ
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


def gpu_indices(runner: Any, host: str, timeout: int = 8) -> tuple[list[int], str | None]:
    """Read GPU indexes with a query (no sudo)."""
    capture = getattr(runner, "capture", None)
    if capture is None:
        return [], "runner cannot query GPU indexes"
    raw = capture(host, list(QUERY_INDEX), timeout=timeout)
    if not isinstance(raw, dict):
        return [], "runner returned a non-object"
    found = parse_gpu_indices(raw.get("stdout") or "")
    if found:
        return found, None
    err = str(raw.get("error") or raw.get("stderr") or "").strip()
    return [], err or "The GPU index could not be read, so the cap was not changed."


def remember_cap(host: str, max_mhz: int | None) -> None:
    """Note what Sparksi just applied on ``host`` (in memory only)."""
    with _MEMO_LOCK:
        _LAST_SET[str(host)] = {"mhz": max_mhz, "at": time.time()}


def with_memo(host: str, gpu: dict[str, Any]) -> dict[str, Any]:
    """Add ``clock_cap_set`` when the driver does not report a locked clock itself."""
    out = dict(gpu or {})
    if out.get("clock_cap_reported") or out.get("clock_cap_mhz") is not None:
        out.pop("clock_cap_set", None)
        return out
    with _MEMO_LOCK:
        memo = _LAST_SET.get(str(host))
        if memo is not None:
            out["clock_cap_set"] = dict(memo)
    return out


def begin_save(host: str) -> bool:
    """Claim the one in-flight save for ``host``. False when another save is running."""
    with _MEMO_LOCK:
        if str(host) in _SAVING:
            return False
        _SAVING.add(str(host))
        return True


def end_save(host: str) -> None:
    with _MEMO_LOCK:
        _SAVING.discard(str(host))


def reset_for_tests() -> None:
    with _MEMO_LOCK:
        _LAST_SET.clear()
        _SAVING.clear()


def apply_clock(runner: Any, host: str, max_mhz: int | None, timeout: int = 8) -> dict[str, Any]:
    """Set or clear the cap with sudo -n, then read the clocks back.

    ``kind`` on a failure is ``"invalid"`` (nothing was run) or ``"failed"`` (sudo, ssh,
    or nvidia-smi refused or errored; the message is passed through as written).
    """
    before = read_clocks(runner, host, timeout=timeout)
    gpu = dict(before.get("gpu") or {})
    err = validate_max_mhz(max_mhz, gpu)
    if err:
        return {"ok": False, "kind": "invalid", "error": err, "gpu": with_memo(host, gpu)}
    indexes, index_err = gpu_indices(runner, host, timeout=timeout)
    if not indexes:
        return {
            "ok": False,
            "kind": "failed",
            "error": index_err or "The GPU index could not be read, so the cap was not changed.",
            "gpu": with_memo(host, gpu),
        }
    run = getattr(runner, "run_gpu_clock", None)
    if run is None:
        return {"ok": False, "kind": "failed", "error": "runner cannot set GPU clocks", "gpu": with_memo(host, gpu)}
    result: dict[str, Any] | None = None
    for done, index in enumerate(indexes):
        result = run(host, index, max_mhz, timeout=timeout)
        ok = isinstance(result, dict) and bool(result.get("ok"))
        if not ok:
            if not isinstance(result, dict):
                message = "runner returned a non-object"
            else:
                message = str(result.get("error") or result.get("stderr") or result.get("stdout") or "nvidia-smi failed").strip()
            if done:
                # An earlier GPU on this Spark already took the new setting.
                remember_cap(host, max_mhz)
            after = read_clocks(runner, host, timeout=timeout)
            return {
                "ok": False,
                "kind": "failed",
                "error": message or "nvidia-smi failed",
                "gpu": with_memo(host, dict(after.get("gpu") or gpu)),
            }
    remember_cap(host, max_mhz)
    after = read_clocks(runner, host, timeout=timeout)
    out_gpu = dict(after.get("gpu") or gpu)
    if max_mhz is None:
        out_gpu.setdefault("clock_cap_mhz", None)
    return {"ok": True, "kind": None, "error": None, "gpu": with_memo(host, out_gpu)}
