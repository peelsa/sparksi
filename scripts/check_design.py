#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run Impeccable detect and fail on findings outside the pinned allowlist.

Inter is the required face, so ``overused-font`` is expected. Peel-deep text
on peel-bright is the filled Load pairing; that one advisory is expected.
Any other finding fails the run. The engine binary is ``IMPECCABLE_BIN``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / "src" / "sparksi" / "static" / "index.html"


def accepted(finding: dict) -> bool:
    kind = finding.get("antipattern")
    if kind == "overused-font":
        return True
    if kind == "low-contrast" and finding.get("advisory"):
        snippet = str(finding.get("snippet") or "").lower()
        return "#133b01" in snippet and "#6aaa0c" in snippet
    return False


def main(argv: list[str]) -> int:
    binary = os.environ.get("IMPECCABLE_BIN") or "impeccable"
    target = Path(argv[1]) if len(argv) > 1 else DEFAULT_TARGET
    proc = subprocess.run(
        [binary, "detect", "--json", str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    raw = (proc.stdout or "").strip()
    if not raw:
        sys.stderr.write(proc.stderr or f"detect produced no output (exit {proc.returncode})\n")
        return 1
    try:
        findings = json.loads(raw)
    except json.JSONDecodeError:
        sys.stderr.write(raw + "\n")
        return 1
    if not isinstance(findings, list):
        sys.stderr.write("detect JSON was not a list\n")
        return 1
    blocked = [item for item in findings if not accepted(item)]
    for item in findings:
        state = "allowed" if accepted(item) else "blocked"
        print(f"{state}: {item.get('antipattern')} {item.get('snippet')}")
    if blocked:
        return 1
    print("design detect: allowed findings only")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
