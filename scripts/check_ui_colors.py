#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail when the dashboard CSS uses a hex color outside the Peelsa card.

Allowed greens are the Peelsa card, plus the near-black and grey neutrals
and white rgba() borders. Any other hex, rgb(), or hsl() is an error.
``--muted-2`` may appear as a token; this script does not decide whether a
caller paints text with it.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "src" / "sparksi" / "static"
SUFFIXES = {".html", ".css", ".js"}

ALLOWED = {
    "133b01",
    "205405",
    "4b9412",
    "6aaa0c",
    "a7df6e",
    "d1f7cb",
    "eefeec",
    "08090a",
    "0f1011",
    "141516",
    "f7f8f8",
    "8a8f98",
    "6b7079",
}

HEX_RE = re.compile(r"#([0-9A-Fa-f]{3,8})\b")
RGB_RE = re.compile(
    r"rgba?\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*[0-9.]+\s*)?\)",
    re.IGNORECASE,
)
HSL_RE = re.compile(r"hsla?\([^)]*\)", re.IGNORECASE)


def _hex6(raw: str) -> str:
    text = raw.lower()
    if len(text) in (3, 4):
        text = "".join(ch * 2 for ch in text[:3])
    return text[:6]


def off_card(text: str) -> list[str]:
    """Return human-readable hits. Empty means the text stays on the card."""
    hits: list[str] = []
    for match in HEX_RE.finditer(text):
        key = _hex6(match.group(1))
        if key not in ALLOWED:
            hits.append(match.group(0))
    for match in RGB_RE.finditer(text):
        red, green, blue = (int(match.group(i)) for i in (1, 2, 3))
        if (red, green, blue) == (255, 255, 255):
            continue
        key = f"{red:02x}{green:02x}{blue:02x}"
        if key not in ALLOWED:
            hits.append(match.group(0))
    for match in HSL_RE.finditer(text):
        hits.append(match.group(0))
    return hits


def scan(root: Path | None = None) -> list[str]:
    root = root or UI_ROOT
    found: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8")
        for hit in off_card(text):
            found.append(f"{path.relative_to(ROOT)}: {hit}")
    return found


def main(argv: list[str]) -> int:
    hits = scan(Path(argv[1]) if len(argv) > 1 else None)
    if hits:
        print("off-card UI color:")
        for hit in hits:
            print(f"  {hit}")
        return 1
    print("ui colors: on card")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
