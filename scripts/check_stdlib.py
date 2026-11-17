#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Fail if the node-shipped scripts import anything outside the standard library.

agent.py and announce.py are copied onto nodes and run with the system
python. They must not import the controller package or any third-party module.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

NODE_SCRIPTS = ("src/sparksi/agent.py", "src/sparksi/announce.py")


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1])
    allowed = set(sys.stdlib_module_names)
    bad = 0
    for rel in NODE_SCRIPTS:
        path = root / rel
        if not path.is_file():
            print(f"{rel}: missing")
            bad += 1
            continue
        extra = imported_modules(path) - {"__future__"} - allowed
        if extra:
            print(f"{rel}: non-stdlib imports: {', '.join(sorted(extra))}")
            bad += 1
        else:
            print(f"{rel}: stdlib only")
    if bad:
        print(f"stdlib guard: {bad} finding(s)")
        return 1
    print("stdlib guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
