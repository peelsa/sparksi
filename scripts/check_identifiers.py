#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Site-identity guard. Fails on site identifiers, private addresses and backups.

The denylist is stored as SHA-256 hashes so this file does not publish the
values it guards against. Extra plain-text words can be supplied by CI from a
secret: SPARKSI_IDENTITY_DENYLIST="word1\\nword2".

Generic checks (no secrets needed):
  * /home/<lowercase-user>/ paths (use /home/USER/)
  * RFC 1918 IPv4 (10/8, 172.16/12, 192.168/16)
  * CGNAT/tailnet 100.64.0.0/10 outside the 100.64.0.0/24 placeholder range
  * *.bak* files, unlock.token, site config files
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import sys
from pathlib import Path

HASHES = {
 "word": [
  "01e76a28977874f8b72265d0d39fa47c4105083556013f84ded1dad7798d01f7",
  "148b21617e48c6f456634cba6e3d7bbafeb70fe58cc60202d957a97d5051e4d4",
  "304ca1638c5effa6832e0e15b958a8f74847efe4df9c3f3187216e921c168fed",
  "382a803dcc0de9618b141c904dde30f933985956df74ce7fd9886494c7a35a60",
  "491d980de148e0c5237013bcc87ee8ed5245fa301ffcfd29b7ae3ed3b79ec06d",
  "65cc5731e18dc4d17148b1bb2dfb57484e9ba7bcd7ae11bc38c6af40ca22a6a7",
  "7baa3d78c929a2733462af9ea246b396f7a583f66a021f530d01e2207da9a013",
  "96d9632f363564cc3032521409cf22a852f2032eec099ed5967c0d000cec607a",
  "9f5a84a6cc3f956445908e80e2ead4cee8c37925d11bf8f78192a32af62eceb0",
  "a03ea09072d789adff29aff6a3758e9294c96ce803915c1456384eaa6e2d2df9",
  "a1d1ff55baa9be435324d11dffbe090166e3d88324cf5ab575ad9f489ebbff39",
  "a51146d38077ff992eba84cf1d5e9b249073503eb74868699016153b031a38f7",
  "af4e572398c5245eaa2aebb12440793aca49988ae6d1cf1e6dde5dc32a72513c",
  "b1417d6a7619d7a4dc7ca778379bbdceb7b0b8e778b99db3a500889b0eb40729",
  "c3d45f26d018654365a6576ccd05138b292f25e3b0f60bdd70551f8fb9b5f28d"
 ],
 "substring": [
  [
   4,
   "a04bcfca8bca030562590798e33268b5119c0e6aa069190af1f9ca5a534cfb6d"
  ],
  [
   5,
   "c4329dad8568556eb9945e766c36e08afe1ba160a696d5ceb4f7c9e6f89abe17"
  ],
  [
   6,
   "01e76a28977874f8b72265d0d39fa47c4105083556013f84ded1dad7798d01f7"
  ],
  [
   6,
   "9a8b86a47c0b442e7ecd54813c43337ff52322a70780e10baf5f0c3f000584ad"
  ],
  [
   7,
   "382a803dcc0de9618b141c904dde30f933985956df74ce7fd9886494c7a35a60"
  ],
  [
   8,
   "304ca1638c5effa6832e0e15b958a8f74847efe4df9c3f3187216e921c168fed"
  ],
  [
   9,
   "9274009c287ae1a0be24627ac5d317b25e1769a0fda657e68740bd5fd2d2e931"
  ],
  [
   36,
   "14aa4f5bf9764c261a3d2071aa7f9cdb8ae491472014e61c86eaed9278afca8e"
  ]
 ]
}

SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "build", "dist", "planning"}
SKIP_FILES = {"LICENSE"}
BAD_NAMES = re.compile(r"(\.bak|unlock\.token$|^sparksi\.toml$|^catalog\.yaml$)")
HOME_RE = re.compile(r"/home/([a-z][a-z0-9_-]*)/")
IPV4_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d{1,3}){3})(?![\d.])")
TOKEN_RE = re.compile(r"[a-z0-9_]+(?:-[a-z0-9_]+)*")
PLACEHOLDER = ipaddress.ip_network("100.64.0.0/24")
CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _tokens(text: str):
    for tok in TOKEN_RE.findall(text):
        yield tok
        if "-" in tok or "_" in tok:
            yield from re.split(r"[-_]", tok)


def check_text(text: str, extra: list[str]) -> list[str]:
    low = text.lower()
    hits: list[str] = []
    words = set(HASHES["word"])
    for tok in set(_tokens(low)):
        if _h(tok) in words:
            hits.append(f"denylisted word (sha256 {_h(tok)[:12]})")
    for length, digest in HASHES["substring"]:
        for i in range(0, max(0, len(low) - length + 1)):
            if _h(low[i:i + length]) == digest:
                hits.append(f"denylisted substring at offset {i} (sha256 {digest[:12]})")
                break
    for w in extra:
        if w and w.lower() in low:
            hits.append(f"secret-list word #{extra.index(w)}")
    for m in HOME_RE.finditer(text):
        hits.append(f"home path /home/{m.group(1)}/")
    for m in IPV4_RE.finditer(text):
        try:
            ip = ipaddress.ip_address(m.group(1))
        except ValueError:
            continue
        if ip in CGNAT and ip not in PLACEHOLDER:
            hits.append(f"tailnet address {ip}")
        elif ip.is_private and ip not in CGNAT and not ip.is_loopback and not ip.is_unspecified \
                and not any(ip in ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24", "198.18.0.0/15")):
            hits.append(f"private address {ip}")
    return hits


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else Path(__file__).resolve().parents[1])
    extra = [w.strip() for w in os.environ.get("SPARKSI_IDENTITY_DENYLIST", "").splitlines() if w.strip()]
    bad = 0
    for path in sorted(root.rglob("*")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if not path.is_file() or path.name in SKIP_FILES:
            continue
        rel = path.relative_to(root)
        if BAD_NAMES.search(path.name):
            print(f"{rel}: file must not be committed")
            bad += 1
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        hits = check_text(str(rel) + "\n" + text, extra)
        for hit in hits:
            print(f"{rel}: {hit}")
        bad += len(hits)
    if bad:
        print(f"identity guard: {bad} finding(s)")
        return 1
    print("identity guard: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
