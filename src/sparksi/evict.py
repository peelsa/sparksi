# SPDX-License-Identifier: Apache-2.0

"""Drop the page cache for one weight directory. Nothing else.

Invoked only as ``python3 -m sparksi.evict /absolute/path``. The path is one
argument. The command does not use a shell or flush every cached page.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

_ARGV = ("python3", "-m", "sparksi.evict")
_DENY = ("/tmp", "/var/tmp", "/dev", "/proc", "/sys", "/root")


def module_sha256() -> str:
    """Hash of this file. The controller compares it with the copy on a node."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def evict_argv(path: str) -> list[str]:
    """The only command Sparksi sends. One absolute path, no shell."""
    return [*_ARGV, str(path)]


def _cache_root(path: str) -> bool:
    """The Hugging Face cache directory, or the hub directory under it."""
    base = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if base == "huggingface" and parent == ".cache":
        return True
    if base == "hub" and parent == "huggingface":
        return True
    return False


def _model_dirs(path: str) -> list[str]:
    if not os.path.isdir(path) or os.path.islink(path):
        return []
    try:
        names = os.listdir(path)
    except OSError:
        return []
    return [name for name in names if name.startswith("models--")]


def eviction_block(path: str, extra_roots: list[str] | None = None) -> str:
    """Why this path must not be evicted. Empty when it is one model's files.

    The cache root and any directory that holds more than one model's
    snapshots are refused. A missing directory is not treated as many models.
    """
    text = os.path.abspath(str(path or "")) if str(path or "").startswith("/") else os.path.normpath(str(path or ""))
    if not text.startswith("/") or text == "/":
        return "path is outside mount roots"
    for root in extra_roots or []:
        norm = os.path.normpath(str(root or ""))
        if norm.startswith("/") and text == norm:
            return "refusing the cache root"
    if _cache_root(text):
        return "refusing the cache root"
    if len(_model_dirs(text)) > 1:
        return "directory holds more than one model's snapshots"
    hub = os.path.join(text, "hub")
    if len(_model_dirs(hub)) > 1:
        return "directory holds more than one model's snapshots"
    return ""


def _refused(path: str) -> str:
    text = os.path.abspath(str(path or ""))
    if not text.startswith("/") or text == "/":
        return "path is outside mount roots"
    parts = text.split("/")
    if ".." in parts:
        return "path is outside mount roots"
    for prefix in _DENY:
        if text == prefix or text.startswith(prefix + os.sep):
            return "path is outside mount roots"
    return ""


def _advise(path: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def advise_tree(path: str) -> int:
    """Advise every regular file under ``path``. Symlinks are skipped."""
    why = _refused(path) or eviction_block(path)
    if why:
        print(f"page cache not cleared: {why}", file=sys.stderr)
        return 1
    root = os.path.abspath(path)
    if not os.path.exists(root):
        print("page cache not cleared: path is missing", file=sys.stderr)
        return 1
    files = [root] if os.path.isfile(root) and not os.path.islink(root) else []
    if os.path.isdir(root) and not os.path.islink(root):
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(dirpath, name))]
            for name in filenames:
                full = os.path.join(dirpath, name)
                if os.path.islink(full) or not os.path.isfile(full):
                    continue
                files.append(full)
    advised = 0
    try:
        for full in files:
            _advise(full)
            advised += 1
    except OSError as exc:
        print(f"page cache not cleared: {exc.strerror or exc}", file=sys.stderr)
        return 1
    print(f"advised {advised} files")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv if argv is None else argv)
    if len(args) != 2:
        print("page cache not cleared: expected one path", file=sys.stderr)
        return 2
    return advise_tree(args[1])


if __name__ == "__main__":
    raise SystemExit(main())
