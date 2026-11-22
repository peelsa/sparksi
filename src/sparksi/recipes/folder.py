# SPDX-License-Identifier: Apache-2.0

"""The files a recipe folder carries, their hashes, and where they land on each Spark.

A recipe folder (``recipes/<id>/<name>/``) holds ``recipe.toml`` plus the files the
recipe mounts: ``files/`` for every Spark and ``roles/<role>/`` for one. A mount
source written ``@files/<path>`` or ``@role/<path>`` names one of them. Before a
start the folder's files are copied to each Spark under
``<recipe_root>/<id>/<name>/<version>/``. The version is the hash of the file
list, so an edit makes a new folder there and never rewrites one a running
container has mounted.
"""

from __future__ import annotations

import copy
import hashlib
import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparksi.config import ALL_ROLES, CFG

FILES_PREFIX = "@files/"
ROLE_PREFIX = "@role/"
CARRIED = ("files", "roles")
MANIFEST = "MANIFEST.sha256"
VERSION_CHARS = 16
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Entry:
    path: str  # relative, POSIX, under files/ or roles/<role>/
    sha256: str
    size: int
    executable: bool


@dataclass
class Folder:
    root: Path
    entries: list[Entry] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def version(self) -> str:
        return version_of(self.entries)

    def paths(self) -> set[str]:
        return {entry.path for entry in self.entries}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def scan(root: Path) -> Folder:
    """Every carried file under ``root``, sorted. Links, devices and oversize files are problems."""
    found = Folder(root=root)
    total = 0
    for top in CARRIED:
        base = root / top
        if not base.exists():
            continue
        if base.is_symlink() or not base.is_dir():
            found.problems.append(f"{top} in the recipe folder must be a folder")
            continue
        for current, dirs, files in os.walk(base, followlinks=False):
            dirs.sort()
            for name in [*dirs]:
                if (Path(current) / name).is_symlink():
                    rel = (Path(current) / name).relative_to(root).as_posix()
                    found.problems.append(f"recipe file {rel} is a link; copy the real file into the folder")
                    dirs.remove(name)
            for name in sorted(files):
                path = Path(current) / name
                rel = path.relative_to(root).as_posix()
                info = path.lstat()
                if stat.S_ISLNK(info.st_mode):
                    found.problems.append(f"recipe file {rel} is a link; copy the real file into the folder")
                    continue
                if not stat.S_ISREG(info.st_mode):
                    found.problems.append(f"recipe file {rel} is not a regular file")
                    continue
                if top == "roles" and (len(rel.split("/")) < 3 or rel.split("/")[1] not in ALL_ROLES):
                    found.problems.append(f"recipe file {rel} must sit under roles/<role>/ for a known role")
                    continue
                if info.st_size > MAX_FILE_BYTES:
                    found.problems.append(f"recipe file {rel} is larger than {MAX_FILE_BYTES >> 20} MiB")
                    continue
                total += info.st_size
                found.entries.append(Entry(rel, _sha256(path), info.st_size, bool(info.st_mode & 0o111)))
    if total > MAX_TOTAL_BYTES:
        found.problems.append(f"the recipe folder's files add up to more than {MAX_TOTAL_BYTES >> 20} MiB")
    found.entries.sort(key=lambda entry: entry.path)
    return found


def manifest_text(entries: list[Entry]) -> str:
    """``sha256sum -c`` format: ``<hash>  <path>`` per line, sorted by path."""
    return "".join(f"{entry.sha256}  {entry.path}\n" for entry in sorted(entries, key=lambda e: e.path))


def version_of(entries: list[Entry]) -> str:
    """Short hash of the manifest plus which files are executable."""
    digest = hashlib.sha256(manifest_text(entries).encode("utf-8"))
    for entry in sorted(entries, key=lambda e: e.path):
        if entry.executable:
            digest.update(f"x {entry.path}\n".encode("utf-8"))
    return digest.hexdigest()[:VERSION_CHARS]


def node_dir(role: str, model_id: str, name: str, version: str) -> str:
    """Where this version lands on the Spark that runs ``role``; "" when that Spark has no usable root."""
    root = CFG.recipe_root(role)
    if not root:
        return ""
    return f"{root}/{model_id}/{name}/{version}"


def _clean_rel(text: str) -> str | None:
    rel = str(text or "")
    if not rel or rel.startswith("/") or "\\" in rel:
        return None
    parts = rel.rstrip("/").split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    return "/".join(parts)


def _covered(rel: str, paths: set[str]) -> bool:
    """A file in the folder, or a folder of files in it."""
    return rel in paths or any(path.startswith(rel + "/") for path in paths)


def folder_source(source: str, role: str) -> str | None:
    """Folder-relative path for an ``@files/`` or ``@role/`` source; None when it is neither."""
    text = str(source or "")
    if text.startswith(FILES_PREFIX):
        rel = _clean_rel(text[len(FILES_PREFIX):])
        return f"files/{rel}" if rel else ""
    if text.startswith(ROLE_PREFIX):
        rel = _clean_rel(text[len(ROLE_PREFIX):])
        return f"roles/{role}/{rel}" if rel else ""
    return None


def uses_folder(recipe: dict[str, Any]) -> bool:
    """True when any role mounts or runs a file from the recipe folder."""
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    for role in roles.values():
        if not isinstance(role, dict):
            continue
        for mount in role.get("mounts") or []:
            if str(mount).startswith(("@files/", "@role/")):
                return True
        if str(role.get("binary") or "").startswith(("@files/", "@role/")):
            return True
    return False


@dataclass
class Resolved:
    recipe: dict[str, Any]
    version: str = ""
    folder: str = ""
    nodes: dict[str, str] = field(default_factory=dict)  # role -> folder on that Spark
    local: dict[str, str] = field(default_factory=dict)  # folder on a Spark -> folder here
    conflicts: list[str] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)


def _resolve_one(source: str, role: str, base: str, paths: set[str], conflicts: list[str]) -> str:
    rel = folder_source(source, role)
    if rel is None:
        return source
    if not rel:
        conflicts.append(f"bad recipe file path {source}")
        return source
    if not _covered(rel, paths):
        conflicts.append(f"recipe file {rel} is missing from the recipe folder")
    return f"{base}/{rel}" if base else source


def resolve(recipe: dict[str, Any], model_id: str, name: str, root: Path | None) -> Resolved:
    """A copy of ``recipe`` with ``@files/`` and ``@role/`` sources turned into Spark paths.

    ``root`` is the recipe folder here, or None for an older-layout recipe. The
    recipe passed in is never changed.
    """
    out = Resolved(recipe=copy.deepcopy(recipe))
    if not uses_folder(recipe):
        return out
    if root is None:
        out.conflicts.append("this recipe names @files/ or @role/ files but is not in a recipe folder")
        return out
    found = scan(root)
    out.conflicts.extend(found.problems)
    out.entries = found.entries
    out.version = found.version
    out.folder = str(root)
    paths = found.paths()
    for role_name, role in (out.recipe.get("roles") or {}).items():
        if not isinstance(role, dict):
            continue
        base = node_dir(str(role_name), model_id, name, out.version)
        if not base:
            out.conflicts.append(f"the {role_name} Spark has no usable recipe_root in sparksi.toml")
        else:
            out.nodes[str(role_name)] = base
            out.local[base] = str(root)
        mounts = []
        for mount in role.get("mounts") or []:
            text = str(mount)
            source, sep, rest = text.partition(":")
            mounts.append(_resolve_one(source, str(role_name), base, paths, out.conflicts) + sep + rest)
        if "mounts" in role:
            role["mounts"] = mounts
        if "binary" in role:
            role["binary"] = _resolve_one(str(role.get("binary") or ""), str(role_name), base, paths, out.conflicts)
    return out


def local_path(node_path: str, local: dict[str, str]) -> str | None:
    """The file here that a Spark path in a copied folder stands for; None when it is not one."""
    text = os.path.normpath(str(node_path or ""))
    for base, here in local.items():
        if text == base:
            return here
        if text.startswith(base + "/"):
            rest = text[len(base) + 1:]
            if ".." in rest.split("/"):
                return None
            return str(Path(here) / rest)
    return None
