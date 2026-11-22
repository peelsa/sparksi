# SPDX-License-Identifier: Apache-2.0

"""Recorded builds a recipe must run: a Docker image ID or a native binary's SHA-256.

A role may record ``image_id`` (``sha256:<64 hex>``, from ``docker image inspect``)
or, for a native role, ``binary_sha256`` and ``library_sha256`` (path -> SHA-256 of
each shared library the binary loads from its own build tree). Each start checks
the Spark that runs the role and refuses when its image or a file does not match.
Nothing is copied: the image, the binary and its libraries stay where they are.
"""

from __future__ import annotations

import json
import re
from typing import Any

from sparksi.config import CFG

_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def image_id(runner: Any, host: str, image: str) -> str | None:
    """The image ID on ``host``; "" when the image is not there; None when the Spark did not answer."""
    capture = getattr(runner, "capture", None)
    if capture is None:
        return None
    try:
        result = capture(host, ["docker", "image", "inspect", image], 8)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(result, dict) or result.get("code") == 255:
        return None
    if not (result.get("ok") or result.get("code") == 0):
        return ""
    try:
        data = json.loads(str(result.get("stdout") or ""))
    except ValueError:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return str(data[0].get("Id") or "")
    return None


def binary_sha256(runner: Any, host: str, path: str) -> str | None:
    """SHA-256 of ``path`` on ``host``; "" when it is not there; None when the Spark did not answer."""
    invoke = getattr(runner, "invoke", None)
    if invoke is None:
        return None
    try:
        result = invoke(host, {"op": "file_sha256", "path": path}, timeout=60)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(result, dict) or not result.get("ok"):
        return None
    return str(result.get("sha256") or "") if result.get("exists") else ""


def native_pins(runner: Any, host: str, binary: str, ld_library_path: str = "") -> tuple[str, dict[str, str]] | None:
    """(binary SHA-256, {library: SHA-256}) read on ``host``; None when the Spark did not answer.

    The binary SHA-256 is "" when the file is not there.
    """
    invoke = getattr(runner, "invoke", None)
    if invoke is None:
        return None
    req = {"op": "file_sha256", "path": binary, "libs": True, "ld_library_path": ld_library_path}
    try:
        result = invoke(host, req, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(result, dict) or not result.get("ok"):
        return None
    if not result.get("exists"):
        return "", {}
    libs = result.get("libs") if isinstance(result.get("libs"), dict) else {}
    return str(result.get("sha256") or ""), {str(k): str(v) for k, v in libs.items()}


def _hashes(runner: Any, host: str, paths: list[str]) -> dict[str, str] | None:
    """Path -> SHA-256 on ``host`` ("" when missing) in one call; None when the Spark did not answer."""
    invoke = getattr(runner, "invoke", None)
    if invoke is None:
        return None
    try:
        result = invoke(host, {"op": "file_sha256", "paths": paths}, timeout=120)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(result, dict) or not result.get("ok") or not isinstance(result.get("files"), dict):
        return None
    out = {}
    for path in paths:
        row = result["files"].get(path)
        if not isinstance(row, dict):
            return None
        out[path] = str(row.get("sha256") or "") if row.get("exists") else ""
    return out


def _library_pins(role: dict[str, Any]) -> dict[str, str] | None:
    """The role's ``library_sha256`` table; None when it is not absolute paths to 64 hex characters."""
    libs = role.get("library_sha256")
    if libs in (None, "", {}):
        return {}
    if not isinstance(libs, dict):
        return None
    out = {}
    for path, digest in libs.items():
        text = str(digest or "").strip().lower()
        if not str(path).startswith("/") or ".." in str(path).split("/") or not _SHA256.fullmatch(text):
            return None
        out[str(path)] = text
    return out


def pin_conflicts(runner: Any, recipe: dict[str, Any]) -> list[str]:
    """One sentence per role file (image, binary or library) that does not match its Spark."""
    conflicts: list[str] = []
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    for name, role in roles.items():
        if not isinstance(role, dict):
            continue
        host = CFG.node_by_role(str(name)).name
        want_image = str(role.get("image_id") or "").strip()
        if want_image:
            image = str(role.get("image") or "")
            if not _IMAGE_ID.fullmatch(want_image) or not image:
                conflicts.append(f"the {name} role's image_id must be sha256: and 64 hex characters, with an image")
            elif runner is not None:
                found = image_id(runner, host, image)
                if found is None:
                    conflicts.append(f"could not check the build of image {image} on {host}")
                elif found != want_image:
                    conflicts.append(f"{host} has a different build of image {image} than this recipe records, so it will not start")
        want_sha = str(role.get("binary_sha256") or "").strip().lower()
        binary = str(role.get("binary") or "")
        libs = _library_pins(role)
        if libs is None:
            conflicts.append(f"the {name} role's library_sha256 must map absolute paths to 64 hex characters")
            continue
        if want_sha and (not _SHA256.fullmatch(want_sha) or not binary.startswith("/")):
            conflicts.append(f"the {name} role's binary_sha256 must be 64 hex characters, with an absolute binary")
            continue
        if runner is None:
            continue
        if want_sha and not libs:
            found_sha = binary_sha256(runner, host, binary)
            if found_sha is None:
                conflicts.append(f"could not check {binary} on {host}")
            elif found_sha != want_sha:
                conflicts.append(f"{binary} on {host} does not match the SHA-256 this recipe records, so it will not start")
            continue
        want = dict(libs)
        if want_sha:
            want[binary] = want_sha
        if not want:
            continue
        found = _hashes(runner, host, sorted(want))
        if found is None:
            conflicts.append(f"could not check {binary or 'the recorded files'} on {host}")
            continue
        for path in sorted(want):
            if found[path] == "":
                conflicts.append(f"{path} is missing on {host}, so it will not start")
            elif found[path] != want[path]:
                conflicts.append(f"{path} on {host} does not match the SHA-256 this recipe records, so it will not start")
    return conflicts
