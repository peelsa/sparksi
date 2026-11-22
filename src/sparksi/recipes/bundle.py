# SPDX-License-Identifier: Apache-2.0

"""Move an older pinned recipe into its own recipe folder, and tidy old backup files.

``bundle`` reads the files a pinned recipe mounts from each Spark (read only),
puts them in ``recipes/<model-id>/<name>/`` and rewrites the mounts to
``@files/`` and ``@role/`` sources. It also records each role's Docker image ID or
native binary SHA-256 and the KV reserve. A dry run writes nothing. A write
builds the folder in a temporary sibling, checks it, renames it into place and
moves the older recipe file into the backups folder. Nothing is deleted.

``tidy_backups`` moves ``recipes.bak-*`` folders and ``*.bak*`` files into
``backups/<date>/``. It only moves. It never deletes or overwrites.
"""

from __future__ import annotations

import copy
import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sparksi.config import CFG
from sparksi.recipes.folder import MAX_FILE_BYTES, MAX_TOTAL_BYTES, Entry, scan, version_of
from sparksi.recipes.pins import image_id, native_pins
from sparksi.recipes.store import (
    RECIPE_FILE,
    backups_dir,
    contained,
    folder_path,
    is_folder_recipe,
    load_recipe,
    named_recipe_path,
    recipe_names,
    recipes_dir,
    recipe_text,
    save_recipe,
    using_recipe,
    valid_recipe_name,
)

START_SCRIPT = "start.sh"
READ_TIMEOUT_S = 120


@dataclass
class Bundle:
    model_id: str
    name: str
    recipe: dict[str, Any] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)  # folder path -> bytes
    executable: dict[str, bool] = field(default_factory=dict)
    sources: dict[str, list[str]] = field(default_factory=dict)  # folder path -> "host:path" read from
    kept: list[str] = field(default_factory=list)  # mounts left as they are
    notes: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def view(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "ok": not self.problems,
            "problems": list(self.problems),
            "notes": list(self.notes),
            "kept": list(self.kept),
            "files": [
                {
                    "path": rel,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "executable": self.executable.get(rel, False),
                    "from": self.sources.get(rel, []),
                }
                for rel, data in sorted(self.files.items())
            ],
            "recipe": self.recipe,
            "version": self.version(),
        }

    def version(self) -> str:
        """The folder version this bundle will have once written; "" when the recipe cannot be saved."""
        try:
            settings = hashlib.sha256(recipe_text(self.recipe).encode("utf-8")).hexdigest()
        except (TypeError, ValueError):
            return ""
        entries = [
            Entry(rel, hashlib.sha256(data).hexdigest(), len(data), self.executable.get(rel, False))
            for rel, data in self.files.items()
        ]
        return version_of(entries, settings)


def _under(path: str, roots: list[str]) -> bool:
    text = os.path.normpath(path)
    for root in roots:
        base = os.path.normpath(str(root or ""))
        if base in ("", ".", "/"):
            continue
        if text == base or text.startswith(base + "/"):
            return True
    return False


def _weights_roots(catalog: dict[str, Any], model_id: str) -> list[str]:
    roots = [str(CFG.hf_cache() or ""), str(CFG.path("hf_hub") or "")]
    for row in catalog.get("library") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == model_id:
            local = row.get("local") if isinstance(row.get("local"), dict) else {}
            roots.extend(str(value) for value in local.values() if value)
    return [root for root in roots if root]


def _read(runner: Any, host: str, path: str) -> tuple[dict[str, Any] | None, str]:
    try:
        result = runner.invoke(host, {"op": "files_read", "paths": [path]}, timeout=READ_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 — reported as one sentence
        return None, str(exc)
    if not isinstance(result, dict) or not result.get("ok"):
        return None, str((result or {}).get("error") or "no answer")
    if result.get("links"):
        return None, f"{path} holds a link, which a recipe folder does not carry"
    found = result.get("files") or {}
    if not any(found.values()):
        return None, f"{path} is not there"
    return found, ""


def _decode(item: dict[str, Any]) -> bytes | None:
    import base64

    try:
        data = base64.b64decode(str(item.get("data") or ""), validate=True)
    except (ValueError, TypeError):
        return None
    if hashlib.sha256(data).hexdigest() != str(item.get("sha256") or ""):
        return None
    return data


def _args_text(role: dict[str, Any]) -> str:
    return " ".join(str(part) for part in role.get("args") or [])


def plan_bundle(
    catalog: dict[str, Any], runner: Any, model_id: str, name: str = "", keep: list[str] | None = None,
) -> Bundle:
    """Read only: what a bundle of this recipe would hold. Writes nothing."""
    names = recipe_names(model_id)
    name = name or (names[0] if len(names) == 1 else "")
    if not name:
        from sparksi.recipes.store import default_recipe_name

        name = default_recipe_name(model_id)
    out = Bundle(model_id=model_id, name=name)
    if not valid_recipe_name(name) or name not in names:
        out.problems.append(f"{model_id} has no recipe named {name}")
        return out
    if is_folder_recipe(model_id, name):
        out.problems.append(f"{model_id} recipe {name} is already in a recipe folder")
        return out
    with using_recipe(model_id, name):
        recipe = copy.deepcopy(load_recipe(model_id))
    if recipe.get("kind") != "pinned" or not isinstance(recipe.get("roles"), dict):
        out.problems.append(f"{model_id} recipe {name} is not a pinned recipe, so there is nothing to bundle")
        return out
    if runner is None:
        out.problems.append("no Sparks to read from")
        return out
    skip = _weights_roots(catalog, model_id)
    keep_set = {os.path.normpath(item) for item in keep or []}
    # container path -> (folder path, bytes by file) for the first role that mounted it
    by_target: dict[str, dict[str, bytes]] = {}
    for role_name, role in sorted(recipe["roles"].items()):
        if not isinstance(role, dict):
            continue
        role_name = str(role_name)
        host = CFG.node_by_role(role_name).name
        mounts = []
        for mount in role.get("mounts") or []:
            text = str(mount)
            source, _sep, rest = text.partition(":")
            target, _sep2, opts = rest.partition(":")
            if (
                not source.startswith("/") or not target.startswith("/")
                or _under(source, skip) or os.path.normpath(source) in keep_set
            ):
                if source.startswith("/"):
                    out.kept.append(f"{role_name}: {text}")
                mounts.append(text)
                continue
            found, why = _read(runner, host, source)
            if found is None:
                out.problems.append(f"could not read {source} on {host} ({why})")
                mounts.append(text)
                continue
            files: dict[str, bytes] = {}
            modes: dict[str, bool] = {}
            for full, item in found.items():
                if not item:
                    continue
                data = _decode(item)
                if data is None:
                    out.problems.append(f"{full} on {host} did not arrive intact")
                    continue
                if len(data) > MAX_FILE_BYTES:
                    out.problems.append(
                        f"{full} on {host} is larger than {MAX_FILE_BYTES >> 20} MiB, so leave its mount as it is with --keep"
                    )
                    continue
                rel = "" if full == source else os.path.relpath(full, source).replace(os.sep, "/")
                files[rel] = data
                modes[rel] = bool(item.get("executable"))
            single = list(files) == [""]
            target_rel = target.strip("/")
            if single and target in _args_text(role).split() and target_rel.endswith(".sh"):
                # The role's start script (D3: one per role, never merged).
                base = f"roles/{role_name}/{START_SCRIPT}"
                new_source = f"@role/{START_SCRIPT}"
            else:
                earlier = by_target.get(target)
                if earlier is None or earlier == files:
                    base = f"files/{target_rel}"
                    new_source = f"@files/{target_rel}"
                    by_target.setdefault(target, files)
                else:
                    base = f"roles/{role_name}/{target_rel}"
                    new_source = f"@role/{target_rel}"
                    out.notes.append(f"{target} differs between Sparks; the {role_name} copy is kept under roles/{role_name}/")
            for rel, data in files.items():
                path = base if rel == "" else f"{base}/{rel}"
                held = out.files.get(path)
                if held is not None and held != data:
                    out.problems.append(f"two sources give different files for {path}")
                    continue
                out.files[path] = data
                out.executable[path] = modes[rel]
                out.sources.setdefault(path, []).append(f"{host}:{source}" + ("" if rel == "" else f"/{rel}"))
            mounts.append(new_source + ":" + target + (":" + opts if opts else ""))
        if "mounts" in role:
            role["mounts"] = mounts
        _record_pins(runner, host, role_name, role, out)
    if sum(len(data) for data in out.files.values()) > MAX_TOTAL_BYTES:
        out.problems.append(
            f"the files to bundle add up to more than {MAX_TOTAL_BYTES >> 20} MiB, so leave the largest mounts as they are with --keep"
        )
    reserve = _card_reserve(catalog, model_id)
    if reserve is not None and not (isinstance(recipe.get("memory"), dict) and "kv_reserve_gib" in recipe["memory"]):
        recipe["memory"] = {"kv_reserve_gib": reserve}
    out.recipe = recipe
    return out


def _record_pins(runner: Any, host: str, role_name: str, role: dict[str, Any], out: Bundle) -> None:
    if role.get("runtime") == "native":
        binary = str(role.get("binary") or "")
        if binary.startswith("/") and not role.get("binary_sha256"):
            env = role.get("env") if isinstance(role.get("env"), dict) else {}
            ld_path = str(env.get("LD_LIBRARY_PATH") or role.get("ld_library_path") or "")
            read = native_pins(runner, host, binary, ld_path)
            if not read or not read[0]:
                out.problems.append(f"could not read the SHA-256 of {binary} on {host}")
                return
            role["binary_sha256"] = read[0]
            if read[1] and not role.get("library_sha256"):
                # Libraries it loads from its own build tree; system libraries are left out.
                role["library_sha256"] = dict(read[1])
                out.notes.append(f"recorded {len(read[1])} libraries {binary} loads from its build tree")
        return
    image = str(role.get("image") or "")
    if image and not role.get("image_id"):
        found = image_id(runner, host, image)
        if found:
            role["image_id"] = found
        else:
            out.problems.append(f"could not read the image ID of {image} on {host}")


def _card_reserve(catalog: dict[str, Any], model_id: str) -> float | None:
    """The KV reserve the model card gives today: its card file, else its catalog row."""
    from sparksi.models import _load_override

    values = [_load_override(model_id).get("kv_reserve_gib")]
    for row in catalog.get("library") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == model_id:
            values.append(row.get("kv_reserve_gib"))
    for value in values:
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def write_bundle(bundle: Bundle, stamp: str | None = None) -> tuple[bool, str]:
    """Write the folder once and move the older recipe file to backups. Returns (ok, sentence)."""
    if bundle.problems:
        return False, bundle.problems[0]
    model_id, name = bundle.model_id, bundle.name
    final = folder_path(model_id, name)
    if final.exists():
        return False, f"{final} already exists, so nothing was written"
    old = named_recipe_path(model_id, name)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{name}.bundle-", dir=final.parent))
    try:
        for rel, data in sorted(bundle.files.items()):
            path = contained(tmp, *rel.split("/"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "xb") as handle:
                handle.write(data)
            os.chmod(path, 0o755 if bundle.executable.get(rel) else 0o644)
        found = scan(tmp)
        if found.problems:
            raise OSError(found.problems[0])
        want = {rel: hashlib.sha256(data).hexdigest() for rel, data in bundle.files.items()}
        if {e.path: e.sha256 for e in found.entries} != want:
            raise OSError("the written folder does not match what was read")
        (tmp / RECIPE_FILE).write_text('kind = "pinned"\n', encoding="utf-8")
        os.rename(tmp, final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    with using_recipe(model_id, name):
        save_recipe(model_id, bundle.recipe)
    moved = ""
    if old.is_file():
        dest = contained(backups_dir(stamp), "bundled", model_id, f"{name}.toml")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return True, f"wrote {final}; the older file stays at {old} because {dest} exists"
        os.replace(old, dest)
        moved = f"; moved the older file to {dest}"
    return True, f"wrote {final}{moved}"


def backup_moves(stamp: str | None = None) -> list[tuple[Path, Path]]:
    """(from, to) for every old backup: recipes.bak-* folders, *.bak* files in the config and recipes folders."""
    recipes = recipes_dir()
    config = recipes.parent
    dest = backups_dir(stamp)
    moves: list[tuple[Path, Path]] = []
    if config.is_dir():
        for child in sorted(config.iterdir()):
            if child.is_symlink():
                continue
            if child.is_dir() and child.name.startswith("recipes.bak"):
                moves.append((child, dest / child.name))
            elif child.is_file() and ".bak" in child.name:
                moves.append((child, dest / child.name))
    if recipes.is_dir():
        for current, dirs, files in os.walk(recipes):
            dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not os.path.islink(os.path.join(current, d)))
            if RECIPE_FILE in files:
                # A recipe folder: its files/ and roles/ are what the recipe mounts, whatever their names.
                dirs[:] = []
            for file_name in sorted(files):
                full = Path(current) / file_name
                if ".bak" in file_name and full.is_file() and not full.is_symlink():
                    rel = full.relative_to(recipes)
                    moves.append((full, dest / "recipes" / rel))
    return moves


def tidy_backups(stamp: str | None = None, dry_run: bool = True) -> list[dict[str, str]]:
    """Move old backups into backups/<date>/. Never deletes; never overwrites."""
    rows = []
    for src, dst in backup_moves(stamp):
        row = {"from": str(src), "to": str(dst), "done": "no"}
        if dst.exists():
            row["note"] = "already there; left in place"
        elif not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)
            row["done"] = "yes"
        rows.append(row)
    return rows
