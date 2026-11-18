# SPDX-License-Identifier: Apache-2.0

"""Named recipes per model. The default lives at <id>.toml; others under named/<id>/."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from sparksi.recipes.pinned_settings import pinned_settings
from sparksi.recipes.service import valid_id, write_recipe
from sparksi.recipes.store import (
    DEFAULT_NAME,
    default_recipe_name,
    load_recipe,
    named_dir,
    named_recipe_path,
    recipe_names,
    recipes_dir,
    set_default_name,
    using_recipe,
    valid_recipe_name,
)
from sparksi.recipes.validate import normalize_speculative

# copied_from is set by the server only: a client cannot unlock a pinned command by sending it.
_FORM_ONLY = ("name", "overwrite", "recipe_name", "recipe_new", "new_name", "copied_from")
NAME_RULE = "Recipe names use letters, digits, ., - and _ (up to 80), starting with a letter or digit."
BLANK = "_blank"


def _bad(model_id: str, name: str) -> tuple[int, dict[str, Any]] | None:
    if not valid_id(model_id):
        return 400, {"error": "bad model id"}
    if not valid_recipe_name(name):
        return 400, {"error": NAME_RULE}
    return None


def _summary(model_id: str, name: str, default: str) -> dict[str, Any]:
    with using_recipe(model_id, name):
        recipe = load_recipe(model_id)
    modified = ""
    try:
        st = named_recipe_path(model_id, name).stat()
        modified = datetime.fromtimestamp(st.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        modified = ""
    kind = str(recipe.get("kind") or "generated")
    copied_from = str(recipe.get("copied_from") or "")
    # A pinned recipe keeps its settings in the launch command; read them from there.
    found = pinned_settings(recipe) if kind == "pinned" else {}
    return {
        "name": name,
        "default": name == default,
        "kind": kind,
        # Save is refused only on an original pinned command; a Save as copy saves in place.
        "locked": kind == "pinned" and not copied_from,
        "copied_from": copied_from,
        "engine": str(recipe.get("engine") or ""),
        "placement": str(recipe.get("placement") or ""),
        "modified": modified,
        # Columns for the recipe list: the main settings, read from the saved file.
        "ctx": _count(recipe.get("ctx")) or found.get("ctx"),
        "max_num_seqs": _count(recipe.get("max_num_seqs")) or found.get("max_num_seqs"),
        "kv_cache_dtype": str(recipe.get("kv_cache_dtype") or found.get("kv_cache_dtype") or ""),
        "speculative": _spec_method(recipe.get("speculative")) or str(found.get("speculative") or ""),
        "tp": found.get("tp"),
        # Name sources for the suggested name: the model the command serves and any quant text.
        "model_hint": str(found.get("model") or ""),
        "quant_hint": str(found.get("quant") or ""),
    }


def _count(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _spec_method(value: Any) -> str:
    mode, _why = normalize_speculative(value)
    return str(mode.get("method") or "") if mode else ""


def _kind(model_id: str, name: str) -> str:
    with using_recipe(model_id, name):
        return str(load_recipe(model_id).get("kind") or "")


def _locked(model_id: str, name: str) -> bool:
    """An original pinned launch command: Save keeps it as it is.

    A Save as copy of one carries ``copied_from``; it is an ordinary recipe that Save updates in place.
    """
    with using_recipe(model_id, name):
        recipe = load_recipe(model_id)
    return str(recipe.get("kind") or "") == "pinned" and not recipe.get("copied_from")


def _keep_pinned(name: str) -> dict[str, Any]:
    """Save never rewrites a pinned launch command; the page answers with Save as."""
    return {
        "error": f"{name} runs a pinned launch command, so Save keeps it as it is. Use Save as to keep your changes as a new recipe.",
        "pinned": True,
        "save_as": True,
    }


def pinned_default_refusal(model_id: str) -> tuple[int, dict[str, Any]] | None:
    """For the older PUT /api/models/<id>/recipe: a pinned default is never rewritten there either."""
    if not valid_id(model_id) or not recipe_names(model_id):
        return None
    name = default_recipe_name(model_id)
    if not _locked(model_id, name):
        return None
    return 409, _keep_pinned(name)


def _form(body: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in body.items() if key not in _FORM_ONLY}


def list_named(model_id: str) -> tuple[int, dict[str, Any]]:
    if not valid_id(model_id):
        return 400, {"error": "bad model id"}
    names = recipe_names(model_id)
    default = default_recipe_name(model_id) if names else ""
    recipes = [_summary(model_id, name, default) for name in names]
    return 200, {"id": model_id, "default": default, "recipes": recipes}


def get_named(model_id: str, name: str) -> tuple[int, dict[str, Any]]:
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    if name not in recipe_names(model_id):
        return 404, {"error": f"no recipe named {name}"}
    with using_recipe(model_id, name):
        recipe = load_recipe(model_id)
    return 200, {"id": model_id, "name": name, "default": name == default_recipe_name(model_id), "recipe": recipe}


def save_named(model_id: str, name: str, body: dict[str, Any], runner: Any = None) -> tuple[int, dict[str, Any]]:
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    if name not in recipe_names(model_id):
        return 404, {"error": f"no recipe named {name}; use Save as"}
    if _locked(model_id, name):
        return 409, _keep_pinned(name)
    with using_recipe(model_id, name):
        code, out = write_recipe(model_id, _form(body), runner)
    if code == 200:
        out["name"] = name
    return code, out


def save_as(model_id: str, body: dict[str, Any], runner: Any = None) -> tuple[int, dict[str, Any]]:
    name = str(body.get("name") or "").strip()
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    names = recipe_names(model_id)
    exists = name in names
    overwrite = body.get("overwrite") is True
    if exists and not overwrite:
        return 409, {"error": f"a recipe named {name} already exists", "exists": True}
    if exists and _locked(model_id, name):
        return 409, _keep_pinned(name)
    if not exists and name == DEFAULT_NAME and names:
        return 400, {"error": "default is reserved for the original recipe"}
    source = str(body.get("recipe_name") or "")
    if body.get("recipe_new"):
        source = ""
    elif source not in names:
        source = default_recipe_name(model_id) if names else ""
    if not names:
        set_default_name(model_id, name)  # first recipe becomes the default
    target = named_recipe_path(model_id, name)
    previous = target.read_bytes() if target.is_file() else None
    if source and source != name:
        source_path = named_recipe_path(model_id, source)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.parent / (target.name + ".tmp")
        tmp.write_bytes(source_path.read_bytes())
        os.replace(tmp, target)
    form = _form(body)
    copied_from = ""
    if source and source != name and (_kind(model_id, source) == "pinned" or form.get("kind") == "pinned"):
        # The copy of a pinned command is the user's own recipe: Save updates it in place.
        copied_from = source
    with using_recipe(model_id, name):
        code, out = write_recipe(model_id, form, runner, copied_from=copied_from)
    if code != 200:
        if previous is None:
            target.unlink(missing_ok=True)
            if not names:
                (named_dir(model_id) / "index.json").unlink(missing_ok=True)
            try:
                named_dir(model_id).rmdir()  # only succeeds when nothing else is there
            except OSError:
                pass
        else:
            target.write_bytes(previous)
        return code, out
    out["name"] = name
    return 200, out


def set_default(model_id: str, name: str) -> tuple[int, dict[str, Any]]:
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    if name not in recipe_names(model_id):
        return 404, {"error": f"no recipe named {name}"}
    current = default_recipe_name(model_id)
    if name == current:
        return list_named(model_id)
    base = recipes_dir() / f"{model_id}.toml"
    moved = named_dir(model_id) / f"{current}.toml"
    src = named_dir(model_id) / f"{name}.toml"
    named_dir(model_id).mkdir(parents=True, exist_ok=True)
    if base.is_file():
        os.replace(base, moved)
    os.replace(src, base)
    set_default_name(model_id, name)
    return list_named(model_id)


def rename_named(model_id: str, name: str, body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    new = str(body.get("new_name") or "").strip()
    bad = _bad(model_id, new)
    if bad is not None:
        return bad
    names = recipe_names(model_id)
    if name not in names:
        return 404, {"error": f"no recipe named {name}"}
    if new == name:
        return list_named(model_id)
    if new == DEFAULT_NAME:
        return 400, {"error": "default is reserved for the original recipe"}
    # A rename moves no launch command: the default only changes its name in
    # index.json and stays at <id>.toml, so Load default recipe is unchanged.
    current = default_recipe_name(model_id)
    if new in names:
        if body.get("overwrite") is not True:
            return 409, {"error": f"a recipe named {new} already exists", "exists": True}
        if new == current:
            return 409, {"error": "the default recipe cannot be replaced by a rename"}
        if _locked(model_id, new):
            return 409, _keep_pinned(new)
        (named_dir(model_id) / f"{new}.toml").unlink(missing_ok=True)
    if name == current:
        set_default_name(model_id, new)
    else:
        os.replace(named_dir(model_id) / f"{name}.toml", named_dir(model_id) / f"{new}.toml")
    return list_named(model_id)


def delete_named(model_id: str, name: str) -> tuple[int, dict[str, Any]]:
    bad = _bad(model_id, name)
    if bad is not None:
        return bad
    if name not in recipe_names(model_id):
        return 404, {"error": f"no recipe named {name}"}
    if name == default_recipe_name(model_id):
        return 409, {"error": "set another recipe as the default first"}
    if _kind(model_id, name) == "pinned" and not any(
        other != name and _kind(model_id, other) == "pinned" for other in recipe_names(model_id)
    ):
        return 409, {"error": "this is the model's only pinned launch command, so it cannot be deleted"}
    (named_dir(model_id) / f"{name}.toml").unlink(missing_ok=True)
    return list_named(model_id)
