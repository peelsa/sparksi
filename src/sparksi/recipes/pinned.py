# SPDX-License-Identifier: Apache-2.0

"""Pinned recipes render the imported argv. Only the bind host is rewritten."""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from sparksi import protect
from sparksi.agent import docker_option_refusal, shell_basename
from sparksi.config import ALL_ROLES, CFG
from sparksi.evict import eviction_block
from sparksi.health import health_paths_for, health_timeout_for
from sparksi.models import ModelCard
from sparksi.placement import needs_text, roles_of
from sparksi.recipes.args import apply_edits, parse_args, parse_launch, rendezvous_ports, rewrite_bind
from sparksi.recipes.serve_port import align_recipe, serve_port_conflicts
from sparksi.recipes.engines.common import docker_stop, launch_for
from sparksi.recipes.spec import Built, Launch
from sparksi.recipes.validate import (
    CTX_MIN,
    PORT_MAX,
    PORT_MIN,
    PINNED_ENGINES,
    binary_allowed,
    ctx_ceiling,
    library_path_allowed,
    normalize_speculative,
)

_SHELL = set(";|&$`<>\n\r\\")
# A pinned command runs on the Sparks it was written for: one role, or the head+worker pair.
_PLACES = (*ALL_ROLES, "both")


# Spark folder -> recipe folder here, for files copied from a recipe folder. A
# mounted script there is read from this machine: the Spark copy is checked by
# hash before the start, so both hold the same bytes.
_FOLDER_FILES: ContextVar[dict[str, str]] = ContextVar("sparksi_folder_files", default={})


@contextmanager
def folder_files(local: dict[str, str]):
    token = _FOLDER_FILES.set(dict(local or {}))
    try:
        yield
    finally:
        _FOLDER_FILES.reset(token)


def roots_for(role: str) -> list[str]:
    """Cache root, that node's extra mount roots and its recipe folder root. ``/`` is never a root."""
    node = CFG.node_by_role(role if role in CFG.roles else "head")
    found: list[str] = []
    for item in (CFG.hf_cache(), CFG.path("hf_hub"), *list(node.mount_roots), CFG.node_recipe_root(node)):
        text = os.path.normpath(str(item or ""))
        if text in ("", ".", "/"):
            continue
        if text not in found:
            found.append(text)
    return found


def _meta(value: str) -> bool:
    return any(ch in _SHELL for ch in value)


def extra_docker_argv(text: str) -> tuple[list[str], list[str]]:
    """Approved ``EXTRA_DOCKER_ARGS`` tokens, and one refusal per other flag.

    Approved flags are ``--gpus``, ``--ipc``, ``--shm-size``, ``--ulimit``,
    ``--network host``, ``--init``, ``--rm``, and detach. Anything else refuses the pin.
    """
    raw = str(text or "").strip()
    if not raw:
        return [], []
    try:
        parts = shlex.split(raw)
    except ValueError:
        return [], ["refusing EXTRA_DOCKER_ARGS"]
    approved: list[str] = []
    refused: list[str] = []
    index = 0
    while index < len(parts):
        token = parts[index]
        inline = token.startswith("--") and "=" in token
        flag, value = token.split("=", 1) if inline else (token, "")
        if flag in _EXTRA_BOOL and not inline:
            approved.append(flag)
            index += 1
            continue
        needs_value = flag in _EXTRA_VALUE or flag in _EXTRA_HOST
        if needs_value and not inline:
            if index + 1 >= len(parts):
                refused.append(f"refusing EXTRA_DOCKER_ARGS {flag}")
                break
            value = parts[index + 1]
        if flag in _EXTRA_HOST:
            if value != "host":
                refused.append(f"refusing EXTRA_DOCKER_ARGS {flag}")
            elif inline:
                approved.append(f"{flag}=host")
            else:
                approved.extend([flag, "host"])
            index += 1 if inline else 2
            continue
        if flag in _EXTRA_VALUE and value and not value.startswith("-") and not _meta(value):
            if inline:
                approved.append(f"{flag}={value}")
            else:
                approved.extend([flag, value])
            index += 1 if inline else 2
            continue
        refused.append(f"refusing EXTRA_DOCKER_ARGS {flag}")
        index += 1 if inline or not needs_value else 2
    return approved, refused


def flag_on(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _model_from_args(args: list[str]) -> str:
    for flag in ("--model", "-m", "--model-path"):
        if flag in args:
            index = args.index(flag)
            if index + 1 < len(args):
                return str(args[index + 1])
    if "serve" in args:
        index = args.index("serve")
        if index + 1 < len(args) and not str(args[index + 1]).startswith("-"):
            return str(args[index + 1])
    return ""


def _hub_snapshot(repo: str, revision: str) -> str:
    text = str(repo or "").strip().strip("/")
    if "/" not in text:
        return ""
    hub = str(CFG.path("hf_hub") or "/home/USER/.cache/huggingface/hub").rstrip("/")
    base = f"{hub}/models--" + text.replace("/", "--")
    rev = str(revision or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", rev):
        return f"{base}/snapshots/{rev.lower()}"
    return f"{base}/snapshots"


def _cache_roots() -> list[str]:
    found: list[str] = []
    for item in (CFG.hf_cache(), CFG.path("hf_hub")):
        text = os.path.normpath(str(item or ""))
        if text.startswith("/") and text not in ("/", ".") and text not in found:
            found.append(text)
    return found


def resolve_evict_path(role: dict[str, Any], role_name: str, *, repo: str = "", revision: str = "") -> str:
    """One model's files under this node's mount roots, or empty when none resolves.

    A missing repo, revision, and model path does not fall back to a mounted
    cache. The cache root, and any directory that holds more than one model's
    snapshots, is never a result.
    """
    roots = roots_for(role_name)
    candidates: list[str] = []
    snap = _hub_snapshot(repo, revision)
    if snap:
        candidates.append(snap)
    model = _model_from_args([str(part) for part in (role.get("args") or [])])
    if model:
        if str(role.get("runtime") or "docker") == "native":
            candidates.append(model)
        else:
            host = host_path_from_mounts(model, [str(item) for item in (role.get("mounts") or [])])
            if host:
                candidates.append(host)
    for path in candidates:
        norm = os.path.normpath(str(path or ""))
        if not norm.startswith("/") or not _under(norm, roots):
            continue
        if eviction_block(norm, extra_roots=_cache_roots()):
            continue
        return norm
    return ""


def evict_enabled(recipe: dict[str, Any], role: dict[str, Any]) -> bool:
    """Role override wins. Otherwise the recipe flag applies to every role."""
    if isinstance(role, dict) and "evict_page_cache" in role:
        return flag_on(role.get("evict_page_cache"))
    return flag_on(recipe.get("evict_page_cache"))


def _under(path: str, roots: list[str]) -> bool:
    text = os.path.normpath(str(path or ""))
    if not text.startswith("/"):
        return False
    for root in roots:
        if text == root or text.startswith(root + os.sep):
            return True
    return False


def _mount_source(mount: str) -> str:
    return str(mount or "").split(":", 1)[0]


def _mount_ends(mount: str) -> tuple[str, str]:
    """Host source and container target of one bind. Options after the target are dropped."""
    parts = str(mount or "").split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        return "", ""
    return parts[0], parts[1]


def host_path_from_mounts(path: str, mounts: list[str]) -> str | None:
    """Map a container path to the host path through the longest matching bind.

    ``None`` means no mount covers the path. A file mount matches only that path.
    """
    text = os.path.normpath(str(path or ""))
    if not text.startswith("/"):
        return None
    best = -1
    found: str | None = None
    for mount in mounts:
        host, target = _mount_ends(str(mount))
        if not host or not target:
            continue
        host = os.path.normpath(host)
        target = os.path.normpath(target)
        if not host.startswith("/") or not target.startswith("/"):
            continue
        if text == target or text.startswith(target + os.sep):
            if len(target) > best:
                best = len(target)
                rest = text[len(target):].lstrip("/")
                found = os.path.normpath(os.path.join(host, rest)) if rest else host
    return found


def _model_path_refused(path: str, role: dict[str, Any], roots: list[str]) -> str | None:
    """A model file must sit under a mount root.

    Native recipes are checked as host paths. Container recipes are translated
    through the role's bind mounts first. An uncovered container path is refused,
    and so is a host source outside the mount roots.
    """
    if str(role.get("runtime") or "docker") == "native":
        if not _under(path, roots):
            return "speculative model must be under a mount root"
        return None
    host = host_path_from_mounts(path, [str(item) for item in (role.get("mounts") or [])])
    if not host:
        return "speculative model is not covered by a mount"
    if not _under(host, roots):
        return "speculative model must be under a mount root"
    return None


def _mount_refused(source: str, roots: list[str]) -> str | None:
    raw = str(source or "").strip()
    if not raw.startswith("/"):
        return f"refusing mount {raw} outside model roots"
    path = os.path.normpath(raw)
    if path in ("/", "/etc", "/root") or path.startswith("/etc/") or path.startswith("/root/"):
        return f"refusing mount {path}"
    if path == "/var/run/docker.sock" or path.endswith("/docker.sock"):
        return "refusing mount /var/run/docker.sock"
    if not _under(path, roots):
        return f"refusing mount {path} outside model roots"
    return None


def _explicit_updates(fields: dict[str, Any], explicit: set[str]) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    for key in (
        "ctx", "gpu_mem_util", "kv_cache_memory_bytes", "kv_cache_dtype",
        "max_num_seqs", "port", "tool_parser", "reasoning_parser", "image",
    ):
        if key in explicit and key in fields:
            updates[key] = fields[key]
    if "speculative" in explicit and isinstance(fields.get("speculative"), dict):
        updates["speculative"] = fields["speculative"]
    elif any(key in explicit for key in ("speculative_method", "speculative_model", "speculative_tokens")):
        updates["speculative"] = {
            "method": str(fields.get("speculative_method") or ""),
            "model": str(fields.get("speculative_model") or ""),
            "tokens": fields.get("speculative_tokens") or 0,
        }
    return updates


def _extra_for(role: str, fields: dict[str, Any], explicit: set[str]) -> list[str] | None:
    by_role = fields.get("extra_args_by_role")
    if isinstance(by_role, dict) and role in by_role and ("extra_args" in explicit or "extra_args_by_role" in explicit):
        return [str(item) for item in by_role[role]]
    key = "extra_args_worker" if role == "worker" else "extra_args"
    if key in explicit:
        return [str(item) for item in (fields.get(key) or [])]
    if role == "head" and "extra_args" in explicit:
        return [str(item) for item in (fields.get("extra_args") or [])]
    return None


def _env_for(role: str, fields: dict[str, Any], explicit: set[str], current: dict[str, str]) -> dict[str, str]:
    if "env" in explicit and isinstance(fields.get("env"), dict):
        raw = fields["env"]
        if role in raw and isinstance(raw[role], dict):
            return {str(key): str(value) for key, value in raw[role].items()}
        if raw and not any(key in ("head", "worker") for key in raw):
            return {str(key): str(value) for key, value in raw.items()}
    key = "env_worker" if role == "worker" else "env_head"
    if key in explicit:
        return _env_pairs(fields.get(key))
    if role == "head" and "env_head" not in explicit and "env" in explicit and isinstance(fields.get("env"), list):
        return _env_pairs(fields.get("env"))
    return dict(current)


def _env_pairs(raw: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, list):
        for item in raw:
            text = str(item)
            if "=" not in text:
                continue
            key, value = text.split("=", 1)
            if key:
                out[key] = value
    return out


def apply_overrides(recipe: dict[str, Any], fields: dict[str, Any], explicit: set[str]) -> dict[str, Any]:
    """Copy a pinned recipe and rewrite only the fields the caller edited."""
    updated = copy.deepcopy(recipe)
    try:
        saved_port = int(recipe.get("port") or 0)
    except (TypeError, ValueError):
        saved_port = 0
    updates = _explicit_updates(fields, explicit)
    engine = str(updated.get("engine") or "other")
    roles = updated.get("roles") if isinstance(updated.get("roles"), dict) else {}
    for name, role in roles.items():
        if not isinstance(role, dict):
            continue
        extra = _extra_for(str(name), fields, explicit)
        role["args"] = apply_edits(list(role.get("args") or []), updates, extra, engine)
        role["env"] = _env_for(str(name), fields, explicit, dict(role.get("env") or {}))
        if "image" in updates and role.get("runtime", "docker") == "docker":
            role["image"] = str(updates["image"])
        if role.get("runtime") == "native":
            if "binary" in explicit and fields.get("binary"):
                role["binary"] = str(fields["binary"])
            if "ld_library_path" in explicit:
                env = dict(role.get("env") or {})
                lib = str(fields.get("ld_library_path") or "")
                if lib:
                    env["LD_LIBRARY_PATH"] = lib
                else:
                    env.pop("LD_LIBRARY_PATH", None)
                role["env"] = env
    if "port" in explicit and fields.get("port") not in (None, ""):
        try:
            updated["port"] = int(fields["port"])
        except (TypeError, ValueError):
            pass
    if "placement" in explicit and str(fields.get("placement") or "") in _PLACES:
        updated["placement"] = str(fields["placement"])
    if "engine" in explicit and str(fields.get("engine") or "") in PINNED_ENGINES:
        updated["engine"] = str(fields["engine"])
    for key in ("repo", "revision"):
        if key not in explicit:
            continue
        text = str(fields.get(key) or "").strip()
        if text:
            updated[key] = text
        else:
            updated.pop(key, None)
    if "evict_page_cache" in explicit:
        updated["evict_page_cache"] = flag_on(fields.get("evict_page_cache"))
    # The serve port has to match in --port and in PORT-style variables.
    # A script that reads PORT would otherwise ignore the flag Sparksi appends.
    try:
        chosen = int(updated.get("port") or 0)
    except (TypeError, ValueError):
        chosen = 0
    if chosen:
        align_recipe(updated, chosen, saved_port=saved_port or None)
    return updated


_TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
_EXTRA_BOOL = {"--init", "--rm", "-d", "--detach"}
_EXTRA_VALUE = {"--gpus", "--ipc", "--shm-size", "--ulimit"}
_EXTRA_HOST = {"--network", "--net"}


def _hub_root() -> str:
    return str(CFG.path("hf_hub") or "/home/USER/.cache/huggingface/hub").rstrip("/")


def snapshot_directory(repo: str, revision: str) -> str:
    """One model's snapshot directory, or empty when the revision is not a hash.

    This is the only directory a saved repo and revision name. It is never the
    cache root.
    """
    text = str(repo or "").strip().strip("/")
    rev = str(revision or "").strip().lower()
    if "/" not in text or not re.fullmatch(r"[0-9a-f]{40}", rev):
        return ""
    folder = "models--" + text.replace("/", "--")
    return f"{_hub_root()}/{folder}/snapshots/{rev}"


def _repo_cache_dir(repo: str) -> str:
    text = str(repo or "").strip().strip("/")
    return "models--" + text.replace("/", "--")


def _capture_code(runner: Any, host: str, argv: list[str]) -> tuple[int | None, str]:
    """Exit code of a read-only argv. Stdout is returned only for a revision hash."""
    capture = getattr(runner, "capture", None) if runner is not None else None
    if capture is None or not host:
        return None, ""
    try:
        result = capture(host, argv, 8)
    except Exception:  # noqa: BLE001 — a failed check is incomplete, and the payload is dropped
        return None, ""
    if not isinstance(result, dict):
        return None, ""
    if result.get("ok") or result.get("code") == 0:
        return 0, str(result.get("stdout") or "")
    try:
        return int(result.get("code")), ""
    except (TypeError, ValueError):
        return None, ""


def _revision_hash(text: str) -> str:
    rev = str(text or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", rev):
        return rev.lower()
    return ""


def _snapshot_complete(runner: Any, host: str, repo: str, revision: str) -> bool:
    """True when the hub snapshot directory is present. Never reads a token."""
    if not repo or "/" not in repo:
        return False
    root = f"{_hub_root()}/{_repo_cache_dir(repo)}"
    rev = _revision_hash(revision)
    if not rev:
        ref_name = str(revision or "").strip() or "main"
        if not re.fullmatch(r"[A-Za-z0-9._-]+", ref_name):
            return False
        code, text = _capture_code(runner, host, ["cat", f"{root}/refs/{ref_name}"])
        if code != 0:
            return False
        rev = _revision_hash(text)
        if not rev:
            return False
    snap = f"{root}/snapshots/{rev}"
    if _capture_code(runner, host, ["test", "-d", snap])[0] != 0:
        return False
    if not _file_there(runner, host, f"{snap}/config.json"):
        return False
    if not _file_there(runner, host, f"{snap}/tokenizer_config.json"):
        return False
    if not (
        _file_there(runner, host, f"{snap}/tokenizer.json")
        or _file_there(runner, host, f"{snap}/tokenizer.model")
    ):
        return False
    code, index_text = _capture_code(runner, host, ["cat", f"{snap}/model.safetensors.index.json"])
    if code == 0:
        shards = _shard_names(index_text)
        if not shards:
            return False
        return all(_file_there(runner, host, f"{snap}/{name}") for name in shards)
    return _file_there(runner, host, f"{snap}/model.safetensors")


def _exit_only(runner: Any, host: str, argv: list[str]) -> int | None:
    """Exit code only. Stdout is not copied, so a secret value is not kept."""
    capture = getattr(runner, "capture", None) if runner is not None else None
    if capture is None or not host:
        return None
    try:
        result = capture(host, argv, 8)
    except Exception:  # noqa: BLE001 — a failed check means the name is not available
        return None
    if not isinstance(result, dict):
        return None
    if result.get("ok") or result.get("code") == 0:
        return 0
    try:
        return int(result.get("code"))
    except (TypeError, ValueError):
        return None


def _token_name_set(runner: Any, host: str, name: str) -> bool:
    """Whether ``printenv NAME`` succeeds. The printed value is discarded."""
    if name not in _TOKEN_NAMES:
        return False
    return _exit_only(runner, host, ["printenv", name]) == 0


def _shard_names(text: str) -> list[str]:
    """Basenames from a safetensors index. An unreadable index is empty."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return []
    weight_map = payload.get("weight_map") if isinstance(payload, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        return []
    found: list[str] = []
    for value in weight_map.values():
        name = os.path.basename(str(value or ""))
        if not name or name != str(value) or name in (".", "..") or not name.endswith(".safetensors"):
            return []
        if name not in found:
            found.append(name)
    return found


def _file_there(runner: Any, host: str, path: str) -> bool:
    return _capture_code(runner, host, ["test", "-f", path])[0] == 0


def apply_token_cache(
    recipe: dict[str, Any],
    runner: Any,
    *,
    repo: str = "",
    revision: str = "",
) -> list[str]:
    """Use the hub cache when it is complete. Otherwise require a token name on the node.

    A complete snapshot sets ``HF_HUB_OFFLINE`` and ``TRANSFORMERS_OFFLINE`` and drops
    the token names from ``env_pass``. An incomplete cache with the name unset refuses
    the pin. The token value and token files are never read.
    """
    conflicts: list[str] = []
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    placement = str(recipe.get("placement") or "")
    wanted = ["head", "worker"] if placement == "both" else [placement] if placement in ALL_ROLES else []
    saw_token = False
    still_required = False
    for role_name in wanted:
        role = roles.get(role_name)
        if not isinstance(role, dict):
            continue
        names = [str(item) for item in (role.get("env_pass") or []) if str(item) in _TOKEN_NAMES]
        if not names:
            continue
        saw_token = True
        host = CFG.node_by_role(role_name).name
        if _snapshot_complete(runner, host, repo, revision):
            env = dict(role.get("env") or {})
            env.update(_OFFLINE_ENV)
            role["env"] = env
            role["env_pass"] = [str(item) for item in (role.get("env_pass") or []) if str(item) not in _TOKEN_NAMES]
            continue
        still_required = True
        if not any(_token_name_set(runner, host, name) for name in names):
            conflicts.append(f"HF token not available on {role_name}; weights not fully cached")
    if saw_token:
        recipe["token_note"] = "Token required" if still_required or conflicts else "Token not required"
    return conflicts


def role_argv(role: dict[str, Any], bind: str) -> list[str]:
    """Launch argv for one role.

    Names in ``env_pass`` are passed as ``-e NAME`` with no value. Docker copies
    each variable from the environment of the process on the node. The recipe
    never contains the value. ``--env-file`` is not used: it would forward every
    key in the file, including keys that are not on the env allowlist.
    """
    args = rewrite_bind([str(part) for part in (role.get("args") or [])], bind)
    if str(role.get("runtime") or "docker") == "native":
        binary = str(role.get("binary") or "llama-server")
        return [binary, *args]
    argv = ["docker", "run", "-d"]
    if role.get("privileged"):
        argv.append("--privileged")
    if role.get("pid"):
        argv.extend(["--pid", str(role["pid"])])
    for cap in role.get("cap_add") or []:
        argv.extend(["--cap-add", str(cap)])
    for opt in role.get("security_opt") or []:
        argv.extend(["--security-opt", str(opt)])
    name = str(role.get("container") or "")
    if name:
        argv.extend(["--name", name])
    if role.get("gpus"):
        argv.extend(["--gpus", str(role["gpus"])])
    if role.get("ipc"):
        argv.extend(["--ipc", str(role["ipc"])])
    argv.extend(["--network", str(role.get("network") or "host")])
    if role.get("shm"):
        argv.extend(["--shm-size", str(role["shm"])])
    extra, _refused = extra_docker_argv(str(role.get("extra_docker_args") or ""))
    argv.extend(extra)
    passed = [str(item) for item in (role.get("env_pass") or []) if str(item)]
    for name in passed:
        argv.extend(["-e", name])
    for key, value in (role.get("env") or {}).items():
        if key in passed:
            continue
        argv.extend(["-e", f"{key}={value}"])
    for mount in role.get("mounts") or []:
        argv.extend(["-v", str(mount)])
    for device in role.get("devices") or []:
        argv.extend(["--device", str(device)])
    entry = str(role.get("entrypoint") or "").strip()
    if entry:
        argv.extend(["--entrypoint", entry])
    image = str(role.get("image") or "")
    if image:
        argv.append(image)
    argv.extend(args)
    return argv


_SHELLS = {"bash", "sh", "dash", "ash"}
# Flags Sparksi itself appends after a script entrypoint. ``-c`` is a shell command, not one of these.
_GENERATED_FLAGS = {
    "--port", "-p", "--host", "--bind",
    "--max-model-len", "--context-length", "--max_seq_len", "--ctx-size",
    "--gpu-memory-utilization", "--mem-fraction-static", "--kv_cache_free_gpu_memory_fraction",
    "--kv-cache-memory-bytes", "--kv-cache-memory", "--kv-cache-dtype",
    "--max-num-seqs",
    "--tool-call-parser", "--reasoning-parser",
    "--speculative-config", "--spec-type", "--spec-draft-n-max", "--model-draft",
    "--speculative-algorithm", "--speculative-num-steps", "--speculative-draft-model-path",
}
_WILD_BIND = re.compile(
    r"(?<![\w])(?:--host|--bind)(?:[ \t]*=[ \t]*|[ \t]+)(?P<q>['\"]?)(?P<addr>0\.0\.0\.0|\[::\]|::|\*)(?P=q)"
)


def _shell_command(
    role: dict[str, Any],
    entry: str,
    entry_flags: list[str],
    roots: list[str],
    role_name: str,
    runner: Any,
) -> str | None:
    """A shell entrypoint is allowed only for one mounted script that passed the bind check.

    ``bash vllm serve`` is refused. ``bash /start.sh`` is kept only when that
    script is under a mount root and the node-side read found no wildcard bind.
    """
    args = [str(part) for part in (role.get("args") or [])]
    explicit = bool(entry or entry_flags)
    if explicit:
        base = shell_basename(entry or (entry_flags[0] if entry_flags else ""))
        rest = args
    elif args:
        base = shell_basename(args[0])
        rest = args[1:]
    else:
        return None
    if not base:
        return None
    if not rest or not _script_path(rest[0]) or not _generated_tail(rest[1:]):
        return f"refusing shell entrypoint {base}"
    script = rest[0]
    if not explicit:
        return None
    if not roots:
        return f"refusing shell entrypoint {base}: mount roots are not configured"
    mounts = [str(item) for item in (role.get("mounts") or [])]
    host = host_path_from_mounts(script, mounts)
    if not host or not _under(host, roots):
        return f"refusing shell entrypoint {base}: script is not under a mount root"
    text = _remote_script(runner, CFG.node_by_role(role_name if role_name in ALL_ROLES else "head").name, host)
    if text is None or _bind_in_text(text, host):
        return f"refusing shell entrypoint {base}: script did not pass the bind check"
    return None


def _generated_tail(rest: list[str]) -> bool:
    """True when ``rest`` is only Sparksi serve flags, with no shell command."""
    index = 0
    while index < len(rest):
        token = rest[index]
        if token == "-c" or token.startswith("-c") or _meta(token):
            return False
        flag = token.split("=", 1)[0]
        if flag not in _GENERATED_FLAGS:
            return False
        if "=" in token:
            if _meta(token.split("=", 1)[1]):
                return False
            index += 1
            continue
        if index + 1 >= len(rest) or rest[index + 1].startswith("-") or _meta(rest[index + 1]):
            return False
        index += 2
    return True


def _script_path(token: str) -> bool:
    """A shell argument that names a script file, not the server binary."""
    text = str(token or "")
    if not text or text.startswith("-"):
        return False
    if text.endswith((".sh", ".bash")):
        return True
    return text.startswith("/") and "." not in os.path.basename(text)


def _looks_like_script(path: str) -> bool:
    base = os.path.basename(path)
    if not base or base in (".", ".."):
        return False
    return base.endswith((".sh", ".bash")) or "." not in base


def _script_statements(line: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        ch = line[index]
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            index += 1
            continue
        if ch == ";" or line.startswith("&&", index) or line.startswith("||", index):
            text = "".join(buf).strip()
            if text:
                parts.append(text)
            buf = []
            index += 1 if ch == ";" else 2
            continue
        buf.append(ch)
        index += 1
    text = "".join(buf).strip()
    if text:
        parts.append(text)
    return parts


def _shell_script(parts: list[str]) -> str:
    index = 0
    while index < len(parts):
        token = parts[index]
        if token == "--":
            index += 1
            break
        if token.startswith("-") and token != "-":
            if "c" in token[1:]:
                return ""
            index += 1
            continue
        break
    if index < len(parts) and parts[index].startswith("/") and _looks_like_script(parts[index]):
        return os.path.normpath(parts[index])
    return ""


def _command_scripts(argv: list[str]) -> list[str]:
    found: list[str] = []
    for index, token in enumerate(argv):
        base = os.path.basename(token)
        if base in _SHELLS:
            script = _shell_script(argv[index + 1:])
            if script and script not in found:
                found.append(script)
        elif index == 0 and token.startswith("/") and _looks_like_script(token) and token not in found:
            found.append(os.path.normpath(token))
    return found


def _container_script(path: str, base: str) -> str:
    text = str(path or "")
    if text.startswith("/"):
        return os.path.normpath(text)
    if not base.startswith("/"):
        return ""
    return os.path.normpath(os.path.join(base, text))


def _referenced_scripts(text: str, container: str = "") -> list[str]:
    found: list[str] = []
    base = os.path.dirname(container) if str(container).startswith("/") else ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for statement in _script_statements(line):
            try:
                parts = shlex.split(statement, posix=True)
            except ValueError:
                continue
            while parts and parts[0] in ("exec", "command", "setsid", "nohup"):
                parts = parts[1:]
            if not parts:
                continue
            script = ""
            if parts[0] in ("source", ".") and len(parts) > 1:
                script = _container_script(str(parts[1]), base)
            else:
                head = os.path.basename(parts[0])
                if head in _SHELLS:
                    script = _shell_script(parts[1:])
                elif parts[0].startswith("/") and _looks_like_script(parts[0]):
                    script = os.path.normpath(parts[0])
            if script and script not in found:
                found.append(script)
    return found


def _bind_in_text(text: str, path: str) -> str | None:
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _WILD_BIND.search(raw)
        if match:
            return f"refusing bind {match.group('addr')} in {path}:{number}"
    return None


def _remote_script(runner: Any, node: str, path: str) -> str | None:
    """File text from the node. ``None`` means it could not be read.

    The controller is read only for a file copied from a recipe folder, which is
    read from that folder here.
    """
    from sparksi.recipes.folder import local_path

    here = local_path(path, _FOLDER_FILES.get())
    if here is not None:
        try:
            with open(here, "rb") as handle:
                data = handle.read(1_000_001)
        except OSError:
            return None
        if len(data) > 1_000_000:
            return None
        return data.decode("utf-8", errors="replace")
    capture = getattr(runner, "capture", None) if runner is not None else None
    if capture is None or not node or not str(path).startswith("/"):
        return None
    try:
        result = capture(node, ["cat", path], 8)
    except Exception:  # noqa: BLE001 — a failed read refuses the pin
        return None
    if not isinstance(result, dict) or not (result.get("ok") or result.get("code") == 0):
        return None
    text = str(result.get("stdout") or "")
    if len(text) > 1_000_000:
        return None
    return text


def _scan_container_script(
    container: str,
    mounts: list[str],
    seen: set[str],
    conflicts: list[str],
    role_name: str,
    node: str,
    runner: Any,
) -> None:
    host = host_path_from_mounts(container, mounts)
    if not host or host in seen:
        # No bind covers this path, so the script lives in the image and is not checked.
        return
    seen.add(host)
    text = _remote_script(runner, node, host)
    if text is None:
        conflicts.append(f"could not read mounted script {host} on {role_name}")
        return
    hit = _bind_in_text(text, host)
    if hit:
        conflicts.append(hit)
    for ref in _referenced_scripts(text, container):
        _scan_container_script(ref, mounts, seen, conflicts, role_name, node, runner)


def _scan_mounted_scripts(role_name: str, role: dict[str, Any], runner: Any) -> list[str]:
    """Wildcard binds in a mounted entry script, read on the node that runs the role.

    A command path that no mount covers is inside the image and is out of scope.
    A mounted script, or a script it sources or execs through a mount, that cannot
    be read refuses the pin.
    """
    if str(role.get("runtime") or "docker") != "docker":
        return []
    mounts = [str(item) for item in (role.get("mounts") or [])]
    conflicts: list[str] = []
    seen: set[str] = set()
    node = CFG.node_by_role(role_name if role_name in ALL_ROLES else "head").name
    for container in _command_scripts([str(part) for part in (role.get("args") or [])]):
        _scan_container_script(container, mounts, seen, conflicts, role_name, node, runner)
    return conflicts


def _scan_host(argv: list[str]) -> str | None:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--host", "--bind") and index + 1 < len(argv):
            if argv[index + 1] in ("0.0.0.0", "::", "*"):
                return "refusing bind 0.0.0.0"
            index += 2
            continue
        if token.startswith("--host=") or token.startswith("--bind="):
            if token.split("=", 1)[1] in ("0.0.0.0", "::", "*"):
                return "refusing bind 0.0.0.0"
        index += 1
    return None


def _scan_port(argv: list[str]) -> str | None:
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in ("--port", "-p") and index + 1 < len(argv) and protect.is_off_limits(argv[index + 1]):
            return "port 8188 is off limits"
        if token.startswith("--port=") and protect.is_off_limits(token.split("=", 1)[1]):
            return "port 8188 is off limits"
        index += 1
    return None


def _roce_allowed(value: str, kind: str) -> bool:
    flag = "--cap-add" if kind == "cap" else "--security-opt"
    return docker_option_refusal(flag, value, allow_roce=True) is None


def uses_roce_fabric(recipe: dict[str, Any]) -> bool:
    """A container recipe on both nodes, head leading, with RoCE HCAs configured."""
    if str(recipe.get("placement") or "") != "both":
        return False
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    if "head" not in roles or "worker" not in roles:
        return False
    for role in roles.values():
        if not isinstance(role, dict) or str(role.get("runtime") or "docker") != "docker":
            return False
    return bool(CFG.has_worker and CFG.head.fabric.hcas and CFG.worker.fabric.hcas)


def recipe_allows_roce(recipe: dict[str, Any]) -> bool:
    """True when launch may apply the RoCE docker allowance.

    Pinned recipes use the role tables. A rendered recipe has nodes instead,
    and still has to be two-node docker with RoCE HCAs configured.
    """
    placement = str(recipe.get("placement") or "")
    covered = roles_of(placement)
    if len(covered) < 2:
        return False
    if not all(CFG.node_by_role(role).fabric.hcas for role in covered):
        return False
    roles = recipe.get("roles")
    if isinstance(roles, dict) and roles:
        return uses_roce_fabric(recipe)
    nodes = [node for node in (recipe.get("nodes") or []) if isinstance(node, dict)]
    if len(nodes) < 2:
        return False
    for node in nodes:
        argv = node.get("argv") or []
        if not argv or str(argv[0]) != "docker":
            return False
    return True


def roce_permissions_line(roles: dict[str, Any]) -> str:
    """The allowed RoCE flags, for a read-only line on the model form."""
    caps: list[str] = []
    opts: list[str] = []
    for role in roles.values():
        if not isinstance(role, dict):
            continue
        for cap in role.get("cap_add") or []:
            if _roce_allowed(str(cap), "cap") and "IPC_LOCK" not in caps:
                caps.append("IPC_LOCK")
        for opt in role.get("security_opt") or []:
            if _roce_allowed(str(opt), "opt") and "label=disable" not in opts:
                opts.append("label=disable")
    parts = [f"--cap-add {cap}" for cap in caps] + [f"--security-opt {opt}" for opt in opts]
    return ", ".join(parts)


def validate_role(
    name: str,
    role: dict[str, Any],
    argv: list[str],
    roots: list[str],
    *,
    allow_roce: bool = False,
    runner: Any = None,
) -> list[str]:
    conflicts: list[str] = []
    runtime = str(role.get("runtime") or "docker")
    if runtime not in ("docker", "native"):
        conflicts.append(f"{name} runtime must be docker or native")
        return conflicts
    if any("dgx-announce" in str(part) for part in argv):
        conflicts.append("refusing dgx-announce")
    host = _scan_host(argv)
    if host:
        conflicts.append(host)
    port = _scan_port(argv)
    if port:
        conflicts.append(port)
    if runtime == "native":
        binary = str(role.get("binary") or (argv[0] if argv else ""))
        if not binary_allowed(binary, roots):
            conflicts.append("binary must be llama-server or an absolute path under a mount root")
        lib = str((role.get("env") or {}).get("LD_LIBRARY_PATH") or "")
        if lib and not library_path_allowed(lib, roots):
            conflicts.append("LD_LIBRARY_PATH must be under a mount root")
    else:
        if role.get("privileged") or "--privileged" in argv:
            conflicts.append("refusing --privileged")
        if str(role.get("pid") or "") == "host" or "--pid=host" in argv or _flag_value(argv, "--pid") == "host":
            conflicts.append("refusing --pid=host")
        cap_values = list(role.get("cap_add") or []) + _flag_values(argv, "--cap-add")
        refused_caps = [item for item in cap_values if not (allow_roce and _roce_allowed(str(item), "cap"))]
        bare_cap = not cap_values and ("--cap-add" in argv or any(str(part).startswith("--cap-add=") for part in argv))
        if refused_caps or bare_cap:
            conflicts.extend(_named_flag("--cap-add", refused_caps))
        opt_values = list(role.get("security_opt") or []) + _flag_values(argv, "--security-opt")
        refused_opts = [item for item in opt_values if not (allow_roce and _roce_allowed(str(item), "opt"))]
        bare_opt = not opt_values and ("--security-opt" in argv or any(str(part).startswith("--security-opt=") for part in argv))
        if refused_opts or bare_opt:
            conflicts.extend(_named_flag("--security-opt", refused_opts))
        network = str(role.get("network") or "host")
        if network != "host":
            conflicts.append("refusing --network")
        for device in list(role.get("devices") or []) + _flag_values(argv, "--device"):
            refused = docker_option_refusal("--device", str(device), allow_roce=allow_roce)
            if refused:
                conflicts.append(refused)
                break
        for item in _flag_values(argv, "--ulimit"):
            refused = docker_option_refusal("--ulimit", str(item), allow_roce=allow_roce)
            if refused and refused not in conflicts:
                conflicts.append(refused)
        entry = str(role.get("entrypoint") or "")
        entry_flags = _flag_values(argv, "--entrypoint")
        # Same per-node mount roots as the bind check below. The agent stand-in
        # list is not consulted.
        if entry:
            refused = docker_option_refusal(
                "--entrypoint", entry, allow_roce=allow_roce, argv=argv, roots=roots,
            )
            if refused:
                conflicts.append(refused)
        for item in entry_flags:
            refused = docker_option_refusal(
                "--entrypoint", str(item), allow_roce=allow_roce, argv=argv, roots=roots,
            )
            if refused and refused not in conflicts:
                conflicts.append(refused)
        shell = _shell_command(role, entry, entry_flags, roots, name, runner)
        if shell and shell not in conflicts:
            conflicts.append(shell)
        image = str(role.get("image") or "")
        if not image:
            conflicts.append("pinned recipe needs an image")
        elif ":latest" in image or image.endswith(":latest"):
            conflicts.append("image tag must be pinned")
        elif _meta(image):
            conflicts.append("image contains shell metacharacters")
        for mount in list(role.get("mounts") or []) + _flag_values(argv, "-v") + _flag_values(argv, "--volume"):
            refused = _mount_refused(_mount_source(str(mount)), roots)
            if refused:
                conflicts.append(refused)
        name_text = str(role.get("container") or "")
        if name_text and _meta(name_text):
            conflicts.append("container name contains shell metacharacters")
        conflicts.extend(_scan_mounted_scripts(name, role, runner))
        _approved, refused_extra = extra_docker_argv(str(role.get("extra_docker_args") or ""))
        conflicts.extend(refused_extra)
    for key, value in (role.get("env") or {}).items():
        if not str(key).replace("_", "").isalnum() or str(key)[:1].isdigit() or _meta(str(key)) or _meta(str(value)):
            conflicts.append("environment contains shell metacharacters")
            break
    return conflicts


def _flag_value(argv: list[str], flag: str) -> str:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            return argv[index + 1]
    prefix = flag + "="
    for token in argv:
        if token.startswith(prefix):
            return token.split("=", 1)[1]
    return ""


def _named_flag(flag: str, values: list[Any]) -> list[str]:
    """One refusal per distinct value, naming the flag. An empty list still refuses the flag."""
    found: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text and text not in found:
            found.append(text)
    if not found:
        return [f"refusing {flag}"]
    return [f"refusing {flag} {item}" for item in found]


def _flag_values(argv: list[str], flag: str) -> list[str]:
    found = []
    index = 0
    while index < len(argv):
        if argv[index] == flag and index + 1 < len(argv):
            found.append(argv[index + 1])
            index += 2
            continue
        if argv[index].startswith(flag + "="):
            found.append(argv[index].split("=", 1)[1])
        index += 1
    return found


FABRIC_ONLY = "IPC_LOCK is only allowed for a two-machine recipe when both Sparks have their high-speed network configured."


def _role_conflicts(name: str, role: dict[str, Any], argv: list[str], roce: bool, runner: Any) -> tuple[list[str], list[str]]:
    """(conflicts, fabric-only conflicts). Fabric-only ones go away when the high-speed network allowance applies."""
    strict = validate_role(name, role, argv, roots_for(name), allow_roce=roce, runner=runner)
    if roce or not strict:
        return strict, []
    relaxed = validate_role(name, role, argv, roots_for(name), allow_roce=True, runner=runner)
    fabric = [item for item in strict if item not in relaxed]
    return [item for item in strict if item not in fabric], fabric


def fabric_only(recipe: dict[str, Any]) -> bool:
    """True when the recipe uses the high-speed network docker flags but cannot have them here."""
    roce = uses_roce_fabric(recipe)
    if roce:
        return False
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    for name, role in roles.items():
        if not isinstance(role, dict) or str(name) not in ALL_ROLES:
            continue
        argv = role_argv(role, CFG.node_by_role(str(name)).tailscale)
        if _role_conflicts(str(name), role, argv, roce, None)[1]:
            return True
    return False


def validate_pinned(recipe: dict[str, Any], *, ctx_max: int | None = None, runner: Any = None, for_save: bool = False) -> list[str]:
    conflicts: list[str] = []
    engine = str(recipe.get("engine") or "")
    if engine not in PINNED_ENGINES:
        conflicts.append(f"unknown engine {engine or '(none)'}")
    try:
        port = int(recipe.get("port"))
    except (TypeError, ValueError):
        conflicts.append("port must be an integer")
        port = 0
    if protect.is_off_limits(port):
        conflicts.append("port 8188 is off limits")
    elif port and (port < PORT_MIN or port > PORT_MAX):
        conflicts.append(f"port {port} is outside {PORT_MIN}-{PORT_MAX}")
    conflicts.extend(serve_port_conflicts(recipe))
    for extra in recipe.get("rendezvous_ports") or []:
        try:
            number = int(extra)
        except (TypeError, ValueError):
            conflicts.append("rendezvous port must be an integer")
            continue
        if protect.is_off_limits(number):
            conflicts.append("port 8188 is off limits")
    placement = str(recipe.get("placement") or "")
    if placement not in _PLACES:
        conflicts.append("placement must be head, worker, or both")
    elif not for_save and needs_text(placement):
        # A pinned command written for two Sparks cannot run on a one-Spark site.
        conflicts.append(f"this pinned command {needs_text(placement)}")
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    if not roles:
        conflicts.append("pinned recipe needs a role")
    wanted = ["head", "worker"] if placement == "both" else [placement] if placement in ALL_ROLES else []
    for name in wanted:
        if name not in roles:
            conflicts.append(f"pinned recipe needs a {name} role")
    parsed_ctx = None
    roce = uses_roce_fabric(recipe)
    fabric = False
    for name, role in roles.items():
        if not isinstance(role, dict):
            conflicts.append(f"{name} role must be a table")
            continue
        if str(name) not in ALL_ROLES:
            conflicts.append(f"unknown role {name}")
            continue
        bind = CFG.node_by_role(str(name)).tailscale
        argv = role_argv(role, bind)
        strict, fabric_items = _role_conflicts(str(name), role, argv, roce, runner)
        conflicts.extend(strict)
        fabric = fabric or bool(fabric_items)
        known, _extra = parse_args(list(role.get("args") or []))
        if "ctx" in known:
            parsed_ctx = known["ctx"]
        spec, why = normalize_speculative(known.get("speculative"))
        if why:
            conflicts.append(why)
        elif spec and spec.get("model"):
            refused = _model_path_refused(str(spec["model"]), role, roots_for(str(name)))
            if refused:
                conflicts.append(refused)
    if fabric and not for_save:
        # Saving keeps the flags; Dry run and Load refuse them here.
        conflicts.append(FABRIC_ONLY)
    if parsed_ctx not in (None, ""):
        try:
            ctx = int(parsed_ctx)
        except (TypeError, ValueError):
            conflicts.append("ctx must be an integer")
        else:
            limit = ctx_ceiling(ctx_max)
            if ctx < CTX_MIN or ctx > limit:
                conflicts.append(f"ctx must be from {CTX_MIN} to {limit}")
    # Preserve order.
    seen: set[str] = set()
    unique = []
    for item in conflicts:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _launch(recipe: dict[str, Any], role_name: str, role: dict[str, Any]) -> Launch:
    bind = CFG.node_by_role(role_name).tailscale
    host = CFG.node_by_role(role_name).name
    argv = role_argv(role, bind)
    env = {str(key): str(value) for key, value in (role.get("env") or {}).items()}
    container = str(role.get("container") or "") if role.get("runtime") != "native" else ""
    stop = docker_stop(container) if container else []
    return launch_for(
        _spec_stub(recipe, role_name, host, bind),
        role_name,
        argv,
        env,
        container=container,
        stop=stop,
    )


def _spec_stub(recipe: dict[str, Any], role: str, host: str, bind: str):
    from sparksi.recipes.spec import Spec

    return Spec(
        model_id=str(recipe.get("model_id") or recipe.get("id") or "model"),
        placement=str(recipe.get("placement") or role),
        port=int(recipe.get("port") or 0),
        paths={},
        hosts={role: host},
        binds={role: bind},
        fabric_ip={},
        fabric_iface={},
        fabric_hcas={},
    )


def render_pinned(
    card: ModelCard,
    recipe: dict[str, Any],
    *,
    catalog: dict[str, Any] | None = None,
) -> dict[str, Any]:
    try:
        saved_port = int(recipe.get("port") or 0)
    except (TypeError, ValueError):
        saved_port = 0
    if saved_port:
        align_recipe(recipe, saved_port, saved_port=saved_port)
    placement = str(recipe.get("placement") or "head")
    if placement in _PLACES and needs_text(placement):
        raise ValueError(f"this pinned command {needs_text(placement)}")
    order = ["worker", "head"] if placement == "both" else [placement if placement in ALL_ROLES else "head"]
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    nodes: list[Launch] = []
    for role_name in order:
        role = roles.get(role_name)
        if not isinstance(role, dict):
            continue
        nodes.append(_launch({**recipe, "model_id": card.id}, role_name, role))
    built = Built(nodes=nodes, order=order, experimental=False, health_paths=["/health", "/v1/models"])
    head = roles.get("head") if isinstance(roles.get("head"), dict) else {}
    worker = roles.get("worker") if isinstance(roles.get("worker"), dict) else {}
    sample = head or worker or {}
    command, known, extras = parse_launch([str(part) for part in (sample.get("args") or [])])
    worker_command, _worker_known, worker_extras = parse_launch([str(part) for part in (worker.get("args") or [])]) if worker else ([], {}, [])
    port = int(recipe.get("port") or known.get("port") or 0)
    image = str(sample.get("image") or "")
    timeout = health_timeout_for(
        catalog,
        {"id": card.id, "timeout_s": recipe.get("timeout_s"), "start_timeout_s": card.timeout_s},
    )
    row = {"id": card.id, "port": port, "kind": str(recipe.get("engine") or "other"), "placement": placement, "probe": None}
    paths = health_paths_for(row) or list(built.health_paths)
    env_head = {str(key): str(value) for key, value in (sample.get("env") or {}).items()}
    env_worker = {str(key): str(value) for key, value in (worker.get("env") or {}).items()}
    fields = {
        "ctx": known.get("ctx"),
        "gpu_mem_util": known.get("gpu_mem_util"),
        "max_num_seqs": known.get("max_num_seqs"),
        "kv_cache_dtype": known.get("kv_cache_dtype") or "",
        "kv_cache_memory_bytes": known.get("kv_cache_memory_bytes"),
        "speculative": known.get("speculative") or {"method": "", "model": "", "tokens": 0},
        "tool_parser": known.get("tool_parser") or "",
        "reasoning_parser": known.get("reasoning_parser") or "",
        "image": image,
        "command": " ".join(command),
        "command_worker": " ".join(worker_command),
        "extra_args": extras,
        "extra_args_worker": worker_extras,
        "env": env_head,
        "env_worker": env_worker,
        "timeout_s": timeout,
        "imported_from": str(recipe.get("imported_from") or ""),
        "port_note": str(recipe.get("port_note") or ""),
        "roce_permissions": roce_permissions_line(roles) if placement == "both" else "",
        "token_note": str(recipe.get("token_note") or ""),
        "repo": str(recipe.get("repo") or card.repo or ""),
        "revision": str(recipe.get("revision") or card.revision or ""),
        "evict_page_cache": flag_on(recipe.get("evict_page_cache")),
    }
    body = built.to_dict()
    for node in body["nodes"]:
        role = roles.get(str(node.get("role") or ""))
        role = role if isinstance(role, dict) else {}
        enabled = evict_enabled(recipe, role)
        node["evict_page_cache"] = enabled
        node["evict_path"] = resolve_evict_path(
            role,
            str(node.get("role") or "head"),
            repo=str(recipe.get("repo") or card.repo or ""),
            revision=str(recipe.get("revision") or card.revision or ""),
        ) if enabled else ""
    body["evict_page_cache"] = flag_on(recipe.get("evict_page_cache"))
    body.update(
        {
            "id": card.id,
            "kind": "pinned",
            "engine": str(recipe.get("engine") or "other"),
            "placement": placement,
            "port": port,
            "timeout_s": timeout,
            "health_paths": [] if protect.is_off_limits(port) or port < PORT_MIN else paths,
            "image": image,
            "ctx": known.get("ctx"),
            "experimental": False,
            "experimental_reason": "",
            "imported_from": str(recipe.get("imported_from") or ""),
            "port_note": str(recipe.get("port_note") or ""),
            "rendezvous_ports": list(recipe.get("rendezvous_ports") or []),
            "fields": fields,
        }
    )
    return body


def known_fields(recipe: dict[str, Any]) -> dict[str, Any]:
    """Fields the form shows, parsed from the head argv (worker extras stay separate)."""
    roles = recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {}
    head = roles.get("head") if isinstance(roles.get("head"), dict) else next(iter(roles.values()), {})
    if not isinstance(head, dict):
        head = {}
    known, extras = parse_args([str(part) for part in (head.get("args") or [])])
    known["extra_args"] = extras
    known["image"] = str(head.get("image") or "")
    known["env"] = dict(head.get("env") or {})
    ports = list(recipe.get("rendezvous_ports") or [])
    if not ports:
        ports = rendezvous_ports([str(part) for part in (head.get("args") or [])])
    known["rendezvous_ports"] = ports
    known["repo"] = str(recipe.get("repo") or "")
    known["revision"] = str(recipe.get("revision") or "")
    return known
