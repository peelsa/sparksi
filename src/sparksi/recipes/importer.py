# SPDX-License-Identifier: Apache-2.0

"""Read a running setup and turn it into a pinned recipe. Never start or stop.

The controller asks a Spark, over SSH, for ``docker inspect``, ``docker compose
config``, ``systemctl cat``, or a start script. Parsing is pure. Saving uses
the recipe store and only happens when the caller did not ask for a dry run.
"""

from __future__ import annotations

import json
import os
import re
import shlex
from datetime import datetime, timezone
from typing import Any

from sparksi import protect
from sparksi.agent import shell_basename
from sparksi.config import CFG
from sparksi.models import card_from_row
from sparksi.recipes.pinned import apply_token_cache, validate_pinned
from sparksi.recipes.store import load_recipe, save_recipe, value_looks_like_token
from sparksi.recipes.validate import PORT_MAX, PORT_MIN, check_port, next_free_port

_SKIP_ENV = {"PATH", "HOSTNAME", "HOME", "TERM", "LANG", "LC_ALL"}
_PORT_ENV = (
    "PORT",
    "SERVER_PORT",
    "VLLM_PORT",
    "SGLANG_PORT",
    "TRTLLM_PORT",
    "LLAMA_PORT",
    "LISTEN_PORT",
    "API_PORT",
    "HTTP_PORT",
    "WEB_PORT",
    "APP_PORT",
    "UVICORN_PORT",
)
_COMPOSE_NAMES = ("docker-compose.yml", "compose.yaml", "compose.yml")
_LAUNCH_PREFIXES = {"exec", "setsid", "nohup", "sudo", "command", "env"}
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_source(text: str) -> tuple[str, str]:
    kind, _, rest = str(text or "").partition(":")
    return kind.strip(), rest.strip()


def readonly_argv(kind: str, value: str) -> list[str]:
    """The only commands the importer runs. Each one is a read."""
    if kind == "container":
        return ["docker", "inspect", value]
    if kind == "compose":
        return ["docker", "compose", "-f", value, "config", "--format", "json"]
    if kind == "unit":
        return ["systemctl", "cat", value]
    if kind == "script":
        return ["cat", value]
    if kind == "ps":
        return ["ps", "-eo", "args"]
    if kind == "containers":
        return ["docker", "ps", "-a", "--format", "{{.Names}}\t{{.Image}}"]
    if kind == "image":
        return ["docker", "image", "inspect", value]
    if kind == "exists":
        return ["test", "-f", value]
    raise ValueError(f"unknown import source {kind}")


def _stamp(source: str) -> str:
    when = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{source} at {when}"


def _env_list(raw: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    items: list[Any]
    if isinstance(raw, dict):
        items = [f"{key}={value}" for key, value in raw.items()]
    elif isinstance(raw, list):
        items = list(raw)
    else:
        return out
    for item in items:
        text = str(item)
        if "=" not in text:
            continue
        key, value = text.split("=", 1)
        if not key or key in _SKIP_ENV:
            continue
        out[key] = value
    return out


def _shm(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    if isinstance(value, str):
        return value
    try:
        number = int(value)
    except (TypeError, ValueError):
        return str(value)
    gib = 1024 ** 3
    if number > 0 and number % gib == 0:
        return f"{number // gib}g"
    return str(number)


def _gpus(host: dict[str, Any]) -> str:
    requests = host.get("DeviceRequests") or []
    for item in requests:
        if not isinstance(item, dict):
            continue
        caps = item.get("Capabilities") or []
        flat = []
        for group in caps:
            if isinstance(group, list):
                flat.extend(str(part) for part in group)
            else:
                flat.append(str(group))
        if "gpu" in flat or item.get("Driver") == "nvidia":
            return "all"
    return ""


def _devices(host: dict[str, Any]) -> list[str]:
    found = []
    for item in host.get("Devices") or []:
        if isinstance(item, str):
            found.append(item)
            continue
        if not isinstance(item, dict):
            continue
        src = str(item.get("PathOnHost") or "")
        dst = str(item.get("PathInContainer") or src)
        if src:
            found.append(src if src == dst or not dst else f"{src}:{dst}")
    return found


def _mounts(payload: dict[str, Any]) -> list[str]:
    found: list[str] = []
    host = payload.get("HostConfig") if isinstance(payload.get("HostConfig"), dict) else {}
    for item in host.get("Binds") or []:
        text = str(item)
        if text and text not in found:
            found.append(text)
    for item in payload.get("Mounts") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("Type") or "bind") not in ("bind", ""):
            continue
        src = str(item.get("Source") or "")
        dst = str(item.get("Destination") or "")
        if not src or not dst:
            continue
        text = f"{src}:{dst}"
        if text not in found and not any(str(have).split(":")[0] == src for have in found):
            found.append(text)
    return found


def _argv_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            return [str(part) for part in shlex.split(value) if str(part)]
        except ValueError:
            return [value] if value else []
    if isinstance(value, list):
        return [str(part) for part in value if str(part)]
    return []


def _split_entrypoint(entry: Any, command: Any) -> tuple[str, list[str]]:
    """Entrypoint executable, then the container args.

    The image entrypoint is not folded into the args. Rendering it again as
    ``--entrypoint`` keeps an override of ``vllm`` from becoming ``bash vllm``.
    """
    tokens = _argv_list(entry)
    args = _argv_list(command)
    if not tokens:
        return "", args
    return _promote_shell_script(tokens[0], tokens[1:] + args)


_SERVE_ENTRY = {"vllm", "python", "python3", "sglang", "trtllm-serve", "llama-server"}


def _promote_shell_script(entrypoint: str, args: list[str]) -> tuple[str, list[str]]:
    """An image entrypoint of ``vllm`` plus ``bash /start.sh`` is the script entry.

    The shell becomes the entrypoint and the script is the only argument.
    """
    if len(args) != 2 or not shell_basename(args[0]):
        return entrypoint, args
    script = args[1]
    if not script.startswith("/") or not script.endswith((".sh", ".bash")):
        return entrypoint, args
    if os.path.basename(entrypoint) not in _SERVE_ENTRY:
        return entrypoint, args
    return args[0], [script]


def guess_engine(args: list[str], image: str) -> str:
    blob = (" ".join(args) + " " + image).lower()
    if "tensorfold" in blob:
        return "tensorfold"
    if "llama-server" in blob or "llama.cpp" in blob:
        return "llamacpp"
    if "sglang" in blob:
        return "sglang"
    if "trtllm" in blob or "tensorrt" in blob:
        return "trtllm"
    if "vllm" in blob:
        return "vllm"
    return "other"


def parse_docker_inspect(payload: Any) -> dict[str, Any]:
    """One ``docker inspect`` document becomes one role."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        raise ValueError("docker inspect did not return an object")
    config = payload.get("Config") if isinstance(payload.get("Config"), dict) else {}
    host = payload.get("HostConfig") if isinstance(payload.get("HostConfig"), dict) else {}
    entrypoint, args = _split_entrypoint(config.get("Entrypoint"), config.get("Cmd"))
    image = str(config.get("Image") or "")
    name = str(payload.get("Name") or "").lstrip("/")
    network = str(host.get("NetworkMode") or "")
    role_body: dict[str, Any] = {
        "runtime": "docker",
        "image": image,
        "env": _env_list(config.get("Env")),
        "mounts": _mounts(payload),
        "network": network or "host",
        "gpus": _gpus(host),
        "ipc": str(host.get("IpcMode") or ""),
        "shm": _shm(host.get("ShmSize")),
        "container": name,
        "args": args,
        "devices": _devices(host),
    }
    if entrypoint:
        role_body["entrypoint"] = entrypoint
    if host.get("Privileged"):
        role_body["privileged"] = True
    pid = str(host.get("PidMode") or "")
    if pid:
        role_body["pid"] = pid
    caps = [str(item) for item in (host.get("CapAdd") or []) if item]
    if caps:
        role_body["cap_add"] = caps
    opts = [str(item) for item in (host.get("SecurityOpt") or []) if item]
    if opts:
        role_body["security_opt"] = opts
    _apply_secret_split(role_body)
    return role_body


def _compose_service(name: str, body: dict[str, Any]) -> dict[str, Any]:
    entrypoint, command = _split_entrypoint(body.get("entrypoint"), body.get("command"))
    volumes = []
    for item in body.get("volumes") or []:
        if isinstance(item, str):
            volumes.append(item)
        elif isinstance(item, dict):
            src = str(item.get("source") or "")
            dst = str(item.get("target") or "")
            if src and dst:
                volumes.append(f"{src}:{dst}")
    gpus = ""
    devices = body.get("deploy") if isinstance(body.get("deploy"), dict) else {}
    reservations = (devices.get("resources") or {}).get("reservations") or {}
    for item in reservations.get("devices") or []:
        caps = item.get("capabilities") or [] if isinstance(item, dict) else []
        if "gpu" in caps:
            gpus = "all"
    network = str(body.get("network_mode") or "")
    role: dict[str, Any] = {
        "runtime": "docker",
        "image": str(body.get("image") or ""),
        "env": _env_list(body.get("environment")),
        "mounts": volumes,
        "network": network or "host",
        "gpus": gpus,
        "ipc": str(body.get("ipc") or ""),
        "shm": _shm(body.get("shm_size")),
        "container": str(body.get("container_name") or name),
        "args": [str(part) for part in command],
        "devices": [],
    }
    if entrypoint:
        role["entrypoint"] = entrypoint
    if body.get("privileged"):
        role["privileged"] = True
    _apply_secret_split(role)
    return role


def _role_name(service: str, index: int, count: int) -> str:
    text = service.lower()
    if "worker" in text or text in ("b", "rank1"):
        return "worker"
    if "head" in text or text in ("a", "rank0"):
        return "head"
    if count == 1:
        return "head"
    return "worker" if index else "head"


def parse_compose_config(payload: Any) -> dict[str, dict[str, Any]]:
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("compose config did not return an object")
    services = payload.get("services") if isinstance(payload.get("services"), dict) else {}
    roles: dict[str, dict[str, Any]] = {}
    items = list(services.items())
    for index, (name, body) in enumerate(items):
        if not isinstance(body, dict):
            continue
        role = _role_name(str(name), index, len(items))
        roles[role] = _compose_service(str(name), body)
    return roles


def _compose_file_arg(parts: list[str]) -> str:
    for flag in ("-f", "--file"):
        if flag in parts:
            index = parts.index(flag)
            if index + 1 < len(parts):
                return parts[index + 1]
    for part in parts:
        if part.startswith("--file="):
            return part.split("=", 1)[1]
    return ""


def _bare_compose(parts: list[str]) -> bool:
    """``docker compose up`` or ``docker-compose up`` with no compose file."""
    if not parts:
        return False
    head = parts[0].rsplit("/", 1)[-1]
    if head == "docker-compose":
        args = parts[1:]
    elif head == "docker" and len(parts) > 1 and parts[1] == "compose":
        args = parts[2:]
    else:
        return False
    return "up" in args and not _compose_file_arg(args)


def parse_unit(text: str) -> dict[str, str]:
    """A unit file points at a compose file, a bare compose up, or a script.

    The unit is not executed. ``WorkingDirectory`` is kept so a wrapper that
    runs ``docker compose up`` without ``-f`` can be followed to the compose file.
    """
    found: dict[str, str] = {}
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "WorkingDirectory":
            directory = value.strip('"').strip("'")
            if directory.startswith("-"):
                directory = directory[1:]
            if directory:
                found["working_directory"] = directory
            continue
        if key not in ("ExecStart", "ExecStartPre"):
            continue
        if key == "ExecStartPre" and (found.get("compose") or found.get("script") or found.get("bare_compose")):
            continue
        try:
            parts = shlex.split(value)
        except ValueError:
            parts = value.split()
        compose = _compose_file_arg(parts) if "compose" in parts or any(part.rsplit("/", 1)[-1] == "docker-compose" for part in parts) else ""
        if compose:
            found["compose"] = compose
            found.pop("script", None)
            found.pop("bare_compose", None)
            continue
        if _bare_compose(parts):
            found["bare_compose"] = "1"
            found.pop("script", None)
            continue
        for part in parts:
            if part.endswith(".sh") or part.endswith(".py"):
                found["script"] = part
                break
    return found


def _native_from_argv(argv: list[str], env: dict[str, str]) -> dict[str, Any]:
    binary = argv[0] if argv else "llama-server"
    return {
        "runtime": "native",
        "binary": binary,
        "args": list(argv[1:]),
        "env": dict(env),
        "image": "",
        "mounts": [],
        "network": "host",
        "container": "",
    }


def _joined_lines(text: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    pending = ""
    start = 1
    for number, raw in enumerate(str(text or "").splitlines(), start=1):
        piece = raw.rstrip()
        if piece.endswith("\\"):
            if not pending:
                start = number
            pending += piece[:-1] + " "
            continue
        lines.append((start if pending else number, (pending + piece).strip()))
        pending = ""
        start = number + 1
    if pending:
        lines.append((start, pending.strip()))
    return lines


def _secret_name(name: str) -> bool:
    """Names whose values must never be written into a recipe."""
    text = str(name or "")
    if not _NAME.fullmatch(text):
        return False
    return text == "HF_TOKEN" or text.endswith(("_TOKEN", "_KEY", "_SECRET"))


# Values from an env file may be stored only for these names. Everything else,
# including secret patterns, is a name-only pass-through.
ENV_VALUE_ALLOW = frozenset({
    "IMAGE",
    "MODEL",
    "MODEL_PATH",
    "SERVED_MODEL_NAME",
    "MAX_MODEL_LEN",
    "GPU_MEMORY_UTILIZATION",
    "TENSOR_PARALLEL_SIZE",
    "PIPELINE_PARALLEL_SIZE",
    "ENABLE_EXPERT_PARALLEL",
    "PORT",
    "HOST",
    "MASTER_PORT",
    "MASTER_ADDR",
    "HEAD_IP",
    "WORKER_IP",
    "IFACE",
    "IB_HCA",
    "TOOL_CALL_PARSER",
    "REASONING_PARSER",
    "KV_CACHE_DTYPE",
    "MODEL_ID",
    "ABL" + "IT",
    "MTP_NUM_SPECULATIVE_" + "TOKENS",
    "FP8_DENSE",
    "QSA_PROFILE",
    "PLE_OFFLOAD",
    "YARN_ENABLE",
    "YARN_FACTOR",
    "IB_GID_INDEX",
    "MAX_NUM_SEQS",
    "MAX_NUM_BATCHED_" + "TOKENS",
    "MM_ENCODER_TP_MODE",
    "MAMBA_SSM_CACHE_DTYPE",
    "NFS_SHARE",
    "VLLM_ALLOW_LONG_MAX_MODEL_LEN",
    "REQUIRE_IDLE_GPU",
    "WORKER_USER",
})
ENV_VALUE_PREFIXES = ("NCCL_", "SPEC_", "SPECULATIVE_", "MTP_")
_ENV_REVIEW = frozenset({"EXTRA_DOCKER_ARGS"})
_SECRET_WORDS = frozenset({"TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"})


def env_name_is_secret(name: str) -> bool:
    """Suffix or whole-word secret names. ``*_TOKENS`` is not a token name."""
    text = str(name or "")
    if not _NAME.fullmatch(text):
        return False
    upper = text.upper()
    if upper in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        return True
    if upper.endswith(("_TOKEN", "_API_KEY", "_SECRET", "_PASSWORD", "_CREDENTIAL")):
        return True
    parts = [part for part in upper.split("_") if part]
    if any(part in _SECRET_WORDS for part in parts):
        return True
    return any(parts[index] == "API" and parts[index + 1] == "KEY" for index in range(len(parts) - 1))


def env_value_allowed(name: str) -> bool:
    """True when an env-file value may be kept. Secret names and unknown keys are not.

    ``EXTRA_DOCKER_ARGS`` is recorded for review, not stored as a value.
    A token-like value is still dropped even when the key is allowlisted.
    """
    text = str(name or "")
    if not _NAME.fullmatch(text):
        return False
    upper = text.upper()
    if upper in _ENV_REVIEW or env_name_is_secret(upper):
        return False
    if upper in ENV_VALUE_ALLOW:
        return True
    return any(upper.startswith(prefix) for prefix in ENV_VALUE_PREFIXES)


def _statements(line: str) -> list[str]:
    """Split one logical line on ``;``, ``&&``, and ``||`` outside quotes.

    A trailing ``&`` ends the command. ``2>&1`` and ``&>`` stay intact.
    """
    parts: list[str] = []
    buf: list[str] = []
    quote = ""
    index = 0
    while index < len(line):
        ch = line[index]
        if quote:
            buf.append(ch)
            if ch == quote and not (quote == '"' and index and line[index - 1] == "\\"):
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            index += 1
            continue
        if line.startswith("[[", index):
            buf.append("[[")
            depth = 2
            index += 2
            while index < len(line) and depth:
                if line.startswith("[[", index):
                    depth += 2
                    buf.append("[[")
                    index += 2
                    continue
                if line.startswith("]]", index):
                    depth -= 2
                    buf.append("]]")
                    index += 2
                    continue
                buf.append(line[index])
                index += 1
            continue
        if ch == ";" or line.startswith("&&", index) or line.startswith("||", index):
            text = "".join(buf).strip()
            if text:
                parts.append(text)
            buf = []
            index += 1 if ch == ";" else 2
            continue
        if ch == "&":
            nxt = line[index + 1] if index + 1 < len(line) else ""
            prev = ""
            for earlier in reversed(buf):
                if not earlier.isspace():
                    prev = earlier
                    break
            if prev == ">" or nxt == ">":
                buf.append(ch)
                index += 1
                continue
            text = "".join(buf).strip()
            if text:
                parts.append(text)
            return parts
        buf.append(ch)
        index += 1
    text = "".join(buf).strip()
    if text:
        parts.append(text)
    return parts


_MARK = "\x1e"
_REDIR_ATTACHED = re.compile(r"^(?:[0-9]*>>.+|[0-9]*>&[0-9]+|&>.+|[0-9]*>.+|[0-9]*<.+)$")
_REDIR_OP = re.compile(r"^(?:[0-9]*>>|[0-9]*>&|&>|[0-9]*>|[0-9]*<)$")


def _mark(name: str) -> str:
    return f"{_MARK}{name}{_MARK}"


def _marked_names(text: str) -> list[str]:
    return re.findall(re.escape(_MARK) + r"([A-Za-z_][A-Za-z0-9_]*)" + re.escape(_MARK), str(text))


def _insert(value: str, quote: str) -> str:
    if quote == '"':
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
    return value


def _closing_brace(text: str, open_at: int) -> int:
    """Index of the ``}`` that matches the ``{`` at ``open_at``. Nested braces count."""
    depth = 0
    index = open_at
    while index < len(text):
        ch = text[index]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError("unbalanced shell parameter expansion")


def _split_param(body: str) -> tuple[str, str, str]:
    """``name``, operator, word. Operators are ``:-`` ``-`` ``:+`` ``+`` ``:=``."""
    if not body or not (body[0].isalpha() or body[0] == "_"):
        raise ValueError("could not parse shell parameter expansion")
    index = 1
    while index < len(body) and (body[index].isalnum() or body[index] == "_"):
        index += 1
    name = body[:index]
    rest = body[index:]
    if not _is_name(name):
        raise ValueError("could not parse shell parameter expansion")
    if rest == "":
        return name, "", ""
    for op in (":-", ":+", ":=", "-", "+"):
        if rest.startswith(op):
            return name, op, rest[len(op):]
    raise ValueError("could not parse shell parameter expansion")


def _positional_at(body: str, positional: dict[str, str] | None, quote: str) -> str:
    if positional is None or body not in positional:
        raise ValueError(f"unresolved shell variable {body}")
    return _insert(positional[body], quote)


def _join_positional(positional: dict[str, str]) -> str:
    order = sorted(int(key) for key in positional if str(key).isdigit())
    return " ".join(positional[str(index)] for index in order)


def _command_subst_at(text: str, index: int) -> tuple[str, int]:
    """Body and the index after a ``$(...)`` that starts at ``index``.

    Parentheses inside double quotes still count, so ``\"$(dirname ...)\"`` nests.
    Single quotes hide parentheses.
    """
    if not text.startswith("$(", index):
        raise ValueError("unresolved shell command substitution")
    depth = 1
    quote = ""
    cursor = index + 2
    while cursor < len(text):
        ch = text[cursor]
        if quote == "'":
            if ch == "'":
                quote = ""
            cursor += 1
            continue
        if ch == "'":
            quote = "'"
            cursor += 1
            continue
        if ch == '"':
            quote = "" if quote == '"' else '"'
            cursor += 1
            continue
        if text.startswith("$(", cursor):
            depth += 1
            cursor += 2
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[index + 2:cursor], cursor + 1
        cursor += 1
    raise ValueError("unresolved shell command substitution")


def _is_script_dir_command(body: str) -> bool:
    """True for the ``dirname`` / ``pwd`` / ``realpath`` idiom of ``$0`` or ``BASH_SOURCE``."""
    text = body.replace('"', "").replace("'", "").strip()
    guard = 0
    while "$(" in text:
        guard += 1
        if guard > 6:
            return False
        start = text.find("$(")
        try:
            inner, end = _command_subst_at(text, start)
        except ValueError:
            return False
        if not _is_script_dir_command(inner):
            return False
        text = (text[:start] + "DIR" + text[end:]).strip()
    text = re.sub(r"\s+", " ", text).strip()
    if text in {"pwd", "DIR"}:
        return True
    if re.fullmatch(r"dirname (?:\$0|\$\{0\}|\$\{BASH_SOURCE\[0\]\}|DIR)", text):
        return True
    if re.fullmatch(r"realpath (?:\$0|\$\{0\}|\$\{BASH_SOURCE\[0\]\}|DIR)", text):
        return True
    return bool(re.fullmatch(r"cd DIR && pwd", text))


def _indirect_value(
    spec: str,
    env: dict[str, str],
    tainted: set[str],
    positional: dict[str, str] | None,
    script_dir: str | None,
) -> str:
    """``${ !name :- word }`` without the leading bang. The value is a known name or the default."""
    if not spec or not (spec[0].isalpha() or spec[0] == "_"):
        raise ValueError("unresolved shell variable !")
    name, op, word = _split_param(spec)
    if name not in env and name not in tainted:
        raise ValueError(f"unresolved shell variable {name}")
    target = "" if name in tainted else str(env.get(name) or "")
    if not _is_name(target):
        if op in (":-", "-", ":="):
            return _expand(word, env, tainted, positional, script_dir) if word else ""
        return ""
    return _expand("${" + target + op + word + "}", env, tainted, positional, script_dir)


def _param_at(
    text: str,
    index: int,
    env: dict[str, str],
    tainted: set[str],
    quote: str,
    positional: dict[str, str] | None = None,
    script_dir: str | None = None,
) -> tuple[str, int]:
    """Expand one ``$...``. A secret, or a value that only carries one, is a marker."""
    if text.startswith("$(", index):
        body, end = _command_subst_at(text, index)
        if script_dir and _is_script_dir_command(body):
            return _insert(script_dir, quote), end
        raise ValueError("unresolved shell command substitution")
    if text.startswith("${", index):
        end = _closing_brace(text, index + 1)
        body = text[index + 2:end]
        if body in ("*", "@"):
            if positional is None:
                raise ValueError(f"unresolved shell variable {body}")
            return _insert(_join_positional(positional), quote), end + 1
        if body.startswith("!"):
            return _insert(_indirect_value(body[1:], env, tainted, positional, script_dir), quote), end + 1
        if body.isdigit():
            return _positional_at(body, positional, quote), end + 1
        name, op, word = _split_param(body)
        secretish = _secret_name(name) or name in tainted
        present = name in env and name not in tainted
        raw = str(env.get(name) or "")
        if op in (":-", ":+", ":=") and present and raw == "":
            present = False
        # Not set in this script: :+ / + are empty, :- / - use the default.
        # The unused word is not expanded, so a nested bare reference does not refuse.
        if op in (":+", "+") and not present:
            chosen = ""
        elif op in (":-", "-", ":=") and not present:
            chosen = _expand(word, env, tainted, positional, script_dir)
            if secretish and _only_marker(chosen):
                chosen = _mark(name)
            elif secretish and chosen == "":
                chosen = _mark(name)
        elif op in (":-", "-", ":=") and present:
            chosen = raw
        elif op in (":+", "+") and present:
            chosen = _expand(word, env, tainted, positional, script_dir)
        elif present and name not in tainted:
            chosen = raw
        elif secretish:
            chosen = _mark(name)
        else:
            raise ValueError(f"unresolved shell variable {name}")
        return _insert(chosen, quote), end + 1
    if text.startswith("$*", index) or text.startswith("$@", index):
        if positional is None:
            raise ValueError("unresolved shell variable *")
        return _insert(_join_positional(positional), quote), index + 2
    if index + 1 < len(text) and text[index + 1].isdigit():
        nxt = index + 1
        while nxt < len(text) and text[nxt].isdigit():
            nxt += 1
        return _positional_at(text[index + 1:nxt], positional, quote), nxt
    match = _NAME.match(text[index + 1:])
    name = match.group(0) if match else ""
    if not _is_name(name):
        raise ValueError("unresolved shell variable")
    nxt = index + 1 + len(name)
    if name in tainted or (_secret_name(name) and name not in env):
        return _insert(_mark(name), quote), nxt
    if name in env:
        return _insert(env[name], quote), nxt
    raise ValueError(f"unresolved shell variable {name}")


def _expand(
    text: str,
    env: dict[str, str],
    tainted: set[str] | None = None,
    positional: dict[str, str] | None = None,
    script_dir: str | None = None,
) -> str:
    """Expand ``$VAR`` and ``${...}`` outside single quotes. Never runs the script.

    ``${...}`` uses brace matching, so nested defaults such as ``${A:-${B:-x}}``
    resolve. A name that is not set in the script is empty for ``:+`` / ``+``
    and uses the default for ``:-`` / ``-``. A secret-like name stays a marker.
    """
    tainted = tainted if tainted is not None else set()
    out: list[str] = []
    quote = ""
    index = 0
    while index < len(text):
        ch = text[index]
        if quote:
            if ch == quote and not (quote == '"' and index and text[index - 1] == "\\"):
                quote = ""
                out.append(ch)
                index += 1
                continue
        elif ch in ("'", '"'):
            quote = ch
            out.append(ch)
            index += 1
            continue
        if quote != "'" and ch == "`":
            raise ValueError("unresolved shell command substitution")
        if quote != "'" and ch == "$":
            piece, index = _param_at(text, index, env, tainted, quote, positional, script_dir)
            out.append(piece)
            continue
        out.append(ch)
        index += 1
    return "".join(out)


def _strip_redirections(argv: list[str]) -> list[str]:
    """Drop shell redirections and stop at a background ``&``."""
    out: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "&":
            break
        if _REDIR_ATTACHED.match(token):
            index += 1
            continue
        if _REDIR_OP.match(token):
            index += 2 if index + 1 < len(argv) else 1
            continue
        out.append(token)
        index += 1
    return out


def _pure_secret(value: str, name: str) -> bool:
    marks = _marked_names(value)
    if not marks:
        return False
    return marks == [name] and value.replace(_mark(name), "") == ""


def _only_marker(value: str) -> bool:
    marks = _marked_names(value)
    if not marks:
        return False
    cleaned = str(value)
    for name in marks:
        cleaned = cleaned.replace(_mark(name), "")
    return cleaned == ""


def _stored_assignment(name: str, value: str, tainted: set[str]) -> tuple[str | None, bool]:
    """``(value to keep, passthrough)``. A secret, or a value that only carries one, is not kept."""
    if _secret_name(name):
        tainted.add(name)
        return None, True
    if _only_marker(value):
        tainted.add(name)
        return None, False
    marks = _marked_names(value)
    if marks:
        raise ValueError(f"unresolved shell variable {marks[0]}")
    return value, False


def _take_env_flags(argv: list[str], *, tolerate_marks: bool = False) -> tuple[list[str], list[str]]:
    """Pull ``-e SECRET`` passthroughs out of argv. Other tokens stay."""
    out: list[str] = []
    passed: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        flag = token in ("-e", "--env")
        spec = ""
        if flag and index + 1 < len(argv):
            spec = argv[index + 1]
        elif token.startswith("--env="):
            flag = True
            spec = token.split("=", 1)[1]
        if flag and spec:
            name, value = spec, ""
            if "=" in spec:
                name, value = spec.split("=", 1)
            if _secret_name(name) and (value == "" or _only_marker(value) or not _marked_names(value)):
                if _marked_names(value) and not _only_marker(value):
                    raise ValueError(f"unresolved shell variable {_marked_names(value)[0]}")
                if name not in passed:
                    passed.append(name)
                index += 2 if token in ("-e", "--env") else 1
                continue
            if _marked_names(spec):
                raise ValueError(f"unresolved shell variable {_marked_names(spec)[0]}")
        if _marked_names(token):
            if not tolerate_marks:
                raise ValueError(f"unresolved shell variable {_marked_names(token)[0]}")
            index += 1
            continue
        out.append(token)
        index += 1
    return out, passed


def _is_name(token: str) -> bool:
    return bool(_NAME.fullmatch(token))


def _is_assignment(token: str) -> bool:
    if "=" not in token:
        return False
    return _is_name(token.split("=", 1)[0])


def _server_binary(token: str) -> bool:
    base = token.rsplit("/", 1)[-1]
    return base == "llama-server"


def _could_be_launch(statement: str) -> bool:
    lowered = statement.strip()
    if lowered.startswith("export ") or lowered.startswith("export\t"):
        return True
    for word in ("setsid", "nohup", "exec", "llama-server", "docker run", "docker compose", "docker-compose"):
        if word in lowered:
            return True
    return False


def _strip_prefixes(parts: list[str], env: dict[str, str]) -> list[str]:
    while parts:
        head = parts[0]
        if _is_assignment(head):
            key, value = head.split("=", 1)
            env[key] = value
            parts = parts[1:]
            continue
        if head not in _LAUNCH_PREFIXES:
            break
        parts = parts[1:]
        while parts and parts[0].startswith("-"):
            flag = parts[0]
            parts = parts[1:]
            if flag in ("-u", "-g") and parts and not parts[0].startswith("-"):
                parts = parts[1:]
    return parts


_ARRAY_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(\+?=)\((.*)\)\s*$", re.DOTALL)
_FUNC_WITH_BRACE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)\s*\{")
_FUNC_NAME_ONLY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*\(\s*\)\s*$")
_IF_LINE = re.compile(r"^if(?:\s|$)")
_KEYWORDS = {"if", "then", "elif", "else", "fi", "for", "do", "done"}
_TRUTHY = {"true", "yes", "on", "1"}
_FALSY = {"false", "no", "off", "0"}
_DOCKER_VALUE_FLAGS = frozenset({
    "-v", "--volume", "--mount", "--network", "--net", "--pid",
    "--cap-add", "--cap-drop", "--security-opt", "--device", "--entrypoint",
    "--gpus", "--name", "-e", "--env", "-p", "--publish", "--expose",
    "--ulimit", "--shm-size", "--ipc", "--user", "-u", "-w", "--workdir",
    "--label", "-l", "--hostname", "--add-host", "--group-add", "--runtime",
    "--memory", "-m", "--cpus", "--cpu-shares", "--restart",
    "--log-driver", "--log-opt", "--tmpfs", "--sysctl", "--pull", "--platform",
    "--cgroupns", "--userns", "--uts", "--memory-swap", "--health-cmd",
    "--health-interval", "--device-cgroup-rule",
})
_MAX_CALL_DEPTH = 8


class _ScriptState:
    def __init__(self, runner: Any = None, host: str = "", cwd: str = "", path: str = "") -> None:
        self.assignments: dict[str, str] = {}
        self.tainted: set[str] = set()
        self.arrays: dict[str, list[str]] = {}
        self.functions: dict[str, list[str]] = {}
        self.events: list[dict[str, Any]] = []
        self.runner = runner
        self.host = host
        self.cwd = cwd
        self.path = path
        self.line = 1
        self.line_text = ""
        self.allexport = False
        self.echo_fns: set[str] = set()


def _first_var(text: str) -> str:
    match = re.search(r"\$\{?!?([A-Za-z_][A-Za-z0-9_]*)", str(text or ""))
    return match.group(1) if match else ""


def _locate(message: str, state: _ScriptState) -> str:
    """Name the file and line on a variable or condition refusal."""
    file = state.path or "script"
    line = state.line or 1
    suffix = f" in {file}:{line}"
    interesting = (
        "unresolved shell variable" in message
        or "could not resolve shell condition" in message
        or message.startswith("could not check")
        or "command substitution" in message
        or message.startswith("could not parse shell")
    )
    if not interesting:
        return message
    if message.strip() == "unresolved shell variable":
        found = _first_var(state.line_text)
        if found:
            message = f"unresolved shell variable {found}"
    if suffix not in message:
        message = f"{message}{suffix}"
    return message


def _note_error(state: _ScriptState, exc: Exception) -> None:
    state.events.append({"kind": "error", "message": _locate(str(exc), state)})


def _is_array_assign(statement: str) -> bool:
    return _ARRAY_ASSIGN.match(statement.strip()) is not None


def _is_function_header(line: str) -> bool:
    return _FUNC_WITH_BRACE.match(line) is not None or _FUNC_NAME_ONLY.match(line) is not None


def _scan_quotes(text: str):
    """Yield ``(index, char, quote)`` with ``quote`` set outside of quotes."""
    quote = ""
    index = 0
    while index < len(text):
        ch = text[index]
        if quote:
            if ch == quote and not (quote == '"' and index and text[index - 1] == "\\"):
                quote = ""
            yield index, ch, quote
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        yield index, ch, ""
        index += 1


def _keywords(text: str) -> list[tuple[str, int, int]]:
    found: list[tuple[str, int, int]] = []
    quote = ""
    index = 0
    while index < len(text):
        ch = text[index]
        if quote:
            if ch == quote and not (quote == '"' and index and text[index - 1] == "\\"):
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            index += 1
            continue
        if ch == "#":
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline
            continue
        if ch.isalpha() or ch == "_":
            end = index + 1
            while end < len(text) and (text[end].isalnum() or text[end] == "_"):
                end += 1
            word = text[index:end]
            prev = text[index - 1] if index else " "
            if word in _KEYWORDS and not (prev.isalnum() or prev == "_"):
                found.append((word, index, end))
            index = end
            continue
        index += 1
    return found


def _matching_brace(text: str, open_at: int) -> int:
    depth = 0
    for index, ch, quote in _scan_quotes(text):
        if quote or index < open_at:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("could not parse shell function")


def _read_balanced(lines: list[str], start: int) -> tuple[str, int, str]:
    """Function text through the matching ``}``, the next line, and any trailing tail."""
    depth = 0
    seen = False
    chunks: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        close_at = -1
        for column, ch, quote in _scan_quotes(line):
            if quote:
                continue
            if ch == "{":
                depth += 1
                seen = True
            elif ch == "}" and seen:
                depth -= 1
                if depth == 0:
                    close_at = column
                    break
        if close_at >= 0:
            chunks.append(line[:close_at + 1])
            return "\n".join(chunks), index + 1, line[close_at + 1:].strip()
        chunks.append(line)
        index += 1
    raise ValueError("could not parse shell function")


def _function_parts(text: str) -> tuple[str, list[str]]:
    match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)", text.strip())
    if not match:
        raise ValueError("could not parse shell function")
    open_at = text.find("{")
    if open_at < 0:
        raise ValueError("could not parse shell function")
    close_at = _matching_brace(text, open_at)
    inner = text[open_at + 1:close_at]
    body = [line.strip() for line in inner.splitlines() if line.strip() and not line.strip().startswith("#")]
    return match.group(1), body


def _read_if(lines: list[str], start: int) -> tuple[str, int, str]:
    depth = 0
    chunks: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        close_at = -1
        for word, word_at, word_end in _keywords(line):
            if word == "if":
                depth += 1
            elif word == "fi":
                depth -= 1
                if depth == 0:
                    close_at = word_end
                    break
        if close_at >= 0:
            chunks.append(line[:close_at])
            return "\n".join(chunks), index + 1, line[close_at:].strip()
        chunks.append(line)
        index += 1
    raise ValueError("could not resolve shell condition: unclosed if")


def _if_branches(text: str) -> list[tuple[str | None, str]]:
    keys = _keywords(text)
    if not keys or keys[0][0] != "if":
        raise ValueError("could not resolve shell condition: unclosed if")
    branches: list[tuple[str | None, str]] = []
    depth = 0
    cond_at = 0
    body_at: int | None = None
    pending: str | None = None
    else_branch = False
    for word, start, end in keys:
        if word == "if":
            depth += 1
            if depth == 1:
                cond_at = end
                body_at = None
                else_branch = False
        elif word == "then" and depth == 1:
            pending = text[cond_at:start].strip().strip(";").strip()
            body_at = end
        elif word == "elif" and depth == 1 and body_at is not None:
            branches.append((pending, text[body_at:start]))
            cond_at = end
            body_at = None
            else_branch = False
        elif word == "else" and depth == 1 and body_at is not None:
            branches.append((pending, text[body_at:start]))
            pending = None
            body_at = end
            else_branch = True
        elif word == "fi":
            if depth == 1 and body_at is not None:
                cond = None if else_branch else pending
                branches.append((cond, text[body_at:start]))
            depth -= 1
    if not branches:
        raise ValueError("could not resolve shell condition: unclosed if")
    return branches


def _split_and_or(text: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    quote = ""
    bracket = 0
    index = 0
    op = ""

    def flush(next_op: str) -> None:
        nonlocal op
        atom = "".join(buf).strip()
        del buf[:]
        if atom:
            parts.append((op, atom))
        op = next_op

    while index < len(text):
        ch = text[index]
        if quote:
            buf.append(ch)
            if ch == quote and not (quote == '"' and index and text[index - 1] == "\\"):
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            index += 1
            continue
        if text.startswith("[[", index):
            bracket += 2
            buf.append("[[")
            index += 2
            continue
        if text.startswith("]]", index) and bracket >= 2:
            bracket -= 2
            buf.append("]]")
            index += 2
            continue
        if ch == "[":
            bracket += 1
            buf.append(ch)
            index += 1
            continue
        if ch == "]" and bracket:
            bracket -= 1
            buf.append(ch)
            index += 1
            continue
        if bracket == 0 and (text.startswith("&&", index) or text.startswith("||", index)):
            flush(text[index:index + 2])
            index += 2
            continue
        buf.append(ch)
        index += 1
    flush("")
    return parts


def _truth_word(word: str, original: str) -> bool:
    low = word.lower()
    if low in _TRUTHY:
        return True
    if low in _FALSY:
        return False
    raise ValueError(f"could not resolve shell condition: {original}")


def _script_dir_of(state: _ScriptState | None) -> str | None:
    """Directory of the script path on the node. No local realpath."""
    if state is None:
        return None
    path = str(state.path or "")
    if path.startswith("/"):
        parent = os.path.dirname(os.path.normpath(path))
        return parent or "/"
    cwd = str(state.cwd or "")
    if cwd.startswith("/"):
        return cwd
    return None


def _node_path(path: str, cwd: str) -> str:
    text = str(path or "")
    if text.startswith("/"):
        return os.path.normpath(text)
    base = str(cwd or "")
    if not base.startswith("/"):
        raise ValueError(f"could not check {text or 'file'} on the node")
    return os.path.normpath(os.path.join(base, text))


def _node_code(state: _ScriptState, argv: list[str]) -> int | None:
    """Exit code of one read-only argv on the script's node. ``None`` means no runner."""
    if state.runner is None or not state.host:
        return None
    capture = getattr(state.runner, "capture", None)
    if capture is None:
        return None
    try:
        result = capture(state.host, argv, 8)
    except Exception as exc:  # noqa: BLE001 — reported as a check failure, without the payload
        raise ValueError(f"could not check {argv[-1]} on the node") from exc
    if not isinstance(result, dict):
        return None
    if result.get("ok") or result.get("code") == 0:
        return 0
    try:
        return int(result.get("code"))
    except (TypeError, ValueError):
        return None


def _node_file_exists(state: _ScriptState, path: str) -> bool:
    absolute = _node_path(path, state.cwd)
    code = _node_code(state, readonly_argv("exists", absolute))
    if code == 0:
        return True
    if code == 1:
        return False
    raise ValueError(f"could not check {absolute} on the node")


def _unquote_env(value: str) -> str:
    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        return text[1:-1]
    if " #" in text:
        text = text.split(" #", 1)[0].rstrip()
    return text


def _env_line(line: str) -> tuple[str, str, str] | None:
    """One env line. ``value``, ``review``, or ``secret``. Secret values are not returned."""
    raw = line.strip()
    if not raw or raw.startswith("#"):
        return None
    if raw.startswith("export ") or raw.startswith("export\t"):
        raw = raw.split(None, 1)[1].strip()
    if "=" not in raw:
        raise ValueError("could not parse env file")
    name, raw_value = raw.split("=", 1)
    name = name.strip()
    if not _is_name(name):
        raise ValueError("could not parse env file")
    if "$" in raw_value or "`" in raw_value:
        raise ValueError(f"refusing expansion in env file for {name}")
    kept = _unquote_env(raw_value)
    if name.upper() == "EVICT_PAGE_CACHE":
        on = str(kept).strip().lower() in ("1", "true", "yes", "on")
        return name, "true" if on else "false", "review"
    if value_looks_like_token(kept):
        return name, "", "secret"
    if name.upper() in _ENV_REVIEW:
        return name, kept, "review"
    if env_value_allowed(name):
        return name, kept, "value"
    return name, "", "secret"


def parse_env_file(text: str) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Parse ``KEY=VALUE`` lines. Secret and unknown values are dropped on the line.

    ``EXTRA_DOCKER_ARGS`` is returned in the review map, not as an env value.
    """
    values: dict[str, str] = {}
    secrets: list[str] = []
    review: dict[str, str] = {}
    for line in str(text or "").splitlines():
        parsed = _env_line(line)
        if parsed is None:
            continue
        name, value, kind = parsed
        if kind == "value":
            values[name] = value
        elif kind == "review":
            review[name] = value
        elif name not in secrets:
            secrets.append(name)
    return values, secrets, review


def _is_env_basename(path: str) -> bool:
    base = os.path.basename(str(path or ""))
    return base == ".env" or base.endswith(".env")


def _read_node_text(state: _ScriptState, path: str) -> str:
    if state.runner is None or not state.host:
        raise ValueError(f"could not read {path} on the node")
    text, err = _read(state.runner, state.host, readonly_argv("script", path))
    if err and not text:
        raise ValueError(f"could not read {path} on the node")
    return text


def _source_env(token: str, state: _ScriptState) -> None:
    path = _node_path(token, state.cwd)
    if not _is_env_basename(path):
        return
    values, secrets, review = parse_env_file(_read_node_text(state, path))
    exported: dict[str, str] = {}
    for name, value in values.items():
        state.assignments[name] = value
        state.tainted.discard(name)
        if state.allexport:
            exported[name] = value
    if exported:
        state.events.append({"kind": "exports", "env": exported, "pass": []})
    passed: list[str] = []
    for name in secrets:
        state.tainted.add(name)
        state.assignments[name] = _mark(name)
        if state.allexport:
            passed.append(name)
    if passed:
        state.events.append({"kind": "pass", "name": passed[0], "names": passed})
    extra = review.get("EXTRA_DOCKER_ARGS") or ""
    if extra:
        state.events.append({"kind": "review", "extra_docker_args": extra})
    if "EVICT_PAGE_CACHE" in review:
        state.events.append({"kind": "evict", "evict_page_cache": review["EVICT_PAGE_CACHE"] == "true"})


def _eval_test(
    inner: str,
    env: dict[str, str],
    tainted: set[str],
    positional: dict[str, str],
    state: _ScriptState | None = None,
) -> bool:
    if not inner.strip():
        raise ValueError("could not resolve shell condition: empty test")
    logic = _split_and_or(inner)
    if len(logic) > 1:
        value = False
        for index, (op, atom) in enumerate(logic):
            if index and ((op == "&&" and not value) or (op == "||" and value)):
                continue
            bit = _eval_test(atom, env, tainted, positional, state)
            if index == 0 or op not in ("&&", "||"):
                value = bit
            elif op == "&&":
                value = value and bit
            else:
                value = value or bit
        return value
    expanded = _expand(inner, env, tainted, positional, _script_dir_of(state))
    try:
        words = shlex.split(expanded, posix=True)
    except ValueError as exc:
        raise ValueError(f"could not resolve shell condition: {inner.strip()}") from exc
    negated = False
    if words and words[0] == "!":
        negated = True
        words = words[1:]
    if len(words) == 2 and words[0] == "-f":
        if state is None:
            raise ValueError(f"could not check {words[1]} on the node")
        exists = _node_file_exists(state, words[1])
        return (not exists) if negated else exists
    if negated:
        raise ValueError(f"could not resolve shell condition: {inner.strip()}")
    if len(words) == 2 and words[0] == "-n":
        return words[1] != ""
    if len(words) == 2 and words[0] == "-z":
        return words[1] == ""
    if len(words) == 3 and words[1] in ("=", "=="):
        return words[0] == words[2]
    if len(words) == 3 and words[1] == "!=":
        return words[0] != words[2]
    raise ValueError(f"could not resolve shell condition: {inner.strip()}")


def _eval_atom(
    atom: str,
    env: dict[str, str],
    tainted: set[str],
    positional: dict[str, str],
    state: _ScriptState | None = None,
) -> bool:
    text = atom.strip()
    if text.startswith("[[") and text.endswith("]]"):
        return _eval_test(text[2:-2], env, tainted, positional, state)
    if text.startswith("[") and text.endswith("]"):
        return _eval_test(text[1:-1], env, tainted, positional, state)
    if text == "test" or text.startswith("test "):
        return _eval_test(text[5:] if text.startswith("test ") else "", env, tainted, positional, state)
    expanded = _expand(text, env, tainted, positional, _script_dir_of(state))
    try:
        words = shlex.split(expanded, posix=True)
    except ValueError as exc:
        raise ValueError(f"could not resolve shell condition: {text}") from exc
    if len(words) == 1 and not _marked_names(words[0]):
        return _truth_word(words[0], text)
    raise ValueError(f"could not resolve shell condition: {text}")


def _eval_cond(
    text: str,
    env: dict[str, str],
    tainted: set[str],
    positional: dict[str, str],
    state: _ScriptState | None = None,
) -> bool:
    parts = _split_and_or(text)
    if not parts:
        raise ValueError(f"could not resolve shell condition: {text.strip() or 'empty'}")
    value = False
    for index, (op, atom) in enumerate(parts):
        if index and ((op == "&&" and not value) or (op == "||" and value)):
            continue
        bit = _eval_atom(atom, env, tainted, positional, state)
        if index == 0 or op not in ("&&", "||"):
            value = bit
        elif op == "&&":
            value = value and bit
        else:
            value = value or bit
    return value


def _splice_arrays(text: str, arrays: dict[str, list[str]]) -> str:
    """Replace ``${name[@]}`` with the array words. An unknown array is refused."""
    out: list[str] = []
    quote = ""
    index = 0
    while index < len(text):
        if quote != "'" and text.startswith("${", index):
            try:
                end = _closing_brace(text, index + 1)
            except ValueError:
                end = -1
            token = text[index:end + 1] if end >= 0 else ""
            match = re.fullmatch(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\[@\]\}", token)
            if match:
                name = match.group(1)
                if name not in arrays:
                    raise ValueError(f"unresolved shell variable {name}")
                words = " ".join(shlex.quote(item) for item in arrays[name])
                if quote == '"' and out and out[-1] == '"':
                    out.pop()
                    out.append(words)
                    if end + 1 < len(text) and text[end + 1] == '"':
                        index = end + 2
                    else:
                        index = end + 1
                    quote = ""
                    continue
                if quote == "":
                    out.append(words)
                    index = end + 1
                    continue
        ch = text[index]
        if quote:
            out.append(ch)
            if ch == quote and not (quote == '"' and index and text[index - 1] == "\\"):
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        out.append(ch)
        index += 1
    return "".join(out)


def _apply_array(statement: str, state: _ScriptState, positional: dict[str, str]) -> None:
    match = _ARRAY_ASSIGN.match(statement.strip())
    if not match:
        raise ValueError("could not parse shell array")
    name, op, inner = match.group(1), match.group(2), match.group(3)
    expanded = _expand(inner, state.assignments, state.tainted, positional, _script_dir_of(state)) if inner.strip() else ""
    words = shlex.split(expanded, posix=True) if expanded.strip() else []
    if op == "=":
        state.arrays[name] = words
    else:
        state.arrays.setdefault(name, []).extend(words)


def _apply_local(parts: list[str], state: _ScriptState, local_names: set[str]) -> None:
    for token in parts[1:]:
        if not _is_assignment(token):
            continue
        name, value = token.split("=", 1)
        stored, _passed = _stored_assignment(name, value, state.tainted)
        local_names.add(name)
        if stored is None:
            state.assignments.pop(name, None)
            if name in state.tainted:
                state.assignments[name] = _mark(name)
        else:
            state.assignments[name] = stored
            state.tainted.discard(name)


def _split_docker_flag(token: str) -> tuple[str, str | None]:
    if token.startswith("--"):
        name, eq, value = token.partition("=")
        return name, (value if eq else None)
    return token, None


def _role_from_docker_run(argv: list[str], prefix_env: dict[str, str], passes: list[str]) -> dict[str, Any]:
    """A ``docker run`` argv becomes one container role. The command is not executed."""
    mounts: list[str] = []
    env: dict[str, str] = {}
    network = ""
    gpus = ""
    ipc = ""
    shm = ""
    container = ""
    caps: list[str] = []
    opts: list[str] = []
    devices: list[str] = []
    privileged = False
    pid = ""
    entry: list[str] = []
    index = 2
    while index < len(argv):
        token = argv[index]
        if token == "--" or not str(token).startswith("-"):
            break
        flag, inline = _split_docker_flag(token)
        value = inline
        step = 1
        if inline is None and flag in _DOCKER_VALUE_FLAGS:
            value = argv[index + 1] if index + 1 < len(argv) else ""
            step = 2
        if flag in ("-v", "--volume") and value:
            mounts.append(value)
        elif flag in ("-e", "--env") and value and "=" in value:
            key, raw = value.split("=", 1)
            if key:
                env[key] = raw
        elif flag in ("--network", "--net") and value is not None:
            network = value
        elif flag == "--gpus" and value is not None:
            gpus = value
        elif flag == "--name" and value is not None:
            container = value
        elif flag == "--cap-add" and value:
            caps.append(value)
        elif flag == "--security-opt" and value:
            opts.append(value)
        elif flag == "--device" and value:
            devices.append(value)
        elif flag == "--privileged":
            privileged = True
        elif flag == "--pid" and value is not None:
            pid = value
        elif flag == "--ipc" and value is not None:
            ipc = value
        elif flag == "--shm-size" and value is not None:
            shm = value
        elif flag == "--entrypoint" and value:
            try:
                entry = shlex.split(value)
            except ValueError:
                entry = [value]
        index += step
    image = argv[index] if index < len(argv) else ""
    args = list(argv[index + 1:])
    entrypoint = ""
    if entry:
        entrypoint = entry[0]
        args = entry[1:] + args
    elif len(args) == 2 and shell_basename(args[0]) and args[1].startswith("/") and args[1].endswith((".sh", ".bash")):
        # ``docker run image bash /start.sh`` with the image entrypoint left as vllm.
        entrypoint, args = args[0], [args[1]]
    for key, value in prefix_env.items():
        if key not in env and key not in _SKIP_ENV and not _secret_name(key):
            env[key] = value
    role: dict[str, Any] = {
        "runtime": "docker",
        "image": image,
        "env": env,
        "mounts": mounts,
        "network": network or "host",
        "gpus": gpus,
        "ipc": ipc,
        "shm": shm,
        "container": container,
        "args": args,
        "devices": devices,
    }
    if entrypoint:
        role["entrypoint"] = entrypoint
    if privileged:
        role["privileged"] = True
    if pid:
        role["pid"] = pid
    if caps:
        role["cap_add"] = caps
    if opts:
        role["security_opt"] = opts
    if passes:
        role["env_pass"] = list(passes)
    return role


def _exec_statement(
    statement: str,
    state: _ScriptState,
    positional: dict[str, str] | None,
    local_names: set[str],
    depth: int,
) -> bool | None:
    """Run one statement. ``False`` means a test failed; ``None`` means it ran."""
    head = statement.strip()
    if head.startswith("[") or head.startswith("[[") or head.startswith("test ") or head == "test":
        return _eval_atom(head, state.assignments, state.tainted, positional, state)
    if _is_array_assign(statement):
        _apply_array(statement, state, positional or {})
        return None
    spliced = _splice_arrays(statement, state.arrays)
    expanded = _expand(spliced, state.assignments, state.tainted, positional, _script_dir_of(state))
    try:
        parts = shlex.split(expanded, posix=True)
    except ValueError as exc:
        raise ValueError("could not parse the shell line") from exc
    if not parts:
        return None
    name = parts[0]
    if name == "set" and len(parts) >= 2 and parts[1] in ("-a", "+a"):
        state.allexport = parts[1] == "-a"
        return
    if name in ("source", ".") and len(parts) > 1:
        if _is_env_basename(parts[1]):
            _source_env(parts[1], state)
        return None
    if name in state.functions:
        _call_function(name, parts[1:], state, depth)
        return None
    if name == "local":
        _apply_local(parts, state, local_names)
        return None
    event = _statement_event(parts, state.assignments, state.tainted)
    if not event:
        return None
    if event.get("kind") == "cd" and str(event.get("path") or "").startswith("/"):
        state.cwd = str(event["path"])
    state.events.append(event)
    return None


def _echo_only(body: list[str]) -> bool:
    if not body:
        return False
    for line in body:
        for statement in _statements(line):
            head = statement.strip()
            if re.match(r"^(?:echo|printf|:|return)(?:\s|$)", head):
                continue
            return False
    return True


def _call_function(name: str, args: list[str], state: _ScriptState, depth: int) -> None:
    if depth >= _MAX_CALL_DEPTH:
        raise ValueError(f"could not resolve shell function {name}")
    saved = dict(state.assignments)
    local_names: set[str] = set()
    positional = {str(index + 1): arg for index, arg in enumerate(args)}
    use_strict = name not in state.echo_fns
    try:
        _run_lines(state.functions.get(name) or [], state, positional, local_names, use_strict, depth + 1)
    finally:
        for key in local_names:
            if key in saved:
                state.assignments[key] = saved[key]
            else:
                state.assignments.pop(key, None)


def _exec_if(
    text: str,
    state: _ScriptState,
    positional: dict[str, str],
    local_names: set[str],
    strict: bool,
    depth: int,
) -> None:
    taken = False
    for cond, body in _if_branches(text):
        if taken:
            break
        if cond is not None and not _eval_cond(cond, state.assignments, state.tainted, positional, state):
            continue
        taken = True
        body_lines = [line.strip() for line in body.splitlines() if line.strip()]
        _run_lines(body_lines, state, positional, local_names, strict, depth)


def _run_tail(
    tail: str,
    state: _ScriptState,
    positional: dict[str, str] | None,
    local_names: set[str],
    strict: bool,
    depth: int,
) -> None:
    text = tail.strip().lstrip(";").strip()
    if text:
        _run_lines([text], state, positional, local_names, strict, depth)


def _flow_parts(line: str) -> list[tuple[str, str]]:
    """Split on ``;``, ``&&``, and ``||``. The operator applies to the following statement."""
    parts: list[tuple[str, str]] = []
    buf: list[str] = []
    quote = ""
    index = 0
    op = ""

    def flush(next_op: str) -> None:
        nonlocal op
        text = "".join(buf).strip()
        del buf[:]
        if text:
            parts.append((op, text))
        op = next_op

    while index < len(line):
        ch = line[index]
        if quote:
            buf.append(ch)
            if ch == quote and not (quote == '"' and index and line[index - 1] == "\\"):
                quote = ""
            index += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            index += 1
            continue
        if line.startswith("[[", index):
            buf.append("[[")
            depth = 2
            index += 2
            while index < len(line) and depth:
                if line.startswith("[[", index):
                    depth += 2
                    buf.append("[[")
                    index += 2
                    continue
                if line.startswith("]]", index):
                    depth -= 2
                    buf.append("]]")
                    index += 2
                    continue
                buf.append(line[index])
                index += 1
            continue
        if ch == ";":
            flush("")
            index += 1
            continue
        if line.startswith("&&", index) or line.startswith("||", index):
            flush(line[index:index + 2])
            index += 2
            continue
        if ch == "&":
            nxt = line[index + 1] if index + 1 < len(line) else ""
            prev = ""
            for earlier in reversed(buf):
                if not earlier.isspace():
                    prev = earlier
                    break
            if prev == ">" or nxt == ">":
                buf.append(ch)
                index += 1
                continue
            flush("")
            return parts
        buf.append(ch)
        index += 1
    flush("")
    return parts


def _read_for(lines: list[str], start: int) -> tuple[str, int, str]:
    depth = 0
    chunks: list[str] = []
    index = start
    while index < len(lines):
        line = lines[index]
        close_at = -1
        for word, _word_at, word_end in _keywords(line):
            if word == "for":
                depth += 1
            elif word == "done":
                depth -= 1
                if depth == 0:
                    close_at = word_end
                    break
        if close_at >= 0:
            chunks.append(line[:close_at])
            return "\n".join(chunks), index + 1, line[close_at:].strip()
        chunks.append(line)
        index += 1
    raise ValueError("could not resolve shell condition: unclosed for")


def _exec_for(
    text: str,
    state: _ScriptState,
    positional: dict[str, str] | None,
    local_names: set[str],
    strict: bool,
    depth: int,
) -> None:
    match = re.match(
        r"^for\s+([A-Za-z_][A-Za-z0-9_]*)\s+in\s+(.*?)(?:;|\n)\s*do\b(.*)\bdone\s*$",
        text.strip(),
        re.S,
    )
    if not match:
        raise ValueError("could not resolve shell condition: for")
    name, words_src, body = match.group(1), match.group(2), match.group(3)
    expanded = _expand(words_src, state.assignments, state.tainted, positional, _script_dir_of(state))
    words = shlex.split(expanded, posix=True) if expanded.strip() else []
    body_lines = [line.strip() for line in body.splitlines() if line.strip()]
    had = name in state.assignments
    saved = state.assignments.get(name, "")
    try:
        for word in words:
            state.assignments[name] = word
            _run_lines(body_lines, state, positional, local_names, strict, depth, origin=state.line)
    finally:
        if had:
            state.assignments[name] = saved
        else:
            state.assignments.pop(name, None)


def _report_shell_error(statement: str, exc: Exception, strict: bool) -> bool:
    text = str(exc)
    return bool(
        strict
        or _could_be_launch(statement)
        or _is_array_assign(statement)
        or _is_source(statement)
        or "command substitution" in text
        or "could not resolve shell condition" in text
        or text.startswith("could not check")
    )


def _numbered(lines: list[Any], origin: int) -> list[tuple[int, str]]:
    numbered: list[tuple[int, str]] = []
    for offset, item in enumerate(lines):
        if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
            numbered.append((int(item[0]), str(item[1])))
        else:
            numbered.append((origin + offset, str(item)))
    return numbered


def _run_lines(
    lines: list[Any],
    state: _ScriptState,
    positional: dict[str, str] | None,
    local_names: set[str],
    strict: bool,
    depth: int,
    origin: int = 1,
) -> None:
    numbered = _numbered(lines, origin)
    texts = [text for _number, text in numbered]
    index = 0
    while index < len(numbered):
        number, raw = numbered[index]
        raw = raw.strip()
        state.line = number
        state.line_text = raw
        if not raw or raw.startswith("#"):
            index += 1
            continue
        if _is_function_header(raw):
            try:
                text, nxt, tail = _read_balanced(texts, index)
                name, body = _function_parts(text)
                state.functions[name] = body
                if _echo_only(body):
                    state.echo_fns.add(name)
                _run_tail(tail, state, positional, local_names, strict, depth)
            except ValueError as exc:
                _note_error(state, exc)
                return
            index = nxt
            continue
        if re.match(r"^for(?:\s|$)", raw):
            try:
                text, nxt, tail = _read_for(texts, index)
                _exec_for(text, state, positional, local_names, strict, depth)
                _run_tail(tail, state, positional, local_names, strict, depth)
            except ValueError as exc:
                _note_error(state, exc)
                return
            index = nxt
            continue
        if _IF_LINE.match(raw):
            try:
                text, nxt, tail = _read_if(texts, index)
                _exec_if(text, state, positional, local_names, strict, depth)
                _run_tail(tail, state, positional, local_names, strict, depth)
            except ValueError as exc:
                _note_error(state, exc)
                return
            index = nxt
            continue
        ok = True
        for op, statement in _flow_parts(raw):
            if op == "&&" and not ok:
                continue
            if op == "||" and ok:
                continue
            try:
                result = _exec_statement(statement, state, positional, local_names, depth)
                ok = True if result is None else bool(result)
            except ValueError as exc:
                ok = False
                if _report_shell_error(statement, exc, strict):
                    _note_error(state, exc)
        index += 1


def _is_source(statement: str) -> bool:
    head = statement.strip()
    return head.startswith("source ") or head.startswith(". ")


def _walk_script(
    text: str,
    runner: Any = None,
    host: str = "",
    cwd: str = "",
    path: str = "",
) -> list[dict[str, Any]]:
    """Assignments, functions, the taken ``if`` branch, and one launch. Nothing is executed.

    Flag tests use values assigned in the script, including ``${NAME:-default}``.
    ``[ -f path ]`` is checked with ``test -f`` on the node. A ``.env`` file is
    parsed as assignments; it is not executed. ``$(dirname ...)`` of ``$0`` or
    ``BASH_SOURCE`` becomes the script directory. A function body is expanded
    when the function is called.
    """
    state = _ScriptState(runner, host, cwd, path)
    _run_lines(_joined_lines(text), state, None, set(), False, 0)
    return state.events


_GUARD_WORDS = {"[", "[[", "test", "if", "elif", "fi", "else", "then"}


def _statement_event(parts: list[str], assignments: dict[str, str], tainted: set[str]) -> dict[str, Any] | None:
    """One statement: assignments, a cd, compose, or a llama-server launch."""
    parts = _strip_redirections(parts)
    if not parts:
        return None
    exporting = parts[0] == "export"
    body = list(parts[1:] if exporting else parts)
    prefix_env: dict[str, str] = {}
    inline: dict[str, str] = {}
    passes: list[str] = []
    if exporting or _is_assignment(body[0] if body else ""):
        index = 0
        while index < len(body):
            token = body[index]
            if _is_assignment(token):
                name, value = token.split("=", 1)
                stored, passed = _stored_assignment(name, value, tainted)
                if stored is None:
                    assignments.pop(name, None)
                    if passed and name not in passes:
                        passes.append(name)
                    elif name in tainted:
                        assignments[name] = _mark(name)
                else:
                    assignments[name] = stored
                    tainted.discard(name)
                    if exporting:
                        prefix_env[name] = stored
                    else:
                        inline[name] = stored
                index += 1
                continue
            if exporting and _is_name(token):
                if _secret_name(token):
                    if token not in passes:
                        passes.append(token)
                elif token not in assignments:
                    raise ValueError(f"unresolved shell variable {token}")
                else:
                    prefix_env[token] = assignments[token]
                index += 1
                continue
            break
        rest = body[index:]
        if not rest:
            if passes and not prefix_env:
                return {"kind": "pass", "name": passes[0], "names": passes}
            if prefix_env or passes:
                return {"kind": "exports", "env": prefix_env, "pass": passes}
            return None
        prefix_env.update(inline)
        body = rest
    while body and body[0] in ("then", "else", "do"):
        body = body[1:]
    if not body:
        return {"kind": "pass", "name": passes[0], "names": passes} if passes else None
    if body and body[0] in ("[", "[[", "test", "if", "elif", "fi"):
        # ``if [ -n "$secret" ]`` makes the following ``-e`` optional. The test itself is not argv.
        _guard_argv, flagged = _take_env_flags(list(body), tolerate_marks=True)
        if flagged:
            return {"kind": "pass", "name": flagged[0], "names": flagged}
        return None
    if body and body[0] == "cd" and len(body) > 1 and str(body[1]).startswith("/"):
        return {"kind": "cd", "path": body[1]}
    argv = _strip_prefixes(list(body), prefix_env)
    scrubbed: dict[str, str] = {}
    for key, value in prefix_env.items():
        if _secret_name(key):
            if key not in passes:
                passes.append(key)
            continue
        marks = _marked_names(str(value))
        if marks:
            raise ValueError(f"unresolved shell variable {marks[0]}")
        if key not in _SKIP_ENV:
            scrubbed[key] = str(value)
    prefix_env = scrubbed
    argv, flagged = _take_env_flags(_strip_redirections(argv))
    for name in flagged:
        if name not in passes:
            passes.append(name)
    if not argv:
        if passes:
            return {"kind": "pass", "name": passes[0], "names": passes}
        return None
    if _bare_compose(argv) or _compose_file_arg(argv):
        return {"kind": "compose", "argv": argv}
    if len(argv) >= 2 and argv[0] == "docker" and argv[1] == "run":
        role = _role_from_docker_run(argv, prefix_env, passes)
        return {"kind": "launch", "runtime": "docker", "role": role, "argv": list(role.get("args") or []), "env": prefix_env, "pass": passes}
    if _server_binary(argv[0]):
        # A prefix assignment is in the child environment even without export.
        return {"kind": "launch", "argv": argv, "env": prefix_env, "pass": passes}
    if passes:
        return {"kind": "pass", "name": passes[0], "names": passes}
    return None


def _docker_role_from_launch(launch: dict[str, Any], exported: dict[str, str], passed: list[str]) -> dict[str, Any]:
    role = dict(launch.get("role") or {})
    env = {key: value for key, value in exported.items() if key not in _SKIP_ENV and not _secret_name(key)}
    for key, value in (role.get("env") or {}).items():
        if key not in _SKIP_ENV and not _secret_name(str(key)):
            env[str(key)] = str(value)
    names: list[str] = []
    for name in list(passed) + list(launch.get("pass") or []) + list(role.get("env_pass") or []):
        if name and str(name) not in names:
            names.append(str(name))
    for name in names:
        env.pop(name, None)
    role["runtime"] = "docker"
    role["env"] = env
    if names:
        role["env_pass"] = names
    else:
        role.pop("env_pass", None)
    _apply_secret_split(role)
    return role


def parse_script(
    text: str,
    *,
    runner: Any = None,
    host: str = "",
    cwd: str = "",
    path: str = "",
) -> dict[str, Any] | None:
    """A start script that launches llama-server or ``docker run``. Nothing is executed.

    ``VAR=value``, ``VAR="value"``, and ``export VAR=...`` in the same script are
    resolved. ``setsid``, ``nohup``, ``exec``, and ``env`` prefixes are stripped.
    Redirections and a trailing ``&`` are not arguments. Exported variables,
    including ``LD_LIBRARY_PATH``, and assignments on the launch line are kept.
    A secret-like name used only as ``-e NAME`` or ``NAME=$NAME`` is recorded
    with no value. An ``if`` uses the script's own flag values, and a function
    body is expanded at the call. ``[ -f .env ]`` is checked on ``host``. A
    sourced env file contributes allowlisted values only. An unresolvable
    variable is ``ValueError``.
    """
    exported: dict[str, str] = {}
    passed: list[str] = []
    launch: dict[str, Any] | None = None
    extra_docker = ""
    evict_page_cache: bool | None = None
    error = ""
    for event in _walk_script(text, runner, host, cwd, path):
        kind = event.get("kind")
        if kind == "error" and not error:
            error = str(event.get("message") or "unresolved shell variable")
        elif kind == "assign" and event.get("export"):
            name = str(event.get("name") or "")
            if name and name not in _SKIP_ENV and not _secret_name(name):
                exported[name] = str(event.get("value") or "")
        elif kind == "exports":
            for name, value in (event.get("env") or {}).items():
                if name not in _SKIP_ENV and not _secret_name(name):
                    exported[str(name)] = str(value)
            for name in event.get("pass") or []:
                if str(name) not in passed:
                    passed.append(str(name))
        elif kind == "pass":
            names = list(event.get("names") or [event.get("name")])
            for name in names:
                if name and str(name) not in passed:
                    passed.append(str(name))
        elif kind == "review":
            extra_docker = str(event.get("extra_docker_args") or extra_docker)
        elif kind == "evict":
            evict_page_cache = bool(event.get("evict_page_cache"))
        elif kind == "launch" and launch is None:
            launch = event
    if error and launch is None:
        raise ValueError(error)
    if launch is None:
        if error:
            raise ValueError(error)
        return None
    if error:
        raise ValueError(error)
    env = dict(exported)
    for key, value in (launch.get("env") or {}).items():
        if key not in _SKIP_ENV and not _secret_name(str(key)):
            env[str(key)] = str(value)
    for name in launch.get("pass") or []:
        if str(name) not in passed:
            passed.append(str(name))
    for name in passed:
        env.pop(name, None)
    if str(launch.get("runtime") or "") == "docker":
        role = _docker_role_from_launch(launch, exported, passed)
    else:
        role = _native_from_argv(list(launch.get("argv") or []), env)
        if passed:
            role["env_pass"] = passed
    if extra_docker:
        role["extra_docker_args"] = extra_docker
    if evict_page_cache is not None:
        role["evict_page_cache"] = evict_page_cache
    return role


def parse_script_compose(text: str) -> dict[str, str]:
    """A script that runs compose. ``compose`` is an ``-f`` path; ``directory`` is a ``cd``."""
    directory = ""
    for event in _walk_script(text):
        if event.get("kind") == "error":
            raise ValueError(str(event.get("message") or "unresolved shell variable"))
        if event.get("kind") == "cd":
            directory = str(event.get("path") or "")
            continue
        if event.get("kind") != "compose":
            continue
        argv = [str(part) for part in (event.get("argv") or [])]
        compose = _compose_file_arg(argv)
        found: dict[str, str] = {}
        if directory:
            found["directory"] = directory
        if compose:
            found["compose"] = compose
        elif _bare_compose(argv):
            found["bare"] = "1"
        return found
    if directory:
        return {"directory": directory}
    return {}


def parse_ps(text: str) -> dict[str, Any] | None:
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if "llama-server" not in line:
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            parts = line.split()
        for index, part in enumerate(parts):
            if part == "llama-server" or part.endswith("/llama-server"):
                return _native_from_argv(_strip_redirections(parts[index:]), {})
    return None


def _port_of(args: list[str]) -> int:
    for index, part in enumerate(args):
        if part in ("--port", "-p") and index + 1 < len(args):
            try:
                return int(args[index + 1])
            except (TypeError, ValueError):
                return 0
        if part.startswith("--port="):
            try:
                return int(part.split("=", 1)[1])
            except (TypeError, ValueError):
                return 0
    return 0


def _rewrite_port(args: list[str], port: int) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        part = args[index]
        if part in ("--port", "-p") and index + 1 < len(args):
            out.extend([part, str(port)])
            index += 2
            continue
        if part.startswith("--port="):
            out.append(f"--port={port}")
            index += 1
            continue
        out.append(part)
        index += 1
    return out


def _row(catalog: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    for row in catalog.get("library") or []:
        if isinstance(row, dict) and str(row.get("id") or "") == model_id:
            return row
    return None


def _launch_start(row: dict[str, Any]) -> list[str]:
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    return [str(part) for part in (launch.get("start") or [])]


def _unit_in_start(start: list[str]) -> str:
    if "systemctl" not in start:
        return ""
    for part in reversed(start):
        if part in ("sudo", "systemctl", "start", "stop", "restart"):
            continue
        if part.startswith("-"):
            continue
        return part if part.endswith(".service") else f"{part}.service"
    return ""


def _script_in_start(start: list[str]) -> str:
    for part in start:
        if part.endswith(".sh") or part.endswith(".py"):
            return part
    return ""


def _container_in_row(row: dict[str, Any]) -> str:
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    return str(ident.get("container") or "")


def guess_source(row: dict[str, Any]) -> str:
    """Catalog hint. A named container is preferred over a start script.

    A systemd unit still wins, because its ExecStart is the real launcher.
    The live import checks that the container exists and falls back to the
    script when it does not.
    """
    start = _launch_start(row)
    unit = _unit_in_start(start)
    if unit:
        return f"unit:{unit}"
    container = _container_in_row(row)
    if container:
        return f"container:{container}"
    script = _script_in_start(start)
    if script:
        return f"script:{script}"
    return ""


def _placement(row: dict[str, Any]) -> str:
    if str(row.get("placement") or "") == "pair":
        return "both"
    host = ""
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    host = str(launch.get("host") or "")
    # The Spark whose name, id or role the row names; anything else is the head.
    return CFG.role_for(host) if host else "head"


def _remap(
    port: int,
    *,
    listening: set[int],
    reserved: set[int],
    claimed: set[int],
    own_ports: set[int] | None = None,
    holders: dict[int, str] | None = None,
) -> tuple[int, str]:
    """Keep a saved in-range port. Only an out-of-range import port moves.

    The model's own active run is not a reason to move the port. A different
    running model keeps the port and is named in the note.
    """
    own = set(own_ports or ())
    held = holders or {}
    if protect.is_off_limits(port) or (port and port in own):
        return port, ""
    if port and not check_port(port, reserved=reserved, claimed=claimed) and port not in listening and port not in held:
        return port, ""
    if port and (port < PORT_MIN or port > PORT_MAX):
        free = next_free_port(listening, reserved, claimed)
        if free:
            return free, f"imported port {port} remapped to {free}"
        return port, ""
    if port and port in held:
        return port, str(held[port])
    return port, ""


def _split_secrets(env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
    """Drop secret values. The names are filled from the node at launch."""
    kept: dict[str, str] = {}
    passed: list[str] = []
    for key, value in env.items():
        if _secret_name(key):
            if key not in passed:
                passed.append(key)
            continue
        kept[key] = value
    return kept, passed


def _apply_secret_split(role: dict[str, Any]) -> None:
    env, passed = _split_secrets(dict(role.get("env") or {}))
    role["env"] = env
    have = [str(item) for item in (role.get("env_pass") or [])]
    for name in passed:
        if name not in have:
            have.append(name)
    if have:
        role["env_pass"] = have


def _sparksi_container(model_id: str, role_name: str, placement: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "_.-" else "-" for ch in str(model_id)).strip(".-") or "model"
    if placement == "both":
        return f"sparksi-{safe}-{role_name}"
    return f"sparksi-{safe}"


def _normalize_caps(values: list[Any]) -> list[str]:
    found: list[str] = []
    for item in values:
        text = str(item or "").strip()
        if text.upper() in ("IPC_LOCK", "CAP_IPC_LOCK"):
            text = "IPC_LOCK"
        if text and text not in found:
            found.append(text)
    return found


def _port_from_env(env: dict[str, Any]) -> tuple[int, str]:
    """A server port that lives only in an environment variable."""
    for name in _PORT_ENV:
        if name not in env:
            continue
        try:
            return int(str(env.get(name) or "").strip()), name
        except (TypeError, ValueError):
            continue
    return 0, ""


def _recipe_from_roles(
    model_id: str,
    row: dict[str, Any],
    roles: dict[str, dict[str, Any]],
    source: str,
    *,
    listening: set[int],
    reserved: set[int],
    claimed: set[int],
    own_ports: set[int] | None = None,
    holders: dict[int, str] | None = None,
) -> dict[str, Any]:
    placement = _placement(row)
    if placement != "both":
        only = next(iter(roles))
        roles = {placement: roles[only]}
    sample = roles.get("head") or next(iter(roles.values()))
    args = [str(part) for part in (sample.get("args") or [])]
    image = str(sample.get("image") or "")
    binary = str(sample.get("binary") or "")
    engine = guess_engine([binary, *args], image) if sample.get("runtime") == "native" else guess_engine(args, image)
    port = _port_of(args)
    if not port:
        port, _env_name = _port_from_env(dict(sample.get("env") or {}))
    new_port, note = _remap(
        port,
        listening=listening,
        reserved=reserved,
        claimed=claimed,
        own_ports=own_ports,
        holders=holders,
    )
    rendezvous: list[int] = []
    if new_port and not protect.is_off_limits(new_port):
        from sparksi.recipes.serve_port import align_role

        for role in roles.values():
            if isinstance(role, dict):
                align_role(role, new_port, saved_port=port or None)
    from sparksi.recipes.args import rendezvous_ports

    for name, role in roles.items():
        if not isinstance(role, dict):
            continue
        if role.get("cap_add"):
            role["cap_add"] = _normalize_caps(list(role.get("cap_add") or []))
        if str(role.get("runtime") or "docker") == "docker":
            role["container"] = _sparksi_container(model_id, str(name), placement)
        for number in rendezvous_ports([str(part) for part in (role.get("args") or [])]):
            if number not in rendezvous:
                rendezvous.append(number)
    evict_flags = [
        bool(role.get("evict_page_cache"))
        for role in roles.values()
        if isinstance(role, dict) and "evict_page_cache" in role
    ]
    recipe: dict[str, Any] = {
        "kind": "pinned",
        "engine": engine,
        "placement": placement,
        "port": new_port or port,
        "imported_from": _stamp(source),
        "port_note": note,
        "rendezvous_ports": rendezvous,
        "roles": roles,
    }
    if evict_flags:
        recipe["evict_page_cache"] = any(evict_flags)
    return recipe


_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,96}/[A-Za-z0-9][A-Za-z0-9_.-]{0,96}$")
_HUB_DIR = re.compile(r"models--([A-Za-z0-9_.]+)--([A-Za-z0-9_.]+)")


def _model_id_of(row: dict[str, Any], roles: dict[str, Any]) -> str:
    """Hugging Face ``org/name`` from the catalog row or the imported argv."""
    catalog_repo = str(row.get("repo") or "").strip().strip("/")
    if _MODEL_ID.fullmatch(catalog_repo):
        return catalog_repo
    for role in roles.values():
        if not isinstance(role, dict):
            continue
        for part in role.get("args") or []:
            text = str(part)
            match = _HUB_DIR.search(text)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
            if _MODEL_ID.fullmatch(text):
                return text
    return ""


def _fill_snapshot(recipe: dict[str, Any], row: dict[str, Any], runner: Any, host: str) -> None:
    """Set repo and revision when the model id and ``refs/main`` both resolve.

    A revision already stored as 40 hex is kept. Otherwise the cache ref is
    read and kept only when it is a hash. Nothing is written when either is
    missing, so the recipe does not name the cache root.
    """
    repo = _model_id_of(row, recipe.get("roles") if isinstance(recipe.get("roles"), dict) else {})
    if not repo:
        return
    listed = str(row.get("revision") or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", listed):
        recipe["repo"] = repo
        recipe["revision"] = listed.lower()
        return
    hub = str(CFG.path("hf_hub") or "/home/USER/.cache/huggingface/hub").rstrip("/")
    folder = "models--" + repo.replace("/", "--")
    text, err = _read(runner, host, ["cat", f"{hub}/{folder}/refs/main"])
    if err or not text:
        return
    found = text.strip().split()[0] if text.strip() else ""
    if not re.fullmatch(r"[0-9a-fA-F]{40}", found):
        return
    recipe["repo"] = repo
    recipe["revision"] = found.lower()


def _read(runner: Any, host: str, argv: list[str]) -> tuple[str, str]:
    capture = getattr(runner, "capture", None)
    if capture is None:
        return "", "no read-only runner"
    try:
        result = capture(host, argv, 8)
    except Exception as exc:  # noqa: BLE001 — the dry run reports the failure
        return "", str(exc)
    if not isinstance(result, dict):
        return "", "runner returned a non-object"
    if result.get("ok") or result.get("code") == 0:
        return str(result.get("stdout") or ""), ""
    return str(result.get("stdout") or ""), str(result.get("error") or result.get("stderr") or "read failed")


def _host_for(row: dict[str, Any]) -> str:
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    return str(launch.get("host") or CFG.head.name)


def image_env(payload: Any) -> dict[str, str]:
    """``Config.Env`` from ``docker image inspect``. Identical pairs are not a launch."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if isinstance(payload, list):
        payload = payload[0] if payload else {}
    if not isinstance(payload, dict):
        return {}
    config = payload.get("Config") if isinstance(payload.get("Config"), dict) else {}
    return _env_list(config.get("Env"))


def without_image_env(role_env: dict[str, str], baked: dict[str, str]) -> dict[str, str]:
    """Drop variables whose value is already the image ENV. Overrides stay."""
    return {key: value for key, value in role_env.items() if baked.get(key) != value}


def _inspect(runner: Any, host: str, name: str, cache: dict[tuple[str, str], tuple[str, str]]) -> tuple[str, str]:
    key = (host, name)
    if key not in cache:
        cache[key] = _read(runner, host, readonly_argv("container", name))
    return cache[key]


def _container_present(text: str, err: str) -> bool:
    if err or not str(text or "").strip():
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    if isinstance(payload, list):
        return bool(payload)
    return isinstance(payload, dict) and bool(payload)


def _preferred_source(
    runner: Any,
    host: str,
    row: dict[str, Any],
    cache: dict[tuple[str, str], tuple[str, str]],
) -> str:
    """Prefer a container that exists. Fall back to the script only when it does not."""
    start = _launch_start(row)
    unit = _unit_in_start(start)
    if unit:
        return f"unit:{unit}"
    container = _container_in_row(row)
    script = _script_in_start(start)
    if container and script:
        text, err = _inspect(runner, host, container, cache)
        if _container_present(text, err):
            return f"container:{container}"
        return f"script:{script}"
    if container:
        return f"container:{container}"
    if script:
        return f"script:{script}"
    return ""


def _drop_image_env(runner: Any, host: str, role: dict[str, Any], cache: dict[Any, Any]) -> None:
    image = str(role.get("image") or "")
    if not image or role.get("runtime") == "native":
        return
    key = ("image", host, image)
    if key not in cache:
        cache[key] = _read(runner, host, readonly_argv("image", image))
    text, err = cache[key]
    if err or not text:
        return
    try:
        baked = image_env(text)
    except (json.JSONDecodeError, ValueError):
        return
    role["env"] = without_image_env(dict(role.get("env") or {}), baked)


def _compose_in_directory(runner: Any, host: str, directory: str) -> str:
    directory = str(directory or "").rstrip("/")
    if not directory.startswith("/"):
        return ""
    for name in _COMPOSE_NAMES:
        path = f"{directory}/{name}"
        _text, err = _read(runner, host, readonly_argv("exists", path))
        if not err:
            return path
    return ""


def _load_roles(
    runner: Any,
    host: str,
    source: str,
    row: dict[str, Any],
    cache: dict[Any, Any] | None = None,
) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    """Return roles, the source note, and problems. Reads only."""
    problems: list[str] = []
    cache = cache if cache is not None else {}
    kind, value = parse_source(source)
    if kind == "container" and not value:
        text, err = _read(runner, host, readonly_argv("containers", ""))
        if err:
            problems.append(err)
        value = _match_container(text, row)
        if not value:
            problems.append("no matching container")
            return {}, source, problems
        source = f"container:{value}"
        kind = "container"
    if kind == "unit":
        text, err = _read(runner, host, readonly_argv("unit", value))
        if err and not text:
            problems.append(err)
            return {}, source, problems
        pointed = parse_unit(text)
        if pointed.get("compose"):
            return _load_roles(runner, host, f"compose:{pointed['compose']}", row, cache)
        if pointed.get("bare_compose"):
            found = _compose_in_directory(runner, host, pointed.get("working_directory") or "")
            if not found:
                problems.append("unit runs docker compose but no compose file was found")
                return {}, source, problems
            return _load_roles(runner, host, f"compose:{found}", row, cache)
        if pointed.get("script"):
            return _load_unit_script(runner, host, pointed, row, cache, source, problems)
        problems.append("unit does not name a compose file or a script")
        return {}, source, problems
    if kind == "compose":
        text, err = _read(runner, host, readonly_argv("compose", value))
        if err and not text:
            problems.append(err)
            return {}, source, problems
        try:
            roles = parse_compose_config(text)
        except (json.JSONDecodeError, ValueError) as exc:
            problems.append(str(exc))
            return {}, source, problems
        return roles, source, problems
    if kind == "script":
        return _load_script(runner, host, value, source, problems)
    if kind == "container":
        return _load_container(runner, host, value, source, row, cache, problems)
    problems.append("unknown import source")
    return {}, source, problems


def _load_unit_script(
    runner: Any,
    host: str,
    pointed: dict[str, str],
    row: dict[str, Any],
    cache: dict[Any, Any],
    source: str,
    problems: list[str],
) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    script = pointed.get("script") or ""
    text, err = _read(runner, host, readonly_argv("script", script))
    if err and not text:
        problems.append(err)
        return {}, source, problems
    try:
        hint = parse_script_compose(text) if text else {}
    except ValueError as exc:
        problems.append(str(exc))
        return {}, source, problems
    compose = hint.get("compose") or ""
    if compose and not compose.startswith("/"):
        base = pointed.get("working_directory") or hint.get("directory") or os.path.dirname(script)
        compose = os.path.normpath(os.path.join(base, compose)) if base else compose
    if compose:
        return _load_roles(runner, host, f"compose:{compose}", row, cache)
    if hint.get("bare"):
        directory = pointed.get("working_directory") or hint.get("directory") or os.path.dirname(script)
        found = _compose_in_directory(runner, host, directory)
        if not found:
            problems.append("unit script runs docker compose but no compose file was found")
            return {}, source, problems
        return _load_roles(runner, host, f"compose:{found}", row, cache)
    return _load_script(runner, host, script, source, problems, text=text)


def _load_script(
    runner: Any,
    host: str,
    path: str,
    source: str,
    problems: list[str],
    text: str | None = None,
) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    if text is None:
        text, err = _read(runner, host, readonly_argv("script", path))
        if err and not text:
            problems.append(err)
    try:
        role = parse_script(
            text,
            runner=runner,
            host=host,
            cwd=os.path.dirname(path) if path else "",
            path=path,
        ) if text else None
    except ValueError as exc:
        problems.append(str(exc))
        return {}, source, problems
    if role is None:
        ps, ps_err = _read(runner, host, readonly_argv("ps", ""))
        if ps_err and not ps:
            problems.append(ps_err or "script has no llama-server command")
        role = parse_ps(ps)
    if role is None:
        problems.append("script has no llama-server command")
        return {}, source, problems
    return {"head": role}, source, problems


def _load_container(
    runner: Any,
    host: str,
    value: str,
    source: str,
    row: dict[str, Any],
    cache: dict[Any, Any],
    problems: list[str],
) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
    text, err = _inspect(runner, host, value, cache)
    if err and not text:
        problems.append(err)
        return {}, source, problems
    try:
        head = parse_docker_inspect(text)
    except (json.JSONDecodeError, ValueError) as exc:
        problems.append(str(exc))
        return {}, source, problems
    _drop_image_env(runner, host, head, cache)
    roles = {"head": head}
    if str(row.get("placement") or "") != "pair":
        return roles, source, problems
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    worker_name = str(ident.get("worker_container") or value)
    if not CFG.has_worker:
        problems.append("this pair needs a worker Spark; this site has 1 Spark")
        return roles, source, problems
    worker_host = CFG.worker.name
    worker_text, worker_err = _inspect(runner, worker_host, worker_name, cache)
    if worker_err or not str(worker_text or "").strip():
        problems.append(worker_err or f"worker container {worker_name} was not found on the worker")
        return roles, source, problems
    try:
        worker = parse_docker_inspect(worker_text)
    except (json.JSONDecodeError, ValueError) as exc:
        problems.append(str(exc))
        return roles, source, problems
    _drop_image_env(runner, worker_host, worker, cache)
    roles["worker"] = worker
    return roles, source, problems


def _match_container(text: str, row: dict[str, Any]) -> str:
    ident = row.get("identity") if isinstance(row.get("identity"), dict) else {}
    wanted = str(ident.get("container") or "")
    names = []
    for line in str(text or "").splitlines():
        name = line.split("\t", 1)[0].strip()
        if name:
            names.append(name)
    if wanted and wanted in names:
        return wanted
    token = str(row.get("id") or "")
    for name in names:
        if token and token in name:
            return name
    return names[0] if names else wanted


def import_model(
    catalog: dict[str, Any],
    runner: Any,
    model_id: str,
    source: str = "",
    *,
    dry_run: bool = False,
    listening: set[int] | None = None,
    reserved: set[int] | None = None,
    claimed: set[int] | None = None,
) -> dict[str, Any]:
    row = _row(catalog, model_id)
    if row is None:
        return {"ok": False, "id": model_id, "conflicts": ["unknown model"], "recipe": None, "saved": False}
    from sparksi.recipes.service import _live_port_holders, _own_ports, _reserved_ports

    own = _own_ports(model_id)
    holders = _live_port_holders(model_id, "both", catalog)
    if reserved is None:
        reserved = _reserved_ports(catalog)
    if claimed is None:
        claimed = set(holders)
    host = _host_for(row)
    cache: dict[Any, Any] = {}
    if not source:
        source = _preferred_source(runner, host, row, cache)
    if not source:
        return {"ok": False, "id": model_id, "conflicts": ["no import source"], "recipe": None, "saved": False}
    roles, note, problems = _load_roles(runner, host, source, row, cache)
    if not roles:
        return {"ok": False, "id": model_id, "conflicts": problems or ["nothing to import"], "recipe": None, "saved": False}
    recipe = _recipe_from_roles(
        model_id,
        row,
        roles,
        note,
        listening=set(listening or ()) - own,
        reserved=set(reserved or ()),
        claimed=set(claimed or ()),
        own_ports=own,
        holders=holders,
    )
    _fill_snapshot(recipe, row, runner, host)
    card = card_from_row(row)
    conflicts = list(problems)
    conflicts.extend(apply_token_cache(
        recipe,
        runner,
        repo=str(recipe.get("repo") or row.get("repo") or ""),
        revision=str(recipe.get("revision") or row.get("revision") or ""),
    ))
    conflicts.extend(validate_pinned(recipe, ctx_max=card.max_ctx, runner=runner))
    saved = False
    if not dry_run and not conflicts:
        try:
            save_recipe(model_id, recipe)
        except ValueError as exc:
            conflicts.append(str(exc))
        else:
            saved = True
    return {
        "ok": not conflicts,
        "id": model_id,
        "conflicts": conflicts,
        "recipe": recipe,
        "saved": saved,
        "imported_from": recipe.get("imported_from") or "",
        "port_note": recipe.get("port_note") or "",
    }


def _service_row(catalog: dict[str, Any], row: dict[str, Any]) -> bool:
    """Services, service_commands, and never_touch rows are not model imports."""
    from sparksi.models import row_is_protected, row_is_service

    if row_is_service(row) or row_is_protected(row, catalog):
        return True
    commands = {tuple(item) for item in CFG.service_commands()}
    launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
    for key in ("start", "stop"):
        argv = tuple(str(part) for part in (launch.get(key) or []))
        if argv and argv in commands:
            return True
    return False


def import_all(catalog: dict[str, Any], runner: Any, *, dry_run: bool = True) -> list[dict[str, Any]]:
    """Dry-run every model row that has a launcher. Services are skipped."""
    out = []
    for row in catalog.get("library") or []:
        if not isinstance(row, dict) or not row.get("id"):
            continue
        if _service_row(catalog, row):
            continue
        launch = row.get("launch") if isinstance(row.get("launch"), dict) else {}
        if not launch.get("start") and not (row.get("identity") or {}).get("container"):
            continue
        out.append(import_model(catalog, runner, str(row["id"]), dry_run=dry_run))
    return out


def saved_copy(model_id: str) -> dict[str, Any]:
    return load_recipe(model_id)
