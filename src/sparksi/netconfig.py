# SPDX-License-Identifier: Apache-2.0

"""The Network page's config edits: roles for machines, written to sparksi.toml.

``change`` decides the new ``[[nodes]]`` (head, worker) and ``[[endpoints]]``
(monitored inference hosts) for one Add, Remove or role change. ``save_tables``
rewrites only those tables, keeps every other line as it was, checks the result
parses to the same settings, keeps a timestamped backup and replaces the file
atomically. Nothing here starts, stops or contacts a machine.
"""

from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import shutil
import tempfile
import tomllib
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from sparksi.config import role_name
from sparksi.hostcheck import check_host, check_name, check_port, check_ssh, http_url

_HEADER_RE = re.compile(
    r"^\s*\[\[?\s*"
    r"([A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*)"
    r"\s*\]\]?\s*(?:#.*)?$"
)

_SCALARS = (str, bool, int, float)


def toml_value(v: Any) -> str:
    """Render a scalar or list-of-scalars as a TOML value."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        # ensure_ascii=False keeps emoji as themselves (TOML rejects \ud83d-style surrogate pairs);
        # json still escapes quotes, backslashes and U+0000-U+001F, and TOML also forbids a raw DEL.
        return json.dumps(v, ensure_ascii=False).replace("\x7f", "\\u007f")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(item) for item in v) + "]"
    raise TypeError(f"unsupported TOML value type: {type(v).__name__}")


def _assign(key: str, value: Any) -> str:
    """``key = value``; a list too long for one line goes one item per line, as people write it."""
    line = f"{key} = {toml_value(value)}"
    if isinstance(value, list) and value and len(line) > 88:
        return f"{key} = [\n" + "".join(f"  {toml_value(item)},\n" for item in value) + "]"
    return line


def render_tables(nodes: list[dict], endpoints: list[dict]) -> str:
    """``[[nodes]]`` (with ``[nodes.<key>]`` sub-tables) then ``[[endpoints]]``, one blank line apart."""
    blocks: list[list[str]] = []
    for name, rows in (("nodes", nodes), ("endpoints", endpoints)):
        for row in rows:
            blocks.append([f"[[{name}]]"] + [_assign(k, v) for k, v in row.items() if not isinstance(v, dict)])
            for key, sub in row.items():
                if isinstance(sub, dict):
                    if name != "nodes":
                        raise TypeError("only nodes have sub-tables")
                    blocks.append([f"[{name}.{key}]"] + [_assign(k, v) for k, v in sub.items()])
    return "\n\n".join("\n".join(block) for block in blocks) + "\n" if blocks else ""


def _bracket_depth(line: str, depth: int) -> int:
    """Track ``[``/``]`` depth outside quoted strings and comments."""
    in_basic = in_literal = escaped = False
    for ch in line:
        if in_basic:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_basic = False
            continue
        if in_literal:
            if ch == "'":
                in_literal = False
            continue
        if ch == "#":
            break
        if ch == '"':
            in_basic = True
        elif ch == "'":
            in_literal = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
    return max(depth, 0)


def _sections(lines: list[str]) -> list[tuple[str, int, int]]:
    """(name, start, end) for each table; end is exclusive. Keys before the first table are not a section."""
    found: list[tuple[str, int]] = []
    depth = 0
    for i, line in enumerate(lines):
        match = _HEADER_RE.match(line) if depth == 0 else None
        if match:
            found.append((match.group(1), i))
            continue
        depth = _bracket_depth(line, depth)
    return [(name, start, found[k + 1][1] if k + 1 < len(found) else len(lines)) for k, (name, start) in enumerate(found)]


def _is_target(name: str) -> bool:
    return name in ("nodes", "endpoints") or name.startswith(("nodes.", "endpoints."))


def _key(kind: str, row: dict) -> str:
    return kind + json.dumps(row, sort_keys=True)


def _original_blocks(lines: list[str], sections: list[tuple[str, int, int]]) -> dict[str, str]:
    """Each existing ``[[nodes]]``/``[[endpoints]]`` entry's own text (comments included), by its parsed value."""
    entries: list[tuple[str, list[str]]] = []
    for name, start, end in sections:
        if name in ("nodes", "endpoints") and lines[start].lstrip().startswith("[["):
            entries.append((name, list(lines[start:end])))
        elif _is_target(name) and entries and name.startswith(entries[-1][0] + "."):
            entries[-1][1].extend(lines[start:end])
    found: dict[str, str] = {}
    for kind, body in entries:
        while body and (not body[-1].strip() or body[-1].lstrip().startswith("#")):
            body.pop()
        block = "".join(body)
        if not block.endswith("\n"):
            block += "\n"
        try:
            rows = tomllib.loads(block).get(kind) or []
        except tomllib.TOMLDecodeError:
            continue
        if len(rows) == 1:
            found.setdefault(_key(kind, rows[0]), block)
    return found


def replace_tables(text: str, nodes: list[dict], endpoints: list[dict]) -> str:
    """Rewrite only the nodes and endpoints tables of ``text``.

    An entry that is unchanged (say, a head and worker that swap) keeps its own
    text, comments included; only new or changed entries are rendered afresh.
    """
    lines = text.splitlines(keepends=True)
    sections = _sections(lines)
    targets: list[tuple[int, int]] = []
    for k, (name, s, e) in enumerate(sections):
        if not _is_target(name):
            continue
        following = sections[k + 1][0] if k + 1 < len(sections) else ""
        if not _is_target(following):
            # Comments and blank lines just above the next table belong to it.
            while e - 1 > s and (not lines[e - 1].strip() or lines[e - 1].lstrip().startswith("#")):
                e -= 1
        targets.append((s, e))
    originals = _original_blocks(lines, sections)
    blocks = []
    for kind, rows in (("nodes", nodes), ("endpoints", endpoints)):
        for row in rows:
            blocks.append(originals.get(_key(kind, row)) or render_tables(*(([row], []) if kind == "nodes" else ([], [row]))))
    rendered = "\n".join(blocks).splitlines(keepends=True)
    if not targets:
        out = list(lines)
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        if out and out[-1].strip():
            out.append("\n")
        return "".join(out + rendered)
    dropped = {i for (s, e) in targets for i in range(s, e)}
    first = targets[0][0]

    def first_kept_after(i: int) -> bool:
        """True when the next kept line is not blank, so a separator is needed."""
        nxt = next((j for j in range(i, len(lines)) if j not in dropped), None)
        return nxt is not None and bool(lines[nxt].strip())

    out: list[str] = []
    for i, line in enumerate(lines):
        if i == first:
            out.extend(rendered)
            if first_kept_after(i):
                out.append("\n")
        if i not in dropped:
            out.append(line)
    return "".join(out)


def _without_tables(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in ("nodes", "endpoints")}


def check_same_elsewhere(
    old_text: str,
    new_text: str,
    nodes: list[dict],
    endpoints: list[dict],
) -> None:
    """Verify the rewrite only touched nodes/endpoints."""
    old = tomllib.loads(old_text)
    new = tomllib.loads(new_text)
    if _without_tables(old) != _without_tables(new):
        raise ValueError("the edit would change other settings")
    if new.get("nodes") != nodes or new.get("endpoints", []) != endpoints:
        raise ValueError("the edit did not produce the requested machines")


def _write_backup(path: Path, data: bytes, now: datetime) -> Path:
    """``<name>.bak-YYYYmmdd-HHMMSS-ffffff`` (plus ``-N`` if taken); created exclusively, never overwritten."""
    stamp = now.strftime("%Y%m%d-%H%M%S-%f")
    mode = path.stat().st_mode & 0o777
    for n in range(1000):
        target = path.with_name(f"{path.name}.bak-{stamp}" + (f"-{n}" if n else ""))
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        except FileExistsError:
            continue
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        shutil.copystat(path, target)
        return target
    raise ValueError("could not pick a free backup name")


def write_text_atomic(path: Path, new_text: str, *, now: datetime | None = None) -> str:
    """Keep a unique backup of ``path``, then replace it atomically with ``new_text``. Returns the backup name."""
    path = Path(path)
    backup_path = _write_backup(path, path.read_bytes(), now or datetime.now())
    backup_name = backup_path.name

    mode = None
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        mode = None

    fd = -1
    tmp_name: str | None = None
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
        if mode is not None:
            os.chmod(tmp_name, mode)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            fd = -1
            f.write(new_text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        tmp_name = None
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp_name is not None:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
        raise

    return backup_name


def save_tables(
    path: Path,
    nodes: list[dict],
    endpoints: list[dict],
    *,
    now: datetime | None = None,
) -> dict:
    """Atomically rewrite nodes/endpoints in the file at ``path``."""
    path = Path(path)
    old_text = path.read_text(encoding="utf-8")
    current = tomllib.loads(old_text)
    if current.get("nodes") == nodes and current.get("endpoints", []) == endpoints:
        return {"changed": False, "backup": None}  # same machines: leave the file (and its comments) alone
    new_text = replace_tables(old_text, nodes, endpoints)
    check_same_elsewhere(old_text, new_text, nodes, endpoints)

    if new_text == old_text:
        return {"changed": False, "backup": None}

    return {"changed": True, "backup": write_text_atomic(path, new_text, now=now)}


MAX_MACHINES = 4  # Sparks ([[nodes]]): head, worker, node3, node4
MAX_MONITORED = 2  # monitored inference hosts ([[endpoints]]) beside them


def slug(text: str) -> str:
    """Lower-case slug: non-alnum runs become '-', trimmed, max 40 chars."""
    s = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return s[:40] or "machine"


def _node_summary(node: dict) -> dict:
    return {
        "id": node.get("id", ""),
        "name": node.get("name", ""),
        "tailscale": node.get("tailscale", ""),
    }


def _endpoint_summary(endpoint: dict) -> dict:
    return {
        "id": endpoint.get("id", ""),
        "name": endpoint.get("name", ""),
        "url": endpoint.get("url", ""),
        "engine": endpoint.get("engine", ""),
    }


def layout(raw: dict, controller_name: str) -> dict:
    """Summarise the site layout for the Network page."""
    nodes = raw["nodes"]
    endpoints = raw.get("endpoints", [])
    out = {"controller": {"name": controller_name}, "head": _node_summary(nodes[0])}
    if len(nodes) >= 2:
        out["worker"] = _node_summary(nodes[1])
    out.update({
        "monitored": [_endpoint_summary(e) for e in endpoints],
        "limit": MAX_MACHINES,
        "count": len(nodes) + len(endpoints),
        "nodes": [{**_node_summary(node), "role": role_name(i)} for i, node in enumerate(nodes)],
    })
    return out


def _host_of(url: str) -> str:
    return urlsplit(str(url)).hostname or ""


def _ref_values(ref: dict) -> list[str]:
    values: list[str] = []
    for key in ("id", "name", "dns", "ips"):
        value = ref.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(v) for v in value)
        else:
            values.append(str(value))
        if key == "dns":
            values.append(str(value).split(".", 1)[0])
    return values


def _machine_values(node_or_endpoint: dict, endpoint: bool) -> list[str]:
    values = [
        str(node_or_endpoint.get("id", "")),
        str(node_or_endpoint.get("name", "")),
    ]
    if endpoint:
        values.append(_host_of(node_or_endpoint.get("url", "")))
    else:
        values.append(str(node_or_endpoint.get("ssh", "")))
        values.append(str(node_or_endpoint.get("tailscale", "")))
    return [v for v in values if v]


def find(raw: dict, ref: dict) -> tuple[str, int] | None:
    """Locate a configured machine by id, name, dns or ip."""
    wanted = [v.casefold() for v in _ref_values(ref) if v]
    if not wanted:
        return None
    for i, node in enumerate(raw["nodes"]):
        have = {v.casefold() for v in _machine_values(node, endpoint=False)}
        if have & set(wanted):
            return ("node", i)
    for i, endpoint in enumerate(raw.get("endpoints", [])):
        have = {v.casefold() for v in _machine_values(endpoint, endpoint=True)}
        if have & set(wanted):
            return ("endpoint", i)
    return None


def _first_ts_ip(ips: list) -> str:
    for ip in ips:
        try:
            addr = ipaddress.ip_address(str(ip).split("%", 1)[0])
        except ValueError:
            continue
        if addr in ipaddress.ip_network("100.64.0.0/10"):
            return str(ip)
    return ""


def machine_label(machine: dict) -> str:
    """Display name of a discovered or typed machine (checked: length cap, no control characters)."""
    for key in ("name", "address", "dns"):
        value = str(machine.get(key) or "").strip()
        if value:
            return check_name(value)
    for ip in machine.get("ips", []) or []:
        value = str(ip).strip()
        if value:
            return check_name(value)
    return "machine"


def check_address(text: object) -> str:
    """A typed address (host name, or private/tailnet IP), trimmed and checked."""
    return check_host(str(text if text is not None else "").strip())


# Engines a monitored host may be saved with ("" = not known yet). Anything else
# Discover reports ("open port", "HTTP service", "needs a key", ...) is not an inference API.
ENGINES = ("", "vLLM", "SGLang", "llama.cpp", "OpenAI-compatible", "Ollama")


def _real_endpoints(machine: dict) -> list[dict]:
    return [ep for ep in machine.get("endpoints", []) or [] if ep.get("engine") and ep.get("engine") in ENGINES]


def _rebuilt(ep: dict) -> tuple[str, str]:
    """A discovered endpoint's (url, engine), rebuilt from its checked host and port; http only."""
    parts = urlsplit(str(ep.get("url", "")))
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError("only http:// inference endpoints can be monitored")
    port = check_port(ep.get("port") if ep.get("port") is not None else parts.port)
    if parts.port is not None and parts.port != port:
        raise ValueError("the endpoint port does not match its address")
    return http_url(parts.hostname, port), str(ep.get("engine") or "")


def endpoint_from(machine: dict) -> tuple[dict, bool]:
    """Build a monitored endpoint; returns (endpoint, answering).

    The URL is always rebuilt here as http://<checked host>:<checked port>; a
    URL sent by the page is never stored as given.
    """
    label = machine_label(machine)
    if "port" in machine:
        port = check_port(machine["port"])
        for ep in _real_endpoints(machine):
            if ep.get("port") == port or urlsplit(str(ep.get("url", ""))).port == port:
                url, engine = _rebuilt(ep)
                return {"id": slug(label), "name": label, "url": url, "engine": engine}, True
        host = check_address(
            machine.get("address")
            or machine.get("dns")
            or _first_ts_ip(machine.get("ips", []))
            or (machine.get("ips") or [""])[0]
            or machine.get("name", "")
        )
        return {"id": slug(label), "name": label, "url": http_url(host, port), "engine": ""}, False
    real = _real_endpoints(machine)
    if not real:
        raise ValueError(
            "nothing answers on this machine right now; choose the port its inference API will use"
        )
    url, engine = _rebuilt(real[0])
    return {"id": slug(label), "name": label, "url": url, "engine": engine}, True


def node_from(machine: dict) -> dict:
    """Build a Spark node dict from a discovered machine. The ssh destination is checked (no option injection)."""
    name = machine_label(machine)
    node = {
        "id": slug(name),
        "name": name,
        "ssh": check_ssh(str(machine.get("dns") or machine.get("address") or name).strip()),
    }
    ts = _first_ts_ip(machine.get("ips", []))
    if ts:
        node["tailscale"] = ts
    return node


def _role_text(role: str) -> str:
    """Plain words for a Spark role: "the worker", "Spark 3"."""
    return f"Spark {role[4:]}" if role.startswith("node") else f"the {role}"


def change(raw: dict, action: str, role: str, machine: dict) -> tuple[list[dict], list[dict], str]:
    """Apply an add / role / remove action; returns (nodes, endpoints, note)."""
    if action not in ("add", "role", "remove"):
        raise ValueError("unknown action")
    if action in ("role", "add") and role not in tuple(role_name(i) for i in range(MAX_MACHINES)) + ("monitor",):
        raise ValueError("unknown role")

    new_raw = copy.deepcopy(raw)
    nodes = new_raw["nodes"]
    endpoints = new_raw.setdefault("endpoints", [])

    found = find(raw, machine)
    if action == "add" and found is not None:
        kind, index = found
        have = (nodes if kind == "node" else endpoints)[index]
        named = have.get("name") or have.get("id") or machine_label(machine)
        if "address" in machine:
            raise ValueError(f"a machine named {named} is already in the setup; pick another name")
        raise ValueError(f"a machine named {named} is already in the setup; change its role under This setup")

    if action == "remove":
        if found is None:
            raise ValueError("not in the config")
        kind, index = found
        if kind == "node":
            if index == 0:
                raise ValueError(
                    "the head runs Sparksi; give the head role to another Spark first"
                )
            if index != len(nodes) - 1:
                # Roles follow node order. Taking out a middle Spark would renumber the
                # ones after it, and recipes saved for them would land on another machine.
                last = role_name(len(nodes) - 1)
                raise ValueError(
                    f"{_role_text(role_name(index))} is not the last Spark; remove {_role_text(last)} first,"
                    f" or swap roles so this Spark is {_role_text(last)}"
                )
            gone = nodes.pop(index)
            return nodes, endpoints, f"removed {gone.get('name') or gone.get('id')}; {len(nodes)} Spark{'s' if len(nodes) != 1 else ''} left"
        gone = endpoints.pop(index)
        return nodes, endpoints, f"removed {gone.get('name') or gone.get('id')}"

    kind, index = found if found is not None else (None, -1)

    if role != "monitor":
        slot = [role_name(i) for i in range(MAX_MACHINES)].index(role)
        if slot > len(nodes):
            raise ValueError(f"add the {role_name(len(nodes))} first")
        if kind == "node":
            if index == slot:
                return nodes, endpoints, "no change"
            if index < len(nodes):
                if slot < len(nodes):
                    nodes[slot], nodes[index] = nodes[index], nodes[slot]
                    pair = {slot, index}
                    if pair == {0, 1}:
                        return nodes, endpoints, "head and worker swapped"
                    return nodes, endpoints, f"{role_name(slot)} and {role_name(index)} swapped"
                raise ValueError("that Spark already has a role")
        if kind == "endpoint":
            old = endpoints.pop(index)
            new_node = node_from(
                {
                    "name": old["name"],
                    "dns": _host_of(old["url"]),
                    "ips": [_host_of(old["url"])],
                }
            )
        else:
            new_node = node_from(machine)
        for j, node in enumerate(nodes):
            if j != slot and node["id"] == new_node["id"]:
                raise ValueError(f'"{new_node["name"]}" conflicts with "{node["name"]}"')
        if slot == len(nodes):
            if len(nodes) >= MAX_MACHINES:
                raise ValueError(f"Sparksi manages up to {MAX_MACHINES} Sparks")
            nodes.append(new_node)
            return nodes, endpoints, f"added {new_node['name']} as {role}"
        old_name = nodes[slot]["name"]
        nodes[slot] = new_node
        return nodes, endpoints, f"replaces {old_name} as {role}"

    if kind == "node":
        raise ValueError("that machine is a Spark here; remove it as a Spark first")
    if kind == "endpoint":
        return nodes, endpoints, "no change"
    new_endpoint, answering = endpoint_from(machine)  # bad input is refused before the limit
    if len(endpoints) + 1 > MAX_MONITORED:
        raise ValueError(
            f"Sparksi holds up to {MAX_MONITORED} monitored hosts beside its Sparks; remove a monitored host first"
        )
    if any(e["id"] == new_endpoint["id"] for e in endpoints):
        raise ValueError(f"a machine named {new_endpoint['name']} is already in the setup; pick another name")
    endpoints.append(new_endpoint)
    note = f"now monitoring {new_endpoint['name']} at {new_endpoint['url']}"
    if not answering:
        note += "; nothing answers on this port right now, so it shows as offline until it does"
    return nodes, endpoints, note
