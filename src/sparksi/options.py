# SPDX-License-Identifier: Apache-2.0

"""The Options page: a few optional settings in sparksi.toml, one key at a time.

Machines and roles live on the Network page. Each save here changes exactly one
``[table] key`` line, keeps every other byte of the file, checks that the file
still parses to the same settings apart from that key, keeps a unique backup and
replaces the file atomically. Options never take the model lock.
"""

from __future__ import annotations

import copy
import posixpath
import re
import tomllib
import unicodedata
from pathlib import Path
from typing import Any

from sparksi import compat
from sparksi.config import DEFAULTS, _merge
from sparksi.hostcheck import check_name
from sparksi.monitor_mirror import refuse_self, upstream_url
from sparksi.netconfig import _bracket_depth, _sections, toml_value, write_text_atomic

OPTIONS: tuple[dict[str, Any], ...] = (
    {
        "key": "archive.root",
        "label": "Storage path",
        "about": "Folder on the storage machine where Archive keeps models and copies them to and from the Sparks.",
        "kind": "path",
        "default": "/mnt/archive",
    },
    {
        "key": "archive.label",
        "label": "Storage name",
        "about": "Name shown for the storage on the Archive page.",
        "kind": "text",
        "default": "Archive",
    },
    {
        "key": "archive.node",
        "label": "Storage machine",
        "about": "The Spark the storage is attached to. Empty means the head.",
        "kind": "node",
        "default": "",
    },
    {
        "key": "controller.monitor_upstream",
        "label": "Monitor mirror",
        "about": "Read the live monitor from another Sparksi (http://host:port on loopback or the tailnet) instead of listening for announcements. Empty turns it off.",
        "kind": "url",
        "default": "",
        "env": "SPARKSI_MONITOR_UPSTREAM",
    },
    {
        "key": "controller.access_log",
        "label": "Request log",
        "about": "Which requests go into the service log: slow ones (over 1 s), all, or none.",
        "kind": "choice",
        "choices": ("slow", "all", "off"),
        "default": "slow",
        "env": "SPARKSI_ACCESS_LOG",
    },
    {
        "key": "controller.metrics_url",
        "label": "Metrics database",
        "about": "VictoriaMetrics address for the token charts (http://host:port on loopback or the tailnet). Empty uses http://127.0.0.1:8428.",
        "kind": "url",
        "default": "",
    },
)

_BY_KEY = {o["key"]: o for o in OPTIONS}
MAX_PATH = 4096

# Never an archive: system trees, and anything one level under /.
_SYSTEM = frozenset(
    "bin boot dev etc lib lib32 lib64 libx32 proc root run sbin snap sys tmp usr var".split()
)


def _split(key: str) -> tuple[str, str]:
    table, _, name = key.partition(".")
    return table, name


def file_value(raw: dict[str, Any], key: str) -> Any:
    """The value in the file, or None when the key is not set."""
    table, name = _split(key)
    section = raw.get(table)
    return section.get(name) if isinstance(section, dict) else None


def _default(opt: dict[str, Any]) -> Any:
    """What Sparksi uses when the file leaves the key out (the built-in defaults)."""
    table, name = _split(opt["key"])
    section = DEFAULTS.get(table)
    if isinstance(section, dict) and name in section:
        return section[name]
    return opt["default"]


def _node_ids(raw: dict[str, Any]) -> list[str]:
    return [str(n.get("id")) for n in raw.get("nodes") or [] if isinstance(n, dict) and n.get("id")]


def _shown(kind: str, value: Any) -> str:
    """What the page shows: never a password or token inside a URL, in any form. Never raises."""
    text = "" if value is None else str(value)
    if kind != "url" or "@" not in text:
        return text
    try:
        scheme = text.find("://")
        if scheme != -1:
            start = scheme + 3
        elif text.startswith("//"):
            start = 2  # scheme-relative: //user:pass@host:port
        else:
            start = 0
        end = len(text)
        for ch in "/?#":
            pos = text.find(ch, start)
            if pos != -1:
                end = min(end, pos)
        at = text.rfind("@", start, end)
        if at == -1:
            return text
        return text[:start] + "***@" + text[at + 1:]
    except Exception:  # noqa: BLE001 - showing a value must not fail the page
        return "***"


def current(raw: dict[str, Any], running: dict[str, Any] | None = None, environ: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Each option as the page shows it. ``running`` is the config this process loaded."""
    environ = environ or {}
    rows = []
    for opt in OPTIONS:
        have = file_value(raw, opt["key"])
        value = _default(opt) if have is None else have
        row: dict[str, Any] = {
            "key": opt["key"],
            "label": opt["label"],
            "about": opt["about"],
            "kind": opt["kind"],
            "value": _shown(opt["kind"], value),
            "default": _shown(opt["kind"], _default(opt)),
            "restart": True,
        }
        if opt["kind"] == "choice":
            row["choices"] = list(opt["choices"])
        elif opt["kind"] == "node":
            row["choices"] = _node_ids(raw)
        if running is not None:
            used = file_value(running, opt["key"])
            row["restart_needed"] = (_default(opt) if used is None else used) != value
        if opt.get("env") and environ.get(opt["env"]):
            row["override"] = opt["env"] + " is set for this process and wins over the file."
        rows.append(row)
    return rows


def _bad_chars(text: str) -> bool:
    return any(unicodedata.category(ch) in ("Cc", "Cf", "Zl", "Zp") for ch in text)


def _inside(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def check_archive_root(value: str, raw: dict[str, Any]) -> str:
    """An absolute, normal folder that is not a system tree and does not overlap a protected path."""
    if not value or len(value) > MAX_PATH or _bad_chars(value) or "\\" in value:
        raise ValueError("the storage path must be an absolute folder such as /mnt/storage")
    if not value.startswith("/") or posixpath.normpath(value) != value or value.startswith("//"):
        raise ValueError("the storage path must be an absolute folder such as /mnt/storage, with no '..', '.' or doubled or trailing '/'")
    parts = value.strip("/").split("/")
    if len(parts) < 2 or parts[0] in _SYSTEM:
        raise ValueError("the storage path must be a folder below a mount point such as /mnt or /media, not a system folder")
    # The protected paths Sparksi really uses: the file merged with the built-in defaults,
    # exactly as Config does, so a file without [paths] or extra_blocked is still covered.
    merged = _merge(DEFAULTS, compat.apply_legacy_paths(raw))
    paths = merged.get("paths") if isinstance(merged.get("paths"), dict) else {}
    archive = merged.get("archive") if isinstance(merged.get("archive"), dict) else {}
    # Never the same as, or a parent of, these; never inside the ones that must stay untouched.
    near: list[str] = [str(paths.get(k) or "") for k in ("home", "hf_hub", "video_service_root")]
    near += [str(p) for p in paths.get("too_broad") or []]
    untouched: list[str] = [str(paths.get("hf_hub") or ""), str(paths.get("video_service_root") or "")]
    untouched += [str(p) for p in paths.get("blocked_prefixes") or []]
    untouched += [str(p) for p in archive.get("extra_blocked") or []]
    for node in merged.get("nodes") or []:
        if isinstance(node, dict):
            untouched += [str(p) for p in node.get("mount_roots") or []]
    for other in near + untouched:
        other = other.rstrip("/")
        if other and _inside(other, value):
            raise ValueError(f"the storage path must not be {other} or a folder that contains it")
    for other in untouched:
        other = other.rstrip("/")
        if other and _inside(value, other):
            raise ValueError(f"the storage path must not be inside {other}")
    return value


def check(
    key: str,
    value: object,
    raw: dict[str, Any],
    own_ports: tuple[int, ...] = (),
    own_names: tuple[str, ...] = (),
) -> str:
    """The cleaned value to store, or ValueError with a plain reason."""
    opt = _BY_KEY.get(key)
    if opt is None:
        raise ValueError("unknown setting")
    if not isinstance(value, str):
        raise ValueError(f"{opt['label']}: the value must be text")
    v = value.strip()
    kind = opt["kind"]
    if kind == "choice":
        if v not in opt["choices"]:
            raise ValueError(f"{opt['label']} must be one of: {', '.join(opt['choices'])}")
        return v
    if kind == "text":
        return check_name(v)
    if kind == "node":
        if v and v not in _node_ids(raw):
            raise ValueError(f"{opt['label']} must be one of the configured Sparks")
        return v
    if kind == "url":
        if not v:
            return ""
        try:
            base = upstream_url(v)
            if key == "controller.monitor_upstream":
                # own_ports: the port(s) this process listens on. controller.ui_port in the file
                # does not choose the listening port (sparksi serve --port does), so it is not used here.
                for port in sorted(set(own_ports)):
                    refuse_self(base, port, own_names)
        except ValueError as exc:
            raise ValueError(str(exc).replace("monitor_upstream", opt["label"])) from None
        return base
    if kind == "path":
        return check_archive_root(v, raw)
    raise ValueError("unknown setting")


_KEY_RE = r'^[ \t]*(?:{0}|"{0}")[ \t]*='


def _value_span(rest: str) -> int:
    """Length of the value at the start of ``rest`` (after '='), not counting a trailing comment."""
    cuts = [m.start() for m in re.finditer("#", rest)] + [len(rest)]
    for cut in cuts:
        piece = rest[:cut]
        if not piece.strip():
            continue
        try:
            tomllib.loads("x = " + piece)
        except tomllib.TOMLDecodeError:
            continue
        return len(piece.rstrip())
    raise ValueError("cannot edit a multi-line value")


def set_key_text(text: str, key: str, value: str) -> str:
    """Set one scalar ``[table] key`` in ``text``, leaving every other byte as it was."""
    table, name = _split(key)
    lines = text.splitlines(keepends=True)
    rendered = toml_value(value)
    section = next(((s, e) for n, s, e in _sections(lines) if n == table and not lines[s].lstrip().startswith("[[")), None)
    if section is None:
        out = text if not text or text.endswith("\n") else text + "\n"
        return out + ("\n" if out.strip() else "") + f"[{table}]\n{name} = {rendered}\n"
    start, end = section
    pattern = re.compile(_KEY_RE.format(re.escape(name)))
    depth = 0
    last = start
    for i in range(start + 1, end):
        line = lines[i]
        if depth == 0 and pattern.match(line):
            body = line.rstrip("\r\n")
            eol = line[len(body):]
            eq = body.index("=") + 1
            rest = body[eq:]
            lead = len(rest) - len(rest.lstrip(" \t"))
            span = _value_span(rest[lead:])
            return "".join(lines[:i]) + body[: eq + lead] + rendered + rest[lead + span:] + eol + "".join(lines[i + 1:])
        depth = _bracket_depth(line, depth)
        if line.strip() and not line.lstrip().startswith("#"):
            last = i
    eol = "\r\n" if lines[start].endswith("\r\n") else "\n"
    if not lines[last].endswith(("\n", "\r")):
        lines[last] += eol
    lines.insert(last + 1, f"{name} = {rendered}{eol}")
    return "".join(lines)


def save(path: Path, key: str, value: str) -> dict[str, Any]:
    """Write one checked value. Refuses any edit that would change something else."""
    path = Path(path)
    old_text = path.read_bytes().decode("utf-8")
    old = tomllib.loads(old_text)
    have = file_value(old, key)
    if have == value or (have is None and value == _default(_BY_KEY[key])):
        return {"changed": False, "backup": None}
    new_text = set_key_text(old_text, key, value)
    want = copy.deepcopy(old)
    table, name = _split(key)
    want.setdefault(table, {})[name] = value
    try:
        same = tomllib.loads(new_text) == want
    except tomllib.TOMLDecodeError:
        same = False
    if not same:
        raise ValueError("the edit would change other settings")
    return {"changed": True, "backup": write_text_atomic(path, new_text)}


def with_value(raw: dict[str, Any], key: str, value: str) -> dict[str, Any]:
    """A copy of ``raw`` with one setting changed (for the load checks before saving)."""
    out = copy.deepcopy(raw)
    table, name = _split(key)
    out.setdefault(table, {})[name] = value
    return out
