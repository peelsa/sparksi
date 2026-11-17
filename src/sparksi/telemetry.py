# SPDX-License-Identifier: Apache-2.0

"""Durable telemetry files. The one-hour chart cache is separate.

Collection appends one JSON line per fresh sample. Nothing in this module
deletes old lines. A GET may read the files and must not write them.

Lines are JSON objects. A missing key means that reading was absent, not
zero. ``t`` is whole UTC seconds. The directory name is the machine role
(``head`` or ``worker``). Files do not store host names, addresses, or paths.

Each day has three files:

* ``YYYY-MM-DD.jsonl`` — the samples, append only
* ``YYYY-MM-DD.tip.json`` — one small JSON object for that day
* ``YYYY-MM-DD.idx`` — byte offset and timestamp about once a minute, append only

``telemetry/days`` gains one line when a day file is first recorded.
``telemetry/<role>.recent.json`` is the role summary. A query of at most 24
hours builds one or two dated paths and does not list the directory. Discovery
reads a byte slice of the catalog, not every name.
"""

from __future__ import annotations

import errno
import json
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

# Every role a Spark can hold (config.role_name). A site lists only the ones it has.
ROLES = ("head", "worker", "node3", "node4")  # config.ALL_ROLES; kept literal so this module stays import-light
FIELDS = ("gpu_pct", "cpu_pct", "mem_pct", "mem_used_gib", "gpu_temp_c", "tok_s", "power_w")
IDX_STRIDE_S = 60
DISCOVER_LIMIT = 30
DISCOVER_MAX = 50
_CATALOG_LINE = re.compile(r"^(head|worker|node3|node4)/(\d{4}-\d{2}-\d{2})\.jsonl$")
_UNCOMMITTED_MAX = 65536

SCHEMA: dict[str, Any] = {
    "name": "sparksi-telemetry",
    "version": 1,
    "record": "one JSON object per line",
    "clock": "t is integer UTC epoch seconds",
    "role": "directory name is the role (head, worker, node3 or node4); each line repeats role",
    "missing": "a key that is absent was not reported; zero is a real zero",
    "retention": "files are kept indefinitely; nothing here deletes them",
    "hot_window_s": 3600,
    "query_span_s": 86400,
    "layout": {
        "samples": "telemetry/<role>/YYYY-MM-DD.jsonl",
        "tip": "telemetry/<role>/YYYY-MM-DD.tip.json",
        "offsets": "telemetry/<role>/YYYY-MM-DD.idx",
        "catalog": "telemetry/days",
        "recent": "telemetry/<role>.recent.json",
    },
    "fields": {
        "t": "integer UTC epoch seconds",
        "role": "head, worker, node3 or node4",
        "gpu_pct": "GPU busy, percent",
        "cpu_pct": "CPU busy, percent",
        "mem_pct": "unified memory in use, percent",
        "mem_used_gib": "unified memory in use, gibibytes",
        "gpu_temp_c": "GPU temperature, celsius",
        "tok_s": "decode tokens per second, summed for models on this role",
        "power_w": "board power, watts",
    },
}


def _site_roles() -> tuple[str, ...]:
    """Roles this site's config lists, in order (head, worker, ...)."""
    try:
        from sparksi.config import CFG

        return tuple(role for role in CFG.roles if role in ROLES)
    except Exception:  # noqa: BLE001
        return ("head", "worker")


def _write_error(exc: OSError) -> str:
    if exc.errno == errno.ENOSPC:
        return "telemetry disk is full"
    if exc.errno in (errno.EACCES, errno.EPERM):
        return "telemetry directory is not writable"
    return "telemetry could not be written"


def _parse_line(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text:
        return None
    try:
        row = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(row, dict) or row.get("t") is None:
        return None
    try:
        int(row["t"])
    except (TypeError, ValueError, OverflowError):
        return None
    return row


def _day(stamp: int) -> str:
    return time.strftime("%Y-%m-%d", time.gmtime(int(stamp)))


def valid_catalog_name(name: str) -> bool:
    """Exact ``role/YYYY-MM-DD.jsonl``. A slash alone is not a day."""
    match = _CATALOG_LINE.fullmatch(name)
    if match is None:
        return False
    try:
        time.strptime(match.group(2), "%Y-%m-%d")
    except ValueError:
        return False
    return True


def _edges(path: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None, int]:
    """First and last complete lines. A torn last line is ignored, not deleted."""
    try:
        with path.open("rb") as handle:
            first_raw = handle.readline()
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size <= 0:
                return None, None, len(first_raw)
            window = min(size, 8192)
            handle.seek(size - window)
            tail = handle.read(window)
    except OSError:
        return None, None, 0
    first = _parse_line(first_raw.decode("utf-8", "replace"))
    last = None
    for line in tail.decode("utf-8", "replace").splitlines():
        parsed = _parse_line(line)
        if parsed is not None:
            last = parsed
    return first, last or first, len(first_raw) + len(tail)


class Archive:
    """Append-only JSONL files under ``root`` plus per-day tip and offset files."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._guard = threading.Lock()
        self.last_error = ""
        self.bytes_read = 0
        self.bytes_written = 0
        # Off unless a test asks. The deque cannot grow with the archive.
        self.trace = False
        self.traced: deque[str] = deque(maxlen=8)
        self._idx_key: tuple[str, str] | None = None
        self._idx_stamp: int | None = None

    def _touch(self, name: str) -> None:
        if self.trace:
            self.traced.append(name)

    def append(self, role: str, sample: dict[str, Any]) -> None:
        """Write one sample. Disk errors stay on ``last_error`` and do not raise."""
        if role not in ROLES:
            return
        try:
            stamp = int(sample.get("t"))
        except (TypeError, ValueError, OverflowError):
            return
        line = {"t": stamp, "role": role}
        for key in FIELDS:
            if key in sample and sample[key] is not None:
                line[key] = sample[key]
        with self._guard:
            try:
                self._append_unlocked(role, line)
            except OSError as exc:
                self.last_error = _write_error(exc)
                self._remember_error()

    def _append_unlocked(self, role: str, line: dict[str, Any]) -> None:
        day = _day(int(line["t"]))
        # The catalog entry is durable before any sample byte. A crash here leaves
        # an empty day that discovery can still see, without scanning for files.
        self._ensure_listed(role, day)
        folder = self.root / role
        folder.mkdir(parents=True, exist_ok=True)
        schema = self.root / "schema.json"
        if not schema.exists():
            schema.write_text(json.dumps(SCHEMA, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        path = folder / f"{day}.jsonl"
        payload = (json.dumps(line, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            if size:
                handle.seek(size - 1)
                last = handle.read(1)
                handle.seek(0, os.SEEK_END)
                if last != b"\n":
                    # Keep the torn bytes. A newline stops them swallowing the next sample.
                    handle.write(b"\n")
                    self.bytes_written += 1
            offset = handle.tell()
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            self.bytes_written += len(payload)
        self._append_meta(role, day, path, line, offset, new_file=size == 0)
        self._clear_error()

    def _append_meta(
        self,
        role: str,
        day: str,
        path: Path,
        line: dict[str, Any],
        offset: int,
        *,
        new_file: bool,
    ) -> None:
        """Record this sample only if the previous tip accounts for every earlier byte."""
        stamp = int(line["t"])
        size = path.stat().st_size
        previous = self._read_tip(self._tip_path(path))
        chained = (
            previous is not None
            and previous.get("samples") is not None
            and previous.get("first_t") is not None
            and previous.get("bytes") == offset
        )
        if chained:
            record = {
                "bytes": size,
                "first_t": int(previous["first_t"]),
                "last_t": stamp,
                "samples": int(previous["samples"]) + 1,
            }
        elif offset == 0 or new_file:
            record = {"bytes": size, "first_t": stamp, "last_t": stamp, "samples": 1}
        else:
            # The tip does not describe the bytes already in the file. Count stays unknown.
            record = {"bytes": size, "first_t": None, "last_t": stamp, "samples": None}
        # Register the day before the tip. A tip must not be the only sign that the day exists.
        self._ensure_listed(role, day)
        self._atomic_json(self._tip_path(path), record)
        self._note_recent(role, record)
        self._note_idx(role, day, path, offset, stamp)

    def _tip_path(self, jsonl: Path) -> Path:
        return jsonl.with_name(jsonl.stem + ".tip.json")

    def _read_tip(self, path: Path) -> dict[str, Any] | None:
        """One small tip file. A torn or partial file is untrusted, not repaired here."""
        try:
            data = path.read_bytes()
        except OSError:
            return None
        self.bytes_read += len(data)
        try:
            row = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(row, dict) or "bytes" not in row or "last_t" not in row:
            return None
        return row

    def _atomic_bytes(self, path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        self.bytes_written += len(raw)

    def _atomic_json(self, path: Path, record: dict[str, Any]) -> None:
        raw = (json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")
        self._atomic_bytes(path, raw)

    def _listed_path(self, role: str, day: str) -> Path:
        return self.root / "listed" / role / day

    def _catalog_commit(self) -> int:
        try:
            return int((self.root / "days.ok").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return 0

    def _set_catalog_commit(self, offset: int) -> None:
        self._atomic_bytes(self.root / "days.ok", f"{int(offset)}\n".encode("ascii"))

    def _mark_listed(self, role: str, day: str) -> None:
        """Completion for one day. Call only after the catalog line is durable."""
        self._atomic_bytes(self._listed_path(role, day), b"ok\n")

    def _uncommitted(self, catalog: Path) -> tuple[int, int, bytes]:
        """Bytes just past the checkpoint. The read starts at the checkpoint.

        A tail read would skip an unmarked line that still sits at that offset.
        """
        if not catalog.is_file():
            return 0, 0, b""
        size = catalog.stat().st_size
        committed = self._catalog_commit()
        if committed < 0 or committed > size:
            committed = 0
        try:
            with catalog.open("rb") as handle:
                handle.seek(committed)
                suffix = handle.read(_UNCOMMITTED_MAX)
        except OSError:
            return committed, size, b""
        self.bytes_read += len(suffix)
        return committed, size, suffix

    def _line_marked(self, name: str) -> bool:
        role, filename = name.split("/", 1)
        day = filename[: -len(".jsonl")]
        return self._listed_path(role, day).is_file()

    def _mark_name(self, name: str) -> None:
        role, filename = name.split("/", 1)
        self._mark_listed(role, filename[: -len(".jsonl")])

    def _commit_resolved_prefix(self, catalog: Path) -> None:
        """Move the checkpoint through marked lines. Stop before an unmarked one."""
        if not catalog.is_file():
            return
        committed, _size, suffix = self._uncommitted(catalog)
        if not suffix:
            return
        offset = 0
        resolved = 0
        while offset < len(suffix):
            newline = suffix.find(b"\n", offset)
            if newline < 0:
                break
            raw = suffix[offset:newline]
            end = newline + 1
            name = raw.decode("utf-8", "replace").strip()
            if valid_catalog_name(name) and not self._line_marked(name):
                break
            resolved = end
            offset = end
        if resolved:
            self._set_catalog_commit(committed + resolved)

    def _mark_visible(self, catalog: Path) -> None:
        """Write markers for complete lines past the checkpoint. Stop on the first failure."""
        _committed, _size, suffix = self._uncommitted(catalog)
        offset = 0
        while offset < len(suffix):
            newline = suffix.find(b"\n", offset)
            if newline < 0:
                break
            name = suffix[offset:newline].decode("utf-8", "replace").strip()
            offset = newline + 1
            if valid_catalog_name(name) and not self._line_marked(name):
                self._mark_name(name)

    def _ensure_listed(self, role: str, day: str) -> None:
        """Record ``role/YYYY-MM-DD.jsonl`` once, before any sample from that day.

        The checkpoint advances only through lines that already have markers.
        Another role cannot move it past a line whose marker is still missing.
        """
        want = f"{role}/{day}.jsonl"
        if not valid_catalog_name(want):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        catalog = self.root / "days"
        if self._line_marked(want):
            self._commit_resolved_prefix(catalog)
            return
        caught_up = False
        for _ in range(64):
            if self._claim_torn(catalog, want):
                self._commit_resolved_prefix(catalog)
                return
            self._separate_other_torn(catalog, want)
            self._mark_visible(catalog)
            self._commit_resolved_prefix(catalog)
            if self._line_marked(want):
                return
            size = catalog.stat().st_size if catalog.is_file() else 0
            if self._catalog_commit() >= size:
                caught_up = True
                break
        if not caught_up:
            raise OSError(errno.EIO, "catalog checkpoint")
        raw = (want + "\n").encode("utf-8")
        with catalog.open("ab") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.bytes_written += len(raw)
        self._mark_name(want)
        self._commit_resolved_prefix(catalog)

    def _claim_torn(self, catalog: Path, want: str) -> bool:
        """A torn tail that is already this exact name becomes one marked line."""
        if not catalog.is_file() or self._ends_with_newline(catalog):
            return False
        _committed, _size, suffix = self._uncommitted(catalog)
        partial = suffix.rsplit(b"\n", 1)[-1]
        if partial.decode("utf-8", "replace") != want:
            return False
        self._separate_tail(catalog)
        self._mark_name(want)
        return True

    def _separate_other_torn(self, catalog: Path, want: str) -> None:
        if not catalog.is_file() or self._ends_with_newline(catalog):
            return
        _committed, _size, suffix = self._uncommitted(catalog)
        partial = suffix.rsplit(b"\n", 1)[-1]
        if partial.decode("utf-8", "replace") == want:
            return
        self._separate_tail(catalog)

    def _ends_with_newline(self, path: Path) -> bool:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                return True
            handle.seek(handle.tell() - 1)
            return handle.read(1) == b"\n"

    def _note_recent(self, role: str, record: dict[str, Any]) -> None:
        recent = {
            "day": _day(int(record["last_t"])),
            "first_t": record.get("first_t"),
            "last_t": record.get("last_t"),
            "samples": record.get("samples"),
        }
        self._atomic_json(self.root / f"{role}.recent.json", recent)

    def _note_idx(self, role: str, day: str, path: Path, offset: int, stamp: int) -> None:
        key = (role, day)
        if self._idx_key != key:
            self._idx_key = key
            self._idx_stamp = self._last_idx_stamp(path.with_suffix(".idx"))
        previous = self._idx_stamp
        if previous is not None and stamp - previous < IDX_STRIDE_S:
            return
        idx = path.with_suffix(".idx")
        self._separate_tail(idx)
        raw = f"{offset} {stamp}\n".encode("utf-8")
        with idx.open("ab") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        self.bytes_written += len(raw)
        self._idx_stamp = stamp

    def _last_idx_stamp(self, path: Path) -> int | None:
        if not path.is_file():
            return None
        tail = self._tail(path, 128)
        if tail and not tail.endswith("\n"):
            tail = tail.rsplit("\n", 1)[0]
        stamp = None
        for line in tail.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                stamp = int(parts[1])
            except ValueError:
                continue
        return stamp

    def _separate_tail(self, path: Path) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            return
        with path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(size - 1)
            last = handle.read(1)
            self.bytes_read += 1
            if last != b"\n":
                handle.seek(0, os.SEEK_END)
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
                self.bytes_written += 1

    def _tail(self, path: Path, limit: int) -> str:
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - limit))
                data = handle.read(limit)
        except OSError:
            return ""
        self.bytes_read += len(data)
        return data.decode("utf-8", "replace")

    def _clear_error(self) -> None:
        self.last_error = ""
        path = self.root / "error"
        if path.is_file():
            path.unlink()

    def _remember_error(self) -> None:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / "error"
            path.write_text(self.last_error + "\n", encoding="utf-8")
        except OSError:
            return

    def _stored_error(self) -> str:
        try:
            return (self.root / "error").read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    def reduce(self, role: str, start: int, end: int, visit: Callable[[dict[str, Any]], None]) -> bool:
        """Stream samples in ``[start, end]``. Read errors are recorded, not raised."""
        if role not in ROLES or end < start:
            return True
        ok = True
        for path in self._paths(role, start, end):
            if not self._scan(path, start, end, visit):
                ok = False
        return ok

    def _scan(self, path: Path, start: int, end: int, visit: Callable[[dict[str, Any]], None]) -> bool:
        offset = self._indexed_offset(path, start)
        self._touch(path.name)
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                while True:
                    raw = handle.readline()
                    if not raw:
                        break
                    self.bytes_read += len(raw)
                    row = _parse_line(raw.decode("utf-8", "replace"))
                    if row is None:
                        continue
                    stamp = int(row["t"])
                    if stamp < start:
                        continue
                    if stamp > end:
                        return True
                    visit(row)
            return True
        except OSError as exc:
            self.last_error = _write_error(exc)
            return False

    def _indexed_offset(self, path: Path, start: int) -> int:
        idx = path.with_suffix(".idx")
        try:
            data = idx.read_bytes()
        except OSError:
            return 0
        self.bytes_read += len(data)
        text = data.decode("utf-8", "replace")
        if data and not data.endswith(b"\n"):
            text = text.rsplit("\n", 1)[0]
        best = 0
        for line in text.splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                offset = int(parts[0])
                stamp = int(parts[1])
            except ValueError:
                continue
            if stamp <= start:
                best = offset
            else:
                break
        return best

    def _paths(self, role: str, start: int, end: int) -> list[Path]:
        """A window of at most 24 hours touches one or two UTC days. No directory listing."""
        days: list[str] = []
        for stamp in (start, end):
            day = _day(stamp)
            if day not in days:
                days.append(day)
        found: list[Path] = []
        for day in days:
            path = self.root / role / f"{day}.jsonl"
            self._touch(f"stat:{role}/{day}.jsonl")
            if path.is_file():
                found.append(path)
        return found

    def discover(
        self,
        *,
        role: str | None = None,
        cursor: str | None = None,
        limit: int = DISCOVER_LIMIT,
    ) -> dict[str, Any]:
        """One page of day files. Does not read sample bodies or rewrite files."""
        if role is not None and role not in ROLES:
            raise ValueError("role")
        try:
            size = int(limit)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("limit") from exc
        if size < 1 or size > DISCOVER_MAX:
            raise ValueError("limit")
        page, follow = self._catalog_page(cursor, size, role)
        files = [self._file_row(name) for name in page]
        error = self.last_error or self._stored_error()
        return {
            "kept": "indefinitely",
            "hot_window_s": 3600,
            "query_span_s": 86400,
            "read": "/api/history",
            "schema_path": "telemetry/schema.json",
            "schema": SCHEMA,
            "last_error": error,
            "note": "Readings from before this archive started were not recorded.",
            "roles": {item: self._recent(item) for item in _site_roles()},
            "limit": size,
            "next": follow,
            "files": files,
        }

    def _catalog_page(self, cursor: str | None, limit: int, role: str | None) -> tuple[list[str], str | None]:
        """Read at most a few catalog lines from ``cursor``. Never the whole lifetime."""
        path = self.root / "days"
        try:
            offset = int(cursor or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("cursor") from exc
        if offset < 0:
            raise ValueError("cursor")
        if not path.is_file():
            return [], None
        cap = limit * 8
        found: list[str] = []
        next_offset = offset
        eof = False
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                if offset > size:
                    raise ValueError("cursor")
                handle.seek(offset)
                scanned = 0
                while len(found) < limit and scanned < cap:
                    raw = handle.readline()
                    if not raw:
                        eof = True
                        break
                    self.bytes_read += len(raw)
                    scanned += 1
                    next_offset = handle.tell()
                    if not raw.endswith(b"\n"):
                        return found, str(next_offset - len(raw))
                    name = raw.decode("utf-8", "replace").strip()
                    if not valid_catalog_name(name):
                        continue
                    item = name.split("/", 1)[0]
                    if role is not None and item != role:
                        continue
                    found.append(name)
        except OSError as exc:
            self.last_error = _write_error(exc)
            return found, None
        if eof:
            return found, None
        return found, str(next_offset)

    def _recent(self, role: str) -> dict[str, Any]:
        path = self.root / f"{role}.recent.json"
        try:
            data = path.read_bytes()
        except OSError:
            return {"first_t": None, "last_t": None, "samples": None, "day": None}
        self.bytes_read += len(data)
        try:
            row = json.loads(data.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return {"first_t": None, "last_t": None, "samples": None, "day": None}
        if not isinstance(row, dict):
            return {"first_t": None, "last_t": None, "samples": None, "day": None}
        return {
            "first_t": row.get("first_t"),
            "last_t": row.get("last_t"),
            "samples": row.get("samples"),
            "day": row.get("day"),
        }

    def _file_row(self, name: str) -> dict[str, Any]:
        role, filename = name.split("/", 1)
        path = self.root / role / filename
        tip = self._read_tip(self._tip_path(path))
        try:
            size = path.stat().st_size
        except OSError:
            size = None
        trusted = (
            tip is not None
            and size is not None
            and tip.get("bytes") == size
            and tip.get("samples") is not None
            and tip.get("first_t") is not None
        )
        if trusted:
            first_t = int(tip["first_t"])
            last_t = int(tip["last_t"])
            samples = int(tip["samples"])
        else:
            first, last, nbytes = _edges(path)
            self.bytes_read += nbytes
            first_t = int(first["t"]) if first else None
            last_t = int(last["t"]) if last else None
            samples = None
        return {
            "role": role,
            "file": filename,
            "path": f"telemetry/{role}/{filename}",
            "first_t": first_t,
            "last_t": last_t,
            "samples": samples,
            "bytes": size,
        }
