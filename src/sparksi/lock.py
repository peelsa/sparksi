# SPDX-License-Identifier: Apache-2.0

"""Lease + flock. Default ON: missing/corrupt/expired ⇒ locked."""

from __future__ import annotations

import fcntl
import json
import os
import secrets
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sparksi.config import CFG, env, state_subdir

DEFAULT_UNLOCK_TTL_S = 15 * 60


class LockedError(RuntimeError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat(timespec="seconds")


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def state_dir() -> Path:
    state = env("SPARKSI_STATE_DIR")
    if state:
        return Path(state)
    lock_env = env("SPARKSI_LOCK")
    if lock_env:
        return Path(lock_env).parent
    configured = str(CFG.controller.get("state_dir") or "")
    if configured:
        return Path(configured).expanduser()
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        return state_subdir(Path(runtime))
    uid = os.getuid()
    if Path(f"/run/user/{uid}").is_dir():
        return state_subdir(Path(f"/run/user/{uid}"))
    return state_subdir(Path.home() / ".local" / "state")


def default_lock_path() -> Path:
    lock_env = env("SPARKSI_LOCK")
    if lock_env:
        return Path(lock_env)
    return state_dir() / "lock"


def default_holder() -> str:
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "unknown"
    host = socket.gethostname()
    return f"{user}@{host}"


@dataclass
class LockState:
    locked: bool
    holder: str | None = None
    pid: int | None = None
    acquired_at: str | None = None
    expires_at: str | None = None
    reason: str | None = None
    source: str = "missing"

    def to_dict(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "locked_for_actions": self.locked,
            "on": "ON" if self.locked else "OFF",
            "holder": self.holder,
            "pid": self.pid,
            "acquired_at": self.acquired_at,
            "expires_at": self.expires_at,
            "reason": self.reason,
            "source": self.source,
        }


def _interpret(raw: dict[str, Any] | None, source: str) -> LockState:
    if raw is None:
        return LockState(locked=True, reason="no lease file", source=source)
    if raw.get("locked") is True:
        return LockState(
            locked=True,
            holder=raw.get("holder"),
            pid=raw.get("pid"),
            acquired_at=raw.get("acquired_at"),
            expires_at=raw.get("expires_at"),
            reason=raw.get("reason") or "held",
            source=source,
        )
    if raw.get("locked") is False:
        exp = _parse_iso(raw.get("expires_at"))
        if exp is None:
            return LockState(
                locked=True,
                holder=raw.get("holder"),
                reason="unlock missing expires_at (fail closed)",
                source=source,
            )
        if _now() >= exp:
            return LockState(
                locked=True,
                holder=raw.get("holder"),
                expires_at=raw.get("expires_at"),
                reason="unlock lease expired",
                source=source,
            )
        return LockState(
            locked=False,
            holder=raw.get("holder"),
            pid=raw.get("pid"),
            acquired_at=raw.get("acquired_at"),
            expires_at=raw.get("expires_at"),
            reason=raw.get("reason") or "explicit unlock",
            source=source,
        )
    return LockState(locked=True, reason="corrupt lease (locked field)", source=source)


class Lock:
    def __init__(self, path: Path | None = None):
        self.path = path or default_lock_path()

    def _with_flock(self, exclusive: bool):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # sibling flock file so readers/writers serialize
        flock_path = self.path.with_suffix(".flock")
        fh = flock_path.open("a+")
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        return fh

    def read(self) -> LockState:
        fh = self._with_flock(False)
        try:
            if not self.path.is_file():
                return _interpret(None, "missing")
            try:
                raw = json.loads(self.path.read_text())
            except (OSError, json.JSONDecodeError):
                return LockState(locked=True, reason="corrupt lease json", source="corrupt")
            if not isinstance(raw, dict):
                return LockState(locked=True, reason="corrupt lease type", source="corrupt")
            return _interpret(raw, "file")
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def _write(self, payload: dict[str, Any]) -> LockState:
        fh = self._with_flock(True)
        try:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2) + "\n")
            tmp.replace(self.path)
            return _interpret(payload, "file")
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            fh.close()

    def acquire(self, reason: str = "lock", holder: str | None = None) -> LockState:
        payload = {
            "locked": True,
            "holder": holder or default_holder(),
            "pid": os.getpid(),
            "acquired_at": _iso(_now()),
            "expires_at": None,
            "reason": reason,
        }
        return self._write(payload)

    def token_path(self) -> Path:
        return self.path.parent / "unlock.token"

    def write_unlock_token(self) -> str:
        tok = secrets.token_urlsafe(24)
        p = self.token_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(tok + "\n")
        os.chmod(p, 0o600)
        return tok

    def verify_token(self, tok: str | None) -> bool:
        if not tok:
            return False
        p = self.token_path()
        if not p.is_file():
            return False
        got = p.read_text().strip()
        if not got:
            return False
        return secrets.compare_digest(got, tok.strip())

    def allow_api_write(self, token: str | None) -> tuple[bool, str]:
        """POST /api/wanted and POST /api/unlock: lock-off OR matching unlock token."""
        st = self.read()
        if not st.locked:
            return True, "lock-off"
        if self.verify_token(token):
            return True, "unlock-token"
        return False, "lock ON and missing/invalid unlock token"

    def release(self, ttl_s: int = DEFAULT_UNLOCK_TTL_S, holder: str | None = None) -> tuple[LockState, str]:
        exp = _now().timestamp() + max(1, int(ttl_s))
        payload = {
            "locked": False,
            "holder": holder or default_holder(),
            "pid": os.getpid(),
            "acquired_at": _iso(_now()),
            "expires_at": _iso(datetime.fromtimestamp(exp, tz=timezone.utc)),
            "reason": f"unlock ttl={ttl_s}s",
        }
        st = self._write(payload)
        tok = self.write_unlock_token()
        return st, tok

    def require_unlocked(self, action: str = "mutate") -> LockState:
        st = self.read()
        if st.locked:
            who = st.holder or "unknown"
            raise LockedError(
                f"refusing {action}: lock is ON (holder={who}; {st.reason}). "
                "sparksi unlock first. No mutation while locked."
            )
        return st
