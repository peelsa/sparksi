# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from sparksi.lock import Lock, LockedError


def test_missing_file_is_locked(tmp_path: Path) -> None:
    lk = Lock(tmp_path / "lock")
    st = lk.read()
    assert st.locked is True
    assert st.source == "missing"


def test_corrupt_is_locked(tmp_path: Path) -> None:
    p = tmp_path / "lock"
    p.write_text("not-json")
    st = Lock(p).read()
    assert st.locked is True
    assert st.source == "corrupt"


def test_unlock_then_require(tmp_path: Path) -> None:
    lk = Lock(tmp_path / "lock")
    st, tok = lk.release(ttl_s=60)
    assert tok
    assert st.locked is False
    st = lk.require_unlocked("test")
    assert st.locked is False
    assert lk.verify_token(tok)
    assert not lk.verify_token("nope")


def test_require_fails_when_locked(tmp_path: Path) -> None:
    lk = Lock(tmp_path / "lock")
    lk.acquire()
    with pytest.raises(LockedError):
        lk.require_unlocked("swap")


def test_api_write_needs_token_when_locked(tmp_path: Path) -> None:
    lk = Lock(tmp_path / "lock")
    lk.acquire()
    ok, why = lk.allow_api_write(None)
    assert ok is False
    st, tok = lk.release(ttl_s=60)
    assert st.locked is False
    ok, why = lk.allow_api_write(None)
    assert ok is True
    lk.acquire()
    ok, _ = lk.allow_api_write(tok)
    assert ok is True


def test_expired_unlock_reverts_to_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lk = Lock(tmp_path / "lock")
    lk.release(ttl_s=1)
    # rewind clock past expiry by rewriting expires_at to the past
    p = tmp_path / "lock"
    text = p.read_text().replace("expires_at", "expires_at")
    # cheaper: write an already-expired lease
    p.write_text(
        '{"locked": false, "holder": "t", "expires_at": "2000-01-01T00:00:00+00:00"}\n'
    )
    st = lk.read()
    assert st.locked is True
    assert "expired" in (st.reason or "")
    del text


def test_unlock_without_expiry_is_locked(tmp_path: Path) -> None:
    p = tmp_path / "lock"
    p.write_text('{"locked": false, "holder": "t"}\n')
    st = Lock(p).read()
    assert st.locked is True
