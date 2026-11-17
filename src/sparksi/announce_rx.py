# SPDX-License-Identifier: Apache-2.0

"""Receive Spark announce datagrams. No SSH, and no socket until tailscale0 has an address."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from typing import Any, Callable

from sparksi.announce import MAX_BYTES, to_peer
from sparksi.config import CFG

PORT = int(CFG.controller.get("announce_port") or 7501)


class Slot:
    def __init__(self, data: dict[str, Any] | None, age: float | None) -> None:
        self._data = data
        self._age = age

    def snapshot(self) -> tuple[dict[str, Any] | None, float | None]:
        return self._data, self._age


class Inbox:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._rows: dict[str, tuple[dict[str, Any], float]] = {}

    def accept(self, src_ip: str, raw: bytes) -> bool:
        if not raw or len(raw) > MAX_BYTES:
            return False
        host = CFG.announce_allow().get(src_ip)
        if not host:
            return False
        try:
            pkt = json.loads(raw.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError):
            return False
        if isinstance(pkt, dict) and "ph" in pkt and "h" not in pkt:
            return self._accept_beat(host, pkt)
        if not isinstance(pkt, dict) or pkt.get("h") != host:
            return False
        data = to_peer(pkt)
        data["announce_from"] = src_ip
        if data.get("mem") in (None, [0.0, 0.0]) and data.get("gpu") == [None, None, None]:
            return False
        with self._lock:
            self._rows[host] = (data, self._clock())
        try:
            from sparksi.llama_day import note

            note(host, data.get("http") or {})
        except Exception:
            pass
        return True

    def _accept_beat(self, host: str, pkt: dict[str, Any]) -> bool:
        """A probe beat: validated and size-bounded by sparksi.heartbeat, filed under the sender's node."""
        from sparksi.heartbeat import BEATS, parse

        beat = parse(pkt)
        if beat is None or beat["ph"] != host:
            return False
        BEATS.note(host, beat)
        return True

    def peer(self, host: str) -> Slot:
        with self._lock:
            row = self._rows.get(host)
        if not row:
            return Slot(None, None)
        data, ts = row
        return Slot(data, self._clock() - ts)


def tailscale_ipv4() -> str | None:
    try:
        out = subprocess.run(
            ["tailscale", "ip", "-4"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip().split()
    except (OSError, subprocess.SubprocessError):
        return None
    for ip in out:
        if ip.startswith("100."):
            return ip
    return None


class Listener:
    """Bind UDP 7501 on the Tailscale address. Rebind when that address changes."""

    def __init__(self, inbox: Inbox) -> None:
        self.inbox = inbox
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, name="dgx-announce-rx", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        sock: socket.socket | None = None
        bound = ""
        while not self._stop.is_set():
            ip = tailscale_ipv4()
            if not ip:
                if sock is not None:
                    sock.close()
                    sock = None
                    bound = ""
                self._stop.wait(2)
                continue
            if sock is None or bound != ip:
                if sock is not None:
                    sock.close()
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((ip, PORT))
                    sock.settimeout(1.0)
                    bound = ip
                    print(f"dgx-announce listen {ip}:{PORT}")
                except OSError as exc:
                    print(f"dgx-announce bind {ip}:{PORT} failed ({exc})")
                    if sock is not None:
                        sock.close()
                    sock = None
                    bound = ""
                    self._stop.wait(2)
                    continue
            try:
                raw, addr = sock.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError:
                sock.close()
                sock = None
                bound = ""
                continue
            self.inbox.accept(addr[0], raw)


INBOX = Inbox()
_LISTENER = Listener(INBOX)


def start_listener() -> None:
    _LISTENER.start()
