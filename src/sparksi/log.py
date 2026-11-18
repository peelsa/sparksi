# SPDX-License-Identifier: Apache-2.0

"""Controller journal.

Records go to stderr, which the controller unit's journal captures, and to
syslog when ``/dev/log`` accepts a connection. Callers use ``logger()``
rather than printing.
"""

from __future__ import annotations

import logging
import socket
import sys
from logging.handlers import SysLogHandler

_READY = False
_SYSLOG_ADDRESS = "/dev/log"


class _QuietSysLogHandler(SysLogHandler):
    """Syslog that drops a record instead of printing a traceback.

    If syslog goes away later (journald restarting, say), each send fails.
    The base handler then prints a traceback for every record. The stderr
    handler still gets every record, and the base handler reconnects on the
    next send, so dropping the syslog copy is enough.
    """

    def handleError(self, record: logging.LogRecord) -> None:
        pass


def _syslog_reachable(address: str) -> bool:
    """True when something is listening on the syslog socket.

    ``SysLogHandler`` ignores a failed connect when it is created, so a
    ``/dev/log`` that points at a dead socket would otherwise be attached.
    """
    for kind in (socket.SOCK_DGRAM, socket.SOCK_STREAM):
        probe = socket.socket(socket.AF_UNIX, kind)
        probe.settimeout(1.0)  # a stuck listener must not hang start-up
        try:
            probe.connect(address)
        except OSError:
            continue
        finally:
            probe.close()
        return True
    return False


def logger() -> logging.Logger:
    """The ``sparksi`` logger, configured once."""
    global _READY
    log = logging.getLogger("sparksi")
    if _READY:
        return log
    log.setLevel(logging.INFO)
    if not any(isinstance(handler, logging.StreamHandler) for handler in log.handlers):
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(logging.Formatter("%(levelname)s %(name)s %(message)s"))
        log.addHandler(stream)
    if _syslog_reachable(_SYSLOG_ADDRESS):
        syslog = _QuietSysLogHandler(address=_SYSLOG_ADDRESS)
        syslog.setFormatter(logging.Formatter("sparksi %(levelname)s %(message)s"))
        log.addHandler(syslog)
    _READY = True
    return log
