# SPDX-License-Identifier: Apache-2.0

"""Controller journal.

Records go to stderr, which the controller unit's journal captures, and to
syslog when ``/dev/log`` is present. Callers use ``logger()`` rather than
printing.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import SysLogHandler

_READY = False


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
    try:
        syslog = SysLogHandler(address="/dev/log")
    except OSError:
        syslog = None
    if syslog is not None:
        syslog.setFormatter(logging.Formatter("sparksi %(levelname)s %(message)s"))
        log.addHandler(syslog)
    _READY = True
    return log
