# SPDX-License-Identifier: Apache-2.0

"""Page errors in WebKit probes that come from leaving a page, not from the page.

When ``page.goto`` or ``page.reload`` leaves a page, WebKit reports that page's
fetches still in flight as page errors ("Fetch API cannot load ... due to access
control checks."). Probes splice in ``WATCH_JS``: errors raised while a navigation
step runs go to ``navErrors``, every other error to ``errors``. ``real_nav_errors``
keeps the navigation-time errors that are not a cancelled fetch to the probe's own
local /api/ server, so anything else still fails the test.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

WATCH_JS = r"""
function watchErrors(page, out) {
  out.errors = out.errors || [];
  out.navErrors = out.navErrors || [];
  let moving = 0;
  page.on("pageerror", (err) => (moving > 0 ? out.navErrors : out.errors).push(String(err)));
  return async (step) => {
    moving += 1;
    try {
      return await step();
    } finally {
      moving -= 1;
    }
  };
}
"""

_PREFIX = r"^(?:TypeError: )?"
_CANCELLED_FETCH = re.compile(_PREFIX + r"Fetch API cannot load (?P<url>\S.*?) due to access control checks\.$")
_CUT_OFF = re.compile(_PREFIX + r"(?:Load failed|The network connection was lost\.)$")
_SCHEME = re.compile(r"^(https?):\s*/+")


def _own_api(url: str) -> bool:
    """True for an http(s) URL on the probe's own local server under /api/.

    WebKit's message can render the scheme as ``http: /``; that is read as ``http://``.
    """
    text = _SCHEME.sub(r"\1://", url.strip())
    if any(ch.isspace() for ch in text):
        return False
    parts = urlsplit(text)
    return parts.scheme in ("http", "https") and parts.hostname in ("127.0.0.1", "localhost") and parts.path.startswith("/api/")


def _cut_off(text: str) -> bool:
    if _CUT_OFF.match(text):
        return True
    match = _CANCELLED_FETCH.match(text)
    return bool(match) and _own_api(match.group("url"))


def real_nav_errors(nav_errors: list[str]) -> list[str]:
    """The navigation-time errors that are not a fetch cut off by the page change."""
    return [text for text in nav_errors if not _cut_off(text)]
