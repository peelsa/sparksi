# SPDX-License-Identifier: Apache-2.0

"""The dashboard is styled from vendored Peelsa tokens and never calls a CDN."""

import threading
from urllib.error import HTTPError
from urllib.request import urlopen

from sparksi.serve import STATIC_DIR, DualServer, Handler, asset_path, page_html


def test_page_uses_peelsa_tokens_and_no_cdn() -> None:
    text = page_html()
    assert "http://" not in text
    assert "https://" not in text
    assert "fonts.googleapis" not in text
    assert "fonts.gstatic" not in text
    for token in (
        "--bg: #08090a",
        "--surface: #0f1011",
        "--text: #f7f8f8",
        "--muted: #8a8f98",
        "--peel-deep: #133B01",
        "--peel-bright: #6AAA0C",
        "--accent: var(--peel-bright)",
        "--accent-hover: var(--zest)",
        "--accent-soft: rgba(106, 170, 12, 0.16)",
        "--accent-ink: var(--peel-deep)",
        "--radius: 8px",
    ):
        assert token in text
    assert "#7CC214" not in text
    assert "7cc214" not in text.lower()
    low = text.lower()
    assert "#1b4d3e" not in low
    assert "#c47a3a" not in low
    assert 'src="sparksi-wordmark-transparent.png"' in text
    assert 'href="favicon.png"' in text
    assert "Sparksi" in text
    assert "fonts/inter-latin-400.woff2" in text
    primary = text.split("button.primary", 1)[1].split("button.danger", 1)[0]
    assert "var(--accent-ink)" in primary
    assert "var(--accent)" in primary
    assert "#fff" not in primary.lower()
    assert "white" not in primary.lower()
    assert "Load on head" not in text
    assert 'id="btn-load-head"' in text
    assert ">Load</button>" in text
    assert "/api/config" in text


def test_vendored_mark_and_font_are_served_locally() -> None:
    license_text = (STATIC_DIR / "fonts" / "OFL.txt").read_text(encoding="utf-8")
    assert "SIL Open Font License" in license_text
    httpd = DualServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        for rel, ctype in (
            ("peelsa.png", "image/png"),
            ("favicon.png", "image/png"),
            ("sparksi-wordmark-transparent.png", "image/png"),
            ("sparksi-wordmark-transparent-2x.png", "image/png"),
            ("fonts/inter-latin-400.woff2", "font/woff2"),
            ("fonts/inter-latin-600.woff2", "font/woff2"),
        ):
            with urlopen(f"{base}/{rel}", timeout=5) as resp:
                assert ctype in resp.headers.get("Content-Type", "")
                assert resp.read() == (STATIC_DIR / rel).read_bytes()
        for blocked in ("/static/peelsa.png", "/fonts/OFL.txt", "/index.html", "/etc/passwd"):
            try:
                urlopen(base + blocked, timeout=5)
            except HTTPError as exc:
                assert exc.code == 404
            else:
                raise AssertionError(blocked)
        assert asset_path("/fonts/../peelsa.png") is None
        assert asset_path("/peelsa.png") == (STATIC_DIR / "peelsa.png").resolve()
        assert asset_path("/sparksi-wordmark-transparent.png") == (
            STATIC_DIR / "sparksi-wordmark-transparent.png"
        ).resolve()
    finally:
        httpd.shutdown()


# Cropped file is 306px tall with 302px of ink. This height makes that ink 40px,
# the same visible height as the previous header icon.
HEADER_MARK_HEIGHT = "calc(40px * 306 / 302)"


def test_header_wordmark_is_present_at_mark_height() -> None:
    text = page_html()
    header = text.split('<header class="brandbar">', 1)[1].split("</header>", 1)[0]
    assert header.count("<img") == 1
    assert 'href="/"' in header
    assert 'src="sparksi-wordmark-transparent.png"' in header
    assert (
        'srcset="sparksi-wordmark-transparent.png 1x, sparksi-wordmark-transparent-2x.png 2x"'
        in header
    )
    assert 'alt="SparkSI"' in header
    assert 'width="1107"' in header
    assert 'height="306"' in header
    assert "peelsa.png" not in header
    assert "<h1>" not in header
    rule = text.split(".brand img", 1)[1].split("}", 1)[0]
    assert f"height: {HEADER_MARK_HEIGHT}" in rule
    assert "width: auto" in rule
