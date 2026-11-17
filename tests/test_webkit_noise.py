# SPDX-License-Identifier: Apache-2.0

"""Only fetches cut off by a page change are dropped from navigation-time errors."""

from webkit_noise import real_nav_errors


def test_cancelled_fetch_wording_is_dropped():
    noise = [
        "Fetch API cannot load http://127.0.0.1:44369/api/status due to access control checks.",
        "TypeError: Load failed",
        "The network connection was lost.",
    ]
    assert real_nav_errors(noise) == []


def test_other_errors_stay():
    real = [
        "TypeError: undefined is not an object (evaluating 'section.open')",
        "Fetch API cannot load http://127.0.0.1:1/api/status due to access control checks. extra",
        "ReferenceError: Can't find variable: paintZones",
    ]
    assert real_nav_errors(real) == real


def test_cancelled_fetches_as_ci_printed_them_are_dropped():
    noise = [
        "Fetch API cannot load http: /127.0.0.1:38013/api/history?machine=a&range=1h&start=1794801000&end=1794802200 due to access control checks.",
        "Fetch API cannot load http: /127.0.0.1:38013/api/token-day due to access control checks.",
        "Fetch API cannot load http://127.0.0.1:38013/api/history?machine=b&range=1h due to access control checks.",
        "TypeError: Fetch API cannot load http://localhost:38013/api/status due to access control checks.",
    ]
    assert real_nav_errors(noise) == []


def test_cancelled_fetches_elsewhere_stay():
    real = [
        "Fetch API cannot load http://example.com/api/status due to access control checks.",
        "Fetch API cannot load http://127.0.0.1:38013/static/app.js due to access control checks.",
        "Fetch API cannot load http://127.0.0.1:38013/api/history due to access control checks. extra",
        "Fetch API cannot load ftp://127.0.0.1/api/status due to access control checks.",
        "Fetch API cannot load http://127.0.0.1:38013/api/a b due to access control checks.",
    ]
    assert real_nav_errors(real) == real
