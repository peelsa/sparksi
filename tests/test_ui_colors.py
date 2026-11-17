# SPDX-License-Identifier: Apache-2.0

"""The UI color gate accepts the Peelsa card and rejects anything else."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.check_ui_colors import off_card, scan  # noqa: E402


def test_dashboard_colors_stay_on_the_card():
    assert scan() == []


def test_off_card_hex_and_rgb_are_rejected():
    assert "#E7A39C" in off_card("color: #E7A39C;")
    assert "#E8D56A" in off_card("color: #E8D56A;")
    assert off_card("color: #6AAA0C; border: 1px solid rgba(255, 255, 255, 0.12);") == []
    assert off_card("color: rgba(106, 170, 12, 0.16);") == []
    assert off_card("color: rgb(255, 0, 0);") == ["rgb(255, 0, 0)"]
