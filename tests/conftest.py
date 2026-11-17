# SPDX-License-Identifier: Apache-2.0

"""Keep the suite off a real Sparksi config in the operator's home directory.

``CFG`` is loaded on first import. These variables are set before that import.
A test that needs a site file sets ``SPARKSI_CONFIG`` itself.
"""

from __future__ import annotations

import os
import shutil

import pytest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HOME = Path("/tmp/sparksi-pytest-home")
_CONFIG = _HOME / ".config" / "sparksi"
_CONFIG.mkdir(parents=True, exist_ok=True)
(_HOME / ".local" / "state").mkdir(parents=True, exist_ok=True)
(_HOME / "run").mkdir(parents=True, exist_ok=True)

_example = _ROOT / "config" / "catalog.example.yaml"
_catalog = _CONFIG / "catalog.yaml"
if _example.is_file():
    shutil.copyfile(_example, _catalog)

os.environ["HOME"] = str(_HOME)
os.environ["XDG_CONFIG_HOME"] = str(_HOME / ".config")
os.environ["XDG_RUNTIME_DIR"] = str(_HOME / "run")
os.environ.pop("SPARKSI_CONFIG", None)
os.environ.pop("SPARKSI_CATALOG", None)
os.environ.pop("SPARKSI_RECIPES_DIR", None)
os.environ.pop("DGX_MEM_CONFIG", None)
os.environ.pop("DGX_MEM_CATALOG", None)
# Most tests exercise the video-service protect preset; tests of the off default unset this.
os.environ["SPARKSI_PROTECT"] = "video-service"
for _legacy in ("SPARKSI_MODEL_TRUTH", "DGX_MEM_MODEL_TRUTH", "SPARKSI_COMFY_STATE"):
    os.environ.pop(_legacy, None)


@pytest.fixture(autouse=True)
def _no_ssh_masters(monkeypatch):
    """No real background ssh master and no ready-run pacing unless a test turns them on."""
    from sparksi.config import CFG

    monkeypatch.setitem(CFG.controller, "ssh_reuse", False)
    monkeypatch.setitem(CFG.controller, "ready_check_s", 0)  # test_ready_check_pace sets its own


@pytest.fixture(autouse=True)
def _fresh_background_reads():
    """Each test starts with no cached node reads and no remembered unreachable hosts."""
    import sparksi.remote as remote_mod
    import sparksi.serve as serve_mod

    serve_mod.READS.clear()
    serve_mod.SLOW_READS.clear()
    serve_mod.PROBES.clear()
    remote_mod.forget_failures()
    yield
    serve_mod.PROBES.clear()
    serve_mod.READS.clear()
    serve_mod.SLOW_READS.clear()
    remote_mod.forget_failures()


_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(params=["legacy", "current"])
def video_catalog(request, monkeypatch):
    """Run a test on the old-style catalog and on the current example.

    ``legacy`` is the catalog shape existing installs still use (video id and
    kind under their old names). Both must give the same protection and plans.
    Returns the video service catalog id for that catalog.
    """
    if request.param == "legacy":
        monkeypatch.setenv("SPARKSI_CATALOG", str(_FIXTURES / "catalog.legacy.yaml"))
        return "comfy"
    monkeypatch.setenv("SPARKSI_CATALOG", str(_ROOT / "config" / "catalog.example.yaml"))
    return "video-service"
