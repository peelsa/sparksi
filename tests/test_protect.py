# SPDX-License-Identifier: Apache-2.0

"""The video-service protect preset: off by default, on when named, on by itself for a legacy install."""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path

import pytest

from sparksi import agent, config as config_mod, protect
from sparksi.apply import is_protected
from sparksi.catalog import load_catalog
from sparksi.config import Config
from sparksi.launch import plan_stop
from sparksi.library import probe_ports
from sparksi.models import row_is_protected
from sparksi.monitor import peer
from sparksi.never import Refused, assert_serve_port
from sparksi.recipes.validate import check_port

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
CURRENT_CONFIG = ROOT / "config" / "sparksi.example.toml"
CURRENT_CATALOG = ROOT / "config" / "catalog.example.yaml"
LEGACY_CONFIG = FIX / "sparksi.legacy.toml"
LEGACY_CATALOG = FIX / "catalog.legacy.yaml"
VIDEO_ROW = {"id": "video-service", "kind": "video_service", "port": 8188, "role": "video"}


def _raw(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def site(monkeypatch, tmp_path):
    """Point the process at one config dict and one catalog file, with no preset env."""
    monkeypatch.delenv("SPARKSI_PROTECT", raising=False)
    for name in protect.LEGACY_MARKERS["env"]:
        monkeypatch.delenv(name, raising=False)

    def use(raw: dict, catalog: Path = CURRENT_CATALOG) -> None:
        monkeypatch.setattr(config_mod, "CFG", Config(raw))
        monkeypatch.setattr(peer, "CFG", config_mod.CFG)
        monkeypatch.setenv("SPARKSI_CATALOG", str(catalog))

    return use


def _new_catalog_without_markers(tmp_path: Path) -> Path:
    # The current example has no legacy markers; check that stays true.
    assert protect.catalog_markers(load_catalog(CURRENT_CATALOG)) == []
    return CURRENT_CATALOG


def test_fresh_install_has_no_protection(site, tmp_path):
    """New-style config, preset off: no reserved 8188, no video row, no process names."""
    site(_raw(CURRENT_CONFIG), _new_catalog_without_markers(tmp_path))
    assert protect.active() == frozenset()
    assert protect.auto_reasons() == []
    assert not protect.is_off_limits(8188)
    assert protect.agent_values() == {"off_limits_ports": [], "video_process_substrings": []}
    assert not any("off limits" in text for text in check_port(8188))
    assert_serve_port(8188)
    assert 8188 not in probe_ports()
    assert not is_protected({"id": "brain-a", "port": 8188, "kind": "vllm"})
    assert not row_is_protected({"id": "brain-a", "port": 8188})
    assert plan_stop(dict(VIDEO_ROW, launch={"stop": ["/srv/example/stop.sh"]}), lock_on=False)["verdict"] == "READY"
    script = peer.remote_script()
    assert "omfy" not in script and ":8188" not in script


def test_explicit_preset_protects(site):
    """[protect] presets = ["video-service"] turns on the same protection a legacy install gets."""
    raw = _raw(CURRENT_CONFIG)
    raw["protect"] = {"presets": ["video-service"]}
    site(raw)
    assert protect.explicit() == {"video-service"} and protect.auto_reasons() == []
    assert protect.is_off_limits(8188) and protect.is_off_limits("8188")
    assert check_port(8188)[0].startswith("port 8188")
    with pytest.raises(Refused):
        assert_serve_port(8188)
    assert is_protected({"id": "brain-a", "port": 8188, "kind": "vllm"})
    assert is_protected(VIDEO_ROW) and row_is_protected(VIDEO_ROW)
    assert plan_stop(VIDEO_ROW, lock_on=False)["verdict"] == "REFUSE"


def test_env_turns_preset_on(site, monkeypatch):
    """SPARKSI_PROTECT=video-service is the same as the file setting."""
    site(_raw(CURRENT_CONFIG))
    monkeypatch.setenv("SPARKSI_PROTECT", "other, video-service")
    assert protect.video_on() and protect.unknown_presets() == ["other"]


def test_legacy_install_auto_enables(site):
    """The old-style config and catalog (the shape existing installs use) turn the preset on."""
    site(_raw(LEGACY_CONFIG), LEGACY_CATALOG)
    reasons = protect.auto_reasons()
    assert "config [paths] comfy_root" in reasons and "catalog kind comfy" in reasons
    assert protect.video_on() and protect.off_limits_ports() == {8188}


@pytest.mark.parametrize(
    "raw, reason",
    [
        ({"paths": {"comfy_root": "/srv/v"}}, "config [paths] comfy_root"),
        ({"paths": {"model_truth": "/srv/m.yaml"}}, "config [paths] model_truth"),
        ({"services": {"video-h3": {"user_controllable": True}}}, "config [services.video-h3]"),
    ],
)
def test_each_config_marker(site, raw, reason):
    """Each legacy config key alone turns the preset on."""
    site(raw)
    assert protect.auto_reasons() == [reason] and protect.video_on()


@pytest.mark.parametrize("name", ["SPARKSI_MODEL_TRUTH", "DGX_MEM_MODEL_TRUTH", "SPARKSI_COMFY_STATE"])
def test_each_env_marker(site, monkeypatch, name):
    """Each legacy env var alone turns the preset on."""
    site({})
    monkeypatch.setenv(name, "/srv/x")
    assert protect.auto_reasons() == [f"env {name}"] and protect.video_on()


@pytest.mark.parametrize(
    "data, reason",
    [
        ({"residents": [{"id": "x", "kind": "comfy"}]}, "catalog kind comfy"),
        ({"residents": [{"id": "comfy", "kind": "service"}]}, "catalog id comfy"),
        ({"library": [{"id": "comfy"}]}, "catalog id comfy"),
        ({"never_touch": {"do_not_stop": ["comfyui"]}}, "catalog never_touch.do_not_stop comfyui"),
        ({"tenants": {"comfy": {}}}, "catalog tenant comfy"),
        ({"seats": {"video": "comfy"}}, "catalog seats.video comfy"),
    ],
)
def test_each_catalog_marker(site, tmp_path, data, reason):
    """Each legacy catalog marker alone turns the preset on."""
    import json

    path = tmp_path / "catalog.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")  # JSON is valid YAML
    site({}, path)
    assert protect.auto_reasons() == [reason] and protect.video_on()


def test_log_auto(site, caplog):
    """serve logs one warning naming the reasons when the preset turned on by itself."""
    log = logging.getLogger("sparksi.test-protect")
    site(_raw(LEGACY_CONFIG), LEGACY_CATALOG)
    with caplog.at_level(logging.WARNING, logger=log.name):
        protect.log_auto(log)
    assert len(caplog.records) == 1
    text = caplog.records[0].getMessage()
    assert (
        "auto-enabled" in text
        and "config [paths] comfy_root" in text
        and '[protect] presets = ["video-service"]' in text
    )
    caplog.clear()
    raw = _raw(LEGACY_CONFIG)
    raw["protect"] = {"presets": ["video-service"]}
    site(raw, LEGACY_CATALOG)
    with caplog.at_level(logging.WARNING, logger=log.name):
        protect.log_auto(log)
    assert caplog.records == []


def test_agent_follows_controller_values():
    """The node agent takes the preset values from the controller, empty lists included."""
    saved = dict(agent._CFG)
    try:
        agent.configure({"off_limits_ports": [], "video_process_substrings": []})
        assert not agent._off_limits(8188) and agent._unmanaged_name_ok("comfyui")
        agent.configure({"off_limits_ports": [8188], "video_process_substrings": ["comfy"]})
        assert agent._off_limits(8188) and not agent._unmanaged_name_ok("ComfyUI")
    finally:
        agent._CFG.clear()
        agent._CFG.update(saved)


def test_announcer_preset_switch(monkeypatch, tmp_path):
    """The announcer reports the video service only with the preset or a legacy marker."""
    from sparksi import announce

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("SPARKSI_PROTECT", raising=False)
    monkeypatch.delenv("SPARKSI_COMFY_STATE", raising=False)
    monkeypatch.delenv("DGX_ANNOUNCE_HOST", raising=False)
    monkeypatch.setattr(announce, "_own_unit_text", lambda: "0::/user.slice/app.slice/sparksi-announce.service\n")
    assert not announce.video_on()
    monkeypatch.setenv("SPARKSI_PROTECT", "video-service")
    assert announce.video_on()
    monkeypatch.setenv("SPARKSI_PROTECT", "")
    assert not announce.video_on()
    monkeypatch.setenv("SPARKSI_COMFY_STATE", str(tmp_path))
    assert announce.video_on()


@pytest.mark.parametrize("marker", ["env", "dir", "unit"])
def test_announcer_legacy_markers(monkeypatch, tmp_path, marker):
    """Each node-side legacy marker turns the announcer preset on by itself."""
    from sparksi import announce

    monkeypatch.setenv("HOME", str(tmp_path))
    for name in ("SPARKSI_PROTECT", "SPARKSI_COMFY_STATE", "DGX_ANNOUNCE_HOST"):
        monkeypatch.delenv(name, raising=False)
    unit = "0::/user.slice/app.slice/sparksi-announce.service\n"
    if marker == "env":
        monkeypatch.setenv("DGX_ANNOUNCE_HOST", "a")
    elif marker == "dir":
        (tmp_path / "services" / "comfyui").mkdir(parents=True)
    else:
        unit = "0::/user.slice/app.slice/dgx-announce.service\n"
    monkeypatch.setattr(announce, "_own_unit_text", lambda: unit)
    assert announce.video_on()


def test_announce_unit_carries_controller_presets(site):
    """The rendered announcer unit names the controller's active presets, or none."""
    from sparksi.cli import render_unit

    site(_raw(LEGACY_CONFIG), LEGACY_CATALOG)
    text = render_unit("announce", prefix="", node_id="a", controller="100.64.0.1:7501")
    assert "Environment=SPARKSI_PROTECT=video-service\n" in text
    site(_raw(CURRENT_CONFIG), CURRENT_CATALOG)
    text = render_unit("announce", prefix="", node_id="a", controller="100.64.0.1:7501")
    assert "Environment=SPARKSI_PROTECT=\n" in text


def _protection(catalog: dict) -> list:
    out = []
    for row in catalog["residents"]:
        out.append([row["id"], is_protected(row), row_is_protected(row, catalog), plan_stop(row, lock_on=False)])
    return out


def test_auto_enabled_legacy_matches_explicit(site, monkeypatch):
    """A legacy install with the preset auto-enabled is protected exactly as with it named."""
    site(_raw(LEGACY_CONFIG), LEGACY_CATALOG)
    cat = load_catalog(LEGACY_CATALOG)
    assert protect.explicit() == frozenset() and protect.video_on()
    auto = _protection(cat)
    monkeypatch.setenv("SPARKSI_PROTECT", "video-service")
    assert auto == _protection(cat)
