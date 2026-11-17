# SPDX-License-Identifier: Apache-2.0

"""Old names still work: config keys, env vars, catalog ids and kinds, API fields.

``tests/fixtures/catalog.legacy.yaml`` and ``sparksi.legacy.toml`` keep the
shape existing installs use. Every check here runs the old shape and the
current example side by side and expects the same protection and plans.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from sparksi import agent, announce, compat, protect
from sparksi.apply import build_plan, is_protected
from sparksi.catalog import load_catalog
from sparksi.config import Config
from sparksi.launch import plan_stop
from sparksi.models import row_is_protected, row_is_service
from sparksi.monitor import peer
from sparksi.wanted import resolve_spec

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"
LEGACY_CATALOG = FIX / "catalog.legacy.yaml"
CURRENT_CATALOG = ROOT / "config" / "catalog.example.yaml"
LEGACY_CONFIG = FIX / "sparksi.legacy.toml"
CURRENT_CONFIG = ROOT / "config" / "sparksi.example.toml"

# Current name -> the name the legacy fixture uses for the same thing.
_TO_LEGACY = (
    ("video-service", "comfy"),
    ("video_service", "comfy"),
)


def _as_legacy(obj) -> str:
    text = json.dumps(obj, sort_keys=True, default=str)
    for new, old in _TO_LEGACY:
        text = text.replace(new, old)
    return text


# --- compat table ------------------------------------------------------------


def test_apply_legacy_paths_copies_when_absent():
    """A legacy [paths] key fills the new key when the new key is absent."""
    raw = {"paths": {"comfy_root": "/srv/v", "model_truth": "/srv/m.yaml"}}
    out = compat.apply_legacy_paths(raw)
    assert out["paths"]["video_service_root"] == "/srv/v"
    assert out["paths"]["model_inventory"] == "/srv/m.yaml"


def test_apply_legacy_paths_new_key_wins():
    """The new key wins when both are set."""
    raw = {"paths": {"comfy_root": "/srv/old", "video_service_root": "/srv/new"}}
    assert compat.apply_legacy_paths(raw)["paths"]["video_service_root"] == "/srv/new"


def test_legacy_env_reads_first_old_name():
    """legacy_env returns the first legacy env value that is set, else None."""
    env = {"SPARKSI_MODEL_TRUTH": "a", "DGX_MEM_MODEL_TRUTH": "b"}
    assert compat.legacy_env("SPARKSI_MODEL_INVENTORY", env) == "a"
    assert compat.legacy_env("SPARKSI_MODEL_INVENTORY", {"DGX_MEM_MODEL_TRUTH": "b"}) == "b"
    assert compat.legacy_env("SPARKSI_VIDEO_STATE", {}) is None


def test_with_legacy_fields_emits_both():
    """API payloads carry the new field names and the old ones."""
    out = compat.with_legacy_fields({"video_service_live": True, "diff_vs_model_inventory": {"drift": 0}})
    assert out["video_service_live"] is True and out["comfy_live"] is True
    assert out["diff_vs_model_truth"] == {"drift": 0}


def test_field_falls_back_to_legacy():
    """field() reads the new name, else the old one (an older server)."""
    assert compat.field({"diff_vs_model_inventory": 1, "diff_vs_model_truth": 2}, "diff_vs_model_inventory") == 1
    assert compat.field({"diff_vs_model_truth": 2}, "diff_vs_model_inventory") == 2


def test_agent_video_flag_both_shapes():
    """The agent HTTP flag is read under its new and its old name."""
    assert compat.agent_video_flag({"video_service": True}) is True
    assert compat.agent_video_flag({"comfy": True}) is True
    assert compat.agent_video_flag({}) is False


@pytest.mark.parametrize("rid", ["comfy", "comfyui", "h3", "minimax-h3", "video-service"])
def test_is_protected_id_case_insensitive(rid):
    """Built-in protected ids match in any case."""
    assert protect.is_protected_id(rid)
    assert protect.is_protected_id(rid.upper())


def test_node_copies_match_compat():
    """agent.py and announce.py are stdlib-only; their copies equal sparksi.compat."""
    assert agent.AGENT_VIDEO_KEY == compat.AGENT_VIDEO_KEY
    assert agent._PRESET_KEYS == tuple(protect.agent_values())
    assert announce.VIDEO_PACKET_KEY == compat.VIDEO_PACKET_KEY
    assert announce.LEGACY_VIDEO_PACKET_KEY == compat.LEGACY_VIDEO_PACKET_KEY
    assert announce.VIDEO_PROCESS_SUBSTRINGS == protect.VIDEO.process_substrings
    assert announce.VIDEO_PRESET == protect.VIDEO_SERVICE
    assert announce.VIDEO_STATE_ENV == ("SPARKSI_VIDEO_STATE",) + compat.LEGACY_ENV_NAMES["SPARKSI_VIDEO_STATE"]
    assert announce.VIDEO_SERVICE_ROOT_FALLBACKS == compat.VIDEO_SERVICE_ROOT_FALLBACKS
    assert {announce.VIDEO_SERVICE_PORT} == set(protect.VIDEO.ports)


# --- announce packet ---------------------------------------------------------


def test_peer_dump_reads_old_and_new_packet_key():
    """A controller reads the video state from an old announcer and a new one."""
    old = announce.to_peer({"h": "a", "mem": [1, 1], "comfy": [1, "video"]})
    new = announce.to_peer({"h": "a", "mem": [1, 1], "video": [1, "video"]})
    assert old["video_service"] == new["video_service"] == {"mode": "video", "rss_kb": 0, "listening": True}


def test_video_state_env_old_name(monkeypatch, tmp_path):
    """The announcer still honors the old state-dir env var."""
    monkeypatch.delenv("SPARKSI_VIDEO_STATE", raising=False)
    monkeypatch.setenv("SPARKSI_COMFY_STATE", str(tmp_path))
    assert announce.video_mode_file() == tmp_path / "mode"
    monkeypatch.setenv("SPARKSI_VIDEO_STATE", str(tmp_path / "new"))
    assert announce.video_mode_file() == tmp_path / "new" / "mode"


# --- config ------------------------------------------------------------------


def _config(path: Path) -> Config:
    return Config(tomllib.loads(path.read_text(encoding="utf-8")), source=path)


def test_legacy_config_keys_read():
    """A config with the old [paths] keys resolves the same paths as the current one."""
    old, new = _config(LEGACY_CONFIG), _config(CURRENT_CONFIG)
    assert "comfy_root" in old.raw["paths"] and "comfy_root" not in tomllib.loads(CURRENT_CONFIG.read_text())["paths"]
    for key in ("video_service_root", "model_inventory"):
        assert old.path(key) == new.path(key)
    assert old.agent_config() == new.agent_config()


def test_model_inventory_env_old_names(monkeypatch, tmp_path):
    """SPARKSI_MODEL_TRUTH and DGX_MEM_MODEL_TRUTH still point at the inventory file."""
    from sparksi.catalog import model_inventory_path

    monkeypatch.delenv("SPARKSI_MODEL_INVENTORY", raising=False)
    monkeypatch.delenv("SPARKSI_MODEL_TRUTH", raising=False)
    monkeypatch.setenv("DGX_MEM_MODEL_TRUTH", str(tmp_path / "b.yaml"))
    assert Path(model_inventory_path()) == tmp_path / "b.yaml"
    monkeypatch.setenv("SPARKSI_MODEL_TRUTH", str(tmp_path / "a.yaml"))
    assert Path(model_inventory_path()) == tmp_path / "a.yaml"
    monkeypatch.setenv("SPARKSI_MODEL_INVENTORY", str(tmp_path / "n.yaml"))
    assert Path(model_inventory_path()) == tmp_path / "n.yaml"


def test_peer_script_same_for_old_and_new_config(monkeypatch):
    """The read-only peer dump script is the same for either config shape."""
    from sparksi import config as config_mod

    out = []
    for path in (LEGACY_CONFIG, CURRENT_CONFIG):
        monkeypatch.setattr(config_mod, "CFG", _config(path))
        monkeypatch.setattr(peer, "CFG", _config(path))
        out.append(peer.remote_script())
    assert out[0] == out[1]


# --- catalog: protection, plans ---------------------------------------------


def _pairs():
    old, new = load_catalog(LEGACY_CATALOG), load_catalog(CURRENT_CATALOG)
    assert [r["id"] for r in old["residents"]].count("comfy") == 1
    assert [r["id"] for r in new["residents"]].count("video-service") == 1
    assert len(old["residents"]) == len(new["residents"])
    return old, new


def test_protection_identical_old_and_new_catalog():
    """Every resident row is protected (or not) the same way in both shapes."""
    old, new = _pairs()
    for a, b in zip(old["residents"], new["residents"]):
        assert is_protected(a) == is_protected(b), a["id"]
        assert row_is_protected(a, old) == row_is_protected(b, new), a["id"]
        assert row_is_service(a) == row_is_service(b), a["id"]
        assert _as_legacy(plan_stop(a, lock_on=False)) == _as_legacy(plan_stop(b, lock_on=False)), a["id"]
    vid_old = next(r for r in old["residents"] if r["id"] == "comfy")
    vid_new = next(r for r in new["residents"] if r["id"] == "video-service")
    for row in (vid_old, vid_new):
        assert is_protected(row)
        assert row_is_protected(row)
        assert plan_stop(row, lock_on=False)["verdict"] == "REFUSE"


def _residents(catalog, live):
    out = []
    for row in catalog["residents"]:
        r = dict(row)
        r["resident"] = r["id"] in live
        out.append(r)
    return out


def _dry(verdict):
    return {"verdict": verdict, "recipe": {}, "floor": {"complete": True, "total_gib": 10.0, "parts": []},
            "margin_gib": 4.0, "needed_gib": 117.0, "mem_total_gib": 121.0, "reasons": ["fixture"]}


@pytest.mark.parametrize("spec", ["video", "trio", "coresident-3", "brain-a,vision-a"])
@pytest.mark.parametrize("live_video", [True, False])
@pytest.mark.parametrize("verdict", ["OK", "NO", "UNCERTAIN"])
def test_plans_identical_old_and_new_catalog(spec, live_video, verdict):
    """Apply plans are the same for both shapes, with the video service live or not."""
    old, new = _pairs()
    plans = []
    for cat, vid in ((old, "comfy"), (new, "video-service")):
        live = {vid, "counsel-a"} if live_video else {"counsel-a"}
        plans.append(_as_legacy(build_plan(cat, resolve_spec(cat, spec), _residents(cat, live), _dry(verdict), lock_on=False)))
    assert plans[0] == plans[1]
