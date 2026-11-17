# SPDX-License-Identifier: Apache-2.0

"""Legacy-name mirrors kept during the rename: packet, /api/monitor, probe and agent."""

import json

from sparksi import agent, announce, compat, probe
from sparksi.monitor import models as monitor_models


def _stub_collectors(monkeypatch, on):
    """Stub video state and collapse other collectors to defaults."""
    monkeypatch.setattr(announce, "_video_state", lambda: [1, "video"])
    monkeypatch.setattr(announce, "video_on", lambda: on)

    def fake_call(fn, default):
        return fn() if fn is announce._video_state else default

    monkeypatch.setattr(announce, "_call", fake_call)


def _stub_probe(monkeypatch, body):
    """Stub port and http helpers for probe tests."""
    monkeypatch.setattr(probe, "port_listening", lambda port, host: True)
    monkeypatch.setattr(probe, "http_get", lambda url: (200, body))


def test_packet_sends_both_video_keys(monkeypatch):
    """Video on: packet carries both the new and legacy video keys."""
    _stub_collectors(monkeypatch, True)
    pkt = announce.sample("host-under-test")
    assert announce.VIDEO_PACKET_KEY in pkt, "new video key missing from packet"
    assert announce.LEGACY_VIDEO_PACKET_KEY in pkt, "legacy video key missing from packet"
    assert pkt[announce.VIDEO_PACKET_KEY] == [1, "video"]
    assert pkt[announce.LEGACY_VIDEO_PACKET_KEY] == [1, "video"]


def test_packet_omits_video_keys_when_off(monkeypatch):
    """Video off: neither video key appears in the packet."""
    _stub_collectors(monkeypatch, False)
    pkt = announce.sample("host-under-test")
    assert announce.VIDEO_PACKET_KEY not in pkt, "new video key present while video off"
    assert announce.LEGACY_VIDEO_PACKET_KEY not in pkt, "legacy video key present while video off"


def test_encode_keeps_both_video_keys(monkeypatch):
    """encode() preserves both video keys in the JSON output."""
    _stub_collectors(monkeypatch, True)
    pkt = announce.sample("host-under-test")
    decoded = json.loads(announce.encode(pkt))
    assert decoded[announce.VIDEO_PACKET_KEY] == [1, "video"]
    assert decoded[announce.LEGACY_VIDEO_PACKET_KEY] == [1, "video"]


def test_to_peer_video_service_shape():
    """to_peer exposes video_service and a separate legacy copy."""
    pkt = {"h": "a", "mem": [1, 1], announce.VIDEO_PACKET_KEY: [1, "video"]}
    out = announce.to_peer(pkt)
    assert out["video_service"] == {"mode": "video", "rss_kb": 0, "listening": True}
    assert out[compat.LEGACY_MONITOR_VIDEO_KEY] == {"mode": "video", "rss_kb": 0, "listening": True}
    assert out["video_service"] is not out[compat.LEGACY_MONITOR_VIDEO_KEY]
    out["video_service"]["rss_kb"] = 99
    assert out[compat.LEGACY_MONITOR_VIDEO_KEY]["rss_kb"] == 0


def test_to_peer_accepts_legacy_key_only():
    """to_peer yields the same result from a legacy-only packet."""
    new_pkt = {"h": "a", "mem": [1, 1], announce.VIDEO_PACKET_KEY: [1, "video"]}
    legacy_pkt = {"h": "a", "mem": [1, 1], announce.LEGACY_VIDEO_PACKET_KEY: [1, "video"]}
    assert announce.to_peer(legacy_pkt) == announce.to_peer(new_pkt)


def test_video_service_from_peer_row():
    """video_service_from_peer builds a legacy-flagged row under protect mode."""
    peer_dict = {"video_service": {"mode": "video", "rss_kb": 1048576, "listening": True}}
    row = monitor_models.video_service_from_peer(peer_dict)
    assert row is not None
    assert row["kind"] == compat.VIDEO_SERVICE_KIND
    assert row["video_service"] is True
    assert row["legacy_kind"] == "comfy"
    assert row[compat.LEGACY_ROW_FLAG] is True
    assert row["port"] == 8188


def test_video_service_row_satisfies_page_predicates():
    """Row matches both the old and the new page predicates."""
    peer_dict = {"video_service": {"mode": "video", "rss_kb": 1048576, "listening": True}}
    row = monitor_models.video_service_from_peer(peer_dict)
    assert row.get("kind") == "comfy" or row.get("port") == 8188
    assert row.get("video_service") is True


def test_probe_http_identity_video(monkeypatch):
    """probe reports both identities for the video stats path."""
    _stub_probe(monkeypatch, '{"system": {}}')
    result = probe.probe_http_identity("host-under-test", 8188, "/system_stats")
    assert result["identity"] == compat.VIDEO_HTTP_LABEL
    assert result["legacy_identity"] == compat.LEGACY_VIDEO_HTTP_LABEL


def test_probe_http_identity_models(monkeypatch):
    """probe reports the model id and no legacy identity for the models path."""
    _stub_probe(monkeypatch, '{"data":[{"id":"m"}]}')
    result = probe.probe_http_identity("host-under-test", 8188, "/v1/models")
    assert result["identity"] == "m"
    assert "legacy_identity" not in result


def test_node_key_constants_match_compat():
    """Node modules mirror the compat key constants."""
    assert announce.LEGACY_MONITOR_VIDEO_KEY == compat.LEGACY_MONITOR_VIDEO_KEY
    assert agent.LEGACY_AGENT_VIDEO_KEY == compat.LEGACY_AGENT_VIDEO_KEY


class _Answer:
    status = 200

    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self, _limit: int) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        return None


def test_agent_reports_both_video_flags(monkeypatch):
    """The node agent sends the new flag and the legacy one an older controller reads."""
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: _Answer(b'{"system": {}}'))
    out = agent._http_identity(9000, "/system_stats")
    assert out[agent.AGENT_VIDEO_KEY] is True
    assert out[agent.LEGACY_AGENT_VIDEO_KEY] is True
    assert compat.agent_video_flag(out) is True
