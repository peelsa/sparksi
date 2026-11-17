# SPDX-License-Identifier: Apache-2.0

"""Fit check: a tailnet proxy is not a running service, and RSS replaces the estimate."""

from __future__ import annotations

import copy

from sparksi.config import CFG
from sparksi.models import running_on
from sparksi.nodes import reset_cache_for_tests
from sparksi.recipes.service import _listening_map, list_model_cards

HEAD = "head"
WORKER = "worker"
FREE = (
    "               total        used        free      shared  buff/cache   available\n"
    "Mem:     128000000000 1000000000 100000000000 0 1000000000 120000000000\n"
)


def _names():
    saved = copy.deepcopy(CFG.raw)
    source = CFG.source
    raw = copy.deepcopy(saved)
    raw["nodes"][0]["id"] = HEAD
    raw["nodes"][0]["name"] = HEAD
    raw["nodes"][0]["tailscale"] = "100.64.0.11"
    raw["nodes"][1]["id"] = WORKER
    raw["nodes"][1]["name"] = WORKER
    raw["nodes"][1]["tailscale"] = "100.64.0.12"
    CFG.__init__(raw, source=source)
    return saved, source


class Node:
    def __init__(self, ss: str, probe: dict) -> None:
        self.ss = ss
        self.probe = probe
        self.calls: list[tuple[str, object]] = []

    def capture(self, host, argv, timeout=8):
        self.calls.append((host, list(argv)))
        prog = argv[0]
        if prog == "free":
            return {"ok": True, "code": 0, "stdout": FREE, "stderr": ""}
        if prog == "ss":
            return {"ok": True, "code": 0, "stdout": self.ss, "stderr": ""}
        if prog == "nvidia-smi":
            return {"ok": True, "code": 0, "stdout": "1, 40\n", "stderr": ""}
        if prog == "docker":
            return {"ok": True, "code": 0, "stdout": "", "stderr": ""}
        if prog == "cat":
            return {"ok": True, "code": 0, "stdout": "MemTotal: 128000000 kB\nMemAvailable: 120000000 kB\n", "stderr": ""}
        return {"ok": False, "code": 1, "stdout": "", "stderr": "", "error": "missing"}

    def invoke(self, host, req, timeout=20):
        self.calls.append((host, dict(req)))
        assert req.get("op") == "probe"
        assert "start" not in req and "stop" not in req
        return self.probe


def _catalog() -> dict:
    return {
        "hosts": {
            HEAD: {"ssh": HEAD, "uma_gib": 121},
            WORKER: {"ssh": WORKER, "uma_gib": 121},
        },
        "host": {"reserved_ports": [8188]},
        "never_touch": {"do_not_stop": ["video-seat"]},
        "tenants": {
            "video": {
                "always": False,
                "measure": "rss",
                "port": 8188,
                "listen_host": "127.0.0.1",
                "cmdline_substr": ["example-service"],
            }
        },
        "library": [{
            "id": "video-seat",
            "title": "Example service",
            "placement": "single",
            "port": 8188,
            "estimate_gib": 40,
            "resident_min_gib": 15,
            "listen_host": HEAD,
            "hosts": [HEAD],
            "local": {HEAD: "/srv/example/service"},
            "identity": {"http": "/system_stats"},
            "launch": {"host": HEAD, "start": ["/srv/example/bin/start.sh"], "stop": ["/srv/example/bin/stop.sh"]},
        }],
    }


def _ss(address: str) -> str:
    return (
        "State Recv-Q Send-Q Local Address:Port Peer Address:Port\n"
        f"LISTEN 0 0 {address}:8188 0.0.0.0:*\n"
    )


def test_tailnet_proxy_is_not_running_and_charges_nothing() -> None:
    saved, source = _names()
    reset_cache_for_tests()
    try:
        node = Node(
            _ss("100.64.0.11"),
            {"ok": True, "listening": {"8188": False}, "http": {}, "containers": None},
        )
        catalog = _catalog()
        confirmed = _listening_map(catalog, node)
        assert 8188 not in confirmed.get(HEAD, set())
        assert running_on(catalog, HEAD, confirmed.get(HEAD, set()), rss_fn=lambda **_kwargs: (40.0, [{}], None)) == 0
        cards = list_model_cards(catalog, node)
        service = cards["services"][0]
        assert service["running"] is False
        assert service["status"] == "down"
        assert any(isinstance(req, dict) and req.get("op") == "probe" and 8188 in (req.get("ports") or []) for _host, req in node.calls)
        assert any(isinstance(req, dict) and (req.get("http_paths") or {}).get("8188") == "/system_stats" for _host, req in node.calls)
    finally:
        CFG.__init__(saved, source=source)
        reset_cache_for_tests()


def test_loopback_identity_is_counted_and_rss_is_charged() -> None:
    saved, source = _names()
    reset_cache_for_tests()
    try:
        node = Node(
            _ss("127.0.0.1"),
            {
                "ok": True,
                "listening": {"8188": True},
                "http": {"8188": {"status": 200, "path": "/system_stats", "comfy": True, "models": []}},
                "containers": None,
            },
        )
        catalog = _catalog()
        confirmed = _listening_map(catalog, node)
        assert 8188 in confirmed.get(HEAD, set())
        charged = running_on(catalog, HEAD, confirmed[HEAD], rss_fn=lambda **_kwargs: (22.0, [{"pid": 1}], None))
        assert charged == 22.0
        cards = list_model_cards(catalog, node)
        assert cards["services"][0]["status"] == "up"
        assert cards["services"][0]["running"] is True
    finally:
        CFG.__init__(saved, source=source)
        reset_cache_for_tests()


def test_resident_min_and_rss_tenant_replace_the_estimate() -> None:
    catalog = _catalog()
    listening = {8188}

    def rss_22(**_kwargs):
        return 22.0, [{"pid": 1}], None

    def rss_4(**_kwargs):
        return 4.0, [{"pid": 1}], None

    assert running_on(catalog, HEAD, listening, rss_fn=rss_22) == 22.0
    assert running_on(catalog, HEAD, listening, rss_fn=rss_4) == 15.0

    tenant_only = _catalog()
    tenant_only["library"][0].pop("resident_min_gib")
    assert running_on(tenant_only, HEAD, listening, rss_fn=lambda **_kwargs: (9.0, [{"pid": 1}], None)) == 9.0
