# SPDX-License-Identifier: Apache-2.0

"""Network config: labels, typed addresses and ports, and monitoring a machine that does not answer yet."""

import copy

import pytest

from sparksi import netconfig


class TestMachineLabel:
    def test_name(self):
        assert netconfig.machine_label({"name": "spark-a"}) == "spark-a"

    def test_address(self):
        assert netconfig.machine_label({"address": "192.0.2.7"}) == "192.0.2.7"

    def test_dns(self):
        assert netconfig.machine_label({"dns": "gpu-box.example.ts.net"}) == "gpu-box.example.ts.net"

    def test_first_ip(self):
        assert netconfig.machine_label({"ips": ["100.64.0.9", "100.64.0.19"]}) == "100.64.0.9"

    def test_fallback(self):
        assert netconfig.machine_label({}) == "machine"

    def test_blanks_skipped(self):
        m = {"name": "", "address": "  ", "dns": "", "ips": ["", "100.64.0.9"]}
        assert netconfig.machine_label(m) == "100.64.0.9"

    def test_blank_ips_only(self):
        assert netconfig.machine_label({"ips": ["", "  "]}) == "machine"


ACCEPT_ADDRESSES = [
    "gpu-box",
    "gpu_box.example.ts.net",
    "192.0.2.7",
    "100.64.0.9",
    " gpu-box ",
]

REJECT_ADDRESSES = [
    "",
    "8.8.8.8",
    "127.0.0.1",
    "-bad",
    "a..b",
    "1.2.3",
    "bad host",
]


class TestCheckAddress:
    @pytest.mark.parametrize("text", ACCEPT_ADDRESSES)
    def test_accept(self, text):
        assert netconfig.check_address(text) == text.strip()

    @pytest.mark.parametrize("text", REJECT_ADDRESSES)
    def test_reject(self, text):
        with pytest.raises(ValueError, match="host name or a private or tailnet IP"):
            netconfig.check_address(text)


ACCEPT_PORTS = [1, 65535, "8000"]
REJECT_PORTS = [0, 65536, "80a", "", True, None, 8000.5]


class TestCheckPort:
    @pytest.mark.parametrize("value", ACCEPT_PORTS, ids=repr)
    def test_accept(self, value):
        assert netconfig.check_port(value) == int(value)

    @pytest.mark.parametrize("value", REJECT_PORTS, ids=repr)
    def test_reject(self, value):
        with pytest.raises(ValueError, match="1 to 65535"):
            netconfig.check_port(value)


class TestEndpointFrom:
    MACHINE = {
        "name": "gpu-box",
        "dns": "gpu-box.example.ts.net",
        "ips": ["100.64.0.9"],
        "endpoints": [
            {"port": 11434, "engine": "Ollama", "url": "http://100.64.0.9:11434"}
        ],
    }

    def test_matching_endpoint(self):
        endpoint, answering = netconfig.endpoint_from(self.MACHINE)
        assert endpoint == {
            "id": "gpu-box",
            "name": "gpu-box",
            "url": "http://100.64.0.9:11434",
            "engine": "Ollama",
        }
        assert answering is True

    def test_explicit_matching_port(self):
        machine = dict(self.MACHINE, port=11434)
        endpoint, answering = netconfig.endpoint_from(machine)
        assert endpoint["url"] == "http://100.64.0.9:11434"
        assert endpoint["engine"] == "Ollama"
        assert answering is True

    def test_explicit_other_port(self):
        machine = dict(self.MACHINE, port=8000)
        endpoint, answering = netconfig.endpoint_from(machine)
        assert endpoint["url"] == "http://gpu-box.example.ts.net:8000"
        assert endpoint["engine"] == ""
        assert answering is False

    def test_no_port_raises(self):
        machine = {
            "name": "",
            "ips": ["192.0.2.7"],
            "endpoints": [{"port": 22, "engine": "open port", "url": "http://192.0.2.7:22"}],
        }
        with pytest.raises(ValueError, match="choose the port"):
            netconfig.endpoint_from(machine)

    def test_string_port(self):
        machine = {
            "name": "",
            "ips": ["192.0.2.7"],
            "port": "8000",
            "endpoints": [{"port": 22, "engine": "open port", "url": "http://192.0.2.7:22"}],
        }
        endpoint, answering = netconfig.endpoint_from(machine)
        assert endpoint["name"] == "192.0.2.7"
        assert endpoint["id"] == "192-0-2-7"
        assert endpoint["url"] == "http://192.0.2.7:8000"
        assert answering is False

    def test_address_field(self):
        endpoint, _ = netconfig.endpoint_from({"address": "198.51.100.4", "port": 8000, "name": ""})
        assert endpoint["name"] == "198.51.100.4"

    def test_public_address_rejected(self):
        with pytest.raises(ValueError):
            netconfig.endpoint_from({"address": "8.8.8.8", "port": 8000})

    def test_bad_port_rejected(self):
        with pytest.raises(ValueError):
            netconfig.endpoint_from({"address": "box", "port": 0})


class TestChange:
    RAW = {"nodes": [{"id": "a", "name": "spark-a"}, {"id": "b", "name": "spark-b"}]}

    def test_add_offline(self):
        nodes, endpoints, note = netconfig.change(
            self.RAW, "add", "monitor", {"address": "198.51.100.4", "port": 8000, "name": "box-two"}
        )
        assert endpoints == [
            {"id": "box-two", "name": "box-two", "url": "http://198.51.100.4:8000", "engine": ""}
        ]
        assert "nothing answers on this port right now" in note
        assert "offline" in note

    def test_too_many_endpoints(self):
        raw = {
            "nodes": self.RAW["nodes"],
            "endpoints": [
                {"id": "x", "name": "x", "url": "http://192.0.2.1:8000", "engine": ""},
                {"id": "y", "name": "y", "url": "http://192.0.2.2:8000", "engine": ""},
            ],
        }
        with pytest.raises(ValueError, match="up to 2 monitored hosts.*remove a monitored host first"):
            netconfig.change(
                raw, "add", "monitor", {"address": "198.51.100.4", "port": 8000, "name": "box-two"}
            )


class TestNodeFrom:
    def test_ip_only(self):
        node = netconfig.node_from({"name": "", "ips": ["100.64.0.5"]})
        assert node["name"] == "100.64.0.5"
        assert node["id"] == "100-64-0-5"
        assert node["ssh"] == "100.64.0.5"
        assert node["tailscale"] == "100.64.0.5"


def _raw():
    return {"nodes": [{"id": "a", "name": "spark-a"}, {"id": "b", "name": "spark-b"}]}


def _full():
    raw = _raw()
    raw["endpoints"] = [
        {"id": "x", "name": "x", "url": "http://192.0.2.5:8000", "engine": ""},
        {"id": "y", "name": "y", "url": "http://192.0.2.6:8000", "engine": ""},
    ]
    return raw


def test_change_rejects_public_address():
    full = _full()
    with pytest.raises(ValueError) as exc:
        netconfig.change(full, "add", "monitor", {"address": "8.8.8.8", "port": 8000})
    assert "private or tailnet" in str(exc.value)


def test_change_rejects_bad_port_before_limit():
    full = _full()
    with pytest.raises(ValueError) as exc:
        netconfig.change(full, "add", "monitor", {"address": "box", "port": 0})
    assert "1 to 65535" in str(exc.value)


def test_change_reject_when_full():
    full = _full()
    with pytest.raises(ValueError) as exc:
        netconfig.change(
            full, "add", "monitor", {"address": "198.51.100.9", "port": 8000, "name": "z"}
        )
    assert "up to 2 monitored hosts" in str(exc.value)


def test_change_duplicate_name_exact():
    raw = _raw()
    raw["endpoints"] = [
        {"id": "x", "name": "x", "url": "http://192.0.2.5:8000", "engine": ""}
    ]
    with pytest.raises(ValueError) as exc:
        netconfig.change(
            raw, "add", "worker", {"address": "198.51.100.9", "port": 8000, "name": "x"}
        )
    assert str(exc.value) == (
        "a machine named x is already in the setup; pick another name"
    )


def test_change_duplicate_name_case_insensitive():
    raw = _raw()
    raw["endpoints"] = [
        {"id": "x", "name": "x", "url": "http://192.0.2.5:8000", "engine": ""}
    ]
    with pytest.raises(ValueError) as exc:
        netconfig.change(
            raw, "add", "worker", {"address": "198.51.100.9", "port": 8000, "name": "X"}
        )
    assert str(exc.value) == (
        "a machine named x is already in the setup; pick another name"
    )


def test_change_discover_duplicate_name():
    raw = _raw()
    with pytest.raises(ValueError) as exc:
        netconfig.change(
            raw,
            "add",
            "worker",
            {
                "name": "spark-a",
                "dns": "spark-a.example.ts.net",
                "ips": ["100.64.0.11"],
                "endpoints": [],
            },
        )
    assert str(exc.value).startswith("a machine named spark-a is already in the setup")


def test_change_does_not_mutate_input_on_refusals():
    raw = _raw()
    raw["endpoints"] = [
        {"id": "x", "name": "x", "url": "http://192.0.2.5:8000", "engine": ""},
        {"id": "y", "name": "y", "url": "http://192.0.2.6:8000", "engine": ""},
    ]
    snapshot = copy.deepcopy(raw)
    cases = [
        ("monitor", {"address": "8.8.8.8", "port": 8000}),
        ("monitor", {"address": "box", "port": 0}),
        ("monitor", {"address": "198.51.100.9", "port": 8000, "name": "z"}),
        ("worker", {"address": "198.51.100.9", "port": 8000, "name": "x"}),
        (
            "worker",
            {
                "name": "spark-a",
                "dns": "spark-a.example.ts.net",
                "ips": ["100.64.0.11"],
                "endpoints": [],
            },
        ),
    ]
    for role, payload in cases:
        with pytest.raises(ValueError):
            netconfig.change(raw, "add", role, payload)
    assert raw == snapshot
