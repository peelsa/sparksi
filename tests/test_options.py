# SPDX-License-Identifier: Apache-2.0

"""Options page: each setting's checks, the one-line byte-preserving save, and the HTTP API (no model lock)."""

import copy
import difflib
import http.client
import json
import shutil
import threading
import tomllib
from pathlib import Path

import pytest

from sparksi import options
from sparksi import serve as serve_mod
from sparksi.lock import Lock
from sparksi.serve import STATE, DualServer, Handler

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- helpers

def ask(port, method, path, body=None, origin=True):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Content-Type": "application/json"}
    if origin is True:
        headers["Origin"] = "http://127.0.0.1:%d" % port
    elif origin == "evil":
        headers["Origin"] = "http://evil.example"
    payload = None
    if body is not None:
        payload = json.dumps(body)
    conn.request(method, path, body=payload, headers=headers)
    resp = conn.getresponse()
    status = resp.status
    raw = resp.read()
    conn.close()
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except Exception:
        parsed = {}
    return status, parsed


def ask_raw(port, method, path, body, content_type):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {
        "Content-Type": content_type,
        "Origin": "http://127.0.0.1:%d" % port,
    }
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    status = resp.status
    raw = resp.read()
    conn.close()
    return status, raw


@pytest.fixture
def server(tmp_path, monkeypatch):
    cfg = tmp_path / "sparksi.toml"
    shutil.copyfile(ROOT / "config" / "sparksi.example.toml", cfg)
    cfg.chmod(0o600)
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    monkeypatch.setenv("SPARKSI_CONFIG", str(cfg))
    monkeypatch.setattr(serve_mod.HOSTS, "names", lambda: {"localhost", "127.0.0.1"})
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release()
    httpd = DualServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield httpd.server_address[1], cfg
    httpd.shutdown()
    httpd.server_close()
    t.join(timeout=5)


def load_toml(path):
    import tomllib
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def changed_lines(old_text, new_text):
    diff = list(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            n=0,
        )
    )
    return [ln for ln in diff if ln.startswith(("+", "-")) and not ln.startswith(("+++", "---"))]


KEYS = [
    "archive.root",
    "archive.label",
    "archive.node",
    "controller.monitor_upstream",
    "controller.access_log",
    "controller.metrics_url",
]


# ---------------------------------------------------------------- 1. check()

class TestCheck:
    def raw(self):
        return load_toml(ROOT / "config" / "sparksi.example.toml")

    def test_access_log_ok(self):
        assert options.check("controller.access_log", "all", self.raw()) == "all"

    def test_access_log_bad_choice(self):
        with pytest.raises(ValueError):
            options.check("controller.access_log", "verbose", self.raw())

    def test_label_ok(self):
        assert options.check("archive.label", "Big storage", self.raw()) == "Big storage"

    @pytest.mark.parametrize("value", ["x" * 101, "a\u0007b"])
    def test_label_refused(self, value):
        with pytest.raises(ValueError):
            options.check("archive.label", value, self.raw())

    @pytest.mark.parametrize("value, want", [("b", "b"), ("", "")])
    def test_node_ok(self, value, want):
        assert options.check("archive.node", value, self.raw()) == want

    def test_node_unknown(self):
        with pytest.raises(ValueError):
            options.check("archive.node", "zz", self.raw())

    def test_monitor_empty_ok(self):
        assert options.check("controller.monitor_upstream", "", self.raw()) == ""

    def test_monitor_no_trailing_slash(self):
        # Another controller on this machine (the file's own ui_port is 7500, so use another port).
        got = options.check("controller.monitor_upstream", "http://127.0.0.1:7600", self.raw())
        assert got == "http://127.0.0.1:7600"

    def test_monitor_trailing_slash_normalised(self):
        got = options.check(
            "controller.monitor_upstream", "http://100.64.0.9:7500/", self.raw()
        )
        assert got == "http://100.64.0.9:7500"

    @pytest.mark.parametrize(
        "value",
        [
            "http://192.0.2.5:7500",
            "https://127.0.0.1:7500",
            "http://u:p@127.0.0.1:7500",
            "http://127.0.0.1:7500/x",
        ],
    )
    def test_monitor_refused(self, value):
        with pytest.raises(ValueError):
            options.check("controller.monitor_upstream", value, self.raw())

    def test_monitor_self_port_refused(self):
        with pytest.raises(ValueError):
            options.check(
                "controller.monitor_upstream",
                "http://127.0.0.1:7601",
                self.raw(),
                own_ports=(7601,),
            )

    def test_metrics_ok(self):
        got = options.check("controller.metrics_url", "http://127.0.0.1:8428", self.raw())
        assert got == "http://127.0.0.1:8428"

    def test_metrics_refused(self):
        with pytest.raises(ValueError):
            options.check("controller.metrics_url", "http://192.0.2.5:8428", self.raw())

    def test_non_string_refused(self):
        with pytest.raises(ValueError):
            options.check("controller.access_log", 1, self.raw())

    def test_unknown_key_refused(self):
        with pytest.raises(ValueError):
            options.check("nope.missing", "x", self.raw())

    def test_root_ok(self):
        for value in ("/mnt/storage", "/media/disk/models", "/srv/storage"):
            assert options.check("archive.root", value, self.raw()) == value

    @pytest.mark.parametrize(
        "value",
        [
            "/",
            "/mnt",
            "mnt/x",
            "/etc/storage",
            "/usr/local/x",
            "/mnt/a/../b",
            "/mnt/a/",
            "/mnt//a",
            "/mnt/a\\b",
            "/mnt/a\nb",
            "/srv/example/cache",
            "/srv/example",
            "/srv/example/cache/sub",
            "/mnt/archive-raw",
            "/mnt/archive-raw/x",
            "/home/USER",
            "/home/USER/.cache/huggingface/hub/x",
            "/home/USER-agent/x",
            "/" + "a/" * 2100 + "b",
        ],
    )
    def test_root_refused(self, value):
        with pytest.raises(ValueError):
            options.check("archive.root", value, self.raw())


# ---------------------------------------------------------------- 2. set_key_text

class TestSetKeyText:
    def text(self):
        return (ROOT / "config" / "sparksi.example.toml").read_text(encoding="utf-8")

    @pytest.mark.parametrize(
        "key, value",
        [
            ("controller.access_log", "all"),
            ("controller.monitor_upstream", "http://127.0.0.1:7500"),
            ("archive.node", "b"),
        ],
    )
    def test_single_line_diff_and_comment_kept(self, key, value):
        old_text = self.text()
        old = load_toml_str(old_text)
        new_text = options.set_key_text(old_text, key, value)
        lines = changed_lines(old_text, new_text)
        assert len(lines) == 2  # one - and one +
        table, name = key.split(".")
        want = copy.deepcopy(old)
        want[table][name] = value
        assert load_toml_str(new_text) == want
        old_line = next(
            ln
            for ln in old_text.splitlines(keepends=True)
            if ln.lstrip().startswith(name)
        )
        new_line = next(
            ln
            for ln in new_text.splitlines(keepends=True)
            if ln.lstrip().startswith(name)
        )
        old_comment = old_line.split("#", 1)[1] if "#" in old_line else None
        new_comment = new_line.split("#", 1)[1] if "#" in new_line else None
        if old_comment is not None:
            assert new_comment == old_comment

    def test_hash_and_quotes_round_trip(self):
        old_text = self.text()
        new_text = options.set_key_text(old_text, "archive.label", 'Box "#1"')
        got = load_toml_str(new_text)
        assert got["archive"]["label"] == 'Box "#1"'

    def test_missing_key_inserted(self):
        old_text = self.text()
        lines = [ln for ln in old_text.splitlines(keepends=True) if not ln.lstrip().startswith("label")]
        stripped = "".join(lines)
        new_text = options.set_key_text(stripped, "archive.label", "Backups")
        got = load_toml_str(new_text)
        assert got["archive"]["label"] == "Backups"
        assert changed_lines(stripped, new_text) == ['+label = "Backups"\n']

    def test_missing_table_appended(self):
        tiny = '[controller]\nname = "c"\n'
        new_text = options.set_key_text(tiny, "archive.root", "/mnt/storage")
        got = load_toml_str(new_text)
        assert got["archive"]["root"] == "/mnt/storage"
        assert got["controller"]["name"] == "c"

    def test_crlf_kept(self):
        crlf = self.text().replace("\n", "\r\n")
        new_text = options.set_key_text(crlf, "controller.access_log", "all")
        edited = next(
            ln for ln in new_text.splitlines(keepends=True) if ln.lstrip().startswith("access_log")
        )
        assert edited.endswith("\r\n")
        assert len(changed_lines(crlf, new_text)) == 2

    def test_multiline_value_refused(self):
        doc = '[controller]\naccess_log = """slow\n"""\n'
        with pytest.raises(ValueError):
            options.set_key_text(doc, "controller.access_log", "all")


def load_toml_str(text):
    import tomllib
    return tomllib.loads(text)


# ---------------------------------------------------------------- 3. save()

class TestSave:
    def test_save_round_trip(self, tmp_path):
        cfg = tmp_path / "sparksi.toml"
        src = ROOT / "config" / "sparksi.example.toml"
        shutil.copyfile(src, cfg)
        cfg.chmod(0o644)
        old_bytes = cfg.read_bytes()
        old_mode = cfg.stat().st_mode & 0o777
        old_text = old_bytes.decode("utf-8")

        result = options.save(cfg, "controller.access_log", "all")
        assert result["changed"] is True
        backup = cfg.parent / result["backup"]
        assert backup.name.startswith("sparksi.toml.bak-")
        assert backup.read_bytes() == old_bytes

        new_text = cfg.read_text(encoding="utf-8")
        assert len(changed_lines(old_text, new_text)) == 2
        assert load_toml(cfg)["controller"]["access_log"] == "all"
        assert cfg.stat().st_mode & 0o777 == old_mode

        again = options.save(cfg, "controller.access_log", "all")
        assert again == {"changed": False, "backup": None}
        backups = list(tmp_path.glob("sparksi.toml.bak-*"))
        assert len(backups) == 1

    def test_save_refuses_other_changes(self, tmp_path, monkeypatch):
        cfg = tmp_path / "sparksi.toml"
        shutil.copyfile(ROOT / "config" / "sparksi.example.toml", cfg)
        old_bytes = cfg.read_bytes()

        real = options.set_key_text

        def noisy(text, key, value):
            out = real(text, key, value)
            return out + "\n[extra]\nthing = 1\n"

        monkeypatch.setattr(options, "set_key_text", noisy)
        with pytest.raises(ValueError):
            options.save(cfg, "controller.access_log", "all")
        assert cfg.read_bytes() == old_bytes


# ---------------------------------------------------------------- 4. HTTP

class TestHttp:
    def test_get_options(self, server):
        port, _cfg = server
        status, body = ask(port, "GET", "/api/options")
        assert status == 200
        items = body["options"]
        assert [it["key"] for it in items] == KEYS
        assert all(it["restart_needed"] is False for it in items)
        values = {it["key"]: it["value"] for it in items}
        assert values["controller.access_log"] == "slow"
        assert values["archive.root"] == "/mnt/archive"
        assert values["archive.node"] == "a"

    def test_post_ok_while_locked(self, server):
        """Settings keep their own write guard; the model lock does not block them."""
        port, cfg = server
        STATE.lock.acquire("test")
        try:
            assert STATE.lock.read().locked
            before = load_toml(cfg)
            status, body = ask(
                port,
                "POST",
                "/api/options",
                body={"key": "controller.access_log", "value": "all"},
            )
            assert status == 200
            assert body["changed"] is True
            assert body["backup"]
            after = load_toml(cfg)
            want = copy.deepcopy(before)
            want["controller"]["access_log"] = "all"
            assert after == want
            status, body = ask(port, "GET", "/api/options")
            assert status == 200
            entry = next(it for it in body["options"] if it["key"] == "controller.access_log")
            assert entry["value"] == "all"
            assert entry["restart_needed"] is True
            assert STATE.lock.read().locked  # a settings save neither needs nor touches the model lock
        finally:
            STATE.lock.release()

    def test_cross_origin_refused(self, server):
        port, cfg = server
        before = cfg.read_bytes()
        status, _ = ask(
            port,
            "POST",
            "/api/options",
            body={"key": "controller.access_log", "value": "all"},
            origin="evil",
        )
        assert status == 403
        assert cfg.read_bytes() == before

    def test_wrong_content_type(self, server):
        port, _cfg = server
        status, _raw = ask_raw(
            port,
            "POST",
            "/api/options",
            json.dumps({"key": "controller.access_log", "value": "all"}),
            "text/plain",
        )
        assert status == 415

    def test_bad_value(self, server):
        port, cfg = server
        before = cfg.read_bytes()
        status, body = ask(
            port,
            "POST",
            "/api/options",
            body={"key": "archive.root", "value": "/etc/x"},
        )
        assert status == 400
        assert body.get("error")
        assert cfg.read_bytes() == before

    def test_unknown_key(self, server):
        port, _cfg = server
        status, body = ask(
            port,
            "POST",
            "/api/options",
            body={"key": "nope.nope", "value": "x"},
        )
        assert status == 400

    def test_page(self, server):
        port, _cfg = server
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("GET", "/options")
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        assert resp.status == 200
        html = raw.decode("utf-8")
        assert 'id="view-options" class="view"' in html
        assert 'href="/options"' in html


def test_missing_key_shows_the_built_in_default_without_a_restart_note():
    """A key left out of the file shows what Sparksi really uses, and no restart is pending for it."""
    from sparksi.config import Config

    raw = load_toml(ROOT / "config" / "sparksi.example.toml")
    del raw["controller"]["access_log"]
    del raw["archive"]["label"]
    rows = {r["key"]: r for r in options.current(raw, Config(copy.deepcopy(raw)).raw)}
    assert rows["controller.access_log"]["value"] == "slow"
    assert rows["controller.access_log"]["restart_needed"] is False
    assert rows["archive.label"]["restart_needed"] is False


def test_env_override_is_reported():
    raw = load_toml(ROOT / "config" / "sparksi.example.toml")
    rows = {r["key"]: r for r in options.current(raw, None, {"SPARKSI_ACCESS_LOG": "all"})}
    assert rows["controller.access_log"]["override"].startswith("SPARKSI_ACCESS_LOG is set")
    assert "override" not in rows["controller.metrics_url"]

def test_toml_value_roundtrip_emoji_and_del():
    import tomllib

    from sparksi.netconfig import toml_value

    cases = ["smile 😀", 'a"b\\c', "tab\there", "del\x7f", "line\u2028sep", "中文"]
    for s in cases:
        rendered = toml_value(s)
        back = tomllib.loads("v = " + rendered)["v"]
        assert back == s
    assert "😀" in toml_value("smile 😀")
    assert "\\ud83d" not in toml_value("smile 😀")


def test_options_save_and_render_tables_emoji(tmp_path):
    import tomllib

    from sparksi import netconfig

    dst = tmp_path / "sparksi.toml"
    shutil.copyfile(ROOT / "config" / "sparksi.example.toml", dst)
    clean = options.check("archive.label", "Box 😀", load_toml(dst))
    assert options.save(dst, "archive.label", clean)["changed"] is True
    assert load_toml(dst)["archive"]["label"] == "Box 😀"
    assert "Box 😀" in dst.read_text(encoding="utf-8")
    text = netconfig.render_tables([{"id": "a", "name": "spark 😀"}], [])
    assert tomllib.loads(text)["nodes"][0]["name"] == "spark 😀"


def test_shown_masks_userinfo_all_forms():
    from sparksi.options import _shown
    inputs = [
        "http://u:secret@h:1",
        "u:secret@h:1",
        "http://u@h",
        "http://a@b@h:1",
        "http://u:secret@[::1:7500",
        "http://u:secret@h:notaport",
    ]
    for s in inputs:
        out = _shown("url", s)
        assert "***@" in out, s
        assert "secret" not in out, s
    assert _shown("url", "u:secret@h:1") == "***@h:1"
    assert _shown("url", "http://a@b@h:1/x") == "http://***@h:1/x"
    assert _shown("url", "http://h:1/a@b") == "http://h:1/a@b"  # an @ in the path is not userinfo


def test_shown_never_raises():
    from sparksi.options import _shown
    for v in [None, 123, object(), "", "://", "@", "http://", "a@b/c?d#e"]:
        assert isinstance(_shown("url", v), str)
        assert isinstance(_shown("other", v), str)


def test_api_options_malformed_ipv6_no_500(server):
    port, cfg_path = server
    text = cfg_path.read_text(encoding="utf-8")
    lines = [
        'monitor_upstream = "http://u:secret@[::1:7500"\n' if line.startswith("monitor_upstream = ") else line
        for line in text.splitlines(keepends=True)
    ]
    assert lines != text.splitlines(keepends=True)
    cfg_path.write_text("".join(lines), encoding="utf-8")
    status, body = ask(port, "GET", "/api/options")
    assert status == 200
    assert "secret" not in json.dumps(body)
    mirror = next(o for o in body["options"] if o["key"] == "controller.monitor_upstream")
    assert mirror["value"] == "http://***@[::1:7500"


# ---------------------------------------------------------------- audit of #81

MINIMAL = """[controller]
name = "controller"
ui_port = 7600

[[nodes]]
id = "a"
name = "spark-a"
ssh = "spark-a"

[[nodes]]
id = "b"
name = "spark-b"
ssh = "spark-b"

[archive]
root = "/mnt/archive"
"""


@pytest.mark.parametrize(
    "value",
    [
        "/mnt/archive-raw",                        # built-in archive.extra_blocked
        "/mnt/archive-raw/x",
        "/home/USER",                              # built-in paths.home
        "/home/USER/.cache/huggingface/hub/x",     # built-in paths.hf_hub
        "/home/USER/services/video",               # built-in paths.comfy_root
        "/home/USER-agent/x",                      # built-in paths.blocked_prefixes
    ],
)
def test_archive_root_uses_built_in_defaults_when_the_file_leaves_them_out(value):
    """No [paths] and no extra_blocked in the file: the defaults Config uses still protect these."""
    raw = tomllib.loads(MINIMAL)
    assert "paths" not in raw and "extra_blocked" not in raw["archive"]
    with pytest.raises(ValueError):
        options.check("archive.root", value, raw)
    assert options.check("archive.root", "/mnt/storage", raw) == "/mnt/storage"


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost.:7600",
        "http://LOCALHOST..:7600",
        "http://127.0.0.2:7600",
        "http://[::1]:7600",
        "http://100.64.0.7:7600",                  # this machine's tailnet IP
        "http://ctl.example.ts.net:7600",          # this machine's MagicDNS name
        "http://ctl.example.ts.net.:7600",
    ],
)
def test_monitor_mirror_refuses_this_machine_on_its_own_port(url):
    raw = tomllib.loads(MINIMAL)
    names = ("localhost", "127.0.0.1", "::1", "ctl", "100.64.0.7", "ctl.example.ts.net")
    with pytest.raises(ValueError, match="points at this controller"):
        options.check("controller.monitor_upstream", url, raw, (7600,), names)


def test_monitor_mirror_allows_another_controller():
    raw = tomllib.loads(MINIMAL)
    names = ("localhost", "127.0.0.1", "ctl", "100.64.0.7", "ctl.example.ts.net")
    ok = options.check("controller.monitor_upstream", "http://127.0.0.1:7500", raw, (7600,), names)
    assert ok == "http://127.0.0.1:7500"
    other = options.check("controller.monitor_upstream", "http://100.64.0.8:7600", raw, (7600,), names)
    assert other == "http://100.64.0.8:7600"


def test_start_mirror_refuses_own_tailnet_name(monkeypatch):
    """The startup path uses the same check with the host allowlist."""
    from sparksi import monitor_mirror

    monkeypatch.setattr(serve_mod.HOSTS, "names", lambda: {"localhost", "127.0.0.1", "ctl.example.ts.net"})
    monkeypatch.setattr(monitor_mirror, "configured_upstream", lambda: "http://ctl.example.ts.net.:7600")
    monkeypatch.setattr(serve_mod.STATE, "monitor_fn", None)
    started = []
    monkeypatch.setattr(monitor_mirror.Mirror, "start", lambda self: started.append(self) or self)
    serve_mod._start_mirror(7600)
    assert started == []
    assert serve_mod.STATE.monitor_fn is None


def test_save_runs_the_config_load_checks_first(server, monkeypatch):
    """A value the load checks refuse is never written (fails if Config(...) before save is removed)."""
    port, cfg = server
    before = cfg.read_bytes()

    def refuse(raw, *a, **k):
        raise ValueError("load check refused it")

    monkeypatch.setattr(serve_mod, "Config", refuse)
    status, body = ask(port, "POST", "/api/options", body={"key": "controller.access_log", "value": "all"})
    assert status == 400
    assert body["error"] == "load check refused it"
    assert cfg.read_bytes() == before
    assert not list(cfg.parent.glob("sparksi.toml.bak-*"))


# ---------------------------------------------------------------- re-audit of #81

def test_mirror_check_uses_only_the_running_port():
    """sparksi serve --port picks the listening port; the file's ui_port (unset, 7500 or 7601) must not matter."""
    own_names = ("localhost", "127.0.0.1", "::1")
    for ui_port in (None, 7500, 7601):
        raw = tomllib.loads(MINIMAL)
        if ui_port is None:
            raw["controller"].pop("ui_port", None)
        else:
            raw["controller"]["ui_port"] = ui_port
        ok = options.check(
            "controller.monitor_upstream",
            "http://127.0.0.1:7500",
            raw,
            own_ports=(7601,),
            own_names=own_names,
        )
        assert ok == "http://127.0.0.1:7500"
        with pytest.raises(ValueError, match="points at this controller"):
            options.check(
                "controller.monitor_upstream",
                "http://127.0.0.1:7500",
                raw,
                own_ports=(7500,),
                own_names=own_names,
            )


def test_http_save_ignores_the_file_ui_port(server):
    """The live server refuses only the port it listens on, not the file's ui_port (7500)."""
    port, cfg_path = server
    other = 7500 if port != 7500 else 7501

    status, body = ask(port, "POST", "/api/options", {"key": "controller.monitor_upstream", "value": f"http://127.0.0.1:{other}"})
    assert status == 200
    saved = tomllib.loads(cfg_path.read_text())
    assert saved["controller"]["monitor_upstream"] == f"http://127.0.0.1:{other}"

    status, body = ask(port, "POST", "/api/options", {"key": "controller.monitor_upstream", "value": f"http://127.0.0.1:{port}"})
    assert status == 400
    assert "points at this controller" in body.get("error", "")
    saved = tomllib.loads(cfg_path.read_text())
    assert saved["controller"]["monitor_upstream"] == f"http://127.0.0.1:{other}"


def test_shown_masks_scheme_relative():
    assert options._shown("url", "//u:secret@h:1") == "//***@h:1"
    shown = options._shown("url", "//u:secret@h:1/x?y")
    assert "secret" not in shown


def test_refuse_self_compares_addresses_not_spellings():
    from sparksi import monitor_mirror

    names = ("ctl.example.ts.net", "100.64.0.7", "fd7a:115c:a1e0::7")
    raising = (
        "http://CTL.Example.TS.NET:7600",
        "http://[::ffff:100.64.0.7]:7600",
        "http://[fd7a:115c:a1e0:0:0:0:0:7]:7600",
        "http://[FD7A:115C:A1E0::7]:7600",
        "http://[::ffff:127.0.0.1]:7600",
    )
    for base in raising:
        with pytest.raises(ValueError, match="points at this controller"):
            monitor_mirror.refuse_self(base, 7600, names)
    for base in ("http://100.64.0.8:7600", "http://100.64.0.7:7601"):
        monitor_mirror.refuse_self(base, 7600, names)
