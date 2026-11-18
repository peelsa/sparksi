# SPDX-License-Identifier: Apache-2.0

"""``sparksi config validate`` and ``show``, and a bad site file at import.

A bad ``sparksi.toml`` must not crash ``import sparksi``. Commands other than
``config`` refuse to run on it and exit 1 with the problem. Tests read files in a
temporary directory only; nothing contacts a machine.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from sparksi import compat, config
from sparksi.cli import main
from sparksi.config import CFG_ERRORS, check, redact


def _nodes(count: int) -> str:
    out = []
    for i in range(count):
        letter = "abcd"[i]
        out.append(
            f'[[nodes]]\nid = "{letter}"\nname = "spark-{letter}"\nssh = "spark-{letter}"\n'
            f'tailscale = "100.64.0.{11 + i}"\n'
        )
    return "\n".join(out)


def _write(tmp_path: Path, text: str, name: str = "sparksi.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("count", "roles"),
    [
        (1, "a (head)"),
        (2, "a (head), b (worker)"),
        (3, "a (head), b (worker), c (node3)"),
        (4, "a (head), b (worker), c (node3), d (node4)"),
    ],
)
def test_validate_accepts_one_to_four_sparks(tmp_path, capsys, count, roles) -> None:
    path = _write(tmp_path, _nodes(count))
    assert main(["config", "validate", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert f"{count} Spark" in captured.out
    assert roles in captured.out
    assert captured.err == ""


BAD = {
    "malformed": ("[[nodes\nid = 1", "(at line"),
    "five-nodes": (_nodes(4) + '\n[[nodes]]\nid = "e"\nssh = "spark-e"\n', "5 [[nodes]]"),
    "duplicate-id": (_nodes(2).replace('id = "b"', 'id = "a"'), "used by an earlier Spark"),
    "bad-ssh": ('[[nodes]]\nid = "a"\nssh = "-oProxyCommand=x"\n', "ssh"),
    "nodes-not-a-list": ('nodes = "x"\n', "[[nodes]]"),
    "table-is-a-string": ('controller = "x"\n\n' + _nodes(1), "[controller] must be a table"),
}


@pytest.mark.parametrize("name", list(BAD))
def test_validate_names_each_kind_of_bad_file(tmp_path, capsys, name) -> None:
    text, needle = BAD[name]
    path = _write(tmp_path, text)
    assert main(["config", "validate", "--file", str(path)]) == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert str(path) in err
    assert needle in err
    assert "sparksi.toml sparksi.toml" not in err
    assert "Traceback" not in err


def test_validate_a_node_value_of_the_wrong_type(tmp_path, capsys) -> None:
    path = _write(tmp_path, '[[nodes]]\nid = "a"\numa_gib = "big"\n')
    assert main(["config", "validate", "--file", str(path)]) == 1
    assert "could not convert" in capsys.readouterr().err


def test_validate_a_missing_file(tmp_path, capsys) -> None:
    path = tmp_path / "absent.toml"
    assert main(["config", "validate", "--file", str(path)]) == 1
    err = capsys.readouterr().err
    assert "cannot read" in err
    assert str(path) in err


def test_validate_a_file_that_is_not_utf8(tmp_path, capsys) -> None:
    path = tmp_path / "sparksi.toml"
    path.write_bytes(b"\xff\xfe")
    assert main(["config", "validate", "--file", str(path)]) == 1
    assert "error:" in capsys.readouterr().err


def test_validate_without_a_file_says_where_it_looked(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SPARKSI_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.chdir(tmp_path)
    assert main(["config", "validate"]) == 1
    assert "no sparksi.toml found" in capsys.readouterr().err


def test_validate_uses_the_file_sparksi_would_load(tmp_path, monkeypatch, capsys) -> None:
    path = _write(tmp_path, _nodes(2))
    monkeypatch.setenv("SPARKSI_CONFIG", str(path))
    assert main(["config", "validate"]) == 0
    assert str(path) in capsys.readouterr().out
    monkeypatch.setenv("SPARKSI_CONFIG", str(tmp_path / "absent.toml"))
    assert main(["config", "validate"]) == 1


def test_validate_warns_but_passes_on_old_names_and_unknown_presets(tmp_path, capsys) -> None:
    old_lines = "".join(f'{old} = "/srv/example/{new}"\n' for new, old in compat.LEGACY_PATH_KEYS.items())
    text = _nodes(2) + f'\n[paths]\n{old_lines}\n[protect]\npresets = ["video-service", "made-up"]\n'
    path = _write(tmp_path, text)
    assert main(["config", "validate", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert "ok:" in captured.out
    for new, old in compat.LEGACY_PATH_KEYS.items():
        assert f"warning: [paths] {old} is deprecated; use [paths] {new}\n" in captured.err
    assert "unknown preset 'made-up'" in captured.err
    assert captured.err.count("warning:") == len(compat.LEGACY_PATH_KEYS) + 1  # video-service is known


def test_validate_does_not_warn_when_both_names_are_set(tmp_path, capsys) -> None:
    both = "".join(f'{old} = "/a"\n{new} = "/b"\n' for new, old in compat.LEGACY_PATH_KEYS.items())
    path = _write(tmp_path, _nodes(1) + f"\n[paths]\n{both}")
    assert main(["config", "validate", "--file", str(path)]) == 0
    assert capsys.readouterr().err == ""


def test_validate_writes_nothing_and_does_not_load_the_file_as_the_live_config(tmp_path) -> None:
    path = _write(tmp_path, _nodes(3))
    before = path.read_bytes()
    live_nodes = [node.id for node in config.CFG.nodes]
    assert main(["config", "validate", "--file", str(path)]) == 0
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["sparksi.toml"]
    assert [node.id for node in config.CFG.nodes] == live_nodes


def test_show_prints_defaults_merged_with_the_file(tmp_path, capsys) -> None:
    path = _write(tmp_path, _nodes(3) + '\n[controller]\nui_port = 7600\n')
    assert main(["config", "show", "--file", str(path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [n["id"] for n in data["nodes"]] == ["a", "b", "c"]
    assert data["controller"]["ui_port"] == 7600
    assert data["controller"]["announce_port"] == 7501  # default kept
    assert data["engines"]["allowed_images"]  # default table present


def test_show_without_a_file_prints_the_built_in_example(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.delenv("SPARKSI_CONFIG", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "none"))
    monkeypatch.chdir(tmp_path)
    assert main(["config", "show"]) == 0
    captured = capsys.readouterr()
    assert "built-in example" in captured.err
    assert json.loads(captured.out) == json.loads(json.dumps(config.Config().effective(), default=str))
    assert [n["id"] for n in json.loads(captured.out)["nodes"]] == ["a", "b"]


def test_show_on_a_bad_file_exits_1(tmp_path, capsys) -> None:
    path = _write(tmp_path, "not = [valid")
    assert main(["config", "show", "--file", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "error:" in captured.err


def test_show_redacts_secrets_and_url_passwords(tmp_path, capsys) -> None:
    text = _nodes(1) + (
        '\n[controller]\nmetrics_url = "http://user:hunter2@127.0.0.1:8428"\n'
        'unlock_token = "abc123"\nmonitor_upstream = "http://100.64.0.9:7500"\n'
        '\n[services.example]\napi_key = "k-9"\nuser_controllable = false\nnote = "keep"\n'
    )
    path = _write(tmp_path, text)
    assert main(["config", "show", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    for secret in ("hunter2", "abc123", "k-9"):
        assert secret not in out
    data = json.loads(out)
    assert data["controller"]["metrics_url"] == "http://<redacted>@127.0.0.1:8428"
    assert data["controller"]["unlock_token"] == "<redacted>"
    assert data["services"]["example"]["api_key"] == "<redacted>"
    assert data["services"]["example"]["note"] == "keep"
    assert data["controller"]["monitor_upstream"] == "http://100.64.0.9:7500"


def test_redact_keeps_empty_and_boolean_values_and_does_not_edit_its_input() -> None:
    value = {"token": "", "secret": None, "password_set": False, "list": [{"api_key": "x"}, "http://a:b@h/"]}
    out = redact(value)
    assert out == {"token": "", "secret": None, "password_set": False, "list": [{"api_key": "<redacted>"}, "http://<redacted>@h/"]}
    assert value["list"][0]["api_key"] == "x"


def test_redact_hides_numbers_and_everything_inside_a_secret_named_table() -> None:
    value = {
        "numeric_token": 123456,
        "float_secret": 1.5,
        "zero_password": 0,
        "flag_token": True,
        "secret": {"value": "s3cret", "port": 22, "deep": {"x": ["a", 7]}},
        "credentials": {"user": "bob"},
        "ports": [7500, 7501],
        "api_key": ["k1", 2],
        "api_token": "t",
        "admin_password": "p",
        "client_secret": "c",
        "private_key": "pk",
        "url": "http://user:pw@h:1/",
    }
    out = redact(value)
    assert out["numeric_token"] == out["float_secret"] == out["zero_password"] == "<redacted>"
    assert out["flag_token"] is True
    assert out["secret"] == {"value": "<redacted>", "port": "<redacted>", "deep": {"x": ["<redacted>", "<redacted>"]}}
    assert out["credentials"] == {"user": "<redacted>"}
    assert out["api_key"] == ["<redacted>", "<redacted>"]
    for key in ("api_token", "admin_password", "client_secret", "private_key"):
        assert out[key] == "<redacted>"
    assert out["ports"] == [7500, 7501]
    assert out["url"] == "http://<redacted>@h:1/"


def test_show_redacts_secret_tables_and_numbers(tmp_path, capsys) -> None:
    text = _nodes(1) + (
        '\n[secret]\nvalue = "s3cret"\nnested = { deep = "d" }\n'
        '\n[credentials]\nuser = "bob"\nport = 22\n'
        '\n[services.example]\nnumeric_token = 123456\napi_token = "t"\nadmin_password = "p"\n'
        'client_secret = "c"\napi_key = ["k1", "k2"]\nflag_token = true\n'
    )
    path = _write(tmp_path, text)
    assert main(["config", "show", "--file", str(path)]) == 0
    out = capsys.readouterr().out
    for clear in ("s3cret", '"d"', "bob", "123456", '"t"', '"p"', '"c"', "k1", "k2"):
        assert clear not in out, clear
    data = json.loads(out)
    assert data["secret"] == {"nested": {"deep": "<redacted>"}, "value": "<redacted>"}
    assert data["credentials"] == {"port": "<redacted>", "user": "<redacted>"}
    example = data["services"]["example"]
    assert example["numeric_token"] == "<redacted>"
    assert example["api_key"] == ["<redacted>", "<redacted>"]
    assert example["flag_token"] is True


def test_the_shipped_defaults_hold_no_secret() -> None:
    data = config.Config().raw
    assert redact(data) == data


def test_effective_is_a_copy() -> None:
    cfg = config.Config()
    out = cfg.effective()
    out["controller"]["ui_port"] = 1
    assert cfg.raw["controller"]["ui_port"] == 7500


def test_config_alone_prints_usage(capsys) -> None:
    assert main(["config"]) == 2
    assert "usage: sparksi config" in capsys.readouterr().err


def test_check_returns_messages_and_never_raises(tmp_path) -> None:
    for text in ("[[nodes", "nodes = 3", "[[nodes]]\nid = 1\n", 'engines = "x"\n'):
        cfg, errors = check(_write(tmp_path, text))
        assert cfg is None
        assert errors and all(isinstance(e, str) and e for e in errors)
    cfg, errors = check(_write(tmp_path, _nodes(4)))
    assert errors == []
    assert [n.id for n in cfg.nodes] == ["a", "b", "c", "d"]


def test_a_valid_reload_clears_the_import_errors(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "CFG_ERRORS", CFG_ERRORS)
    CFG_ERRORS[:] = ["left over from import"]
    snapshot = (dict(config.CFG.raw), config.CFG.source)
    try:
        config.reload(_write(tmp_path, _nodes(2)))
        assert CFG_ERRORS == []
    finally:
        CFG_ERRORS[:] = []
        config.CFG.__init__(snapshot[0], source=snapshot[1])


def test_main_refuses_every_other_command_while_the_import_errors_stand(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sparksi.cli.CFG_ERRORS", ["sparksi.toml: broken"])
    for argv in (["status"], ["runs"], ["tokens"], ["serve", "--port", "7599"], ["evict"], ["install-unit", "--print"]):
        assert main(argv) == 1, argv
        assert capsys.readouterr().err == "sparksi.toml: broken\n"


def test_main_still_runs_config_commands_while_the_import_errors_stand(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setattr("sparksi.cli.CFG_ERRORS", ["sparksi.toml: broken"])
    path = _write(tmp_path, _nodes(2))
    assert main(["config", "validate", "--file", str(path)]) == 0
    assert main(["config", "show", "--file", str(path)]) == 0
    capsys.readouterr()


# --- a fresh process with a bad file -------------------------------------------------

def _run(args: list[str], config_file: Path, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "home" / ".config"),
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "SPARKSI_CONFIG": str(config_file),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, env=env, timeout=60, check=False)


@pytest.fixture()
def bad_file(tmp_path: Path) -> Path:
    return _write(tmp_path, '[[nodes]]\nid = "a"\nssh = "-oProxyCommand=x"\n')


def test_import_survives_a_bad_file_and_logs_it_once(bad_file, tmp_path) -> None:
    code = (
        "import sparksi.cli, sparksi.monitor.__main__, sparksi.serve\n"
        "from sparksi.config import CFG, CFG_ERRORS\n"
        "print(len(CFG_ERRORS), [n.id for n in CFG.nodes])\n"
    )
    proc = _run(["-c", code], bad_file, tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1 ['a', 'b']"  # the built-in example stands in
    assert "Traceback" not in proc.stderr
    assert proc.stderr.count("ignoring invalid config") >= 1
    assert "-oProxyCommand" not in proc.stdout


@pytest.mark.parametrize("argv", [["status"], ["tokens"], ["runs"], ["serve", "--port", "7599"]])
def test_a_command_exits_1_with_the_message_on_a_bad_file(bad_file, tmp_path, argv) -> None:
    proc = _run(["-m", "sparksi.cli", *argv], bad_file, tmp_path)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert "ssh" in proc.stderr
    assert str(bad_file) in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_the_monitor_exits_1_on_a_bad_file(bad_file, tmp_path) -> None:
    proc = _run(["-m", "sparksi.monitor", "--snapshot"], bad_file, tmp_path)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    assert str(bad_file) in proc.stderr
    assert "Traceback" not in proc.stderr
    assert proc.stdout == ""


def test_config_validate_runs_in_a_fresh_process_on_a_bad_file(bad_file, tmp_path) -> None:
    proc = _run(["-m", "sparksi.cli", "config", "validate"], bad_file, tmp_path)
    assert proc.returncode == 1
    assert "error:" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_a_missing_file_named_by_the_environment_does_not_crash_import(tmp_path) -> None:
    proc = _run(["-m", "sparksi.cli", "status"], tmp_path / "absent.toml", tmp_path)
    assert proc.returncode == 1
    assert "cannot read" in proc.stderr
    assert "Traceback" not in proc.stderr


# --- nesting, config path failures, [protect] presets, BOM ---------------------------

def _deep_inline_array() -> str:
    return "a = " + "[" * 100000 + "]" * 100000 + "\n"


def _deep_inline_table() -> str:
    return "a = " + "{a=" * 20000 + "1" + "}" * 20000 + "\n"


def _deep_table_header() -> str:
    return "[" + ".".join(f"k{i}" for i in range(50000)) + "]\nx = 1\n"


DEEP = {
    "inline-array": _deep_inline_array,
    "inline-table": _deep_inline_table,
    "table-header": _deep_table_header,
}


@pytest.mark.parametrize("kind", list(DEEP))
@pytest.mark.parametrize("command", ["validate", "show"])
def test_deeply_nested_files_are_reported_not_raised(tmp_path, capsys, kind, command) -> None:
    path = _write(tmp_path, DEEP[kind]())
    assert main(["config", command, "--file", str(path)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert str(path) in captured.err
    assert captured.err.count("\n") == 1  # one clear line, not a traceback
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("kind", list(DEEP))
def test_check_catches_recursion_and_any_toml_error(tmp_path, kind) -> None:
    cfg, errors = check(_write(tmp_path, DEEP[kind]()))
    assert cfg is None
    assert len(errors) == 1
    assert str(tmp_path) in errors[0]


@pytest.mark.parametrize("kind", list(DEEP))
def test_import_survives_a_deeply_nested_file(tmp_path, kind) -> None:
    path = _write(tmp_path, DEEP[kind]())
    proc = _run(["-c", "import sparksi.cli, sparksi.monitor.__main__; print('imported')"], path, tmp_path)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.strip() == "imported"
    assert "Traceback" not in proc.stderr
    for argv in (["config", "validate"], ["status"]):
        proc = _run(["-m", "sparksi.cli", *argv], path, tmp_path)
        assert proc.returncode == 1, argv
        assert str(path) in proc.stderr
        assert "Traceback" not in proc.stderr


def test_a_home_that_cannot_be_resolved_is_reported_by_import_and_commands(tmp_path) -> None:
    bad = Path("~nosuchuser_zz/x.toml")
    proc = _run(["-c", "import sparksi.cli; print('imported')"], bad, tmp_path)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert proc.stdout.strip() == "imported"
    for argv in (["status"], ["config", "validate"], ["config", "show"]):
        proc = _run(["-m", "sparksi.cli", *argv], bad, tmp_path)
        assert proc.returncode == 1, argv
        assert "SPARKSI_CONFIG='~nosuchuser_zz/x.toml'" in proc.stderr
        assert "Traceback" not in proc.stderr


def test_a_file_option_with_an_unknown_user_is_reported(capsys) -> None:
    assert main(["config", "validate", "--file", "~nosuchuser_zz/x.toml"]) == 1
    err = capsys.readouterr().err
    assert "cannot work out the config path from ~nosuchuser_zz/x.toml" in err
    assert "Traceback" not in err


def test_import_from_a_deleted_working_directory_is_reported(tmp_path) -> None:
    code = (
        "import os, sys\n"
        "d = os.path.join(sys.argv[1], 'gone'); os.mkdir(d); os.chdir(d); os.rmdir(d)\n"
        "import sparksi.cli\n"
        "from sparksi.config import CFG_ERRORS\n"
        "print('imported', CFG_ERRORS)\n"
        "print('rc', sparksi.cli.main(['config', 'validate']))\n"
        "print('rc', sparksi.cli.main(['status']))\n"
    )
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "nowhere"),  # no file there, so the search reaches the working directory
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
    }
    proc = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, env=env, timeout=60, check=False)
    assert proc.returncode == 0, proc.stderr[-400:]
    assert "imported ['cannot look for sparksi.toml: the working directory no longer exists']" in proc.stdout
    assert proc.stdout.count("rc 1") == 2
    assert "cannot look for sparksi.toml" in proc.stderr
    assert "Traceback" not in proc.stderr


@pytest.mark.parametrize(
    ("presets", "expected"),
    [
        ("5", "[protect] presets must be a list of names, not int"),
        ("true", "[protect] presets must be a list of names, not bool"),
        ('{ a = 1 }', "[protect] presets must be a list of names, not dict"),
        ("[2]", "[protect] presets has an item that is not a name: 2"),
        ('[["video-service"]]', "[protect] presets has an item that is not a name: ['video-service']"),
        ('["video-service", 1.5]', "[protect] presets has an item that is not a name: 1.5"),
        ('"made-up"', "[protect] presets names an unknown preset 'made-up'"),
    ],
)
def test_validate_reports_any_shape_of_presets_without_crashing(tmp_path, capsys, presets, expected) -> None:
    path = _write(tmp_path, f"{_nodes(2)}\n[protect]\npresets = {presets}\n")
    assert main(["config", "validate", "--file", str(path)]) == 0
    captured = capsys.readouterr()
    assert f"warning: {expected}" in captured.err
    assert "unknown preset '[" not in captured.err
    assert "Traceback" not in captured.err
    assert "ok:" in captured.out


def test_validate_accepts_a_known_preset_as_a_list_or_a_string(tmp_path, capsys) -> None:
    for form in ('["video-service"]', '"video-service"', "[]"):
        path = _write(tmp_path, f"{_nodes(1)}\n[protect]\npresets = {form}\n")
        assert main(["config", "validate", "--file", str(path)]) == 0
        assert capsys.readouterr().err == "", form


def test_a_file_with_a_byte_order_mark_says_so(tmp_path, capsys) -> None:
    path = tmp_path / "sparksi.toml"
    path.write_bytes(b"\xef\xbb\xbf" + _nodes(1).encode())
    assert main(["config", "validate", "--file", str(path)]) == 1
    err = capsys.readouterr().err
    assert "byte order mark (BOM)" in err
    assert str(path) in err
    path.write_bytes(_nodes(1).encode())  # the same text without it is fine
    assert main(["config", "validate", "--file", str(path)]) == 0


def test_a_directory_named_as_the_file_is_reported(tmp_path, capsys) -> None:
    (tmp_path / "sparksi.toml").mkdir()
    assert main(["config", "validate", "--file", str(tmp_path / "sparksi.toml")]) == 1
    assert "cannot read" in capsys.readouterr().err
