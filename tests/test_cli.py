# SPDX-License-Identifier: Apache-2.0

from sparksi.cli import main


def test_help_exit() -> None:
    assert main([]) == 2


def test_unknown_set(capsys) -> None:
    assert main(["dry-run", "not-a-set"]) == 2
    err = capsys.readouterr().err
    assert "unknown set" in err


def test_refuse_evict(capsys) -> None:
    assert main(["evict"]) == 2
    err = capsys.readouterr().err
    assert "not implemented" in err


def test_tokens_reads_ledger(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    from sparksi.tokens import observe

    observe("counsel-a", prompt=0, gen=0, requests=1, port=8011)
    observe("counsel-a", prompt=100, gen=12, requests=1, port=8011)
    monkeypatch.setattr("sparksi.tokens.sample_vllm", lambda catalog=None: None)
    assert main(["tokens"]) == 0
    out = capsys.readouterr().out
    assert "counsel-a" in out
    assert "12" in out


def test_serve_port_8000_refused(capsys) -> None:
    assert main(["serve", "--port", "8000"]) == 2
    err = capsys.readouterr().err
    assert "8000" in err
    assert "refusing" in err
