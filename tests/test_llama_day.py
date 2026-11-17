# SPDX-License-Identifier: Apache-2.0

from sparksi import llama_day
from sparksi.announce import LlamaCounts, _log_line, to_peer


def _pkt(dp, dg, ps, yp=0, yg=0, dy=20261019):
    return to_peer({"h": "a", "http": {"8014": ["gguf", {
        "dp": dp, "dg": dg, "yp": yp, "yg": yg, "dy": dy, "ps": ps}]}})["http"]


def test_restart_adds_a_process_row_and_never_goes_backwards(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    llama_day.reset_for_tests()
    llama_day.note("a", _pkt(100, 10, 1000))
    llama_day.note("a", _pkt(90, 5, 1000))      # dip (in-flight gone) is ignored
    llama_day.note("a", _pkt(40, 4, 2000))      # restart: new process
    assert llama_day.rows_for("2026-10-19") == [("spark-a", "gguf", 140, 14)]
    llama_day.note("a", _pkt(1, 1, 2000, yp=70, yg=3, dy=20261020))
    assert llama_day.rows_for("2026-10-20") == [("spark-a", "gguf", 1, 1)]
    assert llama_day.rows_for("2026-10-19") == [("spark-a", "gguf", 170, 14)]
    llama_day.reset_for_tests()


def test_vllm_packet_does_not_touch_the_ledger(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    llama_day.reset_for_tests()
    http = to_peer({"h": "b", "http": {"8020": ["qwen", {"pt": 5, "gt": 6}]}})["http"]
    llama_day.note("b", http)
    assert llama_day.rows_for("2026-10-19") == []
    llama_day.reset_for_tests()


def test_log_line_parses_summary_lines():
    p = _log_line("1551.08.540.359 I slot print_timing: id  0 | task 80718 | prompt eval time =  206342.56 ms / 98793 tokens (    2.09 ms per token,   478.78 tokens per second)")
    g = _log_line("1551.08.540.367 I slot print_timing: id  0 | task 80718 |        eval time =    3690.37 ms /    61 tokens (   61.51 ms per token,    16.26 tokens per second)")
    tick = _log_line("1552.28.283.095 I slot print_timing: id  0 | task 81525 | n_gen =    821, tg =  23.55 t/s")
    assert p == (1551 * 60 + 8.54, "p", 98793)
    assert g[1:] == ("g", 61)
    assert tick[1] is None


def test_counts_use_only_the_running_segment(tmp_path, monkeypatch):
    log = tmp_path / "server.log"
    log.write_text(
        "0.00.100.000 I srv llama_server: initializing\n"
        "5.00.000.000 I slot print_timing: id 0 | task 1 | prompt eval time = 1.0 ms / 999 tokens (x)\n"
        "0.00.100.000 I srv llama_server: initializing\n"
        "1.00.000.000 I slot print_timing: id 0 | task 1 | prompt eval time = 1.0 ms / 30 tokens (x)\n"
        "1.00.000.001 I slot print_timing: id 0 | task 1 |        eval time = 1.0 ms / 7 tokens (x)\n"
    )
    import sparksi.announce as a
    c = LlamaCounts()
    row = {"pid": 1, "start": 1_800_000_000.0, "log": str(log), "off": 0, "ino": None,
           "days": {}, "last": -1.0, "rest": ""}
    monkeypatch.setattr(c, "_find", lambda port: row)
    monkeypatch.setattr(a, "_slots_live", lambda port: {"p": 5, "g": 2, "running": 1, "used": 100, "ctx": 1000})
    monkeypatch.setattr(a, "_health_ok", lambda port: True)
    got = c.counts(8014, now=1_800_000_000.0 + 120)
    assert got["dp"] == 35 and got["dg"] == 9 and got["ps"] == 1_800_000_000
    assert got["r"] == 1 and got["cx"] == 0.1 and got["hl"] == 1


def test_monitor_card_shows_llama_server_as_working():
    from sparksi.dash import _llama_row

    rec = {"name": "gguf", "state": "busy", "tok_s": 0.0, "kv_pct": None, "running": 1, "waiting": 0}
    _llama_row(rec, {"llama_process_start": 1.0, "num_requests_running": 1.0,
                     "llama_gen_tps": 33.9, "llama_ctx_used": 0.12, "llama_health_ok": 1})
    assert rec["state"] == "GEN" and rec["tok_s"] == 33.9 and rec["running"] == 1
    assert rec["waiting"] is None and rec["kv_kind"] == "ctx" and round(rec["kv_pct"]) == 12

    idle = {"name": "gguf"}
    _llama_row(idle, {"llama_process_start": 1.0, "num_requests_running": 0.0, "llama_gen_tps": 0.0,
                      "llama_health_ok": 1})
    assert idle["state"] == "idle" and idle["kv_pct"] is None

    down = {"name": "gguf"}
    _llama_row(down, {"llama_process_start": 1.0, "llama_health_ok": 0})
    assert down["state"] == "DOWN" and down["tok_s"] is None
