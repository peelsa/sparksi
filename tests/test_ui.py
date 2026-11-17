# SPDX-License-Identifier: Apache-2.0

import json
import threading
from urllib.request import Request, urlopen

from sparksi.lock import Lock
from sparksi.models import gpu_size_parts, list_models, resolve_modalities
from sparksi.public import refuse_apply_while_video_live, strip_secrets
from sparksi.serve import INDEX_HTML, Handler, STATE, DualServer, page_html, post_apply


HTML = INDEX_HTML.read_text(encoding="utf-8")


def test_index_is_html_not_json() -> None:
    text = page_html()
    assert text.lstrip().lower().startswith("<!doctype html>")
    assert not text.lstrip().startswith("{")
    assert "Lock" in text
    assert "Library" in text
    assert 'class="machines"' in text
    assert "Available harnesses" not in text
    assert "Now in memory" not in text
    assert ">Tokens<" not in text
    assert 'id="token-day"' not in text
    assert "/api/token-life" in text
    assert "out · " in text
    assert "function renderLive()" in text
    assert "hostTodayText" in text
    assert "tokenLifeText" not in text
    assert "modelTokenText" in text
    assert text.find("function renderLive()") < text.find("hostTodayText")
    assert "/api/token-day" in text
    assert 'id="btn-stop-live"' in text
    assert 'id="btn-lib-stop"' not in text
    assert "Installed Models" not in text
    assert "DGX SPARK" not in text
    assert "Load on head" not in text
    assert ">Load</button>" in text
    assert "spark-a" not in text
    assert "spark-b" not in text
    assert "/api/config" in text
    assert 'reason: "load started"' in text
    assert "Stop does not interrupt a load." in text
    assert "libraryIdFor(mod)" in text
    served = text.find("mod.served")
    live = text.find('container_source === "engine"')
    assert served != -1 and live != -1 and served < live
    stop = text.split('el("btn-stop-live")')[1].split('el("btn-job-pause")')[0]
    assert "lastMonitor.loaded" in stop
    assert "wanted[row.id]" in stop
    assert 'live[i].placement === "pair"' not in stop
    assert "model loads on first job" in text
    assert "<th>weights</th>" not in text
    assert "weightsOn" in text
    assert "align-self: start" not in text
    assert "groupMachines" in text
    assert 'href="/models"' in text
    assert "Legacy view" not in text
    assert 'id="view-models"' in text
    assert "Can run now" in text
    assert "Needs copy" in text
    assert "Archive only" in text
    assert "Head only" in text
    assert "waiting_health" in text
    assert 'id="recipe-bar"' in text
    assert 'id="recipe-cmd-save-as"' in text
    assert 'id="recipe-cmd-rename"' in text
    assert 'id="model-save"' not in text
    assert "Load default recipe" in text
    assert 'id="services-h"' in text
    assert "Services" in text
    assert 'id="service-rows"' in text
    assert "Extra arguments" in text
    assert "imported from" in text
    assert 'id="model-default"' not in text
    assert "512K (524288)" in text
    assert "1M (1048576)" in text
    assert "Environment" in text
    assert '" is off limits."' in text
    assert "ComfyUI stays listed and cannot be controlled." not in text
    assert "option.reason" in text
    assert "last heard" in text
    assert "no report yet" in text
    assert ".top .monitor { height: 100%; }" not in text


def test_recipe_default_checkbox_and_context_presets() -> None:
    assert 'id="model-load-default"' in HTML
    assert "Load default recipe" in HTML
    assert "Use default recipe" not in HTML
    assert 'id="model-default-summary"' in HTML
    title = HTML.find('id="model-title"')
    box = HTML.find("model-load-default")
    engine = HTML.find('id="model-engine"')
    placement = HTML.find("2 · Placement")
    port = HTML.find("3 · Port")
    recipe = HTML.find("4 · Recipe")
    fields = HTML.find('id="model-fields"')
    assert title != -1 and box != -1 and engine != -1 and fields != -1
    assert title < box < engine < placement < port < recipe < fields
    assert "fetchPreview(\"defaults\")" in HTML
    assert "lastDefaults.engine" in HTML
    assert "remove.hidden = !!locked" in HTML
    assert "lockSetup(false)" in HTML
    assert "svc.running ? \"up\" : \"down\"" in HTML
    assert "RoCE permissions" in HTML
    assert 'roce.id = "roce-permissions"' in HTML
    assert 'tokenNote.id = "token-required"' in HTML
    assert "values.token_note" in HTML
    assert "Clear page cache before start" in HTML
    assert 'evict.id = "field-evict-cache"' in HTML
    assert "(fact.ports || []).indexOf(8188)" not in HTML
    assert "4K (4096)" in HTML
    assert "[8192, \"8K\"]" in HTML
    assert "128K (131072)" in HTML
    assert "256K (262144)" in HTML
    assert "Model maximum (" not in HTML
    assert "Recorded cap (" in HTML
    assert "Custom…" in HTML
    assert 'custom.id = "field-ctx-custom"' in HTML
    assert 'custom.step = "1024"' in HTML
    assert 'util.step = "0.05"' in HTML
    assert 'seqs.step = "1"' in HTML
    assert 'step = "0.01"' not in HTML
    assert 'from " + sources[sourceKey]' in HTML
    assert "Reset these fields to the default recipe?" in HTML
    assert "rememberDraft" in HTML
    assert "is-locked" in HTML
    assert ".model-setup .field[hidden] { display: none; }" in HTML


def test_no_innerhtml_or_localstorage() -> None:
    assert "innerHTML" not in HTML
    assert "localStorage" not in HTML
    assert "unlock.token" not in HTML


def test_library_rows_select_without_radios() -> None:
    assert 'pick.type = "radio"' not in HTML
    assert "libPicks" in HTML
    assert "lib-sel" in HTML
    assert "Select a library row first." in HTML


def test_nonserve_memory_replaces_floor_table() -> None:
    assert "Non Serve Memory Usage" not in HTML
    assert "<th>part</th>" not in HTML
    assert "<th>unit</th>" not in HTML
    assert "<th>src</th>" not in HTML
    assert "reclaimable cache" not in HTML
    assert 'memCell(hr, "consumer"' not in HTML
    assert "Mem" in HTML and "CPU" in HTML and "GPU" in HTML


def test_now_in_memory_is_per_spark() -> None:
    chunk = HTML.split("function renderLive()")[1].split("function selectedIds")[0]
    assert "groupMachines" in chunk
    assert "portLiveRows" in chunk
    assert "peer unreachable" not in chunk
    assert "nothing in memory" not in chunk
    assert "sparkRoles()" in chunk


def test_no_trio_language_on_page() -> None:
    assert "trio" not in HTML.lower()


def test_web_unlock_has_no_token_field() -> None:
    assert "unlock-token" not in HTML
    assert "from_ui" in HTML


def test_finished_switch_does_not_relock_on_every_poll() -> None:
    chunk = HTML.split("function tickJob()")[1].split("function askLoad")[0]
    assert "/api/lock" not in chunk


def test_port_live_memory_follows_its_port() -> None:
    chunk = HTML.split("function memoryText")[1].split("function weightsOn")[0]
    assert "row.port_live" in chunk
    assert chunk.index("row.port_live") < chunk.index("lastMonitor.loaded")
    live = HTML.split("function renderLive()")[1].split("function selectedIds")[0]
    assert "portLiveRows" in live


def test_unlock_click_does_not_wait_on_models_refresh() -> None:
    chunk = HTML.split("function unlockClick(bar)")[1].split("LOCK_BARS.forEach")[0]
    assert "applyLockPayload" in chunk
    assert "Unlocking…" in chunk
    assert "refreshStatus" in chunk
    assert "refresh()" not in chunk
    assert "/api/models" not in chunk


def test_switch_is_single_button() -> None:
    assert HTML.count('id="btn-load-head"') == 1
    assert HTML.count('id="btn-load-worker"') == 1
    assert HTML.count('id="btn-load-pair"') == 1
    assert 'id="btn-switch"' not in HTML
    assert "btn-plan-trio" not in HTML
    assert "btn-wanted-trio" not in HTML


def test_strip_secrets() -> None:
    raw = {"lock": {"on": "ON"}, "unlock_token": "SECRET", "nested": {"token": "NOPE"}}
    out = strip_secrets(raw)
    blob = json.dumps(out)
    assert "SECRET" not in blob
    assert "NOPE" not in blob
    assert out["lock"]["on"] == "ON"


def test_refuse_apply_8188_independent_of_victims() -> None:
    blocked = refuse_apply_while_video_live(listen_fn=lambda port, host: port == 8188)
    assert blocked is not None
    assert blocked["executed"] is False
    assert "8188" in blocked["error"]
    assert refuse_apply_while_video_live(listen_fn=lambda port, host: False) is None


def test_post_apply_4xx_when_8188_live_even_if_victims_omit(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    STATE.lock = Lock(tmp_path / "lock")
    STATE.lock.release(ttl_s=120)
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(argv)
        return 0

    code, body = post_apply(
        {"spec": "brain-a,vision-a,counsel-a", "victims": []},
        "any-token",
        listen_fn=lambda port, host: True,
        runner=runner,
    )
    assert code >= 400
    assert body.get("executed") is False
    assert calls == []


def test_list_models_has_catalog_ids(monkeypatch, video_catalog) -> None:
    monkeypatch.setattr("sparksi.models._dgx_models_json", lambda: [])
    monkeypatch.setattr("sparksi.models._dir_size_gib", lambda p: 12.3)
    monkeypatch.setattr("sparksi.models.compute_apps", lambda: ([], None))
    rows = list_models()
    ids = {r.get("catalog_id") for r in rows}
    assert "brain-a" in ids
    assert video_catalog in ids
    gem = next(r for r in rows if r.get("catalog_id") == "brain-a")
    assert gem.get("model")
    assert gem.get("quant") == "nvfp4"
    assert gem.get("size_label") == "12.3 GiB"
    assert gem.get("audio") == 0
    assert gem.get("image") == 1
    assert gem.get("video") == 1
    nano = next(r for r in rows if r.get("catalog_id") == "vision-a")
    assert (nano.get("audio"), nano.get("image"), nano.get("video")) == (1, 1, 1)
    video = next(r for r in rows if r.get("catalog_id") == video_catalog)
    assert (video.get("audio"), video.get("image"), video.get("video")) == (1, 1, 1)


def test_catalog_modalities_win_over_weights() -> None:
    got = resolve_modalities({"audio": 0, "image": 1, "video": 1}, None)
    assert got == {"audio": 0, "image": 1, "video": 1}
    missing = resolve_modalities(None, None)
    assert missing == {"audio": 0, "image": 0, "video": 0}


def test_gpu_size_parts_add_up() -> None:
    p = gpu_size_parts(19.16, 29.82)
    assert p["gpu_gib"] == 29.8
    assert abs((p["weight_gib"] + p["kv_gib"]) - p["gpu_gib"]) < 0.05
    assert "+" in p["size_label"] and "=" in p["size_label"]
    disk = gpu_size_parts(17.6, None)
    assert disk["size_label"] == "17.6 GiB"


def _serve(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("SPARKSI_LOCK", str(tmp_path / "lock"))
    lk = Lock(tmp_path / "lock")
    lk.acquire(reason="test")
    (tmp_path / "unlock.token").write_text("PLANTED-UNLOCK-TOKEN-DO-NOT-LEAK\n")
    STATE.lock = lk
    httpd = DualServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, httpd.server_address[1]


def test_get_routes_do_not_leak_unlock_token(tmp_path, monkeypatch) -> None:
    httpd, port = _serve(tmp_path, monkeypatch)
    try:
        planted = "PLANTED-UNLOCK-TOKEN-DO-NOT-LEAK"
        for path in ("/", "/api/status", "/api/wanted"):
            req = Request(f"http://127.0.0.1:{port}{path}")
            with urlopen(req, timeout=30) as resp:
                body = resp.read().decode()
                ctype = resp.headers.get("Content-Type", "")
            if path == "/":
                assert "html" in ctype
                assert body.lstrip().lower().startswith("<!doctype html>")
            assert planted not in body
            assert "unlock_token" not in body
    finally:
        httpd.shutdown()


def test_dgx_models_json_is_cached(monkeypatch) -> None:
    import sparksi.models as models

    models._DISK_CACHE = None
    n = {"c": 0}

    def fetch():
        n["c"] += 1
        return [{"id": "x"}]

    monkeypatch.setattr("sparksi.models._dgx_models_fetch", fetch)
    a = models._dgx_models_json()
    b = models._dgx_models_json()
    assert a == b == [{"id": "x"}]
    assert n["c"] == 1
    models._DISK_CACHE = None


def test_list_models_vllm_tokens_zero_when_empty(tmp_path, monkeypatch, video_catalog) -> None:
    monkeypatch.setenv("SPARKSI_STATE_DIR", str(tmp_path))
    monkeypatch.setattr("sparksi.models._dgx_models_json", lambda: [])
    monkeypatch.setattr("sparksi.models._dir_size_gib", lambda p: 12.3)
    monkeypatch.setattr("sparksi.models.compute_apps", lambda: ([], None))
    rows = list_models()
    nano = next(r for r in rows if r.get("catalog_id") == "vision-a")
    assert nano.get("tok_label") == "0 out · 0 in"
    assert nano.get("life_gen") == 0
    video = next(r for r in rows if r.get("catalog_id") == video_catalog)
    assert video.get("tok_label") == "—"


def test_api_tokens_lists_seated_models(tmp_path, monkeypatch) -> None:
    httpd, port = _serve(tmp_path, monkeypatch)
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/tokens", timeout=15) as resp:
            body = json.loads(resp.read().decode())
        models = body.get("models") or {}
        for cid in ("brain-a", "vision-a", "counsel-a"):
            assert cid in models
            assert "life_gen" in models[cid]
            assert "life_prompt" in models[cid]
            assert "tok_label" in models[cid]
            assert "unlock_token" not in models[cid]
    finally:
        httpd.shutdown()


def test_ui_unlock_without_token(tmp_path, monkeypatch) -> None:
    httpd, port = _serve(tmp_path, monkeypatch)
    try:
        req = Request(
            f"http://127.0.0.1:{port}/api/unlock",
            data=json.dumps({"from_ui": True}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode())
        assert body.get("on") == "OFF"
        assert "unlock_token" not in body
    finally:
        httpd.shutdown()


def test_get_root_content_type_html(tmp_path, monkeypatch) -> None:
    httpd, port = _serve(tmp_path, monkeypatch)
    try:
        with urlopen(f"http://127.0.0.1:{port}/", timeout=30) as resp:
            assert "text/html" in resp.headers.get("Content-Type", "")
            text = resp.read().decode()
        assert "library" in text.lower()
        assert 'id="btn-load-head"' in text
        assert "load on head" not in text.lower()
        assert "spark-a" not in text
        assert "spark-b" not in text
        assert "/api/config" in text
        assert "trio" not in text.lower()
    finally:
        httpd.shutdown()
