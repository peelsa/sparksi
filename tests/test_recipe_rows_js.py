# SPDX-License-Identifier: Apache-2.0

"""Cell text and sort order for the recipe list (details view)."""

import json
import subprocess

from sparksi.serve import INDEX_HTML


def _script() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.split("/* recipe-rows-start */", 1)[1].split("/* recipe-rows-end */", 1)[0]


def _run(expr: str) -> object:
    body = f"process.stdout.write(JSON.stringify({expr}));"
    proc = subprocess.run(["node"], input=_script() + body, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


ROWS = """[
  {name: "b-fast", default: false, placement: "head", modified: "2026-11-10T10:00:00Z", ctx: 32768, max_num_seqs: 8, kv_cache_dtype: "fp8", speculative: "ngram"},
  {name: "A-main", default: true, placement: "both", modified: "2026-11-16T14:40:00Z", ctx: 131072, max_num_seqs: 4, kv_cache_dtype: "auto", speculative: ""},
  {name: "c-bare", default: false, placement: "", modified: "", ctx: null, max_num_seqs: null, kv_cache_dtype: "", speculative: ""},
  {name: "a-two", default: false, placement: "worker", modified: "2026-11-12T09:00:00Z", ctx: 100000, max_num_seqs: 4, kv_cache_dtype: "fp8_e4m3", speculative: ""}
]"""


def _names(key: str, direction: str) -> list[str]:
    got = _run(f"sortRecipes({ROWS}, {json.dumps(key)}, {json.dumps(direction)}).map(function (x) {{ return x.name; }})")
    assert isinstance(got, list)
    return got


def test_ctx_and_kv_text() -> None:
    assert _run('recipeCtxText({ctx: 131072})') == "128k"
    assert _run('recipeCtxText({ctx: 100000})') == "100k"
    assert _run('recipeCtxText({ctx: 512})') == "512"
    assert _run('recipeCtxText({ctx: null})') == ""
    assert _run('recipeKvText({ctx: 131072, kv_cache_dtype: "fp8"})') == "128k \u00b7 fp8"
    assert _run('recipeKvText({ctx: 32768, kv_cache_dtype: "auto"})') == "32k"
    assert _run('recipeKvText({ctx: null, kv_cache_dtype: "fp8"})') == "fp8"
    assert _run('recipeKvText({})') == ""


def test_nodes_and_cells() -> None:
    assert _run('recipeNodesText({placement: "both"})') == "2 (pair)"
    assert _run('recipeNodesText({placement: "pair"})') == "2 (pair)"
    assert _run('recipeNodesText({placement: "worker"})') == "1 (worker)"
    assert _run('recipeNodesText({placement: ""})') == ""
    cells = _run('recipeRowCells({max_num_seqs: 4, placement: "head", speculative: "ngram", ctx: null, kv_cache_dtype: ""})')
    assert cells == {"concurrency": "4", "kv": "", "nodes": "1 (head)", "spec": "ngram"}


def test_phone_line() -> None:
    line = _run('recipeRowLine({default: true, max_num_seqs: 4, ctx: 131072, kv_cache_dtype: "fp8", placement: "pair", speculative: "mtp"}, "Nov 16")')
    assert line == "Default \u00b7 c4 \u00b7 128k fp8 \u00b7 2 nodes \u00b7 spec \u00b7 Nov 16"
    assert _run('recipeRowLine({name: "x"}, "")') == ""


def test_default_first_then_name() -> None:
    assert _names("default", "asc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("default", "desc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("name", "asc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("name", "desc") == ["c-bare", "b-fast", "a-two", "A-main"]


def test_missing_values_sort_last_both_ways() -> None:
    assert _names("concurrency", "asc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("concurrency", "desc") == ["b-fast", "A-main", "a-two", "c-bare"]
    assert _names("ctx", "desc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("modified", "desc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("modified", "asc") == ["b-fast", "a-two", "A-main", "c-bare"]
    assert _names("nodes", "desc") == ["A-main", "a-two", "b-fast", "c-bare"]
    assert _names("spec", "asc") == ["b-fast", "A-main", "a-two", "c-bare"]


def test_sort_does_not_change_its_input() -> None:
    got = _run(f"(function () {{ var rows = {ROWS}; var before = rows.map(function (x) {{ return x.name; }}).join(); sortRecipes(rows, 'name', 'desc'); return rows.map(function (x) {{ return x.name; }}).join() === before; }})()")
    assert got is True
