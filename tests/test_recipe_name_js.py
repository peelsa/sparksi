# SPDX-License-Identifier: Apache-2.0

"""Suggested recipe names: model-quant-concurrency-kvcache-notes, from the recipe's settings."""

import json
import subprocess

from sparksi.serve import INDEX_HTML

EXAMPLE_CARD = {"id": "glm-flash", "title": "GLM 5.3 Flash EXL3 4bpw", "repo": "Org/GLM-5.3-Flash-EXL3", "format": "exl3"}
EXAMPLE_BODY = {"max_num_seqs": 4, "ctx": 131072, "kv_cache_dtype": "fp8", "placement": "both"}


def _script() -> str:
    html = INDEX_HTML.read_text(encoding="utf-8")
    return html.split("/* recipe-name-start */", 1)[1].split("/* recipe-name-end */", 1)[0]


def _run(body: str) -> object:
    proc = subprocess.run(["node"], input=_script() + body, text=True, capture_output=True, check=False)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _suggest(card: dict, body: dict, taken: list[str] | None = None) -> str:
    args = ", ".join(json.dumps(x) for x in (card, body, taken or []))
    got = _run(f"process.stdout.write(JSON.stringify(suggestRecipeName({args})));")
    assert isinstance(got, str)
    return got


def test_full_example_name() -> None:
    assert _suggest(EXAMPLE_CARD, EXAMPLE_BODY) == "glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node"


def test_parts() -> None:
    got = _run(
        """
        process.stdout.write(JSON.stringify({
          model: recipeModelShort({ repo: "Org/Qwen3-30B-A3B-FP8" }),
          modelTitle: recipeModelShort({ title: "Qwen 3 8B", id: "qwen" }),
          modelId: recipeModelShort({ id: "bge-m3" }),
          ggufModel: recipeModelShort({ repo: "x/Llama-3.1-8B-Instruct-Q4_K_M" }),
          q4km: recipeQuant({ repo: "x/Llama-3.1-8B-Instruct-Q4_K_M" }),
          fp8: recipeQuant({ repo: "Org/Qwen3-30B-A3B-FP8" }),
          bf16: recipeQuant({ format: "bf16" }),
          exl3: recipeQuant({ repo: "o/m-EXL3" }),
          none: recipeQuant({ id: "bge-m3" }),
          kv: recipeKv({ ctx: 131072, kv_cache_dtype: "fp8_e4m3" }),
          kvAuto: recipeKv({ ctx: 32768, kv_cache_dtype: "auto" }),
          kvDecimal: recipeKv({ ctx: 100000 }),
          kvCacheOnly: recipeKv({ kv_cache_dtype: "fp8" }),
          kvNone: recipeKv({}),
          notes: recipeNotes({ placement: "both", speculative_method: "ngram" }),
          notesHead: recipeNotes({ placement: "head" })
        }));
        """
    )
    assert got == {
        "model": "qwen3-30b-a3b",
        "modelTitle": "qwen-3-8b",
        "modelId": "bge-m3",
        "ggufModel": "llama-3.1-8b-instruct",
        "q4km": "q4km",
        "fp8": "fp8",
        "bf16": "bf16",
        "exl3": "exl3",
        "none": "",
        "kv": "128k-fp8e4m3",
        "kvAuto": "32k",
        "kvDecimal": "100k",
        "kvCacheOnly": "fp8",
        "kvNone": "",
        "notes": ["2node", "spec"],
        "notesHead": [],
    }


def test_missing_parts_are_skipped_cleanly() -> None:
    assert _suggest({"id": "bge-m3", "title": "BGE M3", "format": "bf16"}, {"ctx": 8192}) == "bge-m3-bf16-8k"
    assert _suggest({"id": "bge-m3"}, {}) == "bge-m3"
    assert _suggest({}, {}) == "recipe"


def test_clash_appends_a_number() -> None:
    name = "glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node"
    assert _suggest(EXAMPLE_CARD, EXAMPLE_BODY, [name]) == name + "-2"
    assert _suggest(EXAMPLE_CARD, EXAMPLE_BODY, [name, name + "-2"]) == name + "-3"


def test_long_names_cut_whole_parts_and_stay_valid() -> None:
    card = {"repo": "o/" + "very-long-model-name-" * 4 + "X"}
    name = _suggest(card, EXAMPLE_BODY)
    assert len(name) <= 80
    assert not name.endswith("-")
    clash = _suggest(card, EXAMPLE_BODY, [name])
    assert len(clash) <= 80 and clash.endswith("-2")
    # The page's own RECIPE_NAME_RE (inside the extracted block), not a copy of it.
    got = _run(f"process.stdout.write(JSON.stringify([RECIPE_NAME_RE.test({json.dumps(name)}), RECIPE_NAME_RE.test({json.dumps(clash)})]));")
    assert got == [True, True]


def test_page_name_rule_matches_the_server() -> None:
    from sparksi.recipes.store import valid_recipe_name

    samples = ["glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node", "a" * 80, "a" * 81, ".hidden", "-x", "a b", "a/b", "A_b.1", ""]
    got = _run(f"process.stdout.write(JSON.stringify({json.dumps(samples)}.map(function (n) {{ return RECIPE_NAME_RE.test(n); }})));")
    assert got == [valid_recipe_name(n) for n in samples]


def test_policy_check_is_a_hint_only() -> None:
    got = _run(
        """
        process.stdout.write(JSON.stringify([
          recipeNameFollowsPolicy("glm-5.3-flash-exl3-4bpw-c4-128k-fp8-2node"),
          recipeNameFollowsPolicy("qwen3-8b-fp8-c8-32k"),
          recipeNameFollowsPolicy("fast"),
          recipeNameFollowsPolicy("Qwen-c4-32k"),
          recipeNameFollowsPolicy("qwen-c4"),
          recipeNameFollowsPolicy("")
        ]));
        """
    )
    assert got == [True, True, False, False, False, False]


def test_quant_is_dropped_from_the_model_even_with_a_trailing_word() -> None:
    out = _run(
        "var card = {title: 'GLM-5.3-Flash EXL3 4bpw instruct'};"
        "process.stdout.write(JSON.stringify([recipeModelShort(card)]))"
    )
    assert out == ["glm-5.3-flash-instruct"]

    body = {"max_num_seqs": 4, "ctx": 850000, "kv_cache_dtype": "fp8", "placement": "both"}
    assert (
        _suggest({"title": "GLM-5.3-Flash EXL3 4bpw instruct", "format": "exl3"}, body)
        == "glm-5.3-flash-instruct-exl3-4bpw-c4-850k-fp8-2node"
    )


def test_iq_quants() -> None:
    out = _run(
        "process.stdout.write(JSON.stringify(["
        'recipeQuant({repo: "x/M-27B-IQ2-mix-MTP"}),'
        'recipeQuant({repo: "x/M-IQ4_XS"}),'
        'recipeQuant({repo: "x/M-IQ3_M"}),'
        'recipeModelShort({repo: "x/M-27B-IQ4_XS"}),'
        'recipeModelShort({repo: "x/M-27B-IQ2-mix-MTP"})]))'
    )
    assert out == ["iq2-mix", "iq4xs", "iq3m", "m-27b", "m-27b-mtp"]


def test_q8_0_is_q8() -> None:
    out = _run(
        "process.stdout.write(JSON.stringify(["
        'recipeKvDtype("q8_0"),'
        'recipeKvDtype("fp8_e4m3"),'
        'recipeKvDtype("auto"),'
        'recipeKv({ctx: 262144, kv_cache_dtype: "q8_0"}),'
        'recipeQuant({repo: "x/M-Q8_0"}),'
        'recipeQuant({repo: "x/M-Q4_1"})]))'
    )
    assert out == ["q8", "fp8e4m3", "", "256k-q8", "q8", "q41"]


PINNED_ITEM = {
    "kind": "pinned",
    "placement": "head",
    "ctx": 262144,
    "max_num_seqs": 1,
    "kv_cache_dtype": "q8_0",
    "speculative": "draft-mtp",
    "model_hint": "m-27b",
    "quant_hint": "M-27B-1.0-IQ2-mix-MTP",
}
PINNED_CARD = {"id": "m", "title": "M 27B IQ2 MTP", "format": "gguf"}


def _from_row(body: dict, item: dict) -> str:
    args = ", ".join(json.dumps(x) for x in (PINNED_CARD, body, item))
    got = _run(f"var s = recipeNameSource({args}); process.stdout.write(JSON.stringify(suggestRecipeName(s.card, s.body, [])));")
    assert isinstance(got, str)
    return got


def test_pinned_rows_fill_the_name() -> None:
    assert _from_row({}, PINNED_ITEM) == "m-27b-iq2-mix-c1-256k-q8-spec"
    # What the form says wins over what the command says.
    assert _from_row({"ctx": 8192}, PINNED_ITEM) == "m-27b-iq2-mix-c1-8k-q8-spec"
    # Only a pinned row is read.
    assert _from_row({}, {**PINNED_ITEM, "kind": "generated"}) == "m-27b-mtp-iq2"


def test_pinned_rows_from_each_kind_of_command() -> None:
    """vLLM, start.sh, TensorFold and llama.cpp rows, as the server builds them."""
    hub = "Org/Model-30B-exl3-4bpw-instruct"
    rows = [
        ({"id": "vl", "title": "Model NVFP4", "format": "nvfp4"},
         {"placement": "both", "ctx": 262144, "max_num_seqs": 8, "kv_cache_dtype": "fp8", "speculative": "mtp",
          "model_hint": "nvidia/Model-8B-Flash-NVFP4", "quant_hint": "Model-8B-Flash-NVFP4"},
         "model-8b-flash-nvfp4-c8-256k-fp8-2node-spec"),
        ({"id": "sh", "title": "Model EXL3 4bpw instruct", "format": "exl3"},
         {"placement": "both", "ctx": 850000, "max_num_seqs": 4, "kv_cache_dtype": "fp8", "speculative": "dflash",
          "model_hint": hub, "quant_hint": "exl3 Model-30B-exl3-4bpw-instruct"},
         "model-30b-instruct-exl3-4bpw-c4-850k-fp8-2node-spec"),
        ({"id": "tf", "title": "Model EXL3 TensorFold", "format": "exl3"},
         {"placement": "both", "ctx": 1048576, "max_num_seqs": 4, "kv_cache_dtype": "fp8", "speculative": "dflash",
          "model_hint": hub, "quant_hint": "Model-30B-exl3-4bpw-instruct"},
         "model-30b-instruct-exl3-4bpw-c4-1024k-fp8-2node-spec"),
        ({"id": "gg", "title": "Model 27B IQ2 MTP", "format": "gguf"},
         {"placement": "head", "ctx": 262144, "max_num_seqs": 1, "kv_cache_dtype": "q8_0", "speculative": "draft-mtp",
          "model_hint": "model-27b", "quant_hint": "Model-27B-1.0-IQ2-mix-MTP"},
         "model-27b-iq2-mix-c1-256k-q8-spec"),
    ]
    for card, row, want in rows:
        args = ", ".join(json.dumps(x) for x in (card, {}, {"kind": "pinned", **row}))
        got = _run(f"var s = recipeNameSource({args}); process.stdout.write(JSON.stringify(suggestRecipeName(s.card, s.body, [])));")
        assert got == want, (card["id"], got)
