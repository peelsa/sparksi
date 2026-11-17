# SPDX-License-Identifier: Apache-2.0

"""Settings read out of pinned launch commands, for the recipe list columns and the suggested name."""

from sparksi.recipes.pinned_settings import pinned_settings

HUB = "/root/.cache/huggingface/hub/models--Org--Model-30B-EXL3-4bpw-instruct/snapshots/abc"


def _pinned(engine: str, args: list[str], env: dict[str, str] | None = None, **top) -> dict:
    return {"kind": "pinned", "engine": engine, "roles": {"head": {"args": args, "env": env or {}}}, **top}


def test_vllm_flags() -> None:
    recipe = _pinned(
        "vllm",
        [
            "vllm", "serve", "/models",
            "--served-model-name", "model-30b",
            "--max-model-len", "262144",
            "--max-num-seqs", "8",
            "--kv-cache-dtype", "fp8",
            "--tensor-parallel-size", "2",
            "--quantization", "fp8",
            "--speculative-config", '{"method":"mtp","num_speculative_tokens":3}',
        ],
    )
    assert pinned_settings(recipe) == {
        "ctx": 262144,
        "max_num_seqs": 8,
        "tp": 2,
        "kv_cache_dtype": "fp8",
        "speculative": "mtp",
        "model": "model-30b",
        "quant": "fp8",
    }


def test_vllm_repo_wins_over_the_model_path() -> None:
    recipe = _pinned("vllm", ["vllm", "serve", HUB, "--max-model-len=32768", "-tp", "2"], repo="nvidia/Model-8B-NVFP4")
    got = pinned_settings(recipe)
    assert got["ctx"] == 32768 and got["tp"] == 2
    assert got["model"] == "nvidia/Model-8B-NVFP4"
    assert got["quant"] == "Model-8B-NVFP4"


def test_llamacpp_flags() -> None:
    recipe = _pinned(
        "llamacpp",
        [
            "-m", "/models/Model-27B-1.0-IQ2-mix-MTP.gguf", "--alias", "model-27b",
            "-c", "262144", "-np", "1", "-ctk", "q8_0", "-ctv", "q8_0",
            "--spec-type", "draft-mtp", "--spec-draft-n-max", "2",
        ],
    )
    assert pinned_settings(recipe) == {
        "ctx": 262144,
        "max_num_seqs": 1,
        "kv_cache_dtype": "q8_0",
        "speculative": "draft-mtp",
        "model": "model-27b",
        "quant": "Model-27B-1.0-IQ2-mix-MTP",
    }


def test_llamacpp_file_name_is_the_model_without_an_alias() -> None:
    got = pinned_settings(_pinned("llamacpp", ["-m", "/models/Model-8B-Q4_K_M.gguf", "-c", "8192"]))
    assert got == {"ctx": 8192, "model": "Model-8B-Q4_K_M", "quant": "Model-8B-Q4_K_M"}


def test_tensorfold_flags_and_env() -> None:
    recipe = _pinned(
        "tensorfold",
        [
            "tensorfold", "serve", HUB, "--tp", "2", "--name", "Model-EXL3",
            "--drafter", "/root/.cache/huggingface/hub/models--Org--Model-DFlash2/snapshots/def",
            "--context", "1048576", "--parallel", "4", "--max-tokens", "32768",
        ],
        {"TF_GLM_KV": "fp8", "TF_GLM_MTP": "auto"},
    )
    assert pinned_settings(recipe) == {
        "ctx": 1048576,
        "max_num_seqs": 4,
        "tp": 2,
        "kv_cache_dtype": "fp8",
        "speculative": "dflash",
        "model": "Org/Model-30B-EXL3-4bpw-instruct",
        "quant": "Model-30B-EXL3-4bpw-instruct",
    }


def test_start_sh_env() -> None:
    env = {
        "MODEL_DIR": HUB,
        "QUANTIZATION": "exl3",
        "MAX_NUM_SEQS": "4",
        "MAX_MODEL_LEN": "850000",
        "KV_CACHE_DTYPE": "fp8",
        "TP": "2",
        "SPEC_METHOD": "dflash",
        "SERVED_MODEL_NAME": "Model-EXL3",
    }
    assert pinned_settings(_pinned("other", ["/start.sh"], env)) == {
        "ctx": 850000,
        "max_num_seqs": 4,
        "tp": 2,
        "kv_cache_dtype": "fp8",
        "speculative": "dflash",
        "model": "Org/Model-30B-EXL3-4bpw-instruct",
        "quant": "exl3 Model-30B-EXL3-4bpw-instruct",
    }


def test_top_level_fields_win_and_unknown_keys_are_left_out() -> None:
    recipe = _pinned("vllm", ["vllm", "serve", "/models", "--max-model-len", "4096", "--kv-cache-dtype", "auto"], ctx=8192)
    assert pinned_settings(recipe) == {"ctx": 8192}
    assert pinned_settings({"kind": "pinned", "ctx": 2048}) == {"ctx": 2048}
    assert pinned_settings({"kind": "pinned", "roles": {"head": {"args": ["--max-num-seqs", "0"], "env": {"TP": "x"}}}}) == {}


def test_junk_shapes_read_as_empty() -> None:
    """A hand-edited or damaged file never raises: odd shapes are simply not read."""
    junk = [
        None,
        "pinned",
        {"kind": "pinned", "roles": "head"},
        {"kind": "pinned", "roles": {"head": "vllm serve"}},
        {"kind": "pinned", "roles": {"head": {"args": "vllm serve /models --max-model-len 4096", "env": ["TP=2"]}}},
        {"kind": "pinned", "roles": {"head": {"args": 42, "env": 7}}, "repo": 5, "rendezvous_ports": "x"},
        {"kind": "pinned", "roles": {"head": {"args": {"--max-model-len": 4096}, "env": None}}},
    ]
    for recipe in junk:
        assert pinned_settings(recipe) == {}, recipe
    # Odd items inside a list still read as text.
    got = pinned_settings({"kind": "pinned", "roles": {"head": {"args": [1, None, "--max-model-len", "4096"], "env": {"TP": 2}}}})
    assert got == {"ctx": 4096, "tp": 2}
