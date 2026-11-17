# SPDX-License-Identifier: Apache-2.0

"""Pure builders. Each returns argv per node and does not read a host out of the environment."""

from sparksi.recipes.engines.custom import build as build_custom
from sparksi.recipes.engines.llamacpp import build as build_llamacpp
from sparksi.recipes.engines.sglang import build as build_sglang
from sparksi.recipes.engines.trtllm import build as build_trtllm
from sparksi.recipes.engines.vllm import build as build_vllm

BUILDERS = {
    "vllm": build_vllm,
    "sglang": build_sglang,
    "trtllm": build_trtllm,
    "llamacpp": build_llamacpp,
    "custom": build_custom,
}
