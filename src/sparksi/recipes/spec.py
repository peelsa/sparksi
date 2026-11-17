# SPDX-License-Identifier: Apache-2.0

"""Values an engine builder needs. Hosts are filled in by the renderer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Launch:
    role: str
    rank: int
    host: str
    argv: list[str]
    env: dict[str, str]
    container: str
    stop_argv: list[str]
    bind: str

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "rank": self.rank,
            "host": self.host,
            "argv": list(self.argv),
            "env": dict(self.env),
            "container": self.container,
            "stop_argv": list(self.stop_argv),
            "bind": self.bind,
        }


@dataclass
class Built:
    nodes: list[Launch]
    order: list[str]
    experimental: bool = False
    experimental_reason: str = ""
    health_paths: list[str] = field(default_factory=lambda: ["/health", "/v1/models"])

    def to_dict(self) -> dict:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "order": list(self.order),
            "experimental": self.experimental,
            "experimental_reason": self.experimental_reason,
            "health_paths": list(self.health_paths),
            "argv_preview": {node.role: list(node.argv) for node in self.nodes},
        }


@dataclass
class Spec:
    """One render. Addresses come from the caller, never from the engine module."""

    model_id: str
    placement: str
    port: int
    paths: dict[str, str]
    hosts: dict[str, str]
    binds: dict[str, str]
    fabric_ip: dict[str, str]
    fabric_iface: dict[str, str]
    fabric_hcas: dict[str, list[str]]
    image: str = ""
    hf_cache: str = "/home/USER/.cache/huggingface"
    ctx: int = 8192
    gpu_mem_util: float = 0.80
    max_num_seqs: int = 4
    kv_cache_dtype: str = "fp8"
    prefix_cache: bool = False
    kv_cache_memory_bytes: int | None = None
    # Structured speculative decoding. A bool is still accepted: True is MTP
    # with one token, False is off. A dict is method, draft model, token count.
    speculative: bool | dict | None = None
    tool_parser: str = ""
    reasoning_parser: str = ""
    extra_flags: list[str] = field(default_factory=list)
    hints: dict[str, bool] = field(default_factory=dict)
    master_port: int = 29501
    dist_port: int = 50000
    scripts: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    forbid_prefix: bool = False
    binary: str = ""
    ld_library_path: str = ""
    max_num_batched_tokens: int | None = None
    attention_backend: str = ""
    mamba_ssm_dtype: str = ""
    quantization: str = ""
    load_format: str = ""
    limit_mm_per_prompt: str = ""
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    alias: str = ""
    reasoning: str = ""
