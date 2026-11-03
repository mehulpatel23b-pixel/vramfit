"""Model definitions and parameter-count parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional

# Bytes per weight for the quantisation formats that matter in practice.
# Keys are the canonical label; values are effective bits per weight.
QUANTS: Dict[str, float] = {
    "f32": 32.0,
    "f16": 16.0,
    "bf16": 16.0,
    "q8": 8.0,
    "q6": 6.5625,
    "q5": 5.5,
    "q4": 4.5,
    "q3": 3.4375,
    "q2": 2.5625,
    "iq2": 2.3125,
}

QUANT_ALIASES = {
    "fp32": "f32",
    "float32": "f32",
    "fp16": "f16",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "int8": "q8",
    "q8_0": "q8",
    "int6": "q6",
    "q6_k": "q6",
    "int5": "q5",
    "q5_k_m": "q5",
    "q5_k_s": "q5",
    "int4": "q4",
    "q4_0": "q4",
    "q4_k_m": "q4",
    "q4_k_s": "q4",
    "gguf-q4": "q4",
    "q3_k_m": "q3",
    "q3_k_s": "q3",
    "q2_k": "q2",
    "iq2_xxs": "iq2",
}

_PARAM_RE = re.compile(r"^\s*([0-9]*\.?[0-9]+)\s*([bmk])?\s*$", re.IGNORECASE)

_SUFFIX = {"": 1.0, "b": 1e9, "m": 1e6, "k": 1e3}


def normalize_quant(name: str) -> str:
    """Map any common quant spelling to a canonical key in QUANTS."""
    key = name.strip().lower().replace("-", "_")
    if key in QUANTS:
        return key
    if key in QUANT_ALIASES:
        return QUANT_ALIASES[key]
    raise ValueError(
        f"unknown quantisation {name!r}. known: {', '.join(sorted(QUANTS))}"
    )


def parse_params(text: str) -> float:
    """Parse a human parameter count into an absolute count.

    Accepts '7b', '7B', '1.5b', '70', '405m', '3k'. A bare number is treated
    as billions, which is how models are named in the wild ('qwen 14' = 14B).
    """
    if isinstance(text, (int, float)):
        return float(text) * 1e9
    m = _PARAM_RE.match(str(text))
    if not m:
        raise ValueError(f"cannot parse parameter count {text!r}")
    value = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if not suffix:
        return value * 1e9
    return value * _SUFFIX[suffix]


@dataclass(frozen=True)
class ModelSpec:
    """A model whose memory footprint we want to compute.

    layers/heads are only needed for the KV-cache estimate; when they are not
    supplied the planner falls back to a shape inferred from the parameter
    count, which is accurate enough to plan a purchase or a download.
    """

    name: str
    params: float
    n_layers: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    hidden_size: int = 0
    vocab_size: int = 151936
    # Trained context length. Planning beyond this is meaningless even if the
    # card has memory for it, so the estimator caps max_context here.
    native_context: int = 131072
    native_quant: Optional[str] = None
    # Parameters actually read per token. For dense models this equals params.
    # For mixture-of-experts it is far smaller: every expert must be resident
    # in memory (so `params` drives the VRAM estimate) but only the routed
    # experts are read during decode (so this drives the speed estimate).
    active_params: Optional[float] = None

    @property
    def read_params(self) -> float:
        """Parameters read per generated token (dense == total)."""
        return self.active_params or self.params

    @property
    def is_moe(self) -> bool:
        return self.active_params is not None and self.active_params < self.params * 0.9

    @property
    def has_attention_shape(self) -> bool:
        """True when the KV cache can be computed from real architecture."""
        return self.n_layers > 0 and self.n_kv_heads > 0 and self.head_dim > 0

    def weights_bytes(self, quant: str) -> float:
        """Raw weight footprint in bytes for a quantisation format."""
        bits = QUANTS[normalize_quant(quant)]
        return self.params * bits / 8.0

    def kv_bytes(self, quant: str, context: int, kv_quant: str = "f16") -> float:
        """KV cache size in bytes.

        kv_bytes_per_element = n_layers * n_kv_heads * head_dim * 2 (K and V)
        total = that * context * bytes_per_element
        """
        shape = self.attention_shape()
        n_layers, n_kv_heads, head_dim = shape
        elem_bits = QUANTS[normalize_quant(kv_quant)]
        per_token = n_layers * n_kv_heads * head_dim * 2 * (elem_bits / 8.0)
        return per_token * context

    def attention_shape(self) -> tuple:
        """Return (n_layers, n_kv_heads, head_dim), inferring when unknown.

        The inference uses the standard transformer relation
        params ~= 12 * n_layers * hidden^2, then head_dim = hidden / heads with
        a grouped-query ratio of 4 KV heads per attention group.
        """
        if self.has_attention_shape:
            return self.n_layers, self.n_kv_heads, self.head_dim

        hidden = self.hidden_size or int(round((self.params / 12.0 / 32) ** 0.5))
        hidden = max(hidden, 64)
        layers = self.n_layers or max(8, int(round(self.params / (12.0 * hidden**2))))
        head_dim = self.head_dim or (hidden // 32 if hidden >= 32 * 64 else 128)
        head_dim = max(head_dim, 32)
        heads = max(1, hidden // head_dim)
        n_kv_heads = self.n_kv_heads or max(1, heads // 4)
        return layers, n_kv_heads, head_dim


# Well-known architectures used by `--model` shortcuts.
MODEL_PRESETS: Dict[str, ModelSpec] = {
    "llama3-8b": ModelSpec("Llama 3 8B", 8.03e9, 32, 8, 128, 4096, 128256, native_context=8192),
    "llama3-70b": ModelSpec("Llama 3 70B", 70.6e9, 80, 8, 128, 8192, 128256, native_context=8192),
    "qwen2.5-7b": ModelSpec("Qwen2.5 7B", 7.62e9, 28, 4, 128, 3584, 152064, native_context=32768),
    "qwen2.5-32b": ModelSpec("Qwen2.5 32B", 32.8e9, 64, 8, 128, 5120, 152064, native_context=32768),
    "qwen2.5-72b": ModelSpec("Qwen2.5 72B", 72.7e9, 80, 8, 128, 8192, 152064, native_context=32768),
    "mistral-7b": ModelSpec("Mistral 7B", 7.24e9, 32, 8, 128, 4096, 32000, native_context=32768),
    "mixtral-8x7b": ModelSpec(
        "Mixtral 8x7B", 46.7e9, 32, 8, 128, 4096, 32000,
        native_context=32768, active_params=12.9e9,
    ),
    "phi3-mini": ModelSpec("Phi-3 Mini", 3.82e9, 32, 32, 96, 3072, 32064, native_context=4096),
    "gemma2-9b": ModelSpec("Gemma 2 9B", 9.24e9, 42, 8, 256, 3584, 256000, native_context=8192),
    "deepseek-v3": ModelSpec(
        "DeepSeek V3", 671e9, 61, 128, 128, 7168, 129280,
        native_context=65536, active_params=37e9,
    ),
}


def resolve_model(text: str) -> ModelSpec:
    """Resolve a preset name or a raw parameter count into a ModelSpec."""
    key = text.strip().lower()
    if key in MODEL_PRESETS:
        return MODEL_PRESETS[key]
    params = parse_params(text)
    return ModelSpec(name=f"custom {text}", params=params)
