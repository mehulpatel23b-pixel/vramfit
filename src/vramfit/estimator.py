"""Core maths: turn a model + device + workload into a fit verdict and speed."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .hardware import GPU, cheapest_options
from .model import ModelSpec, normalize_quant

GB = 1024.0**3

# Verdict thresholds, expressed as a fraction of usable VRAM.
COMFORTABLE = 0.85
TIGHT = 1.0


@dataclass
class PlanOptions:
    """Everything that changes the answer besides the model and the device."""

    quant: str = "q4"
    context: int = 8192
    kv_quant: str = "f16"
    batch: int = 1
    # Runtime scratch: graph buffers, CUDA context, fragmentation. Fraction of
    # weight bytes, floored at 0.5GB. Empirically 6-12% for llama.cpp/GGUF.
    activation_ratio: float = 0.08
    activation_floor_gb: float = 0.5
    # Fraction of theoretical bandwidth actually achieved during decode.
    # Memory-bound decode typically reaches 60-80% of peak.
    bandwidth_efficiency: float = 0.70
    # Leave this fraction of usable VRAM untouched to avoid OOM at long context.
    safety_margin: float = 0.03


@dataclass
class FitPlan:
    model: ModelSpec
    gpu: GPU
    options: PlanOptions
    weights_gb: float = 0.0
    kv_cache_gb: float = 0.0
    activation_gb: float = 0.0
    total_gb: float = 0.0
    usable_gb: float = 0.0
    verdict: str = ""
    headroom_gb: float = 0.0
    utilisation: float = 0.0
    gpu_layers_pct: float = 100.0
    tokens_per_second: float = 0.0
    max_context: int = 0
    notes: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)

    @property
    def fits(self) -> bool:
        return self.verdict in ("comfortable", "tight")

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model.name,
            "gpu": self.gpu.name,
            "quant": normalize_quant(self.options.quant),
            "context": self.options.context,
            "verdict": self.verdict,
            "fits": self.fits,
            "vram": {
                "weights_gb": round(self.weights_gb, 2),
                "kv_cache_gb": round(self.kv_cache_gb, 2),
                "activation_gb": round(self.activation_gb, 2),
                "total_gb": round(self.total_gb, 2),
                "usable_gb": round(self.usable_gb, 2),
                "headroom_gb": round(self.headroom_gb, 2),
                "utilisation_pct": round(self.utilisation * 100, 1),
            },
            "performance": {
                "gpu_layers_pct": round(self.gpu_layers_pct, 1),
                "tokens_per_second": round(self.tokens_per_second, 1),
                "max_context_at_quant": self.max_context,
            },
            "notes": self.notes,
            "alternatives": self.alternatives,
        }


def _activation_bytes(model: ModelSpec, options: PlanOptions) -> float:
    raw = model.weights_bytes(options.quant) * options.activation_ratio
    floor = options.activation_floor_gb * GB
    # Batch multiplies the KV/activation working set, not the model weights.
    return max(raw, floor) * max(1, options.batch) ** 0.5


def estimate(
    model: ModelSpec,
    gpu: GPU,
    options: Optional[PlanOptions] = None,
    include_alternatives: bool = True,
) -> FitPlan:
    """Compute the fit plan for one model on one device."""
    options = options or PlanOptions()
    normalise = normalize_quant

    weights = model.weights_bytes(options.quant)
    kv = model.kv_bytes(options.quant, options.context, options.kv_quant)
    kv *= max(1, options.batch)
    activation = _activation_bytes(model, options)

    total_bytes = weights + kv + activation
    usable = gpu.usable_gb * (1.0 - options.safety_margin) * GB

    plan = FitPlan(
        model=model,
        gpu=gpu,
        options=options,
        weights_gb=weights / GB,
        kv_cache_gb=kv / GB,
        activation_gb=activation / GB,
        total_gb=total_bytes / GB,
        usable_gb=usable / GB,
    )

    ratio = total_bytes / usable if usable else float("inf")
    plan.utilisation = ratio
    plan.headroom_gb = (usable - total_bytes) / GB

    if ratio <= COMFORTABLE:
        plan.verdict = "comfortable"
    elif ratio <= TIGHT:
        plan.verdict = "tight"
    elif ratio <= 1.35:
        plan.verdict = "offload"
    else:
        plan.verdict = "no"

    if plan.verdict in ("comfortable", "tight"):
        plan.gpu_layers_pct = 100.0
    else:
        # Fraction of layers that fit if we spill the remainder to CPU RAM.
        budget = max(0.0, usable - activation - kv * 0.15)
        plan.gpu_layers_pct = min(99.0, 100.0 * budget / weights) if weights else 0.0

    plan.tokens_per_second = _decode_speed(model, gpu, options, plan)
    plan.max_context = _max_context(model, gpu, options)
    plan.notes = _notes(plan)

    if include_alternatives:
        plan.alternatives = [
            f"{g.name} ({g.usable_gb:.0f}GB usable)"
            for g in cheapest_options(total_bytes / GB, limit=4)
            if g.name != gpu.name
        ]

    return plan


def _decode_speed(
    model: ModelSpec, gpu: GPU, options: PlanOptions, plan: FitPlan
) -> float:
    """Estimate decode throughput (tokens/sec) for a memory-bound workload.

    Decode reads the whole weight set once per token. The ceiling is therefore
    effective_bandwidth / bytes_read. With partial offload, the CPU-resident
    share runs at a fraction of GPU bandwidth.
    """
    if plan.gpu_layers_pct <= 0:
        return 0.0

    on_gpu = plan.gpu_layers_pct / 100.0
    # Speed is driven by bytes actually READ per token, which for MoE is the
    # active-parameter share, not the total resident weight set.
    read_gb = plan.weights_gb * (model.read_params / model.params)
    gpu_bytes = read_gb * GB * on_gpu
    cpu_bytes = read_gb * GB * (1.0 - on_gpu)

    eff_bw = gpu.bandwidth_gbs * options.bandwidth_efficiency
    if eff_bw <= 0:
        return 0.0

    gpu_time = gpu_bytes / (eff_bw * 1e9)

    if cpu_bytes > 0:
        # Quadratic penalty: PCIe transfers + CPU compute during partial offload.
        cpu_bw = 45.0 * options.bandwidth_efficiency  # GB/s over PCIe 4.0 x16
        cpu_time = cpu_bytes / (cpu_bw * 1e9)
        gpu_time += cpu_time * 2.5

    tps = 1.0 / gpu_time if gpu_time > 0 else 0.0

    # Unity/unified-memory devices lose throughput to shared-memory contention.
    if gpu.unified:
        tps *= 0.85

    # Long contexts add attention work on top of the weight read.
    if options.context > 32768:
        tps *= 0.85

    return round(tps, 1)


def _max_context(model: ModelSpec, gpu: GPU, options: PlanOptions) -> int:
    """Largest context that still fits in usable VRAM at this quantisation."""
    weights = model.weights_bytes(options.quant)
    activation = _activation_bytes(model, options)
    usable = gpu.usable_gb * (1.0 - options.safety_margin) * GB
    remaining = usable - weights - activation
    if remaining <= 0:
        return 0

    per_token = model.kv_bytes(options.quant, 1, options.kv_quant)
    per_token *= max(1, options.batch)
    if per_token <= 0:
        return 0
    ctx = int(remaining / per_token)
    # Never plan beyond the context the model was trained on.
    ctx = min(ctx, model.native_context)
    # Round down to a practical ladder value.
    for step in (524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024):
        if ctx >= step:
            return step
    return max(0, ctx)


def _notes(plan: FitPlan) -> List[str]:
    notes: List[str] = []
    q = normalize_quant(plan.options.quant)

    notes.append(
        f"weights {plan.weights_gb:.2f}GB + kv {plan.kv_cache_gb:.2f}GB "
        f"+ scratch {plan.activation_gb:.2f}GB"
    )

    if plan.verdict == "comfortable":
        notes.append(
            f"fits with {plan.headroom_gb:.2f}GB spare "
            f"({plan.utilisation * 100:.0f}% of usable VRAM used)"
        )
    elif plan.verdict == "tight":
        notes.append(
            "fits, but under 15% headroom. Long context or a second app on the "
            "desktop can trigger an OOM. Drop to q3 or halve the context."
        )
    elif plan.verdict == "offload":
        notes.append(
            f"does not fit fully. About {plan.gpu_layers_pct:.0f}% of layers can "
            f"live on the GPU; the rest runs from system RAM and is much slower."
        )
    else:
        notes.append(
            "does not fit this device at all, and offloading every layer to "
            "system RAM is not a realistic speed. Needs a smaller model, a "
            "lower quant, or bigger hardware."
        )

    if plan.max_context and plan.max_context < plan.options.context:
        notes.append(
            f"at {q} the largest workable context here is {plan.max_context} "
            f"tokens, not {plan.options.context}"
        )

    if plan.options.context > plan.model.native_context:
        notes.append(
            f"{plan.model.name} was trained for {plan.model.native_context} tokens; "
            f"planning at {plan.options.context} overstates the KV cache you need"
        )

    if plan.options.kv_quant == "f16" and plan.kv_cache_gb > 2.0:
        saved = plan.kv_cache_gb * 0.5
        notes.append(
            f"KV cache in f16 dominates. q8 KV would free ~{saved:.1f}GB."
        )

    if plan.model.is_moe:
        notes.append(
            f"mixture-of-experts: all {plan.model.params / 1e9:.0f}B params must be "
            f"resident in memory, but only {plan.model.read_params / 1e9:.0f}B are "
            "read per token, so real throughput is higher than a dense model of "
            "the same size"
        )

    if q in ("f32", "f16", "bf16"):
        notes.append("unquantised weights are 3-4x larger than q4 for no quality gain at inference")

    return notes
