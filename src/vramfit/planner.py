"""Planning layer: pick the best quantisation for a device, or the device for a model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .estimator import FitPlan, PlanOptions, estimate
from .hardware import GPUS, GPU, cheapest_options, find_gpu
from .model import ModelSpec

# Ordered best-quality-first. Planning walks this list and stops at the first
# quantisation that fits, which is what a developer actually wants to know:
# "how much quality can this card afford?"
QUALITY_ORDER = ["f16", "q8", "q6", "q5", "q4", "q3", "q2", "iq2"]

# Below this, output quality degrades noticeably on reasoning tasks.
QUALITY_FLOOR = "q3"


@dataclass
class Recommendation:
    """A chosen plan plus the evidence behind it."""

    plan: FitPlan
    sweep: List[FitPlan] = field(default_factory=list)
    model: Optional[ModelSpec] = None
    gpu: Optional[GPU] = None

    @property
    def quant(self) -> str:
        return self.plan.options.quant

    @property
    def quality_warning(self) -> Optional[str]:
        idx = QUALITY_ORDER.index(self.quant) if self.quant in QUALITY_ORDER else -1
        floor = QUALITY_ORDER.index(QUALITY_FLOOR)
        if idx > floor:
            return (
                f"{self.quant} is below the practical quality floor "
                f"({QUALITY_FLOOR}). Expect visible degradation on reasoning, "
                "code and long-context tasks."
            )
        return None

    def to_dict(self) -> Dict[str, object]:
        payload = self.plan.to_dict()
        payload["quality_warning"] = self.quality_warning
        payload["quant_table"] = [
            {
                "quant": p.options.quant,
                "total_gb": round(p.total_gb, 2),
                "verdict": p.verdict,
                "tokens_per_second": round(p.tokens_per_second, 1),
            }
            for p in self.sweep
        ]
        return payload


def plan_fit(
    model: ModelSpec, gpu: GPU, options: Optional[PlanOptions] = None
) -> FitPlan:
    """Single-point plan at the quantisation given in options."""
    return estimate(model, gpu, options)


def sweep_quants(
    model: ModelSpec, gpu: GPU, options: Optional[PlanOptions] = None
) -> List[FitPlan]:
    """Evaluate every quantisation, best quality first."""
    base = options or PlanOptions()
    out: List[FitPlan] = []
    for quant in QUALITY_ORDER:
        opts = PlanOptions(
            quant=quant,
            context=base.context,
            kv_quant=base.kv_quant,
            batch=base.batch,
            activation_ratio=base.activation_ratio,
            activation_floor_gb=base.activation_floor_gb,
            bandwidth_efficiency=base.bandwidth_efficiency,
            safety_margin=base.safety_margin,
        )
        out.append(estimate(model, gpu, opts, include_alternatives=False))
    return out


def recommend(
    model: ModelSpec, gpu: GPU, options: Optional[PlanOptions] = None
) -> Recommendation:
    """Choose the highest-quality quantisation that fits this device.

    Preference order: comfortable fit, then tight fit, then the best partial
    offload. If nothing fits at all, return the q4 plan so the caller can
    explain why the combination fails.
    """
    base = options or PlanOptions()
    plans = sweep_quants(model, gpu, base)

    chosen = next((p for p in plans if p.verdict == "comfortable"), None)
    if chosen is None:
        chosen = next((p for p in plans if p.verdict == "tight"), None)
    if chosen is None:
        offloads = [p for p in plans if p.verdict == "offload"]
        chosen = max(offloads, key=lambda p: p.gpu_layers_pct) if offloads else None
    if chosen is None:
        chosen = next(p for p in plans if p.options.quant == base.quant)

    # Recompute with alternatives enabled for the chosen configuration.
    final = estimate(model, gpu, chosen.options, include_alternatives=True)
    return Recommendation(plan=final, sweep=plans, model=model, gpu=gpu)


def recommend_gpu(
    model: ModelSpec, options: Optional[PlanOptions] = None, limit: int = 3
) -> List[FitPlan]:
    """Cheapest single devices that can host this model at the requested quant."""
    opts = options or PlanOptions()
    probe = estimate(model, GPUS["rtx5090"], opts, include_alternatives=False)
    return [
        estimate(model, g, opts, include_alternatives=False)
        for g in cheapest_options(probe.total_gb, limit=limit)
    ]


def compare_gpus(
    model: ModelSpec, keys: List[str], options: Optional[PlanOptions] = None
) -> List[FitPlan]:
    """Evaluate the same model across several named devices."""
    opts = options or PlanOptions()
    return [estimate(model, find_gpu(k), opts, include_alternatives=False) for k in keys]
