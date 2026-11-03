"""Terminal rendering: human-readable tables without any dependencies."""

from __future__ import annotations

from typing import List

from .estimator import FitPlan
from .planner import QUALITY_ORDER, Recommendation

VERDICT_LABEL = {
    "comfortable": "FITS",
    "tight": "TIGHT",
    "offload": "PARTIAL",
    "no": "NO",
}


def bar(value: float, width: int = 24) -> str:
    """A simple ASCII utilisation bar."""
    filled = int(max(0.0, min(1.0, value)) * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def render_plan(plan: FitPlan, colour: bool = True) -> str:
    lines: List[str] = []
    q = plan.options.quant
    lines.append(f"{plan.model.name}  ->  {plan.gpu.name}")
    lines.append(f"quant {q}   context {plan.options.context}   batch {plan.options.batch}")
    lines.append("")
    lines.append(f"  weights      {plan.weights_gb:8.2f} GB")
    lines.append(f"  kv cache     {plan.kv_cache_gb:8.2f} GB")
    lines.append(f"  scratch      {plan.activation_gb:8.2f} GB")
    lines.append("  " + "-" * 24)
    lines.append(f"  total        {plan.total_gb:8.2f} GB")
    lines.append(f"  usable       {plan.usable_gb:8.2f} GB")
    lines.append(f"  headroom     {plan.headroom_gb:8.2f} GB")
    lines.append("")
    lines.append(
        f"  {bar(plan.utilisation)} {plan.utilisation * 100:5.0f}%  "
        f"{VERDICT_LABEL.get(plan.verdict, plan.verdict)}"
    )
    lines.append(f"  est. decode  {plan.tokens_per_second:6.1f} tok/s")
    if plan.gpu_layers_pct < 100:
        lines.append(f"  gpu layers   {plan.gpu_layers_pct:6.1f}%")
    if plan.max_context:
        lines.append(f"  max context  {plan.max_context:6d} tokens")
    lines.append("")
    for note in plan.notes:
        lines.append(f"  - {note}")
    if plan.alternatives:
        lines.append("")
        lines.append("  other devices that fit:")
        for alt in plan.alternatives:
            lines.append(f"    * {alt}")
    return "\n".join(lines)


def render_sweep(sweep: List[FitPlan]) -> str:
    header = f"{'quant':<6}{'total GB':>10}{'tok/s':>9}   verdict"
    rows = [header, "-" * len(header)]
    for p in sweep:
        rows.append(
            f"{p.options.quant:<6}{p.total_gb:>10.2f}"
            f"{p.tokens_per_second:>9.1f}   {VERDICT_LABEL.get(p.verdict, p.verdict)}"
        )
    return "\n".join(rows)


def render_recommendation(rec: Recommendation) -> str:
    parts = [render_plan(rec.plan)]
    parts.append("")
    parts.append("quantisation sweep (best quality first):")
    parts.append(render_sweep(rec.sweep))
    warning = rec.quality_warning
    if warning:
        parts.append("")
        parts.append(f"WARNING: {warning}")
    return "\n".join(parts)


def render_comparison(plans: List[FitPlan]) -> str:
    header = f"{'device':<28}{'usable':>9}{'total':>9}{'tok/s':>9}   verdict"
    rows = [header, "-" * len(header)]
    for p in plans:
        rows.append(
            f"{p.gpu.name[:27]:<28}{p.usable_gb:>9.1f}{p.total_gb:>9.2f}"
            f"{p.tokens_per_second:>9.1f}   {VERDICT_LABEL.get(p.verdict, p.verdict)}"
        )
    return "\n".join(rows)
