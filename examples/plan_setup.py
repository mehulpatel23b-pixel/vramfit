#!/usr/bin/env python3
"""Plan a local inference setup from Python instead of the CLI.

Run:  PYTHONPATH=src python3 examples/plan_setup.py
"""

import json

from vramfit.hardware import find_gpu
from vramfit.model import ModelSpec, resolve_model
from vramfit.estimator import PlanOptions, estimate
from vramfit.planner import recommend, recommend_gpu


def show(title: str, rec) -> None:
    print(f"\n=== {title} ===")
    print(f"model      : {rec.plan.model.name}")
    print(f"device     : {rec.plan.gpu.name} ({rec.plan.usable_gb:.1f} GB usable)")
    print(f"best quant : {rec.quant}")
    print(f"total VRAM : {rec.plan.total_gb:.2f} GB")
    print(f"verdict    : {rec.plan.verdict}")
    print(f"decode     : {rec.plan.tokens_per_second:.1f} tok/s")
    if rec.quality_warning:
        print(f"warning    : {rec.quality_warning}")


def main() -> None:
    # 1. A named preset on a named card.
    rec = recommend(resolve_model("llama3-8b"), find_gpu("rtx4060ti16"))
    show("Llama 3 8B on a 4060 Ti 16GB", rec)

    # 2. A raw parameter count for a model that has no preset yet.
    custom = ModelSpec("internal-finetune", 13e9)
    rec = recommend(custom, find_gpu("rtx4070"), PlanOptions(context=16384))
    show("13B fine-tune on a 4070 12GB", rec)

    # 3. "I need to serve this model, what do I buy?"
    print("\n=== cheapest single cards for Llama 3 70B at q4 ===")
    for plan in recommend_gpu(resolve_model("llama3-70b"), PlanOptions(quant="q4")):
        print(
            f"  {plan.gpu.name:<28} {plan.gpu.vram_gb:>4.0f} GB  "
            f"{plan.tokens_per_second:>5.1f} tok/s  {plan.verdict}"
        )

    # 4. Machine-readable output for a scheduler or CI gate.
    plan = estimate(resolve_model("qwen2.5-7b"), find_gpu("rtx3090"), PlanOptions(quant="q5"))
    print("\n=== JSON payload ===")
    print(json.dumps(plan.to_dict()["performance"], indent=2))

    # 5. Acting on the verdict: pick the largest context that still fits.
    print("\n=== context budget ===")
    print(f"  max context on this card at q5: {plan.max_context} tokens")


if __name__ == "__main__":
    main()
