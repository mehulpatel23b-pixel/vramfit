"""vramfit CLI - python -m vramfit"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from .estimator import PlanOptions, estimate
from .hardware import GPUS, find_gpu
from .model import MODEL_PRESETS, ModelSpec, normalize_quant, parse_params, resolve_model
from .planner import compare_gpus, recommend, recommend_gpu, sweep_quants
from .report import render_comparison, render_plan, render_recommendation, render_sweep

EXIT_FITS = 0
EXIT_NO_FIT = 1
EXIT_USAGE = 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vramfit",
        description="Will this model actually fit on this GPU? Accurate memory math.",
        epilog="examples:\n"
        "  vramfit plan --model llama3-8b --gpu rtx4060ti16\n"
        "  vramfit plan --model 32b --gpu 4090 --context 32768\n"
        "  vramfit best --model qwen2.5-32b --gpu m4max128 --json\n"
        "  vramfit gpu --model llama3-70b --quant q4\n"
        "  vramfit compare --model mixtral-8x7b --gpus 3090,4090,a6000\n"
        "  vramfit list",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"vramfit {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", help="preset name or parameter count, e.g. llama3-8b, 32b")
    common.add_argument("--quant", default="q4", help="f16, q8, q6, q5, q4, q3, q2 (default q4)")
    common.add_argument("--context", type=int, default=8192, help="context window in tokens")
    common.add_argument("--batch", type=int, default=1, help="concurrent sequences")
    common.add_argument("--kv-quant", default="f16", help="kv cache dtype: f16 or q8")
    common.add_argument("--json", action="store_true", help="machine-readable output")

    plan = sub.add_parser("plan", parents=[common], help="plan one model on one device")
    plan.add_argument("--gpu", help="device key, e.g. rtx4090, 5090, m4max128")

    best = sub.add_parser("best", parents=[common], help="pick the best quant that fits")
    best.add_argument("--gpu", help="device key")

    gpu_cmd = sub.add_parser("gpu", parents=[common], help="cheapest devices that can host the model")

    cmp_ = sub.add_parser("compare", parents=[common], help="compare several devices")
    cmp_.add_argument("--gpus", required=True, help="comma-separated device keys")

    sweep = sub.add_parser("sweep", parents=[common], help="show every quantisation for one device")
    sweep.add_argument("--gpu", help="device key")

    sub.add_parser("list", help="list known GPUs and model presets")
    return p


def _options(args) -> PlanOptions:
    normalize_quant(args.quant)  # fail fast on typos
    return PlanOptions(
        quant=args.quant,
        context=args.context,
        batch=args.batch,
        kv_quant=args.kv_quant,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        print("GPUs:")
        for key, gpu in sorted(GPUS.items()):
            kind = "unified" if gpu.unified else "discrete"
            print(f"  {key:<14} {gpu.name:<32} {gpu.vram_gb:>5.0f}GB  {kind}")
        print("\nmodel presets:")
        for key, spec in MODEL_PRESETS.items():
            print(f"  {key:<14} {spec.name:<22} {spec.params / 1e9:>7.1f}B params")
        return EXIT_FITS

    if not args.model:
        parser.error("--model is required")

    try:
        model = resolve_model(args.model)
        options = _options(args)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    if args.command in ("plan", "best", "sweep") and not getattr(args, "gpu", None):
        parser.error("--gpu is required")

    try:
        if args.command == "plan":
            plan = estimate(model, find_gpu(args.gpu), options)
            print(json.dumps(plan.to_dict(), indent=2) if args.json else render_plan(plan))
            return EXIT_FITS if plan.fits else EXIT_NO_FIT

        if args.command == "best":
            rec = recommend(model, find_gpu(args.gpu), options)
            print(
                json.dumps(rec.to_dict(), indent=2)
                if args.json
                else render_recommendation(rec)
            )
            return EXIT_FITS if rec.plan.fits else EXIT_NO_FIT

        if args.command == "sweep":
            plans = sweep_quants(model, find_gpu(args.gpu), options)
            if args.json:
                print(json.dumps([p.to_dict() for p in plans], indent=2))
            else:
                print(render_sweep(plans))
            return EXIT_FITS

        if args.command == "gpu":
            plans = recommend_gpu(model, options)
            if args.json:
                print(json.dumps([p.to_dict() for p in plans], indent=2))
            else:
                print(render_comparison(plans))
            return EXIT_FITS if plans else EXIT_NO_FIT

        if args.command == "compare":
            keys = [k.strip() for k in args.gpus.split(",") if k.strip()]
            plans = compare_gpus(model, keys, options)
            if args.json:
                print(json.dumps([p.to_dict() for p in plans], indent=2))
            else:
                print(render_comparison(plans))
            return EXIT_FITS if any(p.fits for p in plans) else EXIT_NO_FIT

    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    parser.error("unknown command")
    return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
