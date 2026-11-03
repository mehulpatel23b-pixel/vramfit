"""vramfit - will this model actually fit on this GPU?

Accurate, explainable memory math for local LLM inference.
Zero runtime dependencies, pure Python standard library.
"""

from .model import ModelSpec, parse_params, resolve_model
from .planner import PlanOptions, FitPlan, plan_fit
from .hardware import GPU, find_gpu, GPUS
from .estimator import estimate

__version__ = "1.0.0"

__all__ = [
    "ModelSpec",
    "parse_params",
    "resolve_model",
    "PlanOptions",
    "FitPlan",
    "plan_fit",
    "GPU",
    "GPUS",
    "find_gpu",
    "estimate",
    "__version__",
]
