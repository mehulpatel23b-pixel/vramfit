"""Hardware catalogue: VRAM capacity, memory bandwidth and practical headroom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class GPU:
    """A device we can plan inference against."""

    name: str
    vram_gb: float
    bandwidth_gbs: float
    # Fraction of VRAM the driver + display + CUDA context already consume.
    # Measured on real machines: a 24GB card typically exposes ~22.5GB.
    overhead_gb: float = 1.5
    unified: bool = False

    @property
    def usable_gb(self) -> float:
        return max(0.0, self.vram_gb - self.overhead_gb)


GPUS: Dict[str, GPU] = {
    # Consumer NVIDIA
    "rtx3060": GPU("RTX 3060 12GB", 12, 360),
    "rtx3070": GPU("RTX 3070 8GB", 8, 448),
    "rtx3080": GPU("RTX 3080 10GB", 10, 760),
    "rtx3090": GPU("RTX 3090 24GB", 24, 936),
    "rtx4060ti16": GPU("RTX 4060 Ti 16GB", 16, 288),
    "rtx4070": GPU("RTX 4070 12GB", 12, 504),
    "rtx4070tis": GPU("RTX 4070 Ti Super 16GB", 16, 672),
    "rtx4080": GPU("RTX 4080 16GB", 16, 717),
    "rtx4090": GPU("RTX 4090 24GB", 24, 1008),
    "rtx5070": GPU("RTX 5070 12GB", 12, 672),
    "rtx5080": GPU("RTX 5080 16GB", 16, 960),
    "rtx5090": GPU("RTX 5090 32GB", 32, 1792),
    # Prosumer / workstation
    "a6000": GPU("RTX A6000 48GB", 48, 768),
    "l40s": GPU("L40S 48GB", 48, 864),
    "a100-40": GPU("A100 40GB", 40, 1555),
    "a100-80": GPU("A100 80GB", 80, 2039),
    "h100-80": GPU("H100 80GB", 80, 3350),
    # Apple silicon (unified memory: the OS reserves more)
    "m2ultra192": GPU("M2 Ultra 192GB", 192, 800, overhead_gb=16, unified=True),
    "m3max48": GPU("M3 Max 48GB", 48, 400, overhead_gb=6, unified=True),
    "m4max128": GPU("M4 Max 128GB", 128, 546, overhead_gb=10, unified=True),
    # CPU-only baseline for "can I run it at all" comparisons
    "cpu": GPU("CPU only (DDR5 dual channel)", 64, 90, overhead_gb=8, unified=True),
}

ALIASES = {
    "3060": "rtx3060",
    "3070": "rtx3070",
    "3080": "rtx3080",
    "3090": "rtx3090",
    "4060ti": "rtx4060ti16",
    "4070": "rtx4070",
    "4070ti": "rtx4070tis",
    "4080": "rtx4080",
    "4090": "rtx4090",
    "5070": "rtx5070",
    "5080": "rtx5080",
    "5090": "rtx5090",
    "a100": "a100-80",
    "h100": "h100-80",
    "m2ultra": "m2ultra192",
    "m3max": "m3max48",
    "m4max": "m4max128",
    "ddr5": "cpu",
}


def find_gpu(key: str) -> GPU:
    """Look up a GPU by catalogue key, alias, or case-insensitive name match."""
    k = key.strip().lower().replace(" ", "").replace("_", "")
    if k in GPUS:
        return GPUS[k]
    if k in ALIASES:
        return GPUS[ALIASES[k]]
    for gpu in GPUS.values():
        if k in gpu.name.lower().replace(" ", ""):
            return gpu
    raise ValueError(
        f"unknown GPU {key!r}. known: {', '.join(sorted(GPUS))}"
    )


# Devices that hold memory but cannot serve tokens at a useful rate. They are
# valid comparison points but must never be offered as a recommendation.
NOT_BUYABLE = {"cpu"}


def suggest_gpus(needed_gb: float, include_cpu: bool = False) -> List[GPU]:
    """Smallest-first list of single devices that can hold a footprint."""
    fits = [
        g
        for key, g in GPUS.items()
        if g.usable_gb >= needed_gb and (include_cpu or key not in NOT_BUYABLE)
    ]
    return sorted(fits, key=lambda g: g.vram_gb)


def cheapest_options(
    needed_gb: float, limit: int = 3, include_cpu: bool = False
) -> List[GPU]:
    """The fewest-GB devices that can hold the footprint, cheapest first."""
    return suggest_gpus(needed_gb, include_cpu=include_cpu)[:limit]
