"""Tests for the planner: quant sweeps, recommendations and GPU suggestions."""

import unittest

from vramfit.hardware import GPUS
from vramfit.model import MODEL_PRESETS
from vramfit.planner import (
    QUALITY_FLOOR,
    QUALITY_ORDER,
    compare_gpus,
    plan_fit,
    recommend,
    recommend_gpu,
    sweep_quants,
)
from vramfit.estimator import PlanOptions


class TestSweep(unittest.TestCase):
    def test_sweep_covers_every_quant(self):
        plans = sweep_quants(MODEL_PRESETS["llama3-8b"], GPUS["rtx4090"])
        self.assertEqual([p.options.quant for p in plans], QUALITY_ORDER)

    def test_sweep_sizes_increase_with_quality(self):
        plans = sweep_quants(MODEL_PRESETS["qwen2.5-7b"], GPUS["rtx5090"])
        sizes = [p.total_gb for p in plans]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_sweep_respects_context_option(self):
        plans = sweep_quants(
            MODEL_PRESETS["llama3-8b"], GPUS["rtx4090"], PlanOptions(context=4096)
        )
        self.assertTrue(all(p.options.context == 4096 for p in plans))


class TestRecommend(unittest.TestCase):
    def test_picks_highest_quality_that_fits(self):
        rec = recommend(MODEL_PRESETS["llama3-8b"], GPUS["rtx4090"])
        self.assertEqual(rec.plan.verdict, "comfortable")
        # 4090 has 24GB; an 8B model at f16 is 16GB, which should fit.
        self.assertEqual(rec.quant, "f16")

    def test_small_card_forces_lower_quality(self):
        big = recommend(MODEL_PRESETS["qwen2.5-32b"], GPUS["rtx5090"])
        small = recommend(MODEL_PRESETS["qwen2.5-32b"], GPUS["rtx4070"])
        big_idx = QUALITY_ORDER.index(big.quant)
        small_idx = QUALITY_ORDER.index(small.quant)
        self.assertGreaterEqual(small_idx, big_idx)

    def test_quality_warning_below_floor(self):
        rec = recommend(MODEL_PRESETS["llama3-70b"], GPUS["rtx3060"])
        if QUALITY_ORDER.index(rec.quant) > QUALITY_ORDER.index(QUALITY_FLOOR):
            self.assertIsNotNone(rec.quality_warning)
        else:
            self.assertIsNone(rec.quality_warning)

    def test_no_warning_at_or_above_floor(self):
        rec = recommend(MODEL_PRESETS["phi3-mini"], GPUS["rtx4090"])
        self.assertIsNone(rec.quality_warning)

    def test_recommendation_always_returns_a_plan(self):
        # Even hopeless combinations must produce an explainable answer.
        rec = recommend(MODEL_PRESETS["deepseek-v3"], GPUS["rtx3060"])
        self.assertIsNotNone(rec.plan)
        self.assertEqual(rec.plan.verdict, "no")

    def test_to_dict_includes_sweep(self):
        rec = recommend(MODEL_PRESETS["mistral-7b"], GPUS["rtx4060ti16"])
        d = rec.to_dict()
        self.assertEqual(len(d["quant_table"]), len(QUALITY_ORDER))
        self.assertIn("quality_warning", d)


class TestGpuSuggestion(unittest.TestCase):
    def test_suggests_devices_that_all_fit(self):
        plans = recommend_gpu(MODEL_PRESETS["llama3-8b"], PlanOptions(quant="q4"))
        self.assertTrue(plans)
        for p in plans:
            self.assertTrue(p.fits, f"{p.gpu.name} should fit an 8B q4 model")

    def test_smallest_suggestion_is_first(self):
        plans = recommend_gpu(MODEL_PRESETS["mistral-7b"], PlanOptions(quant="q4"))
        sizes = [p.gpu.vram_gb for p in plans]
        self.assertEqual(sizes, sorted(sizes))

    def test_bigger_model_needs_bigger_devices(self):
        small = recommend_gpu(MODEL_PRESETS["llama3-8b"], PlanOptions(quant="q4"))[0]
        big = recommend_gpu(MODEL_PRESETS["llama3-70b"], PlanOptions(quant="q4"))[0]
        self.assertGreater(big.gpu.vram_gb, small.gpu.vram_gb)


class TestCompare(unittest.TestCase):
    def test_compare_returns_one_plan_per_device(self):
        plans = compare_gpus(
            MODEL_PRESETS["llama3-8b"], ["rtx3060", "rtx4090", "a100-80"]
        )
        self.assertEqual(len(plans), 3)
        self.assertEqual([p.gpu.vram_gb for p in plans], [12, 24, 80])

    def test_compare_rejects_unknown_device(self):
        with self.assertRaises(ValueError):
            compare_gpus(MODEL_PRESETS["llama3-8b"], ["rtx3060", "voodoo2"])


class TestPlanFit(unittest.TestCase):
    def test_plan_fit_matches_estimate(self):
        plan = plan_fit(MODEL_PRESETS["gemma2-9b"], GPUS["rtx4070"], PlanOptions(quant="q5"))
        self.assertEqual(plan.options.quant, "q5")
        self.assertGreater(plan.total_gb, 0)


if __name__ == "__main__":
    unittest.main()
