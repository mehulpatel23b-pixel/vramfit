"""Tests for the memory estimator and speed model."""

import unittest

from vramfit.estimator import GB, PlanOptions, estimate
from vramfit.hardware import GPUS, find_gpu
from vramfit.model import MODEL_PRESETS, ModelSpec


class TestVerdicts(unittest.TestCase):
    def test_small_model_on_big_card_is_comfortable(self):
        plan = estimate(MODEL_PRESETS["phi3-mini"], GPUS["rtx5090"])
        self.assertEqual(plan.verdict, "comfortable")
        self.assertTrue(plan.fits)
        self.assertGreater(plan.headroom_gb, 0)

    def test_70b_on_8gb_does_not_fit(self):
        plan = estimate(MODEL_PRESETS["llama3-70b"], GPUS["rtx3070"], PlanOptions(quant="q4"))
        self.assertIn(plan.verdict, ("offload", "no"))
        self.assertFalse(plan.fits)
        self.assertLess(plan.gpu_layers_pct, 100)

    def test_verdict_boundaries_are_ordered(self):
        model = MODEL_PRESETS["llama3-8b"]
        comfortable = estimate(model, GPUS["rtx5090"], PlanOptions(quant="q4"))
        tight = estimate(model, GPUS["rtx3070"], PlanOptions(quant="q8", context=16384))
        self.assertLess(comfortable.utilisation, tight.utilisation)

    def test_total_is_sum_of_parts(self):
        plan = estimate(MODEL_PRESETS["qwen2.5-7b"], GPUS["rtx4070"])
        parts = plan.weights_gb + plan.kv_cache_gb + plan.activation_gb
        self.assertAlmostEqual(plan.total_gb, parts, places=9)

    def test_headroom_consistent_with_total(self):
        plan = estimate(MODEL_PRESETS["mistral-7b"], GPUS["rtx3090"])
        self.assertAlmostEqual(
            plan.headroom_gb, plan.usable_gb - plan.total_gb, places=9
        )


class TestMonotonicity(unittest.TestCase):
    def test_lower_quant_uses_less_memory(self):
        model = MODEL_PRESETS["qwen2.5-32b"]
        gpu = GPUS["rtx4090"]
        sizes = [
            estimate(model, gpu, PlanOptions(quant=q)).total_gb
            for q in ("f16", "q8", "q4", "q2")
        ]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_longer_context_uses_more_memory(self):
        model = MODEL_PRESETS["llama3-8b"]
        gpu = GPUS["rtx4090"]
        a = estimate(model, gpu, PlanOptions(context=4096)).total_gb
        b = estimate(model, gpu, PlanOptions(context=8192)).total_gb
        self.assertGreater(b, a)

    def test_bigger_batch_uses_more_memory(self):
        model = MODEL_PRESETS["llama3-8b"]
        gpu = GPUS["rtx5090"]
        a = estimate(model, gpu, PlanOptions(batch=1)).total_gb
        b = estimate(model, gpu, PlanOptions(batch=8)).total_gb
        self.assertGreater(b, a)

    def test_bigger_card_has_more_headroom(self):
        model = MODEL_PRESETS["qwen2.5-32b"]
        small = estimate(model, GPUS["rtx4070"], PlanOptions(quant="q4"))
        big = estimate(model, GPUS["rtx5090"], PlanOptions(quant="q4"))
        self.assertGreater(big.headroom_gb, small.headroom_gb)


class TestSpeedModel(unittest.TestCase):
    def test_speed_is_positive_when_fits(self):
        plan = estimate(MODEL_PRESETS["llama3-8b"], GPUS["rtx4090"], PlanOptions(quant="q4"))
        self.assertGreater(plan.tokens_per_second, 0)

    def test_70b_is_slower_than_8b_on_same_card(self):
        gpu = GPUS["rtx5090"]
        small = estimate(MODEL_PRESETS["llama3-8b"], gpu, PlanOptions(quant="q4"))
        big = estimate(MODEL_PRESETS["llama3-70b"], gpu, PlanOptions(quant="q4"))
        self.assertGreater(small.tokens_per_second, big.tokens_per_second)

    def test_matches_published_llamacpp_benchmark(self):
        """Reality check: llama.cpp measures ~158 tok/s for Llama 3 8B Q4_0
        on an RTX 4090. The estimator must land in that neighbourhood."""
        plan = estimate(MODEL_PRESETS["llama3-8b"], GPUS["rtx4090"], PlanOptions(quant="q4"))
        self.assertGreater(plan.tokens_per_second, 100)
        self.assertLess(plan.tokens_per_second, 200)

    def test_partial_offload_is_much_slower(self):
        model = MODEL_PRESETS["qwen2.5-32b"]
        opts = PlanOptions(quant="q4")
        on_card = estimate(model, GPUS["rtx5090"], opts)
        off_card = estimate(model, GPUS["rtx4070"], opts)
        self.assertEqual(on_card.gpu_layers_pct, 100)
        self.assertLess(off_card.gpu_layers_pct, 100)
        self.assertLess(off_card.tokens_per_second, on_card.tokens_per_second)

    def test_no_fit_yields_zero_speed(self):
        plan = estimate(
            MODEL_PRESETS["deepseek-v3"], GPUS["rtx3060"], PlanOptions(quant="f16")
        )
        self.assertEqual(plan.verdict, "no")
        self.assertEqual(plan.tokens_per_second, 0)


class TestMaxContext(unittest.TestCase):
    def test_capped_at_native_training_context(self):
        plan = estimate(MODEL_PRESETS["llama3-8b"], GPUS["rtx5090"])
        self.assertLessEqual(plan.max_context, MODEL_PRESETS["llama3-8b"].native_context)

    def test_zero_when_weights_do_not_fit(self):
        plan = estimate(MODEL_PRESETS["deepseek-v3"], GPUS["rtx3060"], PlanOptions(quant="f16"))
        self.assertEqual(plan.max_context, 0)

    def test_larger_card_allows_at_least_as_much_context(self):
        model = MODEL_PRESETS["qwen2.5-7b"]
        small = estimate(model, GPUS["rtx4070"], PlanOptions(quant="q4"))
        big = estimate(model, GPUS["rtx5090"], PlanOptions(quant="q4"))
        self.assertGreaterEqual(big.max_context, small.max_context)


class TestNotesAndSerialisation(unittest.TestCase):
    def test_notes_present_and_strings(self):
        plan = estimate(MODEL_PRESETS["llama3-8b"], GPUS["rtx4060ti16"])
        self.assertTrue(plan.notes)
        self.assertTrue(all(isinstance(n, str) for n in plan.notes))

    def test_no_fit_explains_why(self):
        plan = estimate(MODEL_PRESETS["llama3-70b"], GPUS["rtx3070"])
        joined = " ".join(plan.notes).lower()
        self.assertTrue("fit" in joined)

    def test_to_dict_is_json_serialisable(self):
        import json

        plan = estimate(MODEL_PRESETS["qwen2.5-7b"], GPUS["rtx4090"])
        text = json.dumps(plan.to_dict())
        self.assertIn("comfortable", text)

    def test_to_dict_shape(self):
        plan = estimate(MODEL_PRESETS["qwen2.5-7b"], GPUS["rtx4090"])
        d = plan.to_dict()
        for key in ("model", "gpu", "quant", "verdict", "fits", "vram", "performance"):
            self.assertIn(key, d)
        self.assertIn("total_gb", d["vram"])
        self.assertIn("tokens_per_second", d["performance"])


class TestHardware(unittest.TestCase):
    def test_find_by_key_and_alias(self):
        self.assertEqual(find_gpu("rtx4090").vram_gb, 24)
        self.assertEqual(find_gpu("4090").name, "RTX 4090 24GB")

    def test_unknown_gpu_raises(self):
        with self.assertRaises(ValueError):
            find_gpu("gtx480")

    def test_usable_never_exceeds_vram(self):
        for gpu in GPUS.values():
            self.assertLess(gpu.usable_gb, gpu.vram_gb)
            self.assertGreater(gpu.usable_gb, 0)

    def test_always_an_option_that_fits(self):
        """Regression guard: the catalogue must contain a card for any sane model."""
        plan = estimate(MODEL_PRESETS["llama3-70b"], GPUS["a100-80"], PlanOptions(quant="q4"))
        self.assertTrue(plan.fits)


if __name__ == "__main__":
    unittest.main()


class TestMoE(unittest.TestCase):
    def test_moe_faster_than_dense_of_same_total_size(self):
        dense = ModelSpec("dense-47b", 46.7e9, 32, 8, 128, 4096, 32000)
        moe = MODEL_PRESETS["mixtral-8x7b"]
        gpu = GPUS["a100-80"]
        d = estimate(dense, gpu, PlanOptions(quant="q4"))
        m = estimate(moe, gpu, PlanOptions(quant="q4"))
        self.assertGreater(m.tokens_per_second, d.tokens_per_second)

    def test_moe_uses_same_memory_as_dense_of_same_params(self):
        dense = ModelSpec("dense-47b", 46.7e9, 32, 8, 128, 4096, 32000)
        moe = MODEL_PRESETS["mixtral-8x7b"]
        gpu = GPUS["a100-80"]
        d = estimate(dense, gpu, PlanOptions(quant="q4"))
        m = estimate(moe, gpu, PlanOptions(quant="q4"))
        self.assertAlmostEqual(d.total_gb, m.total_gb, places=6)

    def test_moe_note_present(self):
        plan = estimate(MODEL_PRESETS["mixtral-8x7b"], GPUS["a100-80"])
        joined = " ".join(plan.notes).lower()
        self.assertIn("mixture-of-experts", joined)
