"""Unit tests for model parsing and quantisation handling."""

import unittest

from vramfit.model import (
    MODEL_PRESETS,
    QUANTS,
    ModelSpec,
    normalize_quant,
    parse_params,
    resolve_model,
)


class TestParseParams(unittest.TestCase):
    def test_billions_suffix(self):
        self.assertEqual(parse_params("7b"), 7e9)
        self.assertEqual(parse_params("70B"), 70e9)

    def test_decimal_billions(self):
        self.assertAlmostEqual(parse_params("1.5b"), 1.5e9)

    def test_millions_and_thousands(self):
        self.assertEqual(parse_params("405m"), 405e6)
        self.assertEqual(parse_params("3k"), 3000.0)

    def test_bare_number_means_billions(self):
        # "qwen 14" is how people talk about a 14B model.
        self.assertEqual(parse_params("14"), 14e9)

    def test_numeric_input(self):
        self.assertEqual(parse_params(8), 8e9)

    def test_whitespace_tolerated(self):
        self.assertEqual(parse_params("  8 b "), 8e9)

    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            parse_params("seven")
        with self.assertRaises(ValueError):
            parse_params("")


class TestQuant(unittest.TestCase):
    def test_canonical_passthrough(self):
        for key in QUANTS:
            self.assertEqual(normalize_quant(key), key)

    def test_aliases(self):
        self.assertEqual(normalize_quant("FP16"), "f16")
        self.assertEqual(normalize_quant("q4_k_m"), "q4")
        self.assertEqual(normalize_quant("int8"), "q8")
        self.assertEqual(normalize_quant("Q5-K-M"), "q5")

    def test_unknown_quant_raises(self):
        with self.assertRaises(ValueError):
            normalize_quant("q7_super")

    def test_lower_bits_means_smaller(self):
        spec = ModelSpec("t", 7e9)
        sizes = [spec.weights_bytes(q) for q in ("f16", "q8", "q4", "q2")]
        self.assertEqual(sizes, sorted(sizes, reverse=True))


class TestModelSpec(unittest.TestCase):
    def test_weights_bytes_f16(self):
        spec = ModelSpec("t", 7e9)
        # 7B params at 16 bits = 14e9 bytes
        self.assertAlmostEqual(spec.weights_bytes("f16"), 14e9)

    def test_weights_bytes_q4(self):
        spec = ModelSpec("t", 7e9)
        self.assertAlmostEqual(spec.weights_bytes("q4"), 7e9 * 4.5 / 8)

    def test_known_kv_cache_llama3_8b(self):
        """Llama 3 8B: 32 layers, 8 KV heads, head_dim 128, f16.

        per_token = 32 * 8 * 128 * 2 * 2 bytes = 131072 bytes = 128 KiB
        At 8192 tokens that is exactly 1 GiB.
        """
        spec = MODEL_PRESETS["llama3-8b"]
        kv = spec.kv_bytes("q4", 8192)
        self.assertAlmostEqual(kv, 1024.0**3, delta=1)

    def test_kv_scales_linearly_with_context(self):
        spec = MODEL_PRESETS["llama3-8b"]
        self.assertAlmostEqual(
            spec.kv_bytes("q4", 8192) / spec.kv_bytes("q4", 4096), 2.0, places=6
        )

    def test_q8_kv_halves_cache(self):
        spec = MODEL_PRESETS["llama3-8b"]
        self.assertAlmostEqual(
            spec.kv_bytes("q4", 8192, kv_quant="q8") * 2,
            spec.kv_bytes("q4", 8192, kv_quant="f16"),
            delta=1,
        )

    def test_shape_inferred_when_unknown(self):
        spec = ModelSpec("mystery", 13e9)
        self.assertFalse(spec.has_attention_shape)
        layers, kv_heads, head_dim = spec.attention_shape()
        self.assertGreater(layers, 0)
        self.assertGreater(kv_heads, 0)
        self.assertGreaterEqual(head_dim, 32)

    def test_explicit_shape_wins(self):
        spec = ModelSpec("exact", 7e9, 32, 8, 128, 4096)
        self.assertEqual(spec.attention_shape(), (32, 8, 128))


class TestResolveModel(unittest.TestCase):
    def test_preset_lookup(self):
        self.assertIs(resolve_model("llama3-8b"), MODEL_PRESETS["llama3-8b"])

    def test_preset_lookup_case_insensitive(self):
        self.assertEqual(resolve_model("LLAMA3-8B").name, "Llama 3 8B")

    def test_raw_count_becomes_custom(self):
        spec = resolve_model("13b")
        self.assertAlmostEqual(spec.params, 13e9)
        self.assertIn("custom", spec.name)


if __name__ == "__main__":
    unittest.main()


class TestMoE(unittest.TestCase):
    def test_dense_model_reads_all_params(self):
        spec = MODEL_PRESETS["llama3-8b"]
        self.assertFalse(spec.is_moe)
        self.assertEqual(spec.read_params, spec.params)

    def test_moe_reads_active_params_only(self):
        spec = MODEL_PRESETS["mixtral-8x7b"]
        self.assertTrue(spec.is_moe)
        self.assertLess(spec.read_params, spec.params)
        self.assertAlmostEqual(spec.read_params, 12.9e9)

    def test_moe_still_pays_full_memory(self):
        """Every expert must be resident even though few are read per token."""
        spec = MODEL_PRESETS["mixtral-8x7b"]
        self.assertAlmostEqual(spec.weights_bytes("f16"), 46.7e9 * 2)
