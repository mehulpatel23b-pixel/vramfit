"""End-to-end CLI tests: exit codes, JSON output and error handling."""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout

from vramfit.cli import EXIT_FITS, EXIT_NO_FIT, EXIT_USAGE, main


def run(*argv):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = main(list(argv))
        except SystemExit as exc:  # argparse errors
            code = int(exc.code or 0)
    return code, out.getvalue(), err.getvalue()


class TestPlanCommand(unittest.TestCase):
    def test_fitting_plan_exits_zero(self):
        code, out, _ = run("plan", "--model", "llama3-8b", "--gpu", "rtx4090")
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("FITS", out)

    def test_impossible_plan_exits_one(self):
        code, out, _ = run("plan", "--model", "llama3-70b", "--gpu", "rtx3070")
        self.assertEqual(code, EXIT_NO_FIT)

    def test_json_output_parses(self):
        code, out, _ = run("plan", "--model", "mistral-7b", "--gpu", "rtx4070", "--json")
        self.assertEqual(code, EXIT_FITS)
        payload = json.loads(out)
        self.assertEqual(payload["model"], "Mistral 7B")
        self.assertTrue(payload["fits"])

    def test_json_reports_allocation_breakdown(self):
        _, out, _ = run("plan", "--model", "qwen2.5-7b", "--gpu", "rtx4060ti16", "--json")
        vram = json.loads(out)["vram"]
        self.assertAlmostEqual(
            vram["total_gb"],
            vram["weights_gb"] + vram["kv_cache_gb"] + vram["activation_gb"],
            delta=0.05,
        )

    def test_unknown_gpu_is_usage_error(self):
        code, _, err = run("plan", "--model", "llama3-8b", "--gpu", "voodoo2")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("unknown GPU", err)

    def test_unknown_quant_is_usage_error(self):
        code, _, err = run("plan", "--model", "llama3-8b", "--gpu", "rtx4090", "--quant", "q7")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("quantisation", err)

    def test_bad_model_string_is_usage_error(self):
        code, _, err = run("plan", "--model", "banana", "--gpu", "rtx4090")
        self.assertEqual(code, EXIT_USAGE)


class TestOtherCommands(unittest.TestCase):
    def test_best_prints_sweep(self):
        code, out, _ = run("best", "--model", "llama3-8b", "--gpu", "rtx4090")
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("quantisation sweep", out)
        self.assertIn("q4", out)

    def test_sweep_command(self):
        code, out, _ = run("sweep", "--model", "qwen2.5-7b", "--gpu", "rtx4070")
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("f16", out)

    def test_gpu_command_lists_candidates(self):
        code, out, _ = run("gpu", "--model", "llama3-8b", "--quant", "q4")
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("RTX", out)

    def test_compare_command(self):
        code, out, _ = run(
            "compare", "--model", "llama3-8b", "--gpus", "rtx3060,rtx4090"
        )
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("RTX 3060", out)
        self.assertIn("RTX 4090", out)

    def test_compare_json(self):
        _, out, _ = run(
            "compare", "--model", "llama3-8b", "--gpus", "rtx3060,rtx4090", "--json"
        )
        payload = json.loads(out)
        self.assertEqual(len(payload), 2)
        self.assertIn("verdict", payload[0])

    def test_list_command(self):
        code, out, _ = run("list")
        self.assertEqual(code, EXIT_FITS)
        self.assertIn("rtx4090", out)
        self.assertIn("llama3-8b", out)


class TestArgumentValidation(unittest.TestCase):
    def test_missing_model(self):
        code, _, err = run("plan", "--gpu", "rtx4090")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--model", err)

    def test_missing_gpu(self):
        code, _, err = run("plan", "--model", "llama3-8b")
        self.assertEqual(code, EXIT_USAGE)
        self.assertIn("--gpu", err)

    def test_no_command(self):
        code, _, _ = run()
        self.assertEqual(code, EXIT_USAGE)


if __name__ == "__main__":
    unittest.main()
