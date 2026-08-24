from __future__ import annotations

import argparse
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage8c", ROOT / "scripts/run_stage8c_gameplay_v1.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class Stage8CPerSideFpuCliTests(unittest.TestCase):
    def test_runner_argv_uses_only_explicit_per_side_fpu_flags(self) -> None:
        args = argparse.Namespace(candidate=Path("candidate.onnx"), champion=Path("champion.onnx"), budget=128, c_puct=2.5, concurrency=64, max_batch=96, wait_us=200, bridge_controller="active", fpu_mode="zero", fpu_reduction=0.0, candidate_fpu_mode="parent_value_reduced", candidate_fpu_reduction=0.25, champion_fpu_mode="parent_value_reduced", champion_fpu_reduction=0.25, candidate_budget=128, champion_budget=128, candidate_c_puct=2.5, champion_c_puct=2.5)
        argv = MODULE.build_runner_command(args, Path("out"), Path("openings.txt"))
        self.assertIn("--candidate-fpu-mode", argv)
        self.assertIn("--champion-fpu-mode", argv)
        self.assertNotIn("--fpu-mode", argv)
        self.assertNotIn("--fpu-reduction", argv)
        self.assertEqual(argv[argv.index("--candidate-budget") + 1], "128")
        self.assertEqual(argv[argv.index("--champion-c-puct") + 1], "2.5")


if __name__ == "__main__":
    unittest.main()
