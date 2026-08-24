from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / name)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PRODUCTION = load("autonomous_generation_production_v1.py")
MONITOR = load("monitor_autonomous_generation_production_v1.py")


class AutonomousProductionContractTests(unittest.TestCase):
    def test_approved_control_recipe_binds_profile_and_search(self) -> None:
        recipe = PRODUCTION.recipe(ROOT / "config/autonomous-generation-v1/production-control-3gen-v1.json")
        self.assertEqual(recipe["training"], {"rows_per_epoch": 400000, "epochs": 4, "base_batch": 64, "optimizer": {"name": "AdamW", "lr": 0.001, "weight_decay": 0.0001}, "selection": "best-validation-policy on frozen Teacher100 held-out split"})
        self.assertEqual(recipe["selfplay"]["inference"]["concurrency"], 128)
        self.assertEqual(recipe["selfplay"]["resource_profile"]["max_batch"], 96)
        self.assertEqual(recipe["selfplay"]["search"]["fpu_reduction"], 0.25)

    def test_fake_lineage_semantics_are_explicit_and_cpu_only(self) -> None:
        incumbent = {"id": "champion-2", "checkpoint_sha256": "a", "onnx_sha256": "b"}
        next_incumbent = PRODUCTION.fake_incumbent(incumbent, 2, "candidate")
        self.assertEqual(next_incumbent["id"], "runlocal-g2-candidate")
        self.assertTrue(next_incumbent["run_local"])
        self.assertEqual(PRODUCTION.candidate_ids(1, {"candidate_plan": {"seeds": [1, 2, 3]}}), ["C2-AUTO-G0001-S1", "C2-AUTO-G0001-S2", "C2-AUTO-G0001-S3"])

    def test_monitor_uses_human_duration_format(self) -> None:
        self.assertEqual(MONITOR.duration(38), "38s")
        self.assertEqual(MONITOR.duration(432), "7m 12s")
        self.assertEqual(MONITOR.duration(4080), "1h 08m")


if __name__ == "__main__":
    unittest.main()
