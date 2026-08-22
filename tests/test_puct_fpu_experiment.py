from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_puct_fpu_bank_v1 import canonical_key, transpose_action  # noqa: E402
from analyze_puct_fpu_v1 import nominate  # noqa: E402


class PuctFpuExperimentTests(unittest.TestCase):
    def test_nomination_is_operational_v128_only_and_causal(self) -> None:
        baseline = {
            "configuration": "c1.5-zero-r0-v128", "visits": 128,
            "c_puct": 1.5, "action_agreement": .938, "top3_agreement": .73,
            "policy_tv": .25, "visited_children": 37., "max_depth": 4.,
            "search_seconds": .7,
        }
        rows = [baseline,
                {**baseline, "configuration": "c1.5-zero-r0-v2048", "visits": 2048, "action_agreement": .99},
                {**baseline, "configuration": "c9-zero-r0-v512", "visits": 512, "action_agreement": .999},
                {**baseline, "configuration": "c1.5-parent_value_reduced-r0.25-v128", "fpu_mode": "parent_value_reduced", "fpu_reduction": .25, "action_agreement": .95, "top3_agreement": .86, "policy_tv": .10, "visited_children": 7., "max_depth": 8.},
                {**baseline, "configuration": "c2.5-parent_value_reduced-r0.25-v128", "c_puct": 2.5, "fpu_mode": "parent_value_reduced", "fpu_reduction": .25, "action_agreement": .94, "top3_agreement": .88, "policy_tv": .09, "visited_children": 7., "max_depth": 8.},
                ]
        selected, rule = nominate(rows, "c1.5-zero-r0-v128", 128)
        self.assertEqual([x["configuration"] for x in selected], [
            "c1.5-parent_value_reduced-r0.25-v128",
            "c2.5-parent_value_reduced-r0.25-v128",
        ])
        self.assertTrue(rule["excluded_diagnostic_reference"])

    def test_runtime_estimator_uses_budget_components(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for budget in (128, 512, 2048):
                child = root / f"c1.5-zero-r0-v{budget}-32pos"
                child.mkdir()
                (child / "telemetry.json").write_text(json.dumps({"simulations_per_second": float(budget)}))
            output = root / "estimate.json"
            subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "estimate_puct_fpu_sweep_runtime_v1.py"),
                "--calibration-root", str(root), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            estimate = json.loads(output.read_text())
            self.assertEqual(estimate["requested_simulations"], 66060288)
            self.assertGreater(estimate["expected_seconds"], 0)

    def test_transpose_orbit_is_deduplicated(self) -> None:
        item = {
            "black": [0, 13], "white": [1, 24], "side_to_move": "B", "last_move": 24,
        }
        transposed = {
            "black": sorted(transpose_action(a) for a in item["white"]),
            "white": sorted(transpose_action(a) for a in item["black"]),
            "side_to_move": "W", "last_move": transpose_action(item["last_move"]),
        }
        self.assertEqual(canonical_key(item), canonical_key(transposed))

    def test_bank_records_are_jsonl_merge_friendly(self) -> None:
        record = {"position_id": "T:example:0", "source": "T", "root_visits": [0] * 121}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(json.dumps(record) + "\n")
            self.assertEqual(json.loads(path.read_text())[
                "position_id"], "T:example:0")

    def test_equal_wall_plan_uses_operational_128_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, seconds, simulations in (
                ("c1.5-zero-r0-v128", 1.0, 128),
                ("c1.5-zero-r0-v2048", 10.0, 2048),
                ("c0.75-zero-r0-v128", 2.0, 128),
            ):
                child = root / name
                child.mkdir()
                (child / "results.jsonl").write_text(json.dumps({"search_seconds": seconds, "simulations": simulations}) + "\n")
            output = root / "plan.json"
            subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "scripts" / "plan_puct_fpu_equal_wall_v1.py"),
                "--sweep-root", str(root), "--output", str(output),
            ], check=True, capture_output=True, text=True)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["operational_baseline"], "c1.5-zero-r0-v128")
            self.assertEqual(plan["diagnostic_reference"], "c1.5-zero-r0-v2048")
            self.assertEqual(plan["operational_baseline_mean_seconds"], 1.0)
            self.assertEqual(plan["plans"][0]["equal_wall_budget_estimate"], 64)
            self.assertNotIn("c1.5-zero-r0-v2048", {item["configuration"] for item in plan["plans"]})


if __name__ == "__main__":
    unittest.main()
