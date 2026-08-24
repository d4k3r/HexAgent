from __future__ import annotations

import importlib.util
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


RUN = load("run_native_selfplay_throughput_benchmark_v1.py")
ANALYZE = load("analyze_native_selfplay_throughput_benchmark_v1.py")


class ThroughputContractTests(unittest.TestCase):
    def test_partition_is_deterministic_disjoint_and_complete(self) -> None:
        parts = RUN.partition_workload(9000000, 256, 3)
        self.assertEqual([(p["start_id"], p["end_exclusive"]) for p in parts], [(9000000, 9000086), (9000086, 9000171), (9000171, 9000256)])
        self.assertEqual(sum(p["games"] for p in parts), 256)

    def test_semantic_differences_are_reported(self) -> None:
        one = {"records": {"9": {"literal_winner": "B", "moves": [1, 2], "certificate_ply": 2, "phase_a_rows": 2}}, "fingerprints": {"9": "a"}}
        two = {"records": {"9": {"literal_winner": "W", "moves": [1, 3], "certificate_ply": 3, "phase_a_rows": 3}}, "fingerprints": {"9": "b"}}
        result = ANALYZE.semantic_comparison(one, two)
        self.assertFalse(result["exact_match"])
        self.assertEqual(result["differences"]["winner"], 1)
        self.assertEqual(result["differences"]["full_fingerprint"], 1)


if __name__ == "__main__":
    unittest.main()
