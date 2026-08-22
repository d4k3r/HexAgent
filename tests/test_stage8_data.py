import json
import tempfile
import unittest
from pathlib import Path

from hex_reconstruction.stage8_data import SourceMixture, deterministic_game_split, iter_stage7_examples
from hex_reconstruction.symmetry import transform_example


def synthetic_stage7_game() -> dict:
    moves = [0, 1, 11, 2, 22, 3, 33, 4, 44, 5, 55, 6, 66, 7, 77, 8, 88, 9, 99, 10, 110]
    samples = []
    for ply, move in enumerate(moves):
        visits = [0] * 121; visits[move] = 8
        samples.append({"ply": ply, "side_to_move": "B" if ply % 2 == 0 else "W", "root_visits": visits, "selected_move": move, "z": 1 if ply % 2 == 0 else -1})
    return {"schema": "hex-selfplay-game-v1", "status": "complete", "game_id": 17, "game_seed": 99,
            "model_sha256": "frozen", "configuration_id": "stage7-test", "search_budget": 8, "c_puct": 1.5,
            "exploration": {"dirichlet": False}, "inference": {}, "initial_state": {"side_to_move": "B", "swap": False},
            "moves": moves, "samples": samples, "winner": "B", "game_length": len(moves)}


class Stage8DataTests(unittest.TestCase):
    def test_compact_stage7_round_trip_preserves_soft_policy_and_side_z(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "game-17.json"; path.write_text(json.dumps(synthetic_stage7_game()))
            examples = list(iter_stage7_examples(path))
        self.assertEqual(len(examples), 21)
        self.assertEqual(len(examples[0].policy.pi), 121)
        self.assertEqual(examples[0].policy.raw_visit_counts[0], 8)
        self.assertEqual(examples[0].value.z, 1.0)
        self.assertEqual(examples[1].value.z, -1.0)
        self.assertEqual(len(examples[0].provenance.config_sha256), 64)
        self.assertEqual(examples[0].provenance.search_settings["stage7_configuration_id"], "stage7-test")
        symmetric = transform_example(examples[5])
        self.assertEqual(symmetric.value.z, examples[5].value.z)
        self.assertEqual(sum(symmetric.policy.pi), 1.0)

    def test_game_split_and_mixture_are_reproducible(self):
        split = deterministic_game_split(["g0", "g1", "g2", "g3"], corpus_id="fixture", validation_fraction=0.25)
        self.assertEqual(set(split.values()), {"train", "validation"})
        mixture = SourceMixture({"teacher": 1.0, "selfplay": 1.0})
        self.assertEqual(mixture.schedule(20, seed=4901, epoch=2), mixture.schedule(20, seed=4901, epoch=2))
