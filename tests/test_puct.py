from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from hex_reconstruction.board import BLACK, WHITE, BOARD_AREA, HexBoard
from hex_reconstruction.puct import DeterministicPUCT, Evaluation, SearchConfig, TorchStudentEvaluator, compute_fpu_q
from hex_reconstruction.student_training import Group49Student, architecture_manifest


class FixedEvaluator:
    def __init__(self, *, logits: list[float] | None = None, value: float = 0.0) -> None:
        self.logits = logits or [0.0] * BOARD_AREA
        self.value = value
        self.calls = 0

    def evaluate(self, position: HexBoard) -> Evaluation:
        self.calls += 1
        return Evaluation(self.logits, self.value)


class SideValueEvaluator:
    def evaluate(self, position: HexBoard) -> Evaluation:
        # A leaf value that proves sign changes are reflected on root edges.
        return Evaluation([0.0] * BOARD_AREA, 0.75 if position.side_to_move == BLACK else -0.75)


class PUCTTests(unittest.TestCase):
    def test_fpu_equation_signs_and_prior_mass(self) -> None:
        self.assertEqual(compute_fpu_q("zero", 0.8, 0.0, 0.9), 0.0)
        self.assertAlmostEqual(compute_fpu_q("parent_value_reduced", 0.8, 0.25, 0.0), 0.8)
        self.assertAlmostEqual(compute_fpu_q("parent_value_reduced", 0.8, 0.25, 1.0), 0.55)
        self.assertLess(compute_fpu_q("parent_value_reduced", -0.8, 0.25, 1.0), -0.99)
        self.assertAlmostEqual(compute_fpu_q("parent_value_reduced", 0.0, 0.25, 0.0), 0.0)
        self.assertEqual(compute_fpu_q("parent_value_reduced", 0.2, 2.0, 4.0), -1.0)

    def test_explicit_baseline_configuration_is_identical(self) -> None:
        implicit = DeterministicPUCT(FixedEvaluator(), SearchConfig(simulations=17)).search(HexBoard())
        explicit = DeterministicPUCT(FixedEvaluator(), SearchConfig(simulations=17, c_puct=1.5, fpu_mode="zero", fpu_reduction=0.0)).search(HexBoard())
        self.assertEqual(implicit, explicit)
    def test_only_legal_actions_receive_visits_and_priors(self) -> None:
        board = HexBoard.from_setup(black=[0], white=[1], side_to_move=BLACK, last_move=1)
        logits = [0.0] * BOARD_AREA
        logits[0] = logits[1] = 1000.0
        result = DeterministicPUCT(FixedEvaluator(logits=logits), SearchConfig(simulations=5)).search(board)
        self.assertEqual(result.raw_visits[0], 0)
        self.assertEqual(result.raw_visits[1], 0)
        self.assertEqual(result.priors[0], 0.0)
        self.assertEqual(result.priors[1], 0.0)
        self.assertEqual(sum(result.raw_visits), 5)
        self.assertAlmostEqual(sum(result.policy), 1.0)

    def test_priors_normalize_over_legal_actions(self) -> None:
        board = HexBoard.from_setup(black=[0], white=[1], side_to_move=BLACK)
        logits = [0.0] * BOARD_AREA
        logits[2] = 2.0
        result = DeterministicPUCT(FixedEvaluator(logits=logits), SearchConfig(simulations=0)).search(board)
        self.assertAlmostEqual(sum(result.priors), 1.0)
        self.assertGreater(result.priors[2], result.priors[3])

    def test_immediate_terminal_win_is_backed_up_as_win_for_root_player(self) -> None:
        # Black has a top-to-bottom chain except cell 110; black to move can finish.
        black = [row * 11 for row in range(10)]
        white = [row * 11 + 1 for row in range(10)]
        board = HexBoard.from_setup(black=black, white=white, side_to_move=BLACK, last_move=109)
        logits = [-10.0] * BOARD_AREA
        logits[110] = 10.0
        result = DeterministicPUCT(FixedEvaluator(logits=logits, value=-0.5), SearchConfig(simulations=1)).search(board)
        self.assertEqual(result.selected_action, 110)
        self.assertEqual(result.raw_visits[110], 1)
        self.assertAlmostEqual(result.root_value, 1.0)

    def test_backup_alternates_side_to_move_perspective(self) -> None:
        board = HexBoard()
        logits = [-10.0] * BOARD_AREA
        logits[0] = 10.0
        result = DeterministicPUCT(SideValueEvaluator(), SearchConfig(simulations=1)).search(board)
        # First leaf after black's action is white-to-move (-0.75); root sees +0.75.
        self.assertAlmostEqual(result.root_value, 0.75)

    def test_dominant_prior_and_root_visit_rule(self) -> None:
        logits = [0.0] * BOARD_AREA
        logits[10] = 8.0
        result = DeterministicPUCT(FixedEvaluator(logits=logits), SearchConfig(simulations=4, c_puct=1.5)).search(HexBoard())
        self.assertEqual(result.selected_action, 10)
        self.assertGreater(result.raw_visits[10], 0)

    def test_source_unchanged_and_reproducible(self) -> None:
        board = HexBoard.from_setup(black=[0, 12], white=[1, 13], side_to_move=BLACK, last_move=13)
        before = (tuple(board.cells), board.side_to_move, board.last_move, board.ply)
        search = DeterministicPUCT(FixedEvaluator(), SearchConfig(simulations=9))
        first = search.search(board)
        second = search.search(board)
        self.assertEqual(first, second)
        self.assertEqual(before, (tuple(board.cells), board.side_to_move, board.last_move, board.ply))

    def test_terminal_root_has_no_evaluator_call(self) -> None:
        board = HexBoard.from_setup(black=[row * 11 for row in range(11)], side_to_move=WHITE, last_move=110)
        evaluator = FixedEvaluator()
        result = DeterministicPUCT(evaluator, SearchConfig(simulations=5)).search(board)
        self.assertIsNone(result.selected_action)
        self.assertEqual(result.root_value, -1.0)
        self.assertEqual(evaluator.calls, 0)

    def test_invalid_legal_logits_rejected(self) -> None:
        logits = [0.0] * BOARD_AREA
        logits[0] = float("nan")
        with self.assertRaises(ValueError):
            DeterministicPUCT(FixedEvaluator(logits=logits), SearchConfig(simulations=1)).search(HexBoard())

    def test_torch_evaluator_matches_shared_student_encoder(self) -> None:
        torch.manual_seed(11)
        model = Group49Student(channels=8, blocks=1).eval()
        config = {"architecture": architecture_manifest(model, channels=8, blocks=1)}
        board = HexBoard.from_setup(black=[0, 13], white=[1, 12], side_to_move=BLACK, last_move=12)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "student.pt"
            torch.save({"epoch": 7, "model_state": model.state_dict(), "config": config}, checkpoint)
            evaluator = TorchStudentEvaluator(checkpoint)
            result = evaluator.evaluate(board)
            with torch.no_grad():
                logits, value = model(torch.tensor(board.feature_planes(), dtype=torch.float32).reshape(1, 6, 11, 11))
        self.assertEqual(tuple(float(item) for item in logits[0]), result.policy_logits)
        self.assertEqual(float(value[0]), result.value)


if __name__ == "__main__":
    unittest.main()
