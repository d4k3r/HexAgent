from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from hex_reconstruction.board import BLACK, WHITE, HexBoard
from hex_reconstruction.swap import FrozenKataHexSwapMap, SwapChoice, one_stone_opening_action
from hex_reconstruction.sync import UniversityOpening


class SwapControlTests(unittest.TestCase):
    def test_university_control_event_preserves_physical_opening(self) -> None:
        board = HexBoard(); board.play(60)
        before_cells = tuple(board.cells); before_ply = board.ply; before_legal = board.legal_actions()
        opening = UniversityOpening("first", "second")
        opening.apply_swap(board)
        self.assertEqual(tuple(board.cells), before_cells)
        self.assertEqual(board.ply, before_ply)
        self.assertEqual(board.side_to_move, WHITE)
        self.assertEqual(board.legal_actions(), before_legal)
        self.assertEqual(opening.controller_to_color, {"first": WHITE, "second": BLACK})
        with self.assertRaises(ValueError): opening.apply_swap(board)

    def test_only_the_one_black_opening_is_a_swap_evaluation_state(self) -> None:
        board = HexBoard();
        with self.assertRaises(ValueError): one_stone_opening_action(board)
        board.play(7)
        self.assertEqual(one_stone_opening_action(board), 7)
        board.play(8)
        with self.assertRaises(ValueError): one_stone_opening_action(board)

    def test_reference_backend_preserves_uncertain(self) -> None:
        rows = [{"action": a, "decision": "UNCERTAIN", "evaluation_perspective": "physical_white_to_move", "reference_white_to_move_utility": 0.0, "deepest_budget": 100, "actual_visits": 100, "converged": False} for a in range(121)]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "map.json"
            path.write_text(json.dumps({"schema": "stage75-katahex-reference-map-v1", "openings": rows}))
            backend = FrozenKataHexSwapMap(path)
            board = HexBoard(); board.play(12)
            self.assertEqual(backend.evaluate_one_stone_white_to_move(board).choice, SwapChoice.UNCERTAIN)

    def test_student_backend_value_is_not_a_selected_child_value(self) -> None:
        from hex_reconstruction.puct import Evaluation
        from hex_reconstruction.swap import StudentSearchSwapBackend
        class Constant:
            def evaluate(self, board): return Evaluation([0.0] * 121, -0.25)
        board = HexBoard(); board.play(60)
        result = StudentSearchSwapBackend(Constant(), search_budget=8).evaluate_one_stone_white_to_move(board)
        self.assertEqual(result.choice, SwapChoice.UNCERTAIN)  # policy remains separate
        self.assertEqual(result.evaluation_perspective, "physical_white_to_move")
        self.assertEqual(result.actual_visits, 8)
        self.assertIsNotNone(result.scalar_evaluation)


if __name__ == "__main__":
    unittest.main()
