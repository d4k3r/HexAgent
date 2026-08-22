from __future__ import annotations

import unittest

from hex_reconstruction.board import BLACK, WHITE, BOARD_AREA, HexBoard
from hex_reconstruction.schema import (
    PolicyRecord, ProvenanceRecord, StateRecord, TerminalRecord, TrainingExample,
    TransitionRecord, ValueRecord,
)
from hex_reconstruction.symmetry import (
    board_from_example, transform_example, transformed_training_tensors, transpose_action, transpose_colour_board, transpose_vector,
)


def example_for(board: HexBoard, *, z: float = 1.0) -> TrainingExample:
    pi = [0.0] * BOARD_AREA
    legal = board.legal_actions()
    if legal:
        pi[legal[0]] = 1.0
    return TrainingExample(
        game_id="symmetry-fixture", ply=board.ply,
        state=StateRecord(planes=board.feature_planes(), side_to_move=board.side_to_move),
        policy=PolicyRecord(pi=pi, raw_visit_counts=[int(value * 99) for value in pi], legal_mask=board.legal_mask(), target_kind="root_visits", weight=1.0),
        value=ValueRecord(z=z, z_perspective="side_to_move", teacher_root_value=0.25, teacher_value_type="utility", teacher_value_perspective="side_to_move", weight=1.0),
        source="katahex_teacher", position_status="normal",
        terminal=TerminalRecord(virtual_winner=None, literal_winner=None),
        transition=TransitionRecord(chosen_action=legal[0] if legal else None),
        provenance=ProvenanceRecord("test","test","test","test","model","0" * 64,"test","0" * 64,{},1,"test"),
    )


class SymmetryTests(unittest.TestCase):
    def test_coordinate_and_vector_are_involutions(self) -> None:
        self.assertEqual(transpose_action(2 * 11 + 7), 7 * 11 + 2)
        vector = list(range(BOARD_AREA))
        self.assertEqual(transpose_vector(transpose_vector(vector)), vector)

    def test_empty_early_and_midgame_are_physical_involutions(self) -> None:
        boards = [
            HexBoard(),
            HexBoard.from_setup(black=[0, 13], white=[1, 12], side_to_move=BLACK, last_move=12),
            HexBoard.from_setup(black=[0, 12, 24, 35], white=[1, 13, 25], side_to_move=WHITE, last_move=35),
        ]
        for board in boards:
            transformed = transpose_colour_board(board)
            restored = transpose_colour_board(transformed)
            self.assertEqual(restored.cells, board.cells)
            self.assertEqual(restored.side_to_move, board.side_to_move)
            self.assertEqual(restored.last_move, board.last_move)
            self.assertEqual(restored.feature_planes(), board.feature_planes())

    def test_example_transform_reencodes_every_plane_and_policy(self) -> None:
        board = HexBoard.from_setup(black=[0, 13, 25], white=[1, 12], side_to_move=WHITE, last_move=25)
        example = example_for(board, z=-1.0)
        transformed = transform_example(example)
        semantic = transpose_colour_board(board)
        self.assertEqual(transformed.state.planes, semantic.feature_planes())
        self.assertEqual(transformed.state.side_to_move, BLACK)
        self.assertEqual(transformed.policy.legal_mask, semantic.legal_mask())
        self.assertAlmostEqual(sum(transformed.policy.pi or []), 1.0)
        self.assertEqual(transform_example(transformed).state.planes, example.state.planes)
        self.assertEqual(transform_example(transformed).policy.pi, example.policy.pi)
        self.assertEqual(transformed.value.z, example.value.z)
        self.assertEqual(board_from_example(transformed).feature_planes(), transformed.state.planes)
        state, pi, legal, z = transformed_training_tensors(example)
        self.assertEqual(state, transformed.state.planes)
        self.assertEqual(pi, transformed.policy.pi)
        self.assertEqual(legal, transformed.policy.legal_mask)
        self.assertEqual(z, transformed.value.z)

    def test_terminal_winner_swaps_and_z_is_invariant(self) -> None:
        board = HexBoard.from_setup(black=[row * 11 for row in range(11)], white=[row * 11 + 1 for row in range(10)], side_to_move=WHITE, last_move=110)
        transformed = transpose_colour_board(board)
        self.assertEqual(board.literal_winner(), BLACK)
        self.assertEqual(transformed.literal_winner(), WHITE)
        # Original White-to-move lost (z=-1); transformed Black-to-move also lost.
        example = example_for(board, z=-1.0)
        example.terminal.literal_winner = BLACK
        result = transform_example(example)
        self.assertEqual(result.terminal.literal_winner, WHITE)
        self.assertEqual(result.value.z, -1.0)

    def test_near_terminal_connectivity_is_freshly_recomputed(self) -> None:
        board = HexBoard.from_setup(black=[row * 11 for row in range(10)], white=[row * 11 + 1 for row in range(10)], side_to_move=BLACK, last_move=100)
        transformed = transform_example(example_for(board))
        self.assertEqual(transformed.state.planes, transpose_colour_board(board).feature_planes())


if __name__ == "__main__":
    unittest.main()
