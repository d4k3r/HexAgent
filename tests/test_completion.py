from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from hex_reconstruction.completion import (
    ClassicalLiteralMCTSCompletion,
    KataHexPassForbiddenCompletion,
)
from hex_reconstruction.evaluation import completion_positions
from hex_reconstruction.gtp import GTPClient, MoveKind
from hex_reconstruction.sync import SynchronizedKataHexSession


class CompletionTests(unittest.TestCase):
    def test_classical_mcts_is_seeded_physical_and_deterministic(self) -> None:
        board = completion_positions()[0].board()
        first = ClassicalLiteralMCTSCompletion(iterations=32, seed=1234)
        second = ClassicalLiteralMCTSCompletion(iterations=32, seed=1234)
        move_a = first.choose_action(board, virtual_winner=board.virtual_winner() or "")
        move_b = second.choose_action(board, virtual_winner=board.virtual_winner() or "")
        self.assertEqual(move_a, move_b)
        self.assertTrue(board.is_legal(move_a))
        self.assertIsNone(board.cells[move_a])

    def test_pass_forbidden_uses_completed_physical_play(self) -> None:
        engine = Path(__file__).with_name("fake_stateful_gtp_engine.py")
        with tempfile.TemporaryDirectory() as directory:
            with GTPClient((sys.executable, str(engine)), log_directory=Path(directory)) as client:
                session = SynchronizedKataHexSession(client)
                session.initialize()
                strategy = KataHexPassForbiddenCompletion(client)
                action = strategy.choose_action(session.board, virtual_winner="black")
                self.assertTrue(strategy.engine_played_last_action)
                session.accept_engine_action(action)
                self.assertEqual(len(strategy.diagnostics), 1)
                diagnostic = strategy.diagnostics[0]
                self.assertTrue(any(c.move_kind is MoveKind.PASS for c in diagnostic.candidates))
                self.assertNotEqual(diagnostic.chosen_move, "pass")


if __name__ == "__main__":
    unittest.main()
