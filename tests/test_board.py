from __future__ import annotations

import unittest

from hex_reconstruction.board import BLACK, WHITE, HexBoard, action_to_gtp, gtp_to_action


class BoardTests(unittest.TestCase):
    def test_coordinate_round_trip_and_control_exclusion(self) -> None:
        for action in range(121):
            self.assertEqual(gtp_to_action(action_to_gtp(action)), action)
        for control in ("pass", "resign", "swap"):
            with self.assertRaises(ValueError):
                gtp_to_action(control)

    def test_literal_chain_and_feature_planes(self) -> None:
        board = HexBoard.from_moves(((BLACK, "a1"), (WHITE, "b1")))
        self.assertEqual(board.ply, 2)
        self.assertEqual(len(board.feature_planes()), 6)
        self.assertTrue(all(len(plane) == 121 for plane in board.feature_planes()))


if __name__ == "__main__":
    unittest.main()
