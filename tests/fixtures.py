"""Synthetic, public-safe fixtures for the versioned data contract."""

from hex_reconstruction.board import BLACK, WHITE, HexBoard, gtp_to_action
from hex_reconstruction.schema import PolicyRecord, ProvenanceRecord, StateRecord, TerminalRecord, TrainingExample, TransitionRecord, ValueRecord


def provenance() -> ProvenanceRecord:
    return ProvenanceRecord("0.1.0-test", "public-fixture", "synthetic-fixture", "fixture-v1", "fixture-model", "0" * 64, "fixture", "1" * 64, {"maxVisits": 4}, "fixture-seed", "tests/fixtures.py")


def _example(board: HexBoard, *, source: str, status: str, actions: tuple[int, ...] = (), chosen_action: int | None = None, virtual_winner: str | None = None, literal_winner: str | None = None, teacher_value: float | None = None) -> TrainingExample:
    counts = [0] * 121
    for offset, action in enumerate(actions): counts[action] = len(actions) - offset
    total = sum(counts)
    pi = [value / total for value in counts] if total else None
    return TrainingExample(
        game_id=f"fixture-{source}-{status}", ply=board.ply,
        state=StateRecord(board.feature_planes(), board.side_to_move),
        policy=PolicyRecord(pi, counts if total else None, board.legal_mask(), "mcts_visits" if total else None, 1.0 if total else 0.0),
        value=ValueRecord(1.0, "side_to_move", teacher_value, "utility" if teacher_value is not None else None, "side_to_move" if teacher_value is not None else None, 1.0),
        source=source, position_status=status, terminal=TerminalRecord(virtual_winner, literal_winner), transition=TransitionRecord(chosen_action), provenance=provenance())


def normal_teacher_fixture() -> TrainingExample:
    board = HexBoard.from_moves(((BLACK, "f6"),))
    actions = (gtp_to_action("e6"), gtp_to_action("g6"))
    return _example(board, source="katahex_teacher", status="normal", actions=actions, chosen_action=actions[0], teacher_value=0.25)


def normal_selfplay_fixture() -> TrainingExample:
    board = HexBoard.from_moves(((BLACK, "f6"), (WHITE, "e6")))
    actions = (gtp_to_action("f5"), gtp_to_action("f7"))
    return _example(board, source="group49_selfplay", status="normal", actions=actions, chosen_action=actions[0])


def virtual_completion_fixture() -> TrainingExample:
    black = tuple(gtp_to_action(move) for move in ("a1", "b2", "c3", "d4", "e5", "f6", "g7", "h8", "i9", "j10", "k11"))
    board = HexBoard.from_setup(black, side_to_move=WHITE, last_move=black[-1])
    assert board.virtual_winner() == BLACK and board.literal_winner() is None
    return _example(board, source="completion", status="virtual_terminal", chosen_action=gtp_to_action("a2"), virtual_winner=BLACK)


def literal_terminal_fixture() -> TrainingExample:
    black = tuple(gtp_to_action(f"a{row}") for row in range(1, 12))
    board = HexBoard.from_setup(black, side_to_move=WHITE, last_move=black[-1])
    return _example(board, source="completion", status="literal_terminal", virtual_winner=BLACK, literal_winner=BLACK)
