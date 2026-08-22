"""Small deterministic virtual-terminal completion evaluation positions."""

from __future__ import annotations

from dataclasses import dataclass

from .board import BLACK, WHITE, HexBoard


HISTORICAL_PASS_PREFIX = (
    (BLACK, "d8"), (WHITE, "e4"), (BLACK, "h4"), (WHITE, "g8"),
    (BLACK, "h8"), (WHITE, "g9"), (BLACK, "i9"), (WHITE, "h5"),
    (BLACK, "f5"), (WHITE, "c9"), (BLACK, "b8"), (WHITE, "d9"),
    (BLACK, "f7"), (WHITE, "f6"), (BLACK, "d7"), (WHITE, "e6"),
    (BLACK, "d6"), (WHITE, "e5"), (BLACK, "h6"), (WHITE, "c5"),
    (BLACK, "j4"), (WHITE, "i5"), (BLACK, "j5"), (WHITE, "i6"),
    (BLACK, "j6"), (WHITE, "i4"), (BLACK, "k2"), (WHITE, "j3"),
    (BLACK, "k3"), (WHITE, "k1"), (BLACK, "j2"), (WHITE, "j1"),
    (BLACK, "h3"), (WHITE, "i2"), (BLACK, "i3"), (WHITE, "i7"),
    (BLACK, "k7"), (WHITE, "j7"), (BLACK, "k6"), (WHITE, "h2"),
    (BLACK, "f3"), (WHITE, "j8"), (BLACK, "k8"), (WHITE, "j9"),
    (BLACK, "k9"), (WHITE, "j11"), (BLACK, "j10"), (WHITE, "i11"),
    (BLACK, "i10"), (WHITE, "g3"), (BLACK, "f4"), (WHITE, "g1"),
    (BLACK, "e2"), (WHITE, "e3"), (BLACK, "f2"), (WHITE, "h11"),
    (BLACK, "h10"), (WHITE, "g11"), (BLACK, "f8"), (WHITE, "g5"),
    (BLACK, "g4"), (WHITE, "h7"), (BLACK, "g10"), (WHITE, "f11"),
    (BLACK, "f10"), (WHITE, "e11"), (BLACK, "d10"), (WHITE, "e10"),
    (BLACK, "f9"), (WHITE, "e9"), (BLACK, "a10"), (WHITE, "a11"),
    (BLACK, "b10"), (WHITE, "b11"), (BLACK, "c10"),
)


_BLACK_DIAGONAL = tuple(
    pair
    for index in range(10)
    for pair in ((BLACK, f"{chr(97 + index)}{index + 1}"), (WHITE, f"k{index + 1}"))
) + ((BLACK, "k11"),)

_WHITE_FILLERS = ("k1", "j1", "k2", "i1", "j2", "k3", "h1", "i2", "j3", "k4", "g1")
_WHITE_DIAGONAL = tuple(
    pair
    for index in range(11)
    for pair in ((BLACK, _WHITE_FILLERS[index]), (WHITE, f"{chr(97 + index)}{index + 1}"))
)


@dataclass(frozen=True, slots=True)
class CompletionPosition:
    position_id: str
    moves: tuple[tuple[str, str], ...]
    provenance: str

    def board(self) -> HexBoard:
        board = HexBoard.from_moves(self.moves)
        if board.virtual_winner() is None or board.literal_winner() is not None:
            raise ValueError(f"{self.position_id} is not virtual-only terminal")
        return board

    def to_dict(self) -> dict[str, object]:
        board = self.board()
        return {
            "position_id": self.position_id,
            "moves": [[color, move] for color, move in self.moves],
            "board_cells": list(board.cells),
            "virtual_winner": board.virtual_winner(),
            "side_to_move": board.side_to_move,
            "ply": board.ply,
            "remaining_empty_cells": len(board.legal_actions()),
            "source_provenance": self.provenance,
        }


def completion_positions() -> tuple[CompletionPosition, ...]:
    """Winner/loser-to-move cases for both virtual colours and two densities."""

    return (
        CompletionPosition(
            "historical-pass-ply75-loser-to-move",
            HISTORICAL_PASS_PREFIX,
            "an audited private event ledger; sha256 "
            "9fe0901a2adbe254a21e7bd6e04962aa3f554bf0a703b9cfadbe7a0301607a15",
        ),
        CompletionPosition(
            "historical-pass-ply76-winner-to-move",
            HISTORICAL_PASS_PREFIX + ((WHITE, "a1"),),
            "deterministic derivative of historical ply75 using legal white resistance a1",
        ),
        CompletionPosition(
            "synthetic-black-bridge-loser-to-move",
            _BLACK_DIAGONAL,
            "reconstruction synthetic alternating jump/template regression v1",
        ),
        CompletionPosition(
            "synthetic-white-bridge-loser-to-move",
            _WHITE_DIAGONAL,
            "reconstruction synthetic alternating jump/template regression v1",
        ),
        CompletionPosition(
            "synthetic-white-bridge-winner-to-move",
            _WHITE_DIAGONAL + ((BLACK, "a11"),),
            "deterministic derivative of synthetic white regression using legal black resistance a11",
        ),
    )
