"""Exact physical Black/White colour-transpose symmetry for 11x11 Hex."""
from __future__ import annotations

import copy
from typing import Sequence

from .board import BLACK, BOARD_AREA, BOARD_SIZE, WHITE, HexBoard, opponent
from .schema import TrainingExample


def transpose_action(action: int) -> int:
    """Map physical row-major ``(row, col)`` to ``(col, row)``."""

    if not 0 <= action < BOARD_AREA:
        raise ValueError("action outside physical 11x11 board")
    row, column = divmod(action, BOARD_SIZE)
    return column * BOARD_SIZE + row


def transpose_vector(values: Sequence[object]) -> list[object]:
    if len(values) != BOARD_AREA:
        raise ValueError("expected a 121-action physical vector")
    result = [None] * BOARD_AREA
    for action, value in enumerate(values):
        result[transpose_action(action)] = value
    return result


def board_from_example(example: TrainingExample) -> HexBoard:
    """Rebuild the authoritative physical board from its red/blue/turn/last planes."""

    planes = example.state.planes
    if len(planes) != 6 or any(len(plane) != BOARD_AREA for plane in planes):
        raise ValueError("example does not contain a [6,11,11] state")
    black = [action for action, value in enumerate(planes[0]) if value]
    white = [action for action, value in enumerate(planes[1]) if value]
    if any(planes[0][action] and planes[1][action] for action in range(BOARD_AREA)):
        raise ValueError("example has overlapping physical stones")
    last = [action for action, value in enumerate(planes[3]) if value]
    if len(last) > 1:
        raise ValueError("example has more than one last-move marker")
    board = HexBoard.from_setup(black, white, side_to_move=example.state.side_to_move, last_move=last[0] if last else None)
    if board.feature_planes() != planes:
        raise ValueError("example derived feature planes do not match authoritative board encoding")
    return board


def transpose_colour_board(board: HexBoard) -> HexBoard:
    """Transpose coordinates, swap colours and side-to-move, then recompute DSUs."""

    black = [transpose_action(action) for action, colour in enumerate(board.cells) if colour == WHITE]
    white = [transpose_action(action) for action, colour in enumerate(board.cells) if colour == BLACK]
    return HexBoard.from_setup(
        black,
        white,
        side_to_move=opponent(board.side_to_move),
        last_move=transpose_action(board.last_move) if board.last_move is not None else None,
    )


def _swap_colour(value: str | None) -> str | None:
    return opponent(value) if value is not None else None


def transform_example(example: TrainingExample) -> TrainingExample:
    """Semantic transformation; derived planes are regenerated via ``HexBoard``.

    Final z is invariant because both final physical winner and state side to
    move swap colour. Teacher root value is also role-preserving because its
    recorded perspective is side-to-move in this contract.
    """

    board = transpose_colour_board(board_from_example(example))
    transformed = copy.deepcopy(example)
    transformed.state.planes = board.feature_planes()
    transformed.state.side_to_move = board.side_to_move
    transformed.policy.legal_mask = [bool(value) for value in transpose_vector(example.policy.legal_mask)]
    transformed.policy.pi = [float(value) for value in transpose_vector(example.policy.pi)] if example.policy.pi is not None else None
    transformed.policy.raw_visit_counts = [int(value) for value in transpose_vector(example.policy.raw_visit_counts)] if example.policy.raw_visit_counts is not None else None
    transformed.terminal.virtual_winner = _swap_colour(example.terminal.virtual_winner)
    transformed.terminal.literal_winner = _swap_colour(example.terminal.literal_winner)
    if transformed.transition.chosen_action is not None:
        transformed.transition.chosen_action = transpose_action(transformed.transition.chosen_action)
    return transformed


def transformed_training_tensors(example: TrainingExample) -> tuple[list[list[int]], list[float], list[bool], float]:
    """Efficient projection of the already-qualified semantic transform.

    ``transform_example`` remains the authoritative implementation and
    reconstructs a board/DSUs.  The channel map below is mathematically the
    same operation: colour swap plus coordinate transpose swaps stone planes;
    the turn bit complements; last-move and both own-colour connectivity
    relations transpose in place.  Tests compare this fast projection against
    a fresh semantic board encode, including near-terminal connectivity.
    """

    planes = example.state.planes
    if len(planes) != 6 or any(len(plane) != BOARD_AREA for plane in planes):
        raise ValueError("example does not contain a [6,11,11] state")
    state = [
        [int(value) for value in transpose_vector(planes[1])],  # transformed black = old white
        [int(value) for value in transpose_vector(planes[0])],  # transformed white = old black
        [1 - int(value) for value in transpose_vector(planes[2])],
        [int(value) for value in transpose_vector(planes[3])],
        [int(value) for value in transpose_vector(planes[4])],
        [int(value) for value in transpose_vector(planes[5])],
    ]
    return (
        state,
        [float(value) for value in transpose_vector(example.policy.pi)] if example.policy.pi is not None else [0.0] * BOARD_AREA,
        [bool(value) for value in transpose_vector(example.policy.legal_mask)],
        float(example.value.z) if example.value.z is not None else 0.0,
    )
