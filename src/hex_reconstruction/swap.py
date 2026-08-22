"""Pie-rule decisions separated from physical 121-action Hex search.

This module intentionally contains no threshold that turns a scalar into an
action.  A controller supplies that policy; these objects merely preserve a
diagnostic decision and expose a frozen opening-map backend.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Callable, Protocol

from .board import BOARD_AREA, BLACK, WHITE, HexBoard
from .puct import DeterministicPUCT, Evaluator, SearchConfig


class SwapChoice(str, Enum):
    SWAP = "SWAP"
    KEEP = "KEEP"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True, slots=True)
class SwapDecision:
    choice: SwapChoice
    backend: str
    evaluation_perspective: str
    scalar_evaluation: float | None
    search_budget: int | None
    actual_visits: int | None
    converged: bool
    note: str = ""


class SwapDecisionBackend(Protocol):
    def evaluate_one_stone_white_to_move(self, board: HexBoard) -> SwapDecision: ...


def one_stone_opening_action(board: HexBoard) -> int:
    """Validate the only physical state at which a university swap is legal."""
    stones = [action for action, colour in enumerate(board.cells) if colour is not None]
    if (
        len(stones) != 1
        or board.cells[stones[0]] != BLACK
        or board.side_to_move != WHITE
        or board.ply != 1
    ):
        raise ValueError("swap evaluation requires exactly one black stone and white to move")
    return stones[0]


class FrozenKataHexSwapMap:
    """Offline reference map; unresolved rows remain explicitly uncertain."""
    def __init__(self, path: Path) -> None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != "stage75-katahex-reference-map-v1":
            raise ValueError("not a Stage-7.5 KataHex reference map")
        rows = payload.get("openings")
        if not isinstance(rows, list) or len(rows) != BOARD_AREA:
            raise ValueError("reference map must contain 121 openings")
        self.payload = payload
        self.by_action = {int(row["action"]): row for row in rows}
        if set(self.by_action) != set(range(BOARD_AREA)):
            raise ValueError("reference map action coverage is not exactly 0..120")

    def evaluate_one_stone_white_to_move(self, board: HexBoard) -> SwapDecision:
        action = one_stone_opening_action(board)
        row = self.by_action[action]
        return SwapDecision(
            choice=SwapChoice(row["decision"]),
            backend="frozen_katahex_reference_map",
            evaluation_perspective=row["evaluation_perspective"],
            scalar_evaluation=row.get("reference_white_to_move_utility"),
            search_budget=row.get("deepest_budget"),
            actual_visits=row.get("actual_visits"),
            converged=bool(row.get("converged")),
            note=str(row.get("note", "")),
        )


class StudentSearchSwapBackend:
    """Search-improved Student backend with an injected decision policy.

    ``SearchResult.root_value`` is the visit-weighted mean root-edge value
    sum divided by total visits.  The PUCT implementation stores every root
    edge from the root (physical side-to-move) perspective, so on the required
    opening it is a physical-White-to-move estimate.  No selected-child value
    is substituted for that root estimate.
    """
    def __init__(
        self,
        evaluator: Evaluator,
        *,
        search_budget: int,
        c_puct: float = 1.5,
        policy: Callable[[float], SwapChoice] | None = None,
    ) -> None:
        self.evaluator = evaluator
        self.search_budget = search_budget
        self.c_puct = c_puct
        self.policy = policy

    def evaluate_one_stone_white_to_move(self, board: HexBoard) -> SwapDecision:
        one_stone_opening_action(board)
        result = DeterministicPUCT(
            self.evaluator, SearchConfig(self.search_budget, self.c_puct)
        ).search(board)
        if result.root_value is None:
            return SwapDecision(SwapChoice.UNCERTAIN, "student_puct", "physical_white_to_move", None,
                                self.search_budget, result.root_visits, False, "no root estimate")
        choice = self.policy(result.root_value) if self.policy else SwapChoice.UNCERTAIN
        return SwapDecision(choice, "student_puct", "physical_white_to_move", result.root_value,
                            self.search_budget, result.root_visits, True,
                            "root visit-weighted edge Q; decision policy injected" if self.policy else "root visit-weighted edge Q; no decision policy configured")
