"""Versioned, deterministic trajectory action selection from teacher visits."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

from .board import HexBoard, gtp_to_action
from .gtp import CompletedAnalysis, MoveKind


ROOT_VISIT_SAMPLING_VERSION = "root-visit-sampling-v1"


@dataclass(frozen=True, slots=True)
class DiversityDecision:
    mechanism_id: str
    selected_action: int
    selected_move: str
    engine_selected_action: int
    engine_selected_move: str
    sampled: bool
    sampling_temperature: float | None
    deterministic_draw: float | None
    sampling_physical_ply_limit: int


class DeterministicRootVisitSampler:
    """Sample early actions while leaving the soft search target untouched.

    The random variate is a pure SHA-256 function of the version, experiment
    seed, physical ply, and physical board. It cannot depend on
    timing, process scheduling, Python's mutable RNG state, or swap metadata.
    After the configured opening horizon, KataHex's own final play-selection
    result is retained.
    """

    mechanism_id = ROOT_VISIT_SAMPLING_VERSION

    def __init__(self, *, sampling_physical_ply_limit: int = 20, temperature: float = 1.0) -> None:
        if not 1 <= sampling_physical_ply_limit <= 120:
            raise ValueError("sampling_physical_ply_limit must be in [1,120]")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.sampling_physical_ply_limit = sampling_physical_ply_limit
        self.temperature = temperature

    def choose(
        self,
        analysis: CompletedAnalysis,
        board: HexBoard,
        *,
        experiment_seed: str,
        game_id: str,
    ) -> DiversityDecision:
        if analysis.chosen_move_kind is not MoveKind.PHYSICAL:
            raise ValueError("strategic action selection requires a physical KataHex play")
        engine_action = gtp_to_action(analysis.chosen_move)
        if not board.is_legal(engine_action):
            raise ValueError("KataHex final play is not legal on the local board")

        if board.ply >= self.sampling_physical_ply_limit:
            return DiversityDecision(
                self.mechanism_id,
                engine_action,
                analysis.chosen_move,
                engine_action,
                analysis.chosen_move,
                False,
                None,
                None,
                self.sampling_physical_ply_limit,
            )

        weighted: list[tuple[int, str, float]] = []
        for candidate in analysis.candidates:
            if candidate.move_kind is not MoveKind.PHYSICAL:
                continue
            action = gtp_to_action(candidate.move)
            visits = candidate.visits
            if not board.is_legal(action) or visits is None or visits <= 0:
                continue
            weighted.append((action, candidate.move, float(visits) ** (1.0 / self.temperature)))
        weighted.sort(key=lambda item: item[0])
        total = sum(item[2] for item in weighted)
        if not weighted or total <= 0.0:
            raise ValueError("completed root analysis has no positive-visit legal physical action")

        board_text = "".join("." if cell is None else "B" if cell == "black" else "W" for cell in board.cells)
        key = "\0".join(
            (
                self.mechanism_id,
                experiment_seed,
                str(board.ply),
                board.side_to_move,
                board_text,
            )
        ).encode("utf-8")
        integer = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        draw = integer / float(1 << 64)
        threshold = draw * total
        cumulative = 0.0
        selected_action, selected_move, _ = weighted[-1]
        for action, move, weight in weighted:
            cumulative += weight
            if threshold < cumulative:
                selected_action, selected_move = action, move
                break
        return DiversityDecision(
            self.mechanism_id,
            selected_action,
            selected_move,
            engine_action,
            analysis.chosen_move,
            True,
            self.temperature,
            draw,
            self.sampling_physical_ply_limit,
        )
