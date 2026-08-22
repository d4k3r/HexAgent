"""University-framework adapter for raw reconstructed-Student policy play.

This module deliberately contains no search: its result is raw policy gameplay,
not a PUCT/MCTS strength measurement.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import torch

from .board import BLACK, BOARD_AREA, BOARD_SIZE, HexBoard, WHITE
from .student_training import Group49Student


def action_from_xy(x: int, y: int) -> int:
    if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
        raise ValueError("framework coordinate outside 11x11 board")
    return y * BOARD_SIZE + x


def xy_from_action(action: int) -> tuple[int, int]:
    if not 0 <= action < BOARD_AREA:
        raise ValueError("physical action outside [0,120]")
    return action % BOARD_SIZE, action // BOARD_SIZE


def physical_colour(framework_colour: object) -> str:
    name = getattr(framework_colour, "name", None)
    if name == "RED":
        return BLACK
    if name == "BLUE":
        return WHITE
    raise ValueError(f"unknown university framework colour: {framework_colour!r}")


def framework_board_to_hexboard(board: object, side_to_move: object, last_physical_action: int | None) -> HexBoard:
    """Convert the university x-major Board to the corrected physical encoder."""
    if getattr(board, "size", None) != BOARD_SIZE:
        raise ValueError("Student adapter supports exactly an 11x11 university board")
    black: list[int] = []
    white: list[int] = []
    for x in range(BOARD_SIZE):
        for y in range(BOARD_SIZE):
            colour = board.tiles[x][y].colour
            if colour is None:
                continue
            action = action_from_xy(x, y)
            colour_name = getattr(colour, "name", None)
            if colour_name == "RED":
                black.append(action)
            elif colour_name == "BLUE":
                white.append(action)
            else:
                raise ValueError(f"unknown tile colour: {colour!r}")
    return HexBoard.from_setup(black, white, side_to_move=physical_colour(side_to_move), last_move=last_physical_action)


def encode_framework_board(board: object, side_to_move: object, last_physical_action: int | None) -> list[list[int]]:
    return framework_board_to_hexboard(board, side_to_move, last_physical_action).feature_planes()


def legal_actions(board: object) -> list[int]:
    return [action_from_xy(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if board.tiles[x][y].colour is None]


def select_policy_action(logits: torch.Tensor, legal: list[int], *, mode: Literal["argmax", "sample"] = "argmax", generator: torch.Generator | None = None) -> int:
    """Select only a legal physical action; `sample` is deterministic with its generator."""
    if logits.shape != (BOARD_AREA,):
        raise ValueError(f"expected 121 policy logits, got {tuple(logits.shape)}")
    if not legal:
        raise RuntimeError("no legal physical action")
    mask = torch.full_like(logits, float("-inf"))
    mask[torch.tensor(legal, device=logits.device)] = 0.0
    masked = logits + mask
    if mode == "argmax":
        return int(torch.argmax(masked).item())
    if mode == "sample":
        probabilities = torch.softmax(masked, dim=0)
        return int(torch.multinomial(probabilities, 1, generator=generator).item())
    raise ValueError(f"unknown policy selection mode: {mode}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class StudentPolicyAgent:  # dynamically derives from the university AgentBase below
    pass


def make_student_policy_agent_base(agent_base: type) -> type:
    """Create an AgentBase subclass without making HexAgent a package dependency here."""
    class _StudentPolicyAgent(agent_base):
        gameplay_kind = "raw_student_policy_no_mcts_or_puct"

        def __deepcopy__(self, memo: dict) -> "_StudentPolicyAgent":
            # Game.py defensively deep-copies players every turn. The immutable
            # inference model must be shared for that comparison, not copied.
            clone = self.__class__.__new__(self.__class__)
            memo[id(self)] = clone
            clone.__dict__ = self.__dict__.copy()
            return clone

        def __init__(self, colour: object, checkpoint: Path, *, device: str = "cpu", selection_mode: Literal["argmax", "sample"] = "argmax", sample_seed: int | None = None, swap_mode: Literal["never", "force_turn_2"] = "never") -> None:
            super().__init__(colour)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            architecture = payload["config"]["architecture"]
            self.model = Group49Student(channels=architecture["channels"], blocks=architecture["residual_blocks"]).to(device)
            self.model.load_state_dict(payload["model_state"])
            self.model.eval()
            self.device = torch.device(device)
            self.selection_mode = selection_mode
            self.swap_mode = swap_mode
            self.generator = torch.Generator(device=self.device).manual_seed(sample_seed) if sample_seed is not None else None
            self.checkpoint = Path(checkpoint)
            self.checkpoint_sha256 = sha256_file(self.checkpoint)
            self.last_physical_action: int | None = None

        def make_move(self, turn: int, board: object, opp_move: object | None) -> object:
            if opp_move is not None and not opp_move.is_swap():
                self.last_physical_action = action_from_xy(opp_move.x, opp_move.y)
            # Swap is a university control event, not a network policy action. The
            # forced setting exists solely to qualification-test framework semantics.
            if self.swap_mode == "force_turn_2" and turn == 2:
                from src.Move import Move
                return Move(-1, -1)
            planes = encode_framework_board(board, self.colour, self.last_physical_action)
            state = torch.tensor(planes, dtype=torch.float32, device=self.device).reshape(1, 6, BOARD_SIZE, BOARD_SIZE)
            with torch.no_grad():
                logits, _value = self.model(state)
            action = select_policy_action(logits[0], legal_actions(board), mode=self.selection_mode, generator=self.generator)
            self.last_physical_action = action
            from src.Move import Move
            x, y = xy_from_action(action)
            return Move(x, y)
    return _StudentPolicyAgent
