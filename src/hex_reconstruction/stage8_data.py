"""Immutable Stage-7 compact-corpus adapter and Stage-8 sampling primitives."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .board import BLACK, WHITE, HexBoard
from .schema import PolicyRecord, ProvenanceRecord, StateRecord, TerminalRecord, TrainingExample, TransitionRecord, ValueRecord
from .validation import validate_example

AREA = 121
STAGE7_SCHEMA = "hex-selfplay-game-v1"
STAGE7_SOURCE = "stage7-champion-0"


def _colour(letter: str) -> str:
    if letter == "B":
        return BLACK
    if letter == "W":
        return WHITE
    raise ValueError(f"invalid Stage-7 colour {letter!r}")


def _stage7_config_sha256(game: dict) -> str:
    """Training schema requires a digest; compact Stage-7 records keep an ID."""
    frozen = {key: game[key] for key in ("configuration_id", "search_budget", "c_puct", "exploration", "inference", "model_sha256")}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _game_files(root: Path) -> list[Path]:
    files = sorted((root / "games").glob("game-*.json"))
    if not files:
        raise ValueError(f"no completed Stage-7 games under {root}")
    if list((root / "games").glob("*.partial")):
        raise ValueError("incomplete Stage-7 game files are not a training source")
    return files


def load_stage7_game(path: Path) -> dict:
    game = json.loads(path.read_text())
    required = {"schema", "status", "game_id", "game_seed", "model_sha256", "configuration_id", "search_budget", "c_puct", "exploration", "inference", "initial_state", "moves", "samples", "winner", "game_length"}
    if not required <= game.keys() or game["schema"] != STAGE7_SCHEMA or game["status"] != "complete":
        raise ValueError(f"not a completed {STAGE7_SCHEMA} record: {path}")
    if game["initial_state"] != {"side_to_move": "B", "swap": False}:
        raise ValueError("Stage-8 adapter only accepts frozen no-swap Stage-7 games")
    if not isinstance(game["game_id"], int) or len(game["moves"]) != game["game_length"] or len(game["samples"]) != game["game_length"]:
        raise ValueError("invalid Stage-7 game shape")
    return game


def iter_stage7_examples(path: Path) -> Iterator[TrainingExample]:
    """Reconstruct every pre-move state from the compact immutable game record."""
    game = load_stage7_game(path)
    board = HexBoard()
    winner = _colour(game["winner"])
    for ply, (move, sample) in enumerate(zip(game["moves"], game["samples"])):
        if board.literal_winner() is not None:
            raise ValueError(f"post-terminal Stage-7 move in {path.name}")
        if not isinstance(move, int) or not board.is_legal(move):
            raise ValueError(f"illegal Stage-7 move at ply {ply}")
        if sample.get("ply") != ply or sample.get("selected_move") != move or sample.get("side_to_move") not in ("B", "W"):
            raise ValueError(f"sample/move provenance mismatch at ply {ply}")
        if _colour(sample["side_to_move"]) != board.side_to_move:
            raise ValueError(f"side-to-move mismatch at ply {ply}")
        visits = sample.get("root_visits")
        if not isinstance(visits, list) or len(visits) != AREA or any(type(value) is not int or value < 0 for value in visits):
            raise ValueError(f"invalid root visits at ply {ply}")
        if sum(visits) != game["search_budget"] or visits[move] <= 0:
            raise ValueError(f"root visit accounting mismatch at ply {ply}")
        legal = board.legal_mask()
        if any(count for count, allowed in zip(visits, legal) if not allowed):
            raise ValueError(f"illegal root visit at ply {ply}")
        expected_z = 1.0 if board.side_to_move == winner else -1.0
        if float(sample.get("z")) != expected_z:
            raise ValueError(f"side-relative z mismatch at ply {ply}")
        total = sum(visits)
        pi = [count / total for count in visits]
        example = TrainingExample(
            game_id=f"stage7-c0-{game['game_id']}", ply=ply,
            state=StateRecord(board.feature_planes(), board.side_to_move),
            policy=PolicyRecord(pi, visits, legal, "mcts_visits", 1.0),
            value=ValueRecord(expected_z, "side_to_move", None, None, None, 1.0),
            source="group49_selfplay", position_status="normal",
            terminal=TerminalRecord(None, None), transition=TransitionRecord(move),
            provenance=ProvenanceRecord(
                generator_version="stage7-cpp-selfplay-v1", generator_commit="frozen-stage7",
                engine_repository="reconstruction", engine_commit="frozen-stage7",
                model_filename="student-seed4901-epoch11-dynamic-batch.onnx", model_sha256=game["model_sha256"],
                model_release="stage7-champion-0", config_sha256=_stage7_config_sha256(game),
                search_settings={"stage7_configuration_id": game["configuration_id"], "budget": game["search_budget"], "c_puct": game["c_puct"], "exploration": game["exploration"], "inference": game["inference"]},
                seed=game["game_seed"], raw_log_reference=str(path.resolve()),
            ),
        )
        validate_example(example)
        yield example
        board.play(move)
    if board.literal_winner() != winner:
        raise ValueError(f"literal winner mismatch after replay: {path.name}")


def iter_stage7_corpus(root: Path) -> Iterator[tuple[Path, Iterator[TrainingExample]]]:
    for path in _game_files(root):
        yield path, iter_stage7_examples(path)


def deterministic_game_split(game_ids: Iterable[str], *, corpus_id: str, validation_fraction: float = 0.1) -> dict[str, str]:
    """One immutable game-level split; transforms inherit their source game's side."""
    ids = sorted(set(game_ids))
    if len(ids) < 2 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("need at least two games and a proper validation fraction")
    count = max(1, min(len(ids) - 1, round(len(ids) * validation_fraction)))
    ordered = sorted(ids, key=lambda gid: hashlib.sha256(f"{corpus_id}:{gid}".encode()).hexdigest())
    validation = set(ordered[:count])
    return {gid: "validation" if gid in validation else "train" for gid in ids}


@dataclass(frozen=True)
class SourceMixture:
    """Explicit source weights; sampling is deterministic for a seed and epoch."""
    weights: dict[str, float]

    def __post_init__(self) -> None:
        if not self.weights or any(not math.isfinite(weight) or weight < 0 for weight in self.weights.values()) or not any(weight > 0 for weight in self.weights.values()):
            raise ValueError("source weights must be finite/nonnegative with at least one positive source")

    def schedule(self, draws: int, *, seed: int, epoch: int) -> list[str]:
        if draws < 0:
            raise ValueError("draws must be non-negative")
        names = sorted(name for name, weight in self.weights.items() if weight > 0)
        weights = [self.weights[name] for name in names]
        rng = random.Random(f"stage8-mixture-v1:{seed}:{epoch}")
        return rng.choices(names, weights=weights, k=draws)
