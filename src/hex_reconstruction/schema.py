"""Versioned JSONL contract for reconstructed Hex training examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
BOARD_SIZE = 11
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
PLANE_NAMES = ("red", "blue", "turn", "last_move", "conn_start", "conn_end")
SOURCES = frozenset(("katahex_teacher", "group49_selfplay", "completion"))
POSITION_STATUSES = frozenset(("normal", "virtual_terminal", "literal_terminal"))
COLORS = frozenset(("black", "white"))
VALUE_PERSPECTIVES = frozenset(("side_to_move", "black", "white"))


@dataclass(slots=True)
class StateRecord:
    planes: list[list[int]]
    side_to_move: str
    plane_names: list[str] = field(default_factory=lambda: list(PLANE_NAMES))
    shape: list[int] = field(default_factory=lambda: [6, BOARD_SIZE, BOARD_SIZE])


@dataclass(slots=True)
class PolicyRecord:
    pi: list[float] | None
    raw_visit_counts: list[int] | None
    legal_mask: list[bool]
    target_kind: str | None
    weight: float


@dataclass(slots=True)
class ValueRecord:
    z: float | None
    z_perspective: str
    teacher_root_value: float | None
    teacher_value_type: str | None
    teacher_value_perspective: str | None
    weight: float


@dataclass(slots=True)
class TerminalRecord:
    virtual_winner: str | None
    literal_winner: str | None
    completion_failed: bool = False


@dataclass(slots=True)
class TransitionRecord:
    chosen_action: int | None
    control_event: str | None = None


@dataclass(slots=True)
class ProvenanceRecord:
    generator_version: str
    generator_commit: str
    engine_repository: str
    engine_commit: str
    model_filename: str
    model_sha256: str
    model_release: str
    config_sha256: str
    search_settings: dict[str, Any]
    seed: str | int
    raw_log_reference: str


@dataclass(slots=True)
class TrainingExample:
    game_id: str
    ply: int
    state: StateRecord
    policy: PolicyRecord
    value: ValueRecord
    source: str
    position_status: str
    terminal: TerminalRecord
    transition: TransitionRecord
    provenance: ProvenanceRecord
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingExample":
        return cls(
            schema_version=value["schema_version"],
            game_id=value["game_id"],
            ply=value["ply"],
            state=StateRecord(**value["state"]),
            policy=PolicyRecord(**value["policy"]),
            value=ValueRecord(**value["value"]),
            source=value["source"],
            position_status=value["position_status"],
            terminal=TerminalRecord(**value["terminal"]),
            transition=TransitionRecord(**value["transition"]),
            provenance=ProvenanceRecord(**value["provenance"]),
        )


def write_jsonl(path: Path, examples: Iterable[TrainingExample]) -> None:
    """Serialize finalized examples. Callers should validate before writing."""

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(example.to_json())
            handle.write("\n")


def read_jsonl(path: Path) -> list[TrainingExample]:
    with path.open(encoding="utf-8") as handle:
        return [TrainingExample.from_dict(json.loads(line)) for line in handle if line.strip()]
