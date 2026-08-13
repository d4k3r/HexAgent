"""Atomic per-game run manifests for resumable local pilot generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path
from typing import Any


class GameStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = {
    GameStatus.COMPLETED,
    GameStatus.QUARANTINED,
    GameStatus.FAILED,
}


@dataclass(frozen=True, slots=True)
class GameManifest:
    game_id: str
    status: GameStatus
    attempt: int
    raw_logs: tuple[str, ...]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "game_id": self.game_id,
            "status": self.status.value,
            "attempt": self.attempt,
            "raw_logs": list(self.raw_logs),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GameManifest":
        if value.get("manifest_version") != 1:
            raise ValueError("unsupported game manifest version")
        return cls(
            game_id=value["game_id"],
            status=GameStatus(value["status"]),
            attempt=int(value["attempt"]),
            raw_logs=tuple(value.get("raw_logs", ())),
            metadata=dict(value.get("metadata", {})),
        )


class GameManifestStore:
    """One atomic JSON file per game; completed IDs cannot be re-started."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, game_id: str) -> Path:
        if not game_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for character in game_id):
            raise ValueError("game_id must be a filename-safe identifier")
        return self.directory / f"{game_id}.json"

    def get(self, game_id: str) -> GameManifest | None:
        path = self.path_for(game_id)
        if not path.exists():
            return None
        return GameManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def start(
        self,
        game_id: str,
        *,
        raw_logs: tuple[str, ...],
        metadata: dict[str, Any],
    ) -> GameManifest:
        existing = self.get(game_id)
        if existing is not None and existing.status in TERMINAL_STATUSES:
            raise RuntimeError(
                f"refusing to duplicate terminal game {game_id} ({existing.status.value})"
            )
        attempt = 1 if existing is None else existing.attempt + 1
        manifest = GameManifest(game_id, GameStatus.STARTED, attempt, raw_logs, metadata)
        self._write(manifest)
        return manifest

    def finish(
        self,
        game_id: str,
        status: GameStatus,
        *,
        metadata_update: dict[str, Any] | None = None,
    ) -> GameManifest:
        if status not in TERMINAL_STATUSES:
            raise ValueError("finish status must be completed, quarantined, or failed")
        existing = self.get(game_id)
        if existing is None or existing.status is not GameStatus.STARTED:
            raise RuntimeError("only a started game can be finished")
        metadata = dict(existing.metadata)
        metadata.update(metadata_update or {})
        manifest = GameManifest(game_id, status, existing.attempt, existing.raw_logs, metadata)
        self._write(manifest)
        return manifest

    def interrupt(
        self,
        game_id: str,
        *,
        metadata_update: dict[str, Any] | None = None,
    ) -> GameManifest:
        existing = self.get(game_id)
        if existing is None or existing.status is not GameStatus.STARTED:
            raise RuntimeError("only a started game can be marked interrupted")
        metadata = dict(existing.metadata)
        metadata.update(metadata_update or {})
        manifest = GameManifest(
            game_id,
            GameStatus.INTERRUPTED,
            existing.attempt,
            existing.raw_logs,
            metadata,
        )
        self._write(manifest)
        return manifest

    def mark_started_interrupted(self) -> tuple[GameManifest, ...]:
        changed: list[GameManifest] = []
        for path in sorted(self.directory.glob("*.json")):
            manifest = GameManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if manifest.status is not GameStatus.STARTED:
                continue
            interrupted = GameManifest(
                manifest.game_id,
                GameStatus.INTERRUPTED,
                manifest.attempt,
                manifest.raw_logs,
                manifest.metadata,
            )
            self._write(interrupted)
            changed.append(interrupted)
        return tuple(changed)

    def reopen_failed_as_interrupted(
        self,
        game_id: str,
        *,
        metadata_update: dict[str, Any],
    ) -> GameManifest:
        """Explicitly authorize a preserved failed attempt for a new attempt.

        Callers must archive the failed manifest/checkpoint first. This method
        is intentionally separate from ``start`` so failures are never retried
        silently.
        """

        existing = self.get(game_id)
        if existing is None or existing.status is not GameStatus.FAILED:
            raise RuntimeError("only a failed game can be explicitly reopened")
        metadata = dict(existing.metadata)
        metadata.update(metadata_update)
        reopened = GameManifest(
            existing.game_id,
            GameStatus.INTERRUPTED,
            existing.attempt,
            existing.raw_logs,
            metadata,
        )
        self._write(reopened)
        return reopened

    def list(self) -> tuple[GameManifest, ...]:
        return tuple(
            GameManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.directory.glob("*.json"))
        )

    def update_metadata(self, game_id: str, update: dict[str, Any]) -> GameManifest:
        """Atomically annotate any existing state without changing its status."""

        existing = self.get(game_id)
        if existing is None:
            raise RuntimeError("cannot annotate a missing game manifest")
        metadata = dict(existing.metadata)
        metadata.update(update)
        manifest = GameManifest(
            existing.game_id,
            existing.status,
            existing.attempt,
            existing.raw_logs,
            metadata,
        )
        self._write(manifest)
        return manifest

    def _write(self, manifest: GameManifest) -> None:
        target = self.path_for(manifest.game_id)
        temporary = target.with_suffix(".json.tmp")
        payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n"
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
