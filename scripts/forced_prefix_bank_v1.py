#!/usr/bin/env python3
"""Shared validation helpers for the immutable native-v2 forced-prefix bank."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BOARD_SIZE = 11
BOARD_AREA = BOARD_SIZE * BOARD_SIZE
PREFIX_PLIES = 3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transpose_action(action: int) -> int:
    return (action % BOARD_SIZE) * BOARD_SIZE + action // BOARD_SIZE


def transpose_prefix(actions: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(transpose_action(a) for a in actions)


def canonical_prefix(actions: tuple[int, ...]) -> tuple[int, ...]:
    return min(actions, transpose_prefix(actions))


def orbit_id(actions: tuple[int, ...]) -> str:
    return hashlib.sha256(
        ",".join(map(str, canonical_prefix(actions))).encode("ascii")
    ).hexdigest()[:16]


def legal_prefix(actions: tuple[int, ...]) -> bool:
    return (
        len(actions) == PREFIX_PLIES
        and all(0 <= a < BOARD_AREA for a in actions)
        and len(set(actions)) == PREFIX_PLIES
    )


def load_bank(path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "hex-forced-prefix-bank-v1":
        raise ValueError("wrong forced-prefix bank schema")
    if payload.get("prefix_plies") != PREFIX_PLIES:
        raise ValueError("forced-prefix bank must contain exactly three plies")
    rows = payload.get("prefixes")
    if not isinstance(rows, list) or len(rows) != payload.get("count"):
        raise ValueError("forced-prefix count mismatch")
    exact: set[tuple[int, ...]] = set()
    for index, row in enumerate(rows):
        actions = tuple(int(a) for a in row.get("actions", []))
        if row.get("prefix_id") != f"forced-{index:04d}":
            raise ValueError(f"unstable prefix id at row {index}")
        if not legal_prefix(actions):
            raise ValueError(f"illegal prefix at row {index}: {actions}")
        if actions in exact:
            raise ValueError(f"duplicate prefix at row {index}")
        exact.add(actions)
        if tuple(row.get("canonical_actions", [])) != canonical_prefix(actions):
            raise ValueError(f"canonical prefix mismatch at row {index}")
        if row.get("orbit_id") != orbit_id(actions):
            raise ValueError(f"orbit id mismatch at row {index}")
        partner = row.get("transpose_of")
        if not isinstance(partner, str) or not partner.startswith("forced-"):
            raise ValueError(f"missing transpose partner at row {index}")
    by_id = {row["prefix_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate prefix ids")
    for row in rows:
        partner = by_id.get(row["transpose_of"])
        if partner is None or tuple(partner["actions"]) != transpose_prefix(tuple(row["actions"])):
            raise ValueError(f"transpose partner mismatch for {row['prefix_id']}")
    return payload, rows


def evaluation_overlap(rows: list[dict], openings_path: Path) -> list[dict]:
    payload = json.loads(openings_path.read_text(encoding="utf-8"))
    exact = {tuple(int(a) for a in r.get("actions", [])) for r in rows}
    orbits = {canonical_prefix(tuple(int(a) for a in r.get("actions", []))) for r in rows}
    overlaps = []
    for opening in payload.get("openings", []):
        actions = tuple(int(a) for a in opening.get("opening_moves", []))[:PREFIX_PLIES]
        if len(actions) == PREFIX_PLIES and (actions in exact or canonical_prefix(actions) in orbits):
            overlaps.append({"pair_id": opening.get("pair_id"), "actions": list(actions)})
    return overlaps
