#!/usr/bin/env python3
"""Build the frozen, candidate-independent bank for the C1 PUCT/FPU study.

The builder only reads source corpora.  It materializes compact board setup
records plus immutable provenance; it never runs search and refuses an active
or unaudited native C1 source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

BOARD = 11
AREA = 121


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def entropy(counts: list[int] | None) -> float | None:
    if not counts:
        return None
    total = sum(counts)
    if total <= 0:
        return None
    return -sum((n / total) * math.log(n / total) for n in counts if n)


def phase_band(ply: int) -> str:
    if ply < 10:
        return "opening"
    if ply < 30:
        return "early"
    if ply < 60:
        return "mid"
    return "late"


def entropy_band(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 1.0:
        return "low"
    if value < 2.5:
        return "medium"
    return "high"


def transpose_action(action: int) -> int:
    return (action % BOARD) * BOARD + action // BOARD


def canonical_key(item: dict[str, Any]) -> tuple[Any, ...]:
    black = tuple(sorted(item["black"]))
    white = tuple(sorted(item["white"]))
    side = item["side_to_move"]
    last = item["last_move"]
    direct = (black, white, side, last)
    trans = (
        tuple(sorted(transpose_action(a) for a in white)),
        tuple(sorted(transpose_action(a) for a in black)),
        "W" if side == "B" else "B",
        None if last is None else transpose_action(last),
    )
    return min(direct, trans)


def move_action(move: Any) -> int:
    if isinstance(move, int):
        return move
    if isinstance(move, dict):
        return int(move.get("action", move.get("move")))
    if isinstance(move, (list, tuple)):
        return int(move[-1])
    raise ValueError(f"unsupported move record: {move!r}")


def board_before(moves: list[Any], ply: int) -> tuple[list[int], list[int], str, int | None]:
    black: list[int] = []
    white: list[int] = []
    for i, raw in enumerate(moves[:ply]):
        action = move_action(raw)
        if action < 0 or action >= AREA or action in black or action in white:
            raise ValueError(f"invalid move history at ply {i}: {action}")
        (black if i % 2 == 0 else white).append(action)
    return black, white, ("B" if ply % 2 == 0 else "W"), (move_action(moves[ply - 1]) if ply else None)


def record(*, source: str, source_path: Path, source_sha: str, game_id: str,
           ply: int, moves: list[Any], counts: list[int] | None,
           certificate_ply: int | None = None) -> dict[str, Any]:
    black, white, side, last = board_before(moves, ply)
    item = {
        "schema": "hex-puct-fpu-position-v1",
        "position_id": f"{source}:{game_id}:{ply}",
        "source": source,
        "source_path": str(source_path.resolve()),
        "source_sha256": source_sha,
        "game_id": game_id,
        "ply": ply,
        "side_to_move": side,
        "last_move": last,
        "black": black,
        "white": white,
        "root_visits": counts,
        "policy_entropy": entropy(counts),
        "phase_band": phase_band(ply),
        "entropy_band": entropy_band(entropy(counts)),
        "certificate_ply": certificate_ply,
        "certificate_distance": None if certificate_ply is None else max(0, certificate_ply - ply),
        "disagreement": "unavailable",
    }
    return item


def iter_teacher(root: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    games = sorted(p.parent for p in root.rglob("examples.jsonl"))
    if limit is not None:
        games = games[:limit]
    for game in games:
        trajectory_path = game / "trajectory.json"
        examples_path = game / "examples.jsonl"
        if not trajectory_path.exists():
            raise RuntimeError(f"teacher trajectory missing: {trajectory_path}")
        trajectory = json.loads(trajectory_path.read_text())
        moves = trajectory.get("moves", [])
        source_sha = sha256(trajectory_path)
        for line in examples_path.read_text().splitlines():
            ex = json.loads(line)
            policy = ex.get("policy", {})
            if policy.get("target_kind") != "mcts_visits" or float(policy.get("weight", 0)) <= 0:
                continue
            ply = int(ex["ply"])
            if ply >= len(moves):
                continue
            counts = policy.get("raw_visit_counts")
            if counts is None:
                counts = [round(float(x) * 100) for x in policy.get("pi", [])]
            yield record(source="T", source_path=trajectory_path, source_sha=source_sha,
                         game_id=str(ex["game_id"]), ply=ply, moves=moves, counts=counts,
                         certificate_ply=None)


def iter_historical(manifest_path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    rows = rows[:limit] if limit is not None else rows
    for row in rows:
        if row.get("status") != "retained" or not row.get("validated"):
            continue
        path = Path(row["source_path"])
        if not path.exists() or sha256(path) != row["source_sha256"]:
            raise RuntimeError(f"historical source hash mismatch: {path}")
        game = json.loads(path.read_text())
        for ply in range(int(row["retained_phase_a_rows"])):
            sample = game["samples"][ply]
            yield record(source="H", source_path=path, source_sha=row["source_sha256"],
                         game_id=str(row["game_id"]), ply=ply, moves=game["moves"],
                         counts=sample.get("root_visits"), certificate_ply=row.get("certificate_ply"))


def require_native_complete(root: Path) -> dict[str, Any]:
    manifest_path = root / "run-manifest.json"
    status_path = root / "runner-status.json"
    if not manifest_path.exists() or not status_path.exists():
        raise RuntimeError(f"native source lacks manifest/status: {root}")
    status = json.loads(status_path.read_text())
    if status.get("state") != "complete":
        raise RuntimeError(f"native source is not complete (state={status.get('state')}): {root}")
    audit = root / "postrun-audit.json"
    if not audit.exists():
        candidates = sorted(root.glob("postrun-audit*.json"))
        audit = candidates[-1] if candidates else audit
    if not audit.exists():
        raise RuntimeError(f"native source lacks final audit: {root}")
    audit_data = json.loads(audit.read_text())
    if not audit_data.get("complete", audit_data.get("passed", False)):
        raise RuntimeError(f"native source audit is not complete: {audit}")
    return json.loads(manifest_path.read_text())


def iter_native(root: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    manifest = require_native_complete(root)
    files = sorted((root / "games").glob("game-*.json"))
    if limit is not None:
        files = files[:limit]
    expected = int(manifest["game_ids"]["count"])
    if limit is None and len(files) != expected:
        raise RuntimeError(f"native source game count mismatch: {len(files)} != {expected}")
    for path in files:
        digest = sha256(path)
        game = json.loads(path.read_text())
        if game.get("status") != "accepted" or game.get("schema") != "hex-native-selfplay-game-v2":
            raise RuntimeError(f"invalid native game: {path}")
        cert_ply = game.get("certificate_ply")
        for sample in game.get("samples", []):
            yield record(source="F", source_path=path, source_sha=digest,
                         game_id=str(game["game_id"]), ply=int(sample["ply"]), moves=game["moves"],
                         counts=sample.get("root_visits"), certificate_ply=cert_ply)


def choose(candidates: list[dict[str, Any]], target: int, rng: random.Random, used: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in candidates:
        groups.setdefault((item["side_to_move"], item["phase_band"], item["entropy_band"]), []).append(item)
    for group in groups.values():
        group.sort(key=lambda x: x["position_id"])
        rng.shuffle(group)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    while len(selected) < target and keys:
        progress = False
        for key in keys:
            if not groups[key]:
                continue
            item = groups[key].pop()
            k = canonical_key(item)
            if k in used:
                continue
            used.add(k); selected.append(item); progress = True
            if len(selected) == target: break
        if not progress: break
    return selected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-root", type=Path, required=True)
    ap.add_argument("--historical-manifest", type=Path, required=True)
    ap.add_argument("--native-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026082207)
    ap.add_argument("--max-games-per-source", type=int)
    ap.add_argument("--target-positions", type=int, default=2048)
    args = ap.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"refusing existing bank root: {out}")
    if not args.teacher_root.exists() or not args.historical_manifest.exists() or not args.native_root.exists():
        raise RuntimeError("all three source inputs must exist")
    all_candidates = []
    all_candidates.extend(iter_teacher(args.teacher_root.resolve(), args.max_games_per_source))
    all_candidates.extend(iter_historical(args.historical_manifest.resolve(), args.max_games_per_source))
    all_candidates.extend(iter_native(args.native_root.resolve(), args.max_games_per_source))
    by_source = {s: [x for x in all_candidates if x["source"] == s] for s in ("T", "H", "F")}
    if args.target_positions < 3:
        raise RuntimeError("target-positions must allow at least one position per source")
    base, remainder = divmod(args.target_positions, 3)
    targets = {"T": base + (remainder > 0), "H": base + (remainder > 1), "F": base}
    rng = random.Random(args.seed); used: set[tuple[Any, ...]] = set(); selected = []
    for source in ("T", "H", "F"):
        chosen = choose(by_source[source], targets[source], rng, used)
        if len(chosen) != targets[source]:
            raise RuntimeError(f"source {source} has only {len(chosen)} unique positions; need {targets[source]}")
        selected.extend(chosen)
    selected.sort(key=lambda x: x["position_id"])
    out.mkdir(parents=True)
    records_path = out / "positions.jsonl"
    with records_path.open("w") as f:
        for item in selected:
            f.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema": "hex-puct-fpu-bank-v1",
        "version": 1,
        "selection": {"target_positions": args.target_positions, "seed": args.seed, "source_targets": targets,
                       "candidate_independent": True, "transpose_duplicate_policy": "one canonical orbit representative",
                       "deep_search_fields": "not used; disagreement is unavailable unless source metadata provides it"},
        "records": {"path": str(records_path.resolve()), "sha256": sha256(records_path), "count": len(selected)},
        "sources": {"T": str(args.teacher_root.resolve()), "H": str(args.historical_manifest.resolve()), "F": str(args.native_root.resolve())},
        "source_counts": dict(Counter(x["source"] for x in selected)),
        "strata": {"side": dict(Counter(x["side_to_move"] for x in selected)),
                    "phase_band": dict(Counter(x["phase_band"] for x in selected)),
                    "entropy_band": dict(Counter(x["entropy_band"] for x in selected))},
        "canonical_unique_count": len({canonical_key(x) for x in selected}),
        "source_sha256": {p["source_path"]: p["source_sha256"] for p in selected},
    }
    (out / "bank-manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(out), "positions": len(selected), "source_counts": manifest["source_counts"], "records_sha256": manifest["records"]["sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
