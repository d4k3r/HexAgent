#!/usr/bin/env python3
"""Build the deterministic, candidate-independent native-v2 prefix bank."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from forced_prefix_bank_v1 import (
    PREFIX_PLIES,
    BOARD_AREA,
    canonical_prefix,
    evaluation_overlap,
    orbit_id,
    sha256,
    transpose_prefix,
)


def build(count: int, seed: int) -> list[dict]:
    if count != 1024 or count % 2:
        raise ValueError("this controlled experiment requires exactly 1,024 prefixes")
    rng = random.Random(seed)
    rows: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    # Generate 512 independent prefixes and their geometric transposes.  The
    # first action is cycled deterministically to keep board-cell coverage
    # approximately uniform without consulting a model or strength heuristic.
    attempt = 0
    while len(rows) < count:
        first = attempt % BOARD_AREA
        attempt += 1
        tail = rng.sample([a for a in range(BOARD_AREA) if a != first], 2)
        actions = (first, *tail)
        transposed = transpose_prefix(actions)
        if actions == transposed or actions in seen or transposed in seen:
            continue
        seen.add(actions)
        seen.add(transposed)
        for item in (actions, transposed):
            rows.append({
                "prefix_id": f"forced-{len(rows):04d}",
                "actions": list(item),
                "canonical_actions": list(canonical_prefix(item)),
                "orbit_id": orbit_id(item),
                "transpose_of": f"forced-{len(rows) + 1:04d}" if item == actions else f"forced-{len(rows) - 1:04d}",
                "first_action": item[0],
            })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026082105)
    ap.add_argument("--evaluation-openings", type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise RuntimeError("refusing to overwrite immutable prefix bank")
    rows = build(1024, args.seed)
    first_counts = Counter(r["first_action"] for r in rows)
    payload = {
        "schema": "hex-forced-prefix-bank-v1",
        "count": len(rows),
        "prefix_plies": PREFIX_PLIES,
        "seed": args.seed,
        "generation": "MT19937 fixed-seed; first action cycles 0..120; remaining actions sampled without replacement",
        "transpose_policy": "every generated prefix is paired with coordinate transpose",
        "swap": False,
        "candidate_independent": True,
        "first_move_counts": dict(sorted(first_counts.items())),
        "prefixes": rows,
    }
    if args.evaluation_openings:
        overlaps = evaluation_overlap(rows, args.evaluation_openings)
        if overlaps:
            raise RuntimeError(f"prefix overlaps evaluation opening bank: {overlaps[:3]}")
        payload["evaluation_openings"] = {
            "path": str(args.evaluation_openings.resolve()),
            "sha256": sha256(args.evaluation_openings),
            "overlap_count": 0,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(args.output.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"output": str(args.output.resolve()), "count": len(rows), "sha256": sha256(args.output), "first_move_min": min(first_counts.values()), "first_move_max": max(first_counts.values())}, sort_keys=True))


if __name__ == "__main__":
    main()
