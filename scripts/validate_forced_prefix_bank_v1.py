#!/usr/bin/env python3
"""Read-only audit for the immutable forced-prefix bank."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from forced_prefix_bank_v1 import evaluation_overlap, load_bank, sha256, transpose_prefix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--evaluation-openings", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    payload, rows = load_bank(args.bank)
    exact = [tuple(r["actions"]) for r in rows]
    transposed = [tuple(transpose_prefix(tuple(r["actions"]))) for r in rows]
    overlaps = evaluation_overlap(rows, args.evaluation_openings)
    counts = Counter(r["first_action"] for r in rows)
    orbit_counts = Counter(r["orbit_id"] for r in rows)
    audit = {
        "schema": "hex-forced-prefix-bank-audit-v1",
        "bank_sha256": sha256(args.bank),
        "evaluation_openings_sha256": sha256(args.evaluation_openings),
        "count": len(rows),
        "legal": True,
        "exact_duplicates": len(exact) - len(set(exact)),
        "transpose_orbit_count": len({r["orbit_id"] for r in rows}),
        "transpose_pair_count": sum(1 for n in orbit_counts.values() if n == 2),
        "transpose_orbit_size_min": min(orbit_counts.values()),
        "transpose_orbit_size_max": max(orbit_counts.values()),
        "first_move_min": min(counts.values()),
        "first_move_max": max(counts.values()),
        "evaluation_overlap_count": len(overlaps),
        "evaluation_overlaps": overlaps,
        "swap": payload.get("swap"),
        "passed": len(rows) == 1024 and len(set(exact)) == 1024 and len(orbit_counts) == 512 and all(n == 2 for n in orbit_counts.values()) and not overlaps and payload.get("swap") is False,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit, sort_keys=True))
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
