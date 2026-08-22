#!/usr/bin/env python3
"""Audit forced-prefix usage and prove that no forced move is a policy row."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from forced_prefix_bank_v1 import load_bank, sha256
from run_selfplay_v2_native import read_json, valid_game


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--allow-partial", action="store_true", help="bounded smoke: validate rows without requiring every bank prefix")
    args = ap.parse_args()
    root = args.root.resolve(); manifest = read_json(root / "run-manifest.json")
    bank_payload, bank_rows = load_bank(args.bank.resolve()); by_id = {r["prefix_id"]: r for r in bank_rows}
    if manifest.get("schema_version") != 3 or manifest.get("prefix", {}).get("mode") != "forced":
        raise RuntimeError("root is not a forced native-v2 schema-v3 run")
    if manifest["prefix"].get("bank_sha256") != sha256(args.bank.resolve()):
        raise RuntimeError("prefix bank hash mismatch")
    ids = range(manifest["game_ids"]["start"], manifest["game_ids"]["end_exclusive"])
    counts = Counter(); rows = []; invalid = []
    for gid in ids:
        path = root / "games" / f"game-{gid}.json"
        if not path.exists(): invalid.append({"game_id": gid, "reason": "missing"}); continue
        ok, reason = valid_game(path, manifest, gid)
        if not ok: invalid.append({"game_id": gid, "reason": reason}); continue
        game = read_json(path); pid = game.get("prefix_id"); counts[pid] += 1
        expected = by_id.get(pid)
        if expected is None or game.get("forced_prefix_actions") != expected["actions"] or game.get("forced_prefix_length") != 3:
            invalid.append({"game_id": gid, "reason": "prefix assignment mismatch"})
        if any(int(s.get("ply", -1)) < 3 for s in game.get("samples", [])):
            invalid.append({"game_id": gid, "reason": "forced move emitted as policy row"})
        rows.extend(game.get("samples", []))
    expected_each = manifest["game_ids"]["count"] // len(bank_rows)
    coverage = {pid: counts[pid] for pid in by_id}
    complete_coverage = len(counts) == len(bank_rows) and all(n == expected_each for n in coverage.values()) if expected_each else False
    audit = {
        "schema": "hex-forced-prefix-selfplay-audit-v1", "root": str(root),
        "run_manifest_sha256": hashlib.sha256((root / "run-manifest.json").read_bytes()).hexdigest(),
        "bank_sha256": sha256(args.bank.resolve()), "requested_games": manifest["game_ids"]["count"],
        "accepted_games": sum(counts.values()), "prefix_count": len(bank_rows),
        "prefix_coverage": coverage, "expected_uses_per_prefix": expected_each,
        "missing_prefix_ids": [pid for pid, n in coverage.items() if n == 0],
        "wrong_prefix_use_counts": {pid: n for pid, n in coverage.items() if n != expected_each},
        "phase_a_rows": len(rows), "forced_rows_emitted": sum(int(s.get("ply", 0)) < 3 for s in rows),
        "invalid": invalid, "coverage_complete": complete_coverage,
        "passed": not invalid and (complete_coverage or args.allow_partial),
    }
    output = args.output or root / "forced-prefix-audit.json"
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("requested_games", "accepted_games", "prefix_count", "phase_a_rows", "forced_rows_emitted", "passed")}, sort_keys=True))
    if not audit["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
