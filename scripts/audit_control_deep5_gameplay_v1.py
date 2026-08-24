#!/usr/bin/env python3
"""Audit a completed candidate-vs-CONTROL paired match; never runs gameplay."""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    root = args.root.resolve(); manifest = json.loads((root / "match-manifest.json").read_text()); openings = json.loads(Path(manifest["evaluation_bank"]["path"]).read_text())["openings"]
    if manifest["evaluation_bank"]["sha256"] != sha(Path(manifest["evaluation_bank"]["path"])): raise ValueError("evaluation bank hash mismatch")
    expected = {str(item["pair_id"]): item for item in openings}; records = []; duplicate = []; malformed = []
    for path in sorted((root / "pairs").glob("pair-*.json")):
        try: record = json.loads(path.read_text())
        except json.JSONDecodeError: malformed.append(path.name); continue
        pid = str(record.get("pair_id"));
        if pid in {str(item.get("pair_id")) for item in records}: duplicate.append(pid)
        opening = expected.get(pid)
        if record.get("schema") != "stage8c-pair-v1" or record.get("status") != "complete" or opening is None or record.get("opening_moves") != opening["opening_moves"] or record.get("swap_decision") != opening["swap_decision"]: malformed.append(path.name); continue
        if any(record.get(key, {}).get("candidate_score") not in (0, 1) for key in ("game_a", "game_b")): malformed.append(path.name); continue
        records.append(record)
    wins = sum(int(record["game_a"]["candidate_score"]) + int(record["game_b"]["candidate_score"]) for record in records); games = 2 * len(records)
    summary_path = root / "summary.json"
    if len(records) != 200 or duplicate or malformed or not summary_path.is_file(): raise ValueError(f"incomplete or malformed match: pairs={len(records)} duplicate={duplicate} malformed={malformed} summary={summary_path.is_file()}")
    summary = json.loads(summary_path.read_text())
    if summary.get("games") != 400 or summary.get("pairs") != 200 or summary.get("wins") != wins or summary.get("losses") != games - wins: raise ValueError("summary does not agree with complete pair records")
    candidate_id = manifest.get("candidate", {}).get("id", "candidate")
    result = {"schema": "deep10-control-gameplay-audit-v1" if candidate_id == "C2-DEEP10-v1" else "candidate-control-gameplay-audit-v1", "complete": True, "candidate_id": candidate_id, "candidate_wins": wins, "pairs": len(records), "games": games, "control_wins": games - wins, "paired_score": wins / games, "summary_sha256": sha(summary_path), "match_manifest_sha256": sha(root / "match-manifest.json"), "promotion_qualified": bool(summary.get("promotion_qualified")), "promotion_decision": "manual review required; no recipe approval or champion promotion performed"}
    atomic(args.output.resolve(), result); print(json.dumps(result, sort_keys=True))


if __name__ == "__main__": main()
