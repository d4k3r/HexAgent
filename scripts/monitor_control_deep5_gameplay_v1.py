#!/usr/bin/env python3
"""Read-only live monitor for a prepared 200-pair candidate-vs-CONTROL match."""
from __future__ import annotations

import argparse, json, re, time
from datetime import datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "match-manifest.json").read_text())
    pairs = []
    malformed = 0
    for path in sorted((root / "pairs").glob("pair-*.json")):
        try:
            record = json.loads(path.read_text())
            if record.get("schema") == "stage8c-pair-v1" and record.get("status") == "complete":
                pairs.append(record)
            else:
                malformed += 1
        except (OSError, json.JSONDecodeError):
            malformed += 1
    wins = sum(int(pair["game_a"]["candidate_score"]) + int(pair["game_b"]["candidate_score"]) for pair in pairs)
    games = 2 * len(pairs)
    candidate_black_wins = sum(int(pair["game_a"]["candidate_score"]) for pair in pairs)
    candidate_white_wins = sum(int(pair["game_b"]["candidate_score"]) for pair in pairs)
    started = None
    log = root / "runner.stdout.log"
    if log.exists():
        match = re.search(r"stage8c runner start ([0-9.]+)", log.read_text(errors="replace"))
        if match:
            started = float(match.group(1))
    elapsed = max(0.0, time.time() - started) if started else None
    pair_rate = len(pairs) / elapsed if elapsed and pairs else 0.0
    remaining = max(0, int(manifest["evaluation_bank"]["pairs"]) - len(pairs))
    eta_seconds = remaining / pair_rate if pair_rate else None
    status = {}
    status_path = root / "runner-status.json"
    if status_path.exists():
        try: status = json.loads(status_path.read_text())
        except json.JSONDecodeError: status = {"status_error": True}
    candidate_id = manifest.get("candidate", {}).get("id", "candidate")
    result = {"root": str(root), "candidate_id": candidate_id, "candidate_wins": wins, "candidate_label": candidate_id, "status": status.get("status", "not_started"), "completed_pairs": len(pairs), "expected_pairs": 200, "completed_games": games, "expected_games": 400, "percentage": 100.0 * games / 400.0, "control_wins": games - wins, "score": wins / games if games else None, "candidate_black_wins": candidate_black_wins, "candidate_white_wins": candidate_white_wins, "errors_or_malformed_pairs": malformed, "elapsed_seconds": elapsed, "games_per_hour": games / elapsed * 3600.0 if elapsed and games else None, "eta_seconds": eta_seconds, "estimated_finish_clock": datetime.fromtimestamp(time.time() + eta_seconds).isoformat(timespec="seconds") if eta_seconds is not None else None, "active_pairs": status.get("active_pairs"), "active_games": status.get("active_games"), "service_failure": status.get("reason") or status.get("candidate", {}).get("failure_reason") or status.get("champion", {}).get("failure_reason")}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
