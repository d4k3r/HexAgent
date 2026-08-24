#!/usr/bin/env python3
"""Strictly read-only compact dashboard for autonomous-production-v1."""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path


def load(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unavailable"
    whole = int(seconds)
    hours, remainder = divmod(whole, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def finish(seconds: float | None) -> str:
    return "unavailable" if seconds is None else "~" + datetime.fromtimestamp(time.time() + seconds).strftime("%H:%M")


def selfplay(root: Path, target: int) -> list[str]:
    status = load(root / "runner-status.json", {}) or {}
    done = int(status.get("accepted", 0)); elapsed = float(status.get("elapsed_seconds", 0) or 0)
    rate = done / elapsed if done and elapsed else 0.0
    eta = (target - done) / rate if rate and done < target else (0.0 if done >= target else None)
    return [f"  {root.name}: {done:,}/{target:,} ({100 * done / target:.1f}%) state={status.get('state', 'not_started')}",
            f"    rate: {rate:.3f} games/s  ETA: {duration(eta)}  Finish: {finish(eta)}  quarantine: {status.get('quarantined', 0)}"]


def training(root: Path, candidates: list[str]) -> list[str]:
    lines = []
    for candidate in candidates:
        progress = load(root / candidate / "progress.json", {}) or {}
        done = int(progress.get("completed_base_rows", 0)); rate = float(progress.get("base_rows_per_second", 0) or 0)
        eta = (1_600_000 - done) / rate if rate and done < 1_600_000 else (0.0 if done >= 1_600_000 else None)
        if done or (root / candidate).exists():
            loss = (progress.get("latest_losses") or {}).get("total")
            lines.extend([f"  {candidate}: epoch {progress.get('epoch', 0)}/4  {done:,}/1,600,000 ({100 * done / 1_600_000:.1f}%)",
                          f"    steps: {progress.get('completed_optimizer_steps', 0):,}/25,000  rate: {rate:.1f} rows/s  loss: {loss}  ETA: {duration(eta)}  Finish: {finish(eta)}"])
    return lines or ["  candidates: not started"]


def match(root: Path, label: str, expected_pairs: int) -> list[str]:
    pairs = []
    malformed = 0
    for path in sorted((root / "pairs").glob("pair-*.json")) if root.is_dir() else []:
        record = load(path)
        if record and record.get("status") == "complete": pairs.append(record)
        else: malformed += 1
    games = 2 * len(pairs)
    wins = sum(int(pair.get("game_a", {}).get("candidate_score", 0)) + int(pair.get("game_b", {}).get("candidate_score", 0)) for pair in pairs)
    started = (load(root / "runner-status.json", {}) or {}).get("started_epoch")
    if not started and (root / "runner.stdout.log").is_file():
        match_start = re.search(r"stage8c runner start ([0-9.]+)", (root / "runner.stdout.log").read_text(errors="replace"))
        if match_start:
            started = float(match_start.group(1))
    elapsed = time.time() - float(started) if started else None
    rate = games / elapsed if elapsed and games else 0.0
    eta = (2 * expected_pairs - games) / rate if rate and games < 2 * expected_pairs else (0.0 if games >= 2 * expected_pairs else None)
    score = f"{wins / games:.4f}" if games else "n/a"
    return [f"  {label}: {len(pairs)}/{expected_pairs} pairs, {games}/{2 * expected_pairs} games; candidate score {score}",
            f"    rate: {rate:.3f} games/s  ETA: {duration(eta)}  Finish: {finish(eta)}  malformed: {malformed}"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(); root = args.root.resolve()
    state = load(root / "state.json", {}) or {}
    print("AUTONOMOUS RUN")
    print(f"Root: {root}")
    print(f"Generation: {state.get('generation', 1)} / 3   State/stage: {state.get('state', 'not_started')} / {state.get('stage', 'n/a')}")
    print(f"Current incumbent: {(state.get('incumbent') or {}).get('id', 'n/a')}")
    created = state.get("created_epoch"); print(f"Elapsed: {duration(time.time() - float(created)) if created else 'unavailable'}  Overall ETA: unavailable")
    gen = int(state.get("generation", 1)); gen_root = root / f"generation-{gen:04d}"
    stage = state.get("stage", "")
    if state.get("state") == "COMPLETE":
        final = load(root / "final-summary.json", {}) or {}
        print(f"Completed generations: {len(final.get('completed_generations', []))}/3")
        print(f"Final run-local incumbent: {(final.get('final_run_local_incumbent') or {}).get('id', 'n/a')}")
    elif stage.startswith("SELFPLAY"):
        print("SELF-PLAY")
        if stage == "SELFPLAY_NORMAL": print(*selfplay(gen_root / "selfplay" / "normal", 12_288), sep="\n")
        else:
            print(*selfplay(gen_root / "selfplay" / "normal", 12_288), sep="\n")
            print(*selfplay(gen_root / "selfplay" / "forced", 4_096), sep="\n")
    elif stage == "TRAIN_CANDIDATES":
        print("TRAIN CANDIDATES")
        manifest = load(gen_root / "generation-manifest.json", {}) or {}
        seeds = (manifest.get("candidate_plan") or {}).get("seeds", [])
        candidates = [f"C2-AUTO-G{gen:04d}-S{seed}" for seed in seeds]
        print(*training(gen_root / "training", candidates), sep="\n")
    elif stage == "CANDIDATE_SCREEN":
        print("CANDIDATE SCREEN")
        for child in sorted((gen_root / "screening").glob("*")) if (gen_root / "screening").is_dir() else []:
            print(*match(child, child.name, 32), sep="\n")
    elif stage == "PROMOTION_MATCH":
        print("PROMOTION MATCH")
        for child in sorted((gen_root / "promotion").glob("*")) if (gen_root / "promotion").is_dir() else []:
            print(*match(child, child.name, 200), sep="\n")
    else:
        print("Current stage has no live heavy-work progress file.")


if __name__ == "__main__":
    main()
