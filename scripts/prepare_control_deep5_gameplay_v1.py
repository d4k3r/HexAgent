#!/usr/bin/env python3
"""Prepare, but never execute, the CONTROL/DEEP5 paired gameplay match."""
from __future__ import annotations

import argparse, hashlib, json, os
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--champion-checkpoint", type=Path, required=True)
    parser.add_argument("--openings", type=Path, required=True)
    parser.add_argument("--katahex-map", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", default="C2-DEEP5-v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        plan_only = output.is_dir() and (output / "match-plan.json").is_file() and not (output / "match-manifest.json").exists()
        if not plan_only:
            raise RuntimeError(f"refusing existing match root: {output}")
    candidate, champion, openings, katahex_map, runner = (p.resolve() for p in (args.candidate, args.champion, args.openings, args.katahex_map, args.runner))
    for path in (candidate, champion, args.candidate_checkpoint.resolve(), args.champion_checkpoint.resolve(), openings, katahex_map, runner):
        if not path.is_file():
            raise ValueError(f"missing match input: {path}")
    opening_payload = json.loads(openings.read_text())
    if opening_payload.get("schema") != "hex-puct-fpu-gameplay-openings-v1" or len(opening_payload.get("openings", [])) != 200:
        raise ValueError("evaluation bank must be the frozen 200-pair candidate-independent bank")
    candidate_sha, champion_sha = sha(candidate), sha(champion)
    candidate_checkpoint_sha, champion_checkpoint_sha = sha(args.candidate_checkpoint.resolve()), sha(args.champion_checkpoint.resolve())
    search = {"budget": 128, "c_puct": 2.5, "fpu_mode": "parent_value_reduced", "fpu_reduction": 0.25, "bridge_controller": "active", "literal_winner": "physical terminal authority"}
    execution = {"concurrency": 64, "max_batch": 96, "wait_us": 200, "bootstrap_samples": 20000}
    # This mirrors run_stage8c_gameplay_v1.py's persisted config exactly.
    match_schema = "deep10-control-gameplay-match-manifest-v1" if args.candidate_id == "C2-DEEP10-v1" else "control-deep5-gameplay-match-manifest-v1"
    config = {"schema": "stage8c-gameplay-config-v1", "candidate_id": args.candidate_id, "candidate": {"path": str(candidate), "sha256": candidate_sha}, "champion": {"path": str(champion), "sha256": champion_sha}, "openings": {"path": str(openings), "sha256": sha(openings), "max_pairs": None}, "katahex_map_sha256": sha(katahex_map), "search": {"budget": 128, "c_puct": 2.5, "concurrency": execution["concurrency"], "max_batch": execution["max_batch"], "wait_us": execution["wait_us"], "selection": "root_visit_argmax_lowest_action_tie"}, "bridge_controller": "active", "evaluation_mode": "promotion_protocol", "promotion": "one-sided 95% bootstrap lower pair mean > 0.5", "search_overrides": {"candidate_budget": 128, "champion_budget": 128, "candidate_c_puct": 2.5, "champion_c_puct": 2.5, "candidate_fpu_mode": "parent_value_reduced", "candidate_fpu_reduction": 0.25, "champion_fpu_mode": "parent_value_reduced", "champion_fpu_reduction": 0.25}}
    manifest = {"schema": match_schema, "status": "prepared_not_started", "candidate": {"id": args.candidate_id, "checkpoint": str(args.candidate_checkpoint.resolve()), "checkpoint_sha256": candidate_checkpoint_sha, "onnx": str(candidate), "onnx_sha256": candidate_sha}, "champion": {"id": "C2-CONTROL-v1", "checkpoint": str(args.champion_checkpoint.resolve()), "checkpoint_sha256": champion_checkpoint_sha, "onnx": str(champion), "onnx_sha256": champion_sha}, "evaluation_bank": {"path": str(openings), "sha256": sha(openings), "pairs": 200, "games": 400, "candidate_independent": True}, "katahex_map": {"path": str(katahex_map), "sha256": sha(katahex_map)}, "search": search, "execution": execution, "runner": {"path": str(runner), "sha256": sha(runner), "wrapper": "stage7_cuda12_runtime_v1.sh"}, "lifecycle": {"new": "prepared root with immutable config; first launch uses normal wrapper", "resume": "rerun the identical command; existing complete pair files are skipped by the C++ runner", "official_result": "summary.json emitted only after all 200 pairs complete"}, "decision": {"method": "qualified paired gameplay comparison", "bootstrap_samples": 20000, "promotion_metric": "one-sided 95% bootstrap lower pair mean > 0.5", "recipe_approval": "manual review required"}}
    output.mkdir(parents=True, exist_ok=True)
    atomic(output / "config.json", config)
    manifest["config_sha256"] = sha(output / "config.json")
    atomic(output / "match-manifest.json", manifest)
    print(json.dumps({"output": str(output), "status": manifest["status"], "config_sha256": manifest["config_sha256"], "candidate_onnx_sha256": candidate_sha, "champion_onnx_sha256": champion_sha, "opening_pairs": 200}, sort_keys=True))


if __name__ == "__main__":
    main()
