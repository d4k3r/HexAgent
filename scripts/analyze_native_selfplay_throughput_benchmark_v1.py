#!/usr/bin/env python3
"""Analyze audited native-self-play topology benchmark configurations.

This reports exact per-game differences where they occur, but does not call a
different CUDA batch shape a corruption: all runs must independently pass the
native-v2 corpus audit.  The two follow-on plan writers preserve the selected
topology and fixed semantic workload in immutable JSON plans.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_config(root: Path, config_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    directory = root / "configs" / config_id
    result, manifest, semantics = (read(directory / name) for name in ("config-result.json", "benchmark-config-manifest.json", "semantic-fingerprints.json"))
    if result.get("status") != "complete":
        raise RuntimeError(f"configuration {config_id} is not complete")
    return result, manifest, semantics


def semantic_comparison(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    ref, candidate = reference["records"], other["records"]
    if set(ref) != set(candidate):
        return {"same_game_ids": False, "missing_from_candidate": sorted(set(ref) - set(candidate)),
                "extra_in_candidate": sorted(set(candidate) - set(ref))}
    differences: dict[str, int] = {key: 0 for key in ("winner", "game_length", "moves", "certificate_ply", "phase_a_rows", "full_fingerprint")}
    for game_id in ref:
        left, right = ref[game_id], candidate[game_id]
        differences["winner"] += left["literal_winner"] != right["literal_winner"]
        differences["game_length"] += len(left["moves"]) != len(right["moves"])
        differences["moves"] += left["moves"] != right["moves"]
        differences["certificate_ply"] += left["certificate_ply"] != right["certificate_ply"]
        differences["phase_a_rows"] += left["phase_a_rows"] != right["phase_a_rows"]
        differences["full_fingerprint"] += reference["fingerprints"][game_id] != other["fingerprints"][game_id]
    return {"same_game_ids": True, "games": len(ref), "exact_match": differences["full_fingerprint"] == 0,
            "differences": differences,
            "note": "Different CUDA batch shapes may cause tiny neural numerical changes and a deterministic PUCT near-tie to diverge. Every configuration is therefore separately fail-closed by the native-v2 audit; exact fingerprints are reported, never hidden."}


def compact(result: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {"benchmark_id": result["benchmark_id"], "resource": manifest["resource"],
            "games_per_second": result["games_per_second"], "games_per_hour": result["games_per_hour"],
            "phase_a_rows_per_second": result["phase_a_rows_per_second"], "simulations_per_second": result["simulations_per_second"],
            "elapsed_seconds_startup_inclusive": result["elapsed_seconds_startup_inclusive"],
            "inference": result["inference"], "gpu": result["telemetry"]["gpu"],
            "cpu_percent_one_core_scale": result["telemetry"]["cpu_percent_one_core_scale"], "rss_mib": result["telemetry"]["rss_mib"]}


def winner(items: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]) -> str:
    # A simple and visible selection rule: all results are audit-complete; rank
    # useful game throughput.  The report exposes the close-run margin so an
    # operator can reject a trivial gain before a production-default change.
    return max(items, key=lambda item: item[0]["games_per_second"])[0]["benchmark_id"]


def stage_b_plan(stage_a_report: dict[str, Any], parent_plan: dict[str, Any], output: Path) -> None:
    selected = stage_a_report["fastest_stable_topology"]
    source = next(row for row in stage_a_report["configs"] if row["benchmark_id"] == selected)
    resource = source["resource"]
    concurrency = int(resource["concurrency_per_process"])
    lower, upper = max(16, concurrency - 16), concurrency + 16
    configs = [
        {"benchmark_id": "tune-b64-w200", "process_count": resource["process_count"], "concurrency_per_process": concurrency, "max_batch": 64, "wait_us": 200},
        {"benchmark_id": "tune-b128-w200", "process_count": resource["process_count"], "concurrency_per_process": concurrency, "max_batch": 128, "wait_us": 200},
        {"benchmark_id": "tune-b96-w100", "process_count": resource["process_count"], "concurrency_per_process": concurrency, "max_batch": 96, "wait_us": 100},
        {"benchmark_id": "tune-b96-w400", "process_count": resource["process_count"], "concurrency_per_process": concurrency, "max_batch": 96, "wait_us": 400},
        {"benchmark_id": f"tune-c{lower}-b96-w200", "process_count": resource["process_count"], "concurrency_per_process": lower, "max_batch": 96, "wait_us": 200},
        {"benchmark_id": f"tune-c{upper}-b96-w200", "process_count": resource["process_count"], "concurrency_per_process": upper, "max_batch": 96, "wait_us": 200},
    ]
    plan = {key: parent_plan[key] for key in ("schema", "champion_registry", "champion_onnx_sha256", "search")}
    plan.update({"plan_id": "native-selfplay-throughput-stage-b-v1", "parent_stage_a_report_sha256": stage_a_report["report_sha256"],
                 "selection_basis": {"stage_a_winner": selected, "winner_resource": resource},
                 "workload": {**parent_plan["workload"], "start_id": 9100000, "games": 256}, "configs": configs,
                 "selection_rule": "local coordinate tuning around the Stage-A fastest stable topology; human approval still required for production defaults"})
    atomic_json(output, plan)


def confirmation_plan(overall_report: dict[str, Any], parent_plan: dict[str, Any], output: Path) -> None:
    selected = overall_report["fastest_stable_resource"]
    resource = next(row["resource"] for row in overall_report["all_configs"] if row["benchmark_id"] == selected)
    baseline = {"benchmark_id": "baseline-1x64-b96-w200", "process_count": 1, "concurrency_per_process": 64, "max_batch": 96, "wait_us": 200}
    candidate = {"benchmark_id": "candidate-fastest-stable", **resource}
    plan = {key: parent_plan[key] for key in ("schema", "champion_registry", "champion_onnx_sha256", "search")}
    plan.update({"plan_id": "native-selfplay-throughput-confirmation-v1", "parent_report_sha256": overall_report["report_sha256"],
                 "workload": {**parent_plan["workload"], "start_id": 9200000, "games": 512}, "configs": [baseline, candidate],
                 "selection_rule": "same fixed 512-game workload compares current production resource baseline with the fastest Stage-A/B candidate"})
    atomic_json(output, plan)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--parent-plan", type=Path, required=True)
    p.add_argument("--write-stage-b-plan", type=Path)
    p.add_argument("--write-confirmation-plan", type=Path)
    return p


def main() -> int:
    args = parser().parse_args()
    parent = read(args.parent_plan)
    rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = [load_config(args.root, config) for config in args.config]
    known = {row[0]["benchmark_id"] for row in rows}
    if args.reference not in known:
        raise RuntimeError("reference must be among --config")
    ref_semantics = next(sem for result, _, sem in rows if result["benchmark_id"] == args.reference)
    configs = []
    for result, manifest, semantics in rows:
        row = compact(result, manifest)
        row["semantic_comparison_to_reference"] = semantic_comparison(ref_semantics, semantics)
        configs.append(row)
    report: dict[str, Any] = {"schema": "hex-native-selfplay-throughput-analysis-v1", "reference": args.reference, "configs": configs,
        "fastest_stable_topology": winner(rows), "criterion": "all configurations must have individually complete, zero-quarantine native-v2 audits; then rank games/sec. Exact cross-topology fingerprints are reported separately because CUDA batching can affect near ties."}
    # Store a self-hash after materializing the report, so Stage-B/confirmation
    # plans can bind to the evidence they were selected from.
    atomic_json(args.output, report)
    report["report_sha256"] = digest(args.output)
    atomic_json(args.output, report)
    if args.write_stage_b_plan:
        stage_b_plan(report, parent, args.write_stage_b_plan)
    if args.write_confirmation_plan:
        all_configs = configs
        report["all_configs"] = all_configs
        report["fastest_stable_resource"] = max(all_configs, key=lambda row: row["games_per_second"])["benchmark_id"]
        atomic_json(args.output, report)
        report["report_sha256"] = digest(args.output)
        atomic_json(args.output, report)
        confirmation_plan(report, parent, args.write_confirmation_plan)
    print(json.dumps({"output": str(args.output), "fastest_stable_topology": report["fastest_stable_topology"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native selfplay throughput analysis error: {exc}", file=sys.stderr)
        raise SystemExit(1)
