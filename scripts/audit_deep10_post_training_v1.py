#!/usr/bin/env python3
"""Audit a completed DEEP10 run without loading a model or using CUDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


PARENT_SHA = "a4fdf9adac91468ff966e187a9423d519246288d9fbac470db863b8e5e430288"
EXPECTED = {"teacher": 71808, "deep1600": 8192, "historical": 80000, "normal": 180000, "forced": 60000}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return True


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepared = args.prepared_root.resolve()
    root = args.training_root.resolve()
    config = load(root / "config.json")
    report = load(root / "final-report.json")
    mixture_path = Path(config["mixture_manifest"]).resolve()
    mixture = load(mixture_path)
    prepared_manifest_path = prepared / "prepared-manifest.json"
    prepared_manifest = load(prepared_manifest_path)

    if config.get("candidate_id") != "C2-DEEP10-v1" or report.get("candidate_id") != "C2-DEEP10-v1":
        raise ValueError("DEEP10 candidate identity mismatch")
    if not report.get("passed") or len(report.get("epochs", [])) != 4:
        raise ValueError("DEEP10 training report is incomplete")
    if config.get("parent_champion", {}).get("sha256") != PARENT_SHA:
        raise ValueError("DEEP10 parent identity mismatch")
    if config.get("prepared_data_provenance", {}).get("manifest_sha256") != sha(prepared_manifest_path):
        raise ValueError("DEEP10 prepared bundle identity mismatch")
    if prepared_manifest.get("schema") != "stage8b-prepared-fp32-deep10-v1":
        raise ValueError("DEEP10 prepared schema mismatch")
    if mixture.get("schema") != "stage8b-deep10-mixture-v1" or mixture.get("candidate_id") != "C2-DEEP10-v1":
        raise ValueError("DEEP10 mixture schema mismatch")
    if mixture.get("per_epoch_base_rows") != EXPECTED or mixture.get("base_rows_per_epoch") != 400000 or mixture.get("epochs") != 4 or mixture.get("base_batch") != 64:
        raise ValueError("DEEP10 frozen recipe fields mismatch")
    if mixture.get("deep_repeats") != 2 or config.get("deep_repeats_per_epoch") != 2:
        raise ValueError("DEEP10 two-appearance contract is missing")

    epoch_audits = []
    checkpoint_hashes = {}
    for name, expected_hash in report.get("checkpoint_hashes", {}).items():
        path = root / "checkpoints" / name
        if not path.is_file() or sha(path) != expected_hash:
            raise ValueError(f"checkpoint hash mismatch: {name}")
        checkpoint_hashes[name] = expected_hash
    for epoch in range(1, 5):
        metric = next((item for item in report["epochs"] if item.get("epoch") == epoch), None)
        schedule = load(root / "source-schedules" / f"epoch-{epoch:02d}.json")
        prepared_schedule = load(prepared / "source-schedules" / "deep10" / f"epoch-{epoch:02d}.json")
        if metric is None or metric.get("train", {}).get("base_rows") != 400000:
            raise ValueError(f"epoch {epoch} is incomplete")
        if metric.get("source_base_rows") != EXPECTED or schedule.get("actual_base_rows") != EXPECTED:
            raise ValueError(f"epoch {epoch} source accounting mismatch")
        if schedule.get("row_source_schedule_sha256") != prepared_schedule.get("schedule_sha256"):
            raise ValueError(f"epoch {epoch} schedule identity mismatch")
        epoch_audits.append({"epoch": epoch, "base_rows": 400000, "source_base_rows": EXPECTED, "schedule_sha256": schedule["row_source_schedule_sha256"], "prepared_schedule_file_sha256": sha(prepared / "source-schedules" / "deep10" / f"epoch-{epoch:02d}.json")})

    coverage = load(prepared / "deep-coverage" / "deep10.json")
    if not coverage.get("passed") or len(coverage.get("epochs", [])) != 4:
        raise ValueError("DEEP10 coverage audit is incomplete")
    if any(item.get("deep_rows") != 8192 or item.get("deep_unique_ids") != 4096 or item.get("deep_appearance_count") != 2 or item.get("deep_ids_appearing_once") != 0 or item.get("deep_ids_appearing_gt2") != 0 or item.get("deep_missing_ids") != 0 or item.get("deep_duplicate_ids") != 0 for item in coverage["epochs"]):
        raise ValueError("DEEP10 Deep-ID coverage is not exactly twice per epoch")
    if "champion2-g1-search-v2" in json.dumps(prepared_manifest, sort_keys=True):
        raise ValueError("live C2-G1 source appears in DEEP10 provenance")
    if "best-validation-policy.pt" not in checkpoint_hashes:
        raise ValueError("selected best-validation-policy checkpoint is missing")
    if not finite(report):
        raise ValueError("non-finite DEEP10 training metric")

    result = {
        "schema": "deep10-post-training-audit-v1",
        "passed": True,
        "candidate_id": "C2-DEEP10-v1",
        "root": str(root),
        "config_sha256": sha(root / "config.json"),
        "final_report_sha256": sha(root / "final-report.json"),
        "mixture_manifest_sha256": sha(mixture_path),
        "prepared_manifest_sha256": sha(prepared_manifest_path),
        "parent_checkpoint_sha256": PARENT_SHA,
        "epochs": 4,
        "base_rows_total": 1_600_000,
        "optimizer_steps_total": 25_000,
        "source_counts_per_epoch": EXPECTED,
        "epoch_audits": epoch_audits,
        "deep_coverage": coverage,
        "selected_checkpoint": {"path": str((root / "checkpoints/best-validation-policy.pt").resolve()), "sha256": checkpoint_hashes["best-validation-policy.pt"], "epoch": report["best_validation_policy_epoch"], "best_validation_policy": report["best_validation_policy"]},
        "diagnostics": {"final_train": report["epochs"][-1]["train"], "best_validation_policy": report["best_validation_policy"], "best_validation_policy_epoch": report["best_validation_policy_epoch"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(args.output.name + ".partial")
    partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(partial, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
