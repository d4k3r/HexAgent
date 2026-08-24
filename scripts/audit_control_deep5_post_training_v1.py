#!/usr/bin/env python3
"""Audit completed CONTROL/DEEP5 runs without loading a model or using CUDA."""
from __future__ import annotations

import argparse, hashlib, json, math, os
from pathlib import Path


EXPECTED = {
    "control": {"candidate_id": "C2-CONTROL-v1", "deep1600": 0, "teacher": 80000},
    "deep5": {"candidate_id": "C2-DEEP5-v1", "deep1600": 4096, "teacher": 75904},
}


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


def audit_one(name: str, root: Path, prepared: Path) -> dict:
    expected = EXPECTED[name]
    config = load(root / "config.json")
    report = load(root / "final-report.json")
    mixture_path = Path(config["mixture_manifest"])
    mixture = load(mixture_path)
    prepared_manifest_path = prepared / "prepared-manifest.json"
    prepared_manifest = load(prepared_manifest_path)
    if config["candidate_id"] != expected["candidate_id"] or report["candidate_id"] != expected["candidate_id"]:
        raise ValueError(f"{name}: candidate identity mismatch")
    if not report.get("passed") or len(report.get("epochs", [])) != 4:
        raise ValueError(f"{name}: terminal training report is incomplete")
    if config.get("parent_champion", {}).get("sha256") != "a4fdf9adac91468ff966e187a9423d519246288d9fbac470db863b8e5e430288":
        raise ValueError(f"{name}: parent checkpoint identity mismatch")
    if config.get("prepared_data_provenance", {}).get("manifest_sha256") != sha(prepared_manifest_path):
        raise ValueError(f"{name}: prepared bundle identity mismatch")
    if mixture.get("schema") != "stage8b-control-deep5-mixture-v2":
        raise ValueError(f"{name}: mixture schema mismatch")
    if mixture.get("candidate_id") != expected["candidate_id"] or mixture.get("base_rows_per_epoch") != 400000 or mixture.get("epochs") != 4 or mixture.get("base_batch") != 64:
        raise ValueError(f"{name}: frozen recipe fields mismatch")
    counts = mixture["per_epoch_base_rows"]
    expected_counts = {"teacher": expected["teacher"], "deep1600": expected["deep1600"], "historical": 80000, "normal": 180000, "forced": 60000}
    if counts != expected_counts or sum(counts.values()) != 400000:
        raise ValueError(f"{name}: source accounting mismatch")
    if config.get("mixture") != mixture.get("weights"):
        raise ValueError(f"{name}: config/mixture weights mismatch")
    epoch_rows = []
    actual_checkpoint_hashes = {}
    for checkpoint_name, expected_hash in report.get("checkpoint_hashes", {}).items():
        checkpoint_path = root / "checkpoints" / checkpoint_name
        if not checkpoint_path.is_file() or sha(checkpoint_path) != expected_hash:
            raise ValueError(f"{name}: checkpoint hash mismatch: {checkpoint_name}")
        actual_checkpoint_hashes[checkpoint_name] = expected_hash
    for epoch in range(1, 5):
        metric = next((item for item in report["epochs"] if item.get("epoch") == epoch), None)
        schedule = load(root / "source-schedules" / f"epoch-{epoch:02d}.json")
        prepared_schedule = load(prepared / "source-schedules" / name / f"epoch-{epoch:02d}.json")
        if metric is None:
            raise ValueError(f"{name}: epoch {epoch} accounting mismatch")
        observed_counts = {**expected_counts, **(metric.get("source_base_rows") or {})}
        observed_schedule_counts = {**expected_counts, **(schedule.get("actual_base_rows") or {})}
        if metric.get("train", {}).get("base_rows") != 400000 or observed_counts != expected_counts:
            raise ValueError(f"{name}: epoch {epoch} accounting mismatch")
        if observed_schedule_counts != expected_counts or schedule.get("row_source_schedule_sha256") != prepared_schedule.get("schedule_sha256"):
            raise ValueError(f"{name}: epoch {epoch} schedule identity mismatch")
        if schedule.get("mixed_batch_count") != 6250:
            raise ValueError(f"{name}: epoch {epoch} mixed-batch accounting mismatch")
        epoch_rows.append({"epoch": epoch, "base_rows": metric["train"]["base_rows"], "source_base_rows": metric["source_base_rows"], "schedule_sha256": schedule["row_source_schedule_sha256"], "prepared_schedule_file_sha256": sha(prepared / "source-schedules" / name / f"epoch-{epoch:02d}.json")})
    if sum(item["base_rows"] for item in epoch_rows) != 1600000 or not finite(report):
        raise ValueError(f"{name}: total rows or finite-metric guard failed")
    deep_coverage = load(prepared / "deep-coverage" / f"{name}.json")
    if name == "control" and any(item.get("deep_rows") != 0 for item in deep_coverage["epochs"]):
        raise ValueError("CONTROL: Deep1600 rows are nonzero")
    if name == "deep5" and (not deep_coverage.get("passed") or any(item.get("deep_rows") != 4096 or item.get("deep_unique_ids") != 4096 or item.get("deep_duplicate_ids") != 0 or item.get("deep_missing_ids") != 0 for item in deep_coverage["epochs"])):
        raise ValueError("DEEP5: frozen Deep1600 coverage contract failed")
    source_text = json.dumps(prepared_manifest, sort_keys=True)
    if "champion2-g1-search-v2" in source_text:
        raise ValueError(f"{name}: live C2-G1 source appears in prepared provenance")
    selected = "best-validation-policy.pt"
    if report.get("best_validation_policy_epoch") not in {2, 3} or selected not in actual_checkpoint_hashes:
        raise ValueError(f"{name}: selected best checkpoint is missing or invalid")
    return {"candidate_id": expected["candidate_id"], "root": str(root.resolve()), "config_sha256": sha(root / "config.json"), "final_report_sha256": sha(root / "final-report.json"), "mixture_manifest_sha256": sha(mixture_path), "prepared_manifest_sha256": sha(prepared_manifest_path), "parent_checkpoint_sha256": config["parent_champion"]["sha256"], "epochs": 4, "base_rows_total": 1600000, "optimizer_steps_total": 25000, "source_counts_per_epoch": expected_counts, "epoch_audits": epoch_rows, "deep_coverage": deep_coverage, "selected_checkpoint": {"path": str((root / "checkpoints" / selected).resolve()), "sha256": actual_checkpoint_hashes[selected], "epoch": report["best_validation_policy_epoch"], "best_validation_policy": report["best_validation_policy"]}, "diagnostics": {"final_train": report["epochs"][-1]["train"], "best_validation_policy": report["best_validation_policy"], "best_validation_policy_epoch": report["best_validation_policy_epoch"]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--deep5-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "control-deep5-post-training-audit-v1", "passed": True, "prepared_root": str(args.prepared_root.resolve()), "control": audit_one("control", args.control_root.resolve(), args.prepared_root.resolve()), "deep5": audit_one("deep5", args.deep5_root.resolve(), args.prepared_root.resolve()), "scientific_interpretation": "validation is diagnostic; gameplay decides CONTROL versus DEEP5"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
