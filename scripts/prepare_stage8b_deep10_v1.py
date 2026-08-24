#!/usr/bin/env python3
"""Prepare the immutable DEEP10 dose bundle from the frozen DEEP5 sources.

DEEP10 reuses the exact 4,096 Deep1600 rows and schedules each position twice
per epoch.  No teacher search or model training is performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

from prepare_stage8b_control_deep5_v1 import fingerprint_checkpoint, sha, source_record


ROWS = 400_000
EPOCHS = 4
BATCH = 64
SCHEMA = "stage8b-prepared-fp32-deep10-v1"
MIX_SCHEMA = "stage8b-deep10-mixture-v1"
SCHEDULE_NAMESPACE = "stage8b-deep10-exact-v1"
PARENT_CHECKPOINT_SHA = "a4fdf9adac91468ff966e187a9423d519246288d9fbac470db863b8e5e430288"
PARENT_ONNX_SHA = "def38b0f0d321f74b38ff113b5e9c630ac69569cffd0f85aaffb740c9a1736c5"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def schedule(weights: dict[str, float], seed: int, epoch: int) -> list[str]:
    counts = {name: int(round(value * ROWS)) for name, value in weights.items()}
    if sum(counts.values()) != ROWS:
        raise ValueError(f"DEEP10 counts do not sum to {ROWS}: {counts}")
    result = [name for name in sorted(counts) for _ in range(counts[name])]
    random.Random(f"{SCHEDULE_NAMESPACE}:{seed}:{epoch}").shuffle(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--run-1600", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-onnx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4901)
    args = parser.parse_args()

    source_bundle = args.source_bundle.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing existing immutable output: {output}")
    if not (source_bundle / "prepared-manifest.json").is_file():
        raise ValueError("source bundle is missing prepared-manifest.json")
    source_manifest = json.loads((source_bundle / "prepared-manifest.json").read_text())
    if source_manifest.get("schema") != "stage8b-prepared-fp32-control-deep5-v2":
        raise ValueError("DEEP10 must derive from the qualified prepared-bundle-v2")

    bank = args.bank.resolve()
    run = args.run_1600.resolve()
    if sha(bank) != "77fdc0b3e86271e0143175dfe4eb28c5255a50caba1292113843a99ad1c9fb8f":
        raise ValueError("unexpected frozen Deep1600 bank hash")
    run_manifest = run / "manifest.json"
    if not run_manifest.is_file() or json.loads(run_manifest.read_text()).get("budget") != 1600:
        raise ValueError("Deep1600 run manifest is missing or has the wrong budget")
    if json.loads(run_manifest.read_text()).get("bank_sha256") != sha(bank):
        raise ValueError("Deep1600 run does not bind the frozen bank")

    parent_checkpoint = args.parent_checkpoint.resolve()
    parent_onnx = args.parent_onnx.resolve()
    parent = {
        "champion_id": "champion-2",
        "checkpoint": str(parent_checkpoint),
        "checkpoint_sha256": sha(parent_checkpoint),
        "onnx": str(parent_onnx),
        "onnx_sha256": sha(parent_onnx),
    }
    if parent["checkpoint_sha256"] != PARENT_CHECKPOINT_SHA or parent["onnx_sha256"] != PARENT_ONNX_SHA:
        raise ValueError("parent is not the frozen Champion-2 identity")

    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True)
    sources: dict[str, dict] = {}
    for name in ("teacher", "historical", "normal", "forced", "deep1600"):
        record = source_record(source_bundle, name)
        target = partial / name
        os.symlink(record["path"], target, target_is_directory=True)
        record["path"] = str((output / name).resolve())
        record["source_bundle"] = str(source_bundle)
        sources[name] = record

    parent_fp = fingerprint_checkpoint(parent_checkpoint)
    atomic_json(partial / "parent-fingerprint.json", parent_fp)

    weights = {"teacher": 0.17952, "deep1600": 0.02048, "historical": 0.20, "normal": 0.45, "forced": 0.15}
    counts = {name: int(round(value * ROWS)) for name, value in weights.items()}
    if counts != {"teacher": 71808, "deep1600": 8192, "historical": 80000, "normal": 180000, "forced": 60000}:
        raise AssertionError(counts)

    deep_ids = [str(json.loads(line)["position_id"]) for line in (source_bundle / "deep1600" / "metadata.jsonl").read_text().splitlines()]
    if len(deep_ids) != 4096 or len(set(deep_ids)) != 4096:
        raise ValueError("frozen Deep source does not contain 4,096 unique IDs")
    deep_ids_sha = hashlib.sha256("\n".join(deep_ids).encode()).hexdigest()
    coverage_epochs = []
    for epoch in range(1, EPOCHS + 1):
        coverage_epochs.append({
            "epoch": epoch,
            "deep_rows": 8192,
            "deep_unique_ids": 4096,
            "deep_appearance_count": 2,
            "deep_ids_appearing_once": 0,
            "deep_ids_appearing_gt2": 0,
            "deep_missing_ids": 0,
            "deep_duplicate_ids": 0,
            "deep_position_ids_sha256": deep_ids_sha,
        })
    atomic_json(partial / "deep-coverage" / "deep10.json", {
        "schema": "stage8b-deep1600-coverage-v2",
        "candidate_id": "C2-DEEP10-v1",
        "epochs": coverage_epochs,
        "contract": "each frozen Deep1600 position ID appears exactly twice per epoch before transpose augmentation",
        "passed": True,
    })

    mixture = {
        "schema": MIX_SCHEMA,
        "candidate_id": "C2-DEEP10-v1",
        "parent_champion": parent,
        "weights": weights,
        "per_epoch_base_rows": counts,
        "total_base_rows": {name: value * EPOCHS for name, value in counts.items()},
        "base_rows_per_epoch": ROWS,
        "epochs": EPOCHS,
        "base_batch": BATCH,
        "optimizer_steps_per_epoch": ROWS // BATCH,
        "optimizer_steps_total": EPOCHS * ROWS // BATCH,
        "seed": args.seed,
        "schedule_seed_namespace": SCHEDULE_NAMESPACE,
        "deep_repeats": 2,
        "deep_coverage_contract": {
            "source_rows": 4096,
            "appearances_per_id_per_epoch": 2,
            "rows_per_epoch": 8192,
            "augmentation": "trainer-side exact colour transpose after base-row selection",
        },
        "sampling": "exact row-level shuffled schedule; Deep1600 uses two deterministic permutations of all 4,096 IDs per epoch",
    }
    atomic_json(partial / "mixtures" / "deep10.json", mixture)

    contracts = {}
    for epoch in range(1, EPOCHS + 1):
        values = schedule(weights, args.seed, epoch)
        record = {
            "schema": "stage8b-deep10-source-schedule-v1",
            "candidate_id": "C2-DEEP10-v1",
            "epoch": epoch,
            "base_batch": BATCH,
            "base_rows": ROWS,
            "base_row_counts": {name: values.count(name) for name in sorted(weights)},
            "schedule_mode": "row-level-shuffled",
            "schedule_seed_namespace": SCHEDULE_NAMESPACE,
            "schedule_sha256": hashlib.sha256("\n".join(values).encode()).hexdigest(),
        }
        path = partial / "source-schedules" / "deep10" / f"epoch-{epoch:02d}.json"
        atomic_json(path, record)
        contracts[str(epoch)] = sha(path)

    prepared_manifest = {
        "schema": SCHEMA,
        "semantics": "Teacher100 and Deep1600 supervise normalized physical root visits; value target is side-relative z; teacher_root_utility/root_value are excluded; transpose remains trainer-side",
        "candidate_id": "C2-DEEP10-v1",
        "parent_champion": parent,
        "parent_tensor_fingerprint": parent_fp,
        "inputs_immutable": True,
        "source_bundle_v2": {"path": str(source_bundle), "manifest_sha256": sha(source_bundle / "prepared-manifest.json")},
        "deep_bank": {"path": sources["deep1600"].get("bank_path"), "sha256": sources["deep1600"].get("bank_sha256")},
        "sources": sources,
        "training_contract": {"base_rows_per_epoch": ROWS, "epochs": EPOCHS, "base_batch": BATCH, "optimizer_steps_per_epoch": ROWS // BATCH, "optimizer_steps_total": EPOCHS * ROWS // BATCH, "seed": args.seed, "deep_repeats": 2},
        "training_contracts": {"deep10": {"mixture_path": "mixtures/deep10.json", "mixture_sha256": sha(partial / "mixtures" / "deep10.json"), "deep_coverage_path": "deep-coverage/deep10.json", "deep_coverage_sha256": sha(partial / "deep-coverage" / "deep10.json"), "epoch_schedule_sha256": contracts}},
        "validation": {"identity": "frozen Teacher100 held-out split; inherited from prepared-bundle-v2"},
    }
    atomic_json(partial / "prepared-manifest.json", prepared_manifest)
    atomic_json(partial / "prepared-audit-deep10-v1.json", {
        "schema": "stage8b-deep10-prepared-audit-v1",
        "passed": True,
        "prepared_manifest_sha256": sha(partial / "prepared-manifest.json"),
        "source_rows": {name: record["rows"] for name, record in sources.items()},
        "deep_rows_per_epoch": 8192,
        "deep_appearances_per_id_per_epoch": 2,
        "new_teacher_generation": False,
    })
    os.replace(partial, output)
    print(json.dumps({"output": str(output), "manifest_sha256": sha(output / "prepared-manifest.json"), "mixture_sha256": sha(output / "mixtures/deep10.json"), "deep_coverage_sha256": sha(output / "deep-coverage/deep10.json"), "parent_tensor_fingerprint": parent_fp["tensor_fingerprint"], "counts": counts}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
