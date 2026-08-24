#!/usr/bin/env python3
"""Freeze one autonomous-generation CONTROL training bundle.

The only mutable inputs are the newly audited NORMAL/FORCED native corpora.
Teacher100 and historical salvage are immutable anchors.  No training occurs.
"""
from __future__ import annotations

import argparse, hashlib, json, os, random
from pathlib import Path


ROWS, EPOCHS, BATCH = 400_000, 4, 64
SCHEMA = "stage8b-prepared-fp32-autonomous-control-v1"
MIX_SCHEMA = "stage8b-autonomous-control-mixture-v1"
NAMESPACE = "stage8b-autonomous-control-exact-v1"
COUNTS = {"teacher": 80_000, "historical": 80_000, "normal": 180_000, "forced": 60_000}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def source_record(root: Path, child: str) -> dict:
    manifest_path = root / "prepared-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record = manifest.get("sources", {}).get(child)
    if not isinstance(record, dict) or not isinstance(record.get("rows"), int) or record["rows"] <= 0:
        raise ValueError(f"invalid prepared source {root}/{child}")
    arrays = {}
    for name, expected in record.get("array_sha256", {}).items():
        path = root / child / name
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"prepared source hash mismatch: {path}")
        arrays[name] = expected
    return {"rows": record["rows"], "path": str((root / child).resolve()), "array_sha256": arrays,
            "prepared_manifest": str(manifest_path.resolve()), "prepared_manifest_sha256": sha(manifest_path),
            "prepared_schema": manifest.get("schema")}


def schedule(seed: int, epoch: int) -> list[str]:
    values = [name for name in sorted(COUNTS) for _ in range(COUNTS[name])]
    random.Random(f"{NAMESPACE}:{seed}:{epoch}").shuffle(values)
    return values


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-prepared", type=Path, required=True)
    p.add_argument("--historical-prepared", type=Path, required=True)
    p.add_argument("--normal-prepared", type=Path, required=True)
    p.add_argument("--forced-prepared", type=Path, required=True)
    p.add_argument("--parent-checkpoint", type=Path, required=True)
    p.add_argument("--parent-onnx", type=Path, required=True)
    p.add_argument("--parent-id", required=True)
    p.add_argument("--candidate", action="append", required=True, metavar="ID:SEED")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve()
    if out.exists(): raise RuntimeError("refusing existing immutable autonomous prepared bundle")
    candidates = []
    for item in a.candidate:
        try: candidate, seed_text = item.rsplit(":", 1); seed = int(seed_text)
        except ValueError as exc: raise ValueError("--candidate must be ID:SEED") from exc
        if not candidate: raise ValueError("candidate ID must be nonempty")
        candidates.append((candidate, seed))
    if len({name for name, _ in candidates}) != len(candidates): raise ValueError("candidate IDs must be unique")
    checkpoint, onnx = a.parent_checkpoint.resolve(), a.parent_onnx.resolve()
    if not checkpoint.is_file() or not onnx.is_file(): raise ValueError("parent model artifact is missing")
    parent = {"id": a.parent_id, "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint), "onnx": str(onnx), "onnx_sha256": sha(onnx)}
    inputs = {"teacher": (a.teacher_prepared.resolve(), "teacher"), "historical": (a.historical_prepared.resolve(), "selfplay"), "normal": (a.normal_prepared.resolve(), "selfplay"), "forced": (a.forced_prepared.resolve(), "selfplay")}
    tmp = out.with_name(out.name + ".partial"); tmp.mkdir(parents=True)
    sources = {}
    for name, (root, child) in inputs.items():
        item = source_record(root, child)
        os.symlink(item["path"], tmp / name, target_is_directory=True)
        item["bundle_path"] = name; item["source_kind"] = child; sources[name] = item
    contracts = {}
    weights = {name: count / ROWS for name, count in COUNTS.items()}
    for candidate, seed in candidates:
        payload = {"schema": MIX_SCHEMA, "candidate_id": candidate, "parent_champion": parent,
                   "weights": weights, "per_epoch_base_rows": COUNTS,
                   "total_base_rows": {name: count * EPOCHS for name, count in COUNTS.items()},
                   "base_rows_per_epoch": ROWS, "epochs": EPOCHS, "base_batch": BATCH,
                   "optimizer_steps_per_epoch": ROWS // BATCH, "optimizer_steps_total": ROWS * EPOCHS // BATCH,
                   "seed": seed, "schedule_seed_namespace": NAMESPACE,
                   "sampling": "exact row-level shuffled CONTROL schedule; uniform-with-replacement within each immutable source; trainer-side colour transpose"}
        mix_path = tmp / "mixtures" / f"{candidate}.json"; atomic(mix_path, payload)
        schedule_hashes = {}
        for epoch in range(1, EPOCHS + 1):
            values = schedule(seed, epoch)
            record = {"schema": "stage8b-autonomous-control-source-schedule-v1", "candidate_id": candidate,
                      "epoch": epoch, "base_batch": BATCH, "base_rows": ROWS, "base_row_counts": COUNTS,
                      "schedule_seed_namespace": NAMESPACE, "schedule_sha256": hashlib.sha256("\n".join(values).encode()).hexdigest()}
            path = tmp / "source-schedules" / candidate / f"epoch-{epoch:02d}.json"; atomic(path, record); schedule_hashes[str(epoch)] = sha(path)
        contracts[candidate] = {"mixture_path": f"mixtures/{candidate}.json", "mixture_sha256": sha(mix_path), "epoch_schedule_sha256": schedule_hashes}
    manifest = {"schema": SCHEMA, "inputs_immutable": True, "parent": parent, "sources": sources,
                "semantics": "CONTROL only: exact root-visit soft policy, verified side-relative z, no Deep1600, no Phase-B rows, trainer-side colour transpose",
                "training_contract": {"base_rows_per_epoch": ROWS, "epochs": EPOCHS, "base_batch": BATCH, "optimizer_steps_per_epoch": ROWS // BATCH, "optimizer_steps_total": ROWS * EPOCHS // BATCH, "candidate_seeds": {name: seed for name, seed in candidates}},
                "training_contracts": contracts}
    atomic(tmp / "prepared-manifest.json", manifest)
    atomic(tmp / "prepared-audit-autonomous-control-v1.json", {"schema": "stage8b-autonomous-control-prepared-audit-v1", "passed": True, "prepared_manifest_sha256": sha(tmp / "prepared-manifest.json"), "source_rows": {k: v["rows"] for k, v in sources.items()}, "source_counts_per_epoch": COUNTS, "deep1600_rows": 0})
    os.replace(tmp, out)
    print(json.dumps({"output": str(out), "prepared_manifest_sha256": sha(out / "prepared-manifest.json"), "candidate_count": len(candidates), "source_counts_per_epoch": COUNTS}, sort_keys=True))


if __name__ == "__main__": main()
