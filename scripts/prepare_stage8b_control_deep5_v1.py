#!/usr/bin/env python3
"""Freeze the matched CONTROL/DEEP5 Stage-8B source bundle.

This converts completed Deep1600 policy results into the existing prepared
FP32 arrays.  ``teacher_root_utility`` is deliberately provenance only;
Student z remains the side-relative outcome already recorded for the source
state.  No training or model inference is performed.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, random
from pathlib import Path

import numpy as np

ROWS = 400_000
EPOCHS = 4
BATCH = 64
SCHEMA = "stage8b-prepared-fp32-control-deep5-v2"
MIX_SCHEMA = "stage8b-control-deep5-mixture-v2"
C2_CHECKPOINT_SHA = "a4fdf9adac91468ff966e187a9423d519246288d9fbac470db863b8e5e430288"
C2_ONNX_SHA = "def38b0f0d321f74b38ff113b5e9c630ac69569cffd0f85aaffb740c9a1736c5"

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()

def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def fingerprint_checkpoint(path: Path) -> dict:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("model_state") or payload.get("state_dict")
    if not isinstance(state, dict): raise ValueError("checkpoint has no model_state")
    h = hashlib.sha256(); tensors = 0; parameters = 0
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        raw = value.numpy().tobytes(order="C")
        descriptor = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype), "nbytes": len(raw)}
        h.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode()); h.update(b"\0"); h.update(raw)
        tensors += 1; parameters += value.numel()
    return {"sha256": sha(path), "tensor_fingerprint": h.hexdigest(), "tensors": tensors, "parameters": parameters, "architecture": payload.get("config", {}).get("architecture")}

def source_record(prepared: Path, child: str) -> dict:
    manifest_path = prepared / "prepared-manifest.json"; manifest = json.loads(manifest_path.read_text())
    record = manifest.get("sources", {}).get(child)
    if not record or not isinstance(record.get("rows"), int) or record["rows"] <= 0: raise ValueError(f"invalid prepared source {prepared}/{child}")
    arrays = {}
    for name, expected in record.get("array_sha256", {}).items():
        path = prepared / child / name
        if not path.is_file() or sha(path) != expected: raise ValueError(f"source array mismatch: {path}")
        arrays[name] = expected
    return {"rows": record["rows"], "path": str((prepared / child).resolve()), "array_sha256": arrays, "prepared_manifest": str(manifest_path.resolve()), "prepared_manifest_sha256": sha(manifest_path), "prepared_schema": manifest.get("schema"), "source_kind": child}

SCHEDULE_NAMESPACE = "stage8b-control-deep5-exact-v1"

def exact_schedule(weights: dict[str, float], seed: int, epoch: int) -> list[str]:
    if abs(sum(weights.values()) - 1.0) > 1e-9: raise ValueError("mixture weights must sum to one")
    counts = {key: int(round(value * ROWS)) for key, value in weights.items()}
    if sum(counts.values()) != ROWS: raise ValueError(f"mixture counts do not sum to {ROWS}: {counts}")
    # Canonical source ordering makes the immutable schedule independent of
    # JSON object ordering when the trainer reloads the manifest.
    schedule = [key for key in sorted(counts) for _ in range(counts[key])]
    random.Random(f"{SCHEDULE_NAMESPACE}:{seed}:{epoch}").shuffle(schedule)
    return schedule

def load_source_z(bank_row: dict, cache: dict[tuple[str, int], dict]) -> float:
    root = Path(bank_row["source_root"]); gid = int(bank_row["source_game_id"]); ply = int(bank_row["source_ply"]); key = (str(root), gid)
    if key not in cache:
        path = root / "games" / f"game-{gid}.json"
        if not path.is_file(): raise ValueError(f"source game missing: {path}")
        cache[key] = json.loads(path.read_text())
    samples = cache[key].get("samples", []); matches = [s for s in samples if int(s.get("ply", -1)) == ply]
    if len(matches) != 1: raise ValueError(f"source ply does not resolve uniquely: {key} ply {ply}")
    z = float(matches[0].get("z"));
    if z not in (-1.0, 1.0): raise ValueError(f"source z is not side-relative +/-1: {key} ply {ply}")
    return z

def build_deep_source(bank_path: Path, run_root: Path, out: Path, bank_sha: str, run_manifest_sha: str) -> dict:
    bank = json.loads(bank_path.read_text()); rows = bank.get("positions", [])
    if bank.get("schema") != "deep-teacher-1600-position-bank-v1" or len(rows) != 4096: raise ValueError("Deep bank is not the frozen 4096-position bank")
    result_manifest = json.loads((run_root / "manifest.json").read_text());
    if result_manifest.get("budget") != 1600 or result_manifest.get("bank_sha256") != bank_sha: raise ValueError("Deep1600 run does not bind the frozen bank")
    audit = json.loads((run_root / "postrun-audit.json").read_text())
    if not audit.get("complete") or audit.get("committed") != 4096 or audit.get("missing") or audit.get("duplicate_ids") or audit.get("errors"): raise ValueError("Deep1600 production audit is incomplete")
    result_files = {p.stem.replace("position-", ""): p for p in (run_root / "results").glob("position-*.json")}
    if len(result_files) != 4096 or set(result_files) != {r["position_id"] for r in rows}: raise ValueError("Deep1600 result IDs do not exactly match bank")
    state = np.lib.format.open_memmap(out / "state.npy", mode="w+", dtype=np.float32, shape=(4096, 6, 121))
    pi = np.lib.format.open_memmap(out / "pi.npy", mode="w+", dtype=np.float32, shape=(4096, 121))
    z = np.lib.format.open_memmap(out / "z.npy", mode="w+", dtype=np.float32, shape=(4096,))
    metadata = out / "metadata.jsonl"; cache = {}; ids = []; class_counts = {}; source_counts = {}
    with metadata.open("w", encoding="utf-8") as stream:
        for i, bank_row in enumerate(sorted(rows, key=lambda x: x["position_id"])):
            pid = bank_row["position_id"]; result = json.loads(result_files[pid].read_text());
            if result.get("schema") != "deep-teacher-katahex-result-v1" or result.get("bank_sha256") != bank_sha or result.get("requested_max_visits") != 1600: raise ValueError(f"invalid Deep1600 result {pid}")
            flat = bank_row.get("state_flat"); visits = result.get("raw_visits"); policy = result.get("pi")
            if not isinstance(flat, list) or len(flat) != 6 * 121 or not isinstance(policy, list) or len(policy) != 121 or not isinstance(visits, list) or len(visits) != 121: raise ValueError(f"invalid Deep1600 tensors {pid}")
            if any(not math.isfinite(float(x)) or float(x) < 0 for x in policy) or abs(sum(float(x) for x in policy) - 1.0) > 2e-5: raise ValueError(f"invalid Deep policy normalization {pid}")
            occupied = [bool(flat[i]) or bool(flat[121 + i]) for i in range(121)]
            if any(float(policy[i]) > 1e-8 and occupied[i] for i in range(121)): raise ValueError(f"Deep policy assigns an occupied action {pid}")
            state[i] = np.asarray(flat, dtype=np.float32).reshape(6, 121); pi[i] = np.asarray(policy, dtype=np.float32); z[i] = load_source_z(bank_row, cache)
            ids.append(pid); cls = bank_row["bank_class"]; src = bank_row["source"]; class_counts[cls] = class_counts.get(cls, 0) + 1; source_counts[src] = source_counts.get(src, 0) + 1
            stream.write(json.dumps({"row": i, "position_id": pid, "bank_class": cls, "source": src, "source_game_id": bank_row["source_game_id"], "source_ply": bank_row["source_ply"], "source_manifest_sha256": bank_row["source_manifest_sha256"], "teacher_budget": 1600, "teacher_root_utility_excluded": True, "student_value_target": "source_game_side_relative_z"}, sort_keys=True) + "\n")
    for array in (state, pi, z): array.flush()
    return {"rows": 4096, "array_sha256": {name: sha(out / name) for name in ("state.npy", "pi.npy", "z.npy")}, "metadata_sha256": sha(metadata), "position_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(), "bank_sha256": bank_sha, "run_manifest_sha256": run_manifest_sha, "class_counts": class_counts, "source_counts": source_counts, "value_semantics": "side-relative z from immutable source native Search-V2 sample; KataHex teacher_root_utility is not used"}

def mixture(candidate: str, weights: dict[str, float], parent: dict, seed: int) -> dict:
    counts = {key: int(round(value * ROWS)) for key, value in weights.items()}
    return {"schema": MIX_SCHEMA, "candidate_id": candidate, "parent_champion": parent, "weights": weights,
            "per_epoch_base_rows": counts, "total_base_rows": {k: v * EPOCHS for k, v in counts.items()},
            "base_rows_per_epoch": ROWS, "epochs": EPOCHS, "base_batch": BATCH,
            "optimizer_steps_per_epoch": ROWS // BATCH, "seed": seed, "schedule_seed_namespace": SCHEDULE_NAMESPACE,
            "sampling": "exact row-level shuffled schedule; uniform-with-replacement within each source except deep1600, which is a fixed permutation exactly once per epoch", "deep_coverage_contract": "deep1600 position IDs exactly once per epoch before trainer-side transpose"}

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--bank", type=Path, required=True); p.add_argument("--run-1600", type=Path, required=True); p.add_argument("--teacher-prepared", type=Path, required=True); p.add_argument("--historical-prepared", type=Path, required=True); p.add_argument("--normal-prepared", type=Path, required=True); p.add_argument("--forced-prepared", type=Path, required=True); p.add_argument("--parent-checkpoint", type=Path, required=True); p.add_argument("--parent-onnx", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--seed", type=int, default=4901); a = p.parse_args(); out = a.output.resolve()
    if out.exists(): raise RuntimeError(f"refusing existing immutable output: {out}")
    bank = a.bank.resolve(); run = a.run_1600.resolve(); bank_sha = sha(bank); run_manifest = run / "manifest.json"; run_manifest_sha = sha(run_manifest)
    if bank_sha != "77fdc0b3e86271e0143175dfe4eb28c5255a50caba1292113843a99ad1c9fb8f": raise ValueError("unexpected frozen Deep bank hash")
    parent = {"checkpoint": str(a.parent_checkpoint.resolve()), "checkpoint_sha256": sha(a.parent_checkpoint), "onnx": str(a.parent_onnx.resolve()), "onnx_sha256": sha(a.parent_onnx), "champion_id": "champion-2"}
    if parent["checkpoint_sha256"] != C2_CHECKPOINT_SHA or parent["onnx_sha256"] != C2_ONNX_SHA: raise ValueError("parent is not the frozen Champion-2 identity")
    tmp = out.with_name(out.name + ".partial"); tmp.mkdir(parents=True)
    sources = {"teacher": (a.teacher_prepared.resolve(), "teacher"), "historical": (a.historical_prepared.resolve(), "historical"), "normal": (a.normal_prepared.resolve(), "normal"), "forced": (a.forced_prepared.resolve(), "forced")}; source_records = {}
    for name, (root, child) in sources.items(): source_records[name] = source_record(root, child); os.symlink(source_records[name]["path"], tmp / name)
    deep_dir = tmp / "deep1600"; deep_dir.mkdir(); deep = build_deep_source(bank, run, deep_dir, bank_sha, run_manifest_sha)
    source_records["deep1600"] = {**deep, "path": str((out / "deep1600").resolve()), "source_kind": "deep_teacher1600", "bank_path": str(bank), "run_root": str(run)}
    parent_fp = fingerprint_checkpoint(a.parent_checkpoint); atomic_json(tmp / "parent-fingerprint.json", parent_fp)
    manifest = {"schema": SCHEMA, "semantics": "Teacher100 and Deep1600 supervise full 121-action normalized physical root visits; value target is side-relative z; deep teacher_root_utility is excluded; transpose remains trainer-side", "parent_champion": parent, "parent_tensor_fingerprint": parent_fp, "inputs_immutable": True, "deep_bank": {"path": str(bank), "sha256": bank_sha, "manifest_sha256": sha(bank.parent / "bank-manifest.json")}, "deep1600_run": {"path": str(run), "manifest_sha256": run_manifest_sha}, "sources": source_records}
    training_contracts = {}
    for candidate, weights in {"C2-CONTROL-v1": {"teacher": .20, "deep1600": 0.0, "historical": .20, "normal": .45, "forced": .15}, "C2-DEEP5-v1": {"teacher": .18976, "deep1600": .01024, "historical": .20, "normal": .45, "forced": .15}}.items():
        name = "control" if candidate == "C2-CONTROL-v1" else "deep5"
        payload = mixture(candidate, weights, parent, a.seed); coverage = []
        for epoch in range(1, EPOCHS + 1):
            schedule = exact_schedule(weights, a.seed, epoch); counts = {k: schedule.count(k) for k in sorted(weights)}; atomic_json(tmp / "source-schedules" / name / f"epoch-{epoch:02d}.json", {"schema": "stage8b-control-deep5-source-schedule-v2", "candidate_id": candidate, "epoch": epoch, "base_rows": ROWS, "base_row_counts": counts, "schedule_seed_namespace": SCHEDULE_NAMESPACE, "schedule_sha256": hashlib.sha256("\n".join(schedule).encode()).hexdigest()}); coverage.append({"epoch": epoch, "deep_rows": counts["deep1600"], "deep_unique_ids": 4096 if candidate == "C2-DEEP5-v1" else 0, "deep_duplicate_ids": 0, "deep_missing_ids": 0 if candidate == "C2-DEEP5-v1" else 4096, "deep_position_ids_sha256": deep["position_ids_sha256"] if candidate == "C2-DEEP5-v1" else None})
        atomic_json(tmp / "mixtures" / f"{name}.json", payload)
        atomic_json(tmp / "deep-coverage" / f"{name}.json", {"schema": "stage8b-deep1600-coverage-v1", "candidate_id": candidate, "epochs": coverage, "passed": all(x["deep_rows"] == 4096 and x["deep_unique_ids"] == 4096 and not x["deep_duplicate_ids"] and not x["deep_missing_ids"] for x in coverage) if candidate == "C2-DEEP5-v1" else True})
        training_contracts[name] = {
            "mixture_path": f"mixtures/{name}.json", "mixture_sha256": sha(tmp / "mixtures" / f"{name}.json"),
            "deep_coverage_path": f"deep-coverage/{name}.json", "deep_coverage_sha256": sha(tmp / "deep-coverage" / f"{name}.json"),
            "epoch_schedule_sha256": {str(epoch): sha(tmp / "source-schedules" / name / f"epoch-{epoch:02d}.json") for epoch in range(1, EPOCHS + 1)},
        }
    manifest["training_contracts"] = training_contracts
    atomic_json(tmp / "prepared-manifest.json", manifest)
    atomic_json(tmp / "prepared-audit-v6.json", {"schema": "stage8b-control-deep5-audit-v2", "passed": True, "prepared_manifest_sha256": sha(tmp / "prepared-manifest.json"), "source_rows": {k: v["rows"] for k, v in source_records.items()}, "deep1600_rows": deep["rows"], "deep_value_target": "source_game_side_relative_z", "teacher_root_utility_used": False})
    os.replace(tmp, out); print(json.dumps({"output": str(out), "manifest_sha256": sha(out / "prepared-manifest.json"), "parent_tensor_fingerprint": parent_fp["tensor_fingerprint"], "source_rows": {k: v["rows"] for k, v in source_records.items()}, "deep_coverage": deep["class_counts"] | deep["source_counts"]}, sort_keys=True))

if __name__ == "__main__": main()
