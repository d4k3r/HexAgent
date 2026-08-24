#!/usr/bin/env python3
"""Deterministic, source-aware Stage-8B candidate trainer.

This is intentionally a bounded supervised policy-iteration pilot: it trains
one immutable candidate from champion-0 and never changes champion lineage.
"""
from __future__ import annotations

import argparse, bisect, collections, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from hex_reconstruction.schema import read_jsonl
from hex_reconstruction.stage8_data import SourceMixture, deterministic_game_split, iter_stage7_examples
from hex_reconstruction.student_training import Group49Student, atomic_json, soft_policy_loss, weighted_mse
from hex_reconstruction.symmetry import transformed_training_tensors


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "artifacts/student-training-value-symmetry-v1/expanded-seed4901-symmetry-24epochs-final/checkpoints/best-validation-policy.pt"
ASSEMBLY = ROOT / "artifacts/student-training-expanded-v1/dataset-assembly-v1.json"
SELFPLAY = ROOT / "artifacts/cpp-selfplay-stage7-v1/corpus-128-c128-b96-4h-v1/data"
BASE_BATCH = 64  # original half; exact paired training batch is 128 effective examples


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_complete_prepared_root(root: Path) -> dict:
    """Fail closed: production data must be the audited immutable mmap corpus."""
    root = root.resolve()
    manifest_path = root / "prepared-manifest.json"
    if not manifest_path.is_file(): raise ValueError("prepared root is missing its manifest")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") not in {"stage8b-prepared-fp32-v1", "stage8b-prepared-fp32-clean-selfplay-v2", "stage8b-prepared-fp32-native-selfplay-v2", "stage8b-prepared-fp32-diversity-v1", "stage8b-prepared-fp32-control-deep5-v1", "stage8b-prepared-fp32-control-deep5-v2", "stage8b-prepared-fp32-deep10-v1", "stage8b-prepared-fp32-autonomous-control-v1"} or manifest.get("limited_fixture_rows_per_source") is not None:
        raise ValueError("production prepared root must be complete")
    audit_name = {"stage8b-prepared-fp32-clean-selfplay-v2": "prepared-audit-v2.json", "stage8b-prepared-fp32-native-selfplay-v2": "prepared-audit-v3.json", "stage8b-prepared-fp32-diversity-v1": "prepared-audit-v4.json", "stage8b-prepared-fp32-control-deep5-v1": "prepared-audit-v5.json", "stage8b-prepared-fp32-control-deep5-v2": "prepared-audit-v6.json", "stage8b-prepared-fp32-deep10-v1": "prepared-audit-deep10-v1.json", "stage8b-prepared-fp32-autonomous-control-v1": "prepared-audit-autonomous-control-v1.json"}.get(manifest["schema"], "prepared-audit-v1.json")
    audit_path = root / audit_name
    if not audit_path.is_file(): raise ValueError("prepared root is missing its matching audit")
    audit = json.loads(audit_path.read_text())
    if not audit.get("passed") or audit.get("prepared_manifest_sha256") != digest(manifest_path):
        raise ValueError("prepared root audit does not bind this manifest")
    sources = tuple(manifest.get("sources", {}).keys()) if manifest.get("schema") in {"stage8b-prepared-fp32-diversity-v1", "stage8b-prepared-fp32-control-deep5-v1", "stage8b-prepared-fp32-control-deep5-v2", "stage8b-prepared-fp32-deep10-v1", "stage8b-prepared-fp32-autonomous-control-v1"} else ("teacher", "selfplay")
    for source in sources:
        record = manifest.get("sources", {}).get(source, {})
        if not isinstance(record.get("rows"), int) or record["rows"] <= 0:
            raise ValueError(f"prepared {source} row count is invalid")
        for filename, expected in record.get("array_sha256", {}).items():
            path = root / source / filename
            if not path.is_file() or digest(path) != expected:
                raise ValueError(f"prepared {source} array provenance mismatch: {filename}")
        if source == "deep1600" and record["rows"] != 4096:
            raise ValueError("Deep1600 source must contain exactly the frozen 4096 positions")
    return {"root": str(root), "manifest_path": str(manifest_path), "manifest_sha256": digest(manifest_path),
            "audit_path": str(audit_path), "audit_sha256": digest(audit_path),
            "source_rows": {source: manifest["sources"][source]["rows"] for source in sources}}


def atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary); os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class RowIndex:
    """Uniform-with-replacement row sampler over immutable game files."""
    def __init__(self, entries: list[tuple[Path, int]], kind: str):
        self.entries = [(path, count) for path, count in entries if count > 0]
        self.kind = kind
        self.ends: list[int] = []
        total = 0
        for _, count in self.entries:
            total += count; self.ends.append(total)
        if not total: raise ValueError(f"empty {kind} training source")
        self.total = total
        self.cache: collections.OrderedDict[Path, list] = collections.OrderedDict()

    def descriptor(self, rng: random.Random) -> tuple[Path, int]:
        absolute = rng.randrange(self.total); index = bisect.bisect_right(self.ends, absolute)
        start = self.ends[index - 1] if index else 0
        return self.entries[index][0], absolute - start

    def _rows(self, path: Path) -> list:
        cached = self.cache.get(path)
        if cached is not None:
            self.cache.move_to_end(path); return cached
        if self.kind == "teacher":
            loaded = [example for example in read_jsonl(path) if example.source == "katahex_teacher" and example.position_status == "normal"]
        else:
            loaded = list(iter_stage7_examples(path))
        self.cache[path] = loaded
        if len(self.cache) > 16: self.cache.popitem(last=False)
        return loaded

    def sample(self, rng: random.Random):
        path, offset = self.descriptor(rng)
        return self._rows(path)[offset]


class PreparedRowIndex:
    """Read-only mmap source with the exact old RowIndex absolute row order."""
    def __init__(self, root: Path, kind: str):
        self.kind = kind
        self.state = np.load(root / kind / "state.npy", mmap_mode="r")
        self.pi = np.load(root / kind / "pi.npy", mmap_mode="r")
        self.z = np.load(root / kind / "z.npy", mmap_mode="r")
        self.total = len(self.z)
        if self.total <= 0 or self.state.shape != (self.total, 6, 121) or self.pi.shape != (self.total, 121):
            raise ValueError(f"invalid prepared {kind} tensors")

    def sample_batch(self, rng: random.Random, count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Same random absolute-row draw as RowIndex.descriptor/sample.
        indices = np.asarray([rng.randrange(self.total) for _ in range(count)], dtype=np.int64)
        return np.array(self.state[indices], dtype=np.float32, copy=True), np.array(self.pi[indices], dtype=np.float32, copy=True), np.array(self.z[indices], dtype=np.float32, copy=True)

    def sample_index(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if index < 0 or index >= self.total: raise IndexError(index)
        return (np.array(self.state[index:index + 1], dtype=np.float32, copy=True),
                np.array(self.pi[index:index + 1], dtype=np.float32, copy=True),
                np.array(self.z[index:index + 1], dtype=np.float32, copy=True))


def teacher_entries(assembly_path: Path, split_name: str) -> list[tuple[Path, int]]:
    assembly = json.loads(assembly_path.read_text())
    roots = [Path(item["path"]) for item in assembly["corpora"]]
    result = []
    for game_id, split in assembly["assignments"].items():
        if split != split_name: continue
        matches = [root / "games" / game_id / "examples.jsonl" for root in roots if (root / "games" / game_id / "examples.jsonl").is_file()]
        if len(matches) != 1: raise ValueError(f"teacher game {game_id} resolves to {len(matches)} paths")
        result.append((matches[0], sum(example.source == "katahex_teacher" and example.position_status == "normal" for example in read_jsonl(matches[0]))))
    return result


def selfplay_entries(root: Path, split_name: str) -> list[tuple[Path, int]]:
    paths = sorted((root / "games").glob("game-*.json"))
    ids = [f"stage7-c0-{json.loads(path.read_text())['game_id']}" for path in paths]
    split = deterministic_game_split(ids, corpus_id="stage7-champion-0-v1")
    return [(path, int(json.loads(path.read_text())["game_length"])) for path, identifier in zip(paths, ids) if split[identifier] == split_name]


def tensors(examples, device: torch.device):
    state = []; policy = []; z = []
    for example in examples:
        state.append(example.state.planes); policy.append(example.policy.pi); z.append(example.value.z)
    for example in examples:
        s, p, _legal, value = transformed_training_tensors(example)
        state.append(s); policy.append(p); z.append(value)
    return (torch.tensor(state, dtype=torch.float32, device=device).reshape(-1, 6, 11, 11),
            torch.tensor(policy, dtype=torch.float32, device=device), torch.tensor(z, dtype=torch.float32, device=device))


TRANSPOSE_INDEX = torch.tensor([((index % 11) * 11 + index // 11) for index in range(121)], dtype=torch.long)


def prepared_tensors(state: np.ndarray, policy: np.ndarray, value: np.ndarray, device: torch.device):
    """Vectorized exact materialized colour-transpose, original half first."""
    original_state = torch.from_numpy(state)
    original_policy = torch.from_numpy(policy)
    original_value = torch.from_numpy(value)
    perm = TRANSPOSE_INDEX
    transposed = original_state.index_select(2, perm)
    transformed_state = torch.stack((transposed[:, 1], transposed[:, 0], 1.0 - transposed[:, 2], transposed[:, 3], transposed[:, 4], transposed[:, 5]), dim=1)
    transformed_policy = original_policy.index_select(1, perm)
    full_state = torch.cat((original_state, transformed_state), dim=0).reshape(-1, 6, 11, 11)
    full_policy = torch.cat((original_policy, transformed_policy), dim=0)
    full_value = torch.cat((original_value, original_value), dim=0)
    return full_state.to(device, non_blocking=True), full_policy.to(device, non_blocking=True), full_value.to(device, non_blocking=True)


def validation_rows(limit: int | None = None) -> list:
    index = RowIndex(teacher_entries(ASSEMBLY, "validation"), "teacher")
    # Materialization is bounded (20,082) and gives fixed best-policy selection.
    result = []
    for path, _count in index.entries:
        result.extend(index._rows(path))
        if limit is not None and len(result) >= limit:
            return result[:limit]
    return result


def exact_batch_schedule(weights: dict[str, float], *, seed: int, epoch: int, batches: int, namespace: str = "stage8b-diversity-exact-v1") -> list[str]:
    """Exact source proportions for the native diversity experiment.

    The historical two-source trainer keeps its original random-choice
    schedule. Native BASE/DIVERSE manifests opt into this exact, shuffled
    *row* schedule so a 400,000-row epoch has auditable source counts (in
    particular, 60,000 FORCED rows, which cannot be represented by whole
    64-row source-homogeneous batches).
    """
    if abs(sum(weights.values()) - 1.0) > 1e-9:
        raise ValueError("exact mixture weights must sum to one")
    total_rows = batches * BASE_BATCH
    counts = {name: int(round(weight * total_rows)) for name, weight in weights.items()}
    if sum(counts.values()) != total_rows:
        raise ValueError(f"mixture weights do not produce an exact batch schedule: {counts}")
    # Match the producer's canonical ordering before the deterministic shuffle.
    schedule = [name for name in sorted(counts) for _ in range(counts[name])]
    random.Random(f"{namespace}:{seed}:{epoch}").shuffle(schedule)
    return schedule


def required_field(payload: dict, name: str):
    if name not in payload or payload[name] is None:
        raise ValueError(f"mixture manifest missing required field: {name}")
    return payload[name]


def validate_exact_mixture_manifest(payload: dict, *, args: argparse.Namespace, parent_sha: str, prepared_root: Path | None) -> dict:
    """Validate the versioned exact-mixture contract before model/GPU setup."""
    schema = required_field(payload, "schema")
    supported = {"stage8b-diversity-mixture-v1", "stage8b-control-deep5-mixture-v2", "stage8b-deep10-mixture-v1", "stage8b-autonomous-control-mixture-v1"}
    if schema not in supported:
        if schema == "stage8b-control-deep5-mixture-v1":
            raise ValueError("CONTROL/DEEP5 mixture-v1 is obsolete; regenerate/use prepared-bundle-v2 with mixture-v2")
        raise ValueError(f"unsupported mixture manifest schema: {schema}")
    base_rows = required_field(payload, "base_rows_per_epoch")
    epochs = required_field(payload, "epochs")
    base_batch = required_field(payload, "base_batch")
    try:
        base_rows, epochs, base_batch = int(base_rows), int(epochs), int(base_batch)
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture manifest frozen training fields must be integers") from exc
    if base_rows != args.base_rows_per_epoch or epochs != args.epochs or base_batch != BASE_BATCH:
        raise ValueError("mixture manifest does not match frozen training recipe")
    weights_raw, counts_raw = required_field(payload, "weights"), required_field(payload, "per_epoch_base_rows")
    if not isinstance(weights_raw, dict) or not isinstance(counts_raw, dict):
        raise ValueError("mixture manifest weights/per_epoch_base_rows must be objects")
    if set(weights_raw) != set(counts_raw):
        raise ValueError("mixture manifest source keys differ between weights and per_epoch_base_rows")
    try:
        counts = {str(name): int(value) for name, value in counts_raw.items()}
        weights = {str(name): float(value) for name, value in weights_raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture manifest source counts/weights are malformed") from exc
    if any(value < 0 for value in counts.values()) or sum(counts.values()) != base_rows:
        raise ValueError(f"mixture manifest per_epoch_base_rows must sum to {base_rows}: {counts}")
    if abs(sum(weights.values()) - 1.0) > 1e-9 or any(abs(weights[name] - counts[name] / base_rows) > 1e-9 for name in weights):
        raise ValueError("mixture manifest weights do not exactly match per_epoch_base_rows")
    total_raw = required_field(payload, "total_base_rows")
    if not isinstance(total_raw, dict):
        raise ValueError("mixture manifest total_base_rows must be an object")
    try:
        total_counts = {str(k): int(v) for k, v in total_raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture manifest total_base_rows is malformed") from exc
    if total_counts != {name: count * epochs for name, count in counts.items()}:
        raise ValueError("mixture manifest total_base_rows does not match epoch contract")
    try:
        optimizer_steps = int(required_field(payload, "optimizer_steps_per_epoch"))
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture manifest optimizer_steps_per_epoch must be an integer") from exc
    if optimizer_steps != base_rows // base_batch:
        raise ValueError("mixture manifest optimizer_steps_per_epoch does not match base recipe")
    try:
        seed = int(required_field(payload, "seed"))
    except (TypeError, ValueError) as exc:
        raise ValueError("mixture manifest seed must be an integer") from exc
    if seed != args.seed:
        raise ValueError("mixture manifest seed does not match requested training seed")
    parent = required_field(payload, "parent_champion")
    if not isinstance(parent, dict) or parent.get("checkpoint_sha256") != parent_sha:
        raise ValueError("mixture manifest parent checkpoint identity does not match --parent-checkpoint")
    namespace = "stage8b-diversity-exact-v1"
    if schema in {"stage8b-control-deep5-mixture-v2", "stage8b-deep10-mixture-v1", "stage8b-autonomous-control-mixture-v1"}:
        if required_field(payload, "candidate_id") != args.candidate_id:
            raise ValueError("mixture manifest candidate_id does not match --candidate-id")
        namespace = str(required_field(payload, "schedule_seed_namespace"))
        if schema == "stage8b-control-deep5-mixture-v2":
            if namespace != "stage8b-control-deep5-exact-v1":
                raise ValueError("unsupported CONTROL/DEEP5 schedule_seed_namespace")
            expected = ({"teacher": 80000, "deep1600": 0, "historical": 80000, "normal": 180000, "forced": 60000}
                        if args.candidate_id == "C2-CONTROL-v1" else
                        {"teacher": 75904, "deep1600": 4096, "historical": 80000, "normal": 180000, "forced": 60000}
                        if args.candidate_id == "C2-DEEP5-v1" else None)
            if expected is None or counts != expected:
                raise ValueError("CONTROL/DEEP5 mixture source counts do not match the frozen causal contract")
        elif schema == "stage8b-deep10-mixture-v1":
            if namespace != "stage8b-deep10-exact-v1":
                raise ValueError("unsupported DEEP10 schedule_seed_namespace")
            expected = {"teacher": 71808, "deep1600": 8192, "historical": 80000, "normal": 180000, "forced": 60000}
            if args.candidate_id != "C2-DEEP10-v1" or counts != expected:
                raise ValueError("DEEP10 mixture source counts do not match the frozen dose contract")
        else:
            if namespace != "stage8b-autonomous-control-exact-v1":
                raise ValueError("unsupported autonomous CONTROL schedule_seed_namespace")
            expected = {"teacher": 80000, "historical": 80000, "normal": 180000, "forced": 60000}
            if counts != expected:
                raise ValueError("autonomous CONTROL source counts do not match the frozen recipe")
    if prepared_root is None:
        raise ValueError("exact mixture manifests require --prepared-root")
    provenance = validate_complete_prepared_root(prepared_root)
    prepared_manifest = json.loads((prepared_root / "prepared-manifest.json").read_text())
    prepared_parent = prepared_manifest.get("parent_champion") or prepared_manifest.get("parent") or {}
    if prepared_parent.get("checkpoint_sha256") != parent_sha:
        raise ValueError("prepared root parent Champion identity does not match --parent-checkpoint")
    for source, count in counts.items():
        if count and source not in provenance["source_rows"]:
            raise ValueError(f"prepared root lacks active mixture source: {source}")
    if schema in {"stage8b-control-deep5-mixture-v2", "stage8b-deep10-mixture-v1", "stage8b-autonomous-control-mixture-v1"}:
        name = ("control" if args.candidate_id == "C2-CONTROL-v1" else
                "deep5" if args.candidate_id == "C2-DEEP5-v1" else
                "deep10" if args.candidate_id == "C2-DEEP10-v1" else args.candidate_id)
        schedule = exact_batch_schedule(weights, seed=seed, epoch=1, batches=base_rows // base_batch, namespace=namespace)
        schedule_path = prepared_root / "source-schedules" / name / "epoch-01.json"
        if not schedule_path.is_file() or json.loads(schedule_path.read_text()).get("schedule_sha256") != hashlib.sha256("\n".join(schedule).encode()).hexdigest():
            raise ValueError("prepared source schedule does not bind the trainer schedule")
        contracts = prepared_manifest.get("training_contracts", {})
        contract = contracts.get(name) if isinstance(contracts, dict) else None
        if not isinstance(contract, dict):
            raise ValueError("prepared root is missing the immutable CONTROL/DEEP5 training contract")
        if digest(args.mixture_manifest) != contract.get("mixture_sha256"):
            raise ValueError("mixture manifest does not match the prepared-root immutable training contract")
        if digest(schedule_path) != contract.get("epoch_schedule_sha256", {}).get("1"):
            raise ValueError("prepared source schedule is not bound by the immutable training contract")
        if schema == "stage8b-deep10-mixture-v1":
            coverage_path = prepared_root / "deep-coverage" / "deep10.json"
            if not coverage_path.is_file():
                raise ValueError("prepared root is missing the DEEP10 coverage contract")
            coverage = json.loads(coverage_path.read_text())
            if coverage.get("schema") != "stage8b-deep1600-coverage-v2" or not coverage.get("passed"):
                raise ValueError("DEEP10 coverage contract is not qualified")
            if len(coverage.get("epochs", [])) != epochs:
                raise ValueError("DEEP10 coverage contract does not contain every epoch")
            for item in coverage.get("epochs", []):
                if (item.get("deep_rows") != 8192 or item.get("deep_unique_ids") != 4096 or
                        item.get("deep_appearance_count") != 2 or item.get("deep_ids_appearing_once") != 0 or
                        item.get("deep_ids_appearing_gt2") != 0 or item.get("deep_missing_ids") != 0 or
                        item.get("deep_duplicate_ids") != 0):
                    raise ValueError("DEEP10 coverage is not exactly two appearances per Deep ID")
    return {"schema": schema, "weights": weights, "counts": counts, "seed": seed, "schedule_seed_namespace": namespace,
            "prepared_provenance": provenance, "parent": parent}


def evaluate_policy(model, examples, device: torch.device) -> dict:
    total = 0.0; count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(examples), 128):
            state = torch.tensor([x.state.planes for x in examples[start:start + 128]], dtype=torch.float32, device=device).reshape(-1, 6, 11, 11)
            policy = torch.tensor([x.policy.pi for x in examples[start:start + 128]], dtype=torch.float32, device=device)
            logits, _ = model(state); total += float((-(policy * torch.log_softmax(logits, 1)).sum(1)).sum()); count += len(state)
    return {"soft_policy_ce": total / count, "samples": count}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True); p.add_argument("--candidate-id", required=True)
    p.add_argument("--teacher-weight", type=float, default=None); p.add_argument("--selfplay-weight", type=float, default=None)
    p.add_argument("--epochs", type=int, default=4); p.add_argument("--base-rows-per-epoch", type=int, default=400000)
    p.add_argument("--seed", type=int, default=4901); p.add_argument("--device", default="cuda")
    p.add_argument("--prepared-root", type=Path, help="complete immutable stage8b-prepared-fp32-v1 corpus")
    p.add_argument("--mixture-manifest", type=Path, help="native BASE/DIVERSE exact four-source mixture manifest")
    p.add_argument("--parent-checkpoint", type=Path, help="frozen parent checkpoint; defaults to historical Champion-0 for legacy calls")
    p.add_argument("--allow-limited-prepared-fixture", action="store_true", help="test-only; never production")
    p.add_argument("--progress-interval", type=int, default=250)
    p.add_argument("--smoke-validation-rows", type=int,
                   help="bounded preflight only; canonical candidates omit this option")
    p.add_argument("--preflight", action="store_true",
                   help="validate immutable inputs and epoch-plan identity without creating output or loading a model")
    p.add_argument("--resume", action="store_true",
                   help="resume only from the last atomically committed epoch checkpoint")
    args = p.parse_args(); output = args.output.resolve()
    if output.exists() and not args.resume: raise RuntimeError(f"refusing existing candidate root: {output}")
    if args.resume and not output.exists(): raise RuntimeError(f"--resume requires an existing candidate root: {output}")
    if args.preflight and args.resume: raise ValueError("--preflight cannot be combined with --resume")
    if args.base_rows_per_epoch <= 0 or args.base_rows_per_epoch % BASE_BATCH: raise ValueError("base rows must be a positive multiple of 64")
    if args.epochs <= 0: raise ValueError("epochs must be positive")
    parent_checkpoint = (args.parent_checkpoint or CHAMPION).resolve()
    if not parent_checkpoint.is_file(): raise ValueError(f"parent checkpoint is missing: {parent_checkpoint}")
    parent_sha = digest(parent_checkpoint)
    prepared_provenance = None
    mixture_namespace = "stage8b-diversity-exact-v1"
    deep_repeats = 1
    if args.mixture_manifest:
        mixture_payload = json.loads(args.mixture_manifest.read_text())
        if args.teacher_weight is not None or args.selfplay_weight is not None: raise ValueError("do not combine --mixture-manifest with legacy source weights")
        validated_mixture = validate_exact_mixture_manifest(mixture_payload, args=args, parent_sha=parent_sha,
                                                            prepared_root=args.prepared_root)
        mixture_weights = validated_mixture["weights"]
        mixture_exact = True
        mixture_seed = validated_mixture["seed"]
        mixture_namespace = validated_mixture["schedule_seed_namespace"]
        deep_repeats = int(mixture_payload.get("deep_repeats", 1))
        prepared_provenance = validated_mixture["prepared_provenance"]
    else:
        if args.teacher_weight is None or args.selfplay_weight is None: raise ValueError("legacy mode requires --teacher-weight and --selfplay-weight")
        mixture_weights = {"teacher": args.teacher_weight, "selfplay": args.selfplay_weight}; mixture_exact = False; mixture_seed = args.seed
    mixture = SourceMixture(mixture_weights)
    if args.prepared_root:
        prepared_manifest = json.loads((args.prepared_root / "prepared-manifest.json").read_text())
        if prepared_manifest.get("limited_fixture_rows_per_source") is not None and args.allow_limited_prepared_fixture and prepared_provenance is None:
            prepared_provenance = {"root": str(args.prepared_root.resolve()), "limited_fixture": True,
                                   "manifest_path": str((args.prepared_root / "prepared-manifest.json").resolve()),
                                   "manifest_sha256": digest(args.prepared_root / "prepared-manifest.json")}
        elif prepared_provenance is None:
            prepared_provenance = validate_complete_prepared_root(args.prepared_root)
        indexes = {source: PreparedRowIndex(args.prepared_root, source) for source, weight in mixture.weights.items() if weight > 0}
        if "source_rows" in prepared_provenance and any(indexes[source].total != prepared_provenance["source_rows"][source] for source in indexes):
            raise ValueError("prepared mmap row count does not match manifest")
        if "deep1600" in indexes and mixture_weights.get("deep1600", 0.0) <= 0:
            raise ValueError("Deep1600 source is present but not enabled in the mixture")
    else:
        prepared_manifest = None
        prepared_provenance = None
        if mixture_exact:
            raise ValueError("native diversity mixtures require --prepared-root")
        indexes = {source: (RowIndex(teacher_entries(ASSEMBLY, "train"), "teacher") if source == "teacher" else RowIndex(selfplay_entries(SELFPLAY, "train"), "selfplay")) for source, weight in mixture.weights.items() if weight > 0}
    if args.smoke_validation_rows is not None:
        if args.smoke_validation_rows <= 0: raise ValueError("--smoke-validation-rows must be positive")
    batches = args.base_rows_per_epoch // BASE_BATCH
    if args.preflight:
        epoch_one = (exact_batch_schedule(mixture.weights, seed=mixture_seed, epoch=1, batches=batches,
                                          namespace=mixture_namespace) if mixture_exact else
                     mixture.schedule(batches, seed=args.seed, epoch=1))
        print(json.dumps({"schema": "stage8b-candidate-training-preflight-v1", "passed": True,
                          "candidate_id": args.candidate_id, "parent_checkpoint_sha256": parent_sha,
                          "mixture_manifest": str(args.mixture_manifest.resolve()) if args.mixture_manifest else None,
                          "prepared_data_provenance": prepared_provenance,
                          "source_base_rows_per_epoch": (dict(collections.Counter(epoch_one)) if mixture_exact else
                                                           {key: value * BASE_BATCH for key, value in collections.Counter(epoch_one).items()}),
                          "epochs": args.epochs, "base_rows_per_epoch": args.base_rows_per_epoch,
                          "base_batch": BASE_BATCH, "optimizer_steps_per_epoch": batches,
                          "epoch_1_schedule_sha256": hashlib.sha256("\n".join(epoch_one).encode()).hexdigest(),
                          "output_was_not_created": not output.exists()}, sort_keys=True))
        return
    set_seed(args.seed); output.mkdir(parents=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    payload = torch.load(parent_checkpoint, map_location=device, weights_only=False); arch = payload["config"]["architecture"]
    model = Group49Student(channels=arch["channels"], blocks=arch["residual_blocks"]).to(device); model.load_state_dict(payload["model_state"])
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    validation = validation_rows(args.smoke_validation_rows)
    config = {"schema": "stage8b-candidate-training-v1", "candidate_id": args.candidate_id, "parent_champion": {"path": str(parent_checkpoint), "sha256": parent_sha},
              "mixture": mixture.weights, "seed": args.seed, "epochs": args.epochs, "base_rows_per_epoch": args.base_rows_per_epoch,
              "effective_augmented_examples_per_epoch": args.base_rows_per_epoch * 2, "effective_augmented_examples_total": args.base_rows_per_epoch * 2 * args.epochs,
              "base_batch": BASE_BATCH, "effective_optimizer_batch": BASE_BATCH * 2, "optimizer_steps_per_epoch": batches,
              "architecture": arch, "optimizer": {"AdamW": True, "lr": 1e-3, "weight_decay": 1e-4, "scheduler": None},
              "loss": {"policy": "full soft CE", "value": "MSE", "coefficients": {"policy": 1.0, "value": 1.0}},
              "symmetry": "materialized pair: each optimizer batch has original first half then exact colour-transpose second half", "selection": "best-validation-policy on frozen teacher held-out split", "smoke_validation_rows": args.smoke_validation_rows,
              "input_pipeline": "prepared-fp32-v1-vectorized" if args.prepared_root else "original-python-row-v1", "prepared_data_provenance": prepared_provenance, "mixture_manifest": str(args.mixture_manifest.resolve()) if args.mixture_manifest else None, "exact_batch_schedule": mixture_exact, "deep_repeats_per_epoch": deep_repeats, "progress_interval_steps": args.progress_interval}
    config_path = output / "config.json"
    if args.resume:
        if not config_path.is_file() or json.loads(config_path.read_text()) != config:
            raise RuntimeError("--resume candidate configuration mismatch")
        initial_path = output / "checkpoints/initial-champion0.pt"
        if not initial_path.is_file(): raise RuntimeError("--resume requires the immutable initial checkpoint")
        complete = []
        for path in (output / "checkpoints").glob("epoch-*.pt"):
            try: complete.append((int(path.stem.split("-")[1]), path))
            except (IndexError, ValueError): raise RuntimeError(f"malformed epoch checkpoint: {path}")
        completed_epoch, latest = max(complete, default=(0, None), key=lambda item: item[0])
        if completed_epoch > args.epochs: raise RuntimeError("--resume checkpoint epoch exceeds requested epochs")
        history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines() if line.strip()] if (output / "metrics.jsonl").is_file() else []
        if [row.get("epoch") for row in history] != list(range(1, completed_epoch + 1)):
            raise RuntimeError("--resume metrics history does not match committed checkpoints")
        if latest is not None:
            checkpoint = torch.load(latest, map_location=device, weights_only=False)
            if checkpoint.get("epoch") != completed_epoch or "model_state" not in checkpoint or "optimizer_state" not in checkpoint:
                raise RuntimeError("--resume latest epoch checkpoint is incomplete")
            model.load_state_dict(checkpoint["model_state"]); optimizer.load_state_dict(checkpoint["optimizer_state"])
        if (output / "final-report.json").is_file() and completed_epoch != args.epochs:
            raise RuntimeError("final report exists before every requested epoch is committed")
        best_epoch = min((row["epoch"] for row in history), key=lambda epoch: history[epoch - 1]["validation"]["soft_policy_ce"], default=None)
        best = history[best_epoch - 1]["validation"]["soft_policy_ce"] if best_epoch is not None else math.inf
        started = time.monotonic(); first_epoch = completed_epoch + 1
    else:
        atomic_json(config_path, config); atomic_torch(output / "checkpoints/initial-champion0.pt", {"epoch": 0, "model_state": model.state_dict(), "config": config})
        best = math.inf; best_epoch = None; history = []; started = time.monotonic(); first_epoch = 1
    for epoch in range(first_epoch, args.epochs + 1):
        schedule = (exact_batch_schedule(mixture.weights, seed=mixture_seed, epoch=epoch, batches=batches,
                                         namespace=mixture_namespace) if mixture_exact else
                    mixture.schedule(batches, seed=args.seed, epoch=epoch))
        counts = collections.Counter(schedule)
        source_base_rows = dict(counts) if mixture_exact else {key: value * BASE_BATCH for key, value in counts.items()}
        schedule_record = {"epoch": epoch, "base_batch": BASE_BATCH, "base_rows": args.base_rows_per_epoch, "schedule_mode": "row-level-shuffled" if mixture_exact else "batch-level-weighted", "actual_base_rows": source_base_rows}
        if mixture_exact:
            schedule_record["row_source_schedule_sha256"] = hashlib.sha256("\n".join(schedule).encode("utf-8")).hexdigest()
            schedule_record["mixed_batch_count"] = sum(1 for start in range(0, args.base_rows_per_epoch, BASE_BATCH) if len(set(schedule[start:start + BASE_BATCH])) > 1)
        else:
            schedule_record["batch_sources"] = schedule
        atomic_json(output / f"source-schedules/epoch-{epoch:02d}.json", schedule_record)
        rng = {source: random.Random(f"stage8b-row-v1:{args.seed}:{epoch}:{source}") for source in indexes}
        deep_order = []
        if "deep1600" in indexes:
            if deep_repeats <= 0:
                raise RuntimeError("Deep1600 is enabled but deep_repeats is not positive")
            for repeat in range(deep_repeats):
                permutation = list(range(indexes["deep1600"].total))
                random.Random(f"stage8b-deep1600-order-v{repeat + 1}:{mixture_seed}:{epoch}").shuffle(permutation)
                deep_order.extend(permutation)
        deep_cursor = 0
        sums = {"policy": 0.0, "value": 0.0, "total": 0.0}; model.train()
        epoch_started = time.monotonic()
        step_count = batches
        for step in range(1, step_count + 1):
            if args.prepared_root:
                if mixture_exact:
                    batch_sources = schedule[(step - 1) * BASE_BATCH:step * BASE_BATCH]
                    state_parts = []; policy_parts = []; value_parts = []
                    for source in batch_sources:
                        if source == "deep1600":
                            if deep_cursor >= len(deep_order): raise RuntimeError("Deep1600 coverage overrun")
                            s1, p1, z1 = indexes[source].sample_index(deep_order[deep_cursor]); deep_cursor += 1
                        else:
                            s1, p1, z1 = indexes[source].sample_batch(rng[source], 1)
                        state_parts.append(s1); policy_parts.append(p1); value_parts.append(z1)
                    base_state = np.concatenate(state_parts, axis=0); base_policy = np.concatenate(policy_parts, axis=0); base_value = np.concatenate(value_parts, axis=0)
                else:
                    source = schedule[step - 1]
                    base_state, base_policy, base_value = indexes[source].sample_batch(rng[source], BASE_BATCH)
                state, policy, value = prepared_tensors(base_state, base_policy, base_value, device)
            else:
                source = schedule[step - 1]
                base = [indexes[source].sample(rng[source]) for _ in range(BASE_BATCH)]
                state, policy, value = tensors(base, device)
            optimizer.zero_grad(); logits, prediction = model(state)
            policy_loss = soft_policy_loss(logits, policy, torch.ones(len(state), device=device)); value_loss = weighted_mse(prediction, value, torch.ones(len(state), device=device)); loss = policy_loss + value_loss
            if not torch.isfinite(loss): raise RuntimeError("non-finite training loss")
            loss.backward(); optimizer.step()
            for key, item in (("policy", policy_loss), ("value", value_loss), ("total", loss)): sums[key] += float(item.detach())
            if args.progress_interval > 0 and (step % args.progress_interval == 0 or step == batches):
                elapsed = time.monotonic() - epoch_started; rate = step / max(elapsed, 1e-9)
                atomic_json(output / "progress.json", {"schema": "stage8b-trainer-progress-v1", "candidate_id": args.candidate_id, "epoch": epoch, "completed_optimizer_steps": (epoch - 1) * batches + step, "completed_base_rows": (epoch - 1) * args.base_rows_per_epoch + step * BASE_BATCH, "elapsed_epoch_seconds": elapsed, "base_rows_per_second": step * BASE_BATCH / elapsed, "effective_examples_per_second": step * BASE_BATCH * 2 / elapsed, "latest_losses": {"policy": float(policy_loss.detach()), "value": float(value_loss.detach()), "total": float(loss.detach())}, "estimated_epoch_remaining_seconds": (batches - step) / rate})
        evaluation = evaluate_policy(model, validation, device)
        if deep_order and deep_cursor != len(deep_order): raise RuntimeError(f"Deep1600 coverage incomplete: {deep_cursor}/{len(deep_order)}")
        row = {"epoch": epoch, "train": {key: value / batches for key, value in sums.items()} | {"base_rows": args.base_rows_per_epoch, "effective_examples": args.base_rows_per_epoch * 2}, "validation": evaluation, "source_base_rows": {key: counts.get(key, 0) if mixture_exact else counts.get(key, 0) * BASE_BATCH for key in mixture.weights}, "wall_seconds": time.monotonic() - started}
        history.append(row); (output / "metrics.jsonl").open("a", encoding="utf-8").write(json.dumps(row, sort_keys=True) + "\n")
        checkpoint = {"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "config": config, "validation": evaluation}
        atomic_torch(output / f"checkpoints/epoch-{epoch:02d}.pt", checkpoint); atomic_torch(output / "checkpoints/final.pt", checkpoint)
        if evaluation["soft_policy_ce"] < best:
            best, best_epoch = evaluation["soft_policy_ce"], epoch; atomic_torch(output / "checkpoints/best-validation-policy.pt", checkpoint)
        if epoch == 1 or evaluation["soft_policy_ce"] <= min(item["validation"]["soft_policy_ce"] for item in history): atomic_torch(output / "checkpoints/best-validation-total.pt", checkpoint)
    atomic_json(output / "final-report.json", {"passed": True, "candidate_id": args.candidate_id, "best_validation_policy": best, "best_validation_policy_epoch": best_epoch, "epochs": history, "checkpoint_hashes": {path.name: digest(path) for path in sorted((output / "checkpoints").glob("*.pt"))}, "wall_seconds": time.monotonic() - started})


if __name__ == "__main__": main()
