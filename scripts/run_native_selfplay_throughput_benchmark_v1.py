#!/usr/bin/env python3
"""Run one controlled, auditable native-self-play throughput benchmark plan.

This executor deliberately varies only process topology and the qualified
shared-inference resource settings.  Each worker owns an independent native-v2
corpus root and a deterministic, disjoint part of the same fixed NORMAL game-ID
workload.  It never writes to a production corpus.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIFECYCLE = ROOT / "scripts/run_selfplay_v2_native.py"
DEFAULT_AUDIT = ROOT / "scripts/audit_selfplay_v2_native.py"
DEFAULT_CUDA_WRAPPER = ROOT / "scripts/stage7_cuda12_runtime_v1.sh"
DEFAULT_RUNNER = ROOT / "build/cpp-puct-stage7/hex_native_selfplay_v2_runner"


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_plan(path: Path) -> dict[str, Any]:
    plan = read_json(path)
    if plan.get("schema") != "hex-native-selfplay-throughput-plan-v1":
        raise RuntimeError("wrong throughput plan schema")
    required = ("plan_id", "workload", "champion_registry", "search", "configs")
    if any(key not in plan for key in required):
        raise RuntimeError("incomplete throughput plan")
    if plan["workload"].get("prefix_mode") != "normal":
        raise RuntimeError("v1 topology comparison is intentionally NORMAL-only")
    if plan["workload"].get("games", 0) <= 0:
        raise RuntimeError("invalid workload game count")
    if not plan["configs"]:
        raise RuntimeError("plan has no configurations")
    for config in plan["configs"]:
        if not isinstance(config.get("benchmark_id"), str):
            raise RuntimeError("configuration missing benchmark_id")
        for key in ("process_count", "concurrency_per_process", "max_batch", "wait_us"):
            if int(config.get(key, 0)) <= 0:
                raise RuntimeError(f"invalid resource field {key}")
    return plan


def partition_workload(start_id: int, games: int, process_count: int) -> list[dict[str, int]]:
    """Return deterministic contiguous non-overlapping partitions whose union is exact."""
    if games <= 0 or process_count <= 0 or process_count > games:
        raise ValueError("invalid workload partition")
    q, r = divmod(games, process_count)
    result: list[dict[str, int]] = []
    cursor = start_id
    for worker in range(process_count):
        count = q + (1 if worker < r else 0)
        result.append({"worker_index": worker, "start_id": cursor, "games": count,
                       "end_exclusive": cursor + count})
        cursor += count
    assert cursor == start_id + games
    return result


def _proc_snapshot(roots: list[int]) -> tuple[int, int]:
    """Aggregate CPU ticks and RSS of process trees rooted at *roots* using /proc."""
    parents: dict[int, list[int]] = {}
    stats: dict[int, tuple[int, int]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().split()
            # Linux proc stat: ppid index 3, utime/stime 13/14, rss 23.
            pid, ppid = int(fields[0]), int(fields[3])
            stats[pid] = (int(fields[13]) + int(fields[14]), int(fields[23]))
            parents.setdefault(ppid, []).append(pid)
        except (OSError, ValueError, IndexError):
            continue
    todo, seen = list(roots), set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        todo.extend(parents.get(pid, []))
    ticks = sum(stats[pid][0] for pid in seen if pid in stats)
    rss = sum(stats[pid][1] for pid in seen if pid in stats) * os.sysconf("SC_PAGE_SIZE")
    return ticks, rss


def _gpu_sample() -> tuple[dict[str, float] | None, str | None]:
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used,clocks.sm,clocks.mem",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, str(exc)
    if result.returncode or not result.stdout.strip():
        return None, (result.stderr.strip() or result.stdout.strip() or f"nvidia-smi exit {result.returncode}")
    try:
        fields = [float(x.strip()) for x in result.stdout.splitlines()[0].split(",")]
        if len(fields) != 6:
            raise ValueError("unexpected GPU field count")
        return dict(zip(("utilization_percent", "power_watts", "temperature_c", "vram_mib", "sm_clock_mhz", "memory_clock_mhz"), fields)), None
    except ValueError as exc:
        return None, f"cannot parse nvidia-smi: {exc}: {result.stdout.strip()}"


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p10": None, "p90": None, "max": None}
    ordered = sorted(values)
    def at(q: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p10": at(0.10), "p90": at(0.90), "max": max(values)}


class Telemetry:
    def __init__(self, output: Path, roots: list[int], interval_seconds: float):
        self.output, self.roots, self.interval = output, roots, interval_seconds
        self.samples: list[dict[str, Any]] = []
        self.previous: tuple[float, int] | None = None
        self.gpu_errors: set[str] = set()

    def sample(self) -> None:
        now = time.monotonic()
        ticks, rss = _proc_snapshot(self.roots)
        row: dict[str, Any] = {"monotonic_seconds": now, "rss_bytes": rss}
        if self.previous:
            previous_time, previous_ticks = self.previous
            dt = max(now - previous_time, 1e-9)
            # A native-v2 lifecycle worker replaces its C++ child at wave
            # boundaries.  Summing only live descendants makes the aggregate
            # tick counter decrease at that instant; omit the discontinuity
            # instead of reporting a fictitious negative CPU load.
            if ticks >= previous_ticks:
                row["cpu_percent_one_core_scale"] = 100.0 * (ticks - previous_ticks) / os.sysconf("SC_CLK_TCK") / dt
        self.previous = (now, ticks)
        gpu, error = _gpu_sample()
        if gpu:
            row["gpu"] = gpu
        if error:
            self.gpu_errors.add(error)
        self.samples.append(row)

    def write(self) -> dict[str, Any]:
        telemetry_path = self.output / "telemetry.jsonl"
        with telemetry_path.open("w", encoding="utf-8") as f:
            for row in self.samples:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        gpu_keys = ("utilization_percent", "power_watts", "temperature_c", "vram_mib", "sm_clock_mhz", "memory_clock_mhz")
        summary: dict[str, Any] = {"sample_count": len(self.samples),
            "cpu_percent_one_core_scale": _percentiles([float(r["cpu_percent_one_core_scale"]) for r in self.samples if "cpu_percent_one_core_scale" in r]),
            "rss_mib": _percentiles([float(r["rss_bytes"]) / (1024 * 1024) for r in self.samples]),
            "gpu": {key: _percentiles([float(r["gpu"][key]) for r in self.samples if "gpu" in r]) for key in gpu_keys},
            "gpu_telemetry_errors": sorted(self.gpu_errors)}
        atomic_json(self.output / "telemetry-summary.json", summary)
        return summary


class BenchmarkLock:
    def __init__(self, path: Path):
        self.path, self.fd = path, None
    def __enter__(self) -> "BenchmarkLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = self.path.open("a+")
        try:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another throughput executor holds this benchmark root lock") from exc
        return self
    def __exit__(self, *_: object) -> None:
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN)
            self.fd.close()
        self.path.unlink(missing_ok=True)


def child_command(args: argparse.Namespace, plan: dict[str, Any], config: dict[str, Any], worker: dict[str, int], output: Path, mode: str) -> list[str]:
    workload, search = plan["workload"], plan["search"]
    command = [str(args.cuda_wrapper), str(args.python), str(args.lifecycle), f"--{mode}",
        "--output", str(output), "--run-id", f"{plan['plan_id']}-{config['benchmark_id']}-worker-{worker['worker_index']:02d}",
        "--champion-registry", str(Path(plan["champion_registry"]).resolve()),
        "--start-id", str(worker["start_id"]), "--games", str(worker["games"]),
        "--master-seed", str(workload["master_seed"]), "--budget", str(search["budget"]),
        "--concurrency", str(config["concurrency_per_process"]), "--max-batch", str(config["max_batch"]),
        "--wait-us", str(config["wait_us"]), "--watchdog-seconds", str(search["watchdog_seconds"]),
        "--c-puct", str(search["c_puct"]), "--fpu-mode", str(search["fpu_mode"]),
        "--fpu-reduction", str(search["fpu_reduction"]), "--runner", str(args.runner)]
    if args.recover_stale_lock:
        command.append("--recover-stale-lock")
    return command


def runner_stats(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    candidates: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
            if value.get("status") == "complete" and "requests" in value:
                candidates.append(value)
        except json.JSONDecodeError:
            continue
    # The lifecycle wrapper invokes the C++ runner in waves.  Each completed
    # wave prints one service-stat record, so throughput accounting must retain
    # every record rather than (incorrectly) using just the final wave.
    return candidates


def semantic_record(game: dict[str, Any]) -> dict[str, Any]:
    """Only game/search semantics, deliberately excluding run/config provenance."""
    return {key: game.get(key) for key in ("game_id", "game_seed", "moves", "phase_a_rows", "certificate_ply",
        "certificate_owner", "phase_b_moves", "literal_winner", "classification", "samples")}


def audit_worker(args: argparse.Namespace, worker_root: Path) -> dict[str, Any]:
    audit_path = worker_root / "postrun-audit.json"
    result = subprocess.run([str(args.python), str(args.audit), "--root", str(worker_root), "--output", str(audit_path)],
                            text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"postrun audit failed for {worker_root}: {result.stderr.strip()}")
    return read_json(audit_path)


def aggregate_worker_games(worker_root: Path, expected: dict[str, int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for game_id in range(expected["start_id"], expected["end_exclusive"]):
        game = read_json(worker_root / "games" / f"game-{game_id}.json")
        if game_id in result:
            raise RuntimeError("duplicate game ID while collecting benchmark semantics")
        result[game_id] = semantic_record(game)
    return result


def run_config(args: argparse.Namespace, plan: dict[str, Any], plan_sha: str, config: dict[str, Any], root: Path, mode: str) -> dict[str, Any]:
    config_root = root / "configs" / config["benchmark_id"]
    manifest_path = config_root / "benchmark-config-manifest.json"
    workload = plan["workload"]
    partitions = partition_workload(int(workload["start_id"]), int(workload["games"]), int(config["process_count"]))
    identity = {"schema": "hex-native-selfplay-throughput-config-v1", "plan_id": plan["plan_id"], "plan_sha256": plan_sha,
        "benchmark_id": config["benchmark_id"], "workload": workload, "champion_registry": plan["champion_registry"],
        "champion_onnx_sha256": plan["champion_onnx_sha256"], "search": plan["search"], "resource": config,
        "partitions": partitions, "measurement": "startup-inclusive wall time from worker launch until all audited workers exit"}
    if mode == "new":
        if config_root.exists():
            raise RuntimeError(f"--new refuses existing benchmark config root {config_root}")
        config_root.mkdir(parents=True)
        atomic_json(manifest_path, identity)
    elif not manifest_path.is_file() or read_json(manifest_path) != identity:
        raise RuntimeError(f"--resume immutable configuration mismatch for {config_root}")
    workers: list[tuple[dict[str, int], Path]] = [(part, config_root / "workers" / f"worker-{part['worker_index']:02d}") for part in partitions]
    commands = [child_command(args, plan, config, part, worker_root, mode) for part, worker_root in workers]
    atomic_json(config_root / "commands.json", {"commands": commands})
    launched = time.time(); monotonic_start = time.monotonic(); processes = [subprocess.Popen(command) for command in commands]
    telemetry = Telemetry(config_root, [p.pid for p in processes], args.telemetry_interval)
    try:
        while any(p.poll() is None for p in processes):
            telemetry.sample()
            time.sleep(args.telemetry_interval)
        telemetry.sample()
    except KeyboardInterrupt:
        # The child lifecycle wrappers translate SIGINT into a safe drain. Do not
        # kill a C++ child directly from a benchmark orchestration interruption.
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            process.wait()
        raise
    ended = time.time()
    # The benchmark may be launched from a sandboxed controller while native
    # children run in normal WSL.  Wall-clock epoch timestamps are shared;
    # monotonic namespaces need not be.  Use the former for all reported rates
    # and preserve the latter only as diagnostic evidence.
    elapsed = ended - launched
    elapsed_monotonic_namespace = time.monotonic() - monotonic_start
    telemetry_summary = telemetry.write()
    exit_codes = [p.returncode for p in processes]
    if any(code != 0 for code in exit_codes):
        atomic_json(config_root / "config-result.json", {"schema": "hex-native-selfplay-throughput-result-v1", "status": "failed",
            "exit_codes": exit_codes, "elapsed_seconds_wall_clock": elapsed, "elapsed_seconds_monotonic_namespace": elapsed_monotonic_namespace, "telemetry": telemetry_summary})
        raise RuntimeError(f"benchmark configuration {config['benchmark_id']} worker failures: {exit_codes}")
    audits = [audit_worker(args, worker_root) for _, worker_root in workers]
    if any(not audit.get("complete") or audit.get("quarantined_artifacts") != 0 or audit.get("accepted_games") != part["games"]
           for audit, (part, _) in zip(audits, workers)):
        raise RuntimeError(f"benchmark configuration {config['benchmark_id']} failed corpus integrity guard")
    game_map: dict[int, dict[str, Any]] = {}
    for part, worker_root in workers:
        games = aggregate_worker_games(worker_root, part)
        overlap = set(game_map).intersection(games)
        if overlap:
            raise RuntimeError(f"duplicate IDs across workers: {sorted(overlap)[:3]}")
        game_map.update(games)
    if sorted(game_map) != list(range(int(workload["start_id"]), int(workload["start_id"]) + int(workload["games"]))):
        raise RuntimeError("worker game-ID union does not equal fixed workload")
    atomic_json(config_root / "semantic-fingerprints.json", {"schema": "hex-native-selfplay-throughput-semantics-v1",
        "game_ids": sorted(game_map), "fingerprints": {str(gid): hashlib.sha256(canonical(value)).hexdigest() for gid, value in game_map.items()},
        "records": {str(gid): value for gid, value in game_map.items()}})
    stats = [runner_stats(worker_root / "runner.stdout.log") for _, worker_root in workers]
    all_stats = [stat for records in stats for stat in records]
    requests, batches = sum(int(s.get("requests", 0)) for s in all_stats), sum(int(s.get("batches", 0)) for s in all_stats)
    peak_batch = max((int(s.get("peak_batch", 0)) for s in all_stats), default=0)
    queue_high_water = max((int(s.get("queue_high_water", 0)) for s in all_stats), default=0)
    rows = sum(int(audit["phase_a_rows"]) for audit in audits)
    result = {"schema": "hex-native-selfplay-throughput-result-v1", "status": "complete", "benchmark_id": config["benchmark_id"],
        "started_epoch": launched, "ended_epoch": ended, "elapsed_seconds_startup_inclusive": elapsed,
        "elapsed_seconds_monotonic_namespace": elapsed_monotonic_namespace,
        "accepted_games": int(workload["games"]), "phase_a_rows": rows, "requested_simulations": rows * int(plan["search"]["budget"]),
        "games_per_second": int(workload["games"]) / elapsed, "games_per_hour": 3600 * int(workload["games"]) / elapsed,
        "phase_a_rows_per_second": rows / elapsed, "simulations_per_second": rows * int(plan["search"]["budget"]) / elapsed,
        "inference": {"requests": requests, "batches": batches, "mean_batch": requests / batches if batches else None,
            "peak_batch": peak_batch, "queue_high_water": queue_high_water}, "workers": [{"partition": part, "root": str(worker_root), "audit": audit, "runner_stats": stat}
             for (part, worker_root), audit, stat in zip(workers, audits, stats)], "telemetry": telemetry_summary,
        "semantic_fingerprint_sha256": digest(config_root / "semantic-fingerprints.json")}
    atomic_json(config_root / "config-result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--new", action="store_true")
    mode.add_argument("--resume", action="store_true")
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", action="append", help="one benchmark_id; default is every plan configuration")
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--cuda-wrapper", type=Path, default=DEFAULT_CUDA_WRAPPER)
    p.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    p.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--telemetry-interval", type=float, default=2.0)
    p.add_argument("--recover-stale-lock", action="store_true")
    return p


def main() -> int:
    args = parser().parse_args()
    if args.telemetry_interval < 0.5:
        raise RuntimeError("telemetry interval must be at least 0.5 seconds")
    for path in (args.python, args.cuda_wrapper, args.lifecycle, args.audit, args.runner, args.plan):
        if not path.is_file():
            raise RuntimeError(f"required executable/input missing: {path}")
    plan = load_plan(args.plan.resolve())
    registry = Path(plan["champion_registry"])
    if not registry.is_file():
        raise RuntimeError("champion registry missing")
    champion = read_json(registry)
    if champion.get("onnx", {}).get("onnx_sha256") != plan["champion_onnx_sha256"]:
        raise RuntimeError("plan Champion ONNX identity does not match registry")
    plan_sha = digest(args.plan.resolve())
    root = args.output.resolve()
    root_manifest = root / "benchmark-manifest.json"
    root_identity = {"schema": "hex-native-selfplay-throughput-benchmark-v1", "plan_path": str(args.plan.resolve()),
        "plan_sha256": plan_sha, "plan_id": plan["plan_id"], "champion_onnx_sha256": plan["champion_onnx_sha256"]}
    if args.new:
        # Analysis freezes a follow-on plan under its future stage root before
        # execution.  Permit precisely that public plan file and nothing else;
        # any manifest/config/result remains an overwrite hazard.
        if root.exists() and any(path.name != "plan.json" for path in root.iterdir()):
            raise RuntimeError("--new refuses an existing benchmark root")
        root.mkdir(parents=True, exist_ok=True)
        atomic_json(root_manifest, root_identity)
    elif not root_manifest.is_file() or read_json(root_manifest) != root_identity:
        raise RuntimeError("--resume immutable benchmark plan mismatch")
    requested = set(args.config or [config["benchmark_id"] for config in plan["configs"]])
    configs = [config for config in plan["configs"] if config["benchmark_id"] in requested]
    if len(configs) != len(requested):
        raise RuntimeError(f"unknown benchmark configuration(s): {sorted(requested - {c['benchmark_id'] for c in configs})}")
    summaries: list[dict[str, Any]] = []
    with BenchmarkLock(root / ".throughput-benchmark.lock"):
        for config in configs:
            summaries.append(run_config(args, plan, plan_sha, config, root, "new" if args.new else "resume"))
        aggregate = {"schema": "hex-native-selfplay-throughput-aggregate-v1", "plan_id": plan["plan_id"], "plan_sha256": plan_sha,
            "configs": [{key: result[key] for key in ("benchmark_id", "accepted_games", "phase_a_rows", "elapsed_seconds_startup_inclusive",
                "games_per_second", "games_per_hour", "phase_a_rows_per_second", "simulations_per_second", "inference", "telemetry")}
                for result in summaries]}
        atomic_json(root / "aggregate.json", aggregate)
    print(json.dumps({"status": "complete", "output": str(root), "configs": [r["benchmark_id"] for r in summaries]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"native selfplay throughput benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(1)
