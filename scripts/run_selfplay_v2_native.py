#!/usr/bin/env python3
"""Lifecycle wrapper for immutable native-clean Champion-N self-play v2.

The C++ runner owns PUCT, certificate detection, realisation, and atomic game
commit.  This wrapper owns corpus identity, explicit new/resume semantics,
locking, attempts, live status, and bounded graceful draining.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

from forced_prefix_bank_v1 import load_bank, sha256 as prefix_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BIN = ROOT / "build/cpp-puct-stage7/hex_native_selfplay_v2_runner"
DEFAULT_REGISTRY = ROOT / "artifacts/champions-v1/champion-1.json"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def champion_identity(path: Path) -> dict:
    champion = read_json(path)
    if champion.get("champion_id") != "champion-1":
        raise RuntimeError("native-v2 generation currently requires frozen champion-1 registry record")
    # The immutable registry records the qualified exporter artifact verbatim.
    # Its established field names are `onnx` / `onnx_sha256`, not a second
    # wrapper schema invented by this generator.
    onnx = Path(champion["onnx"]["onnx"])
    if not onnx.is_file() or digest(onnx) != champion["onnx"]["onnx_sha256"]:
        raise RuntimeError("Champion-1 ONNX provenance/hash mismatch")
    return {"registry_path": str(path.resolve()), "registry_sha256": digest(path),
            "champion_id": champion["champion_id"], "onnx_path": str(onnx.resolve()),
            "onnx_sha256": champion["onnx"]["onnx_sha256"],
            "checkpoint_path": champion["checkpoint"]["path"],
            "checkpoint_sha256": champion["checkpoint"]["sha256"]}


def immutable_config(args: argparse.Namespace, champion: dict, prefix: dict | None = None) -> dict:
    native_v3 = args.c_puct is not None or args.fpu_mode is not None or args.fpu_reduction is not None or prefix is not None
    payload = {"schema": "hex-native-selfplay-run-v3" if native_v3 else "hex-native-selfplay-run-v2", "schema_version": 3 if native_v3 else 2,
        "run_id": args.run_id, "champion": champion,
        "game_ids": {"start": args.start_id, "count": args.games,
                     "end_exclusive": args.start_id + args.games},
        "master_seed": args.master_seed,
        "seed_algorithm": "splitmix64(master_seed XOR splitmix64(game_id))",
        "search": {"budget": args.budget, "c_puct": args.c_puct if args.c_puct is not None else 1.5},
        "exploration": {"temperature_plies": 20, "temperature": 1,
                         "after_temperature": "root_visit_argmax", "dirichlet": False},
        "inference": {"concurrency": args.concurrency, "max_batch": args.max_batch,
                      "wait_us": args.wait_us, "watchdog_seconds": args.watchdog_seconds},
        "phase_contract": {"phase_a": "store root before move including certificate-producing move; stop after first valid post-move certificate",
                           "phase_b": "qualified elementary realizer only; no PUCT policy rows; LiteralWinner must equal owner",
                           "value": "z=+1 iff retained state side_to_move equals verified LiteralWinner"}}
    if native_v3:
        payload["search"]["fpu_mode"] = args.fpu_mode or "zero"
        payload["search"]["fpu_reduction"] = args.fpu_reduction if args.fpu_reduction is not None else 0.0
        payload["prefix"] = prefix or {"mode": "normal", "prefix_plies": 0, "bank_sha256": None}
    payload["config_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def valid_game(path: Path, manifest: dict, game_id: int) -> tuple[bool, str]:
    try:
        x = read_json(path)
        expected_schema = "hex-native-selfplay-game-v3" if manifest.get("schema_version") == 3 else "hex-native-selfplay-game-v2"
        assert x["schema"] == expected_schema
        assert x["status"] == "accepted" and x["run_id"] == manifest["run_id"]
        assert x["game_id"] == game_id
        assert x["game_seed"] == splitmix_seed(manifest["master_seed"], game_id)
        assert x["model_sha256"] == manifest["champion"]["onnx_sha256"]
        assert x["config_sha256"] == manifest["config_sha256"]
        assert x["phase_a_rows"] == len(x["samples"])
        assert x["literal_winner"] in {"B", "W"}
        assert all(len(s["root_visits"]) == 121 and s["z"] in {-1, 1} for s in x["samples"])
        assert all((s["z"] == 1) == (s["side_to_move"] == x["literal_winner"]) for s in x["samples"])
        if expected_schema == "hex-native-selfplay-game-v3":
            mode = x.get("prefix_mode")
            assert mode in {"normal", "forced"}
            forced = x.get("forced_prefix_actions", [])
            assert x.get("forced_prefix_length") == len(forced)
            assert len(forced) in {0, 3}
            if mode == "normal":
                assert not forced and x.get("prefix_id") is None and x.get("prefix_bank_sha256") is None
            else:
                assert len(forced) == 3 and len(set(forced)) == 3 and all(0 <= int(a) < 121 for a in forced)
                assert isinstance(x.get("prefix_id"), str)
                assert x.get("prefix_bank_sha256") == manifest.get("prefix", {}).get("bank_sha256")
            assert all(int(s["ply"]) >= len(forced) for s in x["samples"])
            if x["samples"]:
                assert x["samples"][0]["ply"] == len(forced)
        return True, ""
    except Exception as e:  # corrupt files are intentionally never accepted
        return False, str(e)

def valid_quarantine(path: Path, manifest: dict, game_id: int) -> bool:
    try:
        x=read_json(path)
        schema = "hex-native-selfplay-game-v3" if manifest.get("schema_version") == 3 else "hex-native-selfplay-game-v2"
        return x.get("schema")==schema and x.get("status")=="quarantined" and x.get("run_id")==manifest["run_id"] and x.get("game_id")==game_id and x.get("game_seed")==splitmix_seed(manifest["master_seed"],game_id) and x.get("model_sha256")==manifest["champion"]["onnx_sha256"] and x.get("config_sha256")==manifest["config_sha256"] and bool(x.get("reason"))
    except Exception: return False


def splitmix_seed(master: int, game_id: int) -> int:
    def mix(x: int) -> int:
        x = (x + 0x9e3779b97f4a7c15) & ((1 << 64) - 1)
        x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & ((1 << 64) - 1)
        x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & ((1 << 64) - 1)
        return x ^ (x >> 31)
    return mix(master ^ mix(game_id))


class CorpusLock:
    def __init__(self, root: Path, recover_stale: bool, metadata: dict):
        self.path = root / ".native-selfplay-v2.lock"; self.metadata = metadata; self.fd = None
        if self.path.exists() and not recover_stale:
            # flock below is authoritative for liveness; explicit recovery prevents
            # accidental reuse of a stale-looking root.
            raise RuntimeError("lock path exists; use --recover-stale-lock only after confirming no live writer")
    def __enter__(self):
        self.fd = self.path.open("a+")
        try: fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("another native-v2 writer holds the corpus lock")
        self.fd.seek(0); self.fd.truncate(); json.dump(self.metadata, self.fd); self.fd.flush(); os.fsync(self.fd.fileno())
        return self
    def __exit__(self, *_):
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN); self.fd.close()
        self.path.unlink(missing_ok=True)


def append_attempt(root: Path, record: dict) -> None:
    with (root / "run-attempts.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n"); f.flush(); os.fsync(f.fileno())


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True); mode.add_argument("--new", action="store_true"); mode.add_argument("--resume", action="store_true")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--run-id", default="champion-1-native-v2")
    p.add_argument("--champion-registry", type=Path, default=DEFAULT_REGISTRY); p.add_argument("--start-id", type=int, default=0); p.add_argument("--games", type=int, required=True)
    p.add_argument("--master-seed", type=int, default=4901); p.add_argument("--budget", type=int, default=128); p.add_argument("--concurrency", type=int, default=64); p.add_argument("--max-batch", type=int, default=96); p.add_argument("--wait-us", type=int, default=200)
    p.add_argument("--c-puct", type=float); p.add_argument("--fpu-mode", choices=("zero", "parent_value_reduced")); p.add_argument("--fpu-reduction", type=float); p.add_argument("--prefix-bank", type=Path, help="immutable three-ply forced-prefix bank; enables native schema v3")
    p.add_argument("--watchdog-seconds", type=int, default=120); p.add_argument("--drain-timeout-seconds", type=int, default=300); p.add_argument("--recover-stale-lock", action="store_true")
    p.add_argument("--runner", type=Path, default=DEFAULT_BIN); return p


def main() -> int:
    a = parser().parse_args(); out = a.output.resolve(); champion = champion_identity(a.champion_registry.resolve())
    prefix_payload = None
    prefix_rows = []
    if a.prefix_bank is not None:
        prefix_payload, prefix_rows = load_bank(a.prefix_bank.resolve())
        prefix_payload = {"mode": "forced", "prefix_plies": 3, "bank_path": str(a.prefix_bank.resolve()), "bank_sha256": prefix_sha256(a.prefix_bank.resolve()), "bank_count": len(prefix_rows), "assignment": "prefix_index = game_id modulo bank_count"}
    cfg = immutable_config(a, champion, prefix_payload)
    if a.games <= 0 or a.concurrency <= 0 or a.budget <= 0: raise RuntimeError("games, concurrency, and budget must be positive")
    manifest_path = out / "run-manifest.json"
    if a.new:
        if out.exists(): raise RuntimeError("--new refuses existing output root")
        out.mkdir(parents=True); (out / "games").mkdir(); (out / "quarantine").mkdir()
        atomic_json(manifest_path, cfg)
    else:
        if not manifest_path.is_file(): raise RuntimeError("--resume requires initialized run-manifest.json")
        old = read_json(manifest_path)
        if old != cfg: raise RuntimeError("--resume immutable configuration mismatch")
    assignments = out / "prefix-assignments.jsonl"
    if a.prefix_bank is not None:
        content = "".join(f"{gid}|{prefix_rows[gid % len(prefix_rows)]['prefix_id']}|{','.join(map(str, prefix_rows[gid % len(prefix_rows)]['actions']))}\n" for gid in range(a.start_id, a.start_id + a.games))
        if assignments.exists() and assignments.read_text() != content: raise RuntimeError("prefix assignment manifest mismatch")
        if not assignments.exists(): atomic_text(assignments, content)
    attempt = str(uuid.uuid4()); started = time.time(); stop_requested = False; child_pid: int | None = None
    def status(state: str, **extra: object) -> None:
        ids = range(a.start_id, a.start_id + a.games); good = [i for i in ids if (out / "games" / f"game-{i}.json").is_file() and valid_game(out / "games" / f"game-{i}.json", cfg, i)[0]]
        quarantined=[i for i in ids if valid_quarantine(out/"quarantine"/f"game-{i}.json",cfg,i)]
        atomic_json(out / "runner-status.json", {"schema":"hex-native-selfplay-status-v2","state":state,"attempt_id":attempt,"run_id":a.run_id,"target_games":a.games,"accepted":len(good),"quarantined":len(quarantined),"missing":a.games-len(good)-len(quarantined),"elapsed_seconds":time.time()-started,"stop_requested":stop_requested,"child_pid":child_pid,**extra})
    def request_stop(_sig, _frame):
        nonlocal stop_requested
        stop_requested = True; atomic_json(out / "stop-request.json", {"schema":"hex-native-selfplay-stop-v2","requested_epoch":time.time(),"attempt_id":attempt,"reason":"signal"})
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    with CorpusLock(out, a.recover_stale_lock, {"pid":os.getpid(),"hostname":socket.gethostname(),"run_id":a.run_id,"attempt_id":attempt,"started_epoch":started}):
        # A prior stop request is consumed at the start of a deliberate resume.
        if a.resume: (out / "stop-request.json").unlink(missing_ok=True)
        missing=[]
        for gid in range(a.start_id, a.start_id+a.games):
            f=out/"games"/f"game-{gid}.json"
            if f.exists():
                ok, why=valid_game(f,cfg,gid)
                if ok: continue
                target=out/"quarantine"/f"corrupt-game-{gid}-{int(time.time())}.json"
                os.replace(f,target); (target.with_suffix(target.suffix+".reason.txt")).write_text(why+"\n")
            if valid_quarantine(out/"quarantine"/f"game-{gid}.json",cfg,gid):
                continue
            missing.append(gid)
        status("running", active=0)
        failure=""; exit_code=0
        try:
            # Waves bound graceful drain: each C++ invocation has at most concurrency active games.
            while missing and not stop_requested:
                if (out/"stop-request.json").exists(): stop_requested=True; break
                wave=missing[:a.concurrency]; missing=missing[a.concurrency:]
                cmd=[str(a.runner.resolve()),"--model",champion["onnx_path"],"--output",str(out),"--run-id",a.run_id,"--model-sha",champion["onnx_sha256"],"--config-sha",cfg["config_sha256"],"--master-seed",str(a.master_seed),"--start-id",str(wave[0]),"--games",str(len(wave)),"--budget",str(a.budget),"--concurrency",str(min(a.concurrency,len(wave))),"--max-batch",str(a.max_batch),"--wait-us",str(a.wait_us),"--watchdog-seconds",str(a.watchdog_seconds),"--stop-file",str(out/"stop-request.json")]
                if cfg.get("schema_version") == 3:
                    cmd += ["--c-puct", str(cfg["search"]["c_puct"]), "--fpu-mode", cfg["search"]["fpu_mode"], "--fpu-reduction", str(cfg["search"]["fpu_reduction"])]
                if a.prefix_bank is not None:
                    wave_assignments = out / f"prefix-assignments-{wave[0]}-{len(wave)}.jsonl"
                    wave_content = "".join(f"{gid}|{prefix_rows[gid % len(prefix_rows)]['prefix_id']}|{','.join(map(str, prefix_rows[gid % len(prefix_rows)]['actions']))}\n" for gid in wave)
                    if wave_assignments.exists() and wave_assignments.read_text() != wave_content: raise RuntimeError("wave prefix assignment mismatch")
                    if not wave_assignments.exists(): atomic_text(wave_assignments, wave_content)
                    cmd += ["--prefix-assignments", str(wave_assignments), "--prefix-bank-sha", prefix_payload["bank_sha256"]]
                with (out/"runner.stdout.log").open("a") as so, (out/"runner.stderr.log").open("a") as se:
                    # Separate session: Ctrl-C belongs to the lifecycle parent,
                    # which requests a bounded wave drain instead of killing
                    # active C++ games mid-commit.
                    p=subprocess.Popen(cmd,stdout=so,stderr=se,start_new_session=True); child_pid=p.pid
                    deadline=None
                    while p.poll() is None:
                        if (out/"stop-request.json").exists() or stop_requested:
                            stop_requested=True; deadline=deadline or (time.monotonic()+a.drain_timeout_seconds)
                            if time.monotonic()>deadline: p.terminate()
                        status("draining" if stop_requested else "running",active=len(wave))
                        time.sleep(2)
                    if p.returncode:
                        raise RuntimeError(f"native C++ runner exited {p.returncode}; see runner.stderr.log")
                child_pid=None; status("running",active=0)
            state="interrupted_cleanly" if stop_requested else "complete"
            status(state,active=0); exit_code=0
        except Exception as e:
            failure=str(e); exit_code=1; status("failed",failure_reason=failure,active=0)
        finally:
            append_attempt(out,{"schema":"hex-native-selfplay-attempt-v2","attempt_id":attempt,"started_epoch":started,"ended_epoch":time.time(),"mode":"resume" if a.resume else "new","status":"interrupted_cleanly" if stop_requested and not failure else ("failed" if failure else "complete"),"exit_code":exit_code,"failure":failure})
        if failure: raise RuntimeError(failure)
    return 0

if __name__ == "__main__":
    try: raise SystemExit(main())
    except Exception as e: print(f"native selfplay lifecycle error: {e}",file=sys.stderr); raise SystemExit(1)
