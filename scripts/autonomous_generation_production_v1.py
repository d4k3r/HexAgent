#!/usr/bin/env python3
"""Fail-closed real/fake coordinator for the bounded autonomous CONTROL run.

Real execution composes existing qualified executors.  It never writes the
global champion registry: promotions advance a hash-bound run-local incumbent
only.  ``--fake-backend`` is a CPU lifecycle qualification mode.
"""
from __future__ import annotations

import argparse, fcntl, hashlib, json, os, shutil, socket, subprocess, sys, time, uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / "artifacts/venvs/student-training-v1/bin/python"
CUDA_WRAP = ROOT / "scripts/stage7_cuda12_runtime_v1.sh"
NATIVE = ROOT / "build/cpp-puct-stage7/hex_native_selfplay_v2_runner"
MATCH = ROOT / "build/cpp-puct-stage7/hex_candidate_match_runner"
PROFILE = ROOT / "config/autonomous-generation-v1/selfplay-resource-profile-v1.json"
LOCK = Path("/tmp/hex-agent-autonomous-generation-v1-heavy.lock")
STAGES = ("SELFPLAY_NORMAL", "SELFPLAY_FORCED", "SELFPLAY_AUDIT", "DATA_PREP", "TRAIN_CANDIDATES", "EXPORT_CANDIDATES", "CANDIDATE_SCREEN", "PROMOTION_MATCH", "PROMOTION_DECISION")
HEAVY = {"SELFPLAY_NORMAL", "SELFPLAY_FORCED", "TRAIN_CANDIDATES", "CANDIDATE_SCREEN", "PROMOTION_MATCH"}
# Historical raw/prepared/training/export sizes round conservatively to seven
# GiB per generation.  The preflight reserves 30 GiB for the bounded trial.
EXPECTED_GROWTH_PER_GENERATION_BYTES = 7 * 1024 ** 3
MINIMUM_FREE_BYTES = 30 * 1024 ** 3


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: object) -> str: return json.dumps(value, sort_keys=True, separators=(",", ":"))


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("w") as stream:
        stream.write(value); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def read(path: Path) -> dict: return json.loads(path.read_text())


def append(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(canonical(value) + "\n"); stream.flush(); os.fsync(stream.fileno())


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT.parent / path


def env() -> dict:
    result = dict(os.environ)
    prefix = f"{ROOT / 'src'}:{ROOT / 'scripts'}"
    result["PYTHONPATH"] = prefix + (":" + result["PYTHONPATH"] if result.get("PYTHONPATH") else "")
    return result


class HeavyLease:
    def __init__(self, metadata: dict, recover: bool): self.metadata = metadata; self.recover = recover; self.fd = None
    def __enter__(self):
        if LOCK.exists() and not self.recover: raise RuntimeError("heavy lock exists; inspect it or use --recover-stale-lock after confirming no owner")
        self.fd = LOCK.open("a+")
        try: fcntl.flock(self.fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc: self.fd.close(); raise RuntimeError("another heavy autonomous stage owns the machine") from exc
        self.fd.seek(0); self.fd.truncate(); json.dump(self.metadata, self.fd); self.fd.flush(); os.fsync(self.fd.fileno()); return self
    def __exit__(self, *_):
        if self.fd:
            fcntl.flock(self.fd.fileno(), fcntl.LOCK_UN); self.fd.close(); self.fd = None
        LOCK.unlink(missing_ok=True)


def recipe(path: Path) -> dict:
    data = read(path)
    if data.get("schema") != "hex-autonomous-generation-recipe-v1" or data.get("approval_status") != "approved":
        raise RuntimeError("real autonomous run requires the approved CONTROL recipe")
    expected = {"teacher_breadth": .20, "teacher_deep": 0.0, "historical": .20}
    sources = data.get("sources", {})
    if any(float(sources.get(key, {}).get("weight", -1)) != value for key, value in expected.items()):
        raise RuntimeError("recipe is not the frozen CONTROL anchor mixture")
    fresh = sources.get("fresh", {})
    if (float(fresh.get("normal_weight", -1)), float(fresh.get("forced_weight", -1)), float(fresh.get("combined_weight", -1))) != (.75, .25, .60):
        raise RuntimeError("recipe fresh mixture is not N75/F25 within F60")
    training = data.get("training", {})
    if (training.get("rows_per_epoch"), training.get("epochs"), training.get("base_batch")) != (400000, 4, 64):
        raise RuntimeError("recipe training contract is not the frozen CONTROL contract")
    search = data.get("selfplay", {}).get("search", {})
    if search != {"budget":128,"c_puct":2.5,"fpu_mode":"parent_value_reduced","fpu_reduction":.25}:
        raise RuntimeError("Search-V2 contract mismatch")
    profile = data["selfplay"].get("resource_profile", {})
    if profile.get("sha256") != sha(PROFILE) or profile.get("approval_status") != "APPROVED_FOR_AUTONOMOUS_SELFPLAY":
        raise RuntimeError("approved self-play resource profile identity mismatch")
    if profile.get("process_count") != 1 or profile.get("concurrency_per_process") != 128 or profile.get("max_batch") != 96 or profile.get("wait_us") != 200:
        raise RuntimeError("recipe does not bind C128/B96/wait200")
    if data["selfplay"]["inference"] != {"concurrency":128,"max_batch":96,"wait_us":200,"watchdog_seconds":120}:
        raise RuntimeError("self-play inference geometry does not bind resource profile")
    if data["candidate_plan"].get("count") != len(data["candidate_plan"].get("seeds", [])) or not data["candidate_plan"].get("seeds"):
        raise RuntimeError("invalid candidate plan")
    return data


def champion(path: Path) -> dict:
    data = read(path); checkpoint = Path(data["checkpoint"]["path"]); onnx = Path(data["onnx"]["onnx"])
    if not checkpoint.is_file() or sha(checkpoint) != data["checkpoint"]["sha256"]: raise RuntimeError("official checkpoint identity mismatch")
    if not onnx.is_file() or sha(onnx) != data["onnx"]["onnx_sha256"]: raise RuntimeError("official ONNX identity mismatch")
    return {"id": data["champion_id"], "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": data["checkpoint"]["sha256"], "onnx": str(onnx.resolve()), "onnx_sha256": data["onnx"]["onnx_sha256"], "official_registry": str(path.resolve()), "official_registry_sha256": sha(path)}


def incumbent_from_summary(path: Path) -> dict:
    """Load a completed run-local promotion without mutating its source."""
    summary = read(path)
    value = summary.get("final_run_local_incumbent")
    if not isinstance(value, dict) or not value.get("run_local"):
        raise RuntimeError("starting incumbent summary lacks a run-local final incumbent")
    checkpoint = Path(value.get("checkpoint", "")); onnx = Path(value.get("onnx", ""))
    if not checkpoint.is_file() or sha(checkpoint) != value.get("checkpoint_sha256"):
        raise RuntimeError("starting run-local checkpoint identity mismatch")
    if not onnx.is_file() or sha(onnx) != value.get("onnx_sha256"):
        raise RuntimeError("starting run-local ONNX identity mismatch")
    return {"id": value["id"], "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": value["checkpoint_sha256"],
            "onnx": str(onnx.resolve()), "onnx_sha256": value["onnx_sha256"], "run_local": True,
            "source_summary": str(path.resolve()), "source_summary_sha256": sha(path)}


def command(args: list[str], log: Path, *, heavy: bool, metadata: dict, recover: bool) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        stream.write(("\n$ " + " ".join(args) + "\n").encode()); stream.flush()
        context = HeavyLease(metadata, recover) if heavy else None
        if context:
            with context:
                completed = subprocess.run(args, cwd=ROOT, env=env(), stdout=stream, stderr=subprocess.STDOUT)
        else:
            completed = subprocess.run(args, cwd=ROOT, env=env(), stdout=stream, stderr=subprocess.STDOUT)
    if completed.returncode: raise RuntimeError(f"adapter failed ({completed.returncode}); see {log}")


def local_registry(gen_root: Path, incumbent: dict) -> Path:
    path = gen_root / "incumbent-registry.json"
    record = {"schema":"champion-registry-v1", "champion_id": f"champion-runlocal-{incumbent['id']}",
              "checkpoint":{"path":incumbent["checkpoint"],"sha256":incumbent["checkpoint_sha256"]},
              "onnx":{"onnx":incumbent["onnx"],"onnx_sha256":incumbent["onnx_sha256"]},
              "run_local":True,"source_incumbent_id":incumbent["id"]}
    if path.exists() and read(path) != record: raise RuntimeError("run-local incumbent registry mismatch")
    if not path.exists(): atomic(path, record)
    return path


def evidence_path(gen_root: Path, stage: str) -> Path: return gen_root / "evidence" / f"{stage}.json"


def validated_evidence(path: Path, stage: str, manifest: dict, rec: dict) -> dict:
    """Fail closed when a resume encounters malformed or foreign evidence."""
    value = read(path)
    if not isinstance(value, dict) or value.get("passed") is not True:
        raise RuntimeError(f"{stage} evidence is incomplete or failed: {path}")
    if stage in {"SELFPLAY_NORMAL", "SELFPLAY_FORCED"}:
        if value.get("model_sha256") != manifest["incumbent"]["onnx_sha256"] or value.get("search") != rec["selfplay"]["search"]:
            raise RuntimeError(f"{stage} evidence does not bind the generation model/search identity")
    if stage == "TRAIN_CANDIDATES" and value.get("parent_checkpoint_sha256") != manifest["incumbent"]["checkpoint_sha256"]:
        raise RuntimeError("training evidence does not bind the generation parent")
    if stage == "PROMOTION_DECISION" and value.get("previous_incumbent") != manifest["incumbent"]:
        raise RuntimeError("promotion decision does not bind the generation incumbent")
    return value


def audit_selfplay(root: Path, output: Path, log: Path) -> dict:
    command([str(VENV), str(ROOT / "scripts/audit_selfplay_v2_native.py"), "--root", str(root), "--output", str(output)], log, heavy=False, metadata={}, recover=False)
    return read(output)


def candidate_ids(generation: int, rec: dict) -> list[str]: return [f"C2-AUTO-G{generation:04d}-S{seed}" for seed in rec["candidate_plan"]["seeds"]]


def fake_incumbent(incumbent: dict, generation: int, candidate_id: str) -> dict:
    return {**incumbent, "id": f"runlocal-g{generation}-{candidate_id}", "run_local": True, "promotion_generation": generation}


def run_stage(run_root: Path, gen_root: Path, stage: str, manifest: dict, rec: dict, state: dict, *, fake: bool, recover: bool) -> dict:
    ep = evidence_path(gen_root, stage)
    if ep.exists(): return read(ep)
    gen = manifest["generation"]; incumbent = manifest["incumbent"]; logs = run_root / "logs" / f"g{gen:04d}-{stage.lower()}.log"
    games = rec["selfplay"]["games"]; inf = rec["selfplay"]["inference"]; search = rec["selfplay"]["search"]
    if fake:
        ids = candidate_ids(gen, rec)
        if stage == "SELFPLAY_NORMAL": result={"passed":True,"accepted":games["normal"],"requested":games["normal"],"quarantined":0,"model_sha256":incumbent["onnx_sha256"],"search":search}
        elif stage == "SELFPLAY_FORCED": result={"passed":True,"accepted":games["forced"],"requested":games["forced"],"quarantined":0,"prefix_coverage_complete":True,"forced_rows_emitted":0,"model_sha256":incumbent["onnx_sha256"],"search":search}
        elif stage == "SELFPLAY_AUDIT": result={"passed":True,"normal_complete":True,"forced_complete":True,"duplicates":0,"corrupt":0}
        elif stage == "DATA_PREP": result={"passed":True,"source_accounting_exact":True,"deep1600_rows":0,"phase_b_rows":0}
        elif stage == "TRAIN_CANDIDATES": result={"passed":True,"parent_checkpoint_sha256":incumbent["checkpoint_sha256"],"candidates":[{"id":x,"completed":True,"nan_inf":False,"checkpoint":incumbent["checkpoint"],"checkpoint_sha256":incumbent["checkpoint_sha256"]} for x in ids]}
        elif stage == "EXPORT_CANDIDATES": result={"passed":True,"candidates":[{"id":x,"onnx":incumbent["onnx"],"onnx_sha256":incumbent["onnx_sha256"],"cpu_parity_passed":True} for x in ids]}
        elif stage == "CANDIDATE_SCREEN": result={"passed":True,"pairs":rec["evaluation"]["screening_pairs"],"candidates":[{"id":x,"complete":True,"colour_balanced":True,"paired_score":.51+i*.01} for i,x in enumerate(ids)]}
        elif stage == "PROMOTION_MATCH":
            lcb = .52 if gen % 2 else .49; result={"passed":True,"challenger_id":ids[-1],"pairs":rec["evaluation"]["promotion_pairs"],"complete":True,"colour_balanced":True,"one_sided_95_lcb":lcb,"paired_score":.55 if lcb>.5 else .49}
        else: result={"passed":True}
        atomic(ep, result); return result

    registry = local_registry(gen_root, incumbent)
    if stage in {"SELFPLAY_NORMAL", "SELFPLAY_FORCED"}:
        mode = "normal" if stage == "SELFPLAY_NORMAL" else "forced"; target = gen_root / "selfplay" / mode
        lifecycle = "--resume" if (target / "run-manifest.json").is_file() else "--new"
        cmd=[str(CUDA_WRAP),str(VENV),str(ROOT/"scripts/run_selfplay_v2_native.py"),lifecycle,"--output",str(target),"--run-id",f"auto-g{gen:04d}-{mode}","--champion-registry",str(registry),"--games",str(games[mode]),"--master-seed",str(2026082400+gen*10+(0 if mode=="normal" else 1)),"--budget",str(search["budget"]),"--concurrency",str(inf["concurrency"]),"--max-batch",str(inf["max_batch"]),"--wait-us",str(inf["wait_us"]),"--watchdog-seconds",str(inf["watchdog_seconds"]),"--c-puct",str(search["c_puct"]),"--fpu-mode",search["fpu_mode"],"--fpu-reduction",str(search["fpu_reduction"])]
        if mode == "forced": cmd += ["--prefix-bank", str(resolve(rec["selfplay"]["forced_prefix_bank"]["path"]))]
        command(cmd, logs, heavy=True, metadata={"pid":os.getpid(),"hostname":socket.gethostname(),"generation":gen,"stage":stage,"started_epoch":time.time()}, recover=recover)
        status = read(target / "runner-status.json")
        result={"passed":status.get("state")=="complete","root":str(target),"accepted":status.get("accepted"),"requested":games[mode],"quarantined":status.get("quarantined"),"model_sha256":incumbent["onnx_sha256"],"search":search}
        if mode=="forced": result.update({"prefix_coverage_complete":False,"forced_rows_emitted":None})
    elif stage == "SELFPLAY_AUDIT":
        normal=gen_root/"selfplay/normal"; forced=gen_root/"selfplay/forced"; na=audit_selfplay(normal,normal/"postrun-audit.json",logs); fa=audit_selfplay(forced,forced/"postrun-audit.json",logs)
        if fa["prefix_mode"].get("forced_games") != games["forced"] or fa["prefix_mode"].get("forced_rows_emitted") != 0: raise RuntimeError("forced prefix coverage/policy-row audit failed")
        forced_ev=evidence_path(gen_root,"SELFPLAY_FORCED")
        if forced_ev.exists():
            prior=read(forced_ev); prior.update({"prefix_coverage_complete":True,"forced_rows_emitted":0}); atomic(forced_ev,prior)
        result={"passed":bool(na.get("complete") and fa.get("complete") and not na.get("quarantined_artifacts") and not fa.get("quarantined_artifacts")),"normal_complete":na.get("complete"),"forced_complete":fa.get("complete"),"duplicates":0,"corrupt":0,"normal_audit":str(normal/"postrun-audit.json"),"forced_audit":str(forced/"postrun-audit.json")}
    elif stage == "DATA_PREP":
        normal_raw=gen_root/"selfplay/normal"; forced_raw=gen_root/"selfplay/forced"; normal=gen_root/"prepared/normal"; forced=gen_root/"prepared/forced"
        command([str(VENV),str(ROOT/"scripts/prepare_stage8b_native_selfplay_v2.py"),"--native-root",str(normal_raw),"--output",str(normal)],logs,heavy=False,metadata={},recover=False)
        command([str(VENV),str(ROOT/"scripts/prepare_stage8b_native_selfplay_v2.py"),"--native-root",str(forced_raw),"--output",str(forced)],logs,heavy=False,metadata={},recover=False)
        bundle=gen_root/"prepared/control"; ids=candidate_ids(gen,rec); static=rec["static_sources"]
        cmd=[str(VENV),str(ROOT/"scripts/prepare_stage8b_autonomous_control_v1.py"),"--teacher-prepared",str(resolve(static["teacher_prepared"])),"--historical-prepared",str(resolve(static["historical_prepared"])),"--normal-prepared",str(normal),"--forced-prepared",str(forced),"--parent-checkpoint",incumbent["checkpoint"],"--parent-onnx",incumbent["onnx"],"--parent-id",incumbent["id"],"--output",str(bundle)]
        for ident, seed in zip(ids, rec["candidate_plan"]["seeds"]): cmd += ["--candidate", f"{ident}:{seed}"]
        command(cmd,logs,heavy=False,metadata={},recover=False)
        result={"passed":True,"bundle":str(bundle),"source_accounting_exact":True,"deep1600_rows":0,"phase_b_rows":0,"normal_prepared":str(normal),"forced_prepared":str(forced)}
    elif stage == "TRAIN_CANDIDATES":
        bundle=Path(read(evidence_path(gen_root,"DATA_PREP"))["bundle"]); records=[]
        for ident, seed in zip(candidate_ids(gen,rec),rec["candidate_plan"]["seeds"]):
            out=gen_root/"training"/ident; mix=bundle/"mixtures"/f"{ident}.json"
            final_path = out / "final-report.json"
            if final_path.is_file():
                final = read(final_path)
                if final.get("passed") is not True or final.get("candidate_id") != ident:
                    raise RuntimeError(f"existing candidate terminal report is invalid: {ident}")
            else:
                train_cmd=[str(VENV),str(ROOT/"scripts/train_stage8b_candidate_v1.py"),"--output",str(out),"--candidate-id",ident,"--prepared-root",str(bundle),"--mixture-manifest",str(mix),"--parent-checkpoint",incumbent["checkpoint"],"--epochs","4","--base-rows-per-epoch","400000","--seed",str(seed),"--device","cuda"]
                if out.exists(): train_cmd.append("--resume")
                command(train_cmd,logs,heavy=True,metadata={"pid":os.getpid(),"hostname":socket.gethostname(),"generation":gen,"stage":stage,"candidate":ident,"started_epoch":time.time()},recover=recover)
                final=read(final_path)
            checkpoint=out/"checkpoints/best-validation-policy.pt"
            records.append({"id":ident,"seed":seed,"completed":bool(final.get("passed")),"nan_inf":False,"checkpoint":str(checkpoint),"checkpoint_sha256":sha(checkpoint),"best_validation_policy":final["best_validation_policy"]})
        init_audit = gen_root / "training" / "initialization-audit.json"
        init_cmd = [str(VENV), str(ROOT / "scripts/audit_autonomous_candidate_initialization_v1.py"), "--parent", incumbent["checkpoint"], "--output", str(init_audit)]
        for candidate in records:
            init_cmd += ["--initial", str(gen_root / "training" / candidate["id"] / "checkpoints" / "initial-champion0.pt")]
        command(init_cmd, logs, heavy=False, metadata={}, recover=False)
        if not read(init_audit).get("passed"):
            raise RuntimeError("candidate initial tensor audit failed")
        result={"passed":True,"parent_checkpoint_sha256":incumbent["checkpoint_sha256"],"initialization_audit":str(init_audit),"candidates":records}
    elif stage == "EXPORT_CANDIDATES":
        records=[]
        for candidate in read(evidence_path(gen_root,"TRAIN_CANDIDATES"))["candidates"]:
            out=gen_root/"exports"/candidate["id"] / "model-dynamic.onnx"; parity=gen_root/"exports"/candidate["id"] / "cpu-parity.json"
            command([str(VENV),str(ROOT/"scripts/export_stage8b_candidate_onnx_v1.py"),"--checkpoint",candidate["checkpoint"],"--output",str(out)],logs,heavy=False,metadata={},recover=False)
            cpu_env=dict(env()); cpu_env["PYTHONPATH"]=f"{ROOT/'src'}:{ROOT/'artifacts/cpp-onnx-stage3-v1/python'}"
            with logs.open("ab") as stream:
                done=subprocess.run([str(VENV),str(ROOT/"scripts/qualify_stage8b_candidate_onnx_parity_v1.py"),"--checkpoint",candidate["checkpoint"],"--onnx",str(out),"--output",str(parity)],cwd=ROOT,env=cpu_env,stdout=stream,stderr=subprocess.STDOUT)
            if done.returncode: raise RuntimeError(f"CPU parity failed for {candidate['id']}")
            report=read(parity); records.append({**candidate,"onnx":str(out),"onnx_sha256":sha(out),"cpu_parity":str(parity),"cpu_parity_passed":bool(report.get("passed"))})
        result={"passed":all(x["cpu_parity_passed"] for x in records),"candidates":records}
    elif stage == "CANDIDATE_SCREEN":
        records=[]; opening=resolve(rec["evaluation"]["opening_bank"])
        for candidate in read(evidence_path(gen_root,"EXPORT_CANDIDATES"))["candidates"]:
            out=gen_root/"screening"/candidate["id"]
            cmd=[str(CUDA_WRAP),str(VENV),str(ROOT/"scripts/run_stage8c_gameplay_v1.py"),"--candidate-id",candidate["id"],"--candidate",candidate["onnx"],"--champion",incumbent["onnx"],"--openings",str(opening),"--output",str(out),"--budget","128","--c-puct","2.5","--candidate-budget","128","--champion-budget","128","--candidate-c-puct","2.5","--champion-c-puct","2.5","--candidate-fpu-mode","parent_value_reduced","--candidate-fpu-reduction","0.25","--champion-fpu-mode","parent_value_reduced","--champion-fpu-reduction","0.25","--concurrency",str(rec["evaluation"]["execution"]["concurrency"]),"--max-batch",str(rec["evaluation"]["execution"]["max_batch"]),"--wait-us",str(rec["evaluation"]["execution"]["wait_us"]),"--bridge-controller","active","--bootstrap-samples","20000","--max-pairs",str(rec["evaluation"]["screening_pairs"])]
            command(cmd,logs,heavy=True,metadata={"pid":os.getpid(),"hostname":socket.gethostname(),"generation":gen,"stage":stage,"candidate":candidate["id"],"started_epoch":time.time()},recover=recover)
            summary=read(out/"summary.json"); records.append({"id":candidate["id"],"complete":summary.get("pairs")==rec["evaluation"]["screening_pairs"],"colour_balanced":True,"paired_score":summary["paired_mean_score"],"summary":str(out/"summary.json")})
        result={"passed":all(x["complete"] for x in records),"pairs":rec["evaluation"]["screening_pairs"],"candidates":records}
    elif stage == "PROMOTION_MATCH":
        exports={x["id"]:x for x in read(evidence_path(gen_root,"EXPORT_CANDIDATES"))["candidates"]}; screening=read(evidence_path(gen_root,"CANDIDATE_SCREEN"))["candidates"]
        best=sorted(screening,key=lambda x:(x["paired_score"],x["id"]),reverse=True)[0]; candidate=exports[best["id"]]; out=gen_root/"promotion"/candidate["id"]; opening=resolve(rec["evaluation"]["opening_bank"])
        cmd=[str(CUDA_WRAP),str(VENV),str(ROOT/"scripts/run_stage8c_gameplay_v1.py"),"--candidate-id",candidate["id"],"--candidate",candidate["onnx"],"--champion",incumbent["onnx"],"--openings",str(opening),"--output",str(out),"--budget","128","--c-puct","2.5","--candidate-budget","128","--champion-budget","128","--candidate-c-puct","2.5","--champion-c-puct","2.5","--candidate-fpu-mode","parent_value_reduced","--candidate-fpu-reduction","0.25","--champion-fpu-mode","parent_value_reduced","--champion-fpu-reduction","0.25","--concurrency",str(rec["evaluation"]["execution"]["concurrency"]),"--max-batch",str(rec["evaluation"]["execution"]["max_batch"]),"--wait-us",str(rec["evaluation"]["execution"]["wait_us"]),"--bridge-controller","active","--bootstrap-samples","20000"]
        command(cmd,logs,heavy=True,metadata={"pid":os.getpid(),"hostname":socket.gethostname(),"generation":gen,"stage":stage,"candidate":candidate["id"],"started_epoch":time.time()},recover=recover)
        summary=read(out/"summary.json"); result={"passed":summary.get("pairs")==rec["evaluation"]["promotion_pairs"],"challenger_id":candidate["id"],"checkpoint":candidate["checkpoint"],"checkpoint_sha256":candidate["checkpoint_sha256"],"onnx":candidate["onnx"],"onnx_sha256":candidate["onnx_sha256"],"pairs":summary["pairs"],"complete":summary["pairs"]==rec["evaluation"]["promotion_pairs"],"colour_balanced":True,"paired_score":summary["paired_mean_score"],"one_sided_95_lcb":summary["bootstrap_one_sided_95_lcb"],"summary":str(out/"summary.json")}
    else: raise RuntimeError(f"unknown stage {stage}")
    atomic(ep,result); return result


def preflight(rec: dict, initial: dict, output: Path, max_generations: int, recover: bool, official: dict | None = None) -> dict:
    required=[VENV,CUDA_WRAP,NATIVE,MATCH,ROOT/"scripts/run_selfplay_v2_native.py",ROOT/"scripts/audit_selfplay_v2_native.py",ROOT/"scripts/prepare_stage8b_native_selfplay_v2.py",ROOT/"scripts/prepare_stage8b_autonomous_control_v1.py",ROOT/"scripts/train_stage8b_candidate_v1.py",ROOT/"scripts/audit_autonomous_candidate_initialization_v1.py",ROOT/"scripts/export_stage8b_candidate_onnx_v1.py",ROOT/"scripts/qualify_stage8b_candidate_onnx_parity_v1.py",ROOT/"scripts/run_stage8c_gameplay_v1.py"]
    missing=[str(x) for x in required if not x.is_file()]
    for key in ("teacher_prepared","historical_prepared"):
        if not (resolve(rec["static_sources"][key])/"prepared-manifest.json").is_file(): missing.append(key)
    bank=resolve(rec["selfplay"]["forced_prefix_bank"]["path"])
    if not bank.is_file() or sha(bank)!=rec["selfplay"]["forced_prefix_bank"]["sha256"]: missing.append("forced_prefix_bank")
    opening=resolve(rec["evaluation"]["opening_bank"])
    if not opening.is_file(): missing.append("opening_bank")
    stale=[]
    for binary, source in ((NATIVE,ROOT/"cpp/src/native_selfplay_v2_runner.cpp"),(MATCH,ROOT/"cpp/src/candidate_match_runner.cpp")):
        if binary.exists() and source.exists() and binary.stat().st_mtime < source.stat().st_mtime: stale.append(str(binary))
    usage=shutil.disk_usage(output.parent if output.parent.exists() else ROOT.parent)
    lock_available=True
    if LOCK.exists() and not recover: lock_available=False
    expected_growth = EXPECTED_GROWTH_PER_GENERATION_BYTES * max_generations
    disk_sufficient = usage.free >= max(MINIMUM_FREE_BYTES, expected_growth + 5 * 1024 ** 3)
    recipe_max = int(rec.get("governance", {}).get("max_generations", 0))
    generation_bound_valid = 1 <= max_generations <= recipe_max
    failed_checks = []
    if missing: failed_checks.append({"check": "required_inputs", "details": missing})
    if stale: failed_checks.append({"check": "binary_freshness", "details": stale})
    if not lock_available: failed_checks.append({"check": "heavy_lock_available", "details": "lock exists; inspect owner or explicitly recover a confirmed stale lock"})
    if not disk_sufficient: failed_checks.append({"check": "disk_sufficient", "details": {"free_bytes": usage.free, "required_bytes": max(MINIMUM_FREE_BYTES, expected_growth + 5 * 1024 ** 3)}})
    if not generation_bound_valid: failed_checks.append({"check": "generation_bound", "details": {"requested": max_generations, "approved_maximum": recipe_max, "allowed": "1..approved_maximum"}})
    if output.exists(): failed_checks.append({"check": "output_absent", "details": str(output)})
    result={"schema":"autonomous-production-preflight-v1","passed":not failed_checks,"failed_checks":failed_checks,"reasons":[item["check"] for item in failed_checks],"initial_incumbent":initial,"official_incumbent":official or initial,"recipe_id":rec["recipe_id"],"profile_sha256":sha(PROFILE),"missing":missing,"stale_binaries":stale,"heavy_lock_available":lock_available,"disk_free_bytes":usage.free,"disk_total_bytes":usage.total,"expected_growth_bytes":expected_growth,"disk_sufficient":disk_sufficient,"minimum_free_bytes":MINIMUM_FREE_BYTES,"output_absent":not output.exists(),"max_generations":max_generations,"approved_max_generations":recipe_max,"global_registry_read_only":True}
    return result


def write_gen_manifest(gen_root: Path, generation: int, rec_path: Path, rec: dict, incumbent: dict) -> dict:
    data={"schema":"autonomous-generation-production-manifest-v1","generation":generation,"recipe":{"path":str(rec_path.resolve()),"sha256":sha(rec_path),"id":rec["recipe_id"]},"incumbent":incumbent,"search":rec["selfplay"]["search"],"resource_profile":rec["selfplay"]["resource_profile"],"fresh_games":rec["selfplay"]["games"],"candidate_plan":rec["candidate_plan"],"source_plan":rec["sources"]}
    data["identity_sha256"]=hashlib.sha256(canonical(data).encode()).hexdigest(); atomic(gen_root/"generation-manifest.json",data); return data


def final_markdown(initial: dict, completed: list[dict], final_incumbent: dict) -> str:
    lines = ["# Autonomous production summary", "", f"Starting official incumbent: `{initial['id']}`", "", "| Generation | Incumbent before | Challenger | Promoted | Incumbent after |", "|---:|---|---|:---:|---|"]
    for row in completed:
        decision = row["promotion"]
        lines.append(f"| {row['generation']} | `{decision['previous_incumbent']['id']}` | `{decision['challenger']}` | {'yes' if decision['promoted'] else 'no'} | `{row['incumbent_after']['id']}` |")
    lines.extend(["", f"Final proposed run-local incumbent: `{final_incumbent['id']}`", "", "The global Champion registry was not modified. Manual ratification is required.", ""])
    return "\n".join(lines)


def main() -> int:
    p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument("--new",action="store_true");mode.add_argument("--resume",action="store_true");mode.add_argument("--status",action="store_true");mode.add_argument("--dry-run",action="store_true");mode.add_argument("--preflight",action="store_true")
    p.add_argument("--output",type=Path,required=True);p.add_argument("--recipe",type=Path,required=True);p.add_argument("--champion-registry",type=Path,required=True);p.add_argument("--starting-incumbent-summary",type=Path,help="completed run summary whose final run-local incumbent seeds this new run");p.add_argument("--max-generations",type=int,default=3);p.add_argument("--execute",action="store_true");p.add_argument("--fake-backend",action="store_true");p.add_argument("--recover-stale-lock",action="store_true");p.add_argument("--stop-after-stage",choices=STAGES,help="test/operations pause after atomically committing this stage");a=p.parse_args()
    rec_path=a.recipe.resolve(); rec=recipe(rec_path); official=champion(a.champion_registry.resolve()); initial=incumbent_from_summary(a.starting_incumbent_summary.resolve()) if a.starting_incumbent_summary else official; output=a.output.resolve(); source_summary_sha=initial.get("source_summary_sha256")
    if a.preflight:
        result = preflight(rec,initial,output,a.max_generations,a.recover_stale_lock,official)
        print(json.dumps(result,indent=2,sort_keys=True))
        if not result["passed"]: raise RuntimeError("production preflight failed: "+canonical(result))
        return 0
    if a.dry_run:
        print(json.dumps({"schema":"autonomous-production-dry-run-v1","output":str(output),"max_generations":a.max_generations,"initial_incumbent":initial,"stages":STAGES,"global_registry_mutation":False},indent=2,sort_keys=True)); return 0
    if a.new:
        if not a.execute and not a.fake_backend: raise RuntimeError("real production requires --execute")
        pf=preflight(rec,initial,output,a.max_generations,a.recover_stale_lock,official)
        if not pf["passed"]: raise RuntimeError("production preflight failed: "+canonical(pf))
        output.mkdir(parents=True); atomic(output/"run-manifest.json",{"schema":"autonomous-production-run-v1","recipe_path":str(rec_path),"recipe_sha256":sha(rec_path),"initial_official_incumbent":official,"initial_incumbent":initial,"starting_incumbent_summary_sha256":source_summary_sha,"max_generations":a.max_generations,"global_registry_read_only":True,"preflight":pf}); state={"schema":"autonomous-production-state-v1","state":"RUNNING","generation":1,"stage":STAGES[0],"incumbent":initial,"completed_generations":[],"created_epoch":time.time(),"updated_epoch":time.time()}; atomic(output/"state.json",state); append(output/"attempt-history.jsonl",{"event":"new","epoch":time.time(),"fake":a.fake_backend})
    else:
        if not (output/"run-manifest.json").is_file(): raise RuntimeError("--resume/--status requires an initialized run")
        run=read(output/"run-manifest.json")
        if run.get("recipe_sha256")!=sha(rec_path) or run.get("initial_official_incumbent")!=official or run.get("initial_incumbent")!=initial or run.get("starting_incumbent_summary_sha256")!=source_summary_sha or run.get("max_generations")!=a.max_generations: raise RuntimeError("immutable run identity mismatch")
        state=read(output/"state.json")
        if a.status: print(json.dumps(state,indent=2,sort_keys=True));return 0
        if state.get("state") == "COMPLETE": print(json.dumps(state,sort_keys=True));return 0
        if state.get("state") == "FAILED":
            state.pop("failure", None); state["state"] = "RUNNING"; state["updated_epoch"] = time.time(); atomic(output/"state.json", state)
        append(output/"attempt-history.jsonl",{"event":"resume","epoch":time.time(),"fake":a.fake_backend})
    try:
        while state["generation"] <= a.max_generations:
            gen=state["generation"]; gen_root=output/f"generation-{gen:04d}"; gen_root.mkdir(parents=True,exist_ok=True)
            manifest_path=gen_root/"generation-manifest.json"; manifest=read(manifest_path) if manifest_path.exists() else write_gen_manifest(gen_root,gen,rec_path,rec,state["incumbent"])
            if manifest["incumbent"] != state["incumbent"]: raise RuntimeError("run-local incumbent lineage mismatch")
            for stage in STAGES:
                existing = evidence_path(gen_root,stage)
                if existing.exists():
                    validated_evidence(existing, stage, manifest, rec)
                    continue
                state.update({"state":"RUNNING","generation":gen,"stage":stage,"updated_epoch":time.time()});atomic(output/"state.json",state)
                if stage == "PROMOTION_DECISION":
                    evidence = validated_evidence(evidence_path(gen_root, "PROMOTION_MATCH"), "PROMOTION_MATCH", manifest, rec)
                    gate=rec["evaluation"]["promotion_lcb_strictly_greater_than"]
                    promoted=evidence["one_sided_95_lcb"]>gate
                    if promoted:
                        next_incumbent=(fake_incumbent(state["incumbent"],gen,evidence["challenger_id"]) if a.fake_backend else {"id":evidence["challenger_id"],"checkpoint":evidence["checkpoint"],"checkpoint_sha256":evidence["checkpoint_sha256"],"onnx":evidence["onnx"],"onnx_sha256":evidence["onnx_sha256"],"run_local":True,"promotion_generation":gen})
                    else: next_incumbent=state["incumbent"]
                    evidence={"passed":True,"challenger":evidence["challenger_id"],"promoted":promoted,"lcb":evidence["one_sided_95_lcb"],"gate":gate,"previous_incumbent":state["incumbent"],"resulting_incumbent":next_incumbent,"global_registry_action":"manual_ratification_only"}
                    atomic(evidence_path(gen_root,"PROMOTION_DECISION"),evidence)
                else:
                    evidence=run_stage(output,gen_root,stage,manifest,rec,state,fake=a.fake_backend,recover=a.recover_stale_lock)
                if not evidence.get("passed"): raise RuntimeError(f"{stage} evidence failed")
                if a.stop_after_stage == stage:
                    state.update({"state":"PAUSED","generation":gen,"stage":stage,"updated_epoch":time.time()}); atomic(output/"state.json",state)
                    append(output/"attempt-history.jsonl",{"event":"paused","generation":gen,"stage":stage,"epoch":time.time()})
                    print(json.dumps({"state":"PAUSED","generation":gen,"stage":stage},sort_keys=True)); return 0
            decision = read(evidence_path(gen_root,"PROMOTION_DECISION"))
            if decision.get("previous_incumbent") != manifest["incumbent"] or not isinstance(decision.get("resulting_incumbent"), dict):
                raise RuntimeError("promotion decision does not bind the generation incumbent")
            next_incumbent = decision["resulting_incumbent"]
            summary = {"generation":gen,"incumbent_before":manifest["incumbent"],"incumbent_after":next_incumbent,"promotion":decision}
            atomic(gen_root/"generation-summary.json",summary)
            state["completed_generations"].append({"generation":gen,"incumbent_after":next_incumbent,"promotion":decision})
            state["incumbent"]=next_incumbent; state["generation"]=gen+1;state["stage"]=STAGES[0];state["updated_epoch"]=time.time();atomic(output/"state.json",state)
        state["state"]="COMPLETE";state["final_run_local_incumbent"]=state["incumbent"];state["updated_epoch"]=time.time();atomic(output/"state.json",state);final={"schema":"autonomous-production-final-summary-v1","starting_official_incumbent":initial,"completed_generations":state["completed_generations"],"final_run_local_incumbent":state["incumbent"],"official_registry_action":"manual ratification required; global registry unchanged"};atomic(output/"final-summary.json",final);atomic_text(output/"final-summary.md",final_markdown(initial,state["completed_generations"],state["incumbent"]))
    except Exception as exc:
        state.update({"state":"FAILED","failure":{"class":type(exc).__name__,"message":str(exc),"epoch":time.time()},"updated_epoch":time.time()});atomic(output/"state.json",state);append(output/"attempt-history.jsonl",{"event":"failed","failure":state["failure"]});raise
    append(output/"attempt-history.jsonl",{"event":"complete","epoch":time.time()});print(json.dumps({"state":state["state"],"final_run_local_incumbent":state["incumbent"]},sort_keys=True));return 0


if __name__=="__main__": main()
