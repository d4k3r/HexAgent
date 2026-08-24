#!/usr/bin/env python3
"""Durable, fail-closed coordinator for a configured Hex generation.

This is deliberately a thin orchestration layer.  It owns generation identity,
state, locks, evidence gates and reports; qualified self-play/training/match
tools remain the executors.  Real execution requires explicit --execute.
"""
from __future__ import annotations
import argparse, fcntl, hashlib, json, os, socket, subprocess, sys, time, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("PENDING", "SELFPLAY_NORMAL", "SELFPLAY_FORCED", "SELFPLAY_AUDIT", "DATA_PREP", "TRAIN_CANDIDATES", "EXPORT_CANDIDATES", "CANDIDATE_SCREEN", "PROMOTION_MATCH", "PROMOTION_DECISION", "GENERATION_COMPLETE")
HEAVY = {"SELFPLAY_NORMAL", "SELFPLAY_FORCED", "TRAIN_CANDIDATES", "CANDIDATE_SCREEN", "PROMOTION_MATCH"}
TERMINAL = {"FAILED", "BLOCKED", "PLATEAU_STOP", "MAX_GENERATIONS_REACHED", "GENERATION_COMPLETE"}
DEFAULT_HEAVY_LOCK = Path("/tmp/hex-agent-autonomous-generation-v1-heavy.lock")

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def canonical(value: object) -> str: return json.dumps(value,sort_keys=True,separators=(",",":"))
def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(path.name+".partial")
    with tmp.open("w") as f: json.dump(value,f,indent=2,sort_keys=True);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def read(path: Path) -> dict: return json.loads(path.read_text())
def append_event(root: Path, value: dict) -> None:
    """Append-only attempt/event history; it is intentionally not identity data."""
    p=root/"attempt-history.jsonl"; p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("a") as f:
        f.write(canonical(value)+"\n"); f.flush(); os.fsync(f.fileno())

class Lease:
    """Local single-machine heavy-stage lease with explicit stale recovery."""
    def __init__(self, root: Path, metadata: dict, recover: bool=False, lock_path: Path|None=None): self.path=lock_path or Path(os.environ.get("HEX_AUTONOMOUS_HEAVY_LOCK",str(DEFAULT_HEAVY_LOCK)));self.metadata={**metadata,"generation_root":str(root)};self.recover=recover;self.fd=None
    def __enter__(self):
        if self.path.exists() and not self.recover: raise RuntimeError("heavy-resource lock exists; inspect it or use --recover-stale-lock after confirming no live owner")
        self.fd=self.path.open("a+")
        try: fcntl.flock(self.fd.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: self.fd.close();self.fd=None;raise RuntimeError("another local autonomous heavy stage owns the resource") from e
        self.fd.seek(0);self.fd.truncate();json.dump(self.metadata,self.fd);self.fd.flush();os.fsync(self.fd.fileno());return self
    def __exit__(self,*_):
        if self.fd: fcntl.flock(self.fd.fileno(),fcntl.LOCK_UN);self.fd.close()
        self.path.unlink(missing_ok=True)

def recipe(path: Path) -> dict:
    x=read(path)
    if x.get("schema")!="hex-autonomous-generation-recipe-v1": raise RuntimeError("wrong autonomous recipe schema")
    cp=x.get("candidate_plan",{})
    if not isinstance(x.get("recipe_id"),str) or not cp.get("seeds") or cp.get("count")!=len(cp["seeds"]) or len(set(cp["seeds"]))!=len(cp["seeds"]): raise RuntimeError("recipe lacks a unique, exact candidate plan")
    s=x.get("selfplay",{});search=s.get("search",{})
    if any(search.get(k) is None for k in ("budget","c_puct","fpu_mode","fpu_reduction")) or search["fpu_mode"] not in {"zero","parent_value_reduced"}: raise RuntimeError("recipe lacks valid Search contract")
    inference=s.get("inference",{}); profile=s.get("resource_profile",{})
    required_resource=("process_count","concurrency_per_process","max_batch","wait_us")
    if not isinstance(profile,dict) or profile.get("schema")!="hex-native-selfplay-resource-profile-v1" or profile.get("profile_id")!="selfplay-resource-profile-v1" or profile.get("approval_status")!="APPROVED_FOR_AUTONOMOUS_SELFPLAY":
        raise RuntimeError("recipe lacks the approved self-play resource profile")
    if any(profile.get(k) is None for k in required_resource): raise RuntimeError("approved resource profile is incomplete")
    if profile["process_count"] != 1 or profile["concurrency_per_process"] != inference.get("concurrency") or profile["max_batch"] != inference.get("max_batch") or profile["wait_us"] != inference.get("wait_us"):
        raise RuntimeError("recipe inference geometry does not match approved resource profile")
    profile_path=profile.get("path")
    if profile_path:
        resolved=Path(profile_path) if Path(profile_path).is_absolute() else ROOT.parent / profile_path
        if not resolved.is_file() or (profile.get("sha256") and sha(resolved)!=profile["sha256"]): raise RuntimeError("approved resource profile file identity mismatch")
    fresh=x.get("sources",{}).get("fresh",{});
    if abs(float(fresh.get("normal_weight",0))+float(fresh.get("forced_weight",0))-1)>1e-9: raise RuntimeError("fresh weights must sum to one")
    bank=s.get("forced_prefix_bank",{})
    if not bank.get("path") or not bank.get("sha256") or not x.get("evaluation") or not x.get("governance"): raise RuntimeError("recipe lacks required lifecycle contract")
    return x
def champion(path: Path) -> dict:
    x=read(path);o=x.get("onnx",{});c=x.get("checkpoint",{})
    if not x.get("champion_id") or not o.get("onnx_sha256") or not c.get("sha256"): raise RuntimeError("invalid champion registry record")
    model=Path(o["onnx"])
    if not model.is_file() or sha(model)!=o["onnx_sha256"]: raise RuntimeError("champion ONNX identity mismatch")
    if c.get("path"):
        checkpoint=Path(c["path"])
        if not checkpoint.is_file() or sha(checkpoint)!=c["sha256"]: raise RuntimeError("champion checkpoint identity mismatch")
    return {"champion_id":x["champion_id"],"registry_path":str(path.resolve()),"registry_sha256":sha(path),"checkpoint_sha256":c["sha256"],"onnx_path":str(model.resolve()),"onnx_sha256":o["onnx_sha256"]}
def verify_real_assets(rec: dict) -> None:
    """Cheap preflight only; GPU/model execution belongs to qualified executors."""
    raw=Path(rec["selfplay"]["forced_prefix_bank"]["path"])
    bank=raw if raw.is_absolute() else ROOT.parent/raw
    if not bank.is_file() or sha(bank)!=rec["selfplay"]["forced_prefix_bank"]["sha256"]:
        raise RuntimeError("forced-prefix bank identity mismatch")
def identity(gen: int, rec: dict, rec_path: Path, parent: dict) -> dict:
    base={"schema":"hex-autonomous-generation-v1","generation":gen,"recipe":{"id":rec["recipe_id"],"path":str(rec_path.resolve()),"sha256":sha(rec_path)},"parent":parent,"search":rec["selfplay"]["search"],"inference":rec["selfplay"]["inference"],"resource_profile":rec["selfplay"]["resource_profile"],"prefix_bank":rec["selfplay"]["forced_prefix_bank"],"fresh_games":rec["selfplay"]["games"],"candidate_plan":rec["candidate_plan"],"source_plan":rec["sources"]}
    base["identity_sha256"]=hashlib.sha256(canonical(base).encode()).hexdigest();return base
def next_stage(stage: str) -> str:
    return STAGES[STAGES.index(stage)+1] if stage in STAGES and stage!=STAGES[-1] else stage
def generation_root(output: Path, gen: int) -> Path: return output/f"generation-{gen:04d}"

def fake_evidence(stage: str, manifest: dict, rec: dict) -> dict:
    g=manifest["fresh_games"];c=manifest["candidate_plan"]; parent=manifest["parent"]
    if stage=="SELFPLAY_NORMAL": return {"passed":True,"accepted":g["normal"],"expected":g["normal"],"quarantined":0,"model_sha256":parent["onnx_sha256"],"search":manifest["search"]}
    if stage=="SELFPLAY_FORCED": return {"passed":True,"accepted":g["forced"],"expected":g["forced"],"quarantined":0,"model_sha256":parent["onnx_sha256"],"prefix_coverage_complete":True,"forced_rows_emitted":0,"prefix_bank_sha256":manifest["prefix_bank"]["sha256"]}
    if stage=="SELFPLAY_AUDIT": return {"passed":True,"normal_complete":True,"forced_complete":True,"duplicate_ids":0,"corrupt_rows":0}
    if stage=="DATA_PREP": return {"passed":True,"source_accounting_exact":True,"phase_b_rows":0,"source_manifest_bound":True}
    if stage=="TRAIN_CANDIDATES": return {"passed":True,"parent_checkpoint_sha256":parent["checkpoint_sha256"],"candidates":[{"candidate_id":f"g{manifest['generation']}-seed-{s}","seed":s,"completed":True,"nan_inf":False} for s in c["seeds"]]}
    if stage=="EXPORT_CANDIDATES": return {"passed":True,"candidates":[{"candidate_id":f"g{manifest['generation']}-seed-{s}","checkpoint_bound":True,"onnx_exists":True,"cpu_parity_passed":True} for s in c["seeds"]]}
    if stage=="CANDIDATE_SCREEN": return {"passed":True,"pairs":rec["evaluation"]["screening_pairs"],"candidates":[{"candidate_id":f"g{manifest['generation']}-seed-{s}","paired_score":.51+i*.01,"complete":True,"colour_balanced":True} for i,s in enumerate(c["seeds"])]}
    if stage=="PROMOTION_MATCH":
        winner=f"g{manifest['generation']}-seed-{c['seeds'][-1]}";lcb=float(rec.get("fake_backend",{}).get("promotion_lcb",.52));return {"passed":True,"challenger_id":winner,"pairs":rec["evaluation"]["promotion_pairs"],"paired_score":.55,"one_sided_95_lcb":lcb,"complete":True,"colour_balanced":True,"candidate_onnx_sha256":"fake-candidate-onnx"}
    if stage=="PROMOTION_DECISION": return {"passed":True}
    return {"passed":True}

def validate(stage: str, evidence: dict, manifest: dict, rec: dict) -> None:
    if not evidence.get("passed"): raise RuntimeError(f"{stage}: evidence did not pass")
    g=manifest["fresh_games"]
    if stage=="SELFPLAY_NORMAL" and (evidence.get("accepted")!=g["normal"] or evidence.get("quarantined")!=0 or evidence.get("model_sha256")!=manifest["parent"]["onnx_sha256"] or evidence.get("search")!=manifest["search"]): raise RuntimeError("normal self-play evidence mismatch")
    if stage=="SELFPLAY_FORCED" and (evidence.get("accepted")!=g["forced"] or evidence.get("quarantined")!=0 or not evidence.get("prefix_coverage_complete") or evidence.get("forced_rows_emitted")!=0 or evidence.get("prefix_bank_sha256")!=manifest["prefix_bank"]["sha256"]): raise RuntimeError("forced self-play evidence mismatch")
    if stage=="SELFPLAY_AUDIT" and (evidence.get("duplicate_ids") or evidence.get("corrupt_rows") or not evidence.get("normal_complete") or not evidence.get("forced_complete")): raise RuntimeError("self-play audit failed")
    if stage=="DATA_PREP" and (not evidence.get("source_accounting_exact") or evidence.get("phase_b_rows")!=0): raise RuntimeError("prepared data guard failed")
    if stage=="TRAIN_CANDIDATES":
        cs=evidence.get("candidates",[])
        expected=[f"g{manifest['generation']}-seed-{s}" for s in manifest["candidate_plan"]["seeds"]]
        if len(cs)!=len(expected) or sorted(x.get("candidate_id") for x in cs)!=expected or any(not x.get("completed") or x.get("nan_inf") for x in cs) or evidence.get("parent_checkpoint_sha256")!=manifest["parent"]["checkpoint_sha256"]: raise RuntimeError("candidate training guard failed")
    if stage=="EXPORT_CANDIDATES":
        cs=evidence.get("candidates",[]); expected=[f"g{manifest['generation']}-seed-{s}" for s in manifest["candidate_plan"]["seeds"]]
        if len(cs)!=len(expected) or sorted(x.get("candidate_id") for x in cs)!=expected or any(not x.get("checkpoint_bound") or not x.get("onnx_exists") or not x.get("cpu_parity_passed") for x in cs): raise RuntimeError("export guard failed")
    if stage=="CANDIDATE_SCREEN":
        cs=evidence.get("candidates",[]); expected=[f"g{manifest['generation']}-seed-{s}" for s in manifest["candidate_plan"]["seeds"]]
        if evidence.get("pairs")!=rec["evaluation"]["screening_pairs"] or len(cs)!=len(expected) or sorted(x.get("candidate_id") for x in cs)!=expected or any(not x.get("complete") or not x.get("colour_balanced") for x in cs): raise RuntimeError("screen guard failed")
    if stage=="PROMOTION_MATCH" and (evidence.get("pairs")!=rec["evaluation"]["promotion_pairs"] or not evidence.get("complete") or not evidence.get("colour_balanced")): raise RuntimeError("promotion match guard failed")

def report(root: Path, state: dict, manifest: dict) -> None:
    ev={p.stem:read(p) for p in sorted((root/"evidence").glob("*.json"))}; d={"schema":"hex-autonomous-generation-summary-v1","generation":manifest["generation"],"state":state["stage"],"parent":manifest["parent"],"fresh_games":manifest["fresh_games"],"source_plan":manifest["source_plan"],"candidate_plan":manifest["candidate_plan"],"evidence":ev,"resulting_champion":state.get("resulting_champion"),"promotion":state.get("promotion"),"wall_seconds":time.time()-state["created_epoch"]};atomic(root/"generation-summary.json",d)
    (root/"generation-summary.md").write_text(f"# Generation {manifest['generation']}\n\nState: **{state['stage']}**  \nParent: **{manifest['parent']['champion_id']}**  \nResult: **{state.get('resulting_champion','pending')}**\n\nMachine-readable evidence: `generation-summary.json`.\n")
def update_lineage(output: Path, state: dict, manifest: dict) -> None:
    p=output/"lineage-summary.json";x=read(p) if p.exists() else {"schema":"hex-autonomous-lineage-v1","generations":[]}; rows=[r for r in x["generations"] if r["generation"]!=manifest["generation"]];rows.append({"generation":manifest["generation"],"parent":manifest["parent"]["champion_id"],"state":state["stage"],"result":state.get("resulting_champion"),"promotion":state.get("promotion")});x["generations"]=sorted(rows,key=lambda r:r["generation"]);atomic(p,x)

def validate_completed_evidence(root: Path, state: dict, manifest: dict, rec: dict) -> None:
    """A resume treats prior evidence as immutable, not merely as a progress hint."""
    for stage in state.get("completed_stages",[]):
        if stage == "PENDING": continue
        p=root/"evidence"/f"{stage}.json"
        if not p.exists(): raise RuntimeError(f"completed stage lacks evidence: {stage}")
        validate(stage,read(p),manifest,rec)

def run_stage(root: Path, stage: str, manifest: dict, rec: dict, fake: bool, execute: bool, recover: bool) -> dict:
    ep=root/"evidence"/f"{stage}.json"
    if ep.exists(): evidence=read(ep);validate(stage,evidence,manifest,rec);return evidence
    if fake: evidence=fake_evidence(stage,manifest,rec)
    else:
        if not execute: raise RuntimeError(f"{stage} needs --execute with a signed adapter evidence file")
        cmd=rec.get("adapters",{}).get(stage,{}).get("command")
        if not cmd: raise RuntimeError(f"recipe lacks approved adapter command for {stage}")
        mapping={"generation_root":str(root),"generation":str(manifest["generation"]),"parent_onnx":manifest["parent"]["onnx_path"]}
        argv=[str(x).format(**mapping) for x in cmd]
        metadata={"pid":os.getpid(),"hostname":socket.gethostname(),"stage":stage,"generation":manifest["generation"],"started_epoch":time.time()}
        if stage in HEAVY:
            with Lease(root,metadata,recover): rc=subprocess.run(argv,cwd=ROOT).returncode
        else: rc=subprocess.run(argv,cwd=ROOT).returncode
        if rc: raise RuntimeError(f"adapter failed at {stage}: {rc}")
        if not ep.exists(): raise RuntimeError(f"adapter did not create required evidence {ep}")
        evidence=read(ep)
    validate(stage,evidence,manifest,rec);atomic(ep,evidence);return evidence

def advance(root: Path, manifest: dict, state: dict, rec: dict, fake: bool, execute: bool, recover: bool, stop_after: str|None=None) -> dict:
    while state["stage"] not in TERMINAL:
        stage=state["stage"]
        if stage=="PENDING":
            state.setdefault("completed_stages",[]).append(stage);state["stage"]=next_stage(stage);state["updated_epoch"]=time.time();atomic(root/"state.json",state);continue
        evidence=run_stage(root,stage,manifest,rec,fake,execute,recover)
        if stage=="PROMOTION_DECISION":
            m=read(root/"evidence"/"PROMOTION_MATCH.json");gate=float(rec["evaluation"]["promotion_lcb_strictly_greater_than"]); promoted=m["one_sided_95_lcb"]>gate
            state["promotion"]={"challenger":m["challenger_id"],"lcb":m["one_sided_95_lcb"],"gate":gate,"promoted":promoted,"registry_action":"manual_apply_required" if promoted else "incumbent_retained"};state["resulting_champion"]=m["challenger_id"] if promoted else manifest["parent"]["champion_id"]
            if not promoted:
                history=read(root.parent/"lineage-summary.json") if (root.parent/"lineage-summary.json").exists() else {"generations":[]}; prior=sum(1 for r in reversed(history["generations"]) if r.get("promotion",{}).get("promoted") is False)
                if prior+1>=int(rec["governance"]["plateau_consecutive_no_promotion"]): state["stage"]="PLATEAU_STOP"
        if state["stage"] not in TERMINAL: state["stage"]=next_stage(stage)
        state.setdefault("completed_stages",[]).append(stage);state["updated_epoch"]=time.time();atomic(root/"state.json",state);report(root,state,manifest)
        if stop_after==stage:
            append_event(root,{"event":"intentional_stage_pause","stage":stage,"next_stage":state["stage"],"epoch":time.time()})
            return state
    report(root,state,manifest);update_lineage(root.parent,state,manifest);return state

def main() -> int:
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--new',action='store_true');m.add_argument('--resume',action='store_true');m.add_argument('--status',action='store_true');m.add_argument('--dry-run',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--champion-registry',type=Path,required=True);p.add_argument('--generation',type=int,required=True);p.add_argument('--max-generations',type=int);p.add_argument('--stop-after-stage',choices=STAGES[:-1]);p.add_argument('--fake-backend',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--recover-stale-lock',action='store_true');a=p.parse_args();rec=recipe(a.recipe.resolve());parent=champion(a.champion_registry.resolve());root=generation_root(a.output.resolve(),a.generation);manifest_path=root/'generation-manifest.json'
    if a.max_generations is not None and a.generation>a.max_generations:
        print(json.dumps({'state':'MAX_GENERATIONS_REACHED','generation':a.generation,'max_generations':a.max_generations}));return 0
    if not a.fake_backend: verify_real_assets(rec)
    plan=identity(a.generation,rec,a.recipe.resolve(),parent)
    if a.dry_run: print(json.dumps({'schema':'hex-autonomous-generation-dry-run-v1','root':str(root),'identity':plan,'stages':STAGES,'heavy_stages':sorted(HEAVY)},indent=2,sort_keys=True));return 0
    if a.new:
        if root.exists(): raise RuntimeError('--new refuses an existing generation root')
        if rec.get('approval_status')!='approved' and not a.fake_backend: raise RuntimeError('real generation requires an approved recipe')
        root.mkdir(parents=True);atomic(manifest_path,plan);state={'schema':'hex-autonomous-generation-state-v1','generation':a.generation,'stage':'PENDING','completed_stages':[],'created_epoch':time.time(),'updated_epoch':time.time(),'attempt_id':str(uuid.uuid4())};atomic(root/'state.json',state);append_event(root,{'event':'new','attempt_id':state['attempt_id'],'epoch':time.time(),'fake_backend':a.fake_backend})
    else:
        if not manifest_path.exists(): raise RuntimeError('--resume/--status requires existing generation')
        if read(manifest_path)!=plan: raise RuntimeError('immutable generation identity mismatch')
        state=read(root/'state.json')
        if a.status: print(json.dumps({'root':str(root),'state':state,'manifest_sha256':sha(manifest_path)},indent=2,sort_keys=True));return 0
        if state['stage'] in {'FAILED','BLOCKED'}: state['stage']=next((s for s in STAGES if s not in state.get('completed_stages',[])), 'FAILED')
        state['attempt_id']=str(uuid.uuid4()); state['updated_epoch']=time.time(); atomic(root/'state.json',state);append_event(root,{'event':'resume','attempt_id':state['attempt_id'],'epoch':time.time(),'fake_backend':a.fake_backend})
    try:
        validate_completed_evidence(root,state,plan,rec)
        state=advance(root,plan,state,rec,a.fake_backend,a.execute,a.recover_stale_lock,a.stop_after_stage)
    except Exception as e:
        state['stage']='FAILED';state['failure']={'class':type(e).__name__,'message':str(e),'epoch':time.time()};state['updated_epoch']=time.time();atomic(root/'state.json',state);append_event(root,{'event':'failed','attempt_id':state.get('attempt_id'),'failure':state['failure']});report(root,state,plan);update_lineage(root.parent,state,plan);raise
    append_event(root,{'event':'complete_or_pause','attempt_id':state.get('attempt_id'),'state':state['stage'],'epoch':time.time()});print(json.dumps({'root':str(root),'state':state['stage'],'resulting_champion':state.get('resulting_champion')},sort_keys=True));return 0
if __name__=='__main__': main()
