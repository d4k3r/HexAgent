#!/usr/bin/env python3
"""Freeze the deterministic Champion-2 deep-teacher position bank.

The builder reads committed native-v2 Phase-A rows only.  It never writes to
the source corpora.  A bounded deterministic reservoir is used for the
Champion-2 scoring pass; all final records retain their original board and
root-visit provenance.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, statistics
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from hex_reconstruction.board import HexBoard, BOARD_SIZE

ROOT = Path(__file__).resolve().parents[1]
C2_ONNX = ROOT / "artifacts/stage8b-v3-diversity/training/diverse-v1/model-dynamic.onnx"
EVAL_BANK = ROOT / "artifacts/puct-fpu-strength-v1/gameplay-qualifier/openings-v1.json"
AREA = 121

def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def entropy(v: list[int]) -> float:
    s = sum(v)
    return -sum((x / s) * math.log(x / s) for x in v if x) if s else 0.0

def transpose_action(a: int) -> int:
    return (a % BOARD_SIZE) * BOARD_SIZE + a // BOARD_SIZE

def state_digest(board: HexBoard) -> str:
    return hashlib.sha256(bytes(int(x) for p in board.feature_planes() for x in p)).hexdigest()

def orbit_key(board: HexBoard) -> str:
    b = tuple(sorted(transpose_action(i) for i, c in enumerate(board.cells) if c == "white"))
    w = tuple(sorted(transpose_action(i) for i, c in enumerate(board.cells) if c == "black"))
    last = None if board.last_move is None else transpose_action(board.last_move)
    t = (b, w, "white" if board.side_to_move == "black" else "black", last)
    normal = (tuple(i for i, c in enumerate(board.cells) if c == "black"), tuple(i for i, c in enumerate(board.cells) if c == "white"), board.side_to_move, board.last_move)
    return min(hashlib.sha256(repr(normal).encode()).hexdigest(), hashlib.sha256(repr(t).encode()).hexdigest())

def phase(ply: int) -> str:
    return "early" if ply < 10 else "mid" if ply < 30 else "late"

def occ_band(ply: int) -> str:
    return "0-9" if ply < 10 else "10-29" if ply < 30 else "30-59" if ply < 60 else "60+"

def eval_opening_keys(path: Path) -> set[tuple[int, ...]]:
    x = json.loads(path.read_text()); keys = set()
    for row in x.get("openings", []):
        a = tuple(row["opening_moves"])
        keys.add(a); keys.add(tuple(transpose_action(i) for i in a))
    return keys

def row_records(root: Path, source: str, opening_keys: set[tuple[int, ...]], seed: int, pool_size: int) -> list[dict]:
    files = sorted((root / "games").glob("game-*.json"), key=lambda p: int(p.stem.split("-")[-1]))
    manifest = json.loads((root / "run-manifest.json").read_text())
    expected = manifest["game_ids"]["count"]
    if len(files) != expected: raise RuntimeError(f"{root}: incomplete source ({len(files)}/{expected})")
    rng = random.Random(f"deep-teacher-bank-v1:{seed}:{source}"); pool: list[dict] = []; seen = 0
    for path in files:
        game = json.loads(path.read_text())
        if game.get("status") != "accepted": raise RuntimeError(f"non-accepted {path}")
        forced = [int(a) for a in game.get("forced_prefix_actions", [])]
        board = HexBoard()
        for a in forced:
            if not board.is_legal(a): raise RuntimeError(f"illegal prefix {path}")
            board.play(a)
        # Positions derived from an evaluation opening's exact four-ply prefix
        # are excluded; the bank therefore cannot tune to promotion openings.
        eval_overlap = tuple(int(a) for a in game.get("moves", [])[:4]) in opening_keys
        for ri, sample in enumerate(game.get("samples", [])):
            if int(sample["ply"]) != board.ply: raise RuntimeError(f"ply mismatch {path}")
            if source == "forced" and board.ply < 3: raise RuntimeError("forced move emitted as policy row")
            if not eval_overlap:
                rec = {
                    "source": source.upper(), "source_root": str(root.resolve()), "source_run_id": game["run_id"],
                    "source_game_id": int(game["game_id"]), "source_game_seed": int(game["game_seed"]),
                    "source_ply": int(sample["ply"]), "side_to_move": "BLACK" if board.side_to_move == "black" else "WHITE",
                    "last_move": board.last_move, "black_actions": [i for i,c in enumerate(board.cells) if c == "black"],
                    "white_actions": [i for i,c in enumerate(board.cells) if c == "white"],
                    "state": [int(x) for p in board.feature_planes() for x in p], "state_sha256": state_digest(board),
                    "transpose_orbit": orbit_key(board), "root_visits": [int(x) for x in sample["root_visits"]],
                    "root_entropy": entropy([int(x) for x in sample["root_visits"]]), "ply_band": phase(board.ply),
                    "occupancy_band": occ_band(board.ply), "forced_prefix_length": len(forced),
                }
                seen += 1
                if len(pool) < pool_size: pool.append(rec)
                else:
                    j = rng.randrange(seen)
                    if j < pool_size: pool[j] = rec
            board.play(int(sample["selected_move"]))
    if len(pool) < pool_size: raise RuntimeError(f"source pool too small {source}: {len(pool)}")
    return pool

def score_student(records: list[dict], model: Path, batch: int) -> None:
    try:
        import onnxruntime as ort
    except Exception:
        for r in records: r.update({"student_entropy": r["root_entropy"], "student_top_margin": 0.0, "student_search_tv": 0.0, "student_scored": False})
        return
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name; outs = [x.name for x in sess.get_outputs()]
    for start in range(0, len(records), batch):
        group = records[start:start+batch]; x = np.asarray([r["state"] for r in group], dtype=np.float32).reshape(-1,6,11,11)
        result = sess.run(outs, {inp: x}); logits = np.asarray(result[0]); values = np.asarray(result[1]).reshape(-1)
        for j, r in enumerate(group):
            legal = np.asarray([not (r["state"][i] or r["state"][AREA+i]) for i in range(AREA)], bool)
            z = logits[j].astype(np.float64); z[~legal] = -1e30; z -= np.max(z); p = np.exp(z); p /= p.sum()
            root = np.asarray(r["root_visits"], dtype=np.float64); root /= root.sum(); order = np.sort(p[legal])[::-1]
            r.update({"student_entropy": float(-np.sum(p[p>0]*np.log(p[p>0]))), "student_top_margin": float(order[0]-order[1]) if len(order)>1 else 1.0, "student_search_tv": float(.5*np.abs(p-root).sum()), "student_value": float(values[j]), "student_scored": True})

def select_broad(pool: list[dict], target: int, used: set[str]) -> list[dict]:
    # Equal source quotas, with deterministic round-robin across side/phase/
    # occupancy strata; this avoids selecting only common opening trajectories.
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in pool: groups[(r["source"], r["side_to_move"], r["ply_band"], r["occupancy_band"])].append(r)
    for g in groups.values(): g.sort(key=lambda r: (hashlib.sha256((r["state_sha256"]+r["source"]).encode()).hexdigest(), r["source_game_id"], r["source_ply"]))
    chosen=[]; source_counts=Counter(); keys=sorted(groups)
    while len(chosen)<target and keys:
        progress=False
        for k in keys:
            g=groups[k]
            while g and g[0]["state_sha256"] in used: g.pop(0)
            if not g: continue
            if source_counts[k[0]] >= target//2: continue
            chosen.append(g.pop(0)); used.add(chosen[-1]["state_sha256"]); source_counts[k[0]] += 1; progress=True
            if len(chosen)>=target: break
        if not progress: break
    if len(chosen)!=target: raise RuntimeError(f"broad selection only {len(chosen)}/{target}")
    return chosen

def select_hard(pool: list[dict], target: int, used: set[str]) -> list[dict]:
    for r in pool:
        entropy_score = r.get("student_entropy", r["root_entropy"])
        r["hard_score"] = .35*entropy_score + .30*r.get("student_search_tv",0.0) + .20*(1.0-r.get("student_top_margin",0.0)) + .10*r["root_entropy"] + .05*(int(r["state_sha256"][:12],16)/float(16**12))
    chosen=[]
    for r in sorted(pool, key=lambda x:(-x["hard_score"], x["state_sha256"])):
        if r["state_sha256"] in used: continue
        chosen.append(r); used.add(r["state_sha256"])
        if len(chosen)==target: break
    if len(chosen)!=target: raise RuntimeError(f"hard selection only {len(chosen)}/{target}")
    return chosen

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--normal-root",type=Path,required=True);p.add_argument("--forced-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--seed",type=int,default=16002026);p.add_argument("--pool-per-source",type=int,default=8192);p.add_argument("--student-model",type=Path,default=C2_ONNX);p.add_argument("--eval-openings",type=Path,default=EVAL_BANK);p.add_argument("--score-batch",type=int,default=64);a=p.parse_args();out=a.output.resolve()
    if out.exists(): raise RuntimeError(f"refusing existing bank root {out}")
    openings=eval_opening_keys(a.eval_openings.resolve()); n=row_records(a.normal_root.resolve(),"normal",openings,a.seed,a.pool_per_source); f=row_records(a.forced_root.resolve(),"forced",openings,a.seed,a.pool_per_source); pool=n+f; score_student(pool,a.student_model.resolve(),a.score_batch)
    used=set(); broad=select_broad(pool,2048,used); hard=select_hard(pool,2048,used)
    records=[]
    for cls, rows in (("BROAD",broad),("HARD",hard)):
        for i,r in enumerate(rows):
            q=dict(r);q.pop("state",None);q["position_id"]=f"deep-{cls.lower()}-{i:04d}";q["bank_class"]=cls;q["state_flat"]=[int(x) for x in r["state"]];q["source_manifest_sha256"]=sha(Path(r["source_root"])/"run-manifest.json");records.append(q)
    payload={"schema":"deep-teacher-1600-position-bank-v1","selection_before_deep_search":True,"master_seed":a.seed,"student_model":{"path":str(a.student_model.resolve()),"sha256":sha(a.student_model.resolve())},"source_roots":{"NORMAL":str(a.normal_root.resolve()),"FORCED":str(a.forced_root.resolve())},"evaluation_openings":{"path":str(a.eval_openings.resolve()),"sha256":sha(a.eval_openings.resolve()),"excluded_exact_four_ply_prefixes":len(openings)},"pool_per_source":a.pool_per_source,"selection_rule":"BROAD equal NORMAL/FORCED quota with deterministic source/side/phase/occupancy round-robin; HARD descending deterministic Champion-2 weakness score = .35 student entropy + .30 student/Search-V2 TV + .20(1-top-margin) + .10 Search-V2 entropy + .05 novelty; exact state hash uniqueness enforced","positions":records}
    out.mkdir(parents=True); atomic(out/"bank.json",payload); bank_sha=sha(out/"bank.json"); atomic(out/"bank-manifest.json",{"schema":"deep-teacher-1600-bank-manifest-v1","bank_path":str((out/"bank.json").resolve()),"bank_sha256":bank_sha,"positions":4096,"broad":2048,"hard":2048,"source_manifest_sha256":{k:sha(Path(v)/"run-manifest.json") for k,v in payload["source_roots"].items()},"state_exact_duplicates":len(records)-len({r["state_sha256"] for r in records}),"transpose_orbit_count":len({r["transpose_orbit"] for r in records}),"student_scored":sum(bool(r.get("student_scored")) for r in records)})
    print(json.dumps({"output":str(out),"bank_sha256":bank_sha,"positions":len(records),"broad":len(broad),"hard":len(hard),"exact_duplicates":len(records)-len({r['state_sha256'] for r in records}),"transpose_orbits":len({r['transpose_orbit'] for r in records}),"student_scored":sum(bool(r.get('student_scored')) for r in records)},sort_keys=True))
if __name__=='__main__': main()
