#!/usr/bin/env python3
"""Read-only Stage-7 Phase-A prefix salvage manifest generator.

The output intentionally references immutable original game JSON files rather
than copying policy targets: retained root-visits stay byte-identical in the
frozen source records.  A future loader applies `effective_winner` to derive z.
"""
from __future__ import annotations
import argparse, hashlib, json, os, statistics, subprocess
from pathlib import Path
from hex_reconstruction.board import BLACK, WHITE, HexBoard

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNNER = ROOT / "build/cpp-puct-stage7/hex_stage7_prefix_certificate_runner"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def atomic(path: Path, value: object) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(partial, path)

def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    if not (root / "games").is_dir(): root = root / "data"
    files = sorted((root / "games").glob("game-*.json"))
    if not files: raise ValueError(f"no Stage-7 games under {root}")
    return files

def parse_game(path: Path) -> dict:
    game = json.loads(path.read_text())
    required = {"schema","status","game_id","moves","samples","winner","game_length","search_budget"}
    if not required <= game.keys() or game["schema"] != "hex-selfplay-game-v1" or game["status"] != "complete":
        raise ValueError(f"invalid Stage-7 game {path}")
    if len(game["moves"]) != game["game_length"] or len(game["samples"]) != game["game_length"]:
        raise ValueError(f"game length/sample mismatch {path}")
    board=HexBoard(); winner = BLACK if game["winner"] == "B" else WHITE if game["winner"] == "W" else None
    if winner is None: raise ValueError(f"invalid historical winner {path.name}")
    for ply,(move,sample) in enumerate(zip(game["moves"],game["samples"])):
        visits = sample.get("root_visits")
        if sample.get("ply") != ply or sample.get("selected_move") != move or sample.get("side_to_move") not in ("B","W"):
            raise ValueError(f"sample provenance mismatch {path.name}:{ply}")
        if not isinstance(visits,list) or len(visits)!=121 or any(type(v) is not int or v<0 for v in visits) or sum(visits)!=game["search_budget"] or visits[move]<=0:
            raise ValueError(f"invalid root visits {path.name}:{ply}")
        if board.literal_winner() is not None or not isinstance(move,int) or not board.is_legal(move):
            raise ValueError(f"illegal or post-terminal move {path.name}:{ply}")
        if (sample["side_to_move"] == "B") != (board.side_to_move == BLACK):
            raise ValueError(f"side-to-move board mismatch {path.name}:{ply}")
        if any(count for count, allowed in zip(visits,board.legal_mask()) if not allowed):
            raise ValueError(f"illegal visit support {path.name}:{ply}")
        if float(sample.get("z")) != (1.0 if board.side_to_move == winner else -1.0):
            raise ValueError(f"historical side-relative z mismatch {path.name}:{ply}")
        board.play(move)
    if board.literal_winner() != winner: raise ValueError(f"literal winner replay mismatch {path.name}")
    return game

def classify(game: dict, source: Path, certificate: dict) -> dict:
    """Apply the frozen pre-move-row / post-move-certificate boundary."""
    gid = game["game_id"]; literal = game["winner"]; length = game["game_length"]
    ply, owner = certificate.get("certificate_ply"), certificate.get("certificate_owner")
    base = {"schema":"stage7-prefix-salvage-record-v1","game_id":gid,"source_path":str(source.resolve()),
            "source_sha256":sha(source),"historical_literal_winner":literal,"game_length":length,
            "certificate_ply":ply,"certificate_owner":owner,"bridge_count":certificate.get("bridge_count",0),
            "validated":certificate.get("validated",False),"realizer_supported":certificate.get("realizer_supported",False),
            "simultaneous_certificates":certificate.get("simultaneous",False)}
    if ply is None:
        return {**base,"status":"retained","reason":"no_certificate_before_literal_terminal","effective_winner":literal,
                "retained_phase_a_rows":length,"historical_tail_plies":0,"winner_changed":False,
                "policy_source":{"sample_indices":[0,length],"root_visits":"immutable_source_bytes"},"value_source":"original_immutable_z"}
    if not isinstance(ply,int) or not 1 <= ply <= length or owner not in ("B","W"):
        return {**base,"status":"quarantined","reason":"malformed_certificate_result","retained_phase_a_rows":0}
    mover = "B" if ply % 2 else "W"
    if certificate.get("simultaneous"):
        return {**base,"status":"quarantined","reason":"simultaneous_certificates","retained_phase_a_rows":0}
    if not certificate.get("validated") or not certificate.get("realizer_supported"):
        return {**base,"status":"quarantined","reason":"unsupported_or_invalid_certificate","retained_phase_a_rows":0}
    if owner != mover:
        return {**base,"status":"quarantined","reason":"certificate_first_seen_for_nonmoving_side","retained_phase_a_rows":0}
    if ply == length and owner != literal:
        return {**base,"status":"quarantined","reason":"certificate_literal_winner_disagreement_same_move","retained_phase_a_rows":0}
    # `ply` is 1-based post-move board ply. Sample ply-1 is the PUCT state
    # immediately before that move and therefore remains supervised.
    result = {**base,"status":"retained","reason":"certificate_phase_a_boundary","effective_winner":owner,
            "retained_phase_a_rows":ply,"historical_tail_plies":length-ply,"winner_changed":owner != literal,
            "policy_source":{"sample_indices":[0,ply],"root_visits":"immutable_source_bytes"},
            "z_rule":"+1 iff original sample side_to_move equals effective_winner; otherwise -1"}
    if owner != literal:
        result["z_override"] = [1.0 if sample["side_to_move"] == owner else -1.0 for sample in game["samples"][:ply]]
        result["value_source"] = "certificate_owner_side_relative_override"
    else: result["value_source"] = "original_immutable_z"
    return result

def certificates(runner: Path, games: list[tuple[Path,dict]]) -> dict[int,dict]:
    payload = "".join(f"{g['game_id']}|{','.join(map(str,g['moves']))}\n" for _,g in games)
    result = subprocess.run([str(runner)], input=payload, text=True, capture_output=True, check=False)
    if result.returncode: raise RuntimeError(f"certificate runner failed: {result.stderr[-2000:]}")
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    output = {int(row["id"]):row for row in rows}
    if len(output)!=len(games) or any(g["game_id"] not in output for _,g in games): raise ValueError("certificate runner coverage mismatch")
    return output

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--source",type=Path,required=True); ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--runner",type=Path,default=DEFAULT_RUNNER); ap.add_argument("--limit",type=int); args=ap.parse_args()
    if args.output.exists(): raise RuntimeError(f"refusing existing output: {args.output}")
    files=source_files(args.source); files=files[:args.limit] if args.limit is not None else files
    if not args.runner.is_file(): raise RuntimeError(f"missing certificate runner: {args.runner}")
    games=[(path,parse_game(path)) for path in files]; found=certificates(args.runner,games)
    records=[classify(game,path,found[game["game_id"]]) for path,game in games]
    args.output.mkdir(parents=True)
    records_path=args.output/'prefix-manifest.jsonl'; partial=records_path.with_suffix('.jsonl.partial')
    with partial.open('w') as out:
        for record in records: out.write(json.dumps(record,sort_keys=True,separators=(',',':'))+'\n')
    os.replace(partial,records_path)
    kept=[r for r in records if r['status']=='retained']; tails=[r['historical_tail_plies'] for r in kept if r['certificate_ply'] is not None]
    summary={"schema":"stage7-prefix-salvage-summary-v1","source":str(args.source.resolve()),"source_game_count":len(files),
             "records_sha256":sha(records_path),"games_processed":len(records),"certificates_found":sum(r['certificate_ply'] is not None for r in records),
             "retained_games":len(kept),"quarantines":sum(r['status']=='quarantined' for r in records),
             "retained_rows":sum(r.get('retained_phase_a_rows',0) for r in kept),"original_rows":sum(r['game_length'] for r in records),
             "rows_removed":sum(r['game_length'] for r in records)-sum(r.get('retained_phase_a_rows',0) for r in kept),
             "winner_flips":sum(r.get('winner_changed',False) for r in kept),"tails":{"mean":statistics.mean(tails) if tails else None,"median":statistics.median(tails) if tails else None,"p90":sorted(tails)[max(0,(len(tails)*9+9)//10-1)] if tails else None,"max":max(tails) if tails else None},
             "contract":"manifest references original immutable policy bytes; z is derived from effective_winner only"}
    atomic(args.output/'summary.json',summary); print(json.dumps(summary,sort_keys=True))
if __name__=='__main__': main()
