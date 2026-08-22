#!/usr/bin/env python3
"""Turn fixed-bank timings into an explicit equal-wall-clock visit plan.

This is an evidence-based calibration, not a promise of per-move timing
equivalence. The resulting budgets are used only by the later paired gameplay
qualifier; the existing promotion defaults are untouched.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path

def mean(values): return sum(values) / len(values) if values else 0.0

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--sweep-root",type=Path,required=True); ap.add_argument("--baseline",default="c1.5-zero-r0-v128",help="operational equal-wall baseline"); ap.add_argument("--reference",default="c1.5-zero-r0-v2048",help="diagnostic reference only"); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    baseline_path=a.sweep_root/a.baseline/"results.jsonl"
    reference_path=a.sweep_root/a.reference/"results.jsonl"
    if not baseline_path.exists(): raise RuntimeError(f"missing operational baseline: {a.baseline}")
    if not reference_path.exists(): raise RuntimeError(f"missing diagnostic reference: {a.reference}")
    base_lines=baseline_path.read_text().splitlines()
    reference=[json.loads(x) for x in reference_path.read_text().splitlines() if x.strip()]
    base=[json.loads(x) for x in base_lines if x.strip()]
    target_seconds=mean([float(x["search_seconds"]) for x in base])
    plans=[]
    for directory in sorted(a.sweep_root.iterdir()):
        result=directory/"results.jsonl"
        if not directory.is_dir() or not result.exists() or directory.name in {a.baseline, a.reference}: continue
        rows=[json.loads(x) for x in result.read_text().splitlines() if x.strip()]
        if not rows: continue
        seconds_per_sim=mean([float(x["search_seconds"])/max(1,int(x["simulations"])) for x in rows])
        budget=max(1,int(round(target_seconds/max(seconds_per_sim,1e-12))))
        plans.append({"configuration":directory.name,"baseline_mean_seconds":target_seconds,"variant_mean_seconds":mean([x["search_seconds"] for x in rows]),"variant_seconds_per_sim":seconds_per_sim,"equal_wall_budget_estimate":budget,"caveat":"calibrated mean; verify with paired gameplay telemetry"})
    output={"schema":"hex-puct-fpu-equal-wall-plan-v1","operational_baseline":a.baseline,"diagnostic_reference":a.reference,"operational_baseline_mean_seconds":target_seconds,"diagnostic_reference_mean_seconds":mean([float(x["search_seconds"]) for x in reference]),"plans":plans,"not_promotion_protocol":True}
    a.output.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n"); print(json.dumps(output,sort_keys=True))
if __name__ == "__main__": main()
