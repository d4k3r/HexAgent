#!/usr/bin/env python3
"""Compare diagnostic configurations on exactly the same frozen positions."""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import re

def js(a,b):
    m=[(x+y)/2 for x,y in zip(a,b)]
    def kl(x,y): return sum(px*math.log(px/py) for px,py in zip(x,y) if px>0 and py>0)
    return (kl(a,m)+kl(b,m))/2
def tv(a,b): return sum(abs(x-y) for x,y in zip(a,b))/2
def mean(xs): return sum(xs)/len(xs) if xs else None

def config_fields(name: str, row: dict) -> dict:
    match = re.search(r"-v(\d+)$", name)
    if not match:
        raise RuntimeError(f"cannot determine visit budget from configuration: {name}")
    return {
        "c_puct": float(row.get("c_puct", name.split("-", 1)[0][1:])),
        "fpu_mode": row.get("fpu_mode", "zero" if "-zero-" in name else "parent_value_reduced"),
        "fpu_reduction": float(row.get("fpu_reduction", 0.0 if "-zero-" in name else name.split("-r", 1)[1].split("-", 1)[0])),
        "visits": int(row.get("requested_visits", match.group(1))),
    }

def nominate(rows: list[dict], operational_baseline: str, nomination_visits: int, max_nominations: int = 2) -> tuple[list[dict], dict]:
    baseline = next((row for row in rows if row["configuration"] == operational_baseline), None)
    if baseline is None:
        raise RuntimeError(f"missing operational baseline: {operational_baseline}")
    eligible = []
    for row in rows:
        if row["configuration"] == operational_baseline or row["visits"] != nomination_visits:
            continue
        # Conservative evidence gate: the parameter variant must improve or
        # preserve selected-action agreement, reduce policy distance, reduce
        # breadth, and increase depth relative to the operational baseline.
        if not (
            row["action_agreement"] >= baseline["action_agreement"]
            and row["policy_tv"] < baseline["policy_tv"]
            and row["visited_children"] < baseline["visited_children"]
            and row["max_depth"] > baseline["max_depth"]
        ):
            continue
        eligible.append(row)
    ordering = lambda row: (
        -row["action_agreement"], -row["top3_agreement"], row["policy_tv"],
        row["visited_children"], -row["max_depth"], row["search_seconds"],
        row["configuration"],
    )
    ordered = sorted(eligible, key=ordering)
    selected = []
    if ordered:
        selected.append(ordered[0])
        # Prefer a distinct c_puct value for the second slot, making the
        # gameplay test informative about both FPU and exploration pressure.
        for row in ordered[1:]:
            if row["c_puct"] != selected[0]["c_puct"]:
                selected.append(row)
                break
        if len(selected) < max_nominations:
            for row in ordered[1:]:
                if row not in selected:
                    selected.append(row)
                    if len(selected) == max_nominations:
                        break
    rule = {
        "operational_baseline": operational_baseline,
        "nomination_visits": nomination_visits,
        "excluded_diagnostic_reference": True,
        "eligible_count": len(eligible),
        "max_nominations": max_nominations,
        "selection_rule": "v128-only; exclude operational baseline; require action agreement >= baseline, lower policy TV, fewer visited children, and greater depth; rank by action/top3/TV/breadth/depth/runtime; choose a distinct c_puct for slot two where available",
        "diagnostic_reference_does_not_establish_strength": True,
    }
    return selected, rule

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--sweep-root",type=Path,required=True); ap.add_argument("--bank",type=Path,required=True); ap.add_argument("--baseline",default="c1.5-zero-r0-v2048",help="diagnostic comparison reference"); ap.add_argument("--operational-baseline",default="c1.5-zero-r0-v128"); ap.add_argument("--nomination-visits",type=int,default=128); ap.add_argument("--max-nominations",type=int,default=2); ap.add_argument("--output",type=Path,required=True); a=ap.parse_args()
    root=a.sweep_root; dirs=sorted(x for x in root.iterdir() if x.is_dir() and (x/"results.jsonl").exists())
    data={d.name:[json.loads(line) for line in (d/"results.jsonl").read_text().splitlines() if line.strip()] for d in dirs}
    if a.baseline not in data: raise RuntimeError(f"missing baseline {a.baseline}")
    base={x["position_id"]:x for x in data[a.baseline]}; rows=[]
    bank_map={json.loads(line)["position_id"]:json.loads(line) for line in (a.bank/"positions.jsonl").read_text().splitlines() if line.strip()}
    for name, items in data.items():
        if name==a.baseline: continue
        by={x["position_id"]:x for x in items}; common=sorted(set(base)&set(by)); metrics=[]
        for pos in common:
            x,y=by[pos],base[pos]; p=x["root_policy"]; q=y["root_policy"]
            top_p=set(sorted(range(len(p)),key=lambda i:p[i],reverse=True)[:3]); top_q=set(sorted(range(len(q)),key=lambda i:q[i],reverse=True)[:3])
            metric={"tv":tv(p,q),"js":js(p,q),"action_agreement":int(x["selected_action"]==y["selected_action"]),"top3_agreement":int(top_p==top_q),"root_value_abs":abs((x["root_value"] or 0)-(y["root_value"] or 0)),"visited_children":x["visited_children"],"max_depth":x["max_depth"],"search_seconds":x["search_seconds"],"simulations":x["simulations"]}
            metrics.append(metric)
        summary={"configuration":name,"positions":len(metrics),"action_agreement":mean([m["action_agreement"] for m in metrics]),"top3_agreement":mean([m["top3_agreement"] for m in metrics]),"policy_tv":mean([m["tv"] for m in metrics]),"policy_js":mean([m["js"] for m in metrics]),"root_value_abs":mean([m["root_value_abs"] for m in metrics]),"visited_children":mean([m["visited_children"] for m in metrics]),"max_depth":mean([m["max_depth"] for m in metrics]),"search_seconds":mean([m["search_seconds"] for m in metrics]),"simulations_per_second":sum(m["simulations"] for m in metrics)/sum(m["search_seconds"] for m in metrics)}
        summary.update(config_fields(name, items[0]))
        for field in ("source","side_to_move","phase_band","entropy_band"):
            groups=defaultdict(list)
            for pos,metric in zip(common,metrics): groups[bank_map[pos].get(field,"unknown")].append(metric)
            summary[field+"_strata"]={key:{"positions":len(values),"action_agreement":mean([m["action_agreement"] for m in values]),"policy_tv":mean([m["tv"] for m in values]),"visited_children":mean([m["visited_children"] for m in values])} for key,values in sorted(groups.items())}
        rows.append(summary)
    rows.sort(key=lambda x:(-x["action_agreement"],x["search_seconds"]))
    nominations, nomination_rule = nominate(rows, a.operational_baseline, a.nomination_visits, a.max_nominations)
    out={"schema":"hex-puct-fpu-analysis-v2","baseline":a.baseline,"operational_baseline":a.operational_baseline,"comparisons":rows,"nominated_for_gameplay":nominations,"nomination_rule":nomination_rule,"note":"agreement with the diagnostic reference is stability evidence only, not a strength claim"}
    a.output.write_text(json.dumps(out,sort_keys=True,indent=2)+"\n"); print(json.dumps(out,sort_keys=True))
if __name__ == "__main__": main()
