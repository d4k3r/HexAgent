#!/usr/bin/env python3
"""Create, but never execute, a frozen self-play resource benchmark plan."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.recipe.read_text())
 if r.get('schema')!='hex-autonomous-generation-recipe-v1':raise RuntimeError('wrong recipe schema')
 if a.output.exists():raise RuntimeError('refusing existing benchmark plan')
 sp=r['selfplay']; rows=[]
 for i,c in enumerate(r['resource_benchmark']['candidates']): rows.append({'benchmark_id':f'resource-{i:02d}','games':r['resource_benchmark']['game_count'],'search':sp['search'],'resource':c,'fixed':'model, seeds, game IDs, certificate semantics, prefix bank and search semantics'})
 out={'schema':'hex-autonomous-throughput-benchmark-plan-v1','recipe_id':r['recipe_id'],'recipe_sha256':sha(a.recipe),'approved_resource_profile':r['selfplay'].get('resource_profile'),'execution':'manual approval required; no automatic production-default change','metrics':['accepted_games_per_second','elapsed_seconds','mean_batch_size','peak_batch_size','queue_high_water','watchdog_or_errors','GPU_utilization_summary','VRAM_summary','CPU_utilization_summary','RSS_summary','semantic_audit_valid'],'candidates':rows,'selection_rule':'FASTEST_STABLE is descriptive only; explicit human approval is required before changing recipe inference settings'}
 a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');os.replace(tmp,a.output);print(json.dumps({'output':str(a.output),'candidates':len(rows)}))
if __name__=='__main__':main()
