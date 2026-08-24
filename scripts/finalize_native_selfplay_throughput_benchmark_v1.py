#!/usr/bin/env python3
"""Make a confirmation plan and final resource profile from audited results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

def read(path: Path) -> dict[str, Any]: return json.loads(path.read_text())
def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); tmp=path.with_name(path.name+".partial")
    tmp.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n"); os.replace(tmp,path)

p=argparse.ArgumentParser(description=__doc__)
mode=p.add_mutually_exclusive_group(required=True)
mode.add_argument('--write-confirmation-plan',type=Path)
mode.add_argument('--write-profile',type=Path)
p.add_argument('--stage-a-analysis',type=Path)
p.add_argument('--stage-b-analysis',type=Path)
p.add_argument('--parent-plan',type=Path)
p.add_argument('--confirmation-root',type=Path)
p.add_argument('--report',type=Path)
a=p.parse_args()

if a.write_confirmation_plan:
    if not all((a.stage_a_analysis,a.stage_b_analysis,a.parent_plan)):
        raise SystemExit('confirmation planning needs both analyses and the parent plan')
    reports=[read(a.stage_a_analysis),read(a.stage_b_analysis)]; parent=read(a.parent_plan)
    rows=[row for report in reports for row in report['configs']]
    best=max(rows,key=lambda row:row['games_per_second'])
    baseline={'benchmark_id':'baseline-1x64-b96-w200','process_count':1,'concurrency_per_process':64,'max_batch':96,'wait_us':200}
    candidate={**best['resource'], 'benchmark_id':'candidate-fastest-stage-ab'}
    output={key:parent[key] for key in ('schema','champion_registry','champion_onnx_sha256','search')}
    output.update({'plan_id':'native-selfplay-throughput-confirmation-v1','workload':{**parent['workload'],'start_id':9200000,'games':512},'configs':[baseline,candidate],
                   'selection_basis':{'stage_a_analysis_sha256':sha(a.stage_a_analysis),'stage_b_analysis_sha256':sha(a.stage_b_analysis),'fastest_stage_ab':best['benchmark_id'],'fastest_resource':best['resource']},
                   'selection_rule':'confirmation compares the Stage-A/B fastest audit-complete resource candidate with current 1x64/b96/w200 on one new exact 512-game workload'})
    write(a.write_confirmation_plan,output);print(json.dumps({'output':str(a.write_confirmation_plan),'candidate':candidate}));raise SystemExit(0)

if not a.confirmation_root:
    raise SystemExit('profile finalization needs --confirmation-root')
results=[]
for path in sorted((a.confirmation_root/'configs').glob('*/config-result.json')):
    result=read(path)
    if result.get('status')!='complete': raise SystemExit(f'incomplete confirmation result {path}')
    manifest=read(path.parent/'benchmark-config-manifest.json')
    results.append((result,manifest))
if len(results)!=2: raise SystemExit('confirmation requires exactly two completed configs')
baseline=next((r,m) for r,m in results if r['benchmark_id']=='baseline-1x64-b96-w200')
candidate=next((r,m) for r,m in results if r['benchmark_id']!='baseline-1x64-b96-w200')
improvement=100*(candidate[0]['games_per_second']/baseline[0]['games_per_second']-1)
gpu='unavailable'
try:
    q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    if q.returncode==0: gpu=q.stdout.strip()
except (OSError,subprocess.TimeoutExpired): pass
profile={'schema':'hex-native-selfplay-resource-profile-v1','profile_id':'selfplay-resource-profile-v1','status':'PROPOSED_QUALIFIED_REQUIRES_EXPLICIT_APPROVAL','machine_identity':{'operator_declared_cpu':'AMD Ryzen 7 9800X3D','gpu_runtime_query':gpu},
         'champion_onnx_sha256':candidate[1]['champion_onnx_sha256'],'search':candidate[1]['search'],'resource':candidate[1]['resource'],'confirmation':{'baseline':baseline[0],'candidate':candidate[0],'games_per_second_improvement_percent':improvement,'semantic_policy':'each configuration independently passed native-v2 audit; exact per-game fingerprint comparison is retained in confirmation analysis'},
         'default_change_policy':'This profile does not alter production defaults. A human must explicitly approve it for a future recipe.'}
write(a.write_profile,profile)
if a.report:
    text='# Native self-play throughput V1 final summary\n\n'
    text+=f"- Baseline: {baseline[0]['games_per_second']:.4f} games/s\n- Candidate: {candidate[0]['games_per_second']:.4f} games/s\n- Improvement: {improvement:.2f}%\n- Proposed resource: `{json.dumps(candidate[1]['resource'],sort_keys=True)}`\n"
    a.report.parent.mkdir(parents=True,exist_ok=True); tmp=a.report.with_name(a.report.name+'.partial');tmp.write_text(text);os.replace(tmp,a.report)
print(json.dumps({'output':str(a.write_profile),'improvement_percent':improvement,'resource':candidate[1]['resource']}))
