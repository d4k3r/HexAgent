#!/usr/bin/env python3
"""Compare paired 1600/3200 root targets on the frozen 512 subset."""
from __future__ import annotations
import argparse,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path
def tv(a,b):return .5*sum(abs(x-y) for x,y in zip(a,b))
def js(a,b):
 m=[(x+y)/2 for x,y in zip(a,b)];return .5*sum(x*math.log(x/m[i]) for i,x in enumerate(a) if x)+.5*sum(x*math.log(x/m[i]) for i,x in enumerate(b) if x)
def summary(v):return {'n':len(v),'mean':statistics.fmean(v) if v else None,'median':statistics.median(v) if v else None,'p90':sorted(v)[int(.9*(len(v)-1))] if v else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--subset',type=Path,required=True);p.add_argument('--run1600',type=Path,required=True);p.add_argument('--run3200',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sub=json.loads(a.subset.read_text());r16={json.loads(f.read_text())['position_id']:json.loads(f.read_text()) for f in (a.run1600/'results').glob('position-*.json')};r32={json.loads(f.read_text())['position_id']:json.loads(f.read_text()) for f in (a.run3200/'results').glob('position-*.json')};bank={x['position_id']:x for x in json.loads(Path(json.loads((a.run1600/'manifest.json').read_text())['bank_path']).read_text())['positions']};rows=[]
 for s in sub['positions']:
  pid=s['position_id'];x,y=r16.get(pid),r32.get(pid)
  if not x or not y or x.get('teacher_terminal') or y.get('teacher_terminal'):continue
  rows.append({'position_id':pid,'bank_class':s['bank_class'],'source':bank[pid]['source'],'ply_band':s.get('ply_band'),'top1':x['selected_action']==y['selected_action'],'top3':y['selected_action'] in sorted(range(121),key=lambda i:(-x['pi'][i],i))[:3],'tv':tv(x['pi'],y['pi']),'js':js(x['pi'],y['pi']),'value_abs_diff':abs((x.get('teacher_root_utility') or 0)-(y.get('teacher_root_utility') or 0))})
 def agg(v):return {'n':len(v),'top1_agreement':sum(x['top1'] for x in v)/len(v) if v else None,'top3_agreement':sum(x['top3'] for x in v)/len(v) if v else None,'tv':summary([x['tv'] for x in v]),'js':summary([x['js'] for x in v]),'value_abs_diff':summary([x['value_abs_diff'] for x in v])}
 strata={'all':agg(rows)}
 for k in ('bank_class','source','ply_band'):
  for val in sorted({x[k] for x in rows}):strata[f'{k}={val}']=agg([x for x in rows if x[k]==val])
 out={'schema':'deep-teacher-1600-vs-3200-convergence-v1','subset_sha256':hashlib.sha256(a.subset.read_bytes()).hexdigest(),'run1600':str(a.run1600.resolve()),'run3200':str(a.run3200.resolve()),'paired_records':len(rows),'strata':strata,'rows':rows,'interpretation':'descriptive stability only; no arbitrary strength or replacement claim'};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(a.output.resolve()),'paired_records':len(rows),'all':strata['all']},sort_keys=True))
if __name__=='__main__':main()
