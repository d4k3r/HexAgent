#!/usr/bin/env python3
"""Freeze the pre-label 1600/3200 convergence subset."""
from __future__ import annotations
import argparse, hashlib, json, os
from collections import defaultdict
from pathlib import Path

def sha(p: Path) -> str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
 if out.exists(): raise RuntimeError(f'refusing existing subset {out}')
 bank=json.loads(a.bank.read_text()); rows=bank['positions']; assert len(rows)==4096
 def choose(cls):
  pool=[r for r in rows if r['bank_class']==cls]; groups=defaultdict(list)
  for r in pool: groups[(r['source'],r['side_to_move'],r['ply_band'])].append(r)
  for g in groups.values(): g.sort(key=lambda r: hashlib.sha256((r['position_id']+r['state_sha256']).encode()).hexdigest())
  chosen=[]; keys=sorted(groups)
  while len(chosen)<256:
   progress=False
   for k in keys:
    if groups[k]: chosen.append(groups[k].pop(0)); progress=True
    if len(chosen)>=256: break
   if not progress: break
  if len(chosen)!=256: raise RuntimeError(f'{cls} subset only {len(chosen)}')
  return chosen
 chosen=choose('BROAD')+choose('HARD'); chosen.sort(key=lambda r:r['position_id']); payload={'schema':'deep-teacher-1600-convergence-subset-v1','selection_before_teacher_targets':True,'bank_path':str(a.bank.resolve()),'bank_sha256':sha(a.bank),'positions':[{"position_id":r['position_id'],"bank_class":r['bank_class'],"source":r['source'],"source_game_id":r['source_game_id'],"source_ply":r['source_ply'],"side_to_move":r['side_to_move'],"ply_band":r['ply_band'],"student_hard_score":r.get('hard_score')} for r in chosen], 'counts':{'total':512,'BROAD':256,'HARD':256},'budgets':[1600,3200]}
 out.mkdir(parents=True);tmp=out/'subset.json.partial';tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');os.replace(tmp,out/'subset.json');manifest={'schema':'deep-teacher-1600-convergence-subset-manifest-v1','subset_path':str((out/'subset.json').resolve()),'subset_sha256':sha(out/'subset.json'),'bank_sha256':payload['bank_sha256'],'positions':512,'broad':256,'hard':256,'selection_rule':'deterministic round-robin across source, side and ply-band strata; selected before any 1600/3200 result exists'}; (out/'subset-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({**manifest,'output':str(out)},sort_keys=True))
if __name__=='__main__': main()
