#!/usr/bin/env python3
"""Strict read-only validator for the frozen deep-teacher bank/subset."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from hex_reconstruction.board import HexBoard

def sha(p: Path) -> str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bank',type=Path,required=True);p.add_argument('--subset',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();bank=json.loads(a.bank.read_text()); rows=bank.get('positions',[]); errors=[]
 if len(rows)!=4096: errors.append(f'positions={len(rows)}')
 if sum(r.get('bank_class')=='BROAD' for r in rows)!=2048: errors.append('broad count')
 if sum(r.get('bank_class')=='HARD' for r in rows)!=2048: errors.append('hard count')
 if len({r.get('state_sha256') for r in rows})!=len(rows): errors.append('exact duplicate state')
 for r in rows:
  try:
   b=HexBoard.from_setup(r['black_actions'],r['white_actions'],side_to_move=r['side_to_move'].lower(),last_move=r.get('last_move'))
   if b.feature_planes()!=[r['state_flat'][i*121:(i+1)*121] for i in range(6)]: errors.append(f'encoding {r["position_id"]}')
   if len(set(r['black_actions'])&set(r['white_actions'])): errors.append(f'overlap {r["position_id"]}')
   if not b.legal_actions(): errors.append(f'non-live {r["position_id"]}')
   if r.get('source') not in ('NORMAL','FORCED'): errors.append(f'source {r["position_id"]}')
  except Exception as e: errors.append(f'{r.get("position_id")}: {e}')
 out={'schema':'deep-teacher-1600-bank-validation-v1','bank':str(a.bank.resolve()),'bank_sha256':sha(a.bank),'positions':len(rows),'broad':sum(r.get('bank_class')=='BROAD' for r in rows),'hard':sum(r.get('bank_class')=='HARD' for r in rows),'exact_state_duplicates':len(rows)-len({r.get('state_sha256') for r in rows}),'transpose_orbits':len({r.get('transpose_orbit') for r in rows}),'errors':errors,'passed':not errors}
 if a.subset:
  s=json.loads(a.subset.read_text()); ids={r['position_id'] for r in rows}; sp=s['positions']; out['subset']={'path':str(a.subset.resolve()),'sha256':sha(a.subset),'count':len(sp),'broad':sum(r.get('bank_class')=='BROAD' for r in sp),'hard':sum(r.get('bank_class')=='HARD' for r in sp),'all_in_bank':all(r['position_id'] in ids for r in sp),'unique':len({r['position_id'] for r in sp})==len(sp)}
  if len(sp)!=512 or out['subset']['broad']!=256 or out['subset']['hard']!=256 or not out['subset']['all_in_bank'] or not out['subset']['unique']: errors.append('subset invalid');out['passed']=False
 if a.output: a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
 print(json.dumps(out,indent=2,sort_keys=True));
 if errors: raise SystemExit(1)
if __name__=='__main__':main()
