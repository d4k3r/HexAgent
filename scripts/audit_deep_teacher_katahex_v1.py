#!/usr/bin/env python3
"""Audit committed deep-teacher result records."""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();r=a.root.resolve();m=json.loads((r/'manifest.json').read_text());files=list((r/'results').glob('position-*.json'));by={json.loads(f.read_text())['position_id']:f for f in files};expected=set(m['position_ids']);errors=[]
 if set(by)!=expected: errors.append({'missing':sorted(expected-set(by))[:20],'extra':sorted(set(by)-expected)[:20]})
 for pid,f in by.items():
  try:
   x=json.loads(f.read_text());
   if x.get('bank_sha256')!=m['bank_sha256'] or x.get('requested_max_visits')!=m['budget']:errors.append({'position_id':pid,'reason':'identity'})
   if not x.get('teacher_terminal') and (len(x.get('raw_visits',[]))!=121 or sum(x['raw_visits'])!=x.get('actual_physical_root_visits') or abs(sum(x.get('pi',[]))-1)>1e-8):errors.append({'position_id':pid,'reason':'policy'})
  except Exception as e: errors.append({'position_id':pid,'reason':str(e)})
 out={'schema':'deep-teacher-katahex-audit-v1','manifest':str((r/'manifest.json').resolve()),'manifest_sha256':sha(r/'manifest.json'),'budget':m['budget'],'requested':len(expected),'committed':len(by),'missing':len(expected-set(by)),'duplicate_ids':len(files)-len(by),'errors':errors,'complete':not errors and len(by)==len(expected),'attempts':sum(1 for _ in (r/'attempts.jsonl').open()) if (r/'attempts.jsonl').exists() else 0};path=a.output or r/'postrun-audit.json';path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out,sort_keys=True));
 if errors or not out['complete']: raise SystemExit(1)
if __name__=='__main__':main()
