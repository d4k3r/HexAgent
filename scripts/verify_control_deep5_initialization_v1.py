#!/usr/bin/env python3
"""Compare parent and candidate initial model tensors, never optimizer state."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import torch

def fingerprint(path: Path) -> str:
    payload=torch.load(path,map_location='cpu',weights_only=False); state=payload.get('model_state') or payload.get('state_dict'); h=hashlib.sha256()
    for name in sorted(state):
        value=state[name].detach().cpu().contiguous(); raw=value.numpy().tobytes(order='C'); descriptor=json.dumps({'name':name,'shape':list(value.shape),'dtype':str(value.dtype),'nbytes':len(raw)},sort_keys=True,separators=(',',':')); h.update(descriptor.encode());h.update(b'\0');h.update(raw)
    return h.hexdigest()
def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--control-initial',type=Path,required=True);p.add_argument('--deep5-initial',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args(); values={'parent':fingerprint(a.parent),'control_initial':fingerprint(a.control_initial),'deep5_initial':fingerprint(a.deep5_initial)}; report={'schema':'control-deep5-initialization-audit-v1','fingerprints':values,'passed':len(set(values.values()))==1};a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');tmp.replace(a.output);print(json.dumps(report,sort_keys=True));
    if not report['passed']: raise SystemExit(2)
if __name__=='__main__':main()
