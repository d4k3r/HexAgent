#!/usr/bin/env python3
"""Path-parameterized CPU PyTorch/ONNX parity for a Stage-8B candidate."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from hex_reconstruction.board import HexBoard
from hex_reconstruction.student_training import Group49Student

TOL = 2e-5
def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def states() -> np.ndarray:
    rows=[]
    for moves in ((), (0,), (0,1,20,2,40), (60,1,72,2,84,3)):
        b=HexBoard()
        for move in moves: b.play(move)
        rows.append(np.asarray(b.feature_planes(), dtype=np.float32))
    return np.asarray(rows, dtype=np.float32).reshape(-1,6,11,11)
def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--onnx',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False);arch=payload['config']['architecture'];model=Group49Student(channels=arch['channels'],blocks=arch['residual_blocks']);model.load_state_dict(payload['model_state']);model.eval();x=states()
    with torch.no_grad(): tp,tv=model(torch.from_numpy(x))
    session=ort.InferenceSession(str(a.onnx),providers=['CPUExecutionProvider']);op,ov=session.run(['policy_logits','value'],{'state':x});pd=np.abs(tp.numpy()-op);vd=np.abs(tv.numpy().reshape(-1)-np.asarray(ov).reshape(-1));report={'schema':'stage8b-candidate-onnx-cpu-parity-v1','checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha(a.checkpoint),'onnx':str(a.onnx.resolve()),'onnx_sha256':sha(a.onnx),'state_count':len(x),'max_policy_logit_difference':float(pd.max()),'max_value_difference':float(vd.max()),'policy_argmax_agreement':bool(np.all(np.argmax(tp.numpy(),1)==np.argmax(op,1))),'tolerance_abs':TOL,'passed':bool(pd.max()<=TOL and vd.max()<=TOL and np.all(np.argmax(tp.numpy(),1)==np.argmax(op,1)))}
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix('.partial');tmp.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');tmp.replace(a.output);print(json.dumps(report,sort_keys=True));
    if not report['passed']: raise SystemExit(2)
if __name__=='__main__': main()
