#!/usr/bin/env python3
"""Dynamic-batch ONNX export for an immutable Stage-8B candidate checkpoint."""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hex_reconstruction.student_training import Group49Student

def sha(p: Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def main() -> None:
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists(): raise RuntimeError(f'refusing existing ONNX output: {a.output}')
 payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False); arch=payload['config']['architecture'];m=Group49Student(channels=arch['channels'],blocks=arch['residual_blocks']);m.load_state_dict(payload['model_state']);m.eval();a.output.parent.mkdir(parents=True,exist_ok=True)
 torch.onnx.export(m,torch.zeros((1,6,11,11),dtype=torch.float32),a.output,input_names=['state'],output_names=['policy_logits','value'],opset_version=17,dynamic_axes={'state':{0:'B'},'policy_logits':{0:'B'},'value':{0:'B'}},training=torch.onnx.TrainingMode.EVAL,dynamo=False)
 meta={'schema':'stage8b-candidate-dynamic-onnx-v1','checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':sha(a.checkpoint),'onnx':str(a.output.resolve()),'onnx_sha256':sha(a.output),'architecture':arch,'shapes':{'input':['B',6,11,11],'policy':['B',121],'value':['B']}}
 q=a.output.with_suffix('.provenance.json');q.write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n');print(json.dumps(meta))
if __name__=='__main__':main()
