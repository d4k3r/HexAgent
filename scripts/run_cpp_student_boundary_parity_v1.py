#!/usr/bin/env python3
"""Exact C++/Python Student-state parity plus PyTorch Stage-3 golden export."""
from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from hex_reconstruction.board import BLACK,WHITE,BOARD_AREA,HexBoard,action_to_gtp
from hex_reconstruction.puct import TorchStudentEvaluator
from hex_reconstruction.symmetry import transpose_action,transpose_vector

CHECKPOINT=ROOT/'artifacts/student-training-value-symmetry-v1/expanded-seed4901-symmetry-24epochs-final/checkpoints/best-validation-policy.pt'
CHECKPOINT_SHA='b02a60ed210450d4159db28c176aa9a44e01579488e6e3d2d77438784ba481d0'
BANK=(
 ('empty_black',[],[],BLACK,None),('empty_white',[],[],WHITE,None),('first_move_white',[60],[],WHITE,60),
 ('edge_components_black',[0,11,22,33],[1,12,23],BLACK,23),('midgame_white',[0,12,24,36,48],[1,13,25,37],WHITE,37),
 ('swap_ownership_black',[0],[1],BLACK,1),('colour_transpose_black',[0,12,25,37],[1,13,24,36],BLACK,36),
 ('colour_transpose_white',[11,23,24,36],[0,12,35,47],WHITE,36),
 ('literal_terminal_black',[0,11,22,33,44,55,66,77,88,99,110],[1,12,23,34,45,56,67,78,89,100],WHITE,110),
 ('late_dense',[0,2,4,6,8,10,13,15,17,19,21,23,25,27,29,31,33,35,37,39,41,43,45,47,49,51,53,55,57,59,61,63,65,67,69,71,73,75,77,79,81,83,85,87,89,91,93,95,97,99,101,103,105,107,109],[1,3,5,7,9,11,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108],BLACK,109),
)
def board(spec): return HexBoard.from_setup(spec[1],spec[2],side_to_move=spec[3],last_move=spec[4])
def flat(b): return [v for plane in b.feature_planes() for v in plane]
def fail(name,got,want):
 for i,(a,b) in enumerate(zip(got,want)):
  if a!=b:
   plane,rowcol=divmod(i,BOARD_AREA);row,col=divmod(rowcol,11)
   raise AssertionError(f'{name}: plane={plane} row={row} col={col}: Python={b} C++={a}')
 raise AssertionError(f'{name}: tensor length mismatch')
def main():
 exe=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'build/cpp-puct-stage1/hex_student_boundary_runner'
 cpp=json.loads(subprocess.check_output([str(exe)],text=True)); assert len(cpp['cases'])==len(BANK)
 entries=[]
 for case,spec in zip(cpp['cases'],BANK):
  b=board(spec); expected=flat(b); assert case['name']==spec[0]
  if case['tensor']!=expected: fail(spec[0],case['tensor'],expected)
  assert case['legal_mask']==[int(v) for v in b.legal_mask()],spec[0]
  assert case['terminal_winner']==b.literal_winner(),spec[0]
  entries.append((spec[0],b,expected))
 for action,row,col,gtp in cpp['actions']:
  assert action==row*11+col and 0<=row<11 and 0<=col<11
  assert action_to_gtp(action)==gtp and transpose_action(transpose_action(action))==action
 # Semantic encoding symmetry, independent of neural output equivariance.
 by_name={name:(b,tensor) for name,b,tensor in entries}; original=by_name['colour_transpose_black']; transformed=by_name['colour_transpose_white']
 old=[original[1][p*121:(p+1)*121] for p in range(6)]; new=[transformed[1][p*121:(p+1)*121] for p in range(6)]
 assert new[0]==transpose_vector(old[1]) and new[1]==transpose_vector(old[0])
 assert new[2]==[1-v for v in transpose_vector(old[2])]
 assert new[3]==transpose_vector(old[3]) and new[4]==transpose_vector(old[4]) and new[5]==transpose_vector(old[5])
 assert transpose_vector(list(range(121)))[transpose_action(17)]==17
 assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==CHECKPOINT_SHA
 evaluator=TorchStudentEvaluator(CHECKPOINT)
 golden=[]
 for name,b,tensor in entries:
  evaluation=evaluator.evaluate(b)
  golden.append({'position_id':name,'input_sha256':hashlib.sha256(bytes(tensor)).hexdigest(),'input':tensor,'legal_mask':b.legal_mask(),'literal_winner':b.literal_winner(),'policy_logits':list(evaluation.policy_logits),'value':evaluation.value})
 output=ROOT/'artifacts/cpp-student-boundary-v1/student-golden-outputs-v1.json'; output.parent.mkdir(parents=True,exist_ok=True)
 payload={'schema_version':'cpp-student-golden-outputs-v1','checkpoint':str(CHECKPOINT.relative_to(ROOT)),'checkpoint_sha256':CHECKPOINT_SHA,'checkpoint_epoch':11,'selection_manifest':'provenance/student-value-symmetry-checkpoint-selection-v1.json','selection_role':'seed 4901 / best_validation_policy','model_source':{'encoder':'src/hex_reconstruction/board.py:HexBoard.feature_planes','forward':'src/hex_reconstruction/student_training.py:Group49Student','bridge':'src/hex_reconstruction/puct.py:TorchStudentEvaluator'},'architecture':{'architecture':'group49-final-residual-policy-value','input_shape':[6,11,11],'policy_logits':121,'value':'tanh scalar','channels':256,'residual_blocks':10},'bank':'cpp-student-boundary-bank-v1','action_space':'121 physical row-major actions; no controls','entries':golden}
 output.write_text(json.dumps(payload,indent=2)+'\n')
 print(f'C++/Python tensor parity passed: {len(entries)} positions x 726 discrete elements; 121 action mappings passed. Golden outputs: {output}')
if __name__=='__main__': main()
