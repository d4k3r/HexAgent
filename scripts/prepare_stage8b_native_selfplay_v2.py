#!/usr/bin/env python3
"""Materialise accepted native-v2 Phase-A rows for the existing FP32 trainer.

This is deliberately a new immutable prepared root.  It never rewrites the
native corpus and it never synthesises policy rows from Phase B.
"""
from __future__ import annotations
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
from hex_reconstruction.board import HexBoard, BLACK, WHITE

R=Path(__file__).resolve().parents[1]
OLD=R/'artifacts/stage8b-v1/prepared-data-fp32-v1'
def sha(p:Path):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def atomic(p:Path,x:object):
 t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,sort_keys=True,indent=2)+'\n');os.replace(t,p)
def good(path:Path, manifest:dict):
 x=json.loads(path.read_text());expected='hex-native-selfplay-game-v3' if manifest.get('schema_version')==3 else 'hex-native-selfplay-game-v2';return x.get('schema')==expected and x.get('status')=='accepted' and x.get('run_id')==manifest['run_id'] and x.get('config_sha256')==manifest['config_sha256'] and x.get('model_sha256')==manifest['champion']['onnx_sha256'] and x.get('phase_a_rows')==len(x.get('samples',[]))
def main():
 p=argparse.ArgumentParser();p.add_argument('--native-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit-games',type=int);a=p.parse_args();root=a.native_root.resolve();out=a.output.resolve()
 if out.exists():raise RuntimeError('refusing existing prepared root')
 manifest=json.loads((root/'run-manifest.json').read_text()); files=sorted((root/'games').glob('game-*.json'))
 if a.limit_games is not None:files=files[:a.limit_games]
 games=[]
 for f in files:
  if not good(f,manifest):raise RuntimeError(f'invalid accepted native game {f}')
  games.append(json.loads(f.read_text()))
 if not games:raise RuntimeError('no accepted native games')
 n=sum(g['phase_a_rows'] for g in games);tmp=out.with_name(out.name+'.partial');sp=tmp/'selfplay';sp.mkdir(parents=True)
 state=np.lib.format.open_memmap(sp/'state.npy',mode='w+',dtype=np.float32,shape=(n,6,121));pi=np.lib.format.open_memmap(sp/'pi.npy',mode='w+',dtype=np.float32,shape=(n,121));z=np.lib.format.open_memmap(sp/'z.npy',mode='w+',dtype=np.float32,shape=(n,));gameidx=np.lib.format.open_memmap(sp/'game_index.npy',mode='w+',dtype=np.int32,shape=(n,));ply=np.lib.format.open_memmap(sp/'source_ply.npy',mode='w+',dtype=np.int16,shape=(n,)); at=0;entries=[]
 for gi,g in enumerate(games):
  b=HexBoard();begin=at
  forced = g.get('forced_prefix_actions', []) if g.get('schema') == 'hex-native-selfplay-game-v3' else []
  if forced:
   if g.get('prefix_mode') != 'forced' or len(forced) != 3 or g.get('forced_prefix_length') != 3: raise RuntimeError(f'forced prefix metadata mismatch game {g["game_id"]}')
   for action in forced:
    if not b.is_legal(int(action)): raise RuntimeError(f'illegal forced action game {g["game_id"]}: {action}')
    b.play(int(action))
  elif g.get('schema') == 'hex-native-selfplay-game-v3' and g.get('prefix_mode') != 'normal':
   raise RuntimeError(f'unknown prefix mode game {g["game_id"]}')
  for i,s in enumerate(g['samples']):
   expected_ply = len(forced) + i
   if s['ply']!=expected_ply or s['side_to_move'] != ('B' if b.side_to_move==BLACK else 'W') or not b.is_legal(s['selected_move']):raise RuntimeError(f'state/sample mismatch game {g["game_id"]} ply {expected_ply}')
   visits=s['root_visits'];
   if len(visits)!=121 or sum(visits)!=g['search_budget'] or visits[s['selected_move']]<=0:raise RuntimeError('root visits invalid')
   state[at]=np.asarray(b.feature_planes(),dtype=np.float32);pi[at]=np.asarray(visits,dtype=np.float32)/float(sum(visits));z[at]=np.float32(s['z']);gameidx[at]=gi;ply[at]=i;at+=1;b.play(s['selected_move'])
  if at-begin!=g['phase_a_rows']:raise RuntimeError('phase A boundary mismatch')
  entries.append({'entry_index':gi,'game_id':g['game_id'],'game_seed':g['game_seed'],'source_path':str((root/'games'/f"game-{g['game_id']}.json").resolve()),'prepared_begin':begin,'prepared_end':at,'retained_rows':g['phase_a_rows'],'literal_winner':g['literal_winner'],'classification':g['classification'],'prefix_mode':g.get('prefix_mode','normal'),'prefix_id':g.get('prefix_id')})
 for x in (state,pi,z,gameidx,ply):x.flush()
 names=('state.npy','pi.npy','z.npy','game_index.npy','source_ply.npy');rec={'rows':at,'entry_order':entries,'arrays':{'state':'state.npy','pi':'pi.npy','z':'z.npy','game_index':'game_index.npy','source_ply':'source_ply.npy'},'array_sha256':{name:sha(sp/name) for name in names}}
 os.symlink((OLD/'teacher').resolve(),tmp/'teacher');old=json.loads((OLD/'prepared-manifest.json').read_text());
 m={'schema':'stage8b-prepared-fp32-native-selfplay-v2','semantics':'native clean Phase-A only; full 121 soft pi from original root visits; z verified from LiteralWinner; no Phase-B rows; forced-prefix moves are never materialized as policy rows','limited_fixture_rows_per_source':a.limit_games,'parent_inputs':{'native_run_manifest':{'path':str((root/'run-manifest.json').resolve()),'sha256':sha(root/'run-manifest.json')},'native_postrun_audit':str((root/'postrun-audit.json').resolve()) if (root/'postrun-audit.json').exists() else None,'teacher_prepared_manifest':{'path':str((OLD/'prepared-manifest.json').resolve()),'sha256':sha(OLD/'prepared-manifest.json')}},'sources':{'teacher':old['sources']['teacher'],'selfplay':rec}}
 atomic(tmp/'prepared-manifest.json',m);os.replace(tmp,out);audit={'schema':'stage8b-native-prepared-audit-v2','passed':at==n,'prepared_manifest_sha256':sha(out/'prepared-manifest.json'),'teacher_rows':old['sources']['teacher']['rows'],'selfplay_rows':at,'native_games':len(games),'limited_fixture_games':a.limit_games};atomic(out/'prepared-audit-v3.json',audit);print(json.dumps({'output':str(out),'rows':at,'games':len(games),'passed':True}))
if __name__=='__main__':main()
