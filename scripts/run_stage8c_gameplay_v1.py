#!/usr/bin/env python3
"""Minimal resumable paired C++/CUDA Stage-8C gameplay harness."""
from __future__ import annotations
import argparse,fcntl,hashlib,json,os,random,statistics,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[1];RUNNER=R/'build/cpp-puct-stage7/hex_candidate_match_runner';WRAP=R/'scripts/stage7_cuda12_runtime_v1.sh';MAP=R/'artifacts/cpp-swap-stage75-v1/katahex-opening-map-clean-v1/katahex-reference-map.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def validate_completed_pair(path, opening):
 d=json.loads(path.read_text())
 if d.get('schema')!='stage8c-pair-v1' or d.get('status')!='complete': raise RuntimeError(f'invalid completed pair: {path.name}')
 if d.get('pair_id')!=opening['pair_id'] or d.get('opening_moves')!=opening['opening_moves'] or d.get('swap_decision')!=opening['swap_decision']: raise RuntimeError(f'completed pair identity mismatch: {path.name}')
 for key in ('game_a','game_b'):
  game=d.get(key)
  if not isinstance(game,dict) or game.get('candidate_score') not in (0,1) or not isinstance(game.get('moves'),list) or not game['moves']:
   raise RuntimeError(f'malformed completed game: {path.name}:{key}')
 return d

def _effective_fpu(args, side):
 """Resolve the legacy global defaults into explicit per-side settings.

 The C++ match runner accepts per-side FPU settings as the gameplay
 contract.  The global Python options remain accepted for compatibility with
 older callers, but are never emitted on the child argv.
 """
 mode=getattr(args,f'{side}_fpu_mode')
 reduction=getattr(args,f'{side}_fpu_reduction')
 if mode is None: mode=args.fpu_mode
 if reduction is None: reduction=args.fpu_reduction
 return mode,reduction

def build_runner_command(args, out, lines):
 """Build the candidate-match argv without obsolete global FPU flags."""
 candidate_mode,candidate_reduction=_effective_fpu(args,'candidate')
 champion_mode,champion_reduction=_effective_fpu(args,'champion')
 cmd=[str(WRAP),str(RUNNER),'--candidate',str(args.candidate.resolve()),'--champion',str(args.champion.resolve()),'--openings',str(lines.resolve()),'--output',str((out/'pairs').resolve()),'--budget',str(args.budget),'--c-puct',str(args.c_puct),'--concurrency',str(args.concurrency),'--max-batch',str(args.max_batch),'--wait-us',str(args.wait_us),'--bridge-controller',args.bridge_controller,
      '--candidate-fpu-mode',candidate_mode,'--candidate-fpu-reduction',str(candidate_reduction),
      '--champion-fpu-mode',champion_mode,'--champion-fpu-reduction',str(champion_reduction)]
 for key,flag in (('candidate_budget','--candidate-budget'),('champion_budget','--champion-budget'),('candidate_c_puct','--candidate-c-puct'),('champion_c_puct','--champion-c-puct')):
  if getattr(args,key) is not None: cmd += [flag,str(getattr(args,key))]
 return cmd

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate-id',required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--champion',type=Path,required=True);p.add_argument('--openings',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',type=int,default=2048);p.add_argument('--c-puct',type=float,default=1.5);p.add_argument('--fpu-mode',choices=('zero','parent_value_reduced'),default='zero');p.add_argument('--fpu-reduction',type=float,default=0.0);p.add_argument('--candidate-budget',type=int);p.add_argument('--champion-budget',type=int);p.add_argument('--candidate-c-puct',type=float);p.add_argument('--champion-c-puct',type=float);p.add_argument('--candidate-fpu-mode',choices=('zero','parent_value_reduced'));p.add_argument('--champion-fpu-mode',choices=('zero','parent_value_reduced'));p.add_argument('--candidate-fpu-reduction',type=float);p.add_argument('--champion-fpu-reduction',type=float);p.add_argument('--concurrency',type=int,default=32);p.add_argument('--max-batch',type=int,default=96);p.add_argument('--wait-us',type=int,default=200);p.add_argument('--bridge-controller',choices=('off','shadow','active'),default='off');p.add_argument('--bootstrap-samples',type=int,default=20000);p.add_argument('--max-pairs',type=int,help='non-official bounded smoke only');p.add_argument('--diagnostic-only',action='store_true',help='evaluate but forcibly suppress promotion eligibility');a=p.parse_args();out=a.output.resolve();op=json.loads(a.openings.read_text());
 if a.max_pairs is not None: op={**op,'openings':op['openings'][:a.max_pairs]}
 cfg={'schema':'stage8c-gameplay-config-v1','candidate_id':a.candidate_id,'candidate':{'path':str(a.candidate.resolve()),'sha256':sha(a.candidate)},'champion':{'path':str(a.champion.resolve()),'sha256':sha(a.champion)},'openings':{'path':str(a.openings.resolve()),'sha256':sha(a.openings),'max_pairs':a.max_pairs},'katahex_map_sha256':sha(MAP),'search':{'budget':a.budget,'c_puct':a.c_puct,'fpu_mode':a.fpu_mode,'fpu_reduction':a.fpu_reduction,'concurrency':a.concurrency,'max_batch':a.max_batch,'wait_us':a.wait_us,'selection':'root_visit_argmax_lowest_action_tie'},'bridge_controller':a.bridge_controller,'evaluation_mode':'diagnostic_only' if a.diagnostic_only else 'promotion_protocol','promotion':'one-sided 95% bootstrap lower pair mean > 0.5'}
 if a.fpu_mode=='zero' and a.fpu_reduction==0.0: cfg['search'].pop('fpu_mode'); cfg['search'].pop('fpu_reduction')
 overrides={key:getattr(a,key) for key in ('candidate_budget','champion_budget','candidate_c_puct','champion_c_puct','candidate_fpu_mode','champion_fpu_mode','candidate_fpu_reduction','champion_fpu_reduction') if getattr(a,key) is not None}
 if overrides: cfg['search_overrides']=overrides
 if not out.exists():out.mkdir(parents=True);atomic(out/'config.json',cfg)
 elif json.loads((out/'config.json').read_text())!=cfg:raise RuntimeError('resume config/hash mismatch')
 lock=(out/'.stage8c-runner.lock').open('a+')
 try: fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError as e: raise RuntimeError('another Stage-8C runner owns this output root') from e
 lock.seek(0);lock.truncate();lock.write(f'pid={os.getpid()} started={time.time():.6f}\n');lock.flush()
 lines=out/'openings.txt'; content=''.join(f"{x['pair_id']}|{','.join(map(str,x['opening_moves']))}|{x['swap_decision']}\n" for x in op['openings'])
 if lines.exists() and lines.read_text()!=content: raise RuntimeError('resume opening manifest mismatch')
 if not lines.exists(): lines.write_text(content)
 completed=[]
 for opening in op['openings']:
  path=out/'pairs'/f"pair-{opening['pair_id']}.json"
  if path.exists(): completed.append(validate_completed_pair(path,opening))
 atomic(out/'resume-preflight.json',{'schema':'stage8c-resume-preflight-v1','validated_completed_pairs':len(completed),'requested_pairs':len(op['openings']),'config_sha256':sha(out/'config.json')})
 cmd=build_runner_command(a,out,lines)
 stdout_path,stderr_path=out/'runner.stdout.log',out/'runner.stderr.log'
 with stdout_path.open('ab') as stdout, stderr_path.open('ab') as stderr:
  banner=(f'\n=== stage8c runner start {time.time():.6f} ===\n').encode();stdout.write(banner);stderr.write(banner);stdout.flush();stderr.flush()
  process=subprocess.Popen(cmd,cwd=R,stdout=stdout,stderr=stderr)
  rc=process.wait()
 atomic(out/'runner-exit.json',{'schema':'stage8c-runner-exit-v1','exit_code':rc,'command':cmd,'stdout_log':str(stdout_path),'stderr_log':str(stderr_path)})
 if rc:
  tail=stderr_path.read_text(errors='replace')[-1000:]
  raise RuntimeError(f'C++ match runner failed ({rc}); see {stderr_path}: {tail}')
 try:
  telemetry=json.loads(next(line for line in reversed(stdout_path.read_text(errors='replace').splitlines()) if line.strip().startswith('{')))
 except json.JSONDecodeError as e:
  raise RuntimeError(f'C++ match runner produced malformed telemetry: {e}') from e
 if telemetry.get('status')!='complete':raise RuntimeError('C++ match runner did not report completion')
 rows=[];pair_records=[]
 for x in op['openings']:
  f=out/'pairs'/f"pair-{x['pair_id']}.json";d=json.loads(f.read_text());
  if d.get('status')!='complete' or d['opening_moves']!=x['opening_moves'] or d['swap_decision']!=x['swap_decision']:raise RuntimeError(f'malformed pair {x["pair_id"]}')
  scores=[d['game_a']['candidate_score'],d['game_b']['candidate_score']]
  if any(v not in (0,1) for v in scores):raise RuntimeError('invalid Hex result')
  rows.append(sum(scores)/2);pair_records.append(d)
 mean=sum(rows)/len(rows);rng=random.Random(8202601);boots=sorted(sum(rng.choice(rows) for _ in rows)/len(rows) for _ in range(a.bootstrap_samples));lcb=boots[max(0,int(.05*a.bootstrap_samples)-1)]
 game_records=[game for pair in pair_records for game in (pair['game_a'],pair['game_b'])]
 controller_games=[game['bridge_controller'] for game in game_records]
 controller_summary={'mode':a.bridge_controller,'detector_calls':sum(x['detector_calls'] for x in controller_games),'detector_seconds':sum(x['detector_seconds'] for x in controller_games),'controller_selected_moves':sum(game['controller_selected_moves'] for pair in pair_records for game in (pair['game_a'],pair['game_b']))}
 for colour in ('black','white'):
  controller_summary[colour]={'certificates':sum(x[colour]['first_certificate_ply'] is not None for x in controller_games),'required_responses':sum(x[colour]['required_responses'] for x in controller_games),'ignored_required_responses':sum(x[colour]['ignored_required_responses'] for x in controller_games),'paired_responses':sum(x[colour]['successful_paired_responses'] for x in controller_games),'proactive_resolutions':sum(x[colour]['proactive_resolutions'] for x in controller_games),'fail_closed_events':sum(x[colour]['fail_closed_events'] for x in controller_games)}
 tails=[x[colour]['certificate_to_literal_tail'] for x in controller_games for colour in ('black','white') if x[colour]['certificate_to_literal_tail'] is not None]
 controller_summary['certificate_tail_mean']=statistics.mean(tails) if tails else None
 controller_summary['certificate_tail_max']=max(tails) if tails else None
 official=a.max_pairs is None
 reason='diagnostic_only; never promotion eligible' if a.diagnostic_only else (None if official else 'non_official_bounded_run')
 atomic(out/'summary.json',{'schema':'stage8c-gameplay-summary-v1','official':official,'evaluation_mode':cfg['evaluation_mode'],'games':2*len(rows),'wins':int(sum(rows)*2),'losses':2*len(rows)-int(sum(rows)*2),'raw_win_rate':mean,'paired_mean_score':mean,'bootstrap_one_sided_95_lcb':lcb,'promotion_qualified':official and not a.diagnostic_only and lcb>.5,'promotion_reason':reason,'config_sha256':sha(out/'config.json'),'pairs':len(rows),'mean_game_length':statistics.mean(len(game['moves']) for game in game_records),'runner_telemetry':telemetry,'bridge_controller':controller_summary})
 print(json.dumps({'pairs':len(rows),'mean':mean,'lcb':lcb}))
if __name__=='__main__':main()
