"""Safely retire only seed43 after a committed checkpoint, then run two short arms."""
import json,os,subprocess,sys,time,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=Path('/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106')
OLD=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106')
D=OLD/'three_task_b16_micro4_seed43'
def read(p):return json.loads(p.read_text())
def write(p,d):
 q=p.with_suffix('.pending.json');q.write_text(json.dumps(d,indent=2));q.replace(p)
def summarize():
 result={'status':'terminal','arms':[],'primary':'Matched50update unweighted MSE; different total loss scales not compared.'}
 queue=read(R/'queue_status.json')
 hashes=[]
 for weight in [1,9]:
  name=f'tiny16_futureweight{weight}_b16_micro4_seed42';folder=R/name;state=queue.get(name,{})
  row={'weight':weight,'queue_state':state.get('phase'),'completed_updates':state.get('completed_updates',0),'trajectory':{}}
  ini=folder/'initialization_step0.json'
  if ini.exists():hashes.append(read(ini)['sampled_parameter_sha256'])
  for split,path in [('val',folder),('train',folder/'train_monitor')]:
   for p in sorted(path.glob('eval_*.json')):
    data=read(p);z=[x for x in data['rows'] if x['k']==2 and x['split']==split]
    if z:row['trajectory'].setdefault(str(z[0]['update']),{})[split+'_'+z[0]['direction']]=statistics.mean(x['prediction_mse'] for x in z)
  result['arms'].append(row)
 result['initial_parameter_sample_hashes_match']=len(hashes)==2 and len(set(hashes))==1
 result['both_completed']=all(x['queue_state']=='complete' and x['completed_updates']==50 for x in result['arms'])
 write(R/'comparison_report.json',result)
def main():
 import psutil,shutil
 mark=R/'seed43_pause_record.json'
 if not mark.exists():
  base=read(R/'pause_request.json');start=time.monotonic()
  while True:
   meta=read(D/'ckpt_latest.pt.meta.json');saved=meta['meta']['global_step'];history=[json.loads(l) for l in (D/'checkpoint_history.jsonl').read_text().splitlines() if l.strip()]
   committed=any(x['global_step']==saved for x in history)
   status=read(OLD/'status_43.json')['three_task_b16_micro4_seed43']
   write(R/'controller_status.json',{'phase':'waiting_for_next_seed43_checkpoint','observed_saved_update':saved,'require_newer_than':base['saved_update_at_request'],'seed43_training_pid':status.get('pid'),'updated_unix':time.time()})
   if saved>base['saved_update_at_request'] and committed:break
   if time.monotonic()-start>2700:raise RuntimeError('No new verified seed43 checkpoint within45min; not force-killing unsaved work')
   if not psutil.pid_exists(status['pid']):raise RuntimeError('seed43 exited before safe pause; inspect before replacement')
   time.sleep(2)
  child_pid=status['pid'];process=psutil.Process(child_pid)
  if not any('train_shared_sharded.py' in x for x in process.cmdline()):raise RuntimeError('Unexpected process identity')
  saved_bytes=(D/'ckpt_latest.pt').stat().st_size
  if saved_bytes<20*2**30:raise RuntimeError('Unexpected snapshot size')
  # Supervisor signal delivers TERM without its short stop timeout; the queue
  # handles it and terminates only its owned child session after its poll.
  subprocess.run(['supervisorctl','signal','TERM','energywam-three-task-43'],check=True)
  for _ in range(60):
   if not psutil.pid_exists(child_pid):break
   time.sleep(2)
  else:raise RuntimeError('seed43 child still alive; refusing new GPU work')
  latest=read(D/'ckpt_latest.pt.meta.json')['meta']['global_step'];done=read(D/'speed_report.json')['completed_steps']
  record={'status':'paused_by_user_for_weight_experiment','paused_unix':time.time(),'training_pid':child_pid,'recoverable_latest_update':latest,'latest_bytes':saved_bytes,'last_completed_update':done,'completed_updates_not_in_snapshot':max(0,done-latest),'weights_deleted':False,'seed42_untouched':True}
  write(mark,record)
  target=ROOT/'configs/experiments/three_task_full_followup_target_20261105.json';d=read(target);d['status']='seed42_continues_seed43_paused_for_tiny_weight_comparison';d['seed43_pause_record']=str(mark);write(target,d)
 if shutil.disk_usage('/cloud').free<120*2**30:raise RuntimeError('Need120GiB free for two retained arms and serialized writes')
 write(R/'controller_status.json',{'phase':'running_serial_weight_comparison','started_unix':time.time()})
 plan=read(R/'plan.json');plan['status']='launched';write(R/'plan.json',plan)
 code=subprocess.call([sys.executable,'-u',str(ROOT/'scripts/run_iteration_search.py'),'--jobs',str(R/'jobs.json'),'--status',str(R/'queue_status.json')],cwd=ROOT)
 if code:raise RuntimeError(f'Queue exited{code}')
 summarize();write(R/'controller_status.json',{'phase':'comparison_terminal_review_results','report':str(R/'comparison_report.json'),'updated_unix':time.time()})
if __name__=='__main__':main()
