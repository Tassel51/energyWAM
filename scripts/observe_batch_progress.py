"""Read-only wall-clock progress trace for the already running paired jobs."""
import json,time,os,argparse
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
parser=argparse.ArgumentParser();parser.add_argument('--plan',default='batch_size_comparison_plan.json');args=parser.parse_args()
plan=read(R/args.plan);seen={}
for _ in range(2880):
 finished=0
 for job in plan['jobs']:
  folder=R/job['name'];resource=read(folder/'resource_summary.json');speed=read(folder/'speed_report.json');pair=job['physical_gpus'].replace(',','');state=read(R/f'search_status_{pair}.json').get(job['name'],{})
  now=time.time();start=resource.get('started_unix',now);batch=job.get('effective_batch',2 if pair=='01' else 8)
  step=speed.get('completed_steps',0)
  row=dict(unix=now,wall_seconds=now-start,completed_updates=step,completed_sample_visits=step*batch,phase=state.get('phase'),pending_checkpoint=(folder/'ckpt_pending.pt').exists())
  if not folder.exists():continue
  with (folder/'progress_trace.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
  checks=seen.setdefault(job['name'],{})
  for path in folder.glob('eval_*_action.json'):
   u=int(path.name.split('_')[1]);other=folder/f'eval_{u:04d}_future.json'
   if str(u) not in checks and other.exists():
    a,b=read(path),read(other)
    if a.get('rows') and b.get('rows'):checks[str(u)]=dict(update=u,sample_visits=u*batch,evaluation_complete_wall_seconds=now-start,observation_interval_seconds=5)
  tmp=folder/'progress_summary.pending.json';tmp.write_text(json.dumps(dict(started_unix=start,evaluation_checkpoints=checks,last_progress=row),indent=2));tmp.replace(folder/'progress_summary.json')
  if state.get('phase') in ['complete','failed']:finished+=1
 if finished==len(plan['jobs']):print('Both paired jobs terminal; progress trace finalized.',flush=True);break
 time.sleep(5)
else:print('Progress observation deadline exceeded; training was not stopped or restarted.',flush=True)
