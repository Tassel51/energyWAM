"""Read-only live progress and resource attribution for the paired batch trial."""
import json,time,argparse
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def show():
 plan=read(R/'batch_size_comparison_plan.json')
 jobs=plan.get('jobs',[])+plan.get('failure_recheck',{}).get('jobs',[])
 if plan.get('batch4_candidate',{}).get('job'):jobs+=[plan['batch4_candidate']['job']]
 for job in jobs:
  pair=job['physical_gpus'].replace(',','');state=read(R/f'search_status_{pair}.json').get(job['name'],{})
  folder=R/job['name'];speed=read(folder/'speed_report.json');resources=read(folder/'resource_summary.json');row=resources.get('last_sample',{})
  batch=job.get('effective_batch',2 if pair=='01' else 8);visits=speed.get('completed_steps',0)*batch;total=job.get('sample_budget',plan['sample_budget_each']);bars=int(24*min(1,visits/total))
  print(f"B{batch} GPU{job['physical_gpus']} [{('='*bars).ljust(24)}] {visits}/{total} samples | {state.get('phase','prepared')} | {job['name']}")
  print('  Last sampled CPU: %.1f%% (100%%=1 core), process RSS %.2f GiB, USS %.2f GiB' % (row.get('process_cpu_percent_one_core',0),row.get('process_rss_gib',0),row.get('process_uss_gib',0)))
  peaks=resources.get('peaks',{});gpu_peaks=resources.get('gpu_peaks',{})
  print('  Sampled peaks: process RSS %.2f GiB; GPU memory %s GiB' % (peaks.get('process_rss_gib',0),'/'.join('%.2f'%(g.get('memory_used_mib',0)/1024) for g in gpu_peaks.values())))
  pending=folder/'ckpt_pending.pt'
  if pending.exists():
   try:print('  Saving checkpoint: %.2f GiB written (updates may already be complete)'%(pending.stat().st_size/2**30))
   except FileNotFoundError:pass
  print('  Container: current %.2f GiB, anon %.2f GiB, file cache %.2f GiB (shared, not all caused by this job)' % ((row.get('cgroup_memory_current') or 0)/2**30,row.get('cgroup_anon_gib',0),row.get('cgroup_file_gib',0)))
  for gpu in row.get('gpus',[]):print('  GPU%d: %.2f/%.2f GiB, util %.0f%%, power %.0f W' % (gpu['index'],gpu['memory_used_mib']/1024,gpu['memory_total_mib']/1024,gpu['utilization_percent'],gpu['power_w']))
  if state.get('reason') or resources.get('exit_classification'):print('  Exit:',resources.get('exit_classification'),state.get('reason',''))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
 while True:
  if a.watch:print('\033[2J\033[H',end='')
  show()
  if not a.watch:break
  time.sleep(5)
