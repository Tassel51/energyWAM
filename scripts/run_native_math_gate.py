"""Finite supervised real4B native4 precision gate; no optimizer updates."""
import argparse,sys,os,json,subprocess,time,signal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_iteration_search import gpu_idle
from energywam.training.resource_monitor import ResourceMonitor
p=argparse.ArgumentParser();p.add_argument('--pair',required=True);p.add_argument('--mode',choices=['default','strict','tf32'],required=True);p.add_argument('--precision',choices=['configured','float32','float16'],default='configured');p.add_argument('--saved-gpu-budget',type=float);p.add_argument('--no-async-offload',action='store_true');p.add_argument('--dedup-cpu',action='store_true');p.add_argument('--tag');p.add_argument('--wait-idle',action='store_true');p.add_argument('--max-uss',type=float,default=100);a=p.parse_args()
R=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105');folder=R/'gradient_gates'/(a.tag or a.mode);folder.mkdir(parents=True,exist_ok=True)
if a.wait_idle:
 for _ in range(160):
  if gpu_idle(a.pair):break
  time.sleep(15)
 else:raise RuntimeError('Assigned GPU pair remained occupied; no process interrupted')
assert gpu_idle(a.pair),'assigned pair occupied; do not interrupt it'
cmd=[sys.executable,'-u','scripts/check_native_microbatch_gradients.py','--config','configs/experiments/native_capacity_micro4_effective16_seed42.yaml','--manifest',str(R/'train_quick_val_manifest.json'),'--output',str(folder/'gradient_check.json'),'--physical-gpus',a.pair,'--native-microbatch','4','--sample-indices','0,80,160,240','--math-mode',a.mode,'--repeat-reference','--record-all']
cmd+=['--precision',a.precision]
if a.saved_gpu_budget is not None:cmd+=['--saved-gpu-budget',str(a.saved_gpu_budget)]
if a.no_async_offload:cmd+=['--no-async-offload']
if a.dedup_cpu:cmd+=['--dedup-cpu']
env=dict(os.environ,CUDA_VISIBLE_DEVICES=a.pair,ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
with (folder/'console.log').open('a') as log:
 child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 monitor=ResourceMonitor(child.pid,a.pair.split(','),folder);monitor.start();guard=None
 while child.poll() is None:
  if time.time()-monitor.started>900:guard='15minute correctness diagnostic budget'
  if monitor.summary.get('last_sample',{}).get('process_uss_gib',0)>a.max_uss:guard=f'{a.max_uss}GiB owned USS diagnostic budget'
  if guard:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55);break
  time.sleep(2)
 monitor.finish(child.returncode,guard)
report=json.loads((folder/'gradient_check.json').read_text()) if (folder/'gradient_check.json').exists() else {}
(folder/'status.json').write_text(json.dumps(dict(phase='complete',child_returncode=child.returncode,guard=guard,gate_passed=bool(report.get('passed')),scope='No weight updates or formal training; inspect diagnostics before selecting a math mode.'),indent=2))
print(a.mode,'gate_passed',report.get('passed'),'child_returncode',child.returncode,flush=True)
