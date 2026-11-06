from pathlib import Path
import subprocess,os,time,json,signal
r=Path('/root/zzc/energyWAM');o=r/'logs/batch_comparison_20261101';py='/usr/local/miniconda3/envs/py312/bin/python'
while not (o/'training_resumed').exists():time.sleep(5)
print(subprocess.run(['supervisorctl','stop','energywam-gpu23'],capture_output=True,text=True).stdout,flush=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(r),TOKENIZERS_PARALLELISM='false',ENERGYWAM_ALLOW_GPU_TRAINING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
rows=[]
try:
 for per_card in [8,16,32]:
  tag=f'accum_joint_percard{per_card}';folder=o/tag;log=o/(tag+'.log');args=[py,'-u','scripts/benchmark_accumulated_batches.py','--config','configs/gpu23_joint_20261101.yaml','--benchmark-steps','2','--output-dir',str(folder),'--set',f'trainer.per_device_batch_size={per_card*2}','trainer.save_interval=0']
  with log.open('w') as f:
   p=subprocess.Popen(args,cwd=r,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=p.wait(timeout=1000)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=30);code=124
  row=dict(per_card_accumulated_samples=per_card,global_effective_batch=per_card*2,microbatch=1,exit=code)
  report=folder/'speed_report.json'
  if report.exists():
   d=json.loads(report.read_text());row['steps']=[{k:x.get(k) for k in ['end_to_end_seconds','end_to_end_samples_per_second','gradient_sync_seconds','param_grad_nonfinite','cuda_max_alloc_mb','max_rss_mb']} for x in d.get('steps',[])]
  if code:row['error_tail']=log.read_text(errors='replace')[-1600:]
  rows.append(row);(o/'accumulation_summary.json').write_text(json.dumps(rows,indent=2))
finally:
 print(subprocess.run(['supervisorctl','start','energywam-gpu23'],capture_output=True,text=True).stdout,flush=True)
 (o/'all_tests_done').write_text('done')
