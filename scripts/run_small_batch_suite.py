import subprocess,os,json,time,signal
from pathlib import Path
r=Path('/root/zzc/energyWAM');out=r/'logs/batch_comparison_20261101';py='/usr/local/miniconda3/envs/py312/bin/python';results=[]
base=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(r),TOKENIZERS_PARALLELISM='false',ENERGYWAM_ALLOW_GPU_TRAINING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
while not (out/'capacity_done').exists():time.sleep(5)
for batch in [1,2,4]:
 jobs=[]
 for gpu,direction in [(2,'action'),(3,'future')]:
  tag=f'small_b{batch}_{direction}';f=(out/(tag+'.log')).open('w');args=[py,'-u','scripts/benchmark_batch_capacity.py','--batch',str(batch),'--steps','2','--single-device','--direction',direction,'--checkpoint','--offload','--gpu-budget','3','--pin-budget','64','--async-offload','--preallocate-optimizer','--longest-instruction','--tag',tag]
  p=subprocess.Popen(args,cwd=r,env=dict(base,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((p,f,tag,time.monotonic()))
 for p,f,tag,started in jobs:
  try:code=p.wait(timeout=max(1,240-(time.monotonic()-started)))
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=20);code=124
  f.close();text=(out/(tag+'.log')).read_text(errors='replace');reason='completed' if code==0 else ('GPU OOM' if 'OutOfMemoryError' in text or 'CUDA out of memory' in text else ('host RAM safety limit' if 'RAM limit reached' in text else ('time limit' if code==124 else 'error')))
  results.append(dict(tag=tag,exit=code,result=reason));(out/'small_summary.json').write_text(json.dumps(results,indent=2))
(out/'small_done').write_text('done')

print(subprocess.run(['supervisorctl','start','energywam-gpu23'],capture_output=True,text=True).stdout,flush=True)
(out/'training_resumed').write_text('resume requested')
