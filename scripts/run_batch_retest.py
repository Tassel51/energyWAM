from pathlib import Path
import subprocess,os,time,json,signal
r=Path('/root/zzc/energyWAM');out=r/'logs/batch_retest_20261101';py='/usr/local/miniconda3/envs/py312/bin/python';rows=[]
env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',ENERGYWAM_ALLOW_GPU_TRAINING='1',PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')
try:
 for batch in [1,2,4]:
  jobs=[];start=time.monotonic();peak=0;reason=None
  for gpu,d in [(2,'action'),(3,'future')]:
   tag=f'retest_b{batch}_{d}';f=(out/(tag+'.log')).open('w');a=[py,'-u','scripts/benchmark_batch_retest.py','--batch',str(batch),'--steps','2','--single-device','--direction',d,'--checkpoint','--offload','--gpu-budget','3','--pin-budget','128','--async-offload','--preallocate-optimizer','--longest-instruction','--tag',tag];p=subprocess.Popen(a,cwd=r,env=dict(env,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=f,stderr=subprocess.STDOUT,start_new_session=True);jobs.append((p,f,tag))
  while any(p.poll() is None for p,f,t in jobs):
   rss=0
   for p,f,t in jobs:
    q=Path(f'/proc/{p.pid}/status')
    if q.exists():
     for line in q.read_text().splitlines():
      if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
   peak=max(peak,rss)
   if rss>260*2**30:reason='aggregate worker RSS exceeded 260 GiB'
   if time.monotonic()-start>360:reason='360-second bounded test timeout'
   if reason:
    for p,f,t in jobs:
     if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
    break
   time.sleep(.5)
  for p,f,tag in jobs:
   try:code=p.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);code=p.wait()
   f.close();row=dict(microbatch=batch,direction=tag.rsplit('_',1)[1],exit=code,aggregate_peak_rss_gib=peak/2**30,stop_reason=reason)
   report=r/'logs/tuning_gpu23'/tag/'benchmark.json'
   if report.exists():
    d=json.loads(report.read_text());row['steps']=[{k:x.get(k) for k in ['batch_size','wall_seconds','data_seconds','end_to_end_samples_per_second','gpu_peak_allocated_gib','param_grad_nonfinite','max_rss_mb']} for x in d['steps']]
   if code:row['error_tail']=(out/(tag+'.log')).read_text(errors='replace')[-1400:]
   rows.append(row)
  (out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows[-2:]),flush=True)
finally:
 print(subprocess.run(['supervisorctl','start','energywam-gpu23'],capture_output=True,text=True).stdout,flush=True)
 (out/'done').write_text('done')
