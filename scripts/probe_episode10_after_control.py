"""Wait for owned GPU01 trial to finish, preserve reference, then read-only probe."""
from pathlib import Path
import json,time,subprocess,os,sys
from run_iteration_search import gpu_idle
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');name='tinyfit_master_prior_tokenscale_headlr0003_r2';py='/usr/local/miniconda3/envs/py312/bin/python'
while True:
 state=json.loads((R/'search_status_01.json').read_text()).get(name,{})
 if state.get('phase')=='failed':raise RuntimeError('control failed; requires inspection')
 if state.get('phase')=='complete' and gpu_idle('0,1'):break
 time.sleep(15)
subprocess.run([py,'scripts/retire_slow_headlr_control.py'],check=True)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
subprocess.run([py,'-u','scripts/probe_initial_head_gradients.py','--config','configs/experiments/iteration_episode10_prior_tokenscale_seed42.yaml','--manifest',str(R/'episode10_manifest.json'),'--output','logs/episode10_initial_head_gradient_probe.json','--limit','10'],env=env,check=True)
