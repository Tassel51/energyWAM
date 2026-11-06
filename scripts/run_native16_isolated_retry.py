"""Wait for actual native8 termination, then bounded native16 CPU-capacity retry."""
import sys,json,time,subprocess,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_iteration_search import gpu_idle
R=Path('/cloud/zzc/energyWAM/experiments/native_capacity_20261105')
for _ in range(160):
 state=json.loads((R/'status_01.json').read_text()).get('native_capacity_micro8_effective16_seed42',{})
 alive=True
 try:os.kill(state.get('pid',-1),0)
 except ProcessLookupError:alive=False
 if state.get('phase') in ['complete','failed'] and not alive and gpu_idle('0,1') and gpu_idle('2,3'):break
 time.sleep(15)
else:raise RuntimeError('Owned native8 job or GPU pair remained active; no job interrupted')
subprocess.run([sys.executable,'-u','scripts/run_iteration_search.py','--jobs',str(R/'jobs_16_isolated.json'),'--status',str(R/'status_16_isolated.json')],cwd=ROOT,check=True)
