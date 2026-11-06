"""Release completed bounded diagnostic optimizer before main final checkpoint."""
import json,pathlib,time,subprocess,os
R=pathlib.Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
name='no_prior_currentinit_learned_headlr0001_diag_seed42'
for _ in range(120):
 state=json.loads((R/'search_status_23.json').read_text()).get(name,{})
 if state.get('phase')=='complete':
  try:os.kill(state['pid'],0)
  except ProcessLookupError:break
 if state.get('phase')=='failed':raise RuntimeError('learned diagnostic failed; retain logs and inspect')
 print(json.dumps({'waiting_for':name,'phase':state.get('phase'),'completed_updates':state.get('completed_updates')}),flush=True)
 time.sleep(30)
else:raise RuntimeError('observation deadline, no restart attempted')
subprocess.run(['/usr/local/miniconda3/envs/py312/bin/python','scripts/preserve_finished_step_diagnostic.py','--name',name],check=True)
subprocess.run(['/usr/local/miniconda3/envs/py312/bin/python','scripts/report_future_prior_step_ablation.py'],check=True)
print('Both bounded diagnostics complete; model references and all logs retained.',flush=True)
