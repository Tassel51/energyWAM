"""Read-only single-episode adaptive LR study progress."""
import json
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/single_episode_lr_20261108')
def read(p):return json.loads(p.read_text())
s=read(R/'controller_status.json') if (R/'controller_status.json').exists() else {}
print('Study:',s.get('phase','not_started'))
if s.get('phase')=='waiting_safe_old_checkpoint':print('Safely paused old groups:',s.get('paused',{}))
for p in sorted(R.glob('episode*/arm_definition.json')):
 d=read(p);folder=p.parent;speed=read(folder/'speed_report.json') if (folder/'speed_report.json').exists() else {};step=speed.get('completed_steps',0);goal=d['job']['steps'];width=min(25,int(25*step/goal));lr=d['hyperparameters']['trainer']
 print(f"scale={d['scale']:g} GPU{d['job']['physical_gpus']} [{'#'*width}{'-'*(25-width)}] {step}/{goal} updates; {step*16/67:.2f}/50.15 epochs")
 print('  LRs:',{k:lr[k] for k in ['learning_rate','task_learning_rate','energy_head_learning_rate','step_size_learning_rate']})
 if (folder/'best_selection.json').exists():print('  auxiliary-VAL best:',read(folder/'best_selection.json')['global_step'])
