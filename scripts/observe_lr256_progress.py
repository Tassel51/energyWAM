"""Read-only progress for the paired256 resume LR experiment."""
import json
from pathlib import Path
ROOT=Path('/cloud/zzc/energyWAM/experiments/lr256_comparison_20261107')
target=json.loads((ROOT/'plan.json').read_text()).get('target_step',512)
for label,pair in [('03','0,1'),('01','2,3')]:
 name='resume256_lr'+label;folder=ROOT/name
 states=json.loads((ROOT/f'status_{label}.json').read_text()) if (ROOT/f'status_{label}.json').exists() else {}
 state=states.get(name+'_to2048',states.get(name,{}))
 speed=json.loads((folder/'speed_report.json').read_text()) if (folder/'speed_report.json').exists() else {}
 step=speed.get('completed_steps',256);n=max(0,step-256);budget=target-256;width=min(30,int(30*n/budget))
 print(f"GPU{pair} LR x0.{label[-1] if label=='03' else '1'} [{'#'*width}{'-'*(30-width)}] {n}/{budget} new updates; total={step}/{target}; {state.get('phase','pending')}")
 audit=folder/'resume_lr_override_step256.json'
 if audit.exists():print('  verified LRs:',json.loads(audit.read_text())['effective_lrs'])
 if state.get('reason'):print('  reason:',state['reason'])
 if speed.get('steps'):
  rows=speed['steps'][-20:];secs=sum(r['end_to_end_seconds'] for r in rows)/len(rows)
  print(f'  recent {secs:.1f}s/update; training-only remaining {max(0,target-step)*secs/3600:.2f}h (evaluation/save extra)')
