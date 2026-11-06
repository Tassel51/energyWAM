"""Plot observed paired endpoints; incomplete evaluations are never joined."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
DATA=json.loads((R/'future_only_and_learned_steps_results.json').read_text())
labels={'ablation_prior1_currentinit1_seed42':'Action + future prior',
 'future_only_currentinit_learned_seed42':'Future prior only',
 'ablation_prior0_currentinit1_seed42':'Learned steps / no prior',
 'no_prior_currentinit_fixed_seed42':'Fixed steps / no prior',
 'no_prior_currentinit_learned_headlr0001_diag_seed42':'Learned / head LR 0.0001',
 'no_prior_currentinit_fixed_headlr0001_diag_seed42':'Fixed / head LR 0.0001'}
fig,axes=plt.subplots(len(DATA),4,figsize=(17,3.5*len(DATA)),constrained_layout=True)
for row,(pair,comparison) in enumerate(DATA.items()):
 for col,key in enumerate(['train_action','train_future','val_action','val_future']):
  ax=axes[row,col]
  for run in comparison['runs']:
   points=sorted((int(s),v[key]) for s,v in run['trajectory'].items() if key in v)
   if points:
    ax.plot(*zip(*points),marker='.',markersize=3,label=labels[run['name']])
  baselines={run['trajectory'].get('0',{}).get(key) for run in comparison['runs']}
  for baseline in baselines-{None}:ax.axhline(baseline,color='gray',linestyle='--',linewidth=.7,alpha=.5)
  ax.set_title(key.replace('_',' ').title());ax.set_xlabel('Optimizer updates');ax.set_ylabel('Standardized MSE')
  if col==0:ax.legend(fontsize=8)
  if col==3:ax.set_yscale('log')
  if pair=='lower_head_lr_bounded_diagnostic' and col in [1,3]:
   ax.set_yscale('linear');ax.set_ylim(0,max(baselines-{None})*1.1)
  ax.grid(alpha=.25)
fig.suptitle('K=2, effective batch=2; 16 train / 16 held-out windows; seed42\nSame coordinates and paired settings; initial prior advantage is not learned improvement',fontsize=12)
out=Path(__file__).resolve().parents[1]/'docs/figures/LTE_WAM_FUTURE_PRIOR_STEP_ABLATION.png'
fig.savefig(out,dpi=170);plt.close(fig);print(out)
