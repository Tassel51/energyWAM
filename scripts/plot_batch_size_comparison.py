"""Observed sample/time quality and scoped resource curves; no winner fabrication."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');data=json.loads((R/'batch_size_comparison_results.json').read_text())
colors={2:'#2878b5',4:'#23995c',8:'#df8524'}
fig,axes=plt.subplots(3,4,figsize=(17,10),constrained_layout=True)
for row in range(3):
 selected=[run for run in data['runs'] if run['comparison_group']==('lower_headlr00003' if row==2 else 'initial_headlr0001')]
 for col,key in enumerate(['train_action','train_future','val_action','val_future']):
  ax=axes[row,col];baselines=[]
  for run in selected:
   points=[]
   for visits,p in run['matched_sample_trajectory'].items():
    if key not in p:continue
    if row==1:
     if 'evaluation_complete_wall_seconds' not in p:continue
     x=p['evaluation_complete_wall_seconds']/60
    else:x=int(visits)
    points.append((x,p[key]))
   points.sort();b=run['effective_batch'];label=f'B{b}'+(' (stopped)' if run['phase']=='failed' else '')
   if points:
    ax.plot(*zip(*points),marker='.',color=colors[b],label=label)
    if run['phase']=='failed':ax.plot(*points[-1],marker='x',color=colors[b],markersize=8)
   initial=run['matched_sample_trajectory'].get('0',{}).get(key)
   if initial is not None:baselines.append(initial)
  for base in set(baselines):ax.axhline(base,color='gray',linestyle='--',linewidth=.7,alpha=.6)
  ax.set_title(key.replace('_',' ').title());ax.set_ylabel('Standardized MSE')
  ax.set_xlabel('Wall minutes until evaluation observed' if row==1 else 'Training sample visits')
  if col==0:ax.legend(fontsize=8)
  if col in [1,3] and row<2:ax.set_yscale('log')
  if row==2 and baselines:
   vals=[p[key] for run in selected for p in run['matched_sample_trajectory'].values() if key in p]
   if vals and max(vals)-min(vals)<.05*max(baselines):ax.set_ylim(0,max(baselines)*1.1)
  ax.grid(alpha=.25)
fig.suptitle('Effective batch2/4/8; K2, native microbatch1; fixed16 train /16 original val windows\nSame initial model / sample order. Rows1-2: head LR0.001. Row3: bounded paired recheck at head LR0.0003. Gray=own initial baseline. '+data['status'],fontsize=12)
out=Path(__file__).resolve().parents[1]/'docs/figures';fig.savefig(out/'LTE_WAM_BATCH_SIZE_QUALITY.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
for run in data['runs']:
 if run['comparison_group']!='initial_headlr0001':continue
 path=R/run['name']/'resource_trace.jsonl'
 if not path.exists():continue
 trace=[]
 for line in path.read_text().splitlines():
  try:trace.append(json.loads(line))
  except ValueError:pass
 b=run['effective_batch'];color=colors[b]
 for ax,key,title in [(axes[0,0],'process_rss_gib','Owned process-tree CPU RSS (GiB)'),(axes[0,1],'process_cpu_percent_one_core','Owned CPU usage (logical cores)')]:
  points=[(r['elapsed_seconds']/60,r[key]/(100 if key.startswith('process_cpu') else 1)) for r in trace if key in r]
  if points:ax.plot(*zip(*points),color=color,label=f'B{b}',linewidth=1)
  ax.set_title(title)
 for offset,ax in enumerate(axes[1]):
  points=[(r['elapsed_seconds']/60,r['gpus'][offset]['memory_used_mib']/1024) for r in trace if len(r.get('gpus',[]))>offset]
  if points:ax.plot(*zip(*points),color=color,label=f'B{b}',linewidth=1)
  ax.axhline(49140/1024,color='black',linestyle='--',linewidth=.8);ax.set_title(('First shard GPU (physical0/2)' if offset==0 else 'Second shard GPU (physical1/3)')+' memory (GiB)')
for ax in axes.flat:ax.set_xlabel('Wall minutes');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('Process CPU memory is not shared-container total/cache. GPU metrics are device-wide.\nExit evidence and cgroup memory-event deltas are in each resource_summary.json',fontsize=11)
fig.savefig(out/'LTE_WAM_BATCH_SIZE_RESOURCES.png',dpi=160);plt.close(fig)
print('Updated batch quality and resource figures; status:',data['status'])
