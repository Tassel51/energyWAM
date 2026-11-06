"""Paired episode-level comparison of the frozen64-window validation set."""
import json,statistics
from pathlib import Path
import numpy as np
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def main():
 models={b:R/'batch_expanded_validation'/f'batch{b}_tiny16_samples1600_seed42' for b in [2,4]}
 summaries={b:json.loads((p/'summary.json').read_text()) for b,p in models.items()}
 result=dict(scope='Paired differences on64 fixed windows /16 independent validation episodes /2 tasks; stratified episode bootstrap describes episode uncertainty, not training-seed uncertainty or unseen-task generalization. Lower difference favors batch4.',seed=20261105,bootstrap_replicates=10000,metrics={})
 for direction in ['action','future']:
  rows={}
  for b,p in models.items():
   report=json.loads((p/f"eval_{summaries[b]['checkpoint_step']:04d}_{direction}.json").read_text());assert report['weight_updates_during_eval']==0
   rows[b]={x['window_id']:x for x in report['rows'] if x['k']==2};assert len(rows[b])==64
  assert rows[2].keys()==rows[4].keys()
  episodes={}
  for wid,left in rows[2].items():
   right=rows[4][wid];assert (left['episode_index'],left['frame_index'],left['task'])==(right['episode_index'],right['frame_index'],right['task'])
   episodes.setdefault((left['task'],left['episode_index']),[]).append(right['prediction_mse']-left['prediction_mse'])
  tasks=sorted({k[0] for k in episodes});groups=[np.array([statistics.mean(v) for (task,ep),v in episodes.items() if task==t]) for t in tasks]
  assert len(episodes)==16 and all(len(v)==4 for v in episodes.values()) and all(len(g)==8 for g in groups)
  rng=np.random.default_rng(result['seed']);means=sum(g[rng.integers(0,len(g),size=(10000,len(g)))].mean(axis=1) for g in groups)/len(groups)
  baseline_key='zero_mse' if direction=='action' else 'current_visual_persistence_mse'
  baseline=summaries[2]['baselines']['val'][direction][baseline_key]
  assert abs(baseline-summaries[4]['baselines']['val'][direction][baseline_key])<1e-12
  result['metrics'][direction]=dict(batch2=summaries[2]['metrics'][direction],batch4=summaries[4]['metrics'][direction],baseline=baseline,baseline_name=baseline_key,difference_batch4_minus_batch2=float(np.mean([statistics.mean(v) for v in episodes.values()])),episode_bootstrap_95_percentile_interval=list(map(float,np.quantile(means,[.025,.975]))),improvement_over_baseline={str(b):1-summaries[b]['metrics'][direction]/baseline for b in [2,4]})
 (R/'batch_confirmation_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
