"""Paired episode-level screening of residual-energy trials, no test selection."""
import json,statistics,argparse
from pathlib import Path
import numpy as np

def assess(folder):
    points={};groups={}
    for direction in ['action','future']:
        initial=folder/f'eval_0000_{direction}.json'
        if not initial.exists():continue
        baseline={r['window_id']:r for r in json.loads(initial.read_text())['rows'] if r['k']==2}
        for file in sorted(folder.glob(f'eval_*_{direction}.json')):
            report=json.loads(file.read_text());assert report['weight_updates_during_eval']==0
            rows=[r for r in report['rows'] if r['k']==2]
            if not rows:continue
            update=rows[0]['update']
            for split in ['train','val']:
                selected=[r for r in rows if r['split']==split]
                if not selected:continue
                episode={}
                for row in selected:
                    b=baseline[row['window_id']];assert b['episode_index']==row['episode_index'] and b['split']==split
                    episode.setdefault((row['source_dataset'],row['episode_index']),[]).append(row['prediction_mse']-b['prediction_mse'])
                delta=np.array([statistics.mean(v) for v in episode.values()]);rng=np.random.default_rng(42)
                sampled=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
                ref=statistics.mean(baseline[r['window_id']]['prediction_mse'] for r in selected);mse=statistics.mean(r['prediction_mse'] for r in selected)
                row=dict(update=update,split=split,direction=direction,mse=mse,initial_prediction_mse=ref,relative_improvement=1-mse/ref,
                    episodes=len(episode),windows=len(selected),episode_mean_paired_delta=float(delta.mean()),
                    descriptive_episode_bootstrap_95=list(map(float,np.quantile(sampled,[.025,.975]))))
                points[update,split,direction]=row
    updates=sorted({k[0] for k in points})
    sustained=[u for u in updates if all(points.get((u,'train',d),{}).get('relative_improvement',-1)>=.1 for d in ['action','future'])]
    last3=updates[-3:]
    gate=len(last3)==3 and all(u in sustained for u in last3)
    return dict(folder=str(folder),screening_pass=gate,note='10% reduction in BOTH train directions at three latest checks is a screening gate only; episode-level bootstrap is descriptive and does not establish broad generalization',points=list(points.values()))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104');a=p.parse_args();root=Path(a.root)
    folders=list(root.glob('tinyfit_master_prior*'))+list(root.glob('episode10_*'))
    reports=[assess(f) for f in sorted(folders) if f.is_dir() and (f/'eval_0000_action.json').exists()]
    (root/'residual_fit_assessment.json').write_text(json.dumps(reports,indent=2))
    for report in reports:
        last=max((p['update'] for p in report['points']),default=0)
        print(Path(report['folder']).name, 'gate=',report['screening_pass'],[p for p in report['points'] if p['update']==last])
if __name__=='__main__':main()
