#!/usr/bin/env python
"""Summarize authoritative fixed-candidate reports, not moving-batch losses."""
import json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')

def summarize(folder):
    grouped={};per_task={}
    for file in sorted(folder.glob('eval_*.json')):
        data=json.loads(file.read_text())
        if data.get('weight_updates_during_eval')!=0:raise ValueError('probe changed weights')
        for row in data['rows']:
            if row['k']!=2:continue
            key=(row['update'],row['split'],row['direction'])
            grouped.setdefault(key,[]).append(row['prediction_mse'])
            per_task.setdefault((*key,row['task']),[]).append(row['prediction_mse'])
    if not grouped:return None
    updates=sorted({key[0] for key in grouped})
    points=[]
    for update in updates:
        metrics={f'{split}/{direction}':statistics.mean(values) for (u,split,direction),values in grouped.items() if u==update}
        points.append(dict(update=update,metrics=metrics))
    first=points[0]['metrics'];last=points[-1]['metrics']
    gains={key:1-last[key]/first[key] for key in first if key in last}
    records={}
    f=folder/'timing_history.jsonl'
    if f.exists():
        for line in f.read_text().splitlines():
            row=json.loads(line);records[int(row['global_step_completed'])]=row
    baselines=json.loads((folder/'dataset_baselines.json').read_text()) if (folder/'dataset_baselines.json').exists() else {}
    below_mean={f'{split}/{direction}':last.get(f'{split}/{direction}',float('inf'))<value['training_mean_mse']
                for split,dirs in baselines.items() for direction,value in dirs.items()}
    return dict(folder=str(folder),latest_eval_update=updates[-1],latest_training_update=max(records,default=0),
        points=points,relative_improvement=gains,below_training_mean_predictor=below_mean,
        training_seconds=sum(r['end_to_end_seconds'] for r in records.values()),
        per_task=[dict(update=u,split=s,direction=d,task=t,mse=statistics.mean(v)) for (u,s,d,t),v in per_task.items()],
        baselines=baselines)

def main():
    groups={}
    for folder in sorted(OUT.iterdir()):
        if not folder.is_dir():continue
        result=summarize(folder)
        if result:groups[folder.name]=result
    (OUT/'results_summary.json').write_text(json.dumps(groups,indent=2))
    lines=['|实验|最近检查更新|训练 action|训练 future|验证 action|验证 future|','|---|---:|---:|---:|---:|---:|']
    for name,result in groups.items():
        if not (name.endswith('_r2') or name.startswith('episode10_')):continue
        m=result['points'][-1]['metrics']
        values=[f'{m[k]:.6f}' if k in m else '—' for k in ['train/action','train/future','val/action','val/future']]
        lines.append('|'+ '|'.join([name,str(result['latest_eval_update'])]+values)+'|')
    table='\n'.join(lines)
    p=ROOT/'docs/LTE_WAM_MODEL_AND_EXPERIMENT_SUMMARY_CN.md';doc=p.read_text()
    start='<!-- iteration-results:start -->';end='<!-- iteration-results:end -->'
    if start in doc:
        a=doc.index(start);b=doc.index(end,a)+len(end);doc=doc[:a]+start+'\n'+table+'\n'+end+doc[b:]
    else:
        pos=doc.index('\n\n## 1.');doc=doc[:pos]+'\n\n'+start+'\n'+table+'\n'+end+doc[pos:]
    p.write_text(doc)
    print(table)
if __name__=='__main__':main()
