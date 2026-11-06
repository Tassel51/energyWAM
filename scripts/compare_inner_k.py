#!/usr/bin/env python
"""Paired comparisons for a completed K sweep; no model or GPU operations."""
import argparse
import json
from pathlib import Path
import statistics


def load_traces(folder):
    rows={}
    for direction in ['action','future']:
        for line in (folder/f'trace_{direction}.jsonl').read_text().splitlines():
            r=json.loads(line);rows[(r['direction'],r['window_id'],r['seed'],r['k'])]=r
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);config=json.load(open(out/'experiment_config.json'));report=json.load(open(out/'report.json'))
    rows=load_traces(out)
    baseline=load_traces(Path('/cloud/zzc/energyWAM/experiments/inner_diag_20261102'))
    prefix=max(abs(rows[key]['prediction_mse']-r['prediction_mse']) for key,r in baseline.items())
    comparisons=[]
    ks=config['record_k']
    for split in ['train','val']:
        for direction in ['action','future']:
            for ka,kb in zip(ks,ks[1:]):
                b=[r for r in rows.values() if r['split']==split and r['direction']==direction and r['k']==kb]
                paired=[rows[(direction,r['window_id'],r['seed'],ka)] for r in b]
                changes=[rb['prediction_mse']-ra['prediction_mse'] for ra,rb in zip(paired,b)]
                mean_before=statistics.mean(r['prediction_mse'] for r in paired)
                comparisons.append({'split':split,'direction':direction,'from_k':ka,'to_k':kb,
                    'mean_paired_mse_change':statistics.mean(changes),
                    'relative_improvement_percent':-100*statistics.mean(changes)/mean_before,
                    'improved_pair_fraction':sum(d<0 for d in changes)/len(changes)})
    timing=[]
    for k in ks:
        values=[r['refinement_seconds'] for r in rows.values() if r['k']==k]
        timing.append({'k':k,'median_diagnostic_refinement_seconds':statistics.median(values),
                       'mean_diagnostic_refinement_seconds':statistics.mean(values)})
    best=min((r for r in report['aggregates'] if r['split']=='val' and r['direction']=='action'),key=lambda r:r['mean_mse'])
    result={'prefix_vs_previous_max_abs_mse_diff':prefix,'paired_comparisons':comparisons,'timing':timing,
            'best_observed_val_action_k':best['k'],
            'timing_scope':'Frozen model candidate refinement plus diagnostic statistics; excludes condition encoding and model loading. Not training or deployment latency.',
            'scope':'Fixed 20 train + 20 val windows, two seeds, one K=2-trained checkpoint; no training updates or success-rate measurement.'}
    (out/'comparison.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
