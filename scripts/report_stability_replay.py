"""Summarize actual bounded replay logs; no inference from unfinished arms."""
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path('/cloud/zzc/energyWAM/experiments/stability_diagnosis_20261107')
SOURCE=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106/three_task_b16_micro4_seed42')


def read_rows(path):
    out={}
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                row=json.loads(line)
                if 'global_step_completed' in row:out[int(row['global_step_completed'])]=row
            except ValueError:pass
    return out


def summarize():
    original=read_rows(SOURCE/'timing_history.jsonl')
    result=dict(updated_unix=time.time(),scope='12-update replay from680; not a new generalization study',arms={})
    for arm in ['baseline','candidate_clip']:
        name='replay680_'+arm;folder=ROOT/name;status=ROOT/f'status_{arm}.json'
        state=json.loads(status.read_text()).get(name,{}) if status.exists() else {}
        rows=read_rows(folder/'timing_history.jsonl')
        summary=dict(phase=state.get('phase','not_started'),completed_new_updates=len(rows),
                     stop_reason=state.get('reason'),assessments={})
        for split,target in [('train',folder/'train_monitor'),('val',folder)]:
            for path in sorted(target.glob('eval_*.json')):
                data=json.loads(path.read_text())
                values=[r for r in data['rows'] if r['split']==split and r['k']==2]
                if not values:continue
                key=f"{values[0]['update']}/{split}/{values[0]['direction']}"
                traces=[step for row in values for step in row.get('inner_steps',[])]
                summary['assessments'][key]=dict(samples=len(values),mse=statistics.mean(r['prediction_mse'] for r in values),
                    maximum_window_mse=max(r['prediction_mse'] for r in values),
                    mean_clipped_fraction=statistics.mean(t['clipped_fraction'] for t in traces) if traces else None,
                    max_proposed_update_rms=max(t['proposed_update_rms'] for t in traces) if traces else None,
                    by_task={task:statistics.mean(r['prediction_mse'] for r in values if r['task']==task) for task in sorted({r['task'] for r in values})})
        summary['optimizer_updates']=[dict(update=u,action=r['bi/action_loss'],future=r['bi/future_loss'],
             outer_gradient_norm=r['param_grad_norm'],alpha_action=r['diag/alpha_action'],alpha_future=r['diag/alpha_future']) for u,r in rows.items()]
        if arm=='baseline' and rows:
            keys=['bi/action_loss','bi/future_loss','diag/alpha_action','diag/alpha_future']
            comparisons=[dict(update=u,key=k,original=original[u][k],replay=r[k],
                              passed=math.isclose(original[u][k],r[k],rel_tol=1e-5,abs_tol=1e-6)) for u,r in rows.items() for k in keys]
            summary['original_replay_check']=dict(passed=all(c['passed'] for c in comparisons),comparisons=comparisons)
        resource=folder/'resource_summary.json'
        if resource.exists():
            rs=json.loads(resource.read_text())
            summary['resource']={k:rs.get(k) for k in ['exit_classification','guard_reason','returncode','peaks','memory_events_delta']}
        result['arms'][arm]=summary
    terminal=all(a['phase'] in ['complete','failed'] for a in result['arms'].values())
    result['status']='terminal_review' if terminal else 'running'
    result['interpretation']='Compare both the common source and each intervention own680 baseline; any initial benefit from clipping is not learned improvement. Stabilization over12 updates is not evidence of long-run convergence.'
    path=ROOT/'replay_comparison.json';temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
    print(result['status'])
    for name,arm in result['arms'].items():
        n=arm['completed_new_updates'];print(f'{name:16} [{"#"*n:<12}] {n}/12 {arm["phase"]}')
        for key,metric in arm['assessments'].items():
            print(' ',key,round(metric['mse'],6))
    return result


if __name__=='__main__':summarize()
