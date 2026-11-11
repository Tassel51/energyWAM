"""Summarize diagnostic measurements; reconstruct window IDs from logged RNG positions."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import torch


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def summarize(output):
    d=output/'diagnostics';manifest=json.loads((output/'sample_manifest.json').read_text())
    windows=[x for x in manifest['windows'] if x['split']=='train'];orders={}
    traces=rows(d/'window_inner_trace.jsonl');identity={}
    for row in traces:
        role=0 if row['direction']=='action' else 1
        position,remainder=divmod(row['candidate_seed']-43-role,100003)
        assert remainder==0 and (row['update']-1)*16<=position<row['update']*16
        epoch,offset=divmod(position,len(windows))
        if epoch not in orders:orders[epoch]=torch.randperm(len(windows),generator=torch.Generator().manual_seed(43+epoch)).tolist()
        index=orders[epoch][offset];w=windows[index]
        row.update(episode=w['episode_index'],frame=w['frame_index'],train_window_index=index,sample_visit_position=position,
                   identity_source='Exact reconstruction from logged candidate seed, frozen sampler algorithm and fixed manifest; IDs were stripped from prepared batch')
        identity[(row['micro'],row['sample_in_micro'])]=(w['episode_index'],w['frame_index'],index)
    (d/'window_inner_trace_identified.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in traces))
    results={}
    for update in sorted({x['update'] for x in traces}):
        r={'update':update,'directions':{}}
        for direction in ['action','future']:
            data=[x for x in traces if x['update']==update and x['direction']==direction]
            if not data:continue
            r['directions'][direction]={}
            for iteration in [0,1]:
                vs=[x for x in data if x['iteration']==iteration]
                if not vs:continue
                r['directions'][direction][str(iteration)]=dict(samples=len(vs),mean_before=statistics.mean(x['mse_before'] for x in vs),
                    mean_after=statistics.mean(x['mse_after'] for x in vs),max_after=max(x['mse_after'] for x in vs),
                    max_candidate_update_rms=max(x['candidate_update_rms'] for x in vs),
                    top_windows=sorted(vs,key=lambda x:x['mse_after'],reverse=True)[:3])
        results[update]=r
    for row in rows(d/'gradient_clipping.jsonl'):
        r=results[row['update']];r['clipping']={k:v for k,v in row.items() if k not in ['before_parameters','after_parameters']}
        r['top_gradient_parameters']=sorted(row['before_parameters'],key=lambda x:x['norm'],reverse=True)[:12]
    for row in rows(d/'parameter_updates.jsonl'):
        r=results[row['update']];r['optimizer_group_updates']={g:dict(parameter_norm=row['parameter_group_norms'][g],update_norm=v,
               relative_update=v/max(row['parameter_group_norms'][g],1e-30)) for g,v in row['update_group_norms'].items()}
        r['top_relative_parameter_updates']=sorted(row['parameters'],key=lambda x:x['relative_update'],reverse=True)[:10]
    adjoints=rows(d/'candidate_adjoint.jsonl');adj_map={}
    for row in adjoints:
        for i,value in enumerate(row['gradient_rms']):
            ep,fr,idx=identity[(row['micro'],i)]
            adj_map[(row['update'],row['direction'],row['micro'],i,row['iteration'])]=dict(episode=ep,frame=fr,window=idx,rms=value)
    for update,r in results.items():
        amps=[]
        for key,v in adj_map.items():
            u,direction,micro,i,it=key
            if u!=update or it!=0:continue
            last=adj_map.get((u,direction,micro,i,1))
            if last:amps.append(dict(direction=direction,micro=micro,sample_in_micro=i,episode=v['episode'],frame=v['frame'],window=v['window'],
                first_candidate_adjoint_rms=v['rms'],final_candidate_adjoint_rms=last['rms'],amplification=v['rms']/max(last['rms'],1e-30)))
        r['top_inner_backward_amplification']=sorted(amps,key=lambda x:x['amplification'],reverse=True)[:8]
        micros=[]
        for row in rows(d/'micro_parameter_gradients.jsonl'):
            if row['update']!=update:continue
            calls=row['incoming_gradient_calls'];groups=defaultdict(float)
            for x in calls:groups[x['group']]+=x['norm']**2
            micros.append(dict(micro=row['micro'],direction=row['direction'],windows=[identity[(row['micro'],i)] for i in range(4)],
                group_incoming_norms={k:math.sqrt(v) for k,v in groups.items()},
                top_parameters=sorted(calls,key=lambda x:x['norm'],reverse=True)[:5]))
        r['microbatch_gradients']=micros
        endpoint_checks=[]
        for row in traces:
            if row['update']!=update or row['iteration']!=1:continue
            key=(update,row['direction'],row['micro'],row['sample_in_micro'],1)
            actual=adj_map.get(key)
            if actual is None:continue
            count=16*12 if row['direction']=='action' else 3*1*196*384
            weight=1. if row['direction']=='action' else 9.
            expected=2*weight*math.sqrt(row['mse_after'])/(16*count)
            endpoint_checks.append(dict(direction=row['direction'],frame=row['frame'],expected=expected,actual=actual['rms'],
                relative_error=abs(expected-actual['rms'])/max(expected,1e-30)))
        r['endpoint_loss_gradient_scaling']=dict(samples_checked=len(endpoint_checks),
            max_relative_error=max((x['relative_error'] for x in endpoint_checks),default=None),
            formula='2*direction_weight*RMSE/(effective_batch16*valid_elements_per_window)',checks=endpoint_checks)
    inputs=rows(d/'input_statistics.jsonl');stats={}
    for row in inputs:
        for field,v in row['fields'].items():
            s=stats.setdefault(field,dict(max_abs=0.,max_sample_rms=0.,nonfinite_count=0,shapes=[]))
            s['max_abs']=max(s['max_abs'],v['absmax']);s['max_sample_rms']=max(s['max_sample_rms'],max(v['rms']));s['nonfinite_count']+=v['nonfinite']
            if v['shape'] not in s['shapes']:s['shapes'].append(v['shape'])
    summary=dict(scope='Actual training batches of four replayed updates; not full67-window fixed assessment',updates=list(results.values()),input_statistics=stats,
       window_identity_note='Derived exactly from saved candidate seed and deterministic sampler; not directly present in stage-prepared metadata',
       limitations=['Initial step61 is already unstable; cannot locate earliest onset from this replay alone',
                    'Module gradient norms identify where large derivatives occur, not which module uniquely caused them',
                    'No per-window 4B parameter gradients: per-window candidate adjoints and native4 microbatch parameter gradients are recorded'])
    (output/'diagnostic_summary.json').write_text(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    summary=summarize(args.output)
    print(json.dumps(dict(updates=[x['update'] for x in summary['updates']],input_statistics=summary['input_statistics']),indent=2))
