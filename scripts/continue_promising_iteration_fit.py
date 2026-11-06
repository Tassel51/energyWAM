"""One bounded, conditional continuation of the active residual-energy fit."""
import json,time,subprocess,statistics
from pathlib import Path
ROOT=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
NAME='tinyfit_master_prior_tokenscale_r2';FOLDER=ROOT/NAME

def mse(update,direction,split):
    report=json.loads((FOLDER/f'eval_{update:04d}_{direction}.json').read_text())
    assert report['weight_updates_during_eval']==0
    rows=[r for r in report['rows'] if r['k']==2 and r['split']==split]
    assert len(rows)==16
    return statistics.mean(r['prediction_mse'] for r in rows)

def main():
    while True:
        p=FOLDER/'final_report.json'
        if p.exists() and json.loads(p.read_text())['global_step']>=160:break
        status=json.loads((ROOT/'search_status_23.json').read_text()).get(NAME,{})
        if status.get('phase')=='failed':raise RuntimeError('parent trial failed; refusing continuation')
        time.sleep(15)
    if json.loads(p.read_text())['global_step']>160:return
    values={};eligible=True
    for d in ['action','future']:
        initial=mse(0,d,'train');train=mse(160,d,'train')
        values[d]=dict(train160=train,initial_train=initial)
        eligible &= train<.95*initial
        for update in [140,160]:
            val=mse(update,d,'val');values[d][f'val{update}']=val
            eligible &= val<mse(0,d,'val')
    decision=dict(extend_to240=bool(eligible),metrics=values,
        note='5% train improvement plus both held-out directions below initial at checks140/160 permits a bounded continuation only; NOT the final success criterion')
    (FOLDER/'continuation_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision),flush=True)
    if not eligible:return
    meta=json.loads((FOLDER/'ckpt_latest.pt.meta.json').read_text());assert meta['meta']['global_step']==160
    jobs_path=ROOT/'search_jobs_23.json';jobs=json.loads(jobs_path.read_text());jobname=NAME+'_continue240'
    if not any(j['name']==jobname for j in jobs):
        jobs.append(dict(name=jobname,config=f'configs/experiments/iteration_{NAME}.yaml',manifest=str(ROOT/'tinyfit_manifest.json'),steps=240,eval_interval=20,physical_gpus='2,3'))
        tmp=jobs_path.with_suffix('.continue.tmp');tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(jobs_path)
    # Existing controller reloads its queue between jobs. If it already ended,
    # resume through the same managed service instead of spawning a duplicate.
    time.sleep(5)
    status=subprocess.run(['supervisorctl','status','energywam-iteration-search23'],capture_output=True,text=True).stdout
    if 'EXITED' in status:subprocess.run(['supervisorctl','start','energywam-iteration-search23'],check=True)
if __name__=='__main__':main()
