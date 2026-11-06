#!/usr/bin/env python
"""Bounded, staged single-model experiments; leaves formal BI/Stage2 stopped."""
import json,os,sys,subprocess,time,statistics,shutil
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_k_training_comparison import terminate_session,wait_for_free_gpus
PYTHON='/usr/local/miniconda3/envs/py312/bin/python'
OUT=Path('/cloud/zzc/energyWAM/experiments/adaptive_shared_20261103')
PILOT=Path('/cloud/zzc/energyWAM/experiments/train_k248_sharded_20261103')
MANIFEST=PILOT/'sample_manifest.json'
STATE={};START=time.monotonic()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def phase(name,**fields):
    STATE.update(phase=name,**fields);write(OUT/'status.json',STATE);print(json.dumps(STATE),flush=True)

def run(command,folder,env=None,seconds=3600):
    if time.monotonic()-START>20*3600:raise TimeoutError('total experiment budget 20h reached')
    wait_for_free_gpus();folder.mkdir(parents=True,exist_ok=True)
    child_env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    child_env.update(env or {})
    begin=time.monotonic();peak=0.;result={'command':command}
    with (folder/'console.log').open('a') as stream:
        proc=subprocess.Popen(command,cwd=ROOT,env=child_env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while proc.poll() is None:
                current=int(Path('/sys/fs/cgroup/memory.current').read_text())
                mem=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                working=(current-int(mem.get('inactive_file',0)))/2**30;peak=max(peak,working)
                available=next(int(l.split()[1])/1024**2 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
                write(folder/'resource_latest.json',{'working_set_gib':working,'peak_working_set_gib':peak,'available_gib':available,'elapsed_seconds':time.monotonic()-begin})
                if working>320 or available<35:raise RuntimeError('working memory safety limit reached')
                if time.monotonic()-START>20*3600:raise TimeoutError('total experiment budget reached')
                if time.monotonic()-begin>seconds:raise TimeoutError('bounded trial time reached')
                time.sleep(3)
            if proc.returncode:raise RuntimeError(f'process failed: {proc.returncode}')
            result['status']='complete'
        except Exception as exc:
            terminate_session(proc)
            (folder/'ckpt_pending.pt').unlink(missing_ok=True)
            (folder/'ckpt_pending.pt.meta.json').unlink(missing_ok=True)
            result.update(status='failed',reason=str(exc))
        except BaseException:
            terminate_session(proc);raise
    result.update(wall_seconds=time.monotonic()-begin,peak_working_set_gib=peak)
    write(folder/'execution.json',result)
    return result['status']=='complete'

def config(name,k=2,batch=2,alpha=.05,profile=None,seed=42,learning_rate=1e-5):
    c=yaml.safe_load((ROOT/'configs/experiments/train_sharded_k2_20261103.yaml').read_text())
    c['run_id']='adaptive_'+name;c['notes']='Single shared model, two GPUs, fixed train/val windows; controlled sample-budget preexperiment; not final success-rate benchmark.'
    c['solver'].update(train_inner_steps=k,alpha_future=alpha)
    c['trainer'].update(seed=seed,learning_rate=learning_rate,per_device_batch_size=batch,max_train_steps=80//batch,warmup_steps=0,save_interval=max(1,20//batch),output_dir=str(OUT/name))
    if profile:
        c['trainer'].update(saved_tensor_gpu_budgets_gib=profile['gpu_budgets'],saved_tensor_cpu_pin_budget_gib=profile['pin'],saved_tensor_host_cache_limit_gib=profile['cache'])
    path=ROOT/f'configs/experiments/adaptive_{name}_20261103.yaml';path.write_text(yaml.safe_dump(c,sort_keys=False));return path,c

def watch(configs):
    path=ROOT/'configs/wandb_bridge.json';settings=json.loads(path.read_text())
    for cfg in configs:
        rel=str(cfg.relative_to(ROOT));item={'config':rel,'group':'adaptive-one-model-20261103','validation':'fixed train/val conditional MSE, sample-aligned candidate RNG, batch2/8; not task success'}
        if not any(s['config']==rel for s in settings['sources']):settings['sources'].append(item)
    path.write_text(json.dumps(settings,indent=2));subprocess.run(['supervisorctl','restart','energywam-wandb'],check=True,timeout=60)

def train_trial(name,k,batch,alpha,profile,seed=42,learning_rate=1e-5,only_direction=None):
    cfg,c=config(name,k,batch,alpha,profile,seed,learning_rate);folder=OUT/name
    watch([cfg]);command=[PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(cfg),'--steps',str(80//batch),'--fixed-window-manifest',str(MANIFEST),'--probe-eval-interval',str(max(1,20//batch))]
    latest=folder/'ckpt_latest.pt'
    if latest.exists():command+=['--resume',str(latest)]
    ok=run(command,folder,env={'ENERGYWAM_TRIM_BETWEEN_DIRECTIONS':SELECTED_TRIM,'ENERGYWAM_SAMPLE_ALIGNED_RNG':'1','ENERGYWAM_ONLY_DIRECTION':only_direction or ''},seconds=4*3600)
    return trial_result(name) if ok else None

def trial_result(name):
    folder=OUT/name;files=sorted(folder.glob('eval_*_future.json'))
    if not files:return None
    last=int(files[-1].stem.split('_')[1]);result={'name':name,'updates':last,'metrics':{}}
    for update,label in [(0,'initial'),(last,'final')]:
        for direction in ['action','future']:
            p=folder/f'eval_{update:04d}_{direction}.json'
            if not p.exists():continue
            rows=json.loads(p.read_text())['rows']
            for split in ['train','val']:
                for ek in [2,4,8]:
                    values=[r['prediction_mse'] for r in rows if r['split']==split and r['k']==ek]
                    if values:result['metrics'][f'{label}/{split}/{direction}/K{ek}']=statistics.mean(values)
    speed=json.loads((folder/'speed_report.json').read_text())['steps']
    resolved=json.loads((folder/'resolved_config.json').read_text())
    result['train_K']=resolved['solver']['train_inner_steps'];result['batch']=resolved['trainer']['per_device_batch_size']
    result['learning_rate']=resolved['trainer']['learning_rate'];result['alpha_future']=resolved['solver']['alpha_future'];result['window_visits']=last*result['batch']
    result['training_seconds']=sum(r['end_to_end_seconds'] for r in speed)
    result['samples_per_second']=result['window_visits']/result['training_seconds']
    result['wall_seconds']=json.loads((folder/'execution.json').read_text())['wall_seconds']
    return result

def metric(result,direction,split='val',label='final'):
    return result['metrics'][f'{label}/{split}/{direction}/K2']

def prune_nonfinalist_checkpoints(keep):
    removed=[]
    for folder in OUT.iterdir():
        if not folder.is_dir() or folder.name in keep or not (folder/'execution.json').exists():continue
        path=folder/'ckpt_latest.pt'
        if path.exists():
            removed.append({'trial':folder.name,'bytes':path.stat().st_size})
            path.unlink();(folder/'ckpt_latest.pt.meta.json').unlink(missing_ok=True)
    if removed:
        audit=OUT/'checkpoint_pruning.json'
        prior=json.loads(audit.read_text()) if audit.exists() else []
        write(audit,prior+[{'kept':sorted(keep),'removed':removed,'reason':'completed nonfinalist; logs/configuration/evaluation retained'}])


def make_expanded_manifest():
    import random
    from energywam.config import load_config
    from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
    cfg=load_config(ROOT/'configs/experiments/train_sharded_k2_20261103.yaml')
    reader=LeRobotEpisodeReader(resolve_dataset_root(cfg));source=json.loads(MANIFEST.read_text())
    val=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices'])
    rng=random.Random(20261103);windows=[]
    for task in sorted({w['task'] for w in source['windows']}):
        selected=[dict(w) for w in source['windows'] if w['task']==task and w['split']=='val']
        pool=[ep for ep in reader.episodes if ep['episode_index'] in val and ep.get('source_dataset','').split('/')[-2]==task and ep['length']>=17]
        used={w['episode_index'] for w in selected};pairs={(w['episode_index'],w['frame_index']) for w in selected}
        while len(selected)<10:
            choices=[ep for ep in pool if ep['episode_index'] not in used] or pool
            if not choices:raise RuntimeError('no validation episodes for '+task)
            ep=rng.choice(choices);frame=rng.randint(0,ep['length']-17)
            if (ep['episode_index'],frame) in pairs:continue
            pairs.add((ep['episode_index'],frame));used.add(ep['episode_index'])
            selected.append({'window_id':1000+len(windows)+len(selected),'task':task,'split':'val',
                'episode_index':ep['episode_index'],'frame_index':frame,'source_dataset':ep['source_dataset']})
        windows.extend(selected)
    path=OUT/'expanded_val_manifest.json';write(path,dict(source,windows=windows,expanded_selection_seed=20261103))
    return path


def evaluate_expanded(result,manifest,profile):
    name='expanded_'+result['name'];cfg,_=config(name,result['train_K'],result['batch'],result['alpha_future'],profile)
    command=[PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(cfg),'--steps','0','--resume',str(OUT/result['name']/'ckpt_latest.pt'),
        '--fixed-window-manifest',str(manifest),'--evaluate-only']
    if not run(command,OUT/name,seconds=3600):return None
    values={}
    for direction in ['action','future']:
        path=OUT/name/f"eval_{result['updates']:04d}_{direction}.json"
        rows=json.loads(path.read_text())['rows'];rows=[r for r in rows if r['k']==2]
        if len(rows)!=100:raise RuntimeError('expanded validation incomplete')
        values[direction]=statistics.mean(r['prediction_mse'] for r in rows)
    return dict(values,trial=result['name'],batch=result['batch'],train_K=result['train_K'],count=100)


def finish(results,profile,reason,recommendation=None):
    equal_time={}
    comparable=[r for r in results if r and not r.get('diagnostic_only') and not r['name'].endswith('_seed43')]
    if len(comparable)>1:
        common=min(r['training_seconds'] for r in comparable)
        for case in comparable:
            folder=OUT/case['name'];speed=json.loads((folder/'speed_report.json').read_text())['steps']
            elapsed=0.;times={0:0.}
            for row in speed:
                elapsed+=row['end_to_end_seconds'];times[row['global_step_completed']]=elapsed
            updates=[int(p.stem.split('_')[1]) for p in folder.glob('eval_*_future.json')]
            update=max((u for u in updates if u in times and times[u]<=common),default=0)
            metrics={}
            for direction in ['action','future']:
                rows=json.loads((folder/f'eval_{update:04d}_{direction}.json').read_text())['rows']
                metrics[direction]=statistics.mean(r['prediction_mse'] for r in rows if r['split']=='val' and r['k']==2)
            equal_time[case['name']]={'update':update,'window_visits':update*case['batch'],'training_seconds':times[update],
                'common_time_budget_seconds':common,'validation':metrics,'note':'nearest preceding evaluation; excludes validation and checkpoint IO'}
    report={'equal_training_time_comparison':equal_time,'scope':'20 fixed training windows, initial20 validation windows; seed replications and expanded100 validation only when documented complete; not global optimum or task success',
        'profile':profile,'results':results,'failed_trials':[dict(json.loads(p.read_text()),folder=str(p.parent)) for p in OUT.glob('*/execution.json') if json.loads(p.read_text()).get('status')=='failed'],'reason':reason,'recommendation':recommendation,'test_used':False}
    write(OUT/'report.json',report)
    lines=['# 单模型双卡预实验结果','',reason,'','|实验|更新数|验证动作 MSE（预测 K2）|验证未来 MSE（预测 K2）|训练样本/秒|','|---|---:|---:|---:|---:|']
    for r in results:
        if r:lines.append(f"|{r['name']}|{r['updates']}|{metric(r,'action'):.7f}|{metric(r,'future'):.7f}|{r['samples_per_second']:.5f}|")
    lines+=['','小样本结论不能称为绝对最优；是否完成种子复核和扩大验证以 recommendation 字段及 expanded_comparison.json 为准。未解决未来方向前不启动正式长训或 Stage2。',
            '性能短跑见 profiles.json；未来步长只读诊断见 future_scale.json。所有候选从相同预训练基础与任务初始化出发，以相同 80 个窗口访问预算比较；验证预测 K2 为主，同时保存 K4/K8。']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n');phase('complete',reason=reason,recommendation=recommendation)
    try:
        subprocess.run([str(ROOT/'.venv-wandb/bin/python'),'scripts/upload_adaptive_shared_report.py'],cwd=ROOT,check=True,timeout=300)
    except Exception as exc:write(OUT/'upload_error.json',{'error':str(exc)})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('only GPUs2/3')
    phase('waiting_for_existing_K8_pilot')
    deadline=time.monotonic()+2*3600
    while True:
        latest=PILOT/'k8/ckpt_latest.pt.meta.json'
        if latest.exists() and json.loads(latest.read_text())['meta']['global_step']>=2:break
        status=json.loads((PILOT/'status.json').read_text())
        if status.get('8',{}).get('phase')=='failed':break
        speed=PILOT/'k8/speed_report.json'
        if speed.exists():
            rows=json.loads(speed.read_text()).get('steps',[])
            if rows and rows[-1]['end_to_end_seconds']>600:
                write(OUT/'pilot_handoff.json',{'reason':'K8 update exceeds ten minutes; optimize execution before committing further training','observed_step_seconds':rows[-1]['end_to_end_seconds'],'completed_updates':rows[-1]['global_step_completed']})
                break
        if time.monotonic()>deadline:break
        time.sleep(5)
    subprocess.run(['supervisorctl','stop','energywam-sharded-k-comparison'],check=False,timeout=60)
    subprocess.run(['supervisorctl','stop','energywam-sharded-k-report'],check=False,timeout=30)
    wait_for_free_gpus()
    shutil.copy2(MANIFEST,OUT/'sample_manifest.json')
    checkpoint=PILOT/'k2/ckpt_latest.pt'
    candidates=[{'name':'cache_trim_baseline','gpu_budgets':[8.,8.],'pin':64.,'cache':80.,'trim':'1'},
                {'name':'cache_reuse','gpu_budgets':[8.,8.],'pin':64.,'cache':80.,'trim':'0'},
                {'name':'more_gpu_cache_reuse','gpu_budgets':[14.,14.],'pin':96.,'cache':128.,'trim':'0'}]
    measured=[]
    for profile in candidates:
        phase('throughput_profile',candidate=profile['name'])
        name='profile_'+profile['name'];cfg,_=config(name,profile=profile)
        cmd=[PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(cfg),'--steps','5','--resume',str(checkpoint),
             '--fixed-window-manifest',str(MANIFEST),'--profile']
        if run(cmd,OUT/name,env={'ENERGYWAM_TRIM_BETWEEN_DIRECTIONS':profile['trim']},seconds=45*60):
            rows=json.loads((OUT/name/'speed_report.json').read_text())['steps']
            # New folder starts at update2 and contains exactly three measured updates.
            seconds=statistics.median(r['end_to_end_seconds'] for r in rows[-2:])
            peak=max(r.get(f'gpu{d}_peak_allocated_gib',0) for r in rows for d in [0,1])
            measured.append(dict(profile,median_seconds=seconds,gpu_peak_gib=peak))
    write(OUT/'profiles.json',measured)
    # Storage-only optimizations must preserve the controlled three-update trajectory.
    import torch
    reference=None
    for profile in measured:
        sample=torch.load(OUT/('profile_'+profile['name'])/'final_parameter_samples.pt',map_location='cpu',weights_only=True)
        if reference is None:reference=sample
        profile['parameter_samples_match']=all(torch.allclose(sample[n],reference[n],rtol=1e-3,atol=1e-6) for n in reference)
    write(OUT/'profiles.json',measured)
    safe=[p for p in measured if p['gpu_peak_gib']<44.5 and p['parameter_samples_match']]
    if not safe:finish([],None,'没有通过容量和余量检查的性能配置，保留现有成果，需先解决执行问题。');return
    best=min(safe,key=lambda p:p['median_seconds'])
    # The actual chosen cache-trim setting must also be used by learning trials.
    global SELECTED_TRIM
    SELECTED_TRIM=best['trim']
    phase('future_step_scale_diagnosis',selected_profile=best)
    cfg,_=config('future_scale',profile=best)
    diagnostic=[PYTHON,'-u','scripts/diagnose_shared_future_scale.py','--config',str(cfg),'--checkpoint',str(checkpoint),
                '--manifest',str(MANIFEST),'--output',str(OUT/'future_scale.json')]
    if not run(diagnostic,OUT/'future_scale',seconds=45*60):finish([],best,'未来步长诊断失败，未启动进一步长训练。');return
    candidate=json.loads((OUT/'future_scale.json').read_text())['candidate_alpha']
    phase('fit_baseline')
    baseline=train_trial('fit_K2_B2_A005',2,2,.05,best);results=[baseline] if baseline else []
    alternative=None
    if candidate:
        phase('fit_future_step_candidate',alpha=candidate)
        alternative=train_trial('fit_K2_B2_Acandidate',2,2,candidate,best)
        if alternative:results.append(alternative)
    capable=[r for r in results if metric(r,'future','train')<.99*metric(r,'future','train','initial') and metric(r,'future')<1.01*metric(r,'future',label='initial') and metric(r,'action')<1.05*metric(r,'action',label='initial')]
    if baseline:
        capable=[r for r in capable if metric(r,'future')<=1.01*metric(baseline,'future') and metric(r,'action')<=1.01*metric(baseline,'action')]
    if not capable and candidate:
        phase('targeted_learning_rate_check',learning_rate=3e-5)
        stronger=train_trial('fit_K2_B2_Acandidate_LR3e5',2,2,candidate,best,learning_rate=3e-5)
        if stronger:
            results.append(stronger)
            if metric(stronger,'future','train')<.99*metric(stronger,'future','train','initial') and metric(stronger,'future')<1.01*metric(stronger,'future',label='initial') and (not baseline or (metric(stronger,'future')<=1.01*metric(baseline,'future') and metric(stronger,'action')<=1.01*metric(baseline,'action'))):capable.append(stronger)
    if not capable:
        if candidate:
            phase('future_only_diagnostic')
            isolated=train_trial('diagnostic_future_only',2,2,candidate,best,learning_rate=3e-5,only_direction='future')
            if isolated:
                isolated['diagnostic_only']=True;results.append(isolated)
                write(OUT/'future_only_diagnostic.json',{'result':isolated,'interpretation':'Tests learning within 80 window visits; not a candidate for the two-direction objective. Improvement in isolation suggests investigating target conflict; failure is not proof of a code bug.'})
        finish(results,best,'已完成的步长候选未同时满足未来训练误差改善超过 1% 且验证不明显退化；不继续消耗预算筛 K/batch，也不宣称找到适合长训的最优配置。暂用 K2、有效 batch2 作为可运行基线，下一步需定位未来方向学习问题。',{'train_K':2,'batch':2,'status':'provisional runnable baseline only'});return
    chosen=min(capable,key=lambda r:metric(r,'action')/metric(r,'action',label='initial')+metric(r,'future')/metric(r,'future',label='initial'))
    alpha=chosen['alpha_future'];learning_rate=chosen['learning_rate']
    prune_nonfinalist_checkpoints({chosen['name']} | ({baseline['name']} if baseline else set()))
    finalists=[chosen]
    for k in [4,8]:
        phase('K_screen',train_k=k)
        r=train_trial(f'fit_K{k}_B2',k,2,alpha,best,learning_rate=learning_rate)
        if r:results.append(r);finalists.append(r)
    # Preserve direction tradeoffs; score only candidates within 1% of baseline on each direction.
    good=[r for r in finalists if metric(r,'action')<=1.01*metric(chosen,'action') and metric(r,'future')<=1.01*metric(chosen,'future')]
    amin=min(metric(r,'action') for r in good);fmin=min(metric(r,'future') for r in good)
    close=[r for r in good if metric(r,'action')<=1.01*amin and metric(r,'future')<=1.01*fmin]
    kbest=max(close,key=lambda r:r['samples_per_second']) if close else chosen
    k=int(kbest['name'].split('_K')[1].split('_')[0])
    prune_nonfinalist_checkpoints({kbest['name']} | ({baseline['name']} if baseline else set()))
    phase('batch_screen',train_k=k,batch=8)
    larger=train_trial(f'fit_K{k}_B8',k,8,alpha,best,learning_rate=learning_rate)
    if larger:results.append(larger)
    candidates=[kbest]+([larger] if larger else [])
    # Within 1% of the best per-direction metric, prefer throughput.
    amin=min(metric(r,'action') for r in candidates);fmin=min(metric(r,'future') for r in candidates)
    close=[r for r in candidates if metric(r,'action')<=1.01*amin and metric(r,'future')<=1.01*fmin]
    winner=max(close,key=lambda r:r['samples_per_second']) if close else kbest
    repeats=[]
    for finalist in candidates:
        phase('seed_replication',trial=finalist['name'],seed=43)
        repeat=train_trial(finalist['name']+'_seed43',k,finalist['batch'],alpha,best,seed=43,learning_rate=learning_rate)
        if repeat:results.append(repeat);repeats.append(repeat)
    phase('expanded_validation')
    expanded=make_expanded_manifest()
    expanded_scores=[]
    for finalist in candidates+repeats:
        evaluation=evaluate_expanded(finalist,expanded,best)
        if evaluation:expanded_scores.append(evaluation)
    write(OUT/'expanded_comparison.json',expanded_scores)
    # Aggregate the common larger validation protocol over available seeds, while
    # explicitly preserving missing replications in the report.
    aggregate=[]
    for batch in sorted({r['batch'] for r in candidates}):
        rows=[r for r in expanded_scores if r['batch']==batch]
        if len(rows)==2:
            aggregate.append({'batch':batch,'action':statistics.mean(r['action'] for r in rows),
                              'future':statistics.mean(r['future'] for r in rows)})
    if len(aggregate)==len(candidates):
        amin=min(r['action'] for r in aggregate);fmin=min(r['future'] for r in aggregate)
        near=[r for r in aggregate if r['action']<=1.01*amin and r['future']<=1.01*fmin]
        qualified=[r for r in candidates if r['batch'] in {a['batch'] for a in near}]
        if qualified:winner=max(qualified,key=lambda r:r['samples_per_second'])
    recommendation={'train_K':k,'batch':8 if winner is larger else 2,'alpha_future':alpha,'learning_rate':learning_rate,'profile':best,'trial':winner['name'],'seed_replications_complete':len(repeats)==len(candidates),'expanded_validation_complete':len(expanded_scores)==len(candidates+repeats),'status':'best among tested small-sample candidates; not global optimum or final 50-task recipe'}
    finish(results,best,'已完成有条件的 K 与 batch 初筛；推荐仅限已测配置及小样本，种子复核和扩大验证的实际完成情况见报告，不代表绝对最优。',recommendation)

SELECTED_TRIM='0'
if __name__=='__main__':
    import signal
    def stop(signum,frame):raise KeyboardInterrupt('supervisor stop')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as exc:
        phase('failed',error=str(exc));raise
