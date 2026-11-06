#!/usr/bin/env python
"""Matched training-fit diagnostic on two explicitly authorized GPU pairs."""
import json,os,signal,subprocess,sys,time,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_k_training_comparison import terminate_session
PYTHON='/usr/local/miniconda3/envs/py312/bin/python'
OUT=Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103')

def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

def collect(k):
    folder=OUT/f'k{k}';points=[]
    for f in sorted(folder.glob('eval_*_future.json')):
        update=int(f.stem.split('_')[1]);row={'update':update}
        for direction in ['action','future']:
            p=folder/f'eval_{update:04d}_{direction}.json'
            if not p.exists():continue
            data=json.loads(p.read_text())['rows']
            for ek in [2,4]:
                values=[x['prediction_mse'] for x in data if x['split']=='train' and x['k']==ek]
                if len(values)!=80 or not all(math.isfinite(v) for v in values):raise RuntimeError('training-fit evaluation protocol mismatch or nonfinite error')
                row[f'{direction}_K{ek}']=statistics.mean(values)
        if len(row)==5:points.append(row)
    timing=[]
    f=folder/'timing_history.jsonl'
    if f.exists():
        by_step={}
        for line in f.read_text().splitlines():
            try:r=json.loads(line);by_step[r['global_step_completed']]=r
            except ValueError:pass
        timing=sorted(by_step.values(),key=lambda x:x['global_step_completed'])
    elapsed=0.;times={0:0.}
    for r in timing:
        elapsed+=r['end_to_end_seconds'];times[r['global_step_completed']]=elapsed
    for p in points:p['training_seconds']=times.get(p['update'])
    initial=folder/'initialization_step0.json'
    return {'train_K':k,'physical_gpus':[0,1] if k==2 else [2,3],
        'completed_updates':timing[-1]['global_step_completed'] if timing else 0,
        'training_seconds':elapsed,'evaluations':points,
        'initialization':json.loads(initial.read_text()) if initial.exists() else None}

def report(states):
    groups={str(k):collect(k) for k in [2,4]}
    common=min(g['training_seconds'] for g in groups.values())
    equal={}
    for k,g in groups.items():
        rows=[p for p in g['evaluations'] if p['training_seconds'] is not None and p['training_seconds']<=common]
        equal[k]=rows[-1] if rows else None
    init=[g['initialization'] for g in groups.values()]
    matched=all(init) and len({x['sampled_parameter_sha256'] for x in init})==1
    write(OUT/'comparison.json',{'groups':groups,'process_states':states,'sampled_initialization_matches':matched,
        'equal_training_time_budget_seconds':common,'equal_time_nearest_prior_evaluation':equal,
        'scope':'80 fixed training examples from 10 train episodes; no held-out validation or test; cannot establish best generalizing K'})
    lines=['# K=2/4 小样本过拟合检查','',
        '未来步长200，动作步长0.05；有效batch=2，microbatch=1；各200次权重更新，固定80条训练样本来自10条训练轨迹。',
        '训练集检查使用固定候选种子，分别报告预测K=2/4；无验证集或测试集。初始化一致性仅检查每个参数前64个元素，不是全量权重证明。','',
        '|训练K|完成权重更新|最近检查更新|预测K=2动作MSE|预测K=2未来MSE|预测K=4动作MSE|预测K=4未来MSE|','|---|---:|---:|---:|---:|---:|---:|']
    for k,g in groups.items():
        p=g['evaluations'][-1] if g['evaluations'] else {}
        cells=[k,g['completed_updates'],p.get('update','—')]+[f"{p[x]:.7f}" if x in p else '—' for x in ['action_K2','future_K2','action_K4','future_K4']]
        lines.append('|'+ '|'.join(map(str,cells))+'|')
    lines+=['','进程状态：'+json.dumps(states,ensure_ascii=False),'',
        '判断：分别看两个方向是否学会、相同更新次数的误差、相同训练时间的误差和达到共同误差水平的耗时。未完成不能宣布最佳；训练集拟合结果不代表泛化。',
        '检查点间隔20次，共同时间比较使用不超过预算的最近检查点，存在时间余量。最新恢复断点每组仅保留一个。']
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')

def main():
    children={};streams=[];states={};started=time.monotonic()
    def stop(sig,frame):raise KeyboardInterrupt('supervisor stop')
    signal.signal(signal.SIGTERM,stop)
    try:
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if query:raise RuntimeError('GPUs must be idle before launching paired experiment; no existing processes will be killed')
        for k,pair in [(2,'0,1'),(4,'2,3')]:
            folder=OUT/f'k{k}';folder.mkdir(exist_ok=True)
            cfg=ROOT/f'configs/experiments/overfit_k{k}_alpha200_20261103.yaml'
            command=[PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(cfg),'--steps','200',
                '--physical-gpus',pair,'--fixed-window-manifest',str(OUT/'sample_manifest.json'),'--probe-eval-interval','20']
            latest=folder/'ckpt_latest.pt'
            if latest.exists():command+=['--resume',str(latest)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=pair,ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',
                ENERGYWAM_SAMPLE_ALIGNED_RNG='1',ENERGYWAM_ONLY_DIRECTION='',ENERGYWAM_PROBE_MAX_K='4',
                ENERGYWAM_PROBE_SPLITS='train',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
            stream=(folder/'console.log').open('a');streams.append(stream)
            children[k]=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            states[str(k)]={'phase':'running','pid':children[k].pid,'physical_gpus':pair}
        while True:
            current=int(Path('/sys/fs/cgroup/memory.current').read_text());stat=dict(x.split() for x in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
            working=(current-int(stat.get('inactive_file',0)))/2**30
            available=next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
            write(OUT/'resource_latest.json',dict(working_set_gib=working,available_gib=available,wall_seconds=time.monotonic()-started))
            if working>300 or available<35:raise RuntimeError('paired host memory budget exceeded; stop own experiments and retain rolling checkpoints')
            if time.monotonic()-started>12*3600:raise TimeoutError('12-hour diagnostic budget reached')
            for k,proc in children.items():
                if proc.poll() is not None:
                    states[str(k)].update(phase='complete' if proc.returncode==0 else 'failed',returncode=proc.returncode)
            report(states);write(OUT/'status.json',states)
            if all(p.poll() is not None for p in children.values()):break
            time.sleep(10)
    except BaseException as exc:
        write(OUT/'failure.json',{'reason':str(exc),'states':states})
        for p in children.values():terminate_session(p)
        for state in states.values():
            if state['phase']=='running':state['phase']='interrupted'
        report(states);write(OUT/'status.json',states)
        raise
    finally:
        for stream in streams:stream.close()

if __name__=='__main__':main()
