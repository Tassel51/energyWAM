#!/usr/bin/env python
"""Bounded K=2/4/8 training comparison. Does not resume the formal run."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import yaml
ROOT=Path(__file__).resolve().parents[1]
IS_SHARDED=os.environ.get('ENERGYWAM_K_SHARDED')=='1'
OUT=Path('/cloud/zzc/energyWAM/experiments/train_k248_sharded_20261103' if IS_SHARDED else '/cloud/zzc/energyWAM/experiments/train_k248_20261102')
EXECUTION_NOTE='One model layer-sharded over GPUs2/3; both directions and samples sequential microbatch1, one outer update.' if IS_SHARDED else 'K2 concurrent BI; K4/K8 serialized BI to fit RAM. Same objective, different runtime schedule; not a pure K-only speed benchmark.'
PYTHON='/usr/local/miniconda3/envs/py312/bin/python'

def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)

def summarize(status):
    groups={}
    for k in [2,4,8]:
        folder=OUT/f'k{k}'
        records={}
        log=folder/'train_log.jsonl'
        if log.exists():
            for line in log.read_text().splitlines():
                try:
                    row=json.loads(line)
                    if 'global_step' in row:records[int(row['global_step'])+1]=row
                except ValueError:pass
        values=list(records.values())
        evaluations=[]
        for path in sorted(folder.glob('eval_*.json')):
            data=json.loads(path.read_text())
            for split in ['train','val']:
                for ek in [0,2,4,8]:
                    rows=[r for r in data['rows'] if r['split']==split and r['k']==ek]
                    if rows:evaluations.append({'update':rows[0]['update'],'direction':rows[0]['direction'],
                        'split':split,'eval_k':ek,'mse':statistics.mean(r['prediction_mse'] for r in rows),'count':len(rows)})
        speed=folder/'speed_report.json'
        timing=json.loads(speed.read_text())['steps'] if speed.exists() else []
        groups[str(k)]={'updates':max(records,default=0),'samples':max(records,default=0)*2,
            'median_step_seconds':statistics.median(r['end_to_end_seconds'] for r in timing[2:]) if len(timing)>2 else None,
            'training_seconds_excluding_eval_and_save':sum(r['end_to_end_seconds'] for r in timing),
            'evaluations':evaluations,'status':status.get(str(k),{}),
            'timing_by_update':{int(r['global_step_completed']):r['end_to_end_seconds'] for r in timing},
            'rank0_peak_gpu_allocated_gib':max((r.get('cuda_max_alloc_mb',0)/1024 for r in values),default=0),
            'rank0_peak_process_rss_gib':max((r.get('max_rss_mb',0)/1024 for r in values),default=0)}
    completed_groups=[g for k,g in groups.items() if status.get(k,{}).get('phase')=='complete']
    common_time=min((g['training_seconds_excluding_eval_and_save'] for g in completed_groups),default=0)
    equal_time={}
    for k,g in groups.items():
        elapsed=0;times={0:0}
        for update,seconds in sorted(g['timing_by_update'].items()):
            elapsed+=seconds;times[update]=elapsed
        measured=sorted({r['update'] for r in g['evaluations'] if r['update'] in times and times[r['update']]<=common_time})
        chosen=max(measured,default=0)
        equal_time[k]={'update':chosen,'training_seconds':times.get(chosen,0),
                       'validation':[r for r in g['evaluations'] if r['split']=='val' and r['update']==chosen]}
    write(OUT/'comparison.json',{'equal_training_time_budget_seconds':common_time,'equal_time_nearest_prior_evaluation':equal_time,'groups':groups,'scope':'small-sample fitting, not final benchmark selection',
        'same_initialization':True,'effective_batch':2,'normalization':'existing 50-task train-only statistics; identical across groups',
        'training_windows':20,'validation_windows':20,'test_used':False,'execution_note':EXECUTION_NOTE})
    lines=['# K=2/4/8 小样本训练对照','',
           '每组从相同预训练基础和任务模块初始化出发；20 个训练窗口、20 个验证窗口。有效 batch=2；全展开求导。',
           '共用原 50 任务训练统计，不重新拟合 10 任务统计。本轮只检查拟合与成本，不代表最终泛化结论。','',
           '执行方式：三组均为单份模型拆到 GPU2/3、microbatch=1 累积两方向梯度。' if IS_SHARDED else '补跑执行方式：K=2 两方向并行；K=4/8 为控制内存改成两方向轮流计算。目标和梯度累积规则相同，但速度差异包含执行调度成本，不能全部归因于 K。','',
           '|训练 K|完成更新|稳态步耗时（秒）|最终验证动作 MSE（预测 K=2）|最终验证未来 MSE（预测 K=2）|',
           '|---|---:|---:|---:|---:|']
    for k,g in groups.items():
        def mse(direction):
            rows=[r for r in g['evaluations'] if r['split']=='val' and r['eval_k']==2 and r['direction']==direction]
            return f"{max(rows,key=lambda r:r['update'])['mse']:.8f}" if rows and g['updates']>0 else '未完成训练'
        lines.append(f"|{k}|{g['updates']}|{g['median_step_seconds'] if g['median_step_seconds'] is not None else '—'}|{mse('action')}|{mse('future')}|")
    lines+=['','完整数据见 comparison.json；训练 K 与验证 K 分开记录。',
            '若未来方向未改善，不据动作误差单独宣布最佳；未完成的组不按已完成处理。']
    lines+=['','相同训练时间比较（取不超过共同预算的最近评价点，评价间隔会造成预算余量）：','',
            '|训练 K|评价更新数|累计训练秒数|','|---|---:|---:|']
    for k,g in equal_time.items():lines.append(f"|{k}|{g['update']}|{g['training_seconds']:.1f}|")
    finished=all(status.get(str(k),{}).get('phase') in ['complete','failed'] for k in [2,4,8])
    if finished:
        def last(group,split,direction,ek=2):
            selected=[r for r in group['evaluations'] if r['split']==split and r['direction']==direction and r['eval_k']==ek]
            return max(selected,key=lambda r:r['update']) if selected else None
        eligible={k:g for k,g in groups.items() if status.get(k,{}).get('phase')=='complete'}
        lines+=['','## 本轮结论','']
        if len(eligible)<3:
            lines.append('三组训练比较未完成，无法判断 K=2/4/8 哪个学习效果最好。失败原因见 status.json，历史失败与补跑记录分开保存，不能把未训练的初始评测当作训练成绩。')
        baseline=eligible.get('2')
        if baseline:
            baction=last(baseline,'val','action');bfuture=last(baseline,'val','future')
            better=[]
            for k,g in eligible.items():
                if k=='2':continue
                a,f=last(g,'val','action'),last(g,'val','future')
                if a and f and baction and bfuture and a['mse']<baction['mse']*.99 and f['mse']<bfuture['mse']*.99:
                    better.append(k)
            if len(eligible)<2:
                lines.append('只有 K=2 完成，当前仅保留为已验证可运行的基线；不能据此宣称 K=2 更优。没有有效的相同时间跨组比较。')
            elif better:
                lines.append('相同更新预算下，两方向均比 K=2 低超过 1% 的候选：'+','.join(better)+'。这是初筛门槛，尚需更多验证窗口和随机种子确认。')
            else:
                lines.append('本轮没有更大 K 在两方向验证 MSE 上同时比 K=2 降低超过 1%；暂保留 K=2 作为低成本训练基线。1% 是筛选阈值，不是统计显著性检验。')
            initial=[r for r in baseline['evaluations'] if r['split']=='train' and r['direction']=='future' and r['eval_k']==2 and r['update']==0]
            final=last(baseline,'train','future')
            if initial and final and final['mse']>initial[0]['mse']*.99:
                lines.append('K=2 的固定训练样本未来误差也未改善超过 1%；下一步优先排查未来方向的梯度、尺度及精度，不扩大长训。')
        else:lines.append('K=2 基线未完成，本轮无法给出可信的配置选择。')
        lines.append('此结论仅限本轮小样本与单随机种子；不能认定最终 50 任务最优 K。')
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')

def terminate_session(process):
    """Wait for every live worker, even if the launcher exited first."""
    import signal
    def live():
        found=[]
        for item in Path('/proc').iterdir():
            if not item.name.isdigit():continue
            try:
                raw=(item/'stat').read_text()
                fields=raw[raw.rfind(')')+2:].split()
                if int(fields[3])==process.pid and fields[0]!='Z':found.append(int(item.name))
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass
        return found
    for sig,seconds in [(signal.SIGTERM,5),(signal.SIGKILL,30)]:
        for pid in live():
            try:os.kill(pid,sig)
            except ProcessLookupError:pass
        deadline=time.monotonic()+seconds
        while live() and time.monotonic()<deadline:time.sleep(.1)
        if not live():break
    process.wait(timeout=5)
    if live():raise RuntimeError('previous worker session is still alive; refusing next experiment')


def wait_for_free_gpus():
    """Never launch a group while GPUs 2/3 still have any compute process."""
    ids=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    allowed={line.split(',')[1].strip() for line in ids.splitlines() if line.split(',')[0].strip() in ['2','3']}
    deadline=time.monotonic()+45
    while True:
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        busy=[line for line in apps.splitlines() if line.split(',')[0].strip() in allowed]
        if not busy:return
        if time.monotonic()>deadline:raise RuntimeError('GPUs 2/3 still occupied; refusing to overlap experiments: '+str(busy))
        time.sleep(1)


def run_group(k,target,status):
    folder=OUT/f'k{k}';folder.mkdir(exist_ok=True)
    checkpoint=folder/'ckpt_latest.pt'
    config=ROOT/(f'configs/experiments/train_sharded_k{k}_20261103.yaml' if IS_SHARDED else f'configs/experiments/train_k{k}_20261102.yaml')
    cmd=[PYTHON,'-u','scripts/train_shared_sharded.py' if IS_SHARDED else 'scripts/train_replicated.py','--config',str(config),'--steps',str(target),
         '--fixed-window-manifest',str(OUT/'sample_manifest.json'),'--probe-eval-interval','20']
    if checkpoint.exists():cmd+=['--resume',str(checkpoint)]
    wait_for_free_gpus()
    status[str(k)]={'phase':'running','target_updates':target,'serial_bi_directions':k in [4,8],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(OUT/'status.json',status)
    start=time.monotonic()
    with (folder/'console.log').open('a') as stream:
        process=subprocess.Popen(cmd,cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ENERGYWAM_SERIAL_BI='1' if k in [4,8] else '0'),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic()-start>(8 if k in [4,8] else 4)*3600:
                    raise RuntimeError('single-group bounded time budget exceeded')
                # Include child RSS; conservative shared-page double counting is intentional.
                rss=0
                for item in Path('/proc').iterdir():
                    if not item.name.isdigit():continue
                    try:
                        stat=(item/'stat').read_text().split()
                        if int(stat[5])==process.pid:
                            for line in (item/'status').read_text().splitlines():
                                if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
                    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
                available=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
                # RSS can double count pinned/shared mappings. Use cgroup's actual charged
                # memory plus host availability, rather than declaring physical OOM from RSS.
                current=int(Path('/sys/fs/cgroup/memory.current').read_text())
                resource={'group_rss_gib':rss/2**30,'cgroup_memory_current_gib':current/2**30,
                          'available_gib':available/2**30,'elapsed_seconds':time.monotonic()-start}
                write(folder/'resource_latest.json',resource)
                memstat=dict(line.split() for line in Path('/sys/fs/cgroup/memory.stat').read_text().splitlines())
                working=current-int(memstat.get('inactive_file',0))
                resource['cgroup_working_set_gib']=working/2**30
                write(folder/'resource_latest.json',resource)
                if working>320*2**30 or available<35*2**30:
                    raise RuntimeError(f'RAM guard: actual cgroup={current/2**30:.1f}GiB available={available/2**30:.1f}GiB RSS={rss/2**30:.1f}GiB')
                time.sleep(3)
            if process.returncode:raise RuntimeError(f'worker failed ({process.returncode}); see console.log')
            status[str(k)]={'phase':'pilot_complete' if target==2 else 'complete','target_updates':target,'wall_seconds':time.monotonic()-start,'serial_bi_directions':k in [4,8]}
        except BaseException as exc:
            terminate_session(process)
            status[str(k)]={'phase':'failed','reason':str(exc),'target_updates':target}
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
    write(OUT/'status.json',status);summarize(status)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('requires GPUs 2,3 only')
    formal=subprocess.run(['supervisorctl','status','energywam-gpu23'],capture_output=True,text=True).stdout
    if 'STOPPED' not in formal and 'EXITED' not in formal:raise RuntimeError('formal training must remain stopped')
    original=Path('/cloud/zzc/energyWAM/experiments/inner_diag_20261102/sample_manifest.json')
    if not (OUT/'sample_manifest.json').exists():(OUT/'sample_manifest.json').write_bytes(original.read_bytes())
    status=json.loads((OUT/'status.json').read_text()) if (OUT/'status.json').exists() else {}
    # Test capacity for every K before committing the rest of its budget.
    for target in [2,100]:
        for k in [2,4,8]:
            if status.get(str(k),{}).get('phase')=='failed':continue
            meta=OUT/f'k{k}/ckpt_latest.pt.meta.json'
            completed=json.loads(meta.read_text())['meta']['global_step'] if meta.exists() else 0
            if completed>=target:continue
            run_group(k,target,status)
        if target==2:
            fingerprints=[]
            for k in [2,4,8]:
                path=OUT/f'k{k}/initialization_step0.json'
                if path.exists():fingerprints.append(json.loads(path.read_text())['sampled_parameter_sha256'])
            if len(set(fingerprints))>1:raise RuntimeError('different initial parameter fingerprints')
    summarize(status)
    write(OUT/'complete.json',{'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':status})

if __name__=='__main__':
    import signal
    def stop(signum, frame):
        raise KeyboardInterrupt('supervisor requested stop')
    signal.signal(signal.SIGTERM,stop)
    main()
