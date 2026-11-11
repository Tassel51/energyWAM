#!/usr/bin/env python3
"""Read-only, dependency-free terminal dashboard for local lab experiment queues.

Ctrl-C stops this dashboard only. No torch import, GPU allocation, SSH, signals,
training, evaluation or checkpoint writes are performed.
"""
import argparse
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import socket
import subprocess
import sys
import time

TZ=timezone(timedelta(hours=8))
LABELS={'loss':'Future 损失权重','steps':'内层步长','k':'迭代次数 K','future07':'Future步长减半（GPU0/7补跑）','future67':'Future步长减半（GPU6/7原配置补跑）'}
_CACHE={}


def read_json(path):
    path=Path(path)
    try:
        stat=path.stat();key=(stat.st_mtime_ns,stat.st_size)
        cached=_CACHE.get(str(path))
        if cached and cached[0]==key:return cached[1]
        value=json.loads(path.read_text());_CACHE[str(path)]=(key,value);return value
    except (OSError,ValueError):return None


def tail_update(path):
    try:
        with Path(path).open('rb') as f:
            f.seek(0,2);size=f.tell();f.seek(max(0,size-262144));lines=f.read().splitlines()
        for line in reversed(lines):
            try:row=json.loads(line)
            except ValueError:continue
            if 'global_step' in row and 'optimizer_lr/energy_head' in row:
                return int(row['global_step'])+1
    except (OSError,ValueError,TypeError):pass
    return None


def process_info(pid, expected):
    if not pid:return {'alive':False,'state':None}
    try:
        proc=Path('/proc')/str(int(pid));args=(proc/'cmdline').read_bytes().decode(errors='replace').replace('\x00',' ')
        stat=(proc/'stat').read_text().split(') ',1)[1].split()[0]
        return dict(alive=all(word in args for word in expected) and stat!='Z',state=stat)
    except (OSError,ValueError,IndexError):return {'alive':False,'state':None}


def fixed_metrics(folder, split, k):
    folder=Path(folder)
    if not folder.exists():return None
    # Read newest common A/F pair, never silently combine different updates.
    a={p.stem.split('_')[1]:p for p in folder.glob('eval_*_action.json')}
    f={p.stem.split('_')[1]:p for p in folder.glob('eval_*_future.json')}
    for step in sorted(a.keys()&f.keys(),reverse=True):
        result=dict(update=int(step),k=k,split=split);valid=True
        for direction,path in [('action',a[step]),('future',f[step])]:
            report=read_json(path)
            rows=[r for r in (report or {}).get('rows',[]) if r.get('split')==split and r.get('k')==k]
            if not rows:valid=False;break
            result[direction]=sum(float(r['prediction_mse']) for r in rows)/len(rows)
            result[direction+'_samples']=len(rows)
        if valid:return result
    return None


def checkpoint_step(folder, role):
    path=Path(folder)/('ckpt_'+role+'.pt');meta=read_json(str(path)+'.meta.json')
    if not path.is_file():return None
    return (meta or {}).get('meta',{}).get('global_step')


def arm_info(arm, queue):
    folder=Path(arm['folder']);state=queue.get(arm['name'],{})
    phase=state.get('phase','queued');speed=read_json(folder/'speed_report.json');initial=read_json(folder/'initialization_step0.json')
    count=(speed or {}).get('completed_steps');source='speed_report'
    log_count=tail_update(folder/'train_log.jsonl')
    if log_count is not None and (count is None or log_count>count):count=log_count;source='train_log'
    if count is None and initial is not None:count=initial.get('global_step');source='initialization'
    if count is None and isinstance(state.get('completed_updates'),int):count=state['completed_updates'];source='queue_record'
    if count is None and phase in ['queued','waiting_for_idle_gpus','waiting_for_checkpoint_space','waiting_for_gpu_pair_lock']:
        count=0;source='not_started'
    target=state.get('target_updates',arm.get('target_updates',210))
    execution=read_json(folder/'execution_options.json') or {};config=execution.get('config',state.get('config',''))
    expected=['train_shared_sharded.py',str(config)] if config else ['train_shared_sharded.py',str(folder.parent)]
    process=process_info(state.get('pid'),expected)
    if phase=='running' and not process['alive']:phase='process_exited_or_pid_changed'
    k=arm.get('factors',{}).get('train_k',2)
    train=fixed_metrics(folder/'train_monitor','train',k) or fixed_metrics(folder,'train',k)
    val=fixed_metrics(folder,'val',k)
    log=folder/'train_log.jsonl'
    if not log.exists():log=folder/'console.log'
    try:updated=log.stat().st_mtime
    except OSError:updated=None
    resources=read_json(folder/'resource_summary.json') or {}
    reason=state.get('reason') or resources.get('guard_reason') or (resources.get('exit_classification') if phase=='failed' else None)
    return dict(name=arm['name'],factors=arm.get('factors',{}),phase=phase,pid=state.get('pid'),process=process,completed=count,target=target,count_source=source,train=train,val=val,log_updated=updated,log_file=log.name if updated else None,latest=checkpoint_step(folder,'latest'),best=checkpoint_step(folder,'best'),reason=reason)


def study_info(spec):
    root=Path(spec['root']);plan=read_json(root/'plan.json');controller=read_json(root/'controller_status.json')
    queue=read_json(root/'full_status.json');queue_exists=(root/'full_status.json').exists()
    if plan is None:return dict(name=spec['name'],root=str(root),error='plan.json 缺失或暂不可读',arms=[])
    jobs=read_json(plan.get('full_jobs',root/'full_jobs.json')) or []
    targets={j['name']:j.get('steps',210) for j in jobs}
    arms=[arm_info(dict(a,target_updates=targets.get(a['name'],210)),queue or {}) for a in plan['arms']]
    controller_script='run_lab_after_study.py' if (controller or {}).get('phase') in ['waiting_for_preceding_study','blocked_dependency'] else 'run_lab_ablation.py'
    controller_process=process_info((controller or {}).get('pid'),[controller_script,str(root/'plan.json')])
    return dict(name=spec['name'],root=str(root),gpu_pair=plan['gpu_pair'],controller_phase=(controller or {}).get('phase','unknown'),controller_process=controller_process,queue_read_missing=queue_exists and queue is None,arms=arms,confirmed_updates=sum(a['completed'] or 0 for a in arms),unknown_counts=sum(a['completed'] is None for a in arms),total_updates=sum(a['target'] for a in arms),complete_arms=sum(a['phase']=='complete' for a in arms),profile=read_json(root/'profile_status.json'))


def gpu_info():
    try:
        text=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=4)
        return [dict(zip(['index','name','used_mib','free_mib','util'],[x.strip() for x in line.split(',')])) for line in text.splitlines()]
    except (OSError,subprocess.SubprocessError):return []


def bar(done,total,width=22):
    if done is None:return '['+'?'*width+'] 未记录 / '+str(total)
    fill=int(width*max(0,min(done,total))/max(total,1))
    return '['+'#'*fill+'.'*(width-fill)+f'] {done}/{total} ({100*done/max(total,1):.1f}%)'


def metrics_text(label,value):
    if value is None:return label+': 未记录'
    return f"{label}@{value['update']} K={value['k']}: Action {value['action']:.6f} | Future {value['future']:.6f}"


def render(snapshot):
    lines=[f"energyWAM 学校服务器进度 | {snapshot['hostname']} | {snapshot['time']}", '只读面板；Ctrl-C只退出面板。固定评价误差不是当前随机batch误差。','']
    translations={'queued':'排队','running':'运行中','complete':'完成','failed':'失败','interrupted':'中断','waiting_for_idle_gpus':'等待GPU/显存门槛','waiting_for_checkpoint_space':'等待磁盘空间','waiting_for_gpu_pair_lock':'等待本项目GPU锁','process_exited_or_pid_changed':'进程已退出/状态文件未更新'}
    for group in snapshot['studies']:
        if group.get('error'):lines.extend([LABELS.get(group['name'],group['name'])+': '+group['error'],'']);continue
        lines.append(f"【{LABELS.get(group['name'],group['name'])}】GPU {group['gpu_pair']} | 队列阶段 {group['controller_phase']}")
        lines.append('  总进度 '+bar(group['confirmed_updates'],group['total_updates'])+f" | 完成组 {group['complete_arms']}/{len(group['arms'])}"+(' | 有未记录步数，仅显示已确认部分' if group['unknown_counts'] else ''))
        if not group['controller_process']['alive'] and group['controller_phase'] not in ['complete','failed','interrupted','blocked_capacity','blocked_dependency']:
            lines.append('  注意：控制进程已退出或PID已变化，以各组日志为准')
        if group['queue_read_missing']:lines.append('  注意：队列JSON暂不可读；本帧状态可能不完整')
        for i,arm in enumerate(group['arms'],1):
            phase=translations.get(arm['phase'],arm['phase'])
            lines.append(f"  {i}. {arm['name']} | {phase} "+bar(arm['completed'],arm['target']))
            if arm['phase'] not in ['queued','complete']:
                lines.append('     '+metrics_text('固定TRAIN',arm['train'])+' | '+metrics_text('VAL',arm['val']))
                stamp=datetime.fromtimestamp(arm['log_updated'],TZ).strftime('%m-%d %H:%M:%S +0800') if arm['log_updated'] else '未记录'
                process_label=(f" | PID {arm['pid']} 存活({arm['process']['state']})" if arm['process']['alive'] else f" | 原PID {arm['pid']} 已退出或复用" if arm['pid'] else '')
                lines.append(f"     日志 {stamp}"+process_label+(f" ({arm['log_file']})" if arm['log_file'] else '')+f" | checkpoint latest={arm['latest'] if arm['latest'] is not None else '无'} best={arm['best'] if arm['best'] is not None else '无'}")
                if arm['reason']:lines.append('     原因: '+arm['reason'])
        lines.append('')
    indices={i for g in snapshot['studies'] for i in g.get('gpu_pair','').split(',')}
    lines.append('GPU 整卡占用（含共享任务）：编号 | 型号 | 已用/空闲 MiB | 利用率')
    for row in snapshot['gpus']:
        if row['index'] in indices:lines.append(f"  {row['index']:>2} | {row['name'].replace('NVIDIA ','')} | {row['used_mib']}/{row['free_mib']} | {row['util']}%")
    if not snapshot['gpus']:lines.append('  GPU查询不可用')
    return '\n'.join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--index',type=Path,default=Path(__file__).with_name('study_dashboard.json'));p.add_argument('--interval',type=float,default=5);p.add_argument('--once',action='store_true');p.add_argument('--json',action='store_true');args=p.parse_args()
    if args.interval<1:p.error('刷新间隔至少1秒')
    index=read_json(args.index)
    if not index:p.error('缺少或无法读取索引: '+str(args.index))
    try:
        while True:
            snapshot=dict(hostname=socket.gethostname(),time=datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S +0800'),studies=[study_info(g) for g in index['groups']],gpus=gpu_info())
            if sys.stdout.isatty() and not args.once:print('\033[2J\033[H',end='')
            print(json.dumps(snapshot,ensure_ascii=False,indent=2) if args.json else render(snapshot),flush=True)
            if args.once or args.json:return
            time.sleep(args.interval)
    except KeyboardInterrupt:print('\n进度面板已退出；训练任务未受影响。')


if __name__=='__main__':main()
