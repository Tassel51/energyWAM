"""Short controlled future-step training sweep; only physical GPUs2/3."""
import json,os,sys,time,statistics,subprocess,shutil,math
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_adaptive_shared_experiments as A
OUT=Path('/cloud/zzc/energyWAM/experiments/future_step_sweep_20261103')
OLD=Path('/cloud/zzc/energyWAM/experiments/adaptive_shared_20261103')
VALUES=[50,100,200,500,1000]


def measurements(folder,upto=None):
    points={}
    for p in sorted(folder.glob('eval_*_future.json')):
        step=int(p.stem.split('_')[1])
        if upto is not None and step>upto:continue
        pair={}
        for direction in ['action','future']:
            file=folder/f'eval_{step:04d}_{direction}.json'
            if not file.exists():break
            rows=json.loads(file.read_text())['rows']
            values=[r['prediction_mse'] for r in rows if r['split']=='val' and r['k']==2]
            if len(values)!=20 or not all(math.isfinite(v) for v in values):raise RuntimeError('validation protocol mismatch or nonfinite MSE')
            pair[direction]=statistics.mean(values)
        if len(pair)==2:points[step]=pair
    return points


def score(folder,upto):
    points=measurements(folder,upto)
    if upto not in points or 0 not in points:raise RuntimeError('incomplete evaluation')
    tail=[points[u] for u in sorted(points) if u>0][-2:]
    return {'updates':upto,'future_mean':statistics.mean(x['future'] for x in tail),
        'action_mean':statistics.mean(x['action'] for x in tail),
        'future_last':points[upto]['future'],'action_last':points[upto]['action'],
        'future_range':max(x['future'] for x in tail)-min(x['future'] for x in tail),
        'action_initial':points[0]['action'],'future_initial':points[0]['future']}


def select_two(records):
    eligible=[r for r in records if r['action_mean']<=1.01*r['action_initial']]
    return sorted(eligible,key=lambda r:(r['future_mean'],r['future_range'],r['alpha']))[:2]


def trial(alpha,seed,steps):
    name=f'alpha{alpha}_seed{seed}';folder=OUT/name
    cfg=yaml.safe_load((ROOT/'configs/experiments/adaptive_fit_K2_B2_A005_20261103.yaml').read_text())
    cfg['run_id']='future_step_'+name
    cfg['notes']='Future step sweep: fresh matched initialization, K2, global batch2, microbatch1, both BI directions, sample-aligned candidate RNG. Validation K2 on fixed20 windows.'
    cfg['solver']['alpha_future']=float(alpha)
    cfg['trainer'].update(seed=seed,max_train_steps=steps,save_interval=10,output_dir=str(folder))
    path=ROOT/f'configs/experiments/future_step_{name}_20261103.yaml'
    path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    command=[A.PYTHON,'-u','scripts/train_shared_sharded.py','--config',str(path),'--steps',str(steps),
        '--fixed-window-manifest',str(OUT/'sample_manifest.json'),'--probe-eval-interval','10']
    ckpt=folder/'ckpt_latest.pt'
    if ckpt.exists():command+=['--resume',str(ckpt)]
    remaining=3*3600-(time.monotonic()-A.START)
    if remaining<=0:raise TimeoutError('three-hour sweep budget reached')
    ok=A.run(command,folder,env={'ENERGYWAM_TRIM_BETWEEN_DIRECTIONS':'0','ENERGYWAM_SAMPLE_ALIGNED_RNG':'1',
        'ENERGYWAM_ONLY_DIRECTION':'','ENERGYWAM_PROBE_MAX_K':'2','ENERGYWAM_PROBE_SPLITS':'val'},seconds=min(45*60,remaining))
    return dict(score(folder,steps),alpha=alpha,seed=seed,name=name) if ok else None


def snapshot(phase,records,**extra):
    A.write(OUT/'status.json',dict(phase=phase,records=records,**extra))
    lines=['# 未来内层步长短对照','',
        '一份模型拆 GPU2/3；K=2、总 batch=2、microbatch=1；动作步长0.05，模型学习率1e-5，warmup=0。',
        '固定20个训练窗口和20个验证窗口；每个窗口都训练两个方向；各步长独立、相同初始化和样本顺序。',
        '第一轮全部20次更新；第二轮优胜两组续到40次；差距小于1%时，再用seed43各20次复核。','',
        '|未来步长|种子|完成更新|最近两次验证未来均值|最近两次动作均值|末次未来MSE|','|---|---|---:|---:|---:|---:|']
    for r in records:lines.append(f"|{r['alpha']}|{r['seed']}|{r['updates']}|{r['future_mean']:.7f}|{r['action_mean']:.7f}|{r['future_last']:.7f}|")
    lines+=['','当前阶段：'+phase,'','1%只用于初筛，不代表统计显著；结果仅限固定小样本与K=2，不是全50任务最优结论。']
    if 'recommendation' in extra:lines+=['','推荐说明：'+extra['recommendation']]
    (OUT/'RESULTS_CN.md').write_text('\n'.join(lines)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='2,3':raise RuntimeError('only GPUs2/3 authorized')
    OUT.mkdir(parents=True,exist_ok=True);A.OUT=OUT;A.START=time.monotonic()
    handoff=OUT/'handoff.json'
    if not handoff.exists():
        meta=OLD/'fit_K4_B2/ckpt_latest.pt.meta.json'
        saved=json.loads(meta.read_text())['meta']['global_step'] if meta.exists() else 0
        target=min(40,saved+10)
        A.write(handoff,{'target_checkpoint':target,'old_saved_step':saved,'phase':'waiting'})
    h=json.loads(handoff.read_text())
    if h['phase']=='waiting':
        snapshot('等待当前K4完整断点后切换',[])
        deadline=time.monotonic()+45*60
        while time.monotonic()<deadline:
            meta=OLD/'fit_K4_B2/ckpt_latest.pt.meta.json'
            saved=json.loads(meta.read_text())['meta']['global_step'] if meta.exists() else 0
            if saved>=h['target_checkpoint']:break
            status=subprocess.run(['supervisorctl','status','energywam-adaptive-shared'],capture_output=True,text=True).stdout
            if 'RUNNING' not in status:break
            time.sleep(5)
        else:raise TimeoutError('no new complete checkpoint; existing training left intact')
        subprocess.run(['supervisorctl','stop','energywam-adaptive-shared'],check=False,timeout=60)
        A.wait_for_free_gpus()
        h.update(phase='complete',saved_step=saved);A.write(handoff,h)
        old=json.loads((OLD/'status.json').read_text());old.update(phase='paused_for_future_step_sweep',replacement=str(OUT))
        A.write(OLD/'status.json',old)
    source=OLD/'sample_manifest.json'
    if not (OUT/'sample_manifest.json').exists():shutil.copy2(source,OUT/'sample_manifest.json')
    # Register all candidate config files before starting the collector once.
    paths=[]
    for alpha in VALUES:
        for seed in [42,43]:
            name=f'alpha{alpha}_seed{seed}'
            cfg=yaml.safe_load((ROOT/'configs/experiments/adaptive_fit_K2_B2_A005_20261103.yaml').read_text())
            cfg['run_id']='future_step_'+name;cfg['solver']['alpha_future']=float(alpha)
            cfg['trainer'].update(seed=seed,output_dir=str(OUT/name),max_train_steps=20)
            path=ROOT/f'configs/experiments/future_step_{name}_20261103.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False));paths.append(path)
    A.watch(paths)
    records=[];first=[]
    for alpha in VALUES:
        snapshot('第一轮：五组各20次更新',records,current_alpha=alpha)
        r=trial(alpha,42,20)
        if r:records.append(r);first.append(r)
    A.write(OUT/'round1.json',first)
    selected=select_two(first)
    A.prune_nonfinalist_checkpoints({r['name'] for r in selected})
    second=[]
    for candidate in selected:
        snapshot('第二轮：优胜组续到40次更新',records,current_alpha=candidate['alpha'])
        r=trial(candidate['alpha'],42,40)
        if r:records.append(r);second.append(r)
    A.write(OUT/'round2.json',second)
    ranked=select_two(second)
    if not ranked:
        snapshot('完成：没有合格候选',records,recommendation='未完成足够的有效对照或动作误差退化，不能选优。');return
    recommendation=f"本轮优先候选为未来步长 {ranked[0]['alpha']}；应结合末次误差和波动，不能认定最终最优。"
    if len(ranked)==2 and abs(ranked[1]['future_mean']/ranked[0]['future_mean']-1)<.01:
        repetitions=[]
        for candidate in ranked:
            snapshot('第三轮：相近候选换seed43各20次',records,current_alpha=candidate['alpha'])
            r=trial(candidate['alpha'],43,20)
            if r:records.append(r);repetitions.append(r)
        A.write(OUT/'round3.json',repetitions)
        if len(repetitions)==2:
            paired=[]
            for r in repetitions:
                original=next(x for x in first if x['alpha']==r['alpha'])
                paired.append(dict(r,future_mean=(r['future_mean']+original['future_mean'])/2,
                    action_mean=(r['action_mean']+original['action_mean'])/2))
            confirmed=select_two(paired)
            if confirmed:
                recommendation=f"两种子相同20次更新预算下优先候选为 {confirmed[0]['alpha']}；另参考seed42的40次结果。仍需与最终K交叉确认。"
    A.write(OUT/'report.json',{'records':records,'recommendation':recommendation,'scope':'short fixed-window sweep; no test data','handoff':h})
    snapshot('完成',records,recommendation=recommendation)


if __name__=='__main__':
    import signal
    def stop(signum,frame):raise KeyboardInterrupt('supervisor stop')
    signal.signal(signal.SIGTERM,stop)
    try:main()
    except Exception as exc:
        A.write(OUT/'failure.json',{'reason':str(exc)});raise
