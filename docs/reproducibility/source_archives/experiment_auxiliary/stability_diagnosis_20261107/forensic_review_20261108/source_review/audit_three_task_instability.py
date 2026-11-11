"""CPU-only forensic audit of the completed seed42 three-task run.

Reads immutable training records, reconstructs the original sampler and checks
raw action/state data. Optionally recomputes frozen DINO target baselines on CPU.
Never loads Qwen, resumes training, or changes source experiment records.
"""
import argparse
import collections
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pyarrow.parquet as pq
import torch
torch.set_num_threads(1)

from energywam.config import load_config
from energywam.data.energy_dataset import resolve_dataset_root
from energywam.data.episode_dataset import LeRobotEpisodeReader
from energywam.training.run_state import FixedSampleBudgetSampler

BASE = Path('/cloud/zzc/energyWAM/experiments')
SOURCE = BASE/'three_task_full_20261106/three_task_b16_micro4_seed42'
OUT = BASE/'stability_diagnosis_20261107/forensic_review_20261108'
SEGMENTS = [(1,16),(17,256),(257,400),(401,491),(492,520),(521,560),
            (561,600),(601,665),(666,684),(685,689),(690,704)]

def read(p):
    return json.loads(Path(p).read_text())

def save(name, data):
    p=OUT/name
    temp=p.with_suffix(p.suffix+'.pending')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    temp.replace(p)

def summary(values):
    v=np.asarray(values,dtype=float)
    return dict(count=len(v),mean=float(v.mean()),median=float(np.median(v)),
                min=float(v.min()),max=float(v.max()),p95=float(np.quantile(v,.95)))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dino',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=load_config(ROOT/'configs/experiments/three_task_b16_micro4_seed42.yaml')
    source_cfg=read(SOURCE/'resolved_config.json')
    assert cfg.trainer.seed==source_cfg['trainer']['seed']==42
    records=[json.loads(s) for s in (SOURCE/'timing_history.jsonl').read_text().splitlines()]
    trainlog=[json.loads(s) for s in (SOURCE/'train_log.jsonl').read_text().splitlines()]
    assert len(records)==len(trainlog)==704
    assert [r['global_step_completed'] for r in records]==list(range(1,705))
    for r,t in zip(records,trainlog):
        assert r['global_step_completed']==int(t['global_step'])+1
        assert all(r[k]==t[k] for k in ['loss','bi/action_loss','bi/future_loss','param_grad_norm'])
    reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
    episodes={e['episode_index']:e for e in reader.episodes}
    manifest=read(SOURCE/'sample_manifest.json')
    assert reader.dataset_identity==manifest['dataset_identity']
    windows=[w for w in manifest['windows'] if w['split']=='train']
    assert len(windows)==44768
    order=list(FixedSampleBudgetSampler(len(windows),42,0,704*16))
    assert len(order)==len(set(order))==11264
    batches={u:[windows[i] for i in order[(u-1)*16:u*16]] for u in range(1,705)}
    stats=read(ROOT/cfg.data.norm_stats_bundle_path)
    st=stats['action'];mean=np.array(st['mean'],dtype=np.float32);std=np.array(st['std'],dtype=np.float32)
    by_episode=collections.defaultdict(list)
    for u,ws in batches.items():
        for w in ws:by_episode[w['episode_index']].append((u,w))
    checks={};episode_checks=[];anomalies=[]
    columns=['action','observation.state','frame_index','episode_index','timestamp']
    for ix,(ep,used) in enumerate(sorted(by_episode.items())):
        meta=episodes[ep]
        p=reader.root/reader.info['data_path'].format(episode_chunk=0,episode_index=ep)
        assert p.exists(),p
        table=pq.read_table(p,columns=columns)
        a=np.asarray(table['action'].to_pylist(),dtype=np.float32)
        q=np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)
        frames=np.asarray(table['frame_index']);ids=np.asarray(table['episode_index']);times=np.asarray(table['timestamp'])
        ec=dict(episode_index=ep,source_episode_index=meta.get('source_episode_index'),
                source_dataset=meta['source_dataset'],file=str(p),resolved_file=str(p.resolve()),
                metadata_length=meta['length'],rows=len(a),actions_shape=list(a.shape),states_shape=list(q.shape),
                action_finite=bool(np.isfinite(a).all()),state_finite=bool(np.isfinite(q).all()),
                frame_indices_contiguous=bool(np.array_equal(frames,np.arange(len(a)))),
                source_episode_id_matches=bool(np.all(ids==meta['source_episode_index'])),
                timestamps_finite_monotonic=bool(np.isfinite(times).all() and np.all(np.diff(times)>0)))
        ec['valid']=bool(len(a)==meta['length'] and a.shape==(len(a),12) and q.shape==(len(a),16)
                         and all(ec[k] for k in ['action_finite','state_finite','frame_indices_contiguous',
                                                 'source_episode_id_matches','timestamps_finite_monotonic']))
        episode_checks.append(ec)
        if not ec['valid']:anomalies.append(ec)
        for u,w in used:
            t=w['frame_index'];chunk=a[t:t+16];z=(chunk-mean)/(std+st['epsilon'])
            c=dict(update=u,episode_index=ep,frame_index=t,task=w['task'],
                   source_task_matches=w['task']==meta['source_dataset'].split('/')[-2],
                   action_finite=bool(np.isfinite(chunk).all()),state_finite=bool(np.isfinite(q[t]).all()),
                   full_horizon=chunk.shape==(16,12),future_in_bounds=t+16<len(a),
                   raw_action_absmax=float(np.abs(chunk).max()),action_z_absmax=float(np.abs(z).max()),
                   zero_action_mse=float(np.square(z).mean()),state_absmax=float(np.abs(q[t]).max()))
            checks[(ep,t)]=c
        if ix%250==0:print(f'Raw data checked {ix+1}/{len(by_episode)} episodes',flush=True)
    per_update=[]
    diagnostic_keys=['param_grad_norm','param_grad_nonfinite','diag/alpha_action','diag/alpha_future',
                     'diag/alpha_action_after_update','diag/alpha_future_after_update',
                     'diag/step_action_bound_fraction','diag/step_future_bound_fraction',
                     'diag/head_action_gradient_norm','diag/head_weighted_future_gradient_norm',
                     'diag/head_direction_cosine','diag/action_action_update_norm_max',
                     'diag/future_future_update_norm_max','diag/action_action_grad_norm_last',
                     'diag/future_future_grad_norm_last','end_to_end_seconds','data_seconds']
    for r in records:
        u=r['global_step_completed'];ws=batches[u];cs=[checks[(w['episode_index'],w['frame_index'])] for w in ws]
        baseline=statistics.mean(w['zero_action_mse'] for w in cs)
        row=dict(update=u,action_mse=r['bi/action_loss'],future_mse=r['bi/future_loss'],loss=r['loss'],
                 zero_action_baseline_mse=baseline,action_to_zero_baseline_ratio=r['bi/action_loss']/baseline,
                 task_counts=dict(collections.Counter(w['task'] for w in ws)),
                 action_z_absmax=max(w['action_z_absmax'] for w in cs),
                 implied_global_clip_multiplier=min(1.,1./(r['param_grad_norm']+1e-6)),
                 **{k:r[k] for k in diagnostic_keys})
        per_update.append(row)
    metrics=['action_mse','future_mse','loss','zero_action_baseline_mse','action_to_zero_baseline_ratio',
             'action_z_absmax',*diagnostic_keys]
    segments=[]
    for lo,hi in SEGMENTS:
        rs=[r for r in per_update if lo<=r['update']<=hi]
        counts=collections.Counter(w['task'] for u in range(lo,hi+1) for w in batches[u])
        segments.append(dict(start=lo,end=hi,metrics={k:summary([r[k] for r in rs]) for k in metrics},
                             task_counts=dict(counts),task_fractions={k:v/sum(counts.values()) for k,v in counts.items()}))
    fixed=[]
    for p in sorted([*SOURCE.glob('eval_*.json'),*(SOURCE/'train_monitor').glob('eval_*.json')]):
        rs=[r for r in read(p)['rows'] if r['k']==2]
        fixed.append(dict(file=str(p),update=rs[0]['update'],split=rs[0]['split'],direction=rs[0]['direction'],
                          samples=len(rs),mse=statistics.mean(r['prediction_mse'] for r in rs),
                          per_task={t:statistics.mean(r['prediction_mse'] for r in rs if r['task']==t)
                                    for t in sorted({r['task'] for r in rs})}))
    with zipfile.ZipFile(SOURCE/'source_initial.zip') as archive:
        paths=['energywam/training/shared_microbatch.py','energywam/training/trainer.py',
               'energywam/training/offload.py','energywam/training/run_state.py','energywam/training/stages.py',
               'energywam/energy/solver.py','energywam/energy/iteration.py','energywam/models/energy_core.py',
               'energywam/data/energy_dataset.py','energywam/data/episode_dataset.py','energywam/backbones/dino_v3.py']
        source_checks={n:{'archive_sha256':hashlib.sha256(archive.read(n)).hexdigest(),
                          'current_sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest(),
                          'equal':archive.read(n)==(ROOT/n).read_bytes()} for n in paths}
    lrs={k:sorted(set(r[k] for r in records[15:])) for k in records[0] if k.startswith('optimizer_lr/')}
    out=dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
             scope='CPU-only historical audit; no new Qwen evaluation or training. Reconstructed data order; per-sample actual training predictions were not logged.',
             numbering='Completed optimizer updates1..704; raw global_step0..703; W&B train/update uses +1.',
             source=str(SOURCE),record_count=len(records),loss_identity_max_error=max(abs(r['loss']-r['bi/action_loss']-9*r['bi/future_loss']) for r in records),
             source_checks=source_checks,learning_rates_updates16_to704=lrs,
             source_files_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [SOURCE/'timing_history.jsonl',SOURCE/'train_log.jsonl',SOURCE/'sample_manifest.json',SOURCE/'resolved_config.json',ROOT/cfg.data.norm_stats_bundle_path]},
             data=dict(window_visits=len(order),unique_windows=len(checks),unique_episodes=len(by_episode),
                       metadata_anomalies=anomalies,all_windows_valid=all(all(c[k] for k in
                       ['source_task_matches','action_finite','state_finite','full_horizon','future_in_bounds']) for c in checks.values()),
                       action_z_absmax=max(c['action_z_absmax'] for c in checks.values()),
                       task_pool_counts=dict(collections.Counter(w['task'] for w in windows))),
             per_update=per_update,segments=segments,fixed_evaluations=fixed,
             resource_summary=read(SOURCE/'resource_summary.json'))
    save('raw_episode_checks.json',episode_checks)
    save('raw_window_checks.json',list(checks.values()))
    save('reconstructed_batch_windows.json',batches)
    save('analysis.json',out)
    with (OUT/'per_update.csv').open('w') as f:
        fields=['update',*metrics,'implied_global_clip_multiplier','task_counts']
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(per_update)
    print('All704 updates/raw data analysis saved',flush=True)
    if args.dino:
        from energywam.backbones.dino_v3 import FrozenDINOv3
        from energywam.data import episode_dataset as episode_module
        dino=FrozenDINOv3(cfg.backbone.dino_path)
        assert next(dino.parameters()).device.type=='cpu'
        fs=stats['future'];mu=torch.tensor(fs['mean']);sd=torch.tensor(fs['std'])
        selected=sorted({657,max(per_update[:600],key=lambda x:x['action_mse'])['update'],
                         *[x['update'] for x in sorted(per_update,key=lambda x:x['future_mse'],reverse=True)[:2]]})
        details=[]
        for u in selected:
            for w in batches[u]:
                ep=w['episode_index'];t=w['frame_index'];indices=[t,t+5,t+11,t+16]
                frames=[reader.read_frame(ep,cfg.data.camera_order[0],i) for i in indices]
                features=dino.encode_images(frames,device='cpu');z=(features-mu)/(sd+fs['epsilon'])
                details.append(dict(update=u,episode_index=ep,frame_index=t,task=w['task'],
                                    images_shape=[list(x.shape) for x in frames],image_stds=[float(np.std(x)) for x in frames],
                                    features_shape=list(z.shape),features_finite=bool(torch.isfinite(z).all()),
                                    normalized_feature_absmax=float(z.abs().max()),
                                    current_copy_future_mse=float((z[1:]-z[0]).square().mean()),
                                    future_zero_mse=float(z[1:].square().mean()),
                                    source_video=str(reader.video_path(ep,cfg.data.camera_order[0]).resolve())))
                episode_module._VIDEO_CACHE.clear()
            print(f'CPU DINO baseline checked update{u}:16windows',flush=True)
            save('cpu_dino_baselines.json',dict(scope='New CPU recomputation using existing frozen DINO/current files; not original stored target tensors or Qwen predictions. No GPU use.',
                selected_updates=selected,details=details,summary={str(v):dict(
                    model_future_mse=per_update[v-1]['future_mse'],
                    current_copy_future_mse=statistics.mean(x['current_copy_future_mse'] for x in details if x['update']==v),
                    feature_absmax=max(x['normalized_feature_absmax'] for x in details if x['update']==v))
                    for v in selected if any(x['update']==v for x in details)}))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(4,1,figsize=(12,12),sharex=True)
    x=[r['update'] for r in per_update]
    for k,label in [('action_mse','Action MSE'),('future_mse','Future MSE'),('loss','A + 9F')]:
        axs[0].plot(x,[r[k] for r in per_update],label=label,linewidth=.8)
    axs[0].set_yscale('log');axs[0].legend(ncol=3);axs[0].set_ylabel('Raw batch MSE/loss')
    axs[1].plot(x,[r['param_grad_norm'] for r in per_update],linewidth=.8,label='Parameter gradient norm (before clip)')
    axs[1].axhline(1,color='red',linestyle='--',label='Outer clip threshold1');axs[1].set_yscale('log');axs[1].legend();axs[1].set_ylabel('Gradient norm')
    axs[2].plot(x,[r['action_to_zero_baseline_ratio'] for r in per_update],linewidth=.8,label='Action MSE / same-batch zero-action MSE')
    axs[2].axhline(1,color='red',linestyle='--');axs[2].set_yscale('log');axs[2].legend();axs[2].set_ylabel('Ratio')
    for k,label in [('diag/alpha_action','Action step / initial5'),('diag/alpha_future','Future step / initial2000')]:
        axs[3].plot(x,[r[k]/(5 if 'action' in k else 2000) for r in per_update],label=label)
    axs[3].legend();axs[3].set_ylabel('Learned step ratio');axs[3].set_xlabel('Completed optimizer update (1-based)')
    for ax in axs:
        ax.axvspan(501,600,color='orange',alpha=.1);ax.axvspan(685,704,color='red',alpha=.1);ax.grid(alpha=.2)
    fig.suptitle('Three-task seed42: raw records, no smoothing; log axes preserve spikes')
    fig.tight_layout();fig.savefig(OUT/'instability_timeline.png',dpi=170);fig.savefig(OUT/'instability_timeline.pdf');plt.close(fig)
    print('Audit complete:',OUT,flush=True)

if __name__=='__main__':main()
