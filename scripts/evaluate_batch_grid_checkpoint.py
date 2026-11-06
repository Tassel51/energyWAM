"""Read-only conditional evaluation, including VAL-only confirmation manifests.

No training optimizer is allocated. Use only after the checkpoint and protocol
have been selected; confirmation data must not become an LR-search feedback set.
"""
import argparse, hashlib, json, math, os, sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def validate_val_manifest(manifest,allowed,forbidden):
    rows=manifest['windows']
    if not rows or any(w['split']!='val' for w in rows):raise ValueError('Read-only confirmation requires VAL-only windows')
    identities=[(w['episode_index'],w['frame_index']) for w in rows]
    if len(set(identities))!=len(rows) or len({w['window_id'] for w in rows})!=len(rows):raise ValueError('duplicate validation window')
    episodes={w['episode_index'] for w in rows}
    if not episodes<=allowed or not episodes.isdisjoint(forbidden):raise ValueError('noncanonical validation episode or training leakage')

def main():
    p=argparse.ArgumentParser()
    for name in ['config','checkpoint','manifest','output']:p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--physical-gpus',required=True,choices=['0,1','2,3']);p.add_argument('--expected-step',required=True,type=int)
    p.add_argument('--evaluation-seed',type=int,default=2026);a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==a.physical_gpus
    manifest=json.loads(a.manifest.read_text())
    allowed=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices'])
    forbidden=set(json.loads((ROOT/'manifests/train_episodes.json').read_text())['episode_indices'])
    validate_val_manifest(manifest,allowed,forbidden)
    if a.output.resolve()==a.checkpoint.parent.resolve():raise ValueError('evaluation output must not overwrite training directory')
    if (a.output/'train_log.jsonl').exists():raise ValueError('output contains training records')
    a.output.mkdir(parents=True,exist_ok=True)
    source=a.checkpoint.stat();signature=(source.st_dev,source.st_ino,source.st_size,source.st_mtime_ns)
    protocol=dict(config=str(a.config.resolve()),config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),checkpoint=str(a.checkpoint.resolve()),checkpoint_identity=list(signature),expected_step=a.expected_step,manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),evaluation_seed=a.evaluation_seed,eval_k=2,weight_updates=0,optimizer_allocated=False)
    protocol_path=a.output/'evaluation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('output already belongs to a different checkpoint/protocol')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    os.environ.update(ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_PROBE_SPLITS='val',ENERGYWAM_PROBE_MAX_K='2')
    import torch
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import bundle_from_payload,restore
    from energywam.training.parallel import shard_language_model
    from k_training_probe import evaluate,dataset_baselines
    cfg=load_config(a.config)
    if cfg.solver.eval_inner_steps!=2:raise ValueError('this confirmation protocol fixes evaluation K=2')
    if cfg.backbone.dtype in ['float32','fp32']:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    payload=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    if payload['meta']['global_step']!=a.expected_step:raise ValueError('checkpoint does not have the selected budget')
    bundle=bundle_from_payload(payload)
    if bundle is None:raise ValueError('embedded TRAIN-only normalization bundle required')
    torch.manual_seed(cfg.trainer.seed);model=QwenEnergyWAM(cfg,stats_bundle=bundle)
    device=torch.device(cfg.trainer.device);model.to(device);shard_language_model(model,cfg.trainer.language_model_devices,device)
    meta=restore(payload,model=model,optimizer=None,scheduler=None,mode='resume',restore_rng=False);del payload
    if meta.stage!=cfg.stage.stage:raise ValueError('checkpoint stage mismatch')
    model.requires_grad_(False);model.eval()
    class NoOptimizer:
        def zero_grad(self,**kwargs):pass
    trainer=SimpleNamespace(cfg=cfg,model=model,optimizer=NoOptimizer(),global_step=meta.global_step)
    baselines=dataset_baselines(cfg,model,a.manifest,a.output)
    summary=dict(protocol=protocol,stats_fingerprint=bundle.fingerprint,baselines=baselines,metrics={},samples=len(manifest['windows']),scope='Conditional VAL-only Action/Future MSE; no weight updates, optimizer allocation, statistics fitting or robot rollout')
    for rank,direction in enumerate(['action','future']):
        path=a.output/f'eval_{meta.global_step:04d}_{direction}.json'
        if not path.exists():evaluate(trainer,a.manifest,a.output,rank,evaluation_seed=a.evaluation_seed)
        report=json.loads(path.read_text());rows=[x for x in report['rows'] if x['k']==2 and x['split']=='val']
        if len(rows)!=len(manifest['windows']) or report['evaluation_seed']!=a.evaluation_seed or report['weight_updates_during_eval']!=0:raise ValueError('evaluation result/protocol mismatch')
        if any(w['update']!=a.expected_step or w['direction']!=direction or not math.isfinite(w['prediction_mse']) for w in rows):raise ValueError('evaluation checkpoint/metric mismatch')
        expected=[(w['window_id'],w['episode_index'],w['frame_index']) for w in manifest['windows']]
        actual=[(w['window_id'],w['episode_index'],w['frame_index']) for w in rows]
        if actual!=expected:raise ValueError('evaluation windows do not match the frozen manifest')
        summary['metrics'][direction]=sum(x['prediction_mse'] for x in rows)/len(rows)
    after=a.checkpoint.stat()
    if signature!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('source checkpoint changed during evaluation')
    pending=a.output/'summary.pending.json';pending.write_text(json.dumps(summary,indent=2));pending.replace(a.output/'summary.json')
    print(json.dumps(summary['metrics']),flush=True)

if __name__=='__main__':main()
