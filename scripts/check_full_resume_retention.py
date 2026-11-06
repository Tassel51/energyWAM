"""Bounded real-model entrypoint smoke: train, full resume, latest and best."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_iteration_search import gpu_idle
from energywam.training.resource_monitor import ResourceMonitor

def main():
    root=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')
    gate=json.loads((root/'gradient_gates/fp32_dedup10/status.json').read_text())
    assert gate['gate_passed'], 'storage/math correctness gate required'
    for _ in range(160):
        if gpu_idle('2,3'):break
        time.sleep(15)
    else:raise RuntimeError('GPU pair stayed occupied; no existing process interrupted')
    folder=root/'fp32_full_resume_retention_smoke'
    folder.mkdir(exist_ok=True)
    if (folder/'resume_smoke_result.json').exists():return
    # Each phase uses the real training/evaluation/checkpoint entrypoint.
    for target in [1,2]:
        if (folder/'speed_report.json').exists() and json.loads((folder/'speed_report.json').read_text())['completed_steps']>=target:continue
        cmd=[sys.executable,'-u','scripts/train_shared_sharded.py','--config','configs/experiments/fp32_full_resume_retention_smoke.yaml','--steps',str(target),'--physical-gpus','2,3','--fixed-window-manifest',str(root/'resume_smoke_manifest.json'),'--keep-best','--sample-eval-interval','4','--train-monitor-manifest',str(root/'resume_smoke_monitor.json'),'--train-monitor-sample-interval','4']
        checkpoint=folder/'ckpt_latest.pt'
        if checkpoint.exists():cmd+=['--resume',str(checkpoint)]
        elif (folder/'train_log.jsonl').exists():raise RuntimeError('logs without checkpoint: refuse duplicate updates')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='2,3',ENERGYWAM_FIXED_DATA_CACHE='1',ENERGYWAM_SAMPLE_ALIGNED_RNG='1',ENERGYWAM_TRIM_BETWEEN_DIRECTIONS='0',ENERGYWAM_PROBE_MAX_K='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
        with (folder/f'phase{target}.console.log').open('a') as stream:
            child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            monitor=ResourceMonitor(child.pid,['2','3'],folder/f'resources_phase{target}')
            monitor.folder.mkdir(parents=True,exist_ok=True)
            console_alias=monitor.folder/'console.log'
            if not console_alias.exists():console_alias.symlink_to(folder/f'phase{target}.console.log')
            monitor.start()
            try:child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=55)
                monitor.finish(child.returncode,'30 minute smoke phase budget');raise
            monitor.finish(child.returncode)
            if child.returncode:raise RuntimeError(f'resume smoke phase {target} failed: inspect console/resources')
    import torch
    checked={}
    for name in ['latest','best']:
        path=folder/f'ckpt_{name}.pt'
        payload=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert payload['optimizer'] is not None and payload['scheduler'] is not None and payload['rng']
        assert payload['model'] and payload['config']
        checked[name]=dict(step=payload['meta']['global_step'],bytes=path.stat().st_size,optimizer_keys=list(payload['optimizer']),scheduler_last_epoch=payload['scheduler']['last_epoch'])
        del payload
    assert checked['latest']['step']==2
    init=json.loads((folder/'initialization_step1.json').read_text())
    assert init['global_step']==1
    result=dict(passed=True,checked=checked,real_entrypoint_resume_from_step=1,completed_step=2,scope='Two real updates and full resume with fixed conditional validation; no batch-quality conclusion')
    (folder/'resume_smoke_result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
