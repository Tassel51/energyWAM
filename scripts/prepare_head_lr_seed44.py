"""Prepare the explicitly authorized seed44 repeat on cloud storage only."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / 'logs/lab_remote_setup/head_lr_seed44_deployment.json'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    assert not RECEIPT.exists(), 'Already prepared; inspect receipt instead of duplicating'
    previous = Path('/cloud/zzc/energyWAM/experiments/single_episode_head_lr_storagefix_20261110T031718Z')
    original_plan = json.loads((previous / 'plan.json').read_text())
    assert json.loads((previous / 'controller_status.json').read_text())['phase'] == 'interrupted'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = previous.parent / ('single_episode_head_lr_seed44_' + stamp)
    root.mkdir()
    (root / 'configs').mkdir()
    code = Path(original_plan['code'])
    old_job = json.loads((previous / 'jobs.json').read_text())[0]
    original_config = yaml.safe_load(Path(old_job['config']).read_text())
    cfg = copy.deepcopy(original_config)
    name = 'episode3346_headscale1_b16_micro4_seed44'
    cfg['run_id'] = name
    cfg['trainer'].update(seed=44, output_dir=str(root / name))
    cfg['notes'] = 'Seed44 robustness check against seed42 at headLR1e-3; fresh official pretrained model; all other training hyperparameters unchanged; fixed evaluation seed2026 and alternate final2027 unchanged.'
    config = root / 'configs' / (name + '.yaml')
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    check = copy.deepcopy(cfg)
    check['run_id'] = original_config['run_id']
    check['notes'] = original_config['notes']
    check['trainer']['seed'] = original_config['trainer']['seed']
    check['trainer']['output_dir'] = original_config['trainer']['output_dir']
    assert check == original_config
    for file in ['train_val_manifest.json', 'train_monitor_manifest.json', 'environment_manifest.json', 'source_manifest.json']:
        shutil.copy2(previous / file, root / file)
    job = copy.deepcopy(old_job)
    job.update(name=name, config=str(config), manifest=str(root / 'train_val_manifest.json'),
               train_monitor_manifest=str(root / 'train_monitor_manifest.json'),
               runtime_environment_manifest=str(root / 'environment_manifest.json'))
    write(root / 'jobs.json', [job])
    budget = 3 * job['snapshot_bytes_estimate'] + 8 * 2**30
    plan = copy.deepcopy(original_plan)
    for key in ['previous_failed_run', 'source_patch']:
        plan.pop(key, None)
    plan.update(name='single-episode-head-lr-seed44-' + stamp, root=str(root),
                jobs=[dict(name=name, folder=str(root / name), config=str(config), head_lr=.001, head_scale=1.)],
                authorization='User explicitly requested a different seed at headLR x1, requested seed44 fresh run after seed43 diagnostics, and authorized prior checkpoint retirement; cloud-only storage',
                protocol='Fresh seed44; same67TRAIN/16VAL/B16/native4/FP32/K2/210updates/LRs/A+9F/clip1/warmup16; fixed eval seed2026, final alternate2027',
                rerun_mode='Fresh pretrained Qwen/DINO; seed44 initializes task modules, sampler and training RNG; no robot-checkpoint resume',
                comparison='Cross-seed robustness at headLR1e-3; compare fixed TRAIN/VAL separately at matched updates; not a cross-seed LR ranking study',
                tmux_session='energywam_head_lr_seed44_' + stamp,
                required_free_bytes=budget,
                storage_gate_result=str(root / 'metadata_storage_gate/result.json'),
                reference_seed42_study=str(previous), seed=44)
    write(root / 'plan.json', plan)
    folder = root / name
    folder.mkdir()
    write(folder / 'arm_definition.json', dict(hyperparameters=cfg, job=job, seed=44,
          target_updates=210, sample_visits=3360, equivalent_epochs=3360/67,
          initialization=plan['rerun_mode'], fixed_evaluation_seed=2026, alternate_final_evaluation_seed=2027,
          best_criterion='Auxiliary16VAL ownK2 Action MSE +9 Future MSE',
          source_code=str(code), all_non_identity_non_seed_config_fields_equal=True))
    write(root / 'preflight_validation.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
          cloud_only=True, free_bytes=shutil.disk_usage(root).free, required_bytes=budget,
          config_comparison_passed=True, seed_change=[42,44],
          manifest_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in ['train_val_manifest.json','train_monitor_manifest.json']},
          git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          frozen_source_reused_without_modification=str(code)))
    write(RECEIPT, dict(root=str(root), code=str(code), plan=str(root/'plan.json'),
                       python=plan['python'], tmux=plan['tmux_session'], name=name,
                       prepared_at=datetime.now(timezone.utc).isoformat()))
    print(RECEIPT.read_text())


if __name__ == '__main__':
    main()
