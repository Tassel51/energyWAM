import json
from pathlib import Path

import yaml

from scripts import run_head_lr_study as study


def test_head_lr_configs_keep_all_other_training_hyperparameters():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root/'configs/experiments/tiny16_futureweight9_b16_micro4_seed42.yaml').read_text())
    for tag, rate in study.RATES:
        if rate == 1e-3:
            continue
        cfg = yaml.safe_load((root/f'configs/experiments/head_{tag}_tiny16_b16_micro4_seed42.yaml').read_text())
        assert cfg['trainer']['energy_head_learning_rate'] == rate
        expected = json.loads(json.dumps(baseline))
        for c in [cfg, expected]:
            c.pop('run_id')
            c.pop('notes')
            c['trainer'].pop('output_dir')
            c['trainer'].pop('energy_head_learning_rate')
        assert cfg == expected


def test_report_does_not_pick_flat_or_tradeoff_as_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(study, 'OUT', tmp_path)
    monkeypatch.setattr(study, 'ROOT', tmp_path)
    jobs = []
    for name in ['flat', 'future_regression']:
        folder = tmp_path/name
        folder.mkdir()
        config = tmp_path/f'{name}.yaml'
        config.write_text(yaml.safe_dump({'trainer':{'output_dir':str(folder),'energy_head_learning_rate':.001}}))
        (folder/'speed_report.json').write_text(json.dumps({'completed_steps':200}))
        for update in [0,200]:
            rows=[]
            for split in ['train','val']:
                for direction in ['action','future']:
                    mse = 1.0
                    if update and name=='future_regression':
                        mse = .5 if direction=='action' else 1.5
                    rows.append(dict(update=update,split=split,direction=direction,k=2,prediction_mse=mse))
            (folder/f'eval_{update:04d}.json').write_text(json.dumps({'rows':rows}))
        jobs.append(dict(name=name,config=config.name))
    report=study.summarize(jobs)
    assert report['outcome']=='no_eligible_winner'
    assert 'candidate_rate' not in report


def test_partial_results_never_claim_completed_comparison(tmp_path, monkeypatch):
    monkeypatch.setattr(study,'OUT',tmp_path)
    monkeypatch.setattr(study,'ROOT',tmp_path)
    config=tmp_path/'pending.yaml'
    config.write_text(yaml.safe_dump({'trainer':{'output_dir':str(tmp_path/'pending'),'energy_head_learning_rate':.0003}}))
    assert study.summarize([dict(name='pending',config=config.name)])['outcome']=='pending_matched200_review'
