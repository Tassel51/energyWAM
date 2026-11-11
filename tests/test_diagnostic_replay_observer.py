import copy
import json
from pathlib import Path
import torch


def test_observer_preserves_updates_and_records_windows(tmp_path,monkeypatch):
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from scripts.diagnostic_replay_observer import ReplayObserver
    monkeypatch.setenv('ENERGYWAM_SAMPLE_ALIGNED_RNG','1')
    cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
    cfg.trainer.shared_microbatch_size=2;cfg.trainer.per_device_batch_size=4
    cfg.trainer.output_dir=str(tmp_path/'plain')
    torch.manual_seed(42);plain=QwenEnergyWAM(cfg)
    cfg2=copy.deepcopy(cfg);cfg2.trainer.output_dir=str(tmp_path/'observed')
    torch.manual_seed(42);observed=QwenEnergyWAM(cfg2)
    a=SharedMicrobatchTrainer(cfg,plain);b=SharedMicrobatchTrainer(cfg2,observed)
    batch=make_tiny_batch(cfg,batch_size=4,seed=123)
    expected=a.train_step(batch)
    observer=ReplayObserver(b,tmp_path/'trace').install()
    try:actual=b.train_step(batch)
    finally:observer.remove()
    assert expected['loss']==actual['loss']
    assert expected['param_grad_norm']==actual['param_grad_norm']
    for name,value in plain.state_dict().items():assert torch.equal(value,observed.state_dict()[name]),name
    rows=[json.loads(x) for x in (tmp_path/'trace/window_inner_trace.jsonl').read_text().splitlines()]
    assert len(rows)==4*2*2
    clipping=json.loads((tmp_path/'trace/gradient_clipping.jsonl').read_text())
    assert clipping['after_total']<=1.00001
    assert (tmp_path/'trace/parameter_updates.jsonl').exists()
