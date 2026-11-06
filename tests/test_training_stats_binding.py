"""Training entrypoint -> checkpoint -> serving statistics and resume coverage."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from energywam.config import load_config
from energywam.data.normalization import NormStats
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.training.trainer import EnergyWAMTrainer
from energywam.training.checkpoint import load_checkpoint
from scripts.train import build_dataloader
from scripts.serve import resolve_serve_action_stats
from .harness import assert_true, assert_close

ROOT = Path(__file__).resolve().parents[1]

def test_training_stats_saved_and_usable_by_service():
    cfg = load_config(ROOT / 'configs/tiny_cpu.yaml')
    model = QwenEnergyWAM(cfg)
    stats = NormStats(mode='min_max',min=np.array([2,4,6],dtype='float32'),
                      max=np.array([6,8,10],dtype='float32'),version='binding-test')
    loader = SimpleNamespace(dataset=SimpleNamespace(action_stats=stats,
                             reader=SimpleNamespace(dataset_identity='test-data')))
    cfg.backbone.backend='qwen3vl'
    cfg.data.data_mix='training-statistics'
    with patch('energywam.data.energy_dataset.build_dataloader',return_value=loader):
        assert_true(build_dataloader(cfg,model) is loader)
    cfg.backbone.backend='tiny'
    with tempfile.TemporaryDirectory() as folder:
        cfg.trainer.output_dir=folder
        trainer=EnergyWAMTrainer(cfg,model)
        path=trainer.save('trained.pt')
        payload=load_checkpoint(path)
        by_key,key,restored=resolve_serve_action_stats(cfg,payload=payload)
        assert_true(key=='training-statistics',str(key))
        assert_close(restored.min,stats.min,0)
        assert_true(payload['meta']['norm_stats_version']=='binding-test')
        fresh=QwenEnergyWAM(cfg)
        resumed=EnergyWAMTrainer(cfg,fresh)
        resumed.load(path)
        assert_close(fresh.norm_stats[key].max,stats.max,0)
        assert_true(fresh.norm_stats_version=='binding-test')
        again=load_checkpoint(resumed.save('resaved.pt'))
        assert_true(again['meta']['norm_stats']==payload['meta']['norm_stats'])

TESTS=[('training_stats_saved_and_usable_by_service',test_training_stats_saved_and_usable_by_service)]
