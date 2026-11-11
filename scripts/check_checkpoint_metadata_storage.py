"""CPU-only real-filesystem save/resume gate; no formal experiment updates."""
import argparse
import copy
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.data.tiny_data import make_tiny_batch
from energywam.training.trainer import EnergyWAMTrainer
from energywam.training.run_state import save_latest
from energywam.training.best_checkpoint import promote_latest_to_best
from energywam.training.checkpoint_metadata import read_checkpoint_metadata


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--tiny-config',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 assert torch.cuda.device_count()==0,'Hide CUDA for this CPU-only gate'
 cfg=load_config(a.tiny_config);(a.output/'fixture_config.yaml').write_text(a.tiny_config.read_text());cfg.trainer.output_dir=str(a.output/'continuous')
 torch.manual_seed(42);model=QwenEnergyWAM(cfg);trainer=EnergyWAMTrainer(cfg,model)
 batch1=make_tiny_batch(cfg,seed=501);batch2=make_tiny_batch(cfg,seed=502)
 trainer.train_step(batch1);save_latest(trainer,trainer.log_dir)
 promote_latest_to_best(trainer.log_dir,global_step=1,score=1.,metrics={'engineering_fixture':True},criterion='CPU_storage_gate_fixture_not_experiment_selection')
 trainer.train_step(batch2);save_latest(trainer,trainer.log_dir)
 expected={k:v.detach().clone() for k,v in trainer.model.state_dict().items()}
 cfg2=copy.deepcopy(cfg);cfg2.trainer.output_dir=str(a.output/'resume')
 torch.manual_seed(99);model2=QwenEnergyWAM(cfg2);resumed=EnergyWAMTrainer(cfg2,model2)
 resumed.load(trainer.log_dir/'ckpt_best.pt',mode='resume');assert resumed.global_step==1
 resumed.train_step(batch2);save_latest(resumed,resumed.log_dir)
 assert all(torch.equal(expected[k],v) for k,v in resumed.model.state_dict().items())
 assert read_checkpoint_metadata(trainer.log_dir/'ckpt_latest.pt.meta.json')['meta']['global_step']==2
 assert read_checkpoint_metadata(trainer.log_dir/'ckpt_best.pt.meta.json')['meta']['global_step']==1
 assert read_checkpoint_metadata(resumed.log_dir/'ckpt_latest.pt.meta.json')['meta']['global_step']==2
 report=dict(passed=True,scope='Tiny CPU model, actual target filesystem, full save at1/resume to2, bitwise model match; not a real4B GPU trajectory or performance test',cuda_devices=0,continuous_latest=2,continuous_best=1,resumed_latest=2,model_bitwise_equal=True)
 (a.output/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
