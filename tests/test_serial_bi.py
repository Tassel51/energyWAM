"""Verify serialized directions preserve full-unroll gradients and updates."""
from pathlib import Path


def _worker(rank, address, folder):
    import os
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.replicated import ReplicatedTrainer
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=address,timeout=timedelta(seconds=120))
    results=[]
    for serial in ['0','1']:
        os.environ['ENERGYWAM_SERIAL_BI']=serial
        cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
        cfg.solver.train_inner_steps=4
        cfg.trainer.per_device_batch_size=2
        cfg.trainer.output_dir=str(Path(folder)/f'rank{rank}_serial{serial}')
        cfg.trainer.max_grad_norm=0
        cfg.trainer.optimizer='adamw'
        cfg.trainer.preallocate_optimizer_state=False
        torch.manual_seed(88)
        model=QwenEnergyWAM(cfg)
        trainer=ReplicatedTrainer(cfg,model)
        batch=make_tiny_batch(cfg,batch_size=2,seed=10)
        history=[]
        for step in range(2):
            torch.manual_seed(100+rank+step*100003)
            logs=trainer.train_step(batch)
            history.append({'loss':logs['loss'],
                            'grad':{n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None},
                            'state':{n:p.detach().clone() for n,p in model.named_parameters()}})
        results.append(history)
    for reference,serial in zip(*results):
        assert reference['loss']==serial['loss']
        assert reference['grad'].keys()==serial['grad'].keys()
        for name in reference['grad']:
            torch.testing.assert_close(reference['grad'][name],serial['grad'][name],rtol=0,atol=0)
        for name in reference['state']:
            torch.testing.assert_close(reference['state'][name],serial['state'][name],rtol=0,atol=0)
    dist.destroy_process_group()


def test_serial_bi_matches_parallel_gradients_and_updates(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_worker,args=('file://'+str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
