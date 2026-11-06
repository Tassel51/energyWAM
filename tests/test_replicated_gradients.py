"""Numerical BI sum / joint mean decomposition across real Gloo processes."""
from pathlib import Path
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from energywam.config import load_config
from energywam.models.energy_wam import QwenEnergyWAM
from energywam.data.tiny_data import make_tiny_batch
from energywam.training.replicated import ReplicatedTrainer, slice_batch


def make_case(folder, stage):
    cfg = load_config(Path(__file__).parents[1] / 'configs/tiny_cpu.yaml')
    cfg.stage.stage = stage
    cfg.trainer.learning_rate = 0.0
    cfg.trainer.warmup_steps = 0
    cfg.trainer.max_grad_norm = 0.0
    cfg.trainer.output_dir = str(folder)
    torch.manual_seed(17)
    model = QwenEnergyWAM(cfg)
    raw_forward = model.forward
    def seeded_forward(batch):
        torch.manual_seed(int(batch['_candidate_seed'][0]))
        return raw_forward(batch)
    model.forward = seeded_forward
    batch = make_tiny_batch(cfg, batch_size=2, seed=8)
    batch['_candidate_seed'] = [1001, 1002]
    batch['action_mask'] = torch.ones_like(batch['action_mask'])
    batch['future_mask'] = torch.ones_like(batch['future_mask'])
    return cfg, model, batch


def make_real_case(folder, stage):
    from tests.real_fixtures import build_real_model, make_real_batch
    model, cfg, geometry = build_real_model(with_real_dino=False, B=2, seed=17, **{'stage.stage':stage})
    cfg.trainer.learning_rate = 0.0
    cfg.trainer.warmup_steps = 0
    cfg.trainer.max_grad_norm = 0.0
    cfg.trainer.output_dir = str(folder)
    batch = make_real_batch(model,cfg,geometry,B=2,seed=8)
    batch['_candidate_seed'] = [1001,1002]
    def fixed_init(prepared, device, **kwargs):
        actions, futures = [], []
        for seed in prepared['_raw']['_candidate_seed']:
            gen = torch.Generator().manual_seed(seed)
            actions.append(torch.randn(geometry['action_horizon'],geometry['action_dim'],generator=gen))
            futures.append(torch.randn(geometry['num_future_frames'],geometry['num_cameras'],geometry['future_patches'],geometry['future_dim'],generator=gen))
        return torch.stack(actions).to(device), torch.stack(futures).to(device)
    model.stage_runner._init = fixed_init
    return cfg,model,batch


def worker(rank, stage, folder, rendezvous, real):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    cfg, model, batch = (make_real_case if real else make_case)(Path(folder)/str(rank), stage)
    trainer = ReplicatedTrainer(cfg, model)
    trainer.train_step(batch)
    torch.save({n:p.grad for n,p in model.named_parameters() if p.grad is not None}, Path(folder)/f'grad{rank}.pt')
    from energywam.training.checkpoint import _rng_state, _set_rng_state, load_checkpoint
    torch.manual_seed(20000 + rank)
    states = [None, None]
    dist.all_gather_object(states, _rng_state())
    if rank == 0:
        model.distributed_rng_states = states
        trainer.save('distributed.pt')
    dist.barrier()
    saved = Path(folder)/'0'/'distributed.pt'
    trainer.load(saved)
    payload = load_checkpoint(saved)
    _set_rng_state(payload['distributed_rng_states'][rank])
    assert torch.equal(torch.get_rng_state(), states[rank]['torch'])
    assert trainer.global_step == 1
    dist.destroy_process_group()


def run_case(tmp_path, stage, real=False):
    cfg, model, batch = (make_real_case if real else make_case)(tmp_path/'reference', stage)
    model.zero_grad(set_to_none=True)
    if real:
        model(batch)['action_loss'].backward()
    else:
        for direction in (('action','future') if stage == 'bi' else (None,)):
            for index in range(2):
                micro = slice_batch(batch, index, index+1)
                if direction:
                    micro['_bi_only'] = direction
                (model(micro)['action_loss']/2).backward()
    reference = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
    mp.spawn(worker, args=(stage,str(tmp_path),str(tmp_path/'rendezvous'),real), nprocs=2, join=True)
    for rank in range(2):
        actual = torch.load(tmp_path/f'grad{rank}.pt', weights_only=True)
        assert actual.keys() == reference.keys()
        for name in reference:
            torch.testing.assert_close(actual[name], reference[name], rtol=1e-4, atol=1e-6, msg=name)


def test_replicated_bidirectional_gradient_sum(tmp_path):
    run_case(tmp_path, 'bi')


def test_replicated_joint_gradient_mean(tmp_path):
    run_case(tmp_path, 'joint')


def test_replicated_qwen_bi_matches_padded_batch_gradient(tmp_path):
    run_case(tmp_path, 'bi', real=True)


def test_replicated_qwen_joint_matches_padded_batch_gradient(tmp_path):
    run_case(tmp_path, 'joint', real=True)
