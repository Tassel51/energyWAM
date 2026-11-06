"""Numerical checks for batched stats and cross-GPU second-order training."""
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pytest
import torch


def test_batched_statistics_preserve_masked_window_distribution():
    from energywam.config import load_config
    from energywam.data.stats_fit import fit_bundle_from_reader

    cfg = load_config(Path(__file__).parents[1] / 'configs/tiny_cpu.yaml')

    class Reader:
        root = Path('/synthetic')
        dataset_identity = 'synthetic/robot'
        camera_keys = ['left']
        def episode_length(self, ep):
            return 23
        def read_actions(self, ep):
            return np.arange(23 * 3).reshape(23, 3) + 100 * ep
        def read_frame(self, ep, cam, frame):
            return np.full((2, 2, 3), frame + ep, dtype=np.uint8)

    class Dino:
        hidden_dim = 8
        num_patches = 4
        def encode_images(self, images, device='cpu'):
            return torch.stack([torch.full((4, 8), float(im[0, 0, 0])) + torch.arange(8) for im in images])

    for full_horizon in [True, False]:
        cfg.data.require_full_horizon = full_horizon
        for limit in [1, 19, None]:
            kwargs = dict(episodes=[0, 1], dino=Dino(), max_samples=limit)
            ref = fit_bundle_from_reader(Reader(), cfg, **kwargs)
            batched = fit_bundle_from_reader(Reader(), cfg, dino_batch_size=7, **kwargs)
            for kind in ['action', 'future']:
                for field in ['mean', 'std', 'count']:
                    np.testing.assert_allclose(getattr(getattr(ref, kind), field), getattr(getattr(batched, kind), field), rtol=0, atol=0)
            assert {k: v for k, v in ref.provenance.items() if k != 'fitted_at'} == {k: v for k, v in batched.provenance.items() if k != 'fitted_at'}


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two visible GPUs')
def test_sharded_qwen_full_unroll_matches_single_gpu_gradients(tmp_path):
    from tests.real_fixtures import build_real_model, make_real_batch
    from energywam.training.parallel import shard_language_model
    from energywam.training.trainer import EnergyWAMTrainer
    model, cfg, geometry = build_real_model(with_real_dino=False, B=1, seed=88)
    batch = make_real_batch(model, cfg, geometry, B=1)
    model.to('cuda:0').train()

    def backward():
        model.zero_grad(set_to_none=True)
        torch.manual_seed(99)
        out = model(batch)
        out['action_loss'].backward()
        return out['action_loss'].detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}

    ref_loss, ref = backward()
    model.zero_grad(set_to_none=True)
    shard_language_model(model, [0, 1], torch.device('cuda:0'))
    loss, grads = backward()
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref.keys()
    assert any(p.device.index == 1 and p.grad is not None for p in model.parameters())
    for n in grads:
        torch.testing.assert_close(grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    cfg.trainer.offload_saved_tensors = True
    offloaded_loss, offloaded_grads = backward()
    torch.testing.assert_close(offloaded_loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert offloaded_grads.keys() == ref.keys()
    for n in offloaded_grads:
        torch.testing.assert_close(offloaded_grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    cfg.trainer.offload_saved_tensors = False
    model.backbone._mm.language_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    checkpoint_loss, checkpoint_grads = backward()
    torch.testing.assert_close(checkpoint_loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert checkpoint_grads.keys() == ref.keys()
    for n in checkpoint_grads:
        torch.testing.assert_close(checkpoint_grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    cfg.trainer.offload_saved_tensors = True
    cfg.trainer.saved_tensor_gpu_budget_gib = 0.001
    cfg.trainer.saved_tensor_gpu_budgets_gib = [0.001, 0.002]
    mixed_loss, mixed_grads = backward()
    torch.testing.assert_close(mixed_loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert mixed_grads.keys() == ref.keys()
    for n in mixed_grads:
        torch.testing.assert_close(mixed_grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    cfg.trainer.saved_tensor_cpu_pin_budget_gib = 0.02
    cfg.trainer.saved_tensor_async_offload = True
    cfg.trainer.saved_tensor_host_cache_limit_gib = 0.001
    async_loss, async_grads = backward()
    torch.testing.assert_close(async_loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert async_grads.keys() == ref.keys()
    for n in async_grads:
        torch.testing.assert_close(async_grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    # Check optimizer state placement, checkpoint restoration and a resumed step.
    cfg.trainer.device = 'cuda:0'
    cfg.trainer.output_dir = str(tmp_path)
    # Trainer must place/shard the fresh model itself.
    fresh, fresh_cfg, _ = build_real_model(with_real_dino=False, B=1, seed=88)
    fresh_cfg.trainer.device = 'cuda:0'
    fresh_cfg.trainer.language_model_devices = [0, 1]
    fresh_cfg.trainer.output_dir = str(tmp_path)
    fresh_cfg.trainer.learning_rate = 0.0
    fresh_cfg.trainer.max_grad_norm = 0.0
    fresh_cfg.trainer.optimizer = 'adamw8bit'
    fresh_cfg.trainer.preallocate_optimizer_state = True
    trainer = EnergyWAMTrainer(fresh_cfg, fresh)
    torch.manual_seed(99)
    trainer.train_step(batch)
    # Streaming forward/backward must equal the summed-loss gradient, with
    # identical candidate RNG and no intervening parameter update.
    streamed_grads = {n: p.grad.detach().cpu() for n, p in fresh.named_parameters() if p.grad is not None}
    assert streamed_grads.keys() == ref.keys()
    for n in streamed_grads:
        torch.testing.assert_close(streamed_grads[n], ref[n], rtol=1e-3, atol=1e-6, msg=n)
    saved = trainer.save('parallel.pt')
    trainer.load(saved)
    trainer.train_step(batch)
    trainer.switch_stage('joint', saved, optimizer_policy='reset', scheduler_policy='reset')
    assert trainer.optimizer.state
    assert all(state['step'] == 0 for state in trainer.optimizer.state.values())
    trainer.train_step(batch)
