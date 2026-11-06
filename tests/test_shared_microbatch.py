"""Exact objective and one update for logical batches 2 and 8."""
from pathlib import Path
import pytest
import torch


@pytest.mark.parametrize('batch_size',[2,4,8])
def test_shared_accumulation_matches_summed_conditional_objective(tmp_path,batch_size):
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.replicated import slice_batch
    cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
    cfg.trainer.output_dir=str(tmp_path);cfg.trainer.max_grad_norm=0
    cfg.trainer.per_device_batch_size=batch_size
    torch.manual_seed(77);reference=QwenEnergyWAM(cfg)
    torch.manual_seed(77);shared=QwenEnergyWAM(cfg)
    batch=make_tiny_batch(cfg,batch_size=batch_size,seed=10)
    terms=[]
    for role,direction in enumerate(['action','future']):
        torch.manual_seed(cfg.trainer.seed+role)
        for index in range(batch_size):
            micro=dict(slice_batch(batch,index,index+1),_bi_only=direction,_global_step=0)
            terms.append(reference(micro)['action_loss']/batch_size)
    loss=sum(terms);loss.backward()
    class CountSGD(torch.optim.SGD):
        count=0
        def step(self,*args,**kwargs):
            self.count+=1
            return super().step(*args,**kwargs)
    optimizer=CountSGD(shared.parameters(),lr=0.)
    trainer=SharedMicrobatchTrainer(cfg,shared,optimizer=optimizer)
    row=trainer.train_step(batch)
    assert optimizer.count==1 and trainer.global_step==1
    assert row['loss']==pytest.approx(float(loss.detach()),rel=1e-6)
    for (name,p),(other,q) in zip(reference.named_parameters(),shared.named_parameters()):
        assert name==other
        assert (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=1e-7,msg=name)


def test_sample_aligned_rng_makes_batch8_gradient_equal_four_batch2_means(tmp_path,monkeypatch):
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.replicated import slice_batch
    monkeypatch.setenv('ENERGYWAM_SAMPLE_ALIGNED_RNG','1')
    cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
    cfg.trainer.output_dir=str(tmp_path/'b2');cfg.trainer.max_grad_norm=0
    torch.manual_seed(77);model2=QwenEnergyWAM(cfg)
    t2=SharedMicrobatchTrainer(cfg,model2,optimizer=torch.optim.SGD(model2.parameters(),lr=0.))
    batch=make_tiny_batch(cfg,batch_size=8,seed=10)
    mean_grad={}
    for i in range(4):
        t2.train_step(slice_batch(batch,i*2,i*2+2))
        for n,p in model2.named_parameters():
            if p.grad is not None:mean_grad[n]=mean_grad.get(n,torch.zeros_like(p.grad))+p.grad/4
    cfg.trainer.output_dir=str(tmp_path/'b8')
    torch.manual_seed(77);model8=QwenEnergyWAM(cfg)
    t8=SharedMicrobatchTrainer(cfg,model8,optimizer=torch.optim.SGD(model8.parameters(),lr=0.))
    t8.train_step(batch)
    for n,p in model8.named_parameters():
        if p.grad is not None:torch.testing.assert_close(mean_grad[n],p.grad,rtol=2e-4,atol=1e-7,msg=n)


def test_fixed_budget_does_not_drop_windows_for_batch8():
    from energywam.training.run_state import FixedSampleBudgetSampler
    order=list(FixedSampleBudgetSampler(20,42,0,80))
    assert len(order)==80
    for epoch in range(4):assert sorted(order[epoch*20:(epoch+1)*20])==list(range(20))
    batches2=[order[i:i+2] for i in range(0,80,2)]
    batches8=[order[i:i+8] for i in range(0,80,8)]
    assert sum(batches2,[])==sum(batches8,[])
    assert list(FixedSampleBudgetSampler(20,42,16,80))==order[16:]


def test_directional_head_diagnostics_do_not_change_optimizer_update(tmp_path):
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    snapshots=[];rows=[]
    for enabled in [False,True]:
        cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
        cfg.trainer.output_dir=str(tmp_path/str(enabled));cfg.trainer.directional_head_diagnostics=enabled
        torch.manual_seed(921);model=QwenEnergyWAM(cfg)
        trainer=SharedMicrobatchTrainer(cfg,model,optimizer=torch.optim.SGD(model.parameters(),lr=1e-3))
        rows.append(trainer.train_step(make_tiny_batch(cfg,batch_size=2,seed=12)))
        snapshots.append({k:p.detach().clone() for k,p in model.named_parameters()})
    for name in snapshots[0]:torch.testing.assert_close(snapshots[0][name],snapshots[1][name],rtol=0,atol=0)
    assert rows[0]['loss']==rows[1]['loss']
    assert rows[1]['diag/head_action_gradient_norm']>0
    assert rows[1]['diag/head_weighted_future_gradient_norm']>0
    assert abs(rows[1]['diag/head_direction_cosine'])<=1.00001


def test_fixed_candidates_repeat_across_visits_and_match_explicit_eval_generator(tmp_path):
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.data.tiny_data import make_tiny_batch
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    from energywam.training.candidate_seeds import fixed_candidate_seed
    cfg=load_config(Path(__file__).parents[1]/'configs/tiny_cpu.yaml')
    cfg.trainer.fixed_candidate_initialization=True;cfg.trainer.max_grad_norm=0;cfg.trainer.output_dir=str(tmp_path)
    torch.manual_seed(932);model=QwenEnergyWAM(cfg)
    trainer=SharedMicrobatchTrainer(cfg,model,optimizer=torch.optim.SGD(model.parameters(),lr=0.))
    batch=make_tiny_batch(cfg,batch_size=2,seed=33);batch.update(episode_index=[7,8],frame_index=[20,40])
    first=trainer.train_step(batch);grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    second=trainer.train_step(batch)
    assert first['loss']==second['loss']
    for n,p in model.named_parameters():
        if p.grad is not None:torch.testing.assert_close(grads[n],p.grad,rtol=0,atol=0)
    prepared=model._prepare_batch(batch)
    seed=fixed_candidate_seed(cfg.trainer.seed,0,7,20)
    torch.manual_seed(seed);a,z=model.stage_runner._init(prepared,torch.device('cpu'))
    a1,z1=model.stage_runner._init(prepared,torch.device('cpu'),generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(a,a1,rtol=0,atol=0);torch.testing.assert_close(z,z1,rtol=0,atol=0)
    assert seed!=fixed_candidate_seed(cfg.trainer.seed,1,7,20)
