import copy
import pytest
import torch
from energywam.training.master_weights import FP32MasterOptimizer


def test_sub_ulp_updates_accumulate_and_resume_exactly():
    direct=torch.nn.Parameter(torch.tensor([.5],dtype=torch.bfloat16))
    model=torch.nn.Parameter(direct.detach().clone())
    opt0=torch.optim.SGD([direct],lr=1e-4)
    opt=FP32MasterOptimizer(torch.optim.SGD([model],lr=1e-4,momentum=.9))
    for _ in range(20):
        direct.grad=torch.ones_like(direct);opt0.step()
        model.grad=torch.ones_like(model);opt.step()
    assert direct.item()==.5  # All twenty updates were rounded away.
    assert model.item()<.49
    state=copy.deepcopy(opt.state_dict())
    other=torch.nn.Parameter(torch.tensor([.5],dtype=torch.bfloat16))
    resumed=FP32MasterOptimizer(torch.optim.SGD([other],lr=1e-4,momentum=.9))
    resumed.load_state_dict(state)
    for _ in range(5):
        model.grad=torch.ones_like(model);other.grad=torch.ones_like(other)
        opt.step();resumed.step()
    torch.testing.assert_close(model,other,rtol=0,atol=0)
    torch.testing.assert_close(opt.pairs[0][1],resumed.pairs[0][1],rtol=0,atol=0)
    with pytest.raises(ValueError,match='master'):
        resumed.load_state_dict(opt0.state_dict())


def test_scheduler_groups_and_float32_parameters():
    low=torch.nn.Parameter(torch.tensor([.5],dtype=torch.bfloat16))
    full=torch.nn.Parameter(torch.tensor([.5],dtype=torch.float32))
    opt=FP32MasterOptimizer(torch.optim.AdamW([{'params':[low],'lr':1e-4},{'params':[full],'lr':1e-3}],weight_decay=0))
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda n: .5**n)
    low.grad=torch.ones_like(low);full.grad=torch.ones_like(full)
    opt.step();scheduler.step()
    assert full.item()<.5
    assert opt.param_groups[1]['lr']==pytest.approx(5e-4)
    assert opt.base_optimizer.param_groups is opt.param_groups
    opt.zero_grad();assert low.grad is None and full.grad is None


def test_real_energy_trainer_master_checkpoint_next_update_is_identical(tmp_path):
    from .test_iteration_aware import fixture
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    model,cfg,batch=fixture()
    model.backbone._mm.language_model.to(torch.bfloat16)
    cfg.trainer.fp32_master_weights=True;cfg.trainer.optimizer='adamw'
    cfg.trainer.output_dir=str(tmp_path/'original');cfg.trainer.warmup_steps=0
    trainer=SharedMicrobatchTrainer(cfg,model)
    trainer.train_step(batch)
    path=trainer.save('latest.pt')
    other,cfg2,_=fixture();other.backbone._mm.language_model.to(torch.bfloat16)
    cfg2.trainer.fp32_master_weights=True;cfg2.trainer.optimizer='adamw'
    cfg2.trainer.output_dir=str(tmp_path/'restored');cfg2.trainer.warmup_steps=0
    resumed=SharedMicrobatchTrainer(cfg2,other);resumed.load(path)
    row1=trainer.train_step(batch);row2=resumed.train_step(batch)
    assert row1['loss']==row2['loss']
    for (name,p),(name2,q) in zip(model.named_parameters(),other.named_parameters()):
        assert name==name2;torch.testing.assert_close(p,q,rtol=0,atol=0,msg=name)
    for (_,p),(_,q) in zip(trainer.optimizer.pairs,resumed.optimizer.pairs):
        torch.testing.assert_close(p,q,rtol=0,atol=0)


def test_explicit_weight_load_and_optimizer_reset_syncs_master_copy(tmp_path):
    from .test_iteration_aware import fixture
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    model,cfg,batch=fixture();model.backbone._mm.language_model.to(torch.bfloat16)
    cfg.trainer.fp32_master_weights=True;cfg.trainer.optimizer='adamw';cfg.trainer.warmup_steps=0
    cfg.trainer.output_dir=str(tmp_path/'source')
    trainer=SharedMicrobatchTrainer(cfg,model);trainer.train_step(batch)
    path=trainer.save('latest.pt')
    other,cfg2,_=fixture();other.backbone._mm.language_model.to(torch.bfloat16)
    cfg2.trainer.fp32_master_weights=True;cfg2.trainer.optimizer='adamw';cfg2.trainer.warmup_steps=0
    cfg2.trainer.output_dir=str(tmp_path/'target')
    target=SharedMicrobatchTrainer(cfg2,other)
    target.load(path,mode='switch',optimizer_policy='reset',scheduler_policy='reset')
    assert not target.optimizer.state
    for parameter,master in target.optimizer.pairs:
        torch.testing.assert_close(master,parameter.float(),rtol=0,atol=0)
