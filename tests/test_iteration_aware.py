"""CPU checks: actual small Qwen, full double backward, no pretrained-4B claim."""
import io
import pytest
import torch
from energywam.energy.iteration import IterationConditioning, BoundedStepSizes
from energywam.energy.solver import refine


def fixture():
    from .real_fixtures import build_real_model, make_real_batch
    model,cfg,geo=build_real_model(with_real_dino=False,seed=77,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,
        'backbone.dino_num_patches':4,'framework.iteration_conditioning':True,
        'framework.iteration_embedding_dim':16,'framework.iteration_mlp_hidden':16,
        'solver.learnable_step_sizes':True})
    return model,cfg,make_real_batch(model,cfg,geo,B=2)


def test_continuous_pe_and_bounded_steps():
    module=IterationConditioning(48,16,16)
    assert module.encode(0).shape==(1,16)
    assert not torch.equal(module.encode(0),module.encode(.5))
    assert module(0,'action').shape==(1,1,48)
    assert not list(dict(module.named_parameters()).keys())[0].startswith('frequencies')
    steps=BoundedStepSizes(.05,200)
    torch.testing.assert_close(torch.stack(steps()),torch.tensor([.05,200.]))
    for raw in [-100.,100.]:
        with torch.no_grad():steps.raw.fill_(raw)
        alpha=torch.stack(steps())
        assert torch.all(alpha>=steps.initial*.09999) and torch.all(alpha<=steps.initial*10.001)


def test_progress_and_learned_alpha_backprop_through_full_unroll():
    steps=BoundedStepSizes(.05,.2)
    theta=torch.nn.Parameter(torch.tensor(.7))
    seen=[]
    def e(a,z,iteration_progress):
        seen.append(iteration_progress)
        return ((a-theta-iteration_progress)**2).flatten(1).sum(1)+((z-theta)**2).flatten(1).sum(1)
    a,z,_=refine(e,action=torch.ones(1,2,1,requires_grad=True),future=torch.ones(1,1,1,1,1,requires_grad=True),
        num_steps=2,alpha_action=steps()[0],alpha_future=steps()[1],iteration_aware=True)
    (a.square().sum()+z.square().sum()).backward()
    assert seen==[0.,.5]
    assert theta.grad.abs()>0 and torch.all(steps.raw.grad.abs()>0)
    assert torch.isfinite(steps.raw.grad).all()


def test_text_only_condition_tokens_masks_and_full_model_gradients():
    model,cfg,batch=fixture()
    assert model.backbone._mm.visual is None
    assert not any('.visual.' in k for k in model.state_dict())
    cond=model.backbone.encode_condition(batch)
    assert cond.deepstack_visual_embeds is None and cond.visual_pos_masks is None
    assert all(not p.requires_grad for p in model.backbone.dino.parameters())
    a=torch.randn(2,8,3);z=torch.randn(2,3,1,4,8)
    mask=torch.ones(2,8,dtype=torch.bool);mask[:,-1]=False
    g0=model.core.build_action_tokens(a,action_mask=mask,iteration_progress=0)
    g1=model.core.build_action_tokens(a,action_mask=mask,iteration_progress=.5)
    assert g0.embeds.shape==g1.embeds.shape==(2,8,48)
    assert torch.count_nonzero(g1.embeds[:,-1])==0
    assert not torch.equal(g0.embeds[:,:-1],g1.embeds[:,:-1])
    torch.testing.assert_close(g0.embeds,model.core.build_action_tokens(a,action_mask=mask,iteration_progress=0).embeds)
    with pytest.raises(ValueError,match='explicit'):
        model.core.energy(cond,z,a)
    model.zero_grad(set_to_none=True)
    result=model(batch)
    result['action_loss'].backward()
    for name,mod in [('iteration',model.core.iteration_conditioner),('steps',model.core.step_sizes),
                     ('transformer',model.backbone._mm.language_model)]:
        grads=[p.grad for p in mod.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads),name
        assert any(g.abs().sum()>0 for g in grads),name
    for mlp in [model.core.iteration_conditioner.action_mlp,model.core.iteration_conditioner.future_mlp]:
        assert all(p.grad is not None and p.grad.abs().sum()>0 for p in mlp.parameters())
    assert model.core.step_sizes.raw.grad.abs().min()>0
    model.zero_grad(set_to_none=True)
    flags=[p.requires_grad for p in model.parameters()]
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    with torch.inference_mode():
        pred=model.predict({k:batch[k] for k in ['images','langs','states']},mode='joint')
    assert pred['normalized_actions'].shape==(2,8,3)
    assert flags==[p.requires_grad for p in model.parameters()]
    assert all(p.grad is None for p in model.parameters())
    for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    # Strict state round trip covers iteration MLPs, step raw variables and anchors.
    state=io.BytesIO();torch.save(model.state_dict(),state);state.seek(0)
    other,_,_=fixture();other.load_state_dict(torch.load(state,weights_only=True),strict=True)
    torch.testing.assert_close(torch.stack(model.core.effective_step_sizes()),torch.stack(other.core.effective_step_sizes()))


def test_independent_eval_progress_schedules():
    from scripts.k_training_probe import iteration_trace
    seen=[]
    def energy(x,iteration_progress):
        seen.append(iteration_progress)
        return x.square().flatten(1).sum(1)
    rows=iteration_trace(energy,torch.ones(1,2),torch.zeros(1,2),.1,[2,4])
    assert seen==[0,.5,0,.25,.5,.75]
    assert [r['k'] for r in rows]==[0,2,4]
    assert rows[-1]['prediction_mse']<rows[1]['prediction_mse']


def test_optimizer_step_group_and_checkpoint(tmp_path):
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    model,cfg,batch=fixture()
    cfg.trainer.output_dir=str(tmp_path);cfg.trainer.optimizer='adamw'
    cfg.trainer.warmup_steps=0
    trainer=SharedMicrobatchTrainer(cfg,model)
    assert len(trainer.optimizer.param_groups)==2
    assert trainer.optimizer.param_groups[1]['lr']==cfg.trainer.step_size_learning_rate
    before=model.core.step_sizes.raw.detach().clone()
    row=trainer.train_step(batch)
    assert row["diag/iteration_action_grad_norm"]>0
    assert row["diag/step_future_raw_grad"]!=0
    assert row["diag/alpha_action_after_update"]>0
    assert not torch.equal(before,model.core.step_sizes.raw)
    assert torch.isfinite(model.core.step_sizes.raw).all()
    checkpoint=tmp_path/'test.pt';trainer.save(str(checkpoint))
    other,cfg2,_=fixture();cfg2.trainer.output_dir=str(tmp_path/'resumed');cfg2.trainer.optimizer='adamw';cfg2.trainer.warmup_steps=0
    resumed=SharedMicrobatchTrainer(cfg2,other);resumed.load(str(checkpoint))
    assert resumed.global_step==trainer.global_step
    torch.testing.assert_close(other.core.step_sizes.raw,model.core.step_sizes.raw)
    assert resumed.optimizer.state_dict()['param_groups']==trainer.optimizer.state_dict()['param_groups']


def test_legacy_defaults_and_common_initialization():
    from .real_fixtures import build_real_model
    from energywam.config import EnergyWAMConfig
    cfg=EnergyWAMConfig()
    assert cfg.backbone.condition_mode=='qwen_vl'
    assert not cfg.framework.iteration_conditioning and not cfg.solver.learnable_step_sizes
    new,_,_=fixture()
    baseline,_,_=build_real_model(with_real_dino=False,seed=77,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,
        'backbone.dino_num_patches':4})
    newer=dict(new.named_parameters())
    for name,p in baseline.named_parameters():
        torch.testing.assert_close(p,newer[name],rtol=0,atol=0,msg=name)
    assert baseline.core.step_sizes is None and baseline.core.iteration_conditioner is None


def test_bfloat16_checkpointed_language_full_unroll():
    from .real_fixtures import build_real_model,make_real_batch
    model,cfg,geo=build_real_model(with_real_dino=False,seed=78,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,
        'backbone.dino_num_patches':4,'backbone.gradient_checkpointing':True,
        'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,
        'framework.iteration_mlp_hidden':16,'solver.learnable_step_sizes':True})
    model.backbone._mm.language_model.to(torch.bfloat16)
    result=model(make_real_batch(model,cfg,geo,B=1))
    result['action_loss'].backward()
    for p in model.core.iteration_conditioner.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    assert torch.isfinite(model.core.step_sizes.raw.grad).all()
    assert model.core.step_sizes.raw.grad.abs().min()>0


def test_separate_task_lr_partitions_parameters_once(tmp_path):
    from energywam.training.shared_microbatch import SharedMicrobatchTrainer
    model,cfg,batch=fixture()
    cfg.trainer.output_dir=str(tmp_path);cfg.trainer.optimizer='adamw';cfg.trainer.warmup_steps=0
    cfg.trainer.task_learning_rate=1e-4
    trainer=SharedMicrobatchTrainer(cfg,model)
    groups={g['group_name']:g for g in trainer.optimizer.param_groups}
    assert groups['language']['lr']==1e-5
    assert groups['task']['lr']==1e-4
    assert groups['step_sizes']['lr']==1e-3
    all_ids=[id(p) for g in groups.values() for p in g['params']]
    assert len(all_ids)==len(set(all_ids))
    assert set(all_ids)=={id(p) for p in model.parameters() if p.requires_grad}
    assert id(model.core.energy_head.net[1].weight) in {id(p) for p in groups['task']['params']}
    assert id(model.backbone._mm.get_input_embeddings().weight) in {id(p) for p in groups['language']['params']}
    trainer.train_step(batch)
