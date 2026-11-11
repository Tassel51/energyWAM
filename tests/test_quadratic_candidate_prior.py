import torch
import pytest
from torch import nn
from energywam.config import EnergyWAMConfig
from energywam.models.energy_core import EnergyWAMCore
from energywam.energy.types import ConditionBundle
from energywam.energy.solver import refine


@pytest.mark.parametrize('direction',['action','future'])
def test_half_step_with_compensated_prior_preserves_energy_and_candidate_gradients(direction):
    import copy
    cfg=EnergyWAMConfig();cfg.framework.quadratic_candidate_prior=True
    cfg.solver.alpha_action=5.;cfg.solver.alpha_future=2000.
    class Backbone(nn.Module):hidden_size=16
    geometry=dict(action_dim=2,state_dim=4,future_dim=3,future_patches=1,num_cameras=1,action_horizon=8,num_future_frames=3,dino_dim=3)
    baseline=EnergyWAMCore(cfg,Backbone(),geometry)
    changed=copy.deepcopy(cfg)
    setattr(changed.solver,'alpha_'+direction,getattr(changed.solver,'alpha_'+direction)*.5)
    setattr(changed.framework,'quadratic_'+direction+'_weight',.5)
    candidate=EnergyWAMCore(changed,Backbone(),geometry)
    cond=ConditionBundle(group=None,native={'extra_raw':{'future_prior_center':torch.randn(2,1,1,3)}})
    a=torch.randn(2,8,2,requires_grad=True);z=torch.randn(2,3,1,1,3,requires_grad=True)
    first=baseline.quadratic_prior_energy(cond,z,a);second=candidate.quadratic_prior_energy(cond,z,a)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    g1=torch.autograd.grad(first.sum(),(a,z));g2=torch.autograd.grad(second.sum(),(a,z))
    for x,y in zip(g1,g2):torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_pure_prior_one_step_reproduces_zero_and_current_only_baselines():
    cfg=EnergyWAMConfig();cfg.framework.quadratic_candidate_prior=True
    cfg.solver.alpha_action=5.;cfg.solver.alpha_future=2000.
    class Backbone(nn.Module):hidden_size=16
    core=EnergyWAMCore(cfg,Backbone(),dict(action_dim=2,state_dim=4,future_dim=3,
        future_patches=1,num_cameras=1,action_horizon=8,num_future_frames=3,dino_dim=3))
    current=torch.randn(2,1,1,3);cond=ConditionBundle(group=None,native={'extra_raw':{'future_prior_center':current}})
    a=torch.randn(2,8,2);z=torch.randn(2,3,1,1,3)
    ef=lambda a,z:core.quadratic_prior_energy(cond,z,a)
    ap,zp,_=refine(ef,action=a,future=z,alpha_action=5.,alpha_future=2000.,num_steps=1,training=False)
    torch.testing.assert_close(ap,torch.zeros_like(ap),rtol=0,atol=3e-7)
    torch.testing.assert_close(zp,current[:,None].expand_as(zp),rtol=1e-6,atol=2e-6)


def test_future_only_prior_leaves_action_unchanged():
    cfg=EnergyWAMConfig();cfg.framework.quadratic_candidate_prior=True
    cfg.framework.quadratic_action_weight=0.0
    cfg.solver.alpha_action=5.;cfg.solver.alpha_future=2000.
    class Backbone(nn.Module):hidden_size=16
    core=EnergyWAMCore(cfg,Backbone(),dict(action_dim=2,state_dim=4,future_dim=3,
        future_patches=1,num_cameras=1,action_horizon=8,num_future_frames=3,dino_dim=3))
    current=torch.randn(2,1,1,3)
    cond=ConditionBundle(group=None,native={'extra_raw':{'future_prior_center':current}})
    a=torch.randn(2,8,2);z=torch.randn(2,3,1,1,3)
    ap,zp,_=refine(lambda a,z:core.quadratic_prior_energy(cond,z,a),action=a,future=z,
        alpha_action=5.,alpha_future=2000.,num_steps=1,training=False)
    torch.testing.assert_close(ap,a,rtol=0,atol=0)
    torch.testing.assert_close(zp,current[:,None].expand_as(zp),rtol=1e-6,atol=2e-6)


@pytest.mark.parametrize('field', ['quadratic_action_weight','quadratic_future_weight'])
def test_prior_ablation_checkpoint_requires_matching_weights(field):
    from .real_fixtures import build_real_model
    from energywam.config import to_plain_dict
    from energywam.training.checkpoint import restore
    model,cfg,_=build_real_model(with_real_dino=False,seed=99,**{
        'backbone.condition_mode':'text_dino_state','framework.quadratic_candidate_prior':True})
    payload={'config':to_plain_dict(cfg),'model':model.state_dict(),
             'meta':{'stage':'bi','global_step':0}}
    # Historical checkpoints used implicit weights1/1. Preserve their meaning.
    for key in ['quadratic_action_weight','quadratic_future_weight']:
        payload['config']['framework'].pop(key)
    restore(payload,model=model,optimizer=None,scheduler=None,mode='resume')
    setattr(cfg.framework,field,0.)
    with pytest.raises(ValueError,match=field):
        restore(payload,model=model,optimizer=None,scheduler=None,mode='resume')


@pytest.mark.parametrize('action_prior_weight', [0.0, 1.0])
def test_real_qwen_prior_full_unroll_and_no_future_condition_leak(action_prior_weight):
    from .real_fixtures import build_real_model,make_real_batch
    model,cfg,geo=build_real_model(with_real_dino=False,seed=98,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,'backbone.dino_num_patches':4,
        'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,'framework.iteration_mlp_hidden':16,
        'framework.quadratic_candidate_prior':True,
        'framework.quadratic_action_weight':action_prior_weight,'solver.learnable_step_sizes':True})
    batch=make_real_batch(model,cfg,geo,B=1)
    prepared=model._prepare_batch(batch);cond=model._condition(batch,prepared)
    center=cond.native['extra_raw']['future_prior_center']
    changed=dict(batch,future_target=torch.randn_like(prepared['future_target'])*100)
    other=model._condition(changed,prepared).native['extra_raw']['future_prior_center']
    torch.testing.assert_close(center,other,rtol=0,atol=0)
    result=model(batch);result['action_loss'].backward()
    for module in [model.core.energy_head,model.core.iteration_conditioner,model.core.step_sizes]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    with torch.inference_mode():
        prediction=model.predict({k:batch[k] for k in ['images','langs','states']},mode='joint')
    assert torch.isfinite(torch.as_tensor(prediction['normalized_actions'])).all()


def test_zero_residual_head_starts_at_prior_and_can_learn():
    from .real_fixtures import build_real_model,make_real_batch
    model,cfg,geo=build_real_model(with_real_dino=False,seed=97,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,'backbone.dino_num_patches':4,
        'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,'framework.iteration_mlp_hidden':16,
        'framework.quadratic_candidate_prior':True,'framework.energy_head_zero_init':True,
        'solver.learnable_step_sizes':True})
    batch=make_real_batch(model,cfg,geo,B=1)
    with torch.inference_mode():prediction=model.predict({k:batch[k] for k in ['images','langs','states']},mode='joint')
    assert abs(prediction['normalized_actions']).max()<1e-6
    result=model(batch);result['action_loss'].backward()
    assert model.core.energy_head.net[-1].weight.grad.abs().sum()>0
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    optimizer.step();optimizer.zero_grad()
    model(batch)['action_loss'].backward()
    grads=[p.grad for p in model.core.iteration_conditioner.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
