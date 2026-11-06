import torch
from energywam.energy.solver import refine
from energywam.energy.solver_utils import precondition_candidate_grad


def test_rms_ignores_padding_and_is_twice_differentiable():
    x=torch.tensor([[1.,2.,99.],[3.,4.,99.]],dtype=torch.double,requires_grad=True)
    mask=torch.tensor([[True,True,False],[True,True,False]])
    y=precondition_candidate_grad(x,mask,'rms',1e-4)
    assert torch.count_nonzero(y[:,-1])==0
    assert torch.allclose(y[:,:2].square().mean(-1),torch.ones(2,dtype=torch.double),atol=1e-7)
    fn=lambda v:precondition_candidate_grad(v,mask,'rms',1e-4)
    assert torch.autograd.gradcheck(fn,(x,))
    assert torch.autograd.gradgradcheck(fn,(x,))
    zero=precondition_candidate_grad(x,torch.zeros_like(mask),'rms')
    assert torch.isfinite(zero).all() and zero.count_nonzero()==0


def test_energy_scaling_and_frozen_variable_semantics():
    a=torch.full((1,2,1),2.,dtype=torch.float32);z=torch.full((1,1,1,2,3),2.)
    def energy(a,z,iteration_progress=None):return a.square().flatten(1).sum(1)+z.square().flatten(1).sum(1)/7
    results=[]
    for scale in [1.,100.]:
        x,y,_=refine(lambda a,z:scale*energy(a,z),action=a,future=z,num_steps=2,training=False,
            alpha_action=.5,alpha_future=.5,candidate_gradient_normalization='rms',gradient_rms_epsilon=1e-8)
        results.append((x,y))
    for x,y in zip(results[0],results[1]):torch.testing.assert_close(x,y,rtol=1e-6,atol=1e-6)
    x,y,_=refine(energy,action=a,future=z,num_steps=2,training=False,mode='action_given_future',
        alpha_action=.5,candidate_gradient_normalization='rms')
    torch.testing.assert_close(y,z,rtol=0,atol=0)
    assert x.square().mean()<a.square().mean()


def test_normalized_probe_matches_production_solver_and_outer_gradients():
    from scripts.k_training_probe import iteration_trace
    theta=torch.nn.Parameter(torch.tensor(.3))
    initial=torch.tensor([[[2.],[1.]]]);target=torch.zeros_like(initial);z=torch.zeros(1,1,1,1,1)
    def energy(a,z,iteration_progress):return ((a-theta-iteration_progress)**2).flatten(1).sum(1)+z.flatten(1).sum(1)
    a,_,_=refine(energy,action=initial.clone().requires_grad_(),future=z,num_steps=2,mode='action_given_future',
        alpha_action=.5,training=True,iteration_aware=True,candidate_gradient_normalization='rms')
    a.square().mean().backward();assert theta.grad is not None and torch.isfinite(theta.grad) and theta.grad.abs()>0
    rows=iteration_trace(lambda x,iteration_progress:energy(x,z,iteration_progress),initial,target,.5,[2],gradient_normalization='rms')
    assert abs(rows[-1]['prediction_mse']-float(a.detach().square().mean()))<1e-7


def test_real_qwen_two_directions_normalized_full_unroll():
    from .test_iteration_aware import fixture
    model,cfg,batch=fixture();cfg.solver.candidate_gradient_normalization='rms'
    cfg.solver.alpha_action=.2;cfg.solver.alpha_future=.2
    result=model(batch);result['action_loss'].backward()
    for module in [model.core.iteration_conditioner,model.core.energy_head,model.core.step_sizes]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
