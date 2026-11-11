import pytest
import torch
from scripts.k_training_probe import iteration_trace
from energywam.energy.solver import refine


@pytest.mark.parametrize('clip,clamp',[(None,(None,None)),(.1,(None,None)),(.1,(-.5,.5))])
def test_evaluation_trace_matches_production_refine_with_constraints(clip,clamp):
    a=torch.tensor([[[2.,-.7]]]);z=torch.tensor([[[.3,.8]]]);target=torch.zeros_like(a)
    def energy(a,z,iteration_progress=None):
        return ((a-(.2+float(iteration_progress or 0)))**2).flatten(1).sum(1)+z.flatten(1).sum(1)*0
    predicted,_,_=refine(energy,action=a,future=z,mode='action_given_future',
        alpha_action=.6,num_steps=2,training=False,iteration_aware=True,
        clip_action_grad=clip,action_clamp=clamp)
    rows=iteration_trace(lambda x,iteration_progress:energy(x,z,iteration_progress),
        a,target,.6,[2],clip_grad=clip,candidate_clamp=clamp)
    assert rows[-1]['prediction_mse']==pytest.approx(float(predicted.square().mean()),abs=1e-7)
    assert len(rows[-1]['inner_steps'])==2
    if clip is not None:
        assert rows[-1]['inner_steps'][0]['clipped_fraction']>0
