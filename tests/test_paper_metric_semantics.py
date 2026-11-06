"""Independent checks for metric meaning and evaluation-context behavior."""
import torch
from energywam.energy.diagnostics import (energy_ranking_metrics,
    action_gradient_future_sensitivity, future_gradient_action_sensitivity,
    condition_swap_action_trajectory, future_latent_metrics)
from .harness import assert_close, assert_true

def test_auc_is_roc_auc_rather_than_paired_win_rate():
    out=energy_ranking_metrics(torch.tensor([0.,100.]),torch.tensor([1.,101.]),torch.ones(2,dtype=torch.bool))
    assert_close(out['ranking_accuracy'],1.,0.)
    # Across the 2 positives x 2 negatives, 3/4 comparisons win.
    assert_close(out['auc'],.75,0.)
    assert_close(out['conditional_concordance'],1.,0.)

def test_auc_does_not_depend_on_training_margin():
    m=torch.tensor([0.,100.]);n=torch.tensor([1.,101.]);v=torch.ones(2,dtype=torch.bool)
    zero=energy_ranking_metrics(m,n,v,margin=0.)
    shifted=energy_ranking_metrics(m,n,v,margin=10.)
    assert_close(zero['auc'],shifted['auc'],0.)
    assert_close(shifted['ranking_accuracy'],0.,0.)

def test_gradient_diagnostics_work_in_evaluation_contexts():
    def energy(a,z):return (a*z).flatten(1).sum(-1)
    for context in (torch.no_grad,torch.inference_mode):
        with context():
            a=torch.ones(2,2);z1=torch.ones(2,2);z2=2*torch.ones(2,2)
            out=action_gradient_future_sensitivity(energy,a,z1,z2)
            reverse=future_gradient_action_sensitivity(energy,z1,a,2*a)
            traj=condition_swap_action_trajectory(energy,lambda a,z:2*energy(a,z),a,z1,num_steps=2)
        assert_close(out['sensitivity_mean'],1.,1e-6)
        assert_close(reverse['sensitivity_mean'],1.,1e-6)
        assert_true(float(traj['endpoint_l2_mean'])>0)

def test_empty_future_mask_is_not_reported_as_perfect_prediction():
    z=torch.ones(1,3,1,2,4)
    try:future_latent_metrics(z,z,torch.zeros(1,3,1,2,dtype=torch.bool),space='raw_dino')
    except ValueError:return
    raise AssertionError('empty mask must not produce MSE=0 / cosine=1 as a valid metric')

TESTS=[('auc_is_roc_auc_rather_than_paired_win_rate',test_auc_is_roc_auc_rather_than_paired_win_rate),
       ('auc_does_not_depend_on_training_margin',test_auc_does_not_depend_on_training_margin),
       ('gradient_diagnostics_work_in_evaluation_contexts',test_gradient_diagnostics_work_in_evaluation_contexts),
       ('empty_future_mask_is_not_perfect_prediction',test_empty_future_mask_is_not_reported_as_perfect_prediction)]
