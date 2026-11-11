"""Read-only candidate traces, checked against the production inference solver."""
import math
import time
import torch


def trace_candidate(energy_fn, initial, target, alpha, max_k=4):
    initial=initial.detach().float()
    target=target.detach().float()
    candidate=initial.clone()
    rows=[]
    applied_rms=0.0
    initial_rms=float(initial.square().mean().sqrt())
    started=time.perf_counter()
    for k in range(max_k+1):
        refinement_seconds=time.perf_counter()-started if k else 0.0
        candidate=candidate.detach().requires_grad_(True)
        energy=energy_fn(candidate)
        gradient=torch.autograd.grad(energy.sum(),candidate,create_graph=False)[0]
        if not torch.isfinite(candidate).all() or not torch.isfinite(energy).all() or not torch.isfinite(gradient).all():
            raise FloatingPointError(f'non-finite candidate/energy/gradient at K={k}')
        rms=float(gradient.square().mean().sqrt())
        displacement=float((candidate.detach()-initial).square().mean().sqrt())
        rows.append({'refinement_seconds':refinement_seconds,'k':k,'prediction_mse':float((candidate.detach()-target).square().mean()),
                     'energy':float(energy.detach().mean()),'gradient_rms':rms,
                     'gradient_l2':float(gradient.norm()),'applied_update_rms':applied_rms,
                     'next_step_proposal_rms':alpha*rms,
                     'total_displacement_rms':displacement,
                     'relative_total_displacement':displacement/max(initial_rms,1e-12),
                     'candidate_rms':float(candidate.detach().square().mean().sqrt()),
                     'target_rms':float(target.square().mean().sqrt()),'nonfinite':0})
        if k<max_k:
            next_candidate=candidate.detach()-alpha*gradient
            applied_rms=float((next_candidate-candidate.detach()).square().mean().sqrt())
            candidate=next_candidate.detach()
    return rows,candidate.detach()
