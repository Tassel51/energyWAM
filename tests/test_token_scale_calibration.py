import torch
from energywam.energy.encodings import CandidateEncoding


def test_calibration_scales_positions_and_types_without_consuming_extra_rng():
    geo={}
    torch.manual_seed(55);legacy=CandidateEncoding(48,geo);after0=torch.rand(4)
    torch.manual_seed(55);scaled=CandidateEncoding(48,geo,positional_scale=.02,type_embedding_std=.02);after1=torch.rand(4)
    torch.testing.assert_close(after0,after1,rtol=0,atol=0)
    torch.testing.assert_close(scaled.type_embed.weight,legacy.type_embed.weight*.02,rtol=0,atol=0)
    for name,args in [('action',(1,8,torch.device('cpu'),torch.float32)),('dino_current',(1,1,4,torch.device('cpu'),torch.float32))]:
        torch.testing.assert_close(getattr(scaled,name)(*args),getattr(legacy,name)(*args)*.02,rtol=1e-6,atol=1e-8)


def test_real_qwen_calibrated_tokens_full_double_backward():
    from .real_fixtures import build_real_model,make_real_batch
    model,cfg,geo=build_real_model(with_real_dino=False,seed=99,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,'backbone.dino_num_patches':4,
        'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,'framework.iteration_mlp_hidden':16,
        'framework.projector_output_scale':.1,'framework.positional_encoding_scale':.02,'framework.type_embedding_init_std':.02,
        'solver.candidate_gradient_normalization':'rms','solver.learnable_step_sizes':True})
    result=model(make_real_batch(model,cfg,geo,B=1));result['action_loss'].backward()
    for module in [model.core.action_projector,model.core.future_projector,model.core.current_dino_projector,
                   model.core.state_projector,model.core.iteration_conditioner,model.core.energy_head]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
