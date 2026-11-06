import torch
from .real_fixtures import build_real_model,make_real_batch


def test_cached_current_dino_preserves_energy_and_candidate_gradients():
    model,cfg,geo=build_real_model(with_real_dino=False,seed=31,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,'backbone.dino_num_patches':4})
    model.eval();batch=make_real_batch(model,cfg,geo,B=1)
    current=model.backbone.dino_from_images(batch['images'],torch.device('cpu'))
    a=torch.randn(1,8,3,requires_grad=True);z=torch.randn(1,3,1,4,8,requires_grad=True)
    e0=model.core.energy(model.backbone.encode_condition(batch),z,a)
    g0=torch.autograd.grad(e0.sum(),(a,z))
    e1=model.core.energy(model.backbone.encode_condition(dict(batch,current_dino=current)),z,a)
    g1=torch.autograd.grad(e1.sum(),(a,z))
    torch.testing.assert_close(e0,e1,rtol=0,atol=0)
    for x,y in zip(g0,g1):torch.testing.assert_close(x,y,rtol=0,atol=0)
