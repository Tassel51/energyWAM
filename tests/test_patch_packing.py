import copy
import torch
from torch import nn
from energywam.energy.patch_packing import pack_patches
from energywam.models.energy_core import EnergyWAMCore
from energywam.config import EnergyWAMConfig


def core(factor=2):
    cfg=EnergyWAMConfig();cfg.framework.dino_patch_pack_factor=factor
    class Backbone(nn.Module):hidden_size=16
    return EnergyWAMCore(cfg,Backbone(),dict(action_dim=2,state_dim=4,future_dim=3,
        future_patches=196,num_cameras=1,action_horizon=8,num_future_frames=3,dino_dim=3))


def test_packing_spatial_order_retains_all_values_and_gradients():
    x=torch.arange(16.).reshape(1,16,1).requires_grad_()
    packed=pack_patches(x,2)
    torch.testing.assert_close(packed,torch.tensor([[[0.,1.,4.,5.],[2.,3.,6.,7.],[8.,9.,12.,13.],[10.,11.,14.,15.]]]))
    packed.sum().backward();torch.testing.assert_close(x.grad,torch.ones_like(x))
    assert pack_patches(x,1) is x


def test_full_resolution_candidates_group_masks_and_gradients():
    torch.manual_seed(84);m=core()
    z=torch.randn(1,3,1,196,3,requires_grad=True)
    mask=torch.ones_like(z,dtype=torch.bool);mask[:,:, :,0]=False
    polluted=z.detach().clone();polluted[~mask]=float('nan')
    safe=z.detach().clone();safe[~mask]=999999.
    a=m.build_future_tokens(polluted,future_mask=mask)
    b=m.build_future_tokens(safe,future_mask=mask)
    assert a.embeds.shape==(1,147,16)
    torch.testing.assert_close(a.embeds,b.embeds,rtol=0,atol=0)
    m.build_future_tokens(z).embeds.sum().backward()
    assert z.grad.shape==z.shape and torch.isfinite(z.grad).all()
    assert (z.grad.abs().sum(-1)>0).all()
    current=m.build_current_dino_tokens(torch.randn(1,1,196,3))
    assert current.embeds.shape==(1,49,16)
    assert m.build_future_placeholder_tokens(1,None,torch.device('cpu'),torch.float32).embeds.shape==(1,147,16)


def test_real_qwen_full_unroll_packed_candidates_and_checkpoint_contract():
    import pytest
    from .real_fixtures import build_real_model,make_real_batch
    from energywam.config import to_plain_dict
    from energywam.training.checkpoint import restore
    model,cfg,geo=build_real_model(with_real_dino=False,seed=130,**{
        'backbone.condition_mode':'text_dino_state','backbone.dino_hidden_dim':8,'backbone.dino_num_patches':4,
        'framework.dino_patch_pack_factor':2,'framework.iteration_conditioning':True,
        'framework.iteration_embedding_dim':16,'framework.iteration_mlp_hidden':16,
        'solver.learnable_step_sizes':True})
    batch=make_real_batch(model,cfg,geo,B=1);result=model(batch);result['action_loss'].backward()
    for module in [model.core.future_projector,model.core.action_projector,model.core.current_dino_projector,model.core.iteration_conditioner]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum()>0 for g in grads)
    with torch.inference_mode():pred=model.predict({k:batch[k] for k in ['images','langs','states']},mode='joint',return_diagnostics=True)
    assert torch.isfinite(torch.as_tensor(pred['normalized_actions'])).all()
    assert tuple(pred['future_latent'].shape[-2:])==(4,8)
    assert tuple(model._prepare_batch(batch)['future_target'].shape[-2:])==(4,8)
    payload={'config':to_plain_dict(cfg),'model':model.state_dict(),'meta':{'stage':'bi','global_step':0}}
    cfg.framework.dino_patch_pack_factor=1
    with pytest.raises(ValueError,match='dino_patch_pack_factor'):
        restore(payload,model=model,optimizer=None,scheduler=None)


def test_packing_preserves_common_initialization_and_global_rng():
    torch.manual_seed(193);plain=core(factor=1);rng0=torch.rand(4)
    torch.manual_seed(193);packed=core(factor=2);rng1=torch.rand(4)
    torch.testing.assert_close(rng0,rng1,rtol=0,atol=0)
    exceptions={'future_projector.0.weight','current_dino_projector.0.weight'}
    for key,value in plain.state_dict().items():
        if key not in exceptions:torch.testing.assert_close(value,packed.state_dict()[key],rtol=0,atol=0)
