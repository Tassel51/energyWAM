"""Fixed/learned step comparison must not randomize the shared model."""
import torch
from .real_fixtures import build_real_model


def test_learned_step_switch_preserves_all_common_initial_weights_and_rng():
    common={'backbone.condition_mode':'text_dino_state',
        'framework.iteration_conditioning':True,'framework.iteration_embedding_dim':16,
        'framework.iteration_mlp_hidden':16,'framework.energy_head_zero_init':True,
        'solver.alpha_action':5.,'solver.alpha_future':2000.}
    fixed,_,_=build_real_model(with_real_dino=False,seed=42,**common,
        **{'solver.learnable_step_sizes':False})
    rng_fixed=torch.get_rng_state().clone()
    learned,_,_=build_real_model(with_real_dino=False,seed=42,**common,
        **{'solver.learnable_step_sizes':True})
    torch.testing.assert_close(torch.get_rng_state(),rng_fixed,rtol=0,atol=0)
    a=dict(fixed.named_parameters());b=dict(learned.named_parameters())
    assert set(b)-set(a)=={'core.step_sizes.raw'}
    for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
    for x,y in zip(fixed.core.effective_step_sizes(),learned.core.effective_step_sizes()):
        torch.testing.assert_close(torch.as_tensor(x),y,rtol=1e-6,atol=1e-6)
