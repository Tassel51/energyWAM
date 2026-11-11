import copy
import torch
import pytest
from energywam.training.optimizer_offload import OptimizerStateOffloader


def test_nested_state_alias_dtype_and_checkpoint_roundtrip():
    p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=.01)
    p.grad=torch.tensor([.2,.3]);opt.step()
    quant=torch.arange(256,dtype=torch.uint8)
    opt.state[p]['nested']={'first':quant,'alias':[quant,('metadata',17)]}
    before=copy.deepcopy(opt.state_dict());groups=opt.param_groups
    off=OptimizerStateOffloader(opt)
    assert off.offload()==0
    off.restore()
    assert opt.param_groups is groups
    state=opt.state[p]
    assert state['nested']['first'] is state['nested']['alias'][0]
    assert state['nested']['first'].dtype==torch.uint8
    payload=opt.state_dict()
    for key in ['exp_avg','exp_avg_sq','step']:
        assert torch.equal(payload['state'][0][key],before['state'][0][key])
    restored=torch.optim.AdamW([torch.nn.Parameter(p.detach().clone())],lr=.01)
    restored.load_state_dict(payload)
    assert torch.equal(restored.state_dict()['state'][0]['exp_avg'],before['state'][0]['exp_avg'])


def test_paged_states_refused():
    p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.AdamW([p]);t=torch.zeros(2);t.is_paged=True
    opt.state[p]={'state1':t}
    with pytest.raises(ValueError,match='Paged'):
        OptimizerStateOffloader(opt).offload()
