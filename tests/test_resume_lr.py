import copy
import torch
import pytest
from energywam.training.resume_lr import override_resumed_lrs


def test_override_retains_moments_and_survives_scheduler_and_resume():
    p=torch.nn.Parameter(torch.tensor(1.))
    opt=torch.optim.AdamW([{'params':[p],'lr':1e-3}])
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/16))
    for _ in range(20):
        opt.zero_grad();p.square().backward();opt.step();sch.step()
    old=copy.deepcopy(opt.state_dict()['state']);epoch=sch.last_epoch
    override_resumed_lrs(opt,sch,[1e-4])
    assert sch.last_epoch==epoch
    for key,value in old[0].items():
        assert torch.equal(value,opt.state_dict()['state'][0][key])
    assert opt.param_groups[0]['lr']==pytest.approx(1e-4)
    opt.step();sch.step()
    assert sch.get_last_lr()==pytest.approx([1e-4])
    saved_opt=copy.deepcopy(opt.state_dict());saved_sch=copy.deepcopy(sch.state_dict())
    opt.load_state_dict(saved_opt);sch.load_state_dict(saved_sch)
    override_resumed_lrs(opt,sch,[1e-4])
    assert opt.param_groups[0]['lr']==pytest.approx(1e-4)
    assert sch.base_lrs==pytest.approx([1e-4])


def test_warmup_progress_not_reset():
    p=torch.nn.Parameter(torch.tensor(1.));o=torch.optim.AdamW([p],lr=.001)
    s=torch.optim.lr_scheduler.LambdaLR(o,lambda n:min(1.,(n+1)/16))
    override_resumed_lrs(o,s,[.0001])
    assert o.param_groups[0]['lr']==pytest.approx(.0001/16)
