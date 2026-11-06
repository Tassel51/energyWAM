import torch
import pytest
from energywam.training.run_state import ResumableWindowSampler, save_latest


def test_resume_order_across_epoch_and_odd_tail():
    full = ResumableWindowSampler(11, 42)
    first = list(full); second = list(full)
    assert len(first)==10 and len(set(first))==10
    assert list(ResumableWindowSampler(11,42,consumed=6)) == first[6:]
    assert list(ResumableWindowSampler(11,42,consumed=14)) == second[4:]


def test_replacement_preserves_previous_on_failure(tmp_path):
    class Writer:
        fail=False
        step=1
        def save(self,name):
            p=tmp_path/name
            if self.fail:
                p.write_bytes(b'partial')
                raise OSError('interrupted save')
            torch.save({'step':self.step},p)
            return p
    w=Writer();latest=save_latest(w,tmp_path)
    w.fail=True
    with pytest.raises(OSError):save_latest(w,tmp_path)
    assert torch.load(latest)['step']==1
    assert not (tmp_path/'ckpt_pending.pt').exists()
    w.fail=False;w.step=2;save_latest(w,tmp_path)
    assert torch.load(latest)['step']==2
    assert list(tmp_path.glob('*.pt'))==[latest]


def test_wall_time_checkpoint_policy_overrides_steps_and_always_saves_final():
    from energywam.training.run_state import checkpoint_due
    assert not checkpoint_due(20,40,200,20,1500)
    assert checkpoint_due(21,40,1500,20,1500)
    assert checkpoint_due(40,40,400,20,1500)
    assert not checkpoint_due(19,40,2000,20,0)
    assert checkpoint_due(20,40,1,20,0)


def test_unreadable_new_snapshot_never_replaces_latest(tmp_path):
    class Writer:
        broken=False
        def save(self,name):
            path=tmp_path/name
            if self.broken:path.write_bytes(b"invalid torch archive")
            else:torch.save({'step':1},path)
            return path
    writer=Writer();latest=save_latest(writer,tmp_path);writer.broken=True
    with pytest.raises(Exception):save_latest(writer,tmp_path)
    assert torch.load(latest)['step']==1
    assert not (tmp_path/'ckpt_pending.pt').exists()
