import json
from pathlib import Path
import pytest
from energywam.training.checkpoint_metadata import read_checkpoint_metadata,write_checkpoint_metadata


def test_transient_empty_sidecar_is_retried_without_rewriting_weight(tmp_path,monkeypatch):
    p=tmp_path/'checkpoint.meta.json';value={'meta':{'global_step':92}};p.write_text(json.dumps(value))
    original=Path.read_text;calls=[]
    def read(path,*a,**kw):
        if path==p:
            calls.append(1)
            if len(calls)<3:return ''
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',read)
    monkeypatch.setattr('energywam.training.checkpoint_metadata.time.sleep',lambda _:None)
    assert read_checkpoint_metadata(p,expected_meta=value['meta'])==value
    assert len(calls)==3


def test_permanent_mismatch_is_not_silently_accepted(tmp_path,monkeypatch):
    p=tmp_path/'checkpoint.meta.json';p.write_text(json.dumps({'meta':{'global_step':60}}))
    monkeypatch.setattr('energywam.training.checkpoint_metadata.time.sleep',lambda _:None)
    with pytest.raises(ValueError,match='mismatch'):
        read_checkpoint_metadata(p,expected_meta={'global_step':92},attempts=2)


def test_failed_atomic_publish_keeps_previous_sidecar(tmp_path,monkeypatch):
    p=tmp_path/'checkpoint.meta.json';old={'meta':{'global_step':60}};p.write_text(json.dumps(old))
    def fail(*a):raise OSError('injected publication failure')
    monkeypatch.setattr('energywam.training.checkpoint_metadata.os.replace',fail)
    with pytest.raises(OSError):write_checkpoint_metadata(p,{'meta':{'global_step':92}})
    assert json.loads(p.read_text())==old
    assert list(tmp_path.glob('*.pending'))==[]


def test_atomic_write_is_verified_and_leaves_no_pending_file(tmp_path):
    p=tmp_path/'checkpoint.meta.json';value={'meta':{'global_step':92},'config':{'seed':42}}
    write_checkpoint_metadata(p,value)
    assert read_checkpoint_metadata(p)==value
    assert list(tmp_path.glob('*.pending'))==[]


def test_invalid_pending_sidecar_never_replaces_previous_full_checkpoint(tmp_path,monkeypatch):
    import torch
    from energywam.training.run_state import save_latest
    class Writer:
        step=1
        broken=False
        def save(self,name):
            p=tmp_path/name;meta={'global_step':self.step}
            torch.save(dict(model={'weight':torch.tensor([self.step])},optimizer={},scheduler={},meta=meta,config={},rng={},format_version=1),p)
            Path(str(p)+'.meta.json').write_text('' if self.broken else json.dumps({'meta':meta}))
            return p
    monkeypatch.setattr('energywam.training.checkpoint_metadata.time.sleep',lambda _:None)
    w=Writer();p=save_latest(w,tmp_path);w.step=2;w.broken=True
    with pytest.raises(json.JSONDecodeError):save_latest(w,tmp_path)
    assert torch.load(p,weights_only=False)['meta']['global_step']==1
    assert json.loads(Path(str(p)+'.meta.json').read_text())['meta']['global_step']==1
    assert not (tmp_path/'ckpt_pending.pt').exists()
