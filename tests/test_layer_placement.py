from types import SimpleNamespace
import torch
from torch import nn
import pytest
from energywam.training.parallel import shard_language_model


def setup(monkeypatch):
    lm=nn.Module();lm.layers=nn.ModuleList([nn.Identity() for _ in range(36)]);lm.norm=nn.Identity();lm.embed_tokens=nn.Identity()
    model=SimpleNamespace(cfg=SimpleNamespace(backbone=SimpleNamespace(backend='qwen3vl')),backbone=SimpleNamespace(_mm=SimpleNamespace(language_model=lm)))
    seen={};monkeypatch.setattr(torch.cuda,'device_count',lambda:2)
    monkeypatch.setattr('accelerate.dispatch_model',lambda module,device_map,main_device:seen.update(device_map))
    return model,seen


def test_default_placement_identical_and_optional_split_moves_only_selected_layers(monkeypatch):
    model,seen=setup(monkeypatch);shard_language_model(model,[0,1],'cuda:0')
    assert [seen[f'layers.{i}'] for i in range(36)]==['cuda:0']*18+['cuda:1']*18
    shard_language_model(model,[0,1],'cuda:0',layer_counts=[16,20])
    assert [seen[f'layers.{i}'] for i in range(36)]==['cuda:0']*16+['cuda:1']*20
    assert seen['norm']==seen['embed_tokens']=='cuda:0'


@pytest.mark.parametrize('counts',[[0,36],[17,18],[36],[16,20.0]])
def test_invalid_split_rejected(monkeypatch,counts):
    model,_=setup(monkeypatch)
    with pytest.raises(ValueError):shard_language_model(model,[0,1],'cuda:0',layer_counts=counts)
