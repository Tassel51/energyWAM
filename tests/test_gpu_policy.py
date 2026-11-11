import json
import pytest
from energywam.training.gpu_policy import authorized_visibility


def test_local_policy_still_rejects_other_users_gpus(tmp_path):
    p=tmp_path/'policy.json';p.write_text(json.dumps({'allowed_physical_pairs':['2,3']}))
    assert authorized_visibility('2,3',p,visible='2,3')=='2,3'
    with pytest.raises(RuntimeError):authorized_visibility('0,1',p,visible='0,1')
    with pytest.raises(RuntimeError):authorized_visibility('2,3',p,visible='0,1')


def test_remote_uses_uuid_and_is_bound_to_host_and_device_identity(tmp_path,monkeypatch):
    p=tmp_path/'policy.json';cfg={'hostname':'lab','allowed_physical_pairs':['6,7'],'use_uuid_visibility':True,'expected_gpu_uuids':{'6':'GPU-a','7':'GPU-b'}}
    p.write_text(json.dumps(cfg))
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'6, GPU-a\n7, GPU-b\n')
    assert authorized_visibility('6,7',p,hostname='lab',visible='6,7')=='GPU-a,GPU-b'
    with pytest.raises(RuntimeError):authorized_visibility('6,7',p,hostname='cloud',visible='6,7')
    with pytest.raises(RuntimeError):authorized_visibility('6,7',p,hostname='lab',visible='GPU-wrong,GPU-b')
    cfg['expected_gpu_uuids']['6']='GPU-stale';p.write_text(json.dumps(cfg))
    with pytest.raises(RuntimeError):authorized_visibility('6,7',p,hostname='lab',visible='6,7')


@pytest.mark.parametrize('pair',['6,6','-1,7','0','GPU-a,GPU-b','0,1,2'])
def test_invalid_pairs_rejected_before_gpu_access(tmp_path,pair):
    with pytest.raises(ValueError):authorized_visibility(pair,tmp_path/'unused',visible=pair)


def test_shared_device_allocator_cap_uses_logical_index(tmp_path,monkeypatch):
    import torch
    from energywam.training.gpu_policy import apply_allocator_limits
    p=tmp_path/'policy.json';p.write_text(json.dumps(dict(hostname='lab',allowed_physical_pairs=['1,8'],use_uuid_visibility=True,expected_gpu_uuids={'1':'GPU-a','8':'GPU-b'},cuda_allocator_limit_mib={'8':31000})))
    monkeypatch.setattr('socket.gethostname',lambda:'lab')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-a,GPU-b')
    monkeypatch.setattr('subprocess.check_output',lambda *a,**k:'1, GPU-a\n8, GPU-b\n')
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda i:type('Props',(),dict(total_memory=46*2**30))())
    calls=[];monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda f,device:calls.append((f,device)))
    apply_allocator_limits('1,8',p)
    assert calls==[(31000*2**20/(46*2**30),1)]
