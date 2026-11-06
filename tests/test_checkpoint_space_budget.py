import os
import pytest
from scripts.run_iteration_search import checkpoint_space_required


def test_new_full_retention_reserves_latest_best_and_pending(tmp_path):
    size=1024
    assert checkpoint_space_required(tmp_path,size,keep_best=True)==3*size+4*2**30
    assert checkpoint_space_required(tmp_path,size)==2*size+4*2**30
    assert checkpoint_space_required(tmp_path,size,profile=True)==2*2**30


def test_existing_roles_are_counted_by_physical_identity_not_alias_count(tmp_path):
    latest=tmp_path/'ckpt_latest.pt';latest.write_bytes(b'x'*1024)
    best=tmp_path/'ckpt_best.pt';os.link(latest,best)
    assert checkpoint_space_required(tmp_path,1024,keep_best=True)==2048+4*2**30
    best.unlink();best.write_bytes(b'y'*1024)
    assert checkpoint_space_required(tmp_path,1024,keep_best=True)==1024+4*2**30
    with pytest.raises(ValueError):checkpoint_space_required(tmp_path,True)
