import json
import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from k_training_probe import evaluation_budgets,iteration_trace
from run_iteration_search import collect,fixed_mse_failure,gpu_idle
from energywam.training.best_checkpoint import validation_score_from_reports


def test_k3_budget_does_not_change_legacy_evaluation():
    assert evaluation_budgets(2)==[2]
    assert evaluation_budgets(8)==[2,4,8]
    assert evaluation_budgets(3,3,'2,3')==[2,3]
    with pytest.raises(ValueError):evaluation_budgets(3,3,'2')


def test_k3_uses_its_own_iteration_schedule():
    calls=[]
    def energy(x,iteration_progress):
        calls.append(iteration_progress);return .5*x.square().sum()
    rows=iteration_trace(energy,torch.ones(2),torch.zeros(2),.1,[3])
    assert calls==[0,1/3,2/3]
    assert rows[-1]['k']==3
    assert rows[-1]['prediction_mse']==pytest.approx(.9**6)


def test_k3_collect_best_and_failure_guard(tmp_path):
    for direction in ['action','future']:
        rows=[dict(k=k,direction=direction,split='val',prediction_mse=v,update=20,window_id=0,episode_index=1,frame_index=0) for k,v in [(2,9.),(3,.2)]]
        (tmp_path/f'eval_0020_{direction}.json').write_text(json.dumps(dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)))
    assert {p['k'] for p in collect(tmp_path)}=={2,3}
    score,_,criterion=validation_score_from_reports(tmp_path,20,lambda_action=1,lambda_future=9,evaluation_k=3)
    assert score==pytest.approx(2.) and 'K=3' in criterion
    points=[dict(split='train',direction='action',k=3,update=u,mse=m) for u,m in [(0,1.),(20,11.)]]
    assert fixed_mse_failure(points,evaluation_k=3)
    assert fixed_mse_failure(points) is None


def test_shared_gpu_consent_keeps_exclusive_gpu_and_free_floor(monkeypatch):
    def query(argv,**kwargs):
        q=' '.join(argv)
        if 'memory.free' in q:return '1, GPU-a, 32212\n8, GPU-b, 33000\n'
        if 'query-compute-apps' in q:return 'GPU-b, 77\n'
        return '1, GPU-a\n8, GPU-b\n'
    monkeypatch.setattr('subprocess.check_output',query)
    assert not gpu_idle('1,8')
    assert gpu_idle('1,8',dict(indices=['8'],min_free_mib={'8':32000}))
    assert not gpu_idle('1,8',dict(indices=['8'],min_free_mib={'8':34000}))
