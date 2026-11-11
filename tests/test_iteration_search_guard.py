from scripts.run_iteration_search import fixed_mse_failure
from scripts.run_iteration_search import fixed_mse_guard_decision
import pytest

def test_explicit_warning_preserves_default_stop():
    points=[point(0,.1),point(80,1.7)]
    rejection,warning=fixed_mse_guard_decision(points,{})
    assert rejection and warning is None
    rejection,warning=fixed_mse_guard_decision(points,{'fixed_mse_guard_mode':'warn'})
    assert rejection is None and 'update 80' in warning
    assert fixed_mse_guard_decision([point(0,.1),point(80,.2)],{'fixed_mse_guard_mode':'warn'})==(None,None)

@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_warning_mode_still_stops_nonfinite(value):
    rejection,warning=fixed_mse_guard_decision([point(0,.1),point(80,value)],{'fixed_mse_guard_mode':'warn'})
    assert 'Nonfinite' in rejection and warning is None

def test_invalid_guard_mode_rejected():
    with pytest.raises(ValueError):fixed_mse_guard_decision([],{'fixed_mse_guard_mode':'typo'})

def point(update,mse,split='train',k=2):
    return dict(update=update,mse=mse,split=split,k=k,direction='future')

def test_fixed_guard_uses_initial_prediction_and_excludes_other_splits():
    assert fixed_mse_failure([point(0,.1),point(20,.09)]) is None
    assert fixed_mse_failure([point(0,.1),point(20,9.,split='val')]) is None
    assert fixed_mse_failure([point(0,.1),point(20,9.,k=0)]) is None
    assert fixed_mse_failure([point(0,.1),point(10,9.)]) is None
    assert 'future' in fixed_mse_failure([point(0,.1),point(20,9.)])


def test_sample_guard_compares_equal_visits_across_batch_sizes():
    for batch in [8,16,32,64]:
        update=256//batch
        assert fixed_mse_failure([point(0,.1),point(update-1,9.)],effective_batch=batch,min_sample_visits=256) is None
        assert 'future' in fixed_mse_failure([point(0,.1),point(update,9.)],effective_batch=batch,min_sample_visits=256)
