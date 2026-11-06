from scripts.run_iteration_search import fixed_mse_failure

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
