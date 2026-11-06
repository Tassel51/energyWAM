"""An explicit 'val' label cannot by itself prove held-out evaluation."""
from types import SimpleNamespace
from scripts.evaluate import describe_eval_split
from energywam.data.standardization import StatsError
from .harness import assert_true

def reference():
    return SimpleNamespace(provenance={'train_episodes':{'episode_indices':[0,1]}})

def test_disjoint_manifest_is_verified_against_training():
    out=describe_eval_split({'split':'val'},[2,3],reference())
    assert_true(out['guaranteed_heldout'] and out['train_overlap_checked'])

def test_mislabeled_validation_overlapping_training_is_rejected():
    try:describe_eval_split({'split':'val'},[1,2],reference())
    except StatsError as exc:
        assert_true('overlap' in str(exc));return
    raise AssertionError('a val label accepted despite known training episode overlap')

def test_missing_training_reference_or_train_eval_cannot_claim_heldout():
    assert_true(not describe_eval_split({'split':'test'},[2,3],None)['guaranteed_heldout'])
    assert_true(not describe_eval_split({'split':'train'},[0],reference())['guaranteed_heldout'])

def test_nonfinite_statistical_metadata_is_rejected():
    import numpy as np
    from energywam.data.standardization import ZScoreStats
    for field, bad in [('count', np.nan), ('count', np.inf), ('epsilon', np.nan), ('epsilon', np.inf)]:
        kwargs = dict(mean=np.array([0.]), std=np.array([1.]), count=np.array([2.]), epsilon=1e-6)
        kwargs[field] = np.array([bad]) if field == 'count' else bad
        try:
            ZScoreStats(**kwargs)
        except StatsError:
            continue
        raise AssertionError(f'nonfinite {field} accepted')

TESTS=[('nonfinite_statistical_metadata_rejected',test_nonfinite_statistical_metadata_is_rejected),('disjoint_manifest_verified',test_disjoint_manifest_is_verified_against_training),
       ('mislabeled_val_overlap_rejected',test_mislabeled_validation_overlapping_training_is_rejected),
       ('missing_train_reference_not_heldout',test_missing_training_reference_or_train_eval_cannot_claim_heldout)]
