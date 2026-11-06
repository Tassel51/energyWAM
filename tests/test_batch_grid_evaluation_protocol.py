import pytest
from scripts.evaluate_batch_grid_checkpoint import validate_val_manifest

def test_confirmation_accepts_only_disjoint_canonical_validation_episodes():
    rows=[dict(split='val',window_id=1,episode_index=11,frame_index=0),dict(split='val',window_id=2,episode_index=11,frame_index=16)]
    validate_val_manifest({'windows':rows},{11,12},{1,2})
    with pytest.raises(ValueError,match='noncanonical'):validate_val_manifest({'windows':rows},{12},{1,2})
    with pytest.raises(ValueError,match='leakage'):validate_val_manifest({'windows':rows},{11,12},{1,11})
    with pytest.raises(ValueError,match='VAL-only'):validate_val_manifest({'windows':[dict(rows[0],split='train')]},{11},{1})
    with pytest.raises(ValueError,match='VAL-only'):validate_val_manifest({'windows':[dict(rows[0],split='test')]},{11},{1})

def test_duplicate_frames_and_window_ids_cannot_inflate_confirmation_sample_count():
    row=dict(split='val',window_id=1,episode_index=11,frame_index=0)
    for other in [dict(row,window_id=2),dict(row,frame_index=3)]:
        with pytest.raises(ValueError,match='duplicate'):validate_val_manifest({'windows':[row,other]},{11},{1})
