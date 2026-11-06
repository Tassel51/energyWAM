"""Portable preflight refuses mismatched numbering and invalid sample geometry."""
import hashlib,json
from pathlib import Path
import pytest
from scripts.prepare_remote_run import validate_dataset

def fixture(tmp_path):
 d=tmp_path/'robocasa365_target_50tasks';(d/'meta').mkdir(parents=True)
 (d/'meta/info.json').write_text(json.dumps({'robot_type':'PandaOmron'}))
 (d/'meta/modality.json').write_text('{}')
 eps=[{'episode_index':i,'length':40,'source_dataset':'atomic/OpenDrawer/20250816'} for i in [0,1]]
 (d/'meta/episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in eps))
 for i in [0,1]:
  for rel in [f'data/chunk-000/episode_{i:06d}.parquet',f'videos/chunk-000/observation.images.robot0_agentview_left/episode_{i:06d}.mp4']:
   p=d/rel;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
 m={'dataset_identity':'robocasa365_target_50tasks/PandaOmron','windows':[{'split':s,'episode_index':i,'frame_index':0,'source_dataset':eps[i]['source_dataset']} for s,i in [('train',0),('val',1)]]}
 f={'source_meta_sha256':{n:hashlib.sha256((d/'meta'/n).read_bytes()).hexdigest() for n in ['info.json','modality.json','episodes.jsonl']}}
 return d,m,f

def test_valid_portable_dataset(tmp_path):
 d,m,f=fixture(tmp_path);assert validate_dataset(d,m,f)=={'train':1,'val':1}

def test_changed_global_numbering_rejected(tmp_path):
 d,m,f=fixture(tmp_path);(d/'meta/episodes.jsonl').write_text('{}\n')
 with pytest.raises(ValueError,match='metadata differs'):validate_dataset(d,m,f)

def test_invalid_future_horizon_rejected(tmp_path):
 d,m,f=fixture(tmp_path);m['windows'][0]['frame_index']=24
 with pytest.raises(ValueError,match='Window invalid'):validate_dataset(d,m,f)

def test_trajectory_leakage_rejected(tmp_path):
 d,m,f=fixture(tmp_path);m['windows'][1]['episode_index']=0
 with pytest.raises(ValueError,match='leakage'):validate_dataset(d,m,f)

def test_broken_video_symlink_rejected(tmp_path):
 d,m,f=fixture(tmp_path);p=d/'videos/chunk-000/observation.images.robot0_agentview_left/episode_000000.mp4';p.unlink();p.symlink_to(tmp_path/'missing.mp4')
 with pytest.raises(FileNotFoundError,match='Missing/broken'):validate_dataset(d,m,f)
