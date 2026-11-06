"""Regression: official RoboCasa short camera aliases must resolve to feature paths."""
import json,tempfile
from pathlib import Path
from energywam.data.episode_dataset import LeRobotEpisodeReader

def test_official_camera_alias_and_feature_name():
 with tempfile.TemporaryDirectory() as tmp:
  root=Path(tmp);(root/'meta').mkdir()
  info={'robot_type':'PandaOmron','video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
  modality={'video':{name:{'original_key':'observation.images.'+name} for name in ['robot0_agentview_left','robot0_agentview_right']}}
  for name,data in [('info',info),('modality',modality),('stats',{})]:(root/'meta'/f'{name}.json').write_text(json.dumps(data))
  (root/'meta/episodes.jsonl').write_text(json.dumps({'episode_index':0,'length':32,'tasks':['Open the drawer.']})+'\n')
  paths=[]
  for camera in modality['video']:
   path=root/'videos/chunk-000'/modality['video'][camera]['original_key']/'episode_000000.mp4'
   path.parent.mkdir(parents=True);path.touch();paths.append(path)
  reader=LeRobotEpisodeReader(root)
  for alias,path in zip(modality['video'],paths):
   assert reader.video_path(0,alias)==path
   assert reader.video_path(0,modality['video'][alias]['original_key'])==path
  assert reader.video_path(0,'robot0_agentview_left')!=reader.video_path(0,'robot0_agentview_right')
  try:reader.video_path(0,'robot0_unknown_camera')
  except FileNotFoundError:pass
  else:raise AssertionError('Unknown camera silently selected another view')
TESTS=[('official RoboCasa alias selects exact camera feature path',test_official_camera_alias_and_feature_name)]
