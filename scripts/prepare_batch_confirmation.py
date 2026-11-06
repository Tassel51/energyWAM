"""Freeze64 new validation windows before selecting the final batch recommendation."""
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
R=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
small=json.loads((R/'tinyfit_manifest.json').read_text());approved=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices'])
old={w['episode_index'] for w in small['windows']};tasks=sorted({w['task'] for w in small['windows'] if w['split']=='train'})
cfg=load_config(ROOT/'configs/experiments/batch2_tiny16_samples1600_seed42.yaml');reader=LeRobotEpisodeReader(resolve_dataset_root(cfg));rng=random.Random(20261105)
windows=[dict(w) for w in small['windows'] if w['split']=='train']
for task in tasks:
 pool=[e for e in reader.episodes if e['episode_index'] in approved and e['episode_index'] not in old and e.get('source_dataset','').split('/')[-2]==task and e['length']>=20]
 assert len(pool)>=8
 for ep in sorted(rng.sample(pool,8),key=lambda e:e['episode_index']):
  for j in range(4):windows.append(dict(window_id=40000+len(windows),split='val',task=task,episode_index=ep['episode_index'],frame_index=round(j*(ep['length']-17)/3),source_dataset=ep['source_dataset']))
val=[w for w in windows if w['split']=='val'];ids={w['episode_index'] for w in val};assert len(val)==64 and len(ids)==16 and ids<=approved and ids.isdisjoint(old)
data=dict(dataset_identity=small['dataset_identity'],windows=windows,note='Frozen confirmation:8 new approved validation episodes per task x4 windows=64; excludes original training and validation episodes. Original16 training windows retained only for reference baselines. No test used, no normalization fitting. Candidate models still train on original16 windows.')
p=R/'batch_confirmation_manifest.json'
if p.exists():assert json.loads(p.read_text())==data
else:p.write_text(json.dumps(data,indent=2))
print(p,'16 new validation episodes,64 windows; training unchanged')
