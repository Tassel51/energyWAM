"""Make a frozen, episode-disjoint two-task fitting + held-out diagnostic set."""
import json,random
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
source=json.loads(Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103/sample_manifest.json').read_text())
cfg=load_config(ROOT/'configs/experiments/iteration_v1_iter_learned.yaml')
reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
train_ids=set(json.loads((ROOT/'manifests/train_episodes.json').read_text())['episode_indices'])
val_ids=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices'])
rng=random.Random(20261104);windows=[]
for task in ['OpenDrawer','OpenCabinet']:
    eps=list(dict.fromkeys(w['episode_index'] for w in source['windows'] if w['task']==task))
    ep=eps[0]
    assert ep in train_ids
    windows += [dict(w) for w in source['windows'] if w['task']==task and w['episode_index']==ep]
    pool=[ep for ep in reader.episodes if ep['episode_index'] in val_ids and ep.get('source_dataset','').split('/')[-2]==task and ep['length']>=20]
    for val in rng.sample(pool,2):
        for j in range(4):
            windows.append(dict(window_id=1000+len(windows),split='val',task=task,episode_index=val['episode_index'],
                frame_index=round(j*(val['length']-17)/3),source_dataset=val['source_dataset']))
assert len([w for w in windows if w['split']=='train'])==16
assert len([w for w in windows if w['split']=='val'])==16
assert {w['episode_index'] for w in windows if w['split']=='train'}.isdisjoint({w['episode_index'] for w in windows if w['split']=='val'})
p=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104/tinyfit_manifest.json')
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(dict(dataset_identity=source['dataset_identity'],windows=windows,
    note='16 fixed train samples from 2 train episodes; 16 fixed val samples from 4 independent val episodes, 2 tasks; no test; old train-only stats reused'),indent=2))
print(p)
