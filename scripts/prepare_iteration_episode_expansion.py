"""Freeze a 10-train / 10-validation episode protocol, no test access."""
import json,random,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from energywam.config import load_config
from energywam.data.energy_dataset import LeRobotEpisodeReader,resolve_dataset_root
OUT=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104')
def main():
 source=json.loads(Path('/cloud/zzc/energyWAM/experiments/overfit_k24_alpha200_20261103/sample_manifest.json').read_text())
 small=json.loads((OUT/'tinyfit_manifest.json').read_text())
 cfg=load_config(ROOT/'configs/experiments/iteration_tinyfit_master_prior_tokenscale_r2.yaml')
 reader=LeRobotEpisodeReader(resolve_dataset_root(cfg))
 train_ids=set(json.loads((ROOT/'manifests/train_episodes.json').read_text())['episode_indices'])
 val_ids=set(json.loads((ROOT/'manifests/val_episodes.json').read_text())['episode_indices'])
 rng=random.Random(20261104);windows=[dict(w) for w in source['windows'] if w['split']=='train'];tasks=list(dict.fromkeys(w['task'] for w in windows))
 for task in tasks:
  selected=list(dict.fromkeys(w['episode_index'] for w in small['windows'] if w['split']=='val' and w['task']==task))
  pool=[ep for ep in reader.episodes if ep['episode_index'] in val_ids and ep.get('source_dataset','').split('/')[-2]==task and ep['length']>=20]
  records={ep['episode_index']:ep for ep in pool}
  if not selected:selected=[ep['episode_index'] for ep in rng.sample(pool,2)]
  assert len(selected)==2
  for eid in selected:
   ep=records[eid]
   for j in range(8):windows.append(dict(window_id=2000+len(windows),split='val',task=task,episode_index=eid,frame_index=round(j*(ep['length']-17)/7),source_dataset=ep['source_dataset']))
 tr={w['episode_index'] for w in windows if w['split']=='train'};va={w['episode_index'] for w in windows if w['split']=='val'}
 assert len(tr)==len(va)==10 and tr<=train_ids and va<=val_ids and tr.isdisjoint(va)
 assert sum(w['split']=='train' for w in windows)==sum(w['split']=='val' for w in windows)==80
 data=dict(dataset_identity=source['dataset_identity'],windows=windows,note='5 tasks x 2 train episodes x 8 starts =80 fixed train samples; 10 disjoint validation episodes x8 starts=80 fixed held-out; no test. Reuse existing train-only 50-task stats; no validation fitting. Existing 2-task validation episodes retained, denser starts and new fixed candidate IDs.')
 p=OUT/'episode10_manifest.json'
 if p.exists():assert json.loads(p.read_text())==data
 else:p.write_text(json.dumps(data,indent=2))
 print(p, tasks, len(tr),len(va))
if __name__=='__main__':main()
