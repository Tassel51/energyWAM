"""Read-only status of the16-window Future-weight comparison."""
import json,statistics
from pathlib import Path
R=Path('/cloud/zzc/energyWAM/experiments/tiny_future_weight_20261106')
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
print('控制状态:',read(R/'controller_status.json').get('phase','未启动'))
for weight in [1,9]:
 name=f'tiny16_futureweight{weight}_b16_micro4_seed42';folder=R/name;state=read(R/'queue_status.json').get(name,{});speed=read(folder/'speed_report.json');done=speed.get('completed_steps',0);n=min(done,50);bar='█'*int(n/50*30)+'░'*(30-int(n/50*30))
 print(f'Future权重{weight}: [{bar}] {done}/50次更新（{done}遍16个样本） {state.get("phase","排队") }')
 for split,d in [('train',folder/'train_monitor'),('val',folder)]:
  vals={}
  for direction in ['action','future']:
   files=sorted(d.glob(f'eval_*_{direction}.json'))
   if files:
    rows=[x for x in read(files[-1]).get('rows',[]) if x.get('k')==2 and x.get('split')==split]
    if rows:vals[direction]=(rows[0]['update'],round(statistics.mean(x['prediction_mse'] for x in rows),6))
  if vals:print(' ',split,vals)
