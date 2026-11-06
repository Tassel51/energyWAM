"""Read-only progress display for the two three-task runs."""
import json,statistics
from pathlib import Path
ROOT=Path('/cloud/zzc/energyWAM/experiments/three_task_full_20261106')
def read(path):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {}
def duration(seconds):
 hours=seconds/3600
 return f'{hours:.1f}小时' if hours<48 else f'{hours/24:.1f}天'
for seed,pair in [(42,'0,1'),(43,'2,3')]:
 name=f'three_task_b16_micro4_seed{seed}';folder=ROOT/name
 state=read(ROOT/f'status_{seed}.json').get(name,{})
 speed=read(folder/'speed_report.json');done=speed.get('completed_steps',0);total=state.get('target_updates',2798)
 fraction=min(done/total,1);bar='█'*int(fraction*30)+'░'*(30-int(fraction*30))
 rows=speed.get('steps',[]);times=[r['end_to_end_seconds'] for r in rows[-20:] if r.get('end_to_end_seconds',0)>0]
 print(f'GPU {pair} | seed{seed} | {state.get("phase","未启动")}')
 print(f'[{bar}] {fraction:6.2%}  {done}/{total} 次更新，{done*16:,}/{total*16:,} 个样本')
 if times:
  median=statistics.median(times)
  print(f'近{len(times)}步中位耗时 {median:.1f}秒/步 | 剩余纯训练约 {duration((total-done)*median)}（另加验证/保存耗时）')
 else:print('尚无训练测速：正在加载模型或进行第0步基线评估。')
 if state.get('reason'):print('原因：'+state['reason'])
 print()
