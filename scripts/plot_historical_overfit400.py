"""Plot every logged training batch and every actual fixed-set evaluation."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('/cloud/zzc/energyWAM/experiments/iteration_aware_v1_20261104/ablation_prior1_currentinit1_seed42'));p.add_argument('--output',type=Path,default=ROOT/'docs/figures');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 rows={}
 for line in (a.source/'train_log.jsonl').read_text().splitlines():
  x=json.loads(line);rows[int(x['global_step'])+1]=x
 assert set(rows)==set(range(1,401)),'Expected all400 measured updates'
 steps=np.arange(1,401);av=np.array([rows[int(s)]['bi/action_loss'] for s in steps]);fv=np.array([rows[int(s)]['bi/future_loss'] for s in steps]);assert np.isfinite(av).all() and np.isfinite(fv).all()
 evaluations=[]
 for path in sorted(a.source.glob('eval_*.json')):
  groups=collections.defaultdict(list)
  for x in json.loads(path.read_text())['rows']:
   if x['k']==2:groups[(x['update'],x['split'],x['direction'])].append(x['prediction_mse'])
  for (s,split,direction),values in groups.items():evaluations.append(dict(update=s,split=split,direction=direction,mse=float(np.mean(values)),samples=len(values)))
 assert len(evaluations)==84
 for name,fields,values in [('OVERFIT_400_BATCH_MSE.csv',['update','action_mse','future_mse','weighted_total_loss'],[dict(update=int(s),action_mse=float(av[s-1]),future_mse=float(fv[s-1]),weighted_total_loss=rows[int(s)]['loss']) for s in steps]),('OVERFIT_400_FIXED_EVALUATION.csv',['update','split','direction','mse','samples'],evaluations)]:
  with (a.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(values)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','axes.grid':True,'grid.alpha':.18,'savefig.facecolor':'white'})
 blue='#2874A6';orange='#D97924';red='#B83A3A'
 fig,axes=plt.subplots(2,2,figsize=(15,9.3));fig.subplots_adjust(top=.85,bottom=.18,hspace=.4,wspace=.24)
 fig.suptitle('Historical small-set overfit experiment: all 400 updates',fontsize=19,y=.975,weight='bold')
 fig.text(.5,.925,'16 fixed TRAIN windows / 2 episodes | 16 VAL windows / 4 separate episodes | B=2 | K=2 | Future weight=9',ha='center',fontsize=10)
 fig.text(.5,.9,'Future initialized from current DINO; dual quadratic priors; pack2. All MSE values below are unweighted.',ha='center',fontsize=10,color='#555555')
 for col,(name,v) in enumerate([('Action',av),('Future',fv)]):
  ax=axes[0,col];ax.plot(steps,v,color=blue,alpha=.43,lw=.8,marker='.',markersize=2,label='Every measured batch (400 points)')
  ma=np.convolve(v,np.ones(16)/16,mode='valid');ax.plot(steps[15:],ma,color=red,lw=2,label='Trailing 16-update mean')
  ax.set(title=f'{name}: per-update TRAIN batch MSE',xlabel='Update number (batch loss before its optimizer step)',ylabel='Standardized MSE',xlim=(0,405),ylim=(0,float(v.max())*1.08));ax.legend(loc='upper right',fontsize=8)
 for col,direction in enumerate(['action','future']):
  ax=axes[1,col]
  for split,color in [('train',blue),('val',orange)]:
   points=sorted([x for x in evaluations if x['direction']==direction and x['split']==split],key=lambda x:x['update']);x=[v['update'] for v in points];y=[v['mse'] for v in points]
   ax.plot(x,y,'o-',color=color,lw=1.8,ms=3.5,label=f'Fixed {split.upper()} set (16 windows; every20 updates)')
   ax.axhline(y[0],color=color,ls=':',lw=.9,alpha=.7)
   change=(y[-1]/y[0]-1)*100;ax.annotate(f'{y[-1]:.5f} ({change:+.1f}%)',xy=(x[-1],y[-1]),xytext=(-5,10 if split=='val' else -17),textcoords='offset points',ha='right',color=color,fontsize=9,weight='bold')
  ax.set(title=f'{direction.title()}: actual fixed-set evaluations',xlabel='Completed weight updates',ylabel='Standardized MSE',xlim=(0,405));ax.margins(y=.15);ax.legend(loc='upper right' if direction=='action' else 'lower left',fontsize=8)
 fig.text(.06,.045,'Top: every raw batch is shown; batches contain different samples. Red smoothing is explicitly a moving mean.\nBottom: only the 21 real evaluation checkpoints are plotted and connected; no per-update evaluation was recorded.\nTRAIN fitting improved substantially late in training; VAL Future worsened. This is not proof of generalization.',fontsize=9,color='#444444')
 for suffix in ['png','pdf']:fig.savefig(a.output/f'OVERFIT_400_MSE_OVERVIEW.{suffix}',dpi=220)
 plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(20,8),sharey='row');fig.subplots_adjust(top=.86,bottom=.12,wspace=.12,hspace=.35)
 for col,lo in enumerate([1,101,201,301]):
  hi=lo+99;mask=(steps>=lo)&(steps<=hi)
  for row,(name,v) in enumerate([('Action',av),('Future',fv)]):
   ax=axes[row,col];ax.plot(steps[mask],v[mask],'-o',color=blue,lw=.9,ms=2.4)
   ax.set_title(f'{name}: updates {lo}-{hi}',fontsize=11);ax.set_xlabel('Update');ax.set_xlim(lo-2,hi+2);ax.set_ylim(0,v.max()*1.07)
   if col==0:ax.set_ylabel('Unweighted batch MSE')
 fig.suptitle('All400 measured TRAIN batches, expanded into100-update panels',fontsize=18,weight='bold',y=.97)
 fig.text(.5,.92,'No smoothing, no omitted spikes, no interpolated data. Each row shares the same y-axis range across all four panels.',ha='center',fontsize=11)
 fig.text(.06,.04,'B=2: the batch composition changes between updates. Use the fixed-set plots in the overview to judge fitting.',fontsize=10)
 for suffix in ['png','pdf']:fig.savefig(a.output/f'OVERFIT_400_MSE_EVERY_UPDATE.{suffix}',dpi=200)
 plt.close(fig)
 (a.output/'OVERFIT_400_PLOT_PROVENANCE.json').write_text(json.dumps({'source_run':str(a.source),'train_log_sha256':hashlib.sha256((a.source/'train_log.jsonl').read_bytes()).hexdigest(),'batch_points':400,'fixed_evaluation_checkpoints':21,'missing_updates':[],'smoothing':'Trailing16-update arithmetic mean only in overview top row; all raw400 points retained','fixed_eval_interpolation':'Lines connect recorded checkpoints every20updates; no artificial per-update metrics','final_evaluations':[x for x in evaluations if x['update']==400]},indent=2))
 print('Created overview/detail PNG+PDF, raw CSVs and provenance in',a.output)
if __name__=='__main__':main()
