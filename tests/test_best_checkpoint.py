import json
import os
import torch
import pytest
from energywam.training.run_state import save_latest
from energywam.training.best_checkpoint import promote_latest_to_best

class Writer:
 def __init__(self,folder):self.folder=folder;self.step=0
 def save(self,name):
  p=self.folder/name
  torch.save(dict(model={'w':torch.tensor([self.step])},optimizer={'moment':self.step},scheduler={'last_epoch':self.step},rng={'seed':42},config={'seed':42},format_version=1,meta={'global_step':self.step}),p)
  p.with_name(p.name+'.meta.json').write_text(json.dumps({'meta':{'global_step':self.step}}));return p

def promote(folder,step,score):return promote_latest_to_best(folder,global_step=step,score=score,metrics={'val_action':score/2,'val_future':score/2},criterion='fixed-validation-test-score')

def test_best_preserves_full_state_after_latest_rolls_and_keeps_only_one_best(tmp_path):
 w=Writer(tmp_path);save_latest(w,tmp_path);assert promote(tmp_path,0,2.)
 assert os.path.samefile(tmp_path/'ckpt_best.pt',tmp_path/'ckpt_latest.pt')
 w.step=1;save_latest(w,tmp_path);assert not promote(tmp_path,1,3.)
 best=torch.load(tmp_path/'ckpt_best.pt',weights_only=False);assert best['meta']['global_step']==0 and best['optimizer']['moment']==0
 w.step=2;save_latest(w,tmp_path);assert promote(tmp_path,2,1.)
 assert torch.load(tmp_path/'ckpt_best.pt',weights_only=False)['scheduler']['last_epoch']==2
 assert len(list((tmp_path/'.best_snapshots').iterdir()))==1
 assert json.loads((tmp_path/'best_selection.json').read_text())['global_step']==2
 assert len((tmp_path/'best_history.jsonl').read_text().splitlines())==2

def test_failure_keeps_previous_best_and_latest(tmp_path,monkeypatch):
 w=Writer(tmp_path);save_latest(w,tmp_path);promote(tmp_path,0,2.);w.step=1;save_latest(w,tmp_path)
 monkeypatch.setattr('energywam.training.best_checkpoint.os.link',lambda *a: (_ for _ in ()).throw(OSError('no link')))
 monkeypatch.setattr('energywam.training.best_checkpoint.shutil.copyfile',lambda *a: (_ for _ in ()).throw(OSError('disk full')))
 with pytest.raises(OSError):promote(tmp_path,1,1.)
 assert torch.load(tmp_path/'ckpt_best.pt',weights_only=False)['meta']['global_step']==0
 assert torch.load(tmp_path/'ckpt_latest.pt',weights_only=False)['meta']['global_step']==1
 assert len(list((tmp_path/'.best_snapshots').iterdir()))==1

def test_rejects_wrong_step_nonfinite_and_changed_selection_criterion(tmp_path):
 w=Writer(tmp_path);save_latest(w,tmp_path)
 with pytest.raises(ValueError):promote(tmp_path,1,1.)
 with pytest.raises(ValueError):promote(tmp_path,0,float('nan'))
 promote(tmp_path,0,2.)
 with pytest.raises(ValueError):promote_latest_to_best(tmp_path,global_step=0,score=1,metrics={'a':1},criterion='changed')


def test_interrupted_commit_keeps_readable_best_and_recovers_orphan(tmp_path,monkeypatch):
 import energywam.training.best_checkpoint as module
 w=Writer(tmp_path);save_latest(w,tmp_path);promote(tmp_path,0,2.)
 w.step=1;save_latest(w,tmp_path);original=module._pointer
 def interrupted(path,target):
  original(path,target)
  if path.name=='.best_current':raise OSError('simulated interruption immediately after commit')
 monkeypatch.setattr(module,'_pointer',interrupted)
 with pytest.raises(OSError):promote(tmp_path,1,1.)
 assert torch.load(tmp_path/'ckpt_best.pt',weights_only=False)['meta']['global_step']==1
 monkeypatch.setattr(module,'_pointer',original)
 assert not promote(tmp_path,1,1.5)
 assert len(list((tmp_path/'.best_snapshots').iterdir()))==1


def test_validation_selection_uses_only_matched_val_and_declared_weights(tmp_path):
 from energywam.training.best_checkpoint import validation_score_from_reports
 for direction,mse in [('action',.8),('future',.1)]:
  rows=[dict(window_id=1,episode_index=2,frame_index=3,split='val',k=2,update=10,prediction_mse=mse),dict(window_id=4,episode_index=5,frame_index=6,split='train',k=2,update=10,prediction_mse=999.)]
  (tmp_path/f'eval_0010_{direction}.json').write_text(json.dumps(dict(rows=rows,weight_updates_during_eval=0,evaluation_seed=2026)))
 score,metrics,criterion=validation_score_from_reports(tmp_path,10,lambda_action=1,lambda_future=9)
 assert score==pytest.approx(1.7) and metrics['val_future_mse']==.1
 assert 'lambda_future=9' in criterion
 p=tmp_path/'eval_0010_future.json';d=json.loads(p.read_text());d['rows'][0]['episode_index']=7;p.write_text(json.dumps(d))
 with pytest.raises(ValueError):validation_score_from_reports(tmp_path,10,lambda_action=1,lambda_future=9)
