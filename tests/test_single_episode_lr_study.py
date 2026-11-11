import copy
from scripts import run_single_episode_lr_study as s


def arm(name,scale,a,z):
 return dict(name=name,scale=scale,terminal_success=True,fit_score=(a+z)/2,
             train_ratios={'action':a,'future':z},train_mse={'action':a,'future':z},folder=name)


def test_next_wave_moves_toward_lower_winner_and_retires_only_dominated(monkeypatch):
 retired=[];monkeypatch.setattr(s,'retire_weights',lambda folder,*args:retired.append(folder))
 scales,_=s.choose_next([arm('low',.3,.6,.8),arm('high',1.,.9,.95)])
 assert scales==[.1,.6];assert retired==['high']


def test_tradeoff_keeps_both_weights(monkeypatch):
 retired=[];monkeypatch.setattr(s,'retire_weights',lambda folder,*args:retired.append(folder))
 scales,_=s.choose_next([arm('low',.3,.6,.9),arm('high',1.,.8,.7)])
 assert retired==[]


def test_both_fail_moves_lower_without_claiming_winner(monkeypatch):
 monkeypatch.setattr(s,'retire_weights',lambda *args:None)
 scales,reason=s.choose_next([{'name':'a','folder':'a','terminal_success':False},{'name':'b','folder':'b','terminal_success':False}])
 assert scales==[.1,.03];assert 'failed' in reason
