import json
from scripts.watch_checkpoint_retention import observe_folder

def write_latest(folder,step):
    pending=folder/'pending';pending.write_bytes(f'weights{step}'.encode());pending.replace(folder/'ckpt_latest.pt')
    (folder/'ckpt_latest.pt.meta.json').write_text(json.dumps({'meta':{'global_step':step},'config':{'fixed':True}}))

def test_role_replacement_archives_metadata_without_changing_weights(tmp_path):
    write_latest(tmp_path,1);state,events=observe_folder(tmp_path,{})
    archived=state['latest']['metadata_archive'];assert events==[]
    write_latest(tmp_path,2);new,events=observe_folder(tmp_path,state)
    assert json.load(open(archived))['meta']['global_step']==1
    assert events[0]['previous']['step']==1 and events[0]['current']['step']==2
    assert not events[0]['previous_inode_still_selected']
    assert (tmp_path/'ckpt_latest.pt').read_bytes()==b'weights2'
    assert observe_folder(tmp_path,new)[1]==[]

def test_superseded_best_generation_removal_keeps_its_archived_selection(tmp_path):
    import shutil
    write_latest(tmp_path,2)
    old=tmp_path/'.best_snapshots/old';old.mkdir(parents=True)
    (old/'checkpoint.pt').write_bytes(b'weights0')
    (old/'checkpoint.pt.meta.json').write_text(json.dumps({'meta':{'global_step':0}}))
    (old/'selection.json').write_text(json.dumps({'score':2.0}))
    (tmp_path/'ckpt_best.pt').symlink_to('.best_snapshots/old/checkpoint.pt')
    (tmp_path/'ckpt_best.pt.meta.json').symlink_to('.best_snapshots/old/checkpoint.pt.meta.json')
    (tmp_path/'best_selection.json').symlink_to('.best_snapshots/old/selection.json')
    state,_=observe_folder(tmp_path,{})
    for name in ['ckpt_best.pt','ckpt_best.pt.meta.json','best_selection.json']:(tmp_path/name).unlink()
    (tmp_path/'ckpt_best.pt').symlink_to('ckpt_latest.pt')
    (tmp_path/'ckpt_best.pt.meta.json').symlink_to('ckpt_latest.pt.meta.json')
    shutil.rmtree(old)
    _,events=observe_folder(tmp_path,state)
    event=next(x for x in events if x['role']=='best')
    assert event['previous_best_generation_weight_missing']
    from pathlib import Path
    archive=Path(event['previous']['metadata_archive']).parent
    assert json.loads((archive/'selection.json').read_text())['score']==2.0
    assert (tmp_path/'ckpt_latest.pt').read_bytes()==b'weights2'
