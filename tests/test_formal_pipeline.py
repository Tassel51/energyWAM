"""Exercise stage handoff and restart without allocating a model or GPU."""
import importlib.util
from pathlib import Path
import json
import os
import yaml


def test_bi_only_run_and_completed_restart(tmp_path,monkeypatch):
    source=Path(__file__).parents[1]/'scripts/run_gpu23_formal.py'
    spec=importlib.util.spec_from_file_location('formal_pipeline',source)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.os.path,'ismount',lambda path: True)
    monkeypatch.setattr(mod,'ROOT',tmp_path)
    monkeypatch.setattr(mod,'STATUS',tmp_path/'logs/status.json')
    monkeypatch.chdir(tmp_path);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2,3')
    (tmp_path/'configs').mkdir();(tmp_path/'logs').mkdir()
    for stage in ['bi','joint']:
        cfg={'trainer':{'output_dir':f'logs/runs/energywam_formal_{stage}_20261101','max_train_steps':2000 if stage=='bi' else 1000}}
        (tmp_path/f'configs/gpu23_formal_{stage}_20261101.yaml').write_text(yaml.safe_dump(cfg))
    calls=[]
    def run(args,log):
        calls.append(args)
        cfg=yaml.safe_load(Path(args[args.index('--config')+1]).read_text())
        folder=Path(cfg['trainer']['output_dir']);folder.mkdir(parents=True,exist_ok=True)
        if args[0].endswith('train.py'):
            if '--init-from' in args:assert Path(args[args.index('--init-from')+1]).exists()
            (folder/'ckpt_latest.pt').write_text('test checkpoint')
        else:Path(args[args.index('--out')+1]).write_text('{}')
    monkeypatch.setattr(mod,'run',run)
    mod.main()
    assert len(calls)==4
    assert Path('logs/runs/energywam_formal_bi_20261101/ckpt_latest.pt').exists()
    assert not Path('logs/runs/energywam_formal_joint_20261101/ckpt_latest.pt').exists()
    calls.clear();mod.main();assert not calls
    assert json.loads(mod.STATUS.read_text())['phase']=='bi_complete_joint_disabled'
