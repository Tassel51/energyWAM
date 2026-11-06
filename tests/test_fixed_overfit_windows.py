def test_multiple_windows_in_one_episode_are_preserved(monkeypatch):
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    import k_training_probe as probe
    import energywam.data.energy_dataset as data
    captured={}
    def dataset(*args,**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()
    monkeypatch.setattr(data,'EnergyWAMEpisodeDataset',dataset)
    monkeypatch.setattr(data,'resolve_dataset_root',lambda cfg:'unused')
    monkeypatch.setattr(data,'LeRobotEpisodeReader',lambda root:SimpleNamespace(dataset_identity='fixture'))
    bundle=SimpleNamespace(action=SimpleNamespace(to_norm_stats=lambda:None),future=None,provenance={'dataset_identity':'fixture'})
    model=SimpleNamespace(stats_bundle=bundle,backbone=SimpleNamespace(dino=None))
    windows=[dict(split='train',episode_index=7,frame_index=i) for i in [10,20,30]]
    ds,actual=probe.fixed_dataset(None,model,dict(dataset_identity='fixture',windows=windows),'train')
    assert captured['episode_indices']==[7]
    assert ds._index==[(7,10),(7,20),(7,30)]
    assert actual==windows
