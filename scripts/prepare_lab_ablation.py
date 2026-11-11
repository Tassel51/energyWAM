"""Prepare authorized lab weight/inner-step studies; no GPU computation."""
import copy
import json
from pathlib import Path
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
STATE=ROOT/'logs/lab_remote_setup'
SNAPSHOT=24569214417


def main():
    code=json.loads((STATE/'latest_code_release.json').read_text())['code_dir']
    env='/data7/zzc/energyWAM/lab_execution_20261109/environments/energywam_py310_v2/bin/python'
    shared='/data7/zzc/energyWAM/lab_execution_20261109'
    base=yaml.safe_load((ROOT/'configs/experiments/episode3346_lrscale1_b16_micro4_seed42.yaml').read_text())
    base['backbone'].update(base_vlm='/data6/zzc/Qwen3-VL-4B-Instruct',dino_path=shared+'/assets/dino_source_config')
    base['data'].update(data_root_dir=shared+'/assets',norm_stats_bundle_path=shared+'/assets/relocated_stats.json')
    base['framework'].update(quadratic_action_weight=1.,quadratic_future_weight=1.)
    out=STATE/'ablation_preparation';out.mkdir(exist_ok=True)
    groups=[dict(name='loss',root=shared+'/studies/future_weight_20261109',pair='3,4',budgets=[0.,0.],factors=[('weight9_control',9.,5.,2000.),('weight1',1.,5.,2000.)]),
            dict(name='steps',root='/data6/zzc/energywam_lab_studies/inner_steps_20261109',pair='6,7',budgets=[10.,14.],factors=[('steps_control',9.,5.,2000.),('action_half',9.,2.5,2000.),('future_half',9.,5.,1000.)])]
    uuids={'3':'GPU-a5804821-42d9-600f-402c-0d7df8e5658c','4':'GPU-3abf44c3-2344-a902-ce41-df1fc58393e1','6':'GPU-9a1246f3-2243-0b77-9411-7de2c9e1ea9f','7':'GPU-4de65a87-f820-da66-782e-0aab5d22e243'}
    files={};plans=[];gate=groups[1]['root']+'/shared_gradient_gate/result.json'
    for group in groups:
        root=group['root'];policy=root+'/gpu_policy.json'
        files[policy]=json.dumps(dict(hostname='user-SYS-420GP-TNR',allowed_physical_pairs=[group['pair']],use_uuid_visibility=True,expected_gpu_uuids={i:uuids[i] for i in group['pair'].split(',')},reason='User2026-11-09 explicitly assigned these GPUs to this ablation'),indent=2)+'\n'
        jobs=[];arms=[]
        for tag,weight,a,z in group['factors']:
            cfg=copy.deepcopy(base);name='episode3346_'+tag+'_seed42';folder=root+'/'+name
            cfg['run_id']=name;cfg['notes']='Lab controlled ablation; fresh same official4B/seed42,67TRAIN/16auxVAL,B16/native4/K2/FP32,210updates. Same-hardware control. Only declared factor changes; quadratic prior effective coefficients held constant for step tests.'
            cfg['trainer'].update(output_dir=folder,max_train_steps=210,save_interval=20,checkpoint_interval_seconds=1800,saved_tensor_gpu_budgets_gib=group['budgets'])
            cfg['loss']['lambda_future']=weight;cfg['solver'].update(alpha_action=a,alpha_future=z)
            cfg['framework'].update(quadratic_action_weight=a/5.,quadratic_future_weight=z/2000.)
            assert cfg['framework']['quadratic_action_weight']/a==.2
            assert cfg['framework']['quadratic_future_weight']/z==.0005
            config=root+'/configs/'+name+'.yaml';files[config]=yaml.safe_dump(cfg,sort_keys=False)
            factors=dict(lambda_future=weight,initial_alpha_action=a,initial_alpha_future=z,effective_prior_action_precision=.2,effective_prior_future_precision=.0005,learnable_range_action=[a*.5,a*1.5],learnable_range_future=[z*.5,z*1.5])
            job=dict(name=name,config=config,physical_gpus=group['pair'],execution_gpu_policy=policy,steps=210,manifest=code+'/manifests/experiments/single_episode_lr_20261108/train_val_manifest.json',train_monitor_manifest=code+'/manifests/experiments/single_episode_lr_20261108/train_monitor_manifest.json',sample_eval_interval=320,train_monitor_sample_interval=320,alternate_final_eval_seed=2027,keep_best=True,monitor_resources=True,effective_batch=16,guard_min_sample_visits=320,max_fixed_mse_ratio=10,max_process_uss_gib=120,min_system_available_gib=24,max_wall_seconds=43200,snapshot_bytes_estimate=SNAPSHOT)
            jobs.append(job);arms.append(dict(name=name,folder=folder,factors=factors))
            files[folder+'/arm_definition.json']=json.dumps(dict(hyperparameters=cfg,job=job,factors=factors,initialization='Fresh official Qwen plus new modules seed42; no trained checkpoint',best_criterion=f'Aux16VAL Action+{weight:g}Future, distinct from fit comparison'),indent=2)+'\n'
        profile_cfg=copy.deepcopy(base);profile_cfg['run_id']='capacity_'+group['name'];profile_cfg['trainer'].update(output_dir=root+'/capacity_profile',saved_tensor_gpu_budgets_gib=group['budgets'])
        profile_config=root+'/configs/capacity.yaml';files[profile_config]=yaml.safe_dump(profile_cfg,sort_keys=False)
        profile=copy.deepcopy(jobs[0]);profile.update(name='capacity_'+group['name'],config=profile_config,steps=2,profile=True,keep_best=False,max_wall_seconds=1200)
        full_jobs=root+'/full_jobs.json';profile_jobs=root+'/profile_jobs.json'
        files[full_jobs]=json.dumps(jobs,indent=2)+'\n';files[profile_jobs]=json.dumps([profile],indent=2)+'\n'
        plan=dict(hostname='user-SYS-420GP-TNR',name=group['name'],root=root,code=code,python=env,gpu_pair=group['pair'],profile_jobs=profile_jobs,full_jobs=full_jobs,arms=arms,gradient_gate_result=gate,whole_study_checkpoint_budget_bytes=(2*len(arms)+1)*SNAPSHOT+8*2**30,purpose='Future loss weights1/9' if group['name']=='loss' else 'Action/Future inner initialization separately halved, effective quadratic priors unchanged',protocol='210updates=3360visits=50.15equivalent epochs per arm; fixed20update evaluations; no automatic extension',baseline='Original globalLR1/headLR1e-3, not a declared winner of unfinished cloud head-LR study',environment_caveat='Lab torch2.6/cu124 differs from cloud torch2.13/cu132; use same-pair lab controls, do not infer a hyperparameter effect from cross-host differences',execution_caveat='Only GPU-saved tensor cache differs between hardware groups, gradient gate required before full runs; no precision/batch/K/model reduction',retention='Full latest/best per arm, no automatic weight deletion',wandb='No remote credentials copied; authoritative raw logs retained locally on lab and fetched on demand',tmux_session='energywam_lab_'+group['name']+'_20261109')
        if group['name']=='steps':
            plan['gradient_gate_command']=[env,'-u',code+'/scripts/check_native_microbatch_gradients.py','--config',groups[0]['root']+'/configs/capacity.yaml','--manifest',jobs[0]['manifest'],'--output',gate,'--physical-gpus','6,7','--execution-gpu-policy',policy,'--precision','float32','--native-microbatch','4','--sample-indices','0,16,33,66','--math-mode','strict','--reference-gpu-budgets','10','14','--dedup-cpu','--record-all']
        files[root+'/plan.json']=json.dumps(plan,indent=2)+'\n';plans.append(plan)
    (out/'remote_files.json').write_text(json.dumps(files,indent=2)+'\n')
    (STATE/'ablation_deployment.json').write_text(json.dumps(dict(code=code,groups=plans),indent=2)+'\n')
    print(json.dumps(dict(prepared_only=True,groups=[dict(name=p['name'],pair=p['gpu_pair'],arms=len(p['arms']),checkpoint_budget_GiB=p['whole_study_checkpoint_budget_bytes']/2**30) for p in plans]),indent=2))


if __name__=='__main__':main()
