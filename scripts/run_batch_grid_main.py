"""Two bounded batch-comparison rounds, guarded by recorded real-model gates.

This controller never runs LR searches, deletes weights or starts Stage2.
It preserves the complete experiment scope for subsequent evidence-based tuning.
"""
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=Path('/cloud/zzc/energyWAM/experiments/batch_grid_micro4_20261105')

def write(state):
    path=R/'main_controller_status.json';pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(dict(state,updated_unix=time.time()),indent=2));pending.replace(path)

def read(path):return json.loads(Path(path).read_text())

def wait_gate(path,key,label,supervisor_name):
    while True:
        if Path(path).exists() and read(path).get(key):return
        status=subprocess.run(['supervisorctl','status',supervisor_name],capture_output=True,text=True).stdout
        if any(x in status for x in ['EXITED','FATAL','STOPPED']):raise RuntimeError(f'{label} did not pass; {status.strip()}')
        write(dict(phase='waiting_for_preflight',gate=label));time.sleep(15)

def run_queue(jobs,status,log):
    stream=Path(log).open('a')
    child=subprocess.Popen([sys.executable,'-u','scripts/run_iteration_search.py','--jobs',str(jobs),'--status',str(status)],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    return child,stream

def assert_queue(status,expected):
    rows=read(status)
    for name,updates in expected.items():
        row=rows.get(name,{})
        if row.get('phase')!='complete' or row.get('returncode')!=0 or row.get('completed_updates')!=updates:
            raise RuntimeError(f'Queue arm failed or incomplete: {name}: {row.get("phase")} updates={row.get("completed_updates")}')

def main():
    audit=read(R/'data_audit.json')
    for name,digest in audit['files_sha256'].items():
        assert hashlib.sha256((R/name).read_bytes()).hexdigest()==digest,'frozen data changed'
    assert audit['trajectory_disjoint'] and not audit['test_used']
    wait_gate(R/'gradient_gates/fp32_dedup10/status.json','gate_passed','FP32 storage/native gradient','energywam-math-dedup')
    wait_gate(R/'fp32_full_resume_retention_smoke/resume_smoke_result.json','passed','full-resume latest/best smoke','energywam-resume-smoke')
    # Profile all ten tasks' longest selected TRAIN instructions before training.
    profile=read(R/'longest_instruction_capacity_jobs.json')[0]
    status=R/'longest_instruction_capacity_status.json'
    write(dict(phase='preflight_capacity_running',gate='ten-task longest selected TRAIN instructions'))
    child,stream=run_queue(R/'longest_instruction_capacity_jobs.json',status,R/'longest_instruction_controller.log')
    child.wait();stream.close();assert_queue(status,{profile['name']:profile['steps']})
    wait_gate(R/'accumulation_gate/gradient_check.json','passed','effective8/16/32/64 full gradients','energywam-accumulation-gate')
    assert len(read(R/'accumulation_gate/gradient_check.json')['comparisons'])==8
    all_jobs={pair:read(R/f'main_jobs_{pair}.json') for pair in ['01','23']}
    for round_index in [0,1]:
        # Worst-case two distinct full snapshots per newly retained arm, plus
        # one serialized pending write. Never assume best will equal latest.
        size=read(R/'checkpoint_disk_budget.json')['snapshot_bytes_measured']
        required=5*size+4*2**30
        round_jobs={pair:items[round_index] for pair,items in all_jobs.items()}
        retained={}
        for job in round_jobs.values():
            for name in ['ckpt_latest.pt','ckpt_best.pt']:
                path=R/job['name']/name
                if path.exists():
                    stat=path.stat();retained[(stat.st_dev,stat.st_ino)]=stat.st_size
        required=max(0,4*size-sum(retained.values()))+size+4*2**30
        while shutil.disk_usage(R).free<required:
            write(dict(phase='waiting_for_checkpoint_space',round=round_index+1,free_gib=shutil.disk_usage(R).free/2**30,required_gib=required/2**30,note='No latest/best snapshots automatically deleted'))
            time.sleep(30)
        children=[]
        for pair,job in round_jobs.items():
            jobs=R/f'main_round{round_index+1}_jobs_{pair}.json';jobs.write_text(json.dumps([job],indent=2))
            status=R/f'main_round{round_index+1}_status_{pair}.json'
            child,stream=run_queue(jobs,status,R/f'main_round{round_index+1}_controller_{pair}.log')
            children.append((child,stream,status,job))
        write(dict(phase='main_round_running',round=round_index+1,arms=[j['name'] for j in round_jobs.values()],pids=[c.pid for c,_,_,_ in children]))
        while any(c.poll() is None for c,_,_,_ in children):time.sleep(15)
        for child,stream,status,job in children:
            stream.close();assert_queue(status,{job['name']:job['steps']})
        # Both arms must have the same initialized parameters/statistics.
        initialized=[read(R/j['name']/'initialization_step0.json') for j in round_jobs.values()]
        assert len({x['sampled_parameter_sha256'] for x in initialized})==1
        assert len({x['stats_fingerprint'] for x in initialized})==1
    all_initial=[read(R/j['name']/'initialization_step0.json') for items in all_jobs.values() for j in items]
    assert len({x['sampled_parameter_sha256'] for x in all_initial})==1
    write(dict(phase='main_four_arms_complete',next='Assess both validation directions and finite matched-LR probes; this is not the final batch recommendation'))

if __name__=='__main__':
    try:main()
    except Exception as error:
        write(dict(phase='preflight_or_main_failed',error_type=type(error).__name__,reason=str(error)));raise
