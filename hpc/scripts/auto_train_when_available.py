"""Bounded, singleton waiting supervisor for this authorized EnergyWAM trial.
Never changes Slurm limits; never cancels jobs; submits at most two 1-GPU jobs.
"""
import fcntl,json,os,subprocess,time,datetime,traceback
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path.home()/'energywam_storage';LOGS=ROOT/'logs'
STATUS=LOGS/'training_status.json';TZ=ZoneInfo('Asia/Shanghai')

def stamp():return datetime.datetime.now(TZ).isoformat(timespec='seconds')
def write(stage,**kw):
 d=dict(stage=stage,time=stamp(),supervisor_pid=os.getpid(),**kw)
 p=STATUS.with_suffix('.tmp');p.write_text(json.dumps(d,indent=2));p.replace(STATUS)
 print(json.dumps(d),flush=True)
 return d

def run(cmd,timeout=30):
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout)
 if r.returncode:raise RuntimeError(' '.join(cmd[:2])+': '+(r.stdout+r.stderr)[-1600:])
 return r.stdout.strip()

def limit():
 raw=run(['sacctmgr','-nP','show','assoc','where','user=scx8178','format=Cluster,Account,User,MaxSubmitJobs'])
 rows=[l.split('|') for l in raw.splitlines() if l.strip()]
 rows=[r for r in rows if len(r)>=4 and r[0]=='n32-h' and r[1]=='scx8178' and r[2]=='scx8178']
 if len(rows)!=1:raise RuntimeError('Cannot uniquely verify the account submission limit: '+raw)
 value=rows[0][3].strip()
 if value in ('','UNLIMITED','Unlimited','-1'):return True,value or 'unlimited'
 if not value.isdigit():raise RuntimeError('Unknown submission limit '+repr(value))
 return int(value)>0,int(value)

def queue():
 raw=run(['squeue','-h','-u','scx8178','-o','%i|%j|%T'])
 return [l for l in raw.splitlines() if '|energywam' in l.lower()]

def wait_limit(deadline,stage):
 while time.time()<deadline:
  try:
   allowed,value=limit();existing=queue()
   if allowed and not existing:return
   write('waiting_for_account_limit' if not allowed else 'waiting_for_existing_energywam_job',next_stage=stage,MaxSubmitJobs=value,existing_jobs=existing,deadline=datetime.datetime.fromtimestamp(deadline,TZ).isoformat(timespec='seconds'),gpu_allocated=False)
  except Exception as e:write('waiting_resource_check_error',error=str(e),next_stage=stage,gpu_allocated=False)
  time.sleep(60)
 raise TimeoutError('Six-hour waiting budget expired; no additional job submitted')

def submit(script,stage):
 output=run(['sbatch','--parsable',str(LOGS/script)])
 jid=output.split(';')[0].strip()
 if not jid.isdigit():raise RuntimeError('Unexpected sbatch response: '+output)
 (LOGS/(stage+'_jobid.txt')).write_text(jid+'\n');write('submitted',job_id=jid,experiment_stage=stage)
 return jid

def wait_job(jid,stage):
 for _ in range(100):
  q=run(['squeue','-h','-j',jid,'-o','%T|%R'])
  if q:
   write('running' if q.startswith('RUNNING|') else 'queued',job_id=jid,experiment_stage=stage,slurm=q)
  else:
   raw=run(['sacct','-X','-nP','-j',jid,'-o','State,ExitCode'])
   rows=[r.split('|') for r in raw.splitlines() if r.strip()]
   if rows:
    state,code=rows[0][:2]
    write('job_finished',job_id=jid,experiment_stage=stage,slurm_state=state,exit_code=code)
    if state.startswith('COMPLETED') and code=='0:0':return
    if any(state.startswith(x) for x in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']):
     raise RuntimeError(stage+' job '+jid+' ended '+state+' '+code+'; inspect '+str(LOGS))
  time.sleep(30)
 raise TimeoutError('Job monitoring budget exceeded; job preserved, no duplicate submitted')

def main():
 LOGS.mkdir(parents=True,exist_ok=True)
 lock=(LOGS/'training_supervisor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 deadline=time.time()+6*3600
 for marker in ['QWEN_WEIGHTS_READY.json','DINO_AND_METADATA_READY.json','HPC_CODE_VALIDATED.json']:
  if not (LOGS/marker).exists():raise RuntimeError('Missing readiness evidence: '+marker)
 write('ready_waiting_for_resources',deadline=datetime.datetime.fromtimestamp(deadline,TZ).isoformat(timespec='seconds'),max_gpu_jobs=2,gpus_per_job=1,preflight_wall_minutes=15,short_training_wall_minutes=30)
 wait_limit(deadline,'preflight');jid=submit('preflight.slurm','preflight');wait_job(jid,'preflight')
 report=ROOT/'runs'/('preflight-'+jid)/'result.json'
 if not report.exists() or json.loads(report.read_text()).get('stage')!='complete':raise RuntimeError('Preflight has no successful real-model training report')
 wait_limit(deadline,'short_training');jid2=submit('short_training.slurm','short_training');wait_job(jid2,'short_training')
 write('complete',preflight_job=jid,training_job=jid2,notes='Bounded engineering trial; subset statistics; offline metrics, not closed-loop success')
if __name__=='__main__':
 try:main()
 except BlockingIOError:print('Another supervisor already owns this trial; no duplicate launched',flush=True)
 except Exception:write('stopped',error=traceback.format_exc());raise
