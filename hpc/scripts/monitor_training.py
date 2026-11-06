"""Run on the local server. Ctrl+C stops monitoring only, never the supervisor/job."""
import argparse,subprocess,shlex,time
SCRIPT='''from pathlib import Path
import json,subprocess,time
root=Path.home()/"energywam_storage";p=root/"logs/training_status.json"
print(time.strftime("%Y-%m-%d %H:%M:%S %Z"),flush=True)
if not p.exists():print("Supervisor status not available yet");raise SystemExit
s=json.loads(p.read_text());print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
jid=s.get("job_id") or s.get("training_job") or s.get("preflight_job")
if jid:
 q=subprocess.run(["squeue","-h","-j",str(jid),"-o","%i %T %M %N %R"],capture_output=True,text=True,timeout=8)
 print("SLURM:",q.stdout.strip() or "not in active queue",flush=True)
 for stem in ["preflight-", "short-training-"]:
  for suffix in [".log",".err"]:
   f=root/"logs"/(stem+str(jid)+suffix)
   if f.exists():print(str(f)+" recent output:\\n"+f.read_text(errors="replace")[-2600:],flush=True)
 if s.get("stage")=="running":
  try:
   r=subprocess.run(["srun","--overlap","--jobid="+str(jid),"--ntasks=1","--cpus-per-task=1","nvidia-smi","--query-gpu=name,memory.used,memory.total,utilization.gpu","--format=csv"],capture_output=True,text=True,timeout=6)
   print("GPU:",r.stdout or r.stderr,flush=True)
  except subprocess.TimeoutExpired:print("GPU status query timed out; training is preserved",flush=True)
else:print("GPU: not allocated; account limit/resource waiting",flush=True)
f=root/"logs/training_supervisor.log"
if f.exists():print("Recent supervisor output:\\n"+f.read_text(errors="replace")[-1200:],flush=True)
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');p.add_argument('--interval',type=int,default=30);a=p.parse_args()
 try:
  while True:
   subprocess.run(['/data3/xyx_zhuzicheng/.local/bin/papp-cloud-proxy','ssh','scx8178@n32ha14p','python3 -u -c '+shlex.quote(SCRIPT)],timeout=35)
   if a.once:return
   time.sleep(max(15,a.interval))
 except KeyboardInterrupt:print('\nMonitoring exited; background waiting and training are unchanged.')
if __name__=='__main__':main()
