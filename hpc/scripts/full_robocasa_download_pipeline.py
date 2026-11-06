"""Run on HPC: download all official bundles, publish progress, integrate pinned source."""
import argparse,collections,fcntl,hashlib,json,os,shutil,subprocess,sys,time,tarfile,traceback,datetime
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path.home()/'energywam_storage';LOGS=ROOT/'logs';MANIFEST=ROOT/'downloads/full_robocasa365_manifest.json'
STATUS=LOGS/'full_robocasa365_status.json';TZ=ZoneInfo('Asia/Shanghai')

def now():return datetime.datetime.now(TZ).isoformat(timespec='seconds')
def write(path,data):
 p=path.with_name(path.name+'.tmp');p.write_text(json.dumps(data,indent=2));p.replace(path)

def snapshot(manifest,stage,**extra):
 counts=collections.Counter();ready_data=ready_assets=progress=0;active=[];errors=[]
 for entry in manifest['entries']:
  p=ROOT/'status'/(entry['id']+'.json')
  try:rec=json.loads(p.read_text())
  except (OSError,ValueError):rec={}
  state=rec.get('state','new');counts[state]+=1
  if state in ('done','existing'):
   if entry['kind']=='dataset':ready_data+=1
   else:ready_assets+=1
   progress+=entry['archive_bytes']
  else:
   ext='.tar' if entry['kind']=='dataset' else '.zip';base=ROOT/'archives'/(entry['id']+ext)
   sizes=[]
   for candidate in [base,base.with_name(base.name+'.part')]:
    try:sizes.append(candidate.stat().st_size)
    except FileNotFoundError:pass
   n=min(entry['archive_bytes'],max(sizes,default=0));progress+=n
   if state not in ('new','failed','blocked_budget','interrupted'):active.append(dict(id=entry['id'],state=state,bytes=n,total=entry['archive_bytes']))
   if rec.get('error'):errors.append(dict(id=entry['id'],error=rec['error']))
 result=dict(stage=stage,time=now(),pipeline_pid=os.getpid(),tasks_ready=ready_data,tasks_total=50,assets_ready=ready_assets,assets_total=6,robosuite_assets_ready=(LOGS/'ROBOSUITE_ASSETS_READY.json').exists(),counts=dict(counts),archive_progress_bytes=progress,archive_total_bytes=manifest['known_archive_bytes'],archive_progress_percent=round(progress/manifest['known_archive_bytes']*100,2),active=active,errors=errors,robocasa_commit=manifest['robocasa_commit'],**extra)
 write(STATUS,result);return result

def integrate_source(manifest):
 source=ROOT/'downloads/robocasa_source.tar.gz'
 deadline=time.time()+600
 while not source.exists() and time.time()<deadline:
  snapshot(manifest,'waiting_for_source_archive');time.sleep(10)
 if not source.exists():raise RuntimeError('Pinned RoboCasa source archive has not arrived; data/assets remain intact')
 h=hashlib.sha256()
 with source.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 if h.hexdigest()!=manifest['source_archive_sha256']:raise RuntimeError('Pinned source checksum mismatch')
 dest=ROOT/'sim/robocasa365';prefix='robocasa-'+manifest['robocasa_commit'];copied=preserved=0
 with tarfile.open(source) as t:
  for m in t:
   p=Path(m.name)
   if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=prefix or not (m.isdir() or m.isfile()):raise RuntimeError('Unsafe source member')
   rel=Path(*p.parts[1:])
   if not rel.parts:continue
   target=dest/rel
   if m.isdir():target.mkdir(parents=True,exist_ok=True);continue
   if target.exists():preserved+=1;continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as src,target.open('xb') as out:shutil.copyfileobj(src,out,1024**2)
   os.chmod(target,m.mode&0o755);copied+=1
 macro=dest/'robocasa/macros_private.py'
 if not macro.exists():macro.write_text('DATASET_BASE_PATH = '+repr(str(ROOT/'datasets/robocasa365'))+'\n')
 proof=dict(commit=manifest['robocasa_commit'],archive_sha256=h.hexdigest(),source_files_copied=copied,downloaded_files_preserved=preserved,time=now())
 write(dest/'.energywam_source.json',proof);return proof

def main():
 global ROOT,LOGS,MANIFEST,STATUS
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--root',type=Path,default=ROOT)
 parser.add_argument('--python',default=None,help='Worker interpreter; defaults to root/env/bin/python or current Python')
 parser.add_argument('--proxy',default='http://127.0.0.1:38741',help='Download HTTP CONNECT proxy; empty string uses direct access')
 args=parser.parse_args()
 ROOT=args.root.resolve();LOGS=ROOT/'logs';MANIFEST=ROOT/'downloads/full_robocasa365_manifest.json';STATUS=LOGS/'full_robocasa365_status.json'
 worker_python=args.python or (str(ROOT/'env/bin/python') if (ROOT/'env/bin/python').exists() else sys.executable)
 LOGS.mkdir(exist_ok=True);lock=(LOGS/'full_robocasa365_pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 manifest=json.loads(MANIFEST.read_text());assert len(manifest['entries'])==56
 expected=manifest['uncompressed_expected_bytes']
 if expected>200*1024**3:raise RuntimeError('Manifest exceeds 200 GiB task storage budget')
 env=os.environ.copy()
 for k in ['https_proxy','HTTPS_PROXY','http_proxy','HTTP_PROXY']:
  if args.proxy:env[k]=args.proxy
  else:env.pop(k,None)
 if not args.proxy:
  for k in ['ALL_PROXY','all_proxy']:env.pop(k,None)
 env['NO_PROXY']=env['no_proxy']='localhost,127.0.0.1'
 env['PYTHONUNBUFFERED']='1'
 with (LOGS/'full_robocasa365_worker.log').open('a') as out:
  child=subprocess.Popen([worker_python,'-u',str(ROOT/'downloads/robocasa_download_worker.py'),'--manifest',str(MANIFEST),'--root',str(ROOT),'--workers','2','--retries','20','--max-storage-gib','200'],stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=env)
  last=None
  while child.poll() is None:
   s=snapshot(manifest,'downloading',worker_pid=child.pid)
   if last:
    delta=s['archive_progress_bytes']-last[1]
    if delta>=0:s['recent_bytes_per_second']=round(delta/(time.time()-last[0]),1);write(STATUS,s)
   last=(time.time(),s['archive_progress_bytes']);time.sleep(5)
  code=child.wait()
 s=snapshot(manifest,'verifying_results',worker_exit_code=code)
 if code or s['tasks_ready']!=50 or s['assets_ready']!=6:
  snapshot(manifest,'blocked_storage' if s['counts'].get('blocked_budget') else 'failed',worker_exit_code=code);return code or 1
 snapshot(manifest,'integrating_pinned_source');proof=integrate_source(manifest)
 final=snapshot(manifest,'complete',source=proof,notes='Downloads and archive/file counts verified; simulator execution and closed-loop success not tested')
 write(LOGS/'FULL_ROBOCASA365_READY.json',final);print(json.dumps(final),flush=True);return 0
if __name__=='__main__':
 try:sys.exit(main())
 except BlockingIOError:print('Existing download pipeline preserved; no duplicate started',flush=True)
 except Exception:
  try:
   m=json.loads(MANIFEST.read_text());snapshot(m,'failed',error=traceback.format_exc())
  except Exception:pass
  raise
