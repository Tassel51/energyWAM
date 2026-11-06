"""Local watchdog: mirror progress, repair this task's SSH tunnel, release it on success."""
import datetime,fcntl,json,os,signal,subprocess,time,shlex,pty,termios,select,sys
from zoneinfo import ZoneInfo
from pathlib import Path
BASE=Path(__file__).resolve().parents[2];LOGS=BASE/'logs'

def mirror(data):
 p=LOGS/'full_download_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)

def sockets():return sorted(Path('/tmp').glob('energywam-datasets-*'),key=lambda p:p.stat().st_mtime if p.exists() else 0,reverse=True)
def query():
 for sock in sockets():
  try:
   r=subprocess.run(['ssh','-S',str(sock),'energywam-dataset-connection','python3 ~/energywam_storage/downloads/download_connection_status.py'],capture_output=True,text=True,timeout=35)
   if r.returncode==0:return json.loads(r.stdout),sock
  except (ValueError,subprocess.TimeoutExpired):pass
 raise RuntimeError('Dedicated dataset SSH connection unavailable')

def cleanup(sock):
 subprocess.run(['ssh','-S',str(sock),'-O','exit','energywam-dataset-connection'],capture_output=True,timeout=10)
 script=str(BASE/'hpc/scripts/box_download_proxy.py')
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:a=(p/'cmdline').read_bytes().split(b'\0')
  except OSError:continue
  if len(a)>2 and a[0].endswith(b'python3') and a[1]==b'-u' and a[2].decode()==script:os.kill(int(p.name),signal.SIGTERM)

def refresh_login(password):
 # Password is supplied once over stdin, retained only in RAM, never logged.
 master,slave=pty.openpty()
 attrs=termios.tcgetattr(slave);attrs[3]&=~termios.ECHO;termios.tcsetattr(slave,termios.TCSANOW,attrs)
 child=subprocess.Popen(['/data3/xyx_zhuzicheng/.local/bin/lda-hpc','login'],stdin=slave,stdout=slave,stderr=slave,start_new_session=True)
 os.close(slave);captured=b'';sent=False;deadline=time.time()+40
 try:
  while time.time()<deadline:
   if select.select([master],[],[],.2)[0]:
    try:b=os.read(master,4096)
    except OSError:break
    if not b:break
    captured+=b
    if not sent and b'Password:' in captured:
     os.write(master,password.encode()+b'\n');sent=True
   if child.poll() is not None:break
  if child.poll() is None:child.terminate();child.wait(timeout=3)
  return child.returncode==0 and b'login successful' in captured
 finally:os.close(master)

def main():
 lock=(LOGS/'full_download_watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 start=time.time();failed=0
 password=sys.stdin.readline().rstrip('\n') if '--auth-stdin' in sys.argv else None
 last_auth=time.time();last_auth_attempt=0
 while time.time()-start<72*3600:
  if password and time.time()-last_auth>8*3600 and time.time()-last_auth_attempt>600:
   last_auth_attempt=time.time();ok=refresh_login(password);print('Download-session authentication refresh:',bool(ok),flush=True)
   if ok:last_auth=time.time()
  try:
   data,sock=query();data['monitor_updated_at']=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds');failed=0
   if not data.get('pipeline_alive',True) and data.get('stage') not in ('complete','failed','blocked_storage','waiting_for_robosuite_assets'):
    data['previous_stage']=data.get('stage');data['stage']='stopped_unexpectedly';data['error']='Remote pipeline exited without a final status; partial files are preserved'
   mirror(data)
   if data.get('stage')=='complete':cleanup(sock);print('Full dataset/assets download verified complete; own tunnel and relay released',flush=True);return
   if data.get('stage') in ('failed','blocked_storage','stopped_unexpectedly'):
    print('Download pipeline requires attention:',json.dumps(data.get('errors') or data.get('error')),flush=True);return
  except Exception as e:
   failed+=1;mirror(dict(stage='connection_unavailable',error=str(e),consecutive_failures=failed,partial_files_preserved=True))
   try:
    r=subprocess.run([str(BASE/'hpc/tools/papp-dataset-tunnel'),'rsync',str(BASE/'coordination/full_download_20261030/download_manifest.json'),'scx8178@n32ha14p:/home/bingxing2/home/scx8178/energywam_storage/downloads/full_robocasa365_manifest.json'],capture_output=True,text=True,timeout=45)
    print('Dataset tunnel reconnect return code',r.returncode,flush=True)
    if password and 'E202' in (r.stdout+r.stderr) and time.time()-last_auth_attempt>600:
     last_auth_attempt=time.time();ok=refresh_login(password);print('Expired-session authentication restored:',bool(ok),flush=True)
     if ok:last_auth=time.time()
   except subprocess.TimeoutExpired:print('Dataset tunnel reconnect timed out',flush=True)
  time.sleep(60)
 print('72-hour monitor budget expired; downloads and connection preserved',flush=True)
if __name__=='__main__':
 try:main()
 except BlockingIOError:print('Existing download watchdog preserved',flush=True)
