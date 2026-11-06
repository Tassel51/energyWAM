"""Start the authorized download watchdog; credentials stay only in process memory."""
from pathlib import Path
import getpass,subprocess,os,signal,time
base=Path(__file__).resolve().parents[2]
password=getpass.getpass('Existing platform password (memory only): ')
for pid in [1012447]:
 try:
  if str(base/'hpc/scripts/watch_full_download.py').encode() in Path('/proc/%s/cmdline'%pid).read_bytes():os.kill(pid,signal.SIGTERM)
 except FileNotFoundError:pass
time.sleep(.2)
p=subprocess.Popen(['python3','-u',str(base/'hpc/scripts/watch_full_download.py'),'--auth-stdin'],stdin=subprocess.PIPE,stdout=(base/'logs/full_download_watch_20261030.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
p.stdin.write((password+'\n').encode());p.stdin.close();password=None
print('Download watchdog started:',p.pid,flush=True)
