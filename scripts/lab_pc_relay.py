#!/usr/bin/env python3
"""Run on the PERSONAL COMPUTER: relay current code and read-only lab facts.

Uses existing local SSH aliases; never reads or transfers a private key.
The gpu-smoke action creates an isolated venv reusing read-only installed
packages, and performs tiny CUDA operations only on explicitly selected idle GPUs.
No training or model/data transfer is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import contextlib
import uuid

SSH_OPTIONS = ['-o','ForwardAgent=no','-o','ServerAliveInterval=15',
               '-o','ServerAliveCountMax=3','-o','ConnectTimeout=20']
RETRY_LOG = None

# Embedded probe snapshot; no initial connection to compshare is required.
PROBE_SOURCE = "\nimport json, os, shutil, subprocess, socket\nfrom pathlib import Path\ndef command(argv):\n    try:\n        p=subprocess.run(argv, capture_output=True, text=True, timeout=20)\n        return dict(returncode=p.returncode, stdout=p.stdout[:16000], stderr=p.stderr[:2000])\n    except Exception as e:\n        return dict(error=str(e))\nhome=Path.home()\npaths=[home,Path('/data3/xyx_zhuzicheng'),Path('/data3/xyx_zhuzicheng/projects'),Path('/data6'),Path('/data7'),Path('/data7/zzc/energyWAM'),Path('/datasets'),Path('/scratch')]\ndisk=[]\nfor p in paths:\n    if p.is_dir():\n        try:\n            u=shutil.disk_usage(p);disk.append(dict(path=str(p),total=u.total,free=u.free,writable=os.access(p,os.W_OK)))\n        except OSError:pass\nresult=dict(hostname=socket.gethostname(),fqdn=socket.getfqdn(),user=command(['id','-un']),\n    filesystems=command(['df','-hPT']),host_memory=command(['free','-h']),\n    gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv']),\n    gpu_processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),\n    cuda=command(['nvcc','--version']),disk=disk,\n    executables={n:shutil.which(n) for n in ['python3','conda','mamba','micromamba','uv','sbatch','srun','sinfo','qsub','bsub','tmux','rsync']},\n    candidate_environment_managers=[str(p) for p in [home/'miniconda3/bin/conda',Path('/data3/xyx_zhuzicheng/miniconda3/bin/conda')] if p.is_file()],\n    known_historical_asset_locations={str(p):p.exists() for p in [Path('/data7/zzc/energyWAM'),Path('/data3/xyx_zhuzicheng/projects/hpc/energyWAM'),Path('/data6/zzc/Qwen3-VL-4B-Instruct'),Path('/data6/zzc/dinov3-vits16-pretrain-lvd1689m'),Path('/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python')]},\n    slurm_allocation={k:os.environ[k] for k in ['SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_JOB_PARTITION','CUDA_VISIBLE_DEVICES'] if k in os.environ})\nif shutil.which('sinfo'):\n    result['slurm_partitions']=command(['sinfo','-h','-o','%P %a %l %D %G'])\nprint(json.dumps(result,indent=2))\n"

UPLOAD_CODE = '''import os,shutil,sys
from pathlib import Path
p=Path(sys.argv[1]);tmp=p.with_name(p.name+'.upload')
with tmp.open('wb') as f:
    shutil.copyfileobj(sys.stdin.buffer,f)
    f.flush();os.fsync(f.fileno())
tmp.replace(p)
'''

REMOTE_STAGE = r'''
import hashlib,json,os,shutil,socket,tarfile
from pathlib import Path
root=Path(PROJECT_PARENT)/'energyWAM_remote'
parent=root.parent
if not parent.is_dir() or not os.access(parent,os.W_OK):raise RuntimeError('Project parent missing or not writable')
if shutil.disk_usage(parent).free<256*2**20:raise RuntimeError('Less than256MiB free for minimal code staging')
root.mkdir(exist_ok=True)
release=root/'releases'/RELEASE
release.mkdir(parents=True,exist_ok=False)
print(json.dumps(dict(root=str(root),release=str(release),hostname=socket.gethostname())))
'''

REMOTE_SMOKE = r'''
import json,os,shutil,socket,subprocess,sys,time,uuid
from pathlib import Path
if socket.gethostname()!=EXPECTED_HOST:raise RuntimeError('Unexpected GPU host')
if not GPU or any(not x.isdigit() for x in GPU.split(',')):raise RuntimeError('Explicit numeric GPU list required')
selected=GPU.split(',')
if len(selected)!=len(set(selected)):raise RuntimeError('Duplicate GPUs')
def query(args):
    return subprocess.check_output(['nvidia-smi',*args],text=True)
rows=query(['--query-gpu=index,uuid,name,memory.used,memory.free','--format=csv,noheader,nounits'])
devices={r[0].strip():dict(uuid=r[1].strip(),name=r[2].strip(),used_MiB=int(r[3]),free_MiB=int(r[4])) for r in (line.split(',') for line in rows.splitlines())}
apps=query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
for i in selected:
    if i not in devices:raise RuntimeError('GPU not present: '+i)
    d=devices[i]
    if d['used_MiB']>256 or d['free_MiB']<2048 or any(line.split(',')[0].strip()==d['uuid'] for line in apps.splitlines()):
        raise RuntimeError('GPU no longer idle; refusing to interfere: '+i)
root=Path(PROJECT_PARENT)/'energyWAM_remote'
root.mkdir(exist_ok=True)
if shutil.disk_usage(root).free<512*2**20:raise RuntimeError('Insufficient space even for isolated smoke environment')
tag=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:8]
out=root/'validation'/tag;out.mkdir(parents=True,exist_ok=False)
envdir=out/'venv'
subprocess.run([BASE_PYTHON,'-m','venv','--without-pip','--system-site-packages',str(envdir)],check=True,capture_output=True,text=True)
python=envdir/'bin/python'
code="""import json,os,socket,sys,torch,subprocess
torch.set_num_threads(1)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count()==EXPECTED_COUNT, 'Unexpected visible GPU count'
results=[]
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        a=torch.ones((256,256),device='cuda:'+str(i),dtype=torch.float32)
        b=a@a
        torch.cuda.synchronize(i)
        assert bool(torch.all(b==256).item()), 'CUDA computation mismatch'
        results.append(dict(logical_device=i,name=torch.cuda.get_device_name(i),peak_allocated_bytes=torch.cuda.max_memory_allocated(i),value=float(b[0,0].item())))
rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
actual={line.split(',')[0].strip() for line in rows.splitlines() if line.split(',')[1].strip()==str(os.getpid())}
expected=set(EXPECTED_UUIDS)
assert actual==expected, ('Actual GPU placement mismatch',actual,expected)
print(json.dumps(dict(hostname=socket.gethostname(),pid=os.getpid(),python=sys.executable,prefix=sys.prefix,torch=torch.__version__,torch_cuda=torch.version.cuda,CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],actual_gpu_uuids=sorted(actual),devices=results,training_started=False)))
""".replace('EXPECTED_COUNT',str(len(selected))).replace('EXPECTED_UUIDS',repr([devices[i]['uuid'] for i in selected]))
(out/'smoke.py').write_text(code)
env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(devices[i]['uuid'] for i in selected),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
run=subprocess.run([str(python),str(out/'smoke.py')],capture_output=True,text=True,env=env,timeout=90)
(out/'stdout.log').write_text(run.stdout);(out/'stderr.log').write_text(run.stderr)
receipt=dict(hostname=socket.gethostname(),output_dir=str(out),returncode=run.returncode,approved_selected_gpus={i:devices[i] for i in selected},isolation='Independent venv with system-site-packages; base packages read-only, no pip install or base environment changes',training_started=False)
if run.returncode==0:receipt['cuda_result']=json.loads(run.stdout)
(out/'result.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''

REMOTE_VERIFY = r'''
import hashlib,json,os,socket,tarfile
from pathlib import Path
release=Path(RELEASE)
archive=release/'source.tar.gz'
if hashlib.sha256(archive.read_bytes()).hexdigest()!=DIGEST:raise RuntimeError('Archive SHA256 mismatch')
code=release/'code';code.mkdir(exist_ok=False)
with tarfile.open(archive) as t:
    members=t.getmembers()
    names=[m.name for m in members]
    if len(names)!=len(set(names)):raise RuntimeError('Duplicate archive paths')
    for m in members:
        p=Path(m.name)
        if p.is_absolute() or '..' in p.parts or not m.isfile():raise RuntimeError('Unsafe archive member')
    manifest=json.load(t.extractfile('SOURCE_MANIFEST.json'))
    if set(names)!={r['path'] for r in manifest}|{'SOURCE_MANIFEST.json'}:raise RuntimeError('Manifest membership mismatch')
    for row in manifest:
        if hashlib.sha256(t.extractfile(row['path']).read()).hexdigest()!=row['sha256']:raise RuntimeError('File SHA256 mismatch')
    for m in members:
        dest=code/m.name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write(t.extractfile(m).read())
receipt=dict(hostname=socket.gethostname(),code_dir=str(code),archive_sha256=DIGEST,verified_files=len(manifest),
             environment_created=False,gpu_computation_started=False,training_started=False)
(release/'sync_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


def run(argv, *, source=None, retry_safe=False):
    argv=[argv[0],*SSH_OPTIONS,*argv[1:]] if argv[0] in ['ssh','scp'] else argv
    # Only explicitly retry-safe reads/code packaging and file copies are
    # repeated. Environment creation, release extraction and CUDA are not.
    attempts=3 if argv[0]=='scp' or retry_safe else 1
    for attempt in range(1,attempts+1):
        try:
            p=subprocess.run(argv,input=source,text=True,encoding='utf-8',errors='replace',capture_output=True,timeout=600)
            if not p.returncode:return p.stdout
            error=p.stderr[-3000:]
        except subprocess.TimeoutExpired:
            error='Connection timed out after600seconds'
        record_failure(argv,attempt,error)
        transient=any(x in error.lower() for x in ['reset','closed','timed out','broken pipe','unknown error'])
        if attempt==attempts or not transient:break
        print(f'Transfer interrupted; retry {attempt+1}/{attempts}',flush=True)
        time.sleep(attempt*2)
    raise RuntimeError('Command failed: '+shlex.join(argv)+'\n'+error)


def record_failure(argv,attempt,error):
    if RETRY_LOG is not None:
        with RETRY_LOG.open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(at=time.time(),command=argv,attempt=attempt,error=error))+'\n')


def upload_archive(host,local,remote):
    """Stream binary over SSH, avoiding the failing Windows scp/SFTP upload.

    Only this new release's archive is replaced on retry. Source manifests and
    destination code are verified afterwards; existing code is never overwritten.
    """
    command='python3 -c '+shlex.quote(UPLOAD_CODE)+' '+shlex.quote(remote)
    argv=['ssh',*SSH_OPTIONS,host,command]
    for attempt in range(1,4):
        print(f'Uploading code via SSH stream ({attempt}/3)...',flush=True)
        try:
            with Path(local).open('rb') as stream:
                p=subprocess.run(argv,stdin=stream,capture_output=True,timeout=600)
            if p.returncode==0:return
            error=p.stderr.decode('utf-8',errors='replace')[-3000:]
        except subprocess.TimeoutExpired:
            error='SSH stream timed out after600seconds'
        record_failure(argv,attempt,error)
        if attempt==3 or not any(x in error.lower() for x in ['reset','closed','timed out','broken pipe','unknown error']):break
        time.sleep(attempt*2)
    raise RuntimeError('SSH stream upload failed; no code was extracted.\n'+error)


def remote_python(host,source, *, retry_safe=False):
    return json.loads(run(['ssh',host,'python3 -'],source=source,retry_safe=retry_safe))


def return_receipt(local, source_host, remote_path):
    try:
        run(['scp',str(local),source_host+':'+remote_path])
        return True
    except RuntimeError as error:
        print('Result saved locally; return to compshare interrupted. Continue with lab work.\n'+str(error),flush=True)
        return False


def main():
    global RETRY_LOG
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['probe','sync','gpu-smoke'])
    p.add_argument('--source',default='compshare-gpu',help='Existing PERSONAL COMPUTER SSH alias')
    p.add_argument('--target',default='lab-jump',help='Existing PERSONAL COMPUTER SSH alias (with ProxyJump)')
    p.add_argument('--source-root',default='/root/zzc/energyWAM')
    p.add_argument('--expected-host',default='user-SYS-420GP-TNR')
    p.add_argument('--remote-project-parent',default='/data3/xyx_zhuzicheng/projects')
    p.add_argument('--gpu',help='User-authorized GPUs for tiny CUDA check only, e.g.6,7; fresh idle check required')
    p.add_argument('--remote-python',default='/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python',help='Existing remote Python with torch, used read-only to create isolated smoke venv')
    p.add_argument('--out',type=Path,default=Path('energywam_lab_relay_results'))
    p.add_argument('--reuse-cached-bundle',action='store_true',help='Explicitly reuse last verified PC code cache if compshare is unreachable; may be older than current source')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    RETRY_LOG=a.out/'transport_failures.jsonl'
    if any(h.startswith('-') for h in [a.source,a.target]):raise ValueError('Invalid SSH alias')
    cache=a.out/'cache';cache.mkdir(parents=True,exist_ok=True)
    # Persistent cache survives disconnects. Embedded read-only probe removes
    # an unnecessary round-trip to compshare before every lab operation.
    with contextlib.nullcontext(cache) as temp:
        temp=Path(temp);probe_file=temp/'probe.py'
        probe_file.write_text(PROBE_SOURCE,encoding='utf-8')
        facts=remote_python(a.target,probe_file.read_text(encoding='utf-8'),retry_safe=True)
        if facts.get('hostname')!=a.expected_host:raise RuntimeError('Unexpected destination hostname; refusing deployment: '+str(facts.get('hostname')))
        report=a.out/'lab_probe_result.json';report.write_text(json.dumps(facts,indent=2)+'\n')
        delivered=return_receipt(report,a.source,a.source_root+'/logs/lab_remote_setup/pc_relay_probe_result.json')
        print('Read-only lab facts saved; returned_to_compshare:',delivered,'hostname:',facts['hostname'],flush=True)
        if a.action=='probe':return
        if a.action=='gpu-smoke':
            if not a.gpu:p.error('gpu-smoke requires --gpu with user-authorized indices')
            source='GPU='+repr(a.gpu)+'\nBASE_PYTHON='+repr(a.remote_python)+'\nPROJECT_PARENT='+repr(a.remote_project_parent)+'\nEXPECTED_HOST='+repr(a.expected_host)+'\n'+REMOTE_SMOKE
            receipt=remote_python(a.target,source)
            local=a.out/'gpu_smoke_result.json';local.write_text(json.dumps(receipt,indent=2)+'\n')
            return_receipt(local,a.source,a.source_root+'/logs/lab_remote_setup/pc_relay_gpu_smoke_result.json')
            print(json.dumps(receipt,indent=2))
            if receipt['returncode']:raise SystemExit(receipt['returncode'])
            return
        command='cd '+shlex.quote(a.source_root)+' && python3 scripts/lab_remote.py bundle'
        archive=temp/'source.tar.gz'
        cached_report=temp/'bundle_report.json'
        if a.reuse_cached_bundle:
            if not archive.is_file() or not cached_report.is_file():raise RuntimeError('No verified code cache available; cannot skip initial code download')
            bundle=json.loads(cached_report.read_text())
            print('Explicitly reusing cached source tree:',bundle['source_tree_digest'],flush=True)
        else:
            bundle=json.loads(run(['ssh',a.source,command],retry_safe=True))
            pending=temp/'source.tar.gz.download'
            run(['scp',a.source+':'+bundle['archive'],str(pending)])
            if hashlib.sha256(pending.read_bytes()).hexdigest()!=bundle['archive_sha256']:raise RuntimeError('Downloaded source cache checksum mismatch')
            pending.replace(archive)
            cached_report.write_text(json.dumps(bundle,indent=2)+'\n')
        if hashlib.sha256(archive.read_bytes()).hexdigest()!=bundle['archive_sha256']:raise RuntimeError('PC relay archive checksum mismatch')
        release=Path(bundle['archive']).name.removesuffix('.tar.gz')+'_relay_'+uuid.uuid4().hex[:8]
        stage=remote_python(a.target,'RELEASE='+repr(release)+'\nPROJECT_PARENT='+repr(a.remote_project_parent)+'\n'+REMOTE_STAGE)
        # Keep a local receipt even if upload disconnects; do not silently
        # erase evidence of an incomplete remote staging directory.
        (a.out/'pending_stage.json').write_text(json.dumps(stage,indent=2)+'\n')
        upload_archive(a.target,archive,stage['release']+'/source.tar.gz')
        receipt=remote_python(a.target,'RELEASE='+repr(stage['release'])+'\nDIGEST='+repr(bundle['archive_sha256'])+'\n'+REMOTE_VERIFY)
        local=a.out/'sync_receipt.json';local.write_text(json.dumps(receipt,indent=2)+'\n')
        return_receipt(local,a.source,a.source_root+'/logs/lab_remote_setup/pc_relay_sync_receipt.json')
        print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
