#!/usr/bin/env python3
"""Prepare dirty-tree code bundles and inspect lab SSH; never copies credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / 'logs/lab_remote_setup'

REMOTE_PROBE = r'''
import json, os, shutil, subprocess, socket
from pathlib import Path
def command(argv):
    try:
        p=subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return dict(returncode=p.returncode, stdout=p.stdout[:16000], stderr=p.stderr[:2000])
    except Exception as e:
        return dict(error=str(e))
home=Path.home()
paths=[home,Path('/data3/xyx_zhuzicheng'),Path('/data3/xyx_zhuzicheng/projects'),Path('/data6'),Path('/data7'),Path('/data7/zzc/energyWAM'),Path('/datasets'),Path('/scratch')]
disk=[]
for p in paths:
    if p.is_dir():
        try:
            u=shutil.disk_usage(p);disk.append(dict(path=str(p),total=u.total,free=u.free,writable=os.access(p,os.W_OK)))
        except OSError:pass
result=dict(hostname=socket.gethostname(),fqdn=socket.getfqdn(),user=command(['id','-un']),
    filesystems=command(['df','-hPT']),host_memory=command(['free','-h']),
    gpu=command(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free,driver_version','--format=csv']),
    gpu_processes=command(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv']),
    cuda=command(['nvcc','--version']),disk=disk,
    executables={n:shutil.which(n) for n in ['python3','conda','mamba','micromamba','uv','sbatch','srun','sinfo','qsub','bsub','tmux','rsync']},
    candidate_environment_managers=[str(p) for p in [home/'miniconda3/bin/conda',Path('/data3/xyx_zhuzicheng/miniconda3/bin/conda')] if p.is_file()],
    known_historical_asset_locations={str(p):p.exists() for p in [Path('/data7/zzc/energyWAM'),Path('/data3/xyx_zhuzicheng/projects/hpc/energyWAM'),Path('/data6/zzc/Qwen3-VL-4B-Instruct'),Path('/data6/zzc/dinov3-vits16-pretrain-lvd1689m'),Path('/data3/xyx_zhuzicheng/miniconda3/envs/LDA/bin/python')]},
    slurm_allocation={k:os.environ[k] for k in ['SLURM_JOB_ID','SLURM_JOB_GPUS','SLURM_JOB_PARTITION','CUDA_VISIBLE_DEVICES'] if k in os.environ})
if shutil.which('sinfo'):
    result['slurm_partitions']=command(['sinfo','-h','-o','%P %a %l %D %G'])
print(json.dumps(result,indent=2))
'''


def eligible(relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts:
        return False
    if p.parts[0] not in {'energywam','scripts','tests','hpc','configs','stats','manifests','docs'} and len(p.parts) != 1:
        return False
    if any(x.startswith('.') or x in {'__pycache__','node_modules','figures'} for x in p.parts):
        return False
    if any(x in p.name.lower() for x in ['wandb','secret','credential','private','token','password']):
        # Token implementation source names are allowed; secret configs are not.
        if p.suffix != '.py':
            return False
    return p.suffix in {'.py','.md','.txt','.yaml','.yml','.json','.sh','.slurm'}


def bundle(out):
    out.mkdir(parents=True, exist_ok=True)
    paths = subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    chosen = []
    for rel in sorted(set(paths)):
        p = ROOT / rel
        if rel and eligible(rel) and p.is_file() and not p.is_symlink() and p.stat().st_size <= 4*2**20:
            chosen.append((rel,p.read_bytes()))
    if not chosen:
        raise RuntimeError('No source files selected')
    # Add actual current single-episode selection, not its datasets or weights.
    source = Path('/cloud/zzc/energyWAM/experiments/single_episode_lr_20261108')
    for n in ['train_val_manifest.json','train_monitor_manifest.json']:
        if (source/n).is_file():
            chosen.append(('manifests/experiments/single_episode_lr_20261108/'+n,(source/n).read_bytes()))
    digest=hashlib.sha256()
    manifest=[]
    for name,raw in chosen:
        sha=hashlib.sha256(raw).hexdigest();manifest.append(dict(path=name,bytes=len(raw),sha256=sha))
        digest.update(name.encode()+b'\0'+sha.encode()+b'\n')
    tag=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+digest.hexdigest()[:12]
    archive=out/f'energywam_code_{tag}.tar.gz'
    import io
    with tarfile.open(archive,'x:gz') as tar:
        for name,raw in chosen:
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644
            tar.addfile(info,io.BytesIO(raw))
        raw=(json.dumps(manifest,indent=2)+'\n').encode()
        info=tarfile.TarInfo('SOURCE_MANIFEST.json');info.size=len(raw);info.mode=0o644
        tar.addfile(info,io.BytesIO(raw))
    report=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                source_tree_digest=digest.hexdigest(),files=len(manifest),uncompressed_bytes=sum(x['bytes'] for x in manifest),
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                includes_working_tree_changes=True,weights_or_datasets_included=False,
                excluded='git internals, symlinks, environments, caches, credentials/W&B configs, binaries, files>4MiB')
    (out/'bundle_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['bundle','check','probe','export-probe'])
    p.add_argument('--host',default='lab-jump')
    p.add_argument('--user',default='xyx_zhuzicheng')
    p.add_argument('--port',type=int)
    p.add_argument('--identity',type=Path,default=Path('/root/.ssh/id_ed25519_energywam_lab'))
    p.add_argument('--proxy-jump',help='Public jump target, e.g. user@jump.example:22; no passwords')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    a=p.parse_args()
    if a.action=='bundle':
        bundle(a.out);return
    if a.action=='export-probe':
        a.out.mkdir(parents=True,exist_ok=True)
        target=a.out/'lab_probe.py'
        target.write_text(REMOTE_PROBE)
        print(target)
        return
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','ConnectionAttempts=1',
             '-o','StrictHostKeyChecking=yes','-l',a.user]
    if a.port is not None:command+=['-p',str(a.port)]
    if a.identity.exists():
        command+=['-i',str(a.identity),'-o','IdentitiesOnly=yes']
    if a.proxy_jump:command+=['-J',a.proxy_jump]
    command += [a.host, 'python3 -' if a.action=='probe' else 'hostname; id -un']
    r=subprocess.run(command,input=REMOTE_PROBE if a.action=='probe' else None,text=True,capture_output=True,timeout=180)
    a.out.mkdir(parents=True,exist_ok=True)
    result=dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,
                checked_at_utc=datetime.now(timezone.utc).isoformat())
    (a.out/f'{a.action}_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    raise SystemExit(r.returncode)


if __name__=='__main__':
    main()
