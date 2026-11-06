#!/usr/bin/env python
"""Resumable parallel cloud copy: verify before redirecting and deleting locally."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('/cloud/zzc/energyWAM/datasets')
STATUS=ROOT/'logs/cloud_migration_status.json'
NAMES=['robocasa365_archives','robocasa365_assets_archives','robocasa365','sim','sources','models','robocasa365_trainmix']
LOCK=threading.Lock()
STATES={name:'pending' for name in NAMES}

def status(phase,name=None,**extra):
    with LOCK:
        if name:STATES[name]=phase
        data=dict(phase=phase if not name else 'running',directory=name,
                  updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                  directories=dict(STATES),completed=[n for n,s in STATES.items() if s=='complete'],**extra)
        tmp=STATUS.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(STATUS)
        print(json.dumps(data),flush=True)

def migrate(name):
    src=Path('/datasets')/name;dst=DEST/name;old=src.with_name(name+'.migration_local')
    with (ROOT/f'logs/cloud_migration_{name}.lock').open('a') as lockfile:
        fcntl.flock(lockfile,fcntl.LOCK_EX)
        if src.is_symlink():
            if src.resolve()!=dst.resolve():raise RuntimeError(f'unexpected existing link: {src}')
            if old.exists():shutil.rmtree(old)
            status('complete',name);return
        if old.exists():
            if src.exists():raise RuntimeError(f'both source and switch backup exist: {src}')
            old.rename(src)
        dst.mkdir(exist_ok=True)
        status('copying',name)
        log=ROOT/f'logs/cloud_copy_{name}.log'
        with log.open('a') as stream:
            subprocess.run(['rsync','-a','--partial','--append-verify','--info=stats2',str(src)+'/',str(dst)+'/'],stdout=stream,stderr=subprocess.STDOUT,check=True)
        status('verifying_checksums',name)
        result=subprocess.run(['rsync','-aicn','--delete','--out-format=%i|%n',str(src)+'/',str(dst)+'/'],capture_output=True,text=True,check=True)
        differences=[]
        ignored_link_permissions=[]
        for line in result.stdout.splitlines():
            code, separator, relative_name = line.partition('|')
            # UPFS exposes symlink mode 0660 instead of Linux's 0777. Only
            # ignore this exact permissions-only symlink difference. Rsync's
            # checksum and link-target comparisons remain enabled; target,
            # content, missing-file, ownership and all other changes still fail.
            if separator and code == '.L...p.....':
                ignored_link_permissions.append(relative_name)
            elif line.strip():
                differences.append(line)
        if ignored_link_permissions:
            print(json.dumps({'directory':name,'upfs_symlink_mode_only_count':len(ignored_link_permissions),'examples':ignored_link_permissions[:3]}),flush=True)
        if differences:raise RuntimeError(f'{name} checksum/layout mismatch: {chr(10).join(differences)[:2000]}')
        os.rename(src,old)
        try:src.symlink_to(dst,target_is_directory=True)
        except BaseException:
            os.rename(old,src);raise
        status('removing_verified_local_copy',name)
        shutil.rmtree(old)
        status('complete',name)

def main():
    while not os.path.ismount('/cloud'):
        status('waiting_for_cloud_mount');time.sleep(5)
    DEST.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(migrate,NAMES))
    status('complete',destination=str(DEST))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc));raise
