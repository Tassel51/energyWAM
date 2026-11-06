#!/usr/bin/env python
"""Configure only this project's collector; never call shared wandb login."""
import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    if not sys.stdin.isatty():
        raise SystemExit('Run this command in your server terminal; API Key input must be hidden.')
    config_path=ROOT/'configs/wandb_bridge.json'
    cfg=json.loads(config_path.read_text())
    entity=input(f"W&B team/entity [{cfg.get('entity') or 'zhuzc-zhejiang-university'}]: ").strip() or cfg.get('entity') or 'zhuzc-zhejiang-university'
    key=getpass.getpass('Paste your W&B API Key (hidden): ').strip()
    if not key:raise SystemExit('Empty key; nothing changed.')
    # Only this process receives the key. No writes to ~/.netrc or shared settings.
    os.environ['WANDB_API_KEY']=key
    os.environ['WANDB_BASE_URL']=cfg.get('base_url','https://api.wandb.ai')
    import wandb
    try:
        api=wandb.Api(api_key=key,timeout=30)
        default_entity=api.default_entity
        if not default_entity:raise RuntimeError('Account could not be verified')
        api.create_project('lte-wam',entity=entity)
    except Exception:
        raise SystemExit('Account/project access verification failed; nothing changed. Check API Key and entity.')
    secret=Path.home()/'.config/energywam/wandb_api_key'
    secret.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
    temp=secret.with_suffix('.tmp')
    fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    os.fchmod(fd,0o600)
    with os.fdopen(fd,'w') as stream:
        stream.write(key+'\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,secret)
    cfg.update(mode='online',entity=entity,api_key_file=str(secret))
    temp_cfg=config_path.with_suffix('.tmp');temp_cfg.write_text(json.dumps(cfg,indent=2)+'\n');temp_cfg.replace(config_path)
    subprocess.run(['supervisorctl','restart','energywam-wandb'],check=True)
    print(f'Collector enabled for https://wandb.ai/{entity}/lte-wam')
    print('Training was not restarted. Shared W&B login was not changed.')


if __name__=='__main__':main()
