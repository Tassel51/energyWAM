"""Host-scoped NVML GPU authorization and optional UUID CUDA visibility."""
import json
import re
import socket
import subprocess
from pathlib import Path


def authorized_visibility(pair, policy_path, *, hostname=None, visible=None):
    if not re.fullmatch(r'\d+,\d+', pair) or len(set(pair.split(','))) != 2:
        raise ValueError('Expected two distinct nonnegative NVML GPU indices')
    policy = json.loads(Path(policy_path).read_text())
    if pair not in policy['allowed_physical_pairs']:
        raise RuntimeError('Physical GPU pair prohibited by explicit user policy')
    if policy.get('hostname') and policy['hostname'] != (hostname or socket.gethostname()):
        raise RuntimeError('GPU policy belongs to a different host')
    if not policy.get('use_uuid_visibility', False):
        if visible != pair:
            raise RuntimeError('visible GPUs must match explicitly selected physical pair')
        return pair
    if not policy.get('hostname'):
        raise RuntimeError('UUID GPU policy must explicitly bind hostname')
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    mapping = {a.strip():b.strip() for a,b in (row.split(',') for row in rows.splitlines())}
    uuids = ','.join(mapping[i] for i in pair.split(','))
    if visible not in [pair, uuids]:
        raise RuntimeError('Visible GPU selection disagrees with authorized NVML pair')
    expected = policy.get('expected_gpu_uuids', {})
    if any(expected.get(i) != mapping[i] for i in pair.split(',')):
        raise RuntimeError('Authorized GPU UUID identity changed or missing')
    return uuids


def apply_allocator_limits(pair, policy_path):
    """Optional host-policy caps constrain only this process's CUDA allocator."""
    policy=json.loads(Path(policy_path).read_text())
    limits=policy.get('cuda_allocator_limit_mib',{})
    if not limits:return
    # Revalidate host/UUID/pair before applying local CUDA-device indices.
    authorized_visibility(pair,policy_path,visible=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'))
    import torch
    for logical,index in enumerate(pair.split(',')):
        if index in limits:
            total=torch.cuda.get_device_properties(logical).total_memory
            fraction=float(limits[index])*2**20/total
            if not 0<fraction<=1:raise ValueError('Invalid allocator budget')
            torch.cuda.set_per_process_memory_fraction(fraction,device=logical)
