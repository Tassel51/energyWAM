
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
