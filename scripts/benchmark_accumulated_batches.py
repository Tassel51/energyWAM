#!/usr/bin/env python
"""Two GPU workers: opposite BI directions, then data-parallel joint training.

Replicas sum outer gradients manually after each full local second-order
unroll. A microbatch of one preserves global batch two within a 48GiB card.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import timedelta
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--resume')
    p.add_argument('--init-from')
    p.add_argument('--stage-switch-policy', default='reset', choices=['reset','continue'])
    p.add_argument('--steps', type=int)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--benchmark-steps', type=int, default=0)
    p.add_argument('--longest-instruction', action='store_true')
    p.add_argument('--output-dir')
    p.add_argument('--worker', action='store_true')
    p.add_argument('--set', nargs='*', default=[])
    return p.parse_args()


def launch(args):
    devices = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if devices != ['2','3']:
        raise RuntimeError('replicated launcher requires CUDA_VISIBLE_DEVICES=2,3')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0))
        port = sock.getsockname()[1]
    children = []
    for rank, device in enumerate(devices):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, RANK=str(rank), WORLD_SIZE='2',
                   LOCAL_RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port))
        env.setdefault('NCCL_IB_DISABLE','1')
        children.append(subprocess.Popen([sys.executable,'-u',__file__,'--worker',*sys.argv[1:]], cwd=ROOT, env=env))
    try:
        while True:
            codes = [p.poll() for p in children]
            if any(code is not None and code != 0 for code in codes):
                raise RuntimeError(f'replicated worker failed: {codes}')
            if args.benchmark_steps:
                rss_kib = 0
                for child in children:
                    path = Path(f'/proc/{child.pid}/status')
                    if path.exists():
                        for line in path.read_text().splitlines():
                            if line.startswith('VmRSS:'):
                                rss_kib += int(line.split()[1])
                if rss_kib > 300 * 1024**2:
                    raise RuntimeError('benchmark workers exceeded aggregate 300 GiB RAM budget')
            if all(code is not None for code in codes):
                return
            time.sleep(0.5)
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


def worker(args):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    import train as T
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.replicated_batch_benchmark import ReplicatedTrainer
    from energywam.training.checkpoint import load_checkpoint, _rng_state, _set_rng_state

    rank = int(os.environ['RANK'])
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    dist.init_process_group('nccl', timeout=timedelta(minutes=5), device_id=device)
    cfg = load_config(args.config, overrides=args.set)
    if args.steps is not None:
        cfg.trainer.max_train_steps = args.steps
    cfg.trainer.device = 'cuda:0'
    cfg.trainer.language_model_devices = []
    cfg.trainer.saved_tensor_gpu_budgets_gib = []
    output = Path(args.output_dir or cfg.trainer.output_dir)
    cfg.trainer.output_dir = str(output if rank == 0 else output/'rank1')
    if cfg.trainer.per_device_batch_size < 2 or cfg.trainer.per_device_batch_size % 2:
        raise ValueError('replicated runner requires effective global batch=2')
    torch.manual_seed(cfg.trainer.seed)
    bundle = T.resolve_training_stats(cfg, checkpoint_path=args.resume or args.init_from)
    model = QwenEnergyWAM(cfg, stats_bundle=bundle)
    trainer = ReplicatedTrainer(cfg, model)
    if args.resume:
        meta = trainer.load(args.resume)
        payload = load_checkpoint(args.resume)
        states = payload.get('distributed_rng_states')
        if states:
            _set_rng_state(states[rank])
        else:
            torch.manual_seed(cfg.trainer.seed + rank + trainer.global_step * 100003)
        del payload
        print(f'[replicated rank={rank}] resumed global_step={meta.global_step}',flush=True)
    elif args.init_from:
        trainer.switch_stage(cfg.stage.stage, args.init_from, optimizer_policy=args.stage_switch_policy)
        torch.manual_seed(cfg.trainer.seed + rank)
    else:
        torch.manual_seed(cfg.trainer.seed + rank)
    trainer.export_config()
    loader = T.build_dataloader(cfg, model, smoke=args.smoke or bool(args.benchmark_steps))
    if args.longest_instruction:
        dataset = loader.dataset
        allowed = set(model.stats_bundle.provenance['train_episodes']['episode_indices'])
        episodes = [ep for ep in dataset.reader.episodes if ep['episode_index'] in allowed]
        texts = list({ep['tasks'][0] for ep in episodes})
        lengths = dict(zip(texts, map(len, model.backbone.processor.tokenizer(texts,add_special_tokens=False)['input_ids'])))
        longest = max(episodes,key=lambda ep:lengths[ep['tasks'][0]])
        dataset._index = [(longest['episode_index'],i) for i in range(min(16,longest['length']-16))]
        print(f'[replicated rank={rank}] longest instruction episode={longest["episode_index"]} tokens={lengths[longest["tasks"][0]]}',flush=True)
    if args.benchmark_steps:
        dataset = loader.dataset
        needed = cfg.trainer.per_device_batch_size * args.benchmark_steps
        dataset._index = (dataset._index * ((needed + len(dataset._index)-1)//len(dataset._index)))[:needed]
    from energywam.training.run_state import ResumableWindowSampler, save_latest
    sampler = ResumableWindowSampler(len(loader.dataset), cfg.trainer.seed,
                                     consumed=trainer.global_step * cfg.trainer.per_device_batch_size,
                                     shuffle=not(args.smoke or args.benchmark_steps))
    loader = DataLoader(loader.dataset, batch_size=cfg.trainer.per_device_batch_size, sampler=sampler,
                        generator=torch.Generator().manual_seed(cfg.trainer.seed),
                        num_workers=0, drop_last=True, collate_fn=loader.collate_fn)
    role = ('action' if rank==0 else 'future') if cfg.stage.stage=='bi' else 'joint'
    print(f'[replicated rank={rank}] ready: {len(loader)} batches, role={role}',flush=True)
    iterator = iter(loader)
    records = []
    last_saved_step = None
    remaining = args.benchmark_steps or max(0, cfg.trainer.max_train_steps-trainer.global_step)
    for iteration in range(remaining):
        dist.barrier()
        begin = time.monotonic()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        data_seconds = time.monotonic()-begin
        print(f'[replicated rank={rank}] step={trainer.global_step} begin',flush=True)
        row = trainer.train_step(batch)
        host_stats = torch.cuda.memory.host_memory_stats()
        row['pin_host_alloc_seconds_total'] = host_stats.get('host_alloc_time.total',0)/1e6
        row['pin_peak_owned_gib'] = host_stats.get('allocated_bytes.peak',0)/2**30
        torch.cuda.synchronize()
        elapsed = torch.tensor([time.monotonic()-begin], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        row = dict(row, end_to_end_seconds=float(elapsed),data_seconds=data_seconds,
                   end_to_end_samples_per_second=cfg.trainer.per_device_batch_size/float(elapsed),global_step_completed=trainer.global_step)
        records.append(row)
        records = records[-100:]
        if rank == 0:
            output.mkdir(parents=True,exist_ok=True)
            (output/'speed_report.json').write_text(json.dumps({'mode':'replicated','world_size':2,'global_batch':cfg.trainer.per_device_batch_size,'completed_steps':trainer.global_step,'steps':records},indent=2))
            print(f'[replicated] completed={trainer.global_step} seconds={float(elapsed):.3f} samples/s={cfg.trainer.per_device_batch_size/float(elapsed):.5f}',flush=True)
        save_every = cfg.trainer.save_interval
        if not args.benchmark_steps and save_every and (iteration==0 or trainer.global_step%save_every==0):
            states=[None,None]
            dist.all_gather_object(states,_rng_state())
            if rank==0:
                model.distributed_rng_states=states
                save_latest(trainer, output)
            last_saved_step = trainer.global_step
            dist.barrier()
    if not args.benchmark_steps:
        states=[None,None]
        dist.all_gather_object(states,_rng_state())
        if rank==0:
            model.distributed_rng_states=states
            path = output/'ckpt_latest.pt'
            if last_saved_step != trainer.global_step:
                path = save_latest(trainer, output)
            (output/'final_report.json').write_text(json.dumps({'global_step':trainer.global_step,'steps':remaining,'last_loss':records[-1] if records else None,'checkpoint':str(path)},indent=2))
        dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':
    args=arguments()
    worker(args) if args.worker else launch(args)
