#!/usr/bin/env python
"""Bounded real-model training benchmark; no final checkpoint writes."""
import argparse
import json
import sys
import time
import os
import threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', type=int, required=True)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--checkpoint', action='store_true')
    ap.add_argument('--offload', action='store_true')
    ap.add_argument('--gpu-budget', type=float, default=0.0)
    ap.add_argument('--gpu-budgets', default=None, help='JSON list, one budget per visible GPU')
    ap.add_argument('--preallocate-optimizer', action='store_true')
    ap.add_argument('--longest-instruction', action='store_true')
    ap.add_argument('--pin-budget', type=float, default=0.0)
    ap.add_argument('--async-offload', action='store_true')
    ap.add_argument('--single-device', action='store_true')
    ap.add_argument('--direction', choices=['action','future'])
    ap.add_argument('--resume', default=None)
    ap.add_argument('--stage', default='bi', choices=['bi','joint'])
    ap.add_argument('--tag', required=True)
    args = ap.parse_args()
    import torch
    import train as T
    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.trainer import EnergyWAMTrainer
    folder = ROOT / 'logs/tuning_gpu23' / args.tag
    cfg = load_config(f'configs/gpu23_{args.stage}_20261101.yaml')
    cfg.backbone.gradient_checkpointing = args.checkpoint
    cfg.trainer.offload_saved_tensors = args.offload
    cfg.trainer.saved_tensor_gpu_budget_gib = args.gpu_budget
    cfg.trainer.saved_tensor_gpu_budgets_gib = json.loads(args.gpu_budgets) if args.gpu_budgets else []
    cfg.trainer.per_device_batch_size = args.batch
    cfg.trainer.saved_tensor_cpu_pin_budget_gib = args.pin_budget
    cfg.trainer.saved_tensor_async_offload = args.async_offload
    if args.single_device:
        cfg.trainer.language_model_devices = []
        cfg.trainer.device = 'cuda:0'
    if args.direction:
        cfg.stage.stage = 'action_given_future' if args.direction == 'action' else 'future_given_action'
    cfg.trainer.save_interval = 0
    cfg.trainer.output_dir = str(folder)
    cfg.trainer.logging_frequency = 1
    torch.manual_seed(cfg.trainer.seed)
    model = QwenEnergyWAM(cfg, stats_bundle=T.resolve_training_stats(cfg, checkpoint_path=args.resume))
    trainer = EnergyWAMTrainer(cfg, model)
    if args.preallocate_optimizer and not cfg.trainer.preallocate_optimizer_state:
        # Allocate the installed AdamW8bit's zero moments before measuring,
        # so even a one-step probe includes steady-state optimizer memory.
        opt = trainer.optimizer
        opt.check_overrides()
        opt.to_gpu()
        opt.initialized = True
        for gi, group in enumerate(opt.param_groups):
            for pi, parameter in enumerate(group['params']):
                opt.init_state(group, parameter, gi, pi)
    if args.resume:
        if args.direction:
            trainer.load(args.resume, mode='switch', optimizer_policy='continue', scheduler_policy='continue')
        else:
            trainer.load(args.resume)
    trainer.export_config()
    def guard_ram():
        while True:
            time.sleep(1)
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:') and int(line.split()[1]) > 300 * 1024**2:
                    (folder/'ram_limit.json').write_text(json.dumps({'reason':'benchmark RSS exceeded 300 GiB; terminated before host exhaustion'}))
                    print('[benchmark] RAM limit reached', flush=True)
                    os._exit(76)
    threading.Thread(target=guard_ram, daemon=True).start()
    loader = T.build_dataloader(cfg, model, smoke=True)
    if args.longest_instruction:
        dataset = loader.dataset
        allowed = set(model.stats_bundle.provenance['train_episodes']['episode_indices'])
        episodes = [ep for ep in dataset.reader.episodes if ep['episode_index'] in allowed]
        texts = list({ep['tasks'][0] for ep in episodes})
        tokenizer = model.backbone.processor.tokenizer
        lengths = dict(zip(texts, map(len, tokenizer(texts, add_special_tokens=False)['input_ids'])))
        longest = max(episodes, key=lambda ep: lengths[ep['tasks'][0]])
        count = min(max(16, args.batch * args.steps), longest['length'] - 16)
        dataset._index = [(longest['episode_index'], frame) for frame in range(count)]
        print('[benchmark] longest train instruction: '+json.dumps({'episode':longest['episode_index'],'tokens':lengths[longest['tasks'][0]],'source':longest['source_dataset']}), flush=True)
    it = iter(loader)
    rows = []
    for step in range(args.steps):
        for dev in range(torch.cuda.device_count()):
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
        data_start = time.monotonic()
        batch = next(it)
        data_seconds = time.monotonic() - data_start
        t0 = time.monotonic()
        print(f'[benchmark] step={step} begin at {time.strftime("%H:%M:%S", time.gmtime())}', flush=True)
        row = trainer.train_step(batch)
        for dev in range(torch.cuda.device_count()):
            torch.cuda.synchronize(dev)
        row['wall_seconds'] = time.monotonic()-t0
        row['batch_size'] = int(batch['action_target'].shape[0])
        row['samples_per_second'] = row['batch_size']/row['wall_seconds']
        row['data_seconds'] = data_seconds
        row['end_to_end_samples_per_second'] = row['batch_size'] / (data_seconds + row['wall_seconds'])
        row['gpu_peak_allocated_gib'] = [torch.cuda.max_memory_allocated(d)/2**30 for d in range(torch.cuda.device_count())]
        row['gpu_peak_reserved_gib'] = [torch.cuda.max_memory_reserved(d)/2**30 for d in range(torch.cuda.device_count())]
        host_stats = torch.cuda.memory.host_memory_stats()
        row['pin_host_alloc_seconds_total'] = host_stats.get('host_alloc_time.total', 0) / 1e6
        row['pin_peak_owned_gib'] = host_stats.get('allocated_bytes.peak', 0) / 2**30
        rows.append(row)
        print('[benchmark] '+json.dumps(row), flush=True)
        (folder/'benchmark.json').write_text(json.dumps({'arguments':vars(args),'batch':args.batch,'checkpoint':args.checkpoint,'offload':args.offload,'gpu_budget_gib':args.gpu_budget,'gpu_budgets_gib':list(cfg.trainer.saved_tensor_gpu_budgets_gib),'stage':cfg.stage.stage,'completed_steps':len(rows),'steps':rows},indent=2))
    print('[benchmark] COMPLETE', flush=True)

if __name__ == '__main__':
    main()
