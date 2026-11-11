#!/usr/bin/env python
"""EnergyWAM training driver.

Usage::

    python scripts/train.py --config configs/bi.yaml
    python scripts/train.py --config configs/joint.yaml \
        --init-from logs/runs/energywam_bi/ckpt.pt --stage-switch-policy reset
    python scripts/train.py --config configs/tiny_cpu.yaml --steps 200

The driver builds the model, the trainer and the data source declared by the
config.  Full training is **never started implicitly** — the shell wrappers
require ``ENERGYWAM_ALLOW_GPU_TRAINING=1`` and an explicit
``CUDA_VISIBLE_DEVICES``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


def build_dataloader(cfg, model, *, smoke: bool = False):
    """Return a streaming DataLoader (real mode) or a list of tiny batches.

    ``smoke=True`` is the ONLY place that limits the sample count.  Full
    training streams the whole dataset through the DataLoader, honouring
    ``trainer.per_device_batch_size`` and ``trainer.num_workers``.

    The statistics bundle on ``model`` (e.g. embedded in a checkpoint that was
    just restored) is passed down to the loader: a restored run never lets a
    different on-disk fit overwrite its coordinate.
    """
    bundle = getattr(model, "stats_bundle", None)
    if cfg.backbone.backend == "tiny":
        from energywam.data.tiny_data import make_tiny_batch

        n = 32 if smoke else 256
        return [
            make_tiny_batch(
                cfg,
                batch_size=cfg.trainer.per_device_batch_size,
                seed=1000 + i,
                standardizer=bundle,
            )
            for i in range(n)
        ]

    from energywam.data.energy_dataset import build_dataloader as _build

    dino = getattr(getattr(model, "backbone", None), "dino", None)
    max_samples = 16 if smoke else None
    loader = _build(
        cfg, dino=dino, max_samples=max_samples, shuffle=not smoke, stats_bundle=bundle
    )
    # Persist the exact normalization used by this data source for serving.
    stats = loader.dataset.action_stats
    key = str(cfg.data.data_mix or loader.dataset.reader.dataset_identity)
    model.norm_stats = {key: stats}
    model.norm_stats_version = stats.version
    return loader


def resolve_training_stats(cfg, *, checkpoint_path=None):
    """Fail-fast statistics resolution BEFORE any heavy model construction.

    For ``normalization_profile='paper_standardized_v1'`` the bundle embedded in
    the checkpoint being resumed/switched-from is authoritative (so a stale
    ``data.norm_stats_bundle_path`` never breaks a restore); otherwise the
    configured bundle file is fitted-checked here.  Returns ``None`` for the
    legacy profile.  A missing/mismatched bundle raises immediately — before
    the 4B VLM or DINO is loaded.
    """
    from energywam.data.standardization import PROFILE_PAPER
    from energywam.training.checkpoint import bundle_from_payload, load_checkpoint

    if cfg.data.normalization_profile != PROFILE_PAPER:
        return None
    bundle = None
    if checkpoint_path:
        payload = load_checkpoint(checkpoint_path)
        bundle = bundle_from_payload(payload)
        if bundle is None:
            raise RuntimeError(
                f"checkpoint {checkpoint_path} has no embedded statistics bundle but "
                "the config asks for the paper-standardized profile; refusing to "
                "guess a coordinate"
            )
    from energywam.data.energy_dataset import require_future_for_stage, resolve_stats_bundle

    return resolve_stats_bundle(
        cfg,
        stats_bundle=bundle,
        require_future=require_future_for_stage(cfg.stage.stage),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="limit the sample count (smoke only)")
    ap.add_argument("--init-from", default=None, help="checkpoint to load before training")
    ap.add_argument("--stage-switch-policy", default="reset", choices=["reset", "continue"])
    ap.add_argument("--resume", default=None, help="same-stage resume checkpoint")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE dotlist overrides")
    args = ap.parse_args()

    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.trainer import EnergyWAMTrainer

    cfg = load_config(args.config, overrides=args.set)
    if args.steps is not None:
        cfg.trainer.max_train_steps = args.steps
    if cfg.trainer.execution_mode == "replicated":
        import os
        os.execv(sys.executable, [sys.executable, str(ROOT / "scripts/train_replicated.py"), *sys.argv[1:]])

    torch.manual_seed(cfg.trainer.seed)
    print(f"[EnergyWAM] run_id={cfg.run_id} stage={cfg.stage.stage} backend={cfg.backbone.backend}")
    print(f"[EnergyWAM] device={cfg.trainer.device} steps={cfg.trainer.max_train_steps}")
    print(f"[EnergyWAM] normalization_profile={cfg.data.normalization_profile}")

    # fail fast on statistics BEFORE any backbone/4B construction
    stats_bundle = resolve_training_stats(cfg, checkpoint_path=(args.resume or args.init_from))
    if stats_bundle is not None:
        cov = (stats_bundle.provenance or {}).get("coverage", {})
        print(
            f"[EnergyWAM] stats bundle fingerprint={stats_bundle.fingerprint[:12]}... "
            f"coverage={cov.get('kind')} eps={stats_bundle.epsilon:g}"
        )

    model = QwenEnergyWAM(cfg, stats_bundle=stats_bundle)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[EnergyWAM] trainable params: {n_train:,} / total {n_total:,}")

    tr = EnergyWAMTrainer(cfg, model)
    tr.export_config()

    if args.resume:
        meta = tr.load(args.resume, mode="resume")
        print(f"[EnergyWAM] resumed from {args.resume} at step {meta.global_step}")
    elif args.init_from:
        meta = tr.switch_stage(cfg.stage.stage, args.init_from, optimizer_policy=args.stage_switch_policy)
        print(
            f"[EnergyWAM] stage switch from {meta.stage} -> {cfg.stage.stage} "
            f"(optimizer={meta.optimizer_policy}, scheduler={meta.scheduler_policy})"
        )

    smoke = bool(args.smoke) or cfg.run_id.endswith("real_smoke") or cfg.run_id.endswith("_smoke")
    loader = build_dataloader(cfg, model, smoke=smoke)
    n_items = len(loader) if hasattr(loader, "__len__") else -1
    print(f"[EnergyWAM] data source ready ({'smoke-limited' if smoke else 'full stream'}, "
          f"{n_items} batches/epoch)")

    # CLI --steps is the target stage step, including steps already restored.
    # Trainer.fit itself retains its API of running N additional steps.
    remaining = max(0, cfg.trainer.max_train_steps - tr.global_step)
    info = tr.fit(loader, max_steps=remaining)
    print(f"[EnergyWAM] finished {info['steps']} steps in {info['seconds']}s")
    if info["loss_history"]:
        print(
            f"[EnergyWAM] action_loss {info['loss_history'][0]['action_loss']:.4f} "
            f"-> {info['loss_history'][-1]['action_loss']:.4f}"
        )
    ckpt = tr.save(f"{cfg.run_id}_step{tr.global_step}.pt")
    print(f"[EnergyWAM] checkpoint: {ckpt}")
    Path(cfg.trainer.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.trainer.output_dir) / "final_report.json").write_text(json.dumps(info, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
