#!/usr/bin/env python
"""EnergyWAM offline evaluation driver.

Reports **offline action MSE against the demonstration** on an explicitly
labelled subset (``--max-samples``, default 32).  This is NOT a closed-loop
success rate; the report says so in its own fields.

For closed-loop RoboCasa365 success rate use the existing StarVLA evaluation
entry (``examples/simBenchmarks/Robocasa_365/eval_files/run_eval.sh``) —
EnergyWAM does not reimplement or reinterpret that protocol.

The real-data path builds examples through
``energywam.data.energy_dataset.build_eval_examples`` — the same sample
contract as training (``data_mix`` root resolution, configured camera order,
action statistics normalisation, ``sincos`` state), with ``require_future=False``
so offline evaluation never decodes real future frames and needs no DINO
extractor.

Usage::

    python scripts/evaluate.py --config configs/eval.yaml --checkpoint path/to.pt
    python scripts/evaluate.py --config configs/tiny_cpu.yaml --synthetic 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))



def describe_eval_split(manifest, episode_indices, stats_bundle=None):
    """Claim held-out status only after checking the saved training split."""
    from energywam.data.standardization import StatsError

    split = str(manifest.get("split", "")).lower()
    recorded = ((getattr(stats_bundle, "provenance", None) or {})
                .get("train_episodes", {}).get("episode_indices"))
    checked = isinstance(recorded, list) and bool(recorded)
    if checked and split in ("val", "test"):
        overlap = set(map(int, recorded)) & set(map(int, episode_indices))
        if overlap:
            raise StatsError(
                f"held-out {split} episodes overlap the checkpoint training split: {sorted(overlap)}"
            )
    guaranteed = checked and split in ("val", "test")
    return {
        "explicit_selection": True,
        "guaranteed_heldout": guaranteed,
        "train_overlap_checked": checked,
        "split": split,
        "episode_indices": list(episode_indices),
        "note": (
            f"held-out {split} selection verified disjoint from checkpoint train episodes"
            if guaranteed else
            f"NOT guaranteed held-out: declared split={split!r}; training provenance unavailable or train selected"
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--synthetic", type=int, default=0,
                    help="use N synthetic tiny batches instead of a real dataset")
    ap.add_argument("--max-samples", type=int, default=32,
                    help="offline evaluation SUBSET size (default 32). This is a "
                         "subset of the demonstration data, not a closed-loop run.")
    ap.add_argument("--mode", default="joint",
                    choices=["joint", "action_given_future", "future_given_action", "action_only"])
    ap.add_argument("--eval-episodes-manifest", default=None,
                    help="explicit episode manifest selecting the evaluation episodes "
                         "(e.g. a val/test split).  Without it the evaluation is NOT "
                         "guaranteed held-out and the report says so.")
    ap.add_argument("--one-per-task", action="store_true", help="one fixed held-out window per task")
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    import torch

    from energywam.config import load_config
    from energywam.inference.predictor import EnergyWAMPredictor, offline_eval
    from energywam.models.energy_wam import QwenEnergyWAM
    from energywam.training.checkpoint import bundle_from_payload, load_checkpoint, restore

    cfg = load_config(args.config, overrides=args.set)
    device = torch.device(cfg.trainer.device)

    payload = load_checkpoint(args.checkpoint) if args.checkpoint else None
    # paper profile: the checkpoint's embedded statistics bundle is
    # authoritative (external files are only cross-checked, never trusted over it)
    stats_bundle = bundle_from_payload(payload) if payload is not None else None
    model = QwenEnergyWAM(cfg, stats_bundle=stats_bundle).to(device)
    if cfg.trainer.language_model_devices:
        from energywam.training.parallel import shard_language_model

        shard_language_model(model, cfg.trainer.language_model_devices, device)
    if payload is not None:
        restore(payload, model=model, mode="resume", restore_rng=False)
        print(f"[EnergyWAM eval] loaded {args.checkpoint}")
    else:
        print("[EnergyWAM eval] WARNING: evaluating a randomly-initialised model (no --checkpoint).")
    print(f"[EnergyWAM eval] normalization_profile={cfg.data.normalization_profile}")

    action_stats = None
    source = "real_dataset"
    heldout = {
        "explicit_selection": False,
        "guaranteed_heldout": False,
        "split": None,
        "note": (
            "未保证heldout / NOT guaranteed held-out: no explicit held-out episode "
            "selection was made, so the evaluated episodes may overlap the training "
            "split.  These numbers must NOT be reported as held-out results."
        ),
    }
    if args.eval_episodes_manifest and (args.synthetic or cfg.backbone.backend == "tiny"):
        raise SystemExit(
            "--eval-episodes-manifest selects dataset episodes and cannot be combined "
            "with the synthetic tiny path"
        )
    if args.synthetic or cfg.backbone.backend == "tiny":
        from energywam.data.tiny_data import make_tiny_batch

        bundle = getattr(model, "stats_bundle", None)
        n = args.synthetic or args.max_samples
        source = "synthetic_tiny"
        samples = []
        for i in range(n):
            b = make_tiny_batch(cfg, batch_size=1, seed=7000 + i, standardizer=bundle)
            # per-example keys in the predict_action example contract; the tiny
            # backend takes its condition from ``cond_vec``.  The target is in
            # the model action space (zscore under the paper profile, synthetic
            # normalized values otherwise — no physical units either way).
            samples.append(
                {
                    "cond_vec": b["cond_vec"][0],
                    "action": b["action_target"][0],
                    "action_mask": b["action_mask"][0],
                    "action_target_space": b.get("action_target_space", "model_space"),
                }
            )
    else:
        from energywam.data.energy_dataset import build_eval_examples

        eval_episode_indices = None
        if args.eval_episodes_manifest:
            from energywam.data.energy_dataset import (
                LeRobotEpisodeReader,
                resolve_dataset_root,
            )
            from energywam.data.stats_fit import load_episode_manifest

            reader = LeRobotEpisodeReader(resolve_dataset_root(cfg))
            man, eval_episode_indices = load_episode_manifest(
                args.eval_episodes_manifest, reader=reader, expect_split=None
            )
            heldout = describe_eval_split(man, eval_episode_indices, stats_bundle)
        if args.one_per_task:
            if not args.eval_episodes_manifest:
                raise ValueError('--one-per-task requires a held-out episode manifest')
            selected = {}
            allowed = set(eval_episode_indices)
            for ep in reader.episodes:
                if ep['episode_index'] in allowed:
                    key = ep.get('source_dataset') or ep['tasks'][0]
                    selected.setdefault(key, ep['episode_index'])
            samples = []
            for key, ep in sorted(selected.items()):
                part, action_stats = build_eval_examples(
                    cfg, reader=reader, max_samples=1, stats_bundle=stats_bundle,
                    episode_indices=[ep])
                samples.extend(part)
            args.max_samples = len(samples)
        else:
            samples, action_stats = build_eval_examples(
                cfg, max_samples=args.max_samples, stats_bundle=stats_bundle,
                episode_indices=eval_episode_indices,
            )

    # Fixed candidate initialization makes checkpoint comparisons reproducible.
    torch.manual_seed(cfg.trainer.seed)
    predictor = EnergyWAMPredictor(model, action_stats=action_stats)
    out_path = Path(args.out or (ROOT / "logs" / "eval_report.json"))
    report = offline_eval(
        predictor,
        samples,
        mode=args.mode,
        log_path=out_path,
        extra={
            "subset": {
                "source": source,
                "max_samples": int(args.max_samples),
                "num_samples_evaluated": len(samples),
                "is_full_dataset": False,
                "note": "Offline subset of the demonstration data. Not a closed-loop run.",
            },
            "device": str(device),
            "backend": cfg.backbone.backend,
            "normalization_profile": cfg.data.normalization_profile,
            "heldout": heldout,
            "action_target_space": (
                "normalized_with_training_stats" if action_stats is not None else "normalized_synthetic_no_physical_units"
            ),
        },
    )
    print(json.dumps(report, indent=2))
    print(f"[EnergyWAM eval] report written to {out_path}")
    if args.one_per_task:
        report['per_task_action_mse'] = dict(zip(sorted(selected), report['per_sample_mse']))
        out_path.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
