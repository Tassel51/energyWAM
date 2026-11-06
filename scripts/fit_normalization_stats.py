#!/usr/bin/env python
"""Fit the paper §4.8 training-split statistics bundle (offline, read-only).

Produces the versioned JSON bundle consumed by
``data.normalization_profile='paper_standardized_v1'``: action mean/std/count
per action dimension and future mean/std/count per DINO channel, plus the full
provenance (train episode manifest + hash, dataset identity, ordered cameras,
H/offsets, DINO identity/config hash, preprocess version, dimensions, coverage,
content fingerprint).

Guarantees
----------
* **train split only** — ``--split`` must be ``train``; ``val`` / ``test`` are
  rejected, and episodes listed in ``--val-manifest`` / ``--test-manifest`` are
  refused as overlaps;
* an explicit training ``--episodes-manifest`` is required (the manifest format
  is documented in docs/STANDARDIZED_BASELINE.md) and every listed episode is
  verified to exist in the dataset;
* accumulation is streaming Welford/Chan in float64 with valid masks; the
  dataset is opened read-only and nothing is written but ``--out``;
* existing output is never overwritten unless ``--overwrite`` is given;
* ``--max-samples`` marks the bundle ``coverage=subset`` — it is NOT a full
  training-split statistic;
* ``--no-future`` fits action statistics only (action_only training does not
  read future ground truth);
* only the frozen DINO is loaded for feature extraction — **no Qwen**;
* ``--synthetic-tiny`` produces an explicitly **synthetic** bundle for the tiny
  test mode (marked ``coverage=synthetic_train_split``), never a real statistic.

This script is never run implicitly; full-dataset fitting is an explicit,
potentially long operation.

Usage::

    python scripts/fit_normalization_stats.py --config configs/bi.yaml \\
        --split train --episodes-manifest manifests/train_episodes.json \\
        --val-manifest manifests/val_episodes.json \\
        --out stats/robocasa365_open_drawer_target_human_paper_standardized_v1.json

    python scripts/fit_normalization_stats.py --config configs/tiny_paper_standardized.yaml \\
        --split train --synthetic-tiny --overwrite \\
        --out configs/tiny_paper_standardized.synthetic_bundle.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="run config (geometry / DINO / epsilon)")
    ap.add_argument("--split", required=True,
                    help="dataset split to fit on — only 'train' is accepted")
    ap.add_argument("--episodes-manifest", default=None,
                    help="JSON manifest listing the TRAIN episodes (required unless --synthetic-tiny)")
    ap.add_argument("--val-manifest", default=None, help="optional val manifest; its episodes must not overlap")
    ap.add_argument("--test-manifest", default=None, help="optional test manifest; its episodes must not overlap")
    ap.add_argument("--out", required=True, help="output bundle JSON path")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing an existing --out file")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="SMOKE ONLY: limit the number of windows; the bundle is then marked coverage=subset")
    ap.add_argument("--no-future", action="store_true",
                    help="fit action statistics only (action_only: no future images are read)")
    ap.add_argument("--synthetic-tiny", action="store_true",
                    help="produce an explicitly SYNTHETIC bundle for the tiny test mode (no dataset, no DINO)")
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE dotlist config overrides")
    ap.add_argument("--device", default="cpu",
                    help="device for the frozen DINO body (e.g. cuda, cuda:1); "
                         "features are still returned on CPU for accumulation. "
                         "cpu (default) keeps the original CPU-only behaviour.")
    ap.add_argument("--dino-batch-size", type=int, default=1,
                    help="batch frozen DINO frames while fitting identical masked windows")
    args = ap.parse_args()

    from energywam.config import load_config
    from energywam.data.standardization import StatsError, save_bundle

    # --- split guard: statistics are fitted on TRAIN only -------------------
    if str(args.split).lower() != "train":
        raise SystemExit(
            f"[fit-stats] refusing to fit on split={args.split!r}: statistics come from "
            "the TRAIN split only (paper §6.3); val/test are used for evaluation, never "
            "for fitting."
        )

    cfg = load_config(args.config, overrides=args.set)
    out_path = Path(args.out)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(
            f"[fit-stats] refusing to overwrite existing {out_path}; pass --overwrite "
            "explicitly to replace it."
        )

    if args.synthetic_tiny:
        from energywam.data.tiny_data import synthetic_train_bundle

        bundle = synthetic_train_bundle(
            cfg,
            n_windows=int(args.max_samples or 512),
            epsilon=float(cfg.data.standardization_epsilon),
            include_future=not args.no_future,
        )
        print("[fit-stats] SYNTHETIC tiny bundle (not a real dataset statistic)")
    else:
        if not args.episodes_manifest:
            raise SystemExit(
                "[fit-stats] --episodes-manifest is required: statistics must be fitted "
                "on an explicit TRAIN episode list (see docs/STANDARDIZED_BASELINE.md "
                "for the manifest format)."
            )
        from energywam.data.energy_dataset import resolve_dataset_root
        from energywam.data.stats_fit import (
            build_dino_extractor,
            fit_bundle_from_reader,
            load_train_manifest,
        )
        from energywam.data.episode_dataset import LeRobotEpisodeReader

        reader = LeRobotEpisodeReader(resolve_dataset_root(cfg))
        others = [p for p in (args.val_manifest, args.test_manifest) if p]
        manifest, episodes = load_train_manifest(
            args.episodes_manifest, reader=reader, other_manifest_paths=others
        )
        print(f"[fit-stats] dataset {reader.dataset_identity}: {len(episodes)} train episodes")
        dino = None if args.no_future else build_dino_extractor(cfg)
        if dino is not None and str(args.device) != "cpu":
            dino.to(args.device)
        bundle = fit_bundle_from_reader(
            reader,
            cfg,
            episodes=episodes,
            dino=dino,
            include_future=not args.no_future,
            max_samples=args.max_samples,
            epsilon=float(cfg.data.standardization_epsilon),
            manifest=manifest,
            dino_batch_size=args.dino_batch_size,
        )

    save_bundle(bundle, out_path, overwrite=args.overwrite)
    cov = (bundle.provenance or {}).get("coverage", {})
    print(f"[fit-stats] wrote {out_path}")
    print(f"[fit-stats] fingerprint={bundle.fingerprint}")
    print(f"[fit-stats] coverage={cov.get('kind')} windows={cov.get('num_windows')}")
    print(f"[fit-stats] action dim={bundle.action.dim} count~{float(bundle.action.count.mean()):.1f}")
    if bundle.future is not None:
        print(f"[fit-stats] future dim={bundle.future.dim} count~{float(bundle.future.count.mean()):.1f}")
    else:
        print("[fit-stats] future statistics: NONE (--no-future)")
    print(f"[fit-stats] epsilon={bundle.epsilon:g} coordinate={bundle.coordinate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
