#!/usr/bin/env python
"""Probe a real LeRobot dataset and print a de-identified sample report.

Run:
    python scripts/data_probe.py --root /data6/zzc/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/<task>
    python scripts/data_probe.py --root <robocasa365 task dir> --offsets 5 11 16 --horizon 16

The report is written to ``logs/data_probe_<dataset>.json`` and printed.
It never prints raw sensor values beyond min/max, only shapes and indices.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from energywam.data.episode_dataset import DatasetSpec, LeRobotEpisodeReader, probe_one_sample


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--offsets", type=int, nargs="+", default=[5, 11, 16])
    ap.add_argument("--num-cameras", type=int, default=1)
    ap.add_argument("--no-frames", action="store_true", help="skip video decoding (faster)")
    args = ap.parse_args()

    spec = DatasetSpec(
        action_horizon=args.horizon,
        future_offsets=tuple(args.offsets),
        num_cameras=args.num_cameras,
    )

    if args.no_frames:
        reader = LeRobotEpisodeReader(args.root)
        ep = int(reader.episodes[0]["episode_index"])
        L = reader.episode_length(ep)
        actions = reader.read_actions(ep)
        states = reader.read_states(ep)
        report = {
            "dataset_identity": reader.dataset_identity,
            "episode_index": ep,
            "episode_length": L,
            "action_shape": list(actions.shape),
            "state_shape": list(states.shape),
            "cameras": list(reader.camera_keys),
            "num_episodes": len(reader.episodes),
            "info_keys": sorted(reader.info.get("features", {}).keys()),
        }
    else:
        report = probe_one_sample(args.root, spec)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(__file__).resolve().parents[1] / "logs"
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.root).name.replace("/", "_")
    (out / f"data_probe_{name}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"-> {out / f'data_probe_{name}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
