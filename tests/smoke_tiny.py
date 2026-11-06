"""Quick end-to-end tiny smoke (also usable as a script).

Run:  python tests/smoke_tiny.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from energywam.config import load_config
from energywam.data.tiny_data import make_tiny_batch
from energywam.models.energy_wam import QwenEnergyWAM


def main() -> int:
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "tiny_cpu.yaml")
    torch.manual_seed(0)
    model = QwenEnergyWAM(cfg)
    model.train()

    batch = make_tiny_batch(cfg, batch_size=3, seed=7, include_padding=True)
    out = model(batch)
    print("loss:", float(out["action_loss"]))
    for k in sorted(out):
        if k != "action_loss":
            v = out[k]
            print(f"  {k}: {float(v) if v.dim() == 0 else tuple(v.shape)}")

    out["action_loss"].backward()
    grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            grads.append(name)
    print(f"params with nonzero grad: {len(grads)} / {sum(1 for _ in model.parameters())}")
    print("  sample:", grads[:8])

    model.eval()
    pred = model.predict(batch, mode="joint", return_diagnostics=True)
    print("predict keys:", sorted(pred))
    print("normalized_actions shape:", pred["normalized_actions"].shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
