"""Engineering validation: gradient-checkpointed trunk vs un-checkpointed trunk.

The energy unroll needs ``create_graph=True`` (double backward).  Activation
checkpointing is only safe if it yields the SAME outer gradients as the plain
forward; this probe proves that on the real backbone before a run trusts
``backbone.gradient_checkpointing=true``.

One K=1 / batch=1 forward+backward is run twice on the identical batch and RNG
state (K=1 fits the 48G card without checkpointing, so the reference exists):
once with checkpointing OFF, once ON.  Per-parameter max abs/rel gradient
differences and the loss difference are printed.  PASS == differences at
floating-point noise level (bf16: <= ~1e-2 relative on tiny-magnitude grads).

    python scripts/probe_checkpoint_grads.py \
        --config configs/local4090_20261031_real_smoke.yaml \
        --set solver.train_inner_steps=1 trainer.device=cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="KEY=VALUE dotlist overrides")
    args = ap.parse_args()

    import torch

    from energywam.config import load_config
    from energywam.models.energy_wam import QwenEnergyWAM
    import train as T

    cfg = load_config(args.config, overrides=args.set)
    torch.manual_seed(cfg.trainer.seed)
    device = torch.device(cfg.trainer.device)
    print(f"[probe] config={args.config} overrides={args.set}")

    stats_bundle = T.resolve_training_stats(cfg)
    model = QwenEnergyWAM(cfg, stats_bundle=stats_bundle)
    model.to(device)
    loader = T.build_dataloader(cfg, model, smoke=True)
    batch = next(iter(loader))
    print("[probe] batch ready")

    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state(device)

    def one_pass(ckpt: bool):
        lm = model.backbone._mm.language_model
        if ckpt:
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            lm.gradient_checkpointing_disable()
        model.train()
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(rng_cpu)
        torch.cuda.set_rng_state(rng_cuda, device)
        out = model(dict(batch))
        loss = out["action_loss"]
        loss.backward()
        grads = {
            n: p.grad.detach().float().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return float(loss), grads

    print("[probe] pass 1: checkpointing OFF (reference)")
    loss0, g0 = one_pass(False)
    print(f"[probe] reference loss={loss0:.6f} tensors_with_grad={len(g0)}")
    print("[probe] pass 2: checkpointing ON")
    loss1, g1 = one_pass(True)
    print(f"[probe] checkpoint loss={loss1:.6f} tensors_with_grad={len(g1)}")

    keys0, keys1 = set(g0), set(g1)
    if keys0 != keys1:
        print(f"[probe] FAIL: parameter sets differ: only-ref={keys0 - keys1} only-ckpt={keys1 - keys0}")
        return 1

    worst_abs, worst_rel, worst_name = 0.0, 0.0, ""
    bad = []
    for n in sorted(keys0):
        a, b = g0[n], g1[n]
        d = (a - b).abs().max().item()
        scale = max(a.abs().max().item(), 1e-12)
        r = d / scale
        if d > worst_abs:
            worst_abs = d
        if r > worst_rel:
            worst_rel, worst_name = r, n
        if r > 2e-2 and d > 1e-4:
            bad.append((n, d, r))
    dl = abs(loss0 - loss1)
    print(f"[probe] loss |diff| = {dl:.3e}")
    print(f"[probe] grad max abs diff = {worst_abs:.3e}")
    print(f"[probe] grad max rel diff = {worst_rel:.3e} (at {worst_name})")
    if bad:
        print(f"[probe] FAIL: {len(bad)} tensors exceed tolerance:")
        for n, d, r in bad[:20]:
            print(f"    {n}: abs={d:.3e} rel={r:.3e}")
        return 1
    print("[probe] PASS: checkpointed gradients match the reference")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
