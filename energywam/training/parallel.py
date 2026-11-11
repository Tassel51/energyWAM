"""Layer model parallelism with differentiable inter-device transfers."""
import torch


def shard_language_model(model, devices, primary, *, layer_counts=None):
    from accelerate import dispatch_model

    devices = [int(d) for d in devices]
    if len(devices) < 2 or len(set(devices)) != len(devices):
        raise ValueError("language_model_devices requires at least two distinct GPUs")
    if any(d < 0 or d >= torch.cuda.device_count() for d in devices):
        raise ValueError(f"unavailable CUDA device in {devices}")
    if model.cfg.backbone.backend != "qwen3vl":
        raise ValueError("language model sharding requires qwen3vl")
    lm = model.backbone._mm.language_model
    # Embeddings, rotary positions and final norm stay with conditioning/head.
    device_map = {name: str(primary) for name, _ in lm.named_children() if name != "layers"}
    if layer_counts:
        if len(layer_counts)!=len(devices) or any(type(n) is not int or n<1 for n in layer_counts) or sum(layer_counts)!=len(lm.layers):
            raise ValueError('Layer counts must be positive integers per device, summing to all language layers')
        placements=[d for d,n in zip(devices,layer_counts) for _ in range(n)]
    else:
        placements=[devices[i*len(devices)//len(lm.layers)] for i in range(len(lm.layers))]
    for i,d in enumerate(placements):
        device_map[f"layers.{i}"] = f"cuda:{d}"
    dispatch_model(lm, device_map=device_map, main_device=str(primary))
    print(f"[EnergyWAM] differentiable language layer sharding: {device_map}", flush=True)
