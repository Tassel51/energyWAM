"""Fixtures for REAL-path tests: full ``QwenEnergyWAM`` on a small random Qwen3-VL
plus the **real** frozen DINOv3 checkpoint.

These build the *top-level* ``QwenEnergyWAM`` (not the adapter in isolation), so
the tests exercise .to/.train/state_dict/forward/predict/checkpoint the way a
real run would.

Scope: structure + gradient plumbing.  This is explicitly NOT a pretrained-4B
validation — the 4B weights are not loaded and no GPU is used.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DINO_PATH = "/data6/zzc/dinov3-vits16-pretrain-lvd1689m"


def tiny_qwen_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )

    vision = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        num_heads=4,
        intermediate_size=64,
        out_hidden_size=48,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=64,
        deepstack_visual_indexes=[0, 1],
    )
    text = Qwen3VLTextConfig(
        vocab_size=256,
        hidden_size=48,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=12,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        rope_scaling={"type": "mrope", "mrope_section": [4, 4, 4]},
        attention_dropout=0.0,
        hidden_dropout=0.0,
        mlp_dropout=0.0,
    )
    return Qwen3VLConfig(
        vision_config=vision.to_dict(),
        text_config=text.to_dict(),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )


# The stub mirrors the real AutoProcessor output contract:
#   pixel_values  [num_images * patches, 3 * temporal_patch * patch * patch]
#   image_grid_thw [num_images, 3]
#   left-padded input_ids carrying <vision_start> + N <image> + <vision_end>
class StubQwen3VLProcessor:
    """Local processor producing native-looking Qwen3-VL batches.

    Image geometry is derived from the same numbers the real processor uses
    (patch 16, temporal patch 2, spatial merge 2) so the vision tower and
    ``get_rope_index`` see a consistent grid.
    """

    PATCH = 16
    TEMPORAL = 2
    MERGE = 2

    class tokenizer:
        pad_token_id = 0

    def __init__(self, grid: int = 14):
        self.grid = grid

    @property
    def llm_tokens_per_image(self) -> int:
        return (self.grid * self.grid) // (self.MERGE * self.MERGE)

    def apply_chat_template(self, messages, tokenize=True, padding=True,
                            add_generation_prompt=True, return_dict=True,
                            return_tensors="pt"):
        B = len(messages)
        rows, grids = [], []
        for msgs in messages:
            content = msgs[0]["content"]
            n_img = sum(1 for c in content if c["type"] == "image")
            text = "".join(c["text"] for c in content if c["type"] == "text")
            toks = []
            for _ in range(n_img):
                toks += [102] + [100] * self.llm_tokens_per_image + [103]
            toks += [10, 11] + [20 + (ord(ch) % 200) for ch in text[:12]]
            rows.append(toks)
            grids.append([[1, self.grid, self.grid]] * n_img)
        max_len = max(len(r) for r in rows)
        input_ids = torch.zeros(B, max_len, dtype=torch.long)
        attention_mask = torch.zeros(B, max_len, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, max_len - len(r):] = torch.tensor(r)
            attention_mask[i, max_len - len(r):] = 1
        n_images = sum(len(g) for g in grids)
        patches = self.grid * self.grid
        # DETERMINISTIC pixel values derived from the actual input images, so a
        # state_dict round-trip sees identical native inputs (a torch.randn here
        # would make every call different and hide real loading bugs).
        flat_imgs = [c["image"] for msgs in messages for c in msgs[0]["content"]
                     if c["type"] == "image"]
        cols = 3 * self.TEMPORAL * self.PATCH * self.PATCH
        rows_px = torch.zeros(n_images * patches, cols)
        for i, img in enumerate(flat_imgs[:n_images]):
            arr = np.asarray(img).astype("uint8")
            if arr.ndim == 3 and arr.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"expected HxWx3 image, got {arr.shape}")
            g = self.grid * self.PATCH
            if arr.shape[0] != g or arr.shape[1] != g:
                arr = np.asarray(
                    Image.fromarray(arr).resize((g, g), Image.BICUBIC), dtype="uint8"
                )
            flat = torch.from_numpy(arr.astype("float32") / 255.0).reshape(-1)
            # seed from the image content so the same image always yields the
            # same synthetic patch pixels
            seed = int(flat[:64].sum().item() * 1e6) % (2**31 - 1)
            gen = torch.Generator().manual_seed(seed)
            rows_px[i * patches : (i + 1) * patches] = (
                flat[:cols] if flat.numel() >= cols
                else flat.repeat((cols // flat.numel()) + 1)[:cols]
            ).unsqueeze(0).expand(patches, cols).clone()
            rows_px[i * patches : (i + 1) * patches] += 0.01 * torch.randn(
                patches, cols, generator=gen
            )
        pixel_values = rows_px
        image_grid_thw = torch.tensor([g for sample in grids for g in sample], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
        }


class StubDINO(torch.nn.Module):
    """Deterministic stand-in for the frozen DINO when the real checkpoint is
    not wanted.  Same interface as :class:`FrozenDINOv3`, small geometry."""

    def __init__(self, patches: int = 4, dim: int = 8):
        super().__init__()
        self.patches = patches
        self.hidden_dim = dim
        self.image_size = 224
        self.proj = torch.nn.Conv2d(3, dim, kernel_size=112, stride=112)

    def encode_images(self, images, device="cpu"):
        import numpy as _np

        self.to(device)
        out = []
        for img in images:
            a = _np.asarray(img).astype("float32") / 255.0
            if a.ndim == 2:
                a = _np.stack([a] * 3, axis=-1)
            t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(device)
            if t.shape[-1] != 224:
                t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            f = self.proj(t).flatten(2).transpose(1, 2)  # [1, patches, dim]
            out.append(f)
        return torch.cat(out, dim=0).detach()


def make_images(B: int, V: int = 1, size: int = 224, seed: int = 0):
    """Native synthetic RGB images as uint8 HxWx3 numpy arrays."""
    rng = np.random.RandomState(seed)
    return [
        [rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8) for _ in range(V)]
        for _ in range(B)
    ]


def build_real_model(
    *,
    B: int = 2,
    with_real_dino: bool = True,
    seed: int = 0,
    device: str = "cpu",
    stats_bundle=None,
    **cfg_overrides,
):
    """Build a complete ``QwenEnergyWAM`` for the real path.

    ``stats_bundle`` (a paper §4.8 ``NormalizationBundle``) selects the
    standardized coordinate; without it the model keeps the legacy contract.
    """
    from transformers import Qwen3VLForConditionalGeneration

    from energywam.backbones.dino_v3 import FrozenDINOv3
    from energywam.backbones.qwen3vl_adapter import Qwen3VLEnergyBackbone
    from energywam.config import EnergyWAMConfig
    from energywam.models.energy_wam import QwenEnergyWAM

    torch.manual_seed(seed)
    cfg_model = tiny_qwen_config()
    model = Qwen3VLForConditionalGeneration(cfg_model)
    model.config.use_cache = False

    cfg = EnergyWAMConfig()
    cfg.backbone.backend = "qwen3vl"
    cfg.backbone.dtype = "float32"
    cfg.backbone.attn_implementation = "eager"
    cfg.data.action_dim = 3
    cfg.data.state_dim = 4
    cfg.data.action_horizon = 8
    cfg.data.num_cameras = 1
    cfg.data.future_offsets = [3, 5, 8]
    cfg.trainer.candidate_dtype = "float32"
    cfg.trainer.device = device
    cfg.solver.train_inner_steps = 2
    cfg.solver.eval_inner_steps = 2
    cfg.solver.alpha_action = 0.1
    cfg.solver.alpha_future = 0.1
    for k, v in cfg_overrides.items():
        node = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            node = getattr(node, p)
        setattr(node, parts[-1], v)

    if with_real_dino:
        if not os.path.exists(DINO_PATH):
            raise FileNotFoundError(f"real DINO checkpoint missing: {DINO_PATH}")
        dino = FrozenDINOv3(DINO_PATH, freeze=True, dtype=torch.float32)
    else:
        dino = StubDINO(patches=4, dim=8)

    processor = StubQwen3VLProcessor()
    if cfg.backbone.condition_mode == "text_dino_state":
        class TextTokenizer:
            def __call__(self, texts, padding=True, return_tensors="pt"):
                rows = [[10]+[20+ord(c)%70 for c in t[:12]] for t in texts]
                n = max(map(len, rows))
                ids = torch.zeros(len(rows), n, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for i, row in enumerate(rows):
                    ids[i, -len(row):] = torch.tensor(row)
                    mask[i, -len(row):] = 1
                return dict(input_ids=ids, attention_mask=mask)
        processor = TextTokenizer()
    backbone = Qwen3VLEnergyBackbone(cfg, model=model, processor=processor, dino=dino)
    ew = QwenEnergyWAM(cfg, backbone=backbone, stats_bundle=stats_bundle)

    geometry = dict(ew.geometry)
    if with_real_dino:
        # keep the real DINO geometry: 196 patches x 384 dims
        geometry["future_patches"] = 196
        geometry["future_dim"] = 384
        geometry["dino_dim"] = 384
    else:
        geometry["future_patches"] = 4
        geometry["future_dim"] = 8
        geometry["dino_dim"] = 8
    ew.geometry = geometry
    # rebuild the core projectors against the real geometry
    from energywam.models.energy_core import EnergyWAMCore

    ew.core = EnergyWAMCore(cfg, backbone, geometry)
    # Training and prediction must exercise the same rebuilt core, including
    # when the stub DINO changes the candidate geometry to 4 patches x 8 dims.
    ew.stage_runner.core = ew.core

    ew = ew.to(device)
    return ew, cfg, geometry


def make_real_batch(ew, cfg, geometry, B: int = 2, seed: int = 1, standardizer=None):
    """A full training batch for the real path, including Z* from real DINO.

    With ``standardizer`` (a paper §4.8 ``NormalizationBundle``) the action /
    future targets are converted into the standardized coordinate and the batch
    carries the ``*_target_space`` markers the model verifies.
    """
    from energywam.data.energy_dataset import collate_energy_batch

    images = make_images(B, V=cfg.data.num_cameras, seed=seed)
    states = [np.random.RandomState(seed + i).randn(cfg.data.state_dim).astype("float32") for i in range(B)]

    # Z* from the REAL frozen DINO
    offsets = list(cfg.data.future_offsets)
    Z_list, z_mask_list = [], []
    for b in range(B):
        fut = []
        for k, off in enumerate(offsets):
            fut.append([images[b][v] for v in range(cfg.data.num_cameras)])
        flat = [img for frame in fut for img in frame]
        with torch.no_grad():
            feats = ew.backbone.dino.encode_images(flat, device="cpu")
        feats = feats.detach().float()
        F, V, P, D = len(offsets), cfg.data.num_cameras, feats.shape[1], feats.shape[2]
        Z_list.append(feats.view(F, V, P, D).numpy())
        z_mask_list.append(np.ones((F, V, P), dtype="float32"))

    A = np.random.RandomState(seed).randn(B, cfg.data.action_horizon, cfg.data.action_dim).astype("float32")
    Z = np.stack(Z_list)
    out = {
        "images": images,
        "langs": ["do the thing", "a considerably longer instruction string"][:B],
        "states": states,
        "action_target": A,
        "action_mask": np.ones_like(A),
        "future_target": Z,
        "future_mask": np.stack(z_mask_list),
        "future_offsets": np.tile(np.asarray(offsets, dtype="int64"), (B, 1)),
    }
    if standardizer is not None:
        out["action_target"] = standardizer.action.normalize(A).astype("float32")
        out["future_target"] = standardizer.future.normalize(Z).astype("float32")
        out["action_target_space"] = "standardized"
        out["future_target_space"] = "standardized"
    return out
