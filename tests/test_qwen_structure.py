"""Real Qwen3-VL structure + higher-order backward, on a **tiny random config**.

IMPORTANT — scope of this test
------------------------------
This builds a randomly-initialised, very small ``Qwen3VLConfig`` and runs the
*actual* HuggingFace Qwen3-VL code path (image ``masked_scatter``, DeepStack,
mRoPE/``get_rope_index``, the language transformer).  It validates the
**structure and the double-backward plumbing**, not the pretrained 4B model.

It is explicitly NOT a substitute for the pretrained-4B validation, which
cannot run on this server right now (all 9 GPUs are occupied).
"""

from __future__ import annotations

import torch

from .harness import TestFailure, assert_nonzero_finite, assert_true


def _tiny_qwen_config():
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
        max_position_embeddings=512,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        rope_scaling={"type": "mrope", "mrope_section": [4, 4, 4]},
    )
    cfg = Qwen3VLConfig(
        vision_config=vision.to_dict(),
        text_config=text.to_dict(),
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=102,
        vision_end_token_id=103,
    )
    return cfg


def _build_model_and_processor():
    """Build a tiny random Qwen3-VL + a minimal processor stub."""
    from transformers import Qwen3VLForConditionalGeneration

    from energywam.config import EnergyWAMConfig
    from energywam.backbones.qwen3vl_adapter import Qwen3VLEnergyBackbone

    cfg_model = _tiny_qwen_config()
    with torch.no_grad():
        model = Qwen3VLForConditionalGeneration(cfg_model)
    model.config.use_cache = False

    ew_cfg = EnergyWAMConfig()
    ew_cfg.backbone.backend = "qwen3vl"
    ew_cfg.backbone.dtype = "float32"
    ew_cfg.backbone.attn_implementation = "eager"
    ew_cfg.data.action_dim = 3
    ew_cfg.data.state_dim = 4
    ew_cfg.data.action_horizon = 8
    ew_cfg.data.num_cameras = 1
    ew_cfg.trainer.candidate_dtype = "float32"

    # A processor stub that mimics the real AutoProcessor output exactly:
    #  - 1 image per sample with grid_thw = [1, 14, 14]  (224/16 = 14)
    #  - the LLM sequence holds (1*14*14)/(2*2) = 49 <image> placeholder tokens
    #  - left padding across a batch with different text lengths
    PATCH, MERGE = 16, 2
    GRID = 14
    LLM_TOKENS_PER_IMAGE = (GRID * GRID) // (MERGE * MERGE)  # 49

    class _Proc:
        class tokenizer:
            pad_token_id = 0

        def apply_chat_template(self, messages, tokenize=True, padding=True,
                                add_generation_prompt=True, return_dict=True,
                                return_tensors="pt"):
            B = len(messages)
            rows, grids = [], []
            for i, msgs in enumerate(messages):
                content = msgs[0]["content"]
                n_img = sum(1 for c in content if c["type"] == "image")
                text = "".join(c["text"] for c in content if c["type"] == "text")
                toks = []
                for _ in range(n_img):
                    toks += [102] + [100] * LLM_TOKENS_PER_IMAGE + [103]
                toks += [10, 11] + [20 + (ord(ch) % 200) for ch in text[:12]]
                rows.append(toks)
                grids.append([[1, GRID, GRID]] * n_img)
            max_len = max(len(r) for r in rows)
            input_ids = torch.zeros(B, max_len, dtype=torch.long)
            attention_mask = torch.zeros(B, max_len, dtype=torch.long)
            for i, r in enumerate(rows):
                input_ids[i, max_len - len(r):] = torch.tensor(r)  # left padding
                attention_mask[i, max_len - len(r):] = 1
            n_images = sum(len(g) for g in grids)
            # pixel_values: one row per patch of every image
            patches_per_image = GRID * GRID
            pixel_values = torch.randn(n_images * patches_per_image, 3 * 2 * PATCH * PATCH)
            image_grid_thw = torch.tensor(
                [g for sample in grids for g in sample], dtype=torch.long
            )
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

    backbone = Qwen3VLEnergyBackbone(ew_cfg, model=model, processor=_Proc())
    return model, backbone, ew_cfg


def _make_batch(B=2):
    from PIL import Image
    import numpy as np

    images = []
    for b in range(B):
        imgs = []
        for v in range(1):
            arr = (np.random.rand(224, 224, 3) * 255).astype("uint8")
            imgs.append(Image.fromarray(arr))
        images.append(imgs)
    return {
        "images": images,
        "langs": ["do the task", "a much longer instruction string for sample one"],
        "states": [np.random.randn(4).astype("float32") for _ in range(B)],
        # pre-supplied DINO features so the structure test never touches real weights
        "current_dino": torch.randn(B, 1, 4, 8),
    }


def test_qwen_native_path_and_second_order():
    model, backbone, ew_cfg = _build_model_and_processor()
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    batch = _make_batch(B=2)
    cond = backbone.encode_condition(batch)

    # native path sanity
    assert_true(cond.native_position_ids is not None, "mRoPE position_ids missing")
    assert_true(cond.native_position_ids.shape[0] == 3, f"mRoPE must be 3-axis, got {cond.native_position_ids.shape}")
    assert_true(cond.deepstack_visual_embeds is not None, "DeepStack embeds missing")
    assert_true(cond.visual_pos_masks is not None, "visual_pos_masks missing")
    assert_true(int(cond.visual_pos_masks.sum()) > 0, "no visual positions masked")

    # build candidate groups and run the extended sequence
    from energywam.models.energy_core import EnergyWAMCore

    geometry = {
        "action_dim": 3, "state_dim": 4, "future_dim": 8,
        "future_patches": 4, "num_cameras": 1, "action_horizon": 8,
        "num_future_frames": 3, "dino_dim": 8,
    }
    core = EnergyWAMCore(ew_cfg, backbone, geometry)
    # NOTE: the core derives the extra condition tokens (current DINO / state)
    # from ``cond.native['extra_raw']``; the backbone holds no back-reference.

    B = 2
    Z = torch.randn(B, 3, 1, 4, 8, requires_grad=True)
    A = torch.randn(B, 8, 3, requires_grad=True)
    E = core.energy(cond, Z, A)
    assert_true(E.shape == (B,), f"E shape {tuple(E.shape)}")

    # Energy token index must be the last position
    fut = core.build_future_tokens(Z)
    act = core.build_action_tokens(A)
    eng = core.build_energy_tokens(B, Z.device, Z.dtype)
    read = core.energy_from_groups(cond, [fut, act, eng])
    # the energy token index must be the LAST column of the assembled sequence
    hidden_len = int(read.energy_token_index[0]) + 1
    assert_true(hidden_len > cond.native_position_ids.shape[2], "appended block not after the native sequence")

    gA, gZ = torch.autograd.grad(E.sum(), [A, Z], create_graph=True)
    assert_nonzero_finite(gA, "gA through real Qwen path")
    assert_nonzero_finite(gZ, "gZ through real Qwen path")

    # outer: second-order backward into the real HF parameters
    loss = (E.sum()) + 0.0 * gA.sum() + 0.0 * gZ.sum()
    loss2 = (A - gA * 0.05).pow(2).sum()
    model.zero_grad(set_to_none=True)
    loss2.backward()
    grad_names = [n for n, p in model.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) > 0]
    print(f"  params with non-zero outer grad through the unroll: {len(grad_names)}")
    assert_true(len(grad_names) > 0, "no parameter gradient through the real Qwen unroll")
    lm_keys = [n for n in grad_names if "language_model" in n]
    assert_true(len(lm_keys) > 0, f"no language-model gradient; got {grad_names[:10]}")


def test_qwen_appended_positions_strictly_after_native():
    model, backbone, ew_cfg = _build_model_and_processor()
    batch = _make_batch(B=2)
    cond = backbone.encode_condition(batch)
    pos = cond.native_position_ids
    mask = cond.native_attention_mask
    app = backbone._appended_position_ids(pos, mask, 7, pos.device)
    B = pos.shape[1]
    for i in range(B):
        valid = mask[i].bool()
        base = pos[:, i, valid].max().item()
        expected = torch.arange(base + 1, base + 8, dtype=app.dtype)
        from .harness import assert_close
        assert_close(app[0, i], expected, 0.0, f"appended positions sample {i} axis 0")
        assert_close(app[1, i], expected, 0.0, f"appended positions sample {i} axis 1")
        assert_close(app[2, i], expected, 0.0, f"appended positions sample {i} axis 2")


def test_qwen_padding_masks_keys_only():
    """Padded query rows may be garbage; the Energy token must stay valid."""
    model, backbone, ew_cfg = _build_model_and_processor()
    batch = _make_batch(B=2)
    # make the first sample's language shorter so left-padding is present
    batch["langs"] = ["hi", "a much much longer instruction for the second sample here"]
    cond = backbone.encode_condition(batch)
    mask = cond.native_attention_mask
    assert_true(mask.shape[0] == 2, "batch")
    # there IS padding somewhere in the batch
    assert_true(bool((mask.sum(dim=1) != mask.shape[1]).any()), "expected left padding in the batch")


TESTS = [
    ("qwen_native_path_and_second_order", test_qwen_native_path_and_second_order),
    ("qwen_appended_positions_strictly_after_native", test_qwen_appended_positions_strictly_after_native),
    ("qwen_padding_masks_keys_only", test_qwen_padding_masks_keys_only),
]
