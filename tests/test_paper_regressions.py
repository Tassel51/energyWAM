"""Regression tests for the paper-audit fixes (2026-10-29).

Each test reproduces the exact pre-fix behaviour and must FAIL against it:

* F1 ``lm_head`` — an untied (unused) output head stayed trainable; a tied
  head must never be frozen because its weight IS the shared text embedding.
* F2 ``FrozenDINOv3.encode_images`` moved the shared module (``self.to``),
  dragging the model's frozen encoder to whatever device a caller named.
* F3 ``initialization_future='current_dino_noise'`` was unreachable through the
  real training forward / ``predict`` (``current_dino`` is never populated
  there), so it raised "requires current_dino features".
* F4 the ``action_only`` training data path still decoded future RGB and ran
  frozen DINO on it although the stage cannot read any future.
"""

from __future__ import annotations

import numpy as np
import torch

from .harness import TestFailure, assert_close, assert_true


# ---------------------------------------------------------------------------
# F1 — lm_head freezing policy
# ---------------------------------------------------------------------------
def _headless_freeze_setup(tie: bool):
    """Tiny Qwen3-VL with ``tie_word_embeddings`` set as requested."""
    from transformers import Qwen3VLForConditionalGeneration

    from energywam.backbones.qwen3vl_adapter import Qwen3VLEnergyBackbone
    from energywam.config import EnergyWAMConfig
    from .real_fixtures import StubDINO, StubQwen3VLProcessor, tiny_qwen_config

    cfg_model = tiny_qwen_config()
    cfg_model.text_config.tie_word_embeddings = bool(tie)
    cfg_model.tie_word_embeddings = bool(tie)
    model = Qwen3VLForConditionalGeneration(cfg_model)
    model.tie_weights()  # mirror from_pretrained post-init
    model.config.use_cache = False

    cfg = EnergyWAMConfig()
    cfg.backbone.backend = "qwen3vl"
    cfg.backbone.attn_implementation = "eager"
    cfg.data.action_dim = 3
    cfg.data.state_dim = 4
    cfg.backbone.tune_language_model = True
    backbone = Qwen3VLEnergyBackbone(
        cfg, model=model, processor=StubQwen3VLProcessor(), dino=StubDINO(patches=4, dim=8)
    )
    return model, backbone


def test_lm_head_frozen_when_untied_and_unused():
    """F1: a standalone (untied) lm_head is never used by the energy path and
    must not be counted as trainable.  Pre-fix it stayed requires_grad=True."""
    model, backbone = _headless_freeze_setup(tie=False)
    lm_head = model.lm_head
    emb = model.model.language_model.embed_tokens.weight
    assert_true(all(p is not emb for p in lm_head.parameters()),
                "test setup: expected an UNTIED lm_head")
    for n, p in lm_head.named_parameters():
        assert_true(not p.requires_grad,
                    f"unused untied lm_head.{n} is still trainable (pre-fix behaviour)")
    # the text embeddings the language model DOES use must stay trainable
    assert_true(emb.requires_grad,
                "embed_tokens was frozen — text input embeddings must stay trainable "
                "when tune_language_model=True")


def test_lm_head_tied_never_freezes_shared_embedding():
    """F1 guard: with ``tie_word_embeddings=True`` (the real Qwen3-VL-4B config)
    ``lm_head.weight`` IS ``embed_tokens.weight``; freezing it would silently
    freeze the text embeddings."""
    model, backbone = _headless_freeze_setup(tie=True)
    emb = model.model.language_model.embed_tokens.weight
    assert_true(model.lm_head.weight is emb,
                "test setup: expected a TIED lm_head (as in Qwen3-VL-4B-Instruct)")
    assert_true(emb.requires_grad,
                "tied lm_head/embed_tokens was frozen — shared embedding must keep "
                "the language-model trainability policy")


# ---------------------------------------------------------------------------
# F2 — shared DINO must not be moved by encode_images
# ---------------------------------------------------------------------------
def _spy_dino():
    """FrozenDINOv3 with a recording ``.to`` and a fake body (no real weights)."""
    from energywam.backbones.dino_v3 import FrozenDINOv3

    class FakeCfg:
        model_type = "dinov3"
        hidden_size = 384
        patch_size = 16
        num_register_tokens = 4

    class FakeBody(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeCfg()
            self.lin = torch.nn.Linear(3 * 16 * 16, 384)  # real params -> .to() would move them

        def forward(self, pixel_values=None, **kw):
            n = pixel_values.shape[0]
            total = 5 + (224 // 16) ** 2
            out = torch.zeros(n, total, 384, device=pixel_values.device)
            return type("Out", (), {"last_hidden_state": out})()

    class SpyDINO(FrozenDINOv3):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.model_path = "spy"
            self.image_size = 224
            self.patch_size = 16
            self.hidden_dim = 384
            self.num_special_tokens = 5
            self.num_patches = 196
            self._dtype = torch.float32
            self.body = FakeBody()
            self.to_calls = []

        def to(self, *args, **kwargs):
            self.to_calls.append(args)
            return super().to(*args, **kwargs)

    return SpyDINO()


def test_encode_images_never_moves_the_shared_module():
    """F2: ``encode_images`` must not call ``self.to`` — the dataset and the
    model share one extractor and the dataset asks for CPU features; moving the
    module there (pre-fix) dragged the model's encoder off its device."""
    d = _spy_dino()
    imgs = [np.zeros((224, 224, 3), dtype="uint8")]
    feats = d.encode_images(imgs, device="cpu")
    assert_true(len(d.to_calls) == 0,
                f"encode_images moved the shared module: {d.to_calls} (pre-fix behaviour)")
    assert_true(feats.shape == (1, 196, 384), f"feature shape {tuple(feats.shape)}")
    assert_true(feats.device.type == "cpu", f"features must come back on the requested device, got {feats.device}")


def test_shared_dino_dataset_extraction_does_not_touch_model_device():
    """F2 end-to-end: building a future target through the dataset (which passes
    ``device='cpu'``) must leave the shared module's parameters where they are."""
    from energywam.data.energy_dataset import EnergyWAMEpisodeDataset

    d = _spy_dino()
    before = [p.device for p in d.parameters()]

    class FakeReader:
        dataset_identity = "fake/ep"
        camera_keys = ["cam0"]
        episodes = [{"episode_index": 0, "length": 40, "tasks": ["t"]}]

        def episode_length(self, ep):
            return 40

        def read_actions(self, ep):
            return np.zeros((40, 3), dtype="float32")

        def read_states(self, ep):
            return np.zeros((40, 4), dtype="float32")

        def read_language(self, ep):
            return "t"

        def read_frame(self, ep, cam, idx):
            return np.zeros((224, 224, 3), dtype="uint8")

    from energywam.data.normalization import NormStats
    from energywam.config import EnergyWAMConfig

    cfg = EnergyWAMConfig()
    cfg.data.action_dim = 3
    cfg.data.state_dim = 4
    cfg.data.state_normalization = "none"
    cfg.data.action_horizon = 8
    cfg.data.future_offsets = [3, 5, 8]
    cfg.data.num_cameras = 1
    cfg.data.camera_order = ("cam0",)
    ds = EnergyWAMEpisodeDataset(
        cfg,
        reader=FakeReader(),
        action_stats=NormStats(mode="none"),
        state_stats=None,
        dino=d,
    )
    item = ds[0]  # triggers _future_dino_target -> dino.encode_images(..., device='cpu')
    after = [p.device for p in d.parameters()]
    assert_true(len(d.to_calls) == 0,
                f"dataset future-target extraction moved the shared DINO: {d.to_calls}")
    assert_true(before == after, "shared DINO parameter devices changed during dataset access")
    assert_true(item["future_target"].shape == (3, 1, 196, 384),
                f"future target shape {item['future_target'].shape}")


# ---------------------------------------------------------------------------
# F3 — current_dino_noise reachable from the real forward / predict
# ---------------------------------------------------------------------------
def test_current_dino_noise_works_in_real_forward_and_predict():
    """F3: with ``initialization_future='current_dino_noise'`` the real training
    forward and ``predict`` must work WITHOUT a hand-built ``current_dino``
    key (pre-fix they raised "requires current_dino features")."""
    from .real_fixtures import build_real_model, make_real_batch

    # with_real_dino=True: the stage runner's geometry (384/196) matches the
    # real DINO features the condition computes.
    ew, cfg, geo = build_real_model(with_real_dino=True, seed=41)
    cfg.init.initialization_future = "current_dino_noise"
    cfg.init.init_future_noise_std = 0.05
    batch = make_real_batch(ew, cfg, geo, B=2, seed=9)
    assert_true("current_dino" not in batch, "test setup: batch must not carry current_dino")

    try:
        out = ew(batch)
    except ValueError as exc:
        raise TestFailure(
            f"training forward rejected current_dino_noise without a manual "
            f"current_dino key (pre-fix behaviour): {exc}"
        )
    assert_true(torch.isfinite(out["action_loss"]), "loss not finite")

    slim = {"images": batch["images"], "langs": batch["langs"], "states": batch["states"]}
    try:
        pred = ew.predict(slim, mode="joint", return_diagnostics=True)
    except ValueError as exc:
        raise TestFailure(
            f"predict rejected current_dino_noise without a manual current_dino "
            f"key (pre-fix behaviour): {exc}"
        )
    assert_true(pred["normalized_actions"].shape[0] == 2, "predict batch size")

    # determinism: the local seeded generator must still make predict repeatable
    pred2 = ew.predict(slim, mode="joint")
    assert_close(pred["normalized_actions"], pred2["normalized_actions"], 1e-6,
                 "predict is not deterministic under a fixed seed")


def test_explicit_current_dino_still_wins():
    """F3 guard: an explicitly supplied ``current_dino`` keeps priority over the
    condition-derived features (previous manual-call semantics preserved)."""
    from .real_fixtures import build_real_model, make_real_batch

    ew, cfg, geo = build_real_model(with_real_dino=True, seed=42)
    cfg.init.initialization_future = "current_dino_noise"
    batch = make_real_batch(ew, cfg, geo, B=2, seed=10)
    custom = torch.full((2, geo["num_cameras"], geo["future_patches"], geo["future_dim"]), 7.0)
    batch2 = dict(batch)
    batch2["current_dino"] = custom
    out = ew(batch2)
    assert_true(torch.isfinite(out["action_loss"]), "loss not finite")


# ---------------------------------------------------------------------------
# F4 — action_only data path must not extract the future
# ---------------------------------------------------------------------------
def test_action_only_dataloader_skips_future_extraction():
    """F4: ``build_dataloader`` for ``stage=action_only`` must not decode future
    frames (pre-fix it always did, wasting frozen-DINO passes and reading real
    future data the stage cannot use)."""
    import json
    import tempfile
    from pathlib import Path

    import energywam.data.energy_dataset as ed
    from energywam.config import EnergyWAMConfig

    class CountingReader:
        dataset_identity = "fake/ep"

        def __init__(self):
            self.episodes = [{"episode_index": 0, "length": 40, "tasks": ["t"]}]
            self.frames_read = []

        def episode_length(self, ep):
            return 40

        def read_actions(self, ep):
            return np.zeros((40, 3), dtype="float32")

        def read_states(self, ep):
            return np.zeros((40, 4), dtype="float32")

        def read_language(self, ep):
            return "t"

        def read_frame(self, ep, cam, idx):
            self.frames_read.append(int(idx))
            return np.zeros((224, 224, 3), dtype="uint8")

    class StubDino:
        hidden_dim = 8

        def encode_images(self, images, device="cpu"):
            return torch.zeros(len(images), 4, 8)

    with tempfile.TemporaryDirectory() as td:
        stats_path = Path(td) / "stats.json"
        stats_path.write_text(json.dumps({
            "action": {"min": [-1.0] * 3, "max": [1.0] * 3}
        }))
        cfg = EnergyWAMConfig()
        cfg.backbone.backend = "qwen3vl"
        cfg.stage.stage = "action_only"
        cfg.data.action_dim = 3
        cfg.data.state_dim = 4
        cfg.data.state_normalization = "none"
        cfg.data.action_horizon = 8
        cfg.data.future_offsets = [3, 5, 8]
        cfg.data.num_cameras = 1
        cfg.data.camera_order = ("cam0",)
        cfg.data.norm_stats_path = str(stats_path)
        cfg.trainer.per_device_batch_size = 2

        orig_root, orig_reader = ed.resolve_dataset_root, ed.LeRobotEpisodeReader
        readers = []

        def fake_reader(root):
            r = CountingReader()
            readers.append(r)
            return r

        ed.resolve_dataset_root = lambda c: Path(td)
        ed.LeRobotEpisodeReader = fake_reader
        try:
            loader = ed.build_dataloader(cfg, dino=StubDino())
        finally:
            ed.resolve_dataset_root, ed.LeRobotEpisodeReader = orig_root, orig_reader

        # materialise one batch first so the read counter is populated
        items = next(iter(loader))
        r = readers[0]
        future_offsets_abs = {3, 5, 8}
        decoded_future = [i for i in r.frames_read if i in future_offsets_abs]
        assert_true(loader.dataset.require_future is False,
                    "action_only dataset still requires the future (pre-fix behaviour)")
        assert_true(len(decoded_future) == 0,
                    f"action_only path decoded future frames: {decoded_future} (pre-fix behaviour)")
        assert_true(len(r.frames_read) > 0, "current frame was not read at all")

        # the resulting batch must still be usable by the action_only stage
        assert_true(items["action_target"].shape == (2, 8, 3), "action target shape")
        assert_true(tuple(items["future_target"].shape[:1]) == (2,),
                    "future target batch dim")

        # every other stage keeps the future supervision it trains against
        cfg2 = cfg
        cfg2.stage.stage = "bi"
        assert_true(ed.require_future_for_stage("bi") is True,
                    "bi stage must keep future extraction")
        assert_true(ed.require_future_for_stage("joint") is True,
                    "joint stage must keep future extraction")
        assert_true(ed.require_future_for_stage("action_only") is False,
                    "action_only must not extract the future")


TESTS = [
    ("lm_head_frozen_when_untied_and_unused", test_lm_head_frozen_when_untied_and_unused),
    ("lm_head_tied_never_freezes_shared_embedding", test_lm_head_tied_never_freezes_shared_embedding),
    ("encode_images_never_moves_the_shared_module", test_encode_images_never_moves_the_shared_module),
    ("shared_dino_dataset_extraction_does_not_touch_model_device",
     test_shared_dino_dataset_extraction_does_not_touch_model_device),
    ("current_dino_noise_works_in_real_forward_and_predict",
     test_current_dino_noise_works_in_real_forward_and_predict),
    ("explicit_current_dino_still_wins", test_explicit_current_dino_still_wins),
    ("action_only_dataloader_skips_future_extraction",
     test_action_only_dataloader_skips_future_extraction),
]
