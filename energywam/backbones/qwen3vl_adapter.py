"""Real Qwen3-VL energy backbone adapter.

Scope
-----
This is a *small* adapter around ``transformers``' Qwen3-VL implementation
(4.57.0).  It does **not** rewrite Transformers; it reuses the model's own
methods and drives the language model directly so that candidate tokens can be
appended **before** the language transformer runs (the engineering requirements
forbid bolting A/Z onto the hidden states after all language layers).

The condition path is embedding-only: Qwen ViT + its native visual merger,
and text token lookup.  It never contextualises C with the language decoder.
``run_sequence`` is the sole language-transformer entry point, called once per
energy evaluation on the complete sequence.  Iterative energy minimisation
requires a new evaluation for each updated candidate; "once" does not mean
once for an entire solver rollout.

What is preserved from the native path
--------------------------------------
* image token replacement / feature injection (``masked_scatter``)
* DeepStack multi-layer visual fusion (``deepstack_visual_embeds`` +
  ``visual_pos_masks``)
* ``image_grid_thw`` and mRoPE ``position_ids`` from ``get_rope_index``
* padding / attention mask, variable language length, multi-view images

What is appended
----------------
The block ``[T_future, T_action, e_E]`` (plus the current-DINO / state tokens
that make up the rest of C) is concatenated after the native sequence.  Its
mRoPE positions start at ``(per-sample native valid max) + 1`` and the three
axes share the same consecutive numbering.  Independent future-time / spatial /
camera / type encodings carry the actual semantics (see
:mod:`energywam.energy.encodings`).

Attention / higher-order backward
---------------------------------
``attn_implementation`` is configurable; the reference is ``eager`` because it
is plain matmul+softmax and therefore supports the double backward the full
unroll needs.  KV cache, gradient checkpointing and torch.compile are disabled
for this framework.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from ..config import EnergyWAMConfig
from ..energy.types import ConditionBundle, TokenGroup


class Qwen3VLEnergyBackbone(nn.Module):
    """Adapter that exposes the EnergyWAM backbone interface."""

    def __init__(
        self,
        cfg: EnergyWAMConfig,
        model: Optional[nn.Module] = None,
        processor=None,
        dino: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self._external_model = model is not None
        self.model, self.processor = self._load(cfg, model, processor)
        self._enforce_attn_implementation(cfg.backbone.attn_implementation)
        self.hidden_size = int(self._text_config.hidden_size)
        self.dtype = getattr(torch, cfg.backbone.dtype, torch.float32)

        # ``self.dino`` is assigned EXACTLY ONCE in __init__ so the parameter
        # key set is stable for the lifetime of the module (a lazy property that
        # created submodules on first use would make state_dict keys differ
        # before/after the first forward and break strict checkpoint loading).
        # Tests inject a small stub; real runs load the frozen DINOv3 checkpoint.
        self.dino: nn.Module = dino if dino is not None else self._load_dino(cfg)

        if cfg.backbone.condition_mode == "text_dino_state":
            # Remove duplicate visual tower, keep language trunk and text embedding.
            self._mm.visual = None
        if cfg.backbone.freeze_dino:
            self.dino.requires_grad_(False)
            self.dino.eval()
        self._apply_freezing(cfg)
        # Never allow the fused/higher-order-incompatible paths for this framework.
        self._disable_incompatible_optimisations(cfg)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def _load(self, cfg: EnergyWAMConfig, model, processor):
        if model is not None:
            return model, processor
        import os

        path = cfg.backbone.base_vlm
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Qwen3-VL weights not found at {path!r}.  Real mode requires the "
                "pretrained weights; use backbone.backend='tiny' for the explicit "
                "mock test mode."
            )
        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        attn = cfg.backbone.attn_implementation
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                path,
                attn_implementation=attn,
                dtype=getattr(torch, cfg.backbone.dtype, torch.float32),
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load Qwen3-VL from {path!r} with attn_implementation={attn!r}: {exc}"
            ) from exc
        processor = (AutoTokenizer.from_pretrained(path, padding_side="left")
                     if cfg.backbone.condition_mode == "text_dino_state"
                     else AutoProcessor.from_pretrained(path))
        return model, processor

    def _load_dino(self, cfg: EnergyWAMConfig):
        """Eagerly load the frozen DINOv3 checkpoint as a registered submodule.

        Real mode refuses to run without it (a batch may still supply
        pre-computed ``current_dino`` features, but the module is always present
        so checkpoint keys never change).
        """
        from .dino_v3 import FrozenDINOv3

        return FrozenDINOv3(
            cfg.backbone.dino_path,
            image_size=cfg.backbone.dino_image_size,
            patch_size=cfg.backbone.dino_patch_size,
            hidden_dim=cfg.backbone.dino_hidden_dim,
            num_special_tokens=cfg.backbone.dino_num_special_tokens,
            num_patches=cfg.backbone.dino_num_patches,
            freeze=cfg.backbone.freeze_dino,
            dtype=torch.float32,
        )

    def _enforce_attn_implementation(self, name: str) -> None:
        """Force the attention backend on every submodule, including an
        externally supplied model.

        This matters for the full unroll: the reference is ``eager`` because the
        SDPA / flash kernels do not implement the double backward that
        ``create_graph=True`` needs.  We measured this on the current stack —
        see logs/test_results_test_qwen_structure.json, where the first run with
        the default ``sdpa`` failed with
        ``derivative for aten::_scaled_dot_product_flash_attention_for_cpu_backward
        is not implemented`` and succeeded once ``eager`` was enforced.
        """
        targets = [self.model, self._mm, self._mm.language_model, self._mm.visual]
        for sub in targets:
            if sub is None:
                continue
            cfg_obj = getattr(sub, "config", None)
            if cfg_obj is not None:
                try:
                    cfg_obj._attn_implementation = name
                except Exception:
                    pass
        main_cfg = getattr(self.model, "config", None)
        if main_cfg is not None:
            try:
                main_cfg._attn_implementation = name
            except Exception:
                pass

    def _apply_freezing(self, cfg: EnergyWAMConfig) -> None:
        b = cfg.backbone
        # vision encoder
        if self._mm.visual is not None:
            for p in self._mm.visual.parameters():
                p.requires_grad_(bool(b.tune_qwen_vision))
            if b.freeze_qwen_vision and not b.tune_qwen_vision:
                self._mm.visual.eval()
        # language model
        for p in self._mm.language_model.parameters():
            p.requires_grad_(bool(b.tune_language_model))
        if not b.tune_language_model:
            self._mm.language_model.eval()
        # token embeddings are part of the language model; keep them tied to the
        # same policy (the requirements warn not to delete weights needed for
        # text input when lm_head is tied — we simply never touch lm_head here).
        self._freeze_unused_lm_head()
        if b.lora_enabled:
            self._apply_lora(b)

    def _freeze_unused_lm_head(self) -> None:
        """Freeze ``lm_head`` **only** when it is a genuinely unused parameter.

        The energy path drives ``language_model`` directly and never computes
        vocab logits, so a standalone ``lm_head`` is dead weight: leaving it
        trainable inflates the trainable-parameter count and the grad-norm
        bookkeeping for a tensor that can never receive a gradient.

        However, when the checkpoint ties the output head to the token
        embeddings (Qwen3-VL-4B-Instruct: ``tie_word_embeddings: true``),
        ``lm_head.weight`` **is** ``embed_tokens.weight`` — the very matrix the
        language model needs to embed the text input.  Freezing it would silently
        freeze text embeddings, so a tied head is left exactly on the
        language-model policy applied above and is never touched here.
        """
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            return
        emb_weight = None
        try:
            emb_weight = self.model.get_input_embeddings().weight
        except Exception:
            emb = getattr(self._mm.language_model, "embed_tokens", None)
            emb_weight = getattr(emb, "weight", None)
        tied = emb_weight is not None and any(p is emb_weight for p in lm_head.parameters())
        if not tied:
            for p in lm_head.parameters():
                p.requires_grad_(False)

    def _apply_lora(self, b) -> None:
        """Optional, explicitly-labelled LoRA (NOT the main method)."""
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise ImportError("LoRA requires the optional `peft` package") from exc
        lora_cfg = LoraConfig(
            r=int(b.lora_r),
            lora_alpha=int(b.lora_alpha),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        self._mm.language_model = get_peft_model(self._mm.language_model, lora_cfg)

    def _disable_incompatible_optimisations(self, cfg=None) -> None:
        m = self.model
        # Gradient checkpointing + double backward (create_graph=True) is not
        # assumed to work by default, so it stays OFF unless the config turns it
        # on explicitly.  The opt-in path uses the non-reentrant implementation,
        # which supports higher-order gradients; scripts/probe_checkpoint_grads.py
        # validates gradient agreement against the un-checkpointed trunk before a
        # real run trusts it.
        use_ckpt = bool(cfg is not None and getattr(cfg.backbone, "gradient_checkpointing", False))
        if use_ckpt:
            lm = getattr(self._mm, "language_model", None)
            if lm is not None and hasattr(lm, "gradient_checkpointing_enable"):
                lm.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                print("[EnergyWAM] gradient checkpointing ON (use_reentrant=False) on the language trunk")
        for sub in (m, getattr(m, "language_model", None), getattr(m, "visual", None)):
            if sub is None:
                continue
            if use_ckpt and getattr(sub, "gradient_checkpointing", False):
                continue
            if hasattr(sub, "gradient_checkpointing"):
                try:
                    sub.gradient_checkpointing = False
                except Exception:
                    pass
            if hasattr(sub, "gradient_checkpointing_enable"):
                pass
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = False
        # cache should never be used in the energy path
        self._use_cache = False

    # ------------------------------------------------------------------
    @property
    def _text_config(self):
        cfg = self.model.config
        return getattr(cfg, "text_config", cfg)

    @property
    def _mm(self):
        """The ``Qwen3VLModel`` that owns ``visual`` / ``language_model`` and the
        multimodal helpers (``get_image_features`` / ``get_placeholder_mask`` /
        ``get_rope_index``).  ``Qwen3VLForConditionalGeneration`` wraps it as
        ``.model``; a bare ``Qwen3VLModel`` is used as-is."""
        inner = getattr(self.model, "model", None)
        return inner if inner is not None else self.model

    # ------------------------------------------------------------------
    # condition encoding
    # ------------------------------------------------------------------
    def encode_condition(self, batch: Dict[str, Any]) -> ConditionBundle:
        """Build embedding-only ``C`` from raw StarVLA-style fields.

        Qwen image features and text embeddings are prepared here, with no
        language-transformer forward.  Candidate tokens must be appended by
        ``run_sequence`` before the language layers see this condition.

        Expected keys: ``images`` (List[List[PIL]]), ``langs`` (List[str]),
        ``states`` (List[np.ndarray] or None), optional ``current_dino``.
        """
        images = batch.get("images")
        langs = batch.get("langs")
        states = batch.get("states")
        if images is None or langs is None:
            raise ValueError("real Qwen path requires batch['images'] and batch['langs']")
        device = next(self.model.parameters()).device

        if self.cfg.backbone.condition_mode == "text_dino_state":
            tokenizer = getattr(self.processor, "tokenizer", self.processor)
            # Text only: no RGB placeholders, visual merger, or DeepStack inputs.
            native = tokenizer(list(langs), padding=True, return_tensors="pt")
            input_ids = native["input_ids"].to(device)
            attention_mask = native["attention_mask"].to(device)
            inputs_embeds = self._mm.get_input_embeddings()(input_ids)
            positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
            position_ids = positions.unsqueeze(0).expand(3, -1, -1)
            deepstack_visual_embeds = visual_pos_masks = None
        else:
            # --- 1. native multimodal inputs ---
            native = self._build_native_inputs(images, langs, device)
            input_ids = native["input_ids"]
            attention_mask = native["attention_mask"]
    
            # --- 2. embedding-only Qwen path; NO language transformer here ---
            inputs_embeds, deepstack_visual_embeds, visual_pos_masks = self._embed_vision_language(native)
    
            # --- 3. mRoPE position ids for the native sequence ---
            position_ids, _ = self._mm.get_rope_index(
                input_ids,
                native.get("image_grid_thw"),
                native.get("video_grid_thw"),
                attention_mask=attention_mask,
            )

        # --- 4. raw tensors for the extra condition tokens (current DINO + state)
        # These are converted to TokenGroups by EnergyWAMCore (which owns the
        # projectors and the semantic encodings).  The backbone deliberately
        # holds NO reference back to the core, so there is no module cycle.
        extra_raw: Dict[str, torch.Tensor] = {}
        cur_dino = batch.get("current_dino")
        if cur_dino is None:
            cur_dino = self.dino_from_images(images, device)
        else:
            cur_dino = torch.as_tensor(cur_dino).to(device)
        extra_raw["dino"] = cur_dino
        if states is not None and states[0] is not None:
            import numpy as np

            extra_raw["state"] = torch.as_tensor(
                np.stack([np.asarray(s, dtype="float32") for s in states]), device=device
            )

        B = input_ids.shape[0]
        native_lengths = attention_mask.sum(dim=1).long()
        return ConditionBundle(
            group=None,
            native={
                "inputs_embeds": inputs_embeds,
                "extra_raw": extra_raw,
                "batch_size": torch.tensor(B, device=device),
            },
            native_lengths=native_lengths,
            native_position_ids=position_ids,
            native_attention_mask=attention_mask,
            deepstack_visual_embeds=deepstack_visual_embeds,
            visual_pos_masks=visual_pos_masks,
        )

    def _embed_vision_language(
        self, native: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
        """Qwen ViT/merger + text embedding lookup, without language layers.

        Preserve native image-placeholder replacement and DeepStack features.
        The latter are injected during the single subsequent language pass;
        they do not require an earlier language-model forward.
        """
        input_ids = native["input_ids"]
        inputs_embeds = self._mm.get_input_embeddings()(input_ids)
        deepstack_visual_embeds = None
        visual_pos_masks = None
        if native.get("pixel_values") is not None:
            image_embeds, deepstack_visual_embeds = self._mm.get_image_features(
                native["pixel_values"], native["image_grid_thw"]
            )
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self._mm.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
        return inputs_embeds, deepstack_visual_embeds, visual_pos_masks

    # ------------------------------------------------------------------
    def _build_native_inputs(self, images, langs, device) -> Dict[str, torch.Tensor]:
        messages = []
        for imgs, instruction in zip(images, langs):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])
        batch_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        out = {}
        for k, v in batch_inputs.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out

    def dino_from_images(self, images, device) -> torch.Tensor:
        """Return ``[B, V, P, D]`` current-frame DINO patch features."""
        B = len(images)
        V = len(images[0])
        flat = [img for sample in images for img in sample]
        feats = self.dino.encode_images(flat, device=device)  # [B*V, P, D]
        return feats.view(B, V, feats.shape[1], feats.shape[2])

    # ------------------------------------------------------------------
    # sequence assembly
    # ------------------------------------------------------------------
    def run_sequence(
        self, cond: ConditionBundle, appended: Sequence[TokenGroup]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concat ``[native C_vl, extra C, T_future, T_action, e_E]`` and run the
        language model exactly once for this energy evaluation.

        C_vl contains pre-decoder embeddings, not language hidden states.  The
        ViT and text lookup are reused from ``encode_condition``; updated A/Z
        candidates require another call here, not another condition encoding.
        Returns ``(hidden [B, S, D], energy_index [B])``.
        """
        if cond.native is None or "inputs_embeds" not in cond.native:
            raise ValueError("Qwen backbone requires a ConditionBundle from encode_condition")
        inputs_embeds = cond.native["inputs_embeds"]
        attention_mask = cond.native_attention_mask
        position_ids = cond.native_position_ids
        device = inputs_embeds.device
        B, S_native, D = inputs_embeds.shape

        # --- extra condition tokens (current DINO / state), built by the core ---
        extra_group = cond.group
        extra_embeds = extra_group.embeds if extra_group is not None else None
        extra_mask = extra_group.mask if extra_group is not None else None

        # --- candidate groups (built by EnergyWAMCore) ---
        app_embeds = torch.cat([g.embeds for g in appended], dim=1).to(dtype=inputs_embeds.dtype)
        app_mask = torch.cat([g.mask for g in appended], dim=1)

        parts = [inputs_embeds]
        if extra_embeds is not None:
            parts.append(extra_embeds.to(dtype=inputs_embeds.dtype))
        parts.append(app_embeds)
        full_embeds = torch.cat(parts, dim=1)

        parts_m = [attention_mask.to(dtype=torch.long)]
        if extra_mask is not None:
            parts_m.append(extra_mask.to(dtype=torch.long))
        parts_m.append(app_mask.to(dtype=torch.long))
        full_mask = torch.cat(parts_m, dim=1)

        # --- position ids: appended block sits strictly after the native max ---
        appended_len = full_embeds.shape[1] - S_native
        app_pos = self._appended_position_ids(
            position_ids, attention_mask, appended_len, device
        )  # [3, B, appended_len]
        full_position_ids = torch.cat([position_ids.to(device), app_pos], dim=2)

        # --- DeepStack: only the native visual positions receive it ---
        visual_pos_masks = cond.visual_pos_masks
        if visual_pos_masks is not None:
            pad = torch.zeros(
                B, full_mask.shape[1] - visual_pos_masks.shape[1], dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, pad], dim=1)

        hidden = self._mm.language_model(
            input_ids=None,
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            position_ids=full_position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=None,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=cond.deepstack_visual_embeds,
        )
        last = hidden.last_hidden_state if hasattr(hidden, "last_hidden_state") else hidden[0]

        # Energy token: LAST token of the sequence (the final appended group's
        # last column).  It is always valid, and the causal layout guarantees it
        # can see C, T_future and T_action.
        e_index = torch.full((B,), last.shape[1] - 1, device=last.device, dtype=torch.long)
        return last, e_index

    # ------------------------------------------------------------------
    def _appended_position_ids(
        self,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        appended_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Positions for the appended block.

        Per sample, start at ``(max native valid mRoPE position) + 1`` and number
        consecutively with the **same** values on all three mRoPE axes.
        """
        B = position_ids.shape[1]
        pos = position_ids.to(device)  # [3, B, S]
        mask = attention_mask.to(device).to(dtype=torch.bool)  # [B, S]
        # max over the 3 axes and over valid (non-padding) positions
        masked = pos.masked_fill(~mask.unsqueeze(0), torch.iinfo(torch.int64).min)
        base = masked.amax(dim=(0, 2))  # [B]
        # if a sample has no valid native token, fall back to 0
        base = torch.where(mask.any(dim=1), base, torch.zeros_like(base))
        start = base + 1
        offs = torch.arange(appended_len, device=device, dtype=position_ids.dtype)
        app = start.view(B, 1) + offs.view(1, appended_len)  # [B, L]
        return app.unsqueeze(0).expand(3, -1, -1).contiguous()

    # ------------------------------------------------------------------
