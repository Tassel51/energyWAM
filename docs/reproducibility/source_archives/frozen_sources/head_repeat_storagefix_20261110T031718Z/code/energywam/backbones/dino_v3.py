"""Frozen DINOv3-ViT-S feature extractor.

Checkpoint used on this server
------------------------------
``/data6/zzc/dinov3-vits16-pretrain-lvd1689m`` (HuggingFace format)

* hidden size 384
* patch size 16
* 4 register tokens
* image size 224  ->  14x14 = 196 patches

The feature we supervise on is the **spatial patch tokens only**: the CLS token
and the 4 register tokens are removed.  Nothing here silently swaps in DINOv2,
a different resolution or a randomly initialised model — a mismatch raises.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn


class FrozenDINOv3(nn.Module):
    """Thin wrapper over a HF ``DINOv3ViTModel``-style checkpoint.

    The module is frozen by default (``requires_grad_(False)``) and run under
    ``torch.no_grad()`` in :meth:`extract_patches`, which returns **detached**
    features — the DINO target is a constant for the energy optimisation.
    """

    def __init__(
        self,
        model_path: str,
        *,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_dim: int = 384,
        num_special_tokens: int = 5,
        num_patches: int = 196,
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_path = model_path
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.hidden_dim = int(hidden_dim)
        self.num_special_tokens = int(num_special_tokens)
        self.num_patches = int(num_patches)
        self._dtype = dtype

        self.body = self._load(model_path, dtype)
        self._assert_config()
        if freeze:
            self.freeze()

    # ------------------------------------------------------------------
    def _load(self, model_path: str, dtype: torch.dtype):
        import os

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"DINO checkpoint not found: {model_path!r}. "
                "Real mode requires the frozen DINOv3 weights; refusing to "
                "silently substitute another model."
            )
        try:
            from transformers import AutoModel
        except ImportError as exc:  # pragma: no cover - transformers is a hard dep of the env
            raise ImportError("transformers is required for the real DINO path") from exc

        body = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        return body

    def _assert_config(self) -> None:
        cfg = getattr(self.body, "config", None)
        if cfg is None:
            return
        # Validate the architecture NAME, not just the tensor sizes — a
        # different ViT with hidden=384 must not be silently accepted.
        model_type = str(getattr(cfg, "model_type", "") or "")
        if model_type and "dinov3" not in model_type.lower():
            raise ValueError(
                f"checkpoint model_type={model_type!r} is not a DINOv3 model. "
                "Refusing to silently substitute another backbone."
            )
        hidden = getattr(cfg, "hidden_size", None)
        if hidden is not None and int(hidden) != self.hidden_dim:
            raise ValueError(
                f"DINO hidden_size mismatch: checkpoint={hidden}, expected={self.hidden_dim}. "
                "Refusing to continue with a wrong backbone."
            )
        ps = getattr(cfg, "patch_size", None)
        if ps is not None and int(ps) != self.patch_size:
            raise ValueError(f"DINO patch_size mismatch: checkpoint={ps}, expected={self.patch_size}")
        n_reg = getattr(cfg, "num_register_tokens", None)
        if n_reg is not None and int(n_reg) + 1 != self.num_special_tokens:
            raise ValueError(
                f"DINO special-token count mismatch: checkpoint registers={n_reg} "
                f"(+1 CLS) != expected {self.num_special_tokens}"
            )
        expected_patches = (self.image_size // self.patch_size) ** 2
        if expected_patches != self.num_patches:
            raise ValueError(
                f"expected {expected_patches} patches for {self.image_size}/{self.patch_size}, "
                f"config says {self.num_patches}"
            )

    # ------------------------------------------------------------------
    def freeze(self) -> None:
        self.body.eval()
        for p in self.body.parameters():
            p.requires_grad_(False)

    def train(self, mode: bool = True):  # noqa: D401 - nn.Module API
        """Keep the frozen body in eval mode regardless of the outer mode."""
        super().train(mode)
        self.body.eval()
        return self

    # ------------------------------------------------------------------
    def preprocess(self, images: Sequence) -> torch.Tensor:
        """PIL / ndarray / tensor images -> normalised float tensor [N, 3, S, S].

        Uses the DINOv3 reference normalisation.  Deterministic: resize to
        ``image_size`` (bicubic) and normalise.  No random crop / geometric
        augmentation is applied here, so a transition's current and future
        frames are preprocessed consistently (the requirements ask for temporal
        consistency when augmentation is used; we simply do not use any).
        """
        from PIL import Image

        import numpy as np

        out = []
        for img in images:
            if isinstance(img, torch.Tensor):
                t = img
                # uint8 tensors are 0..255 and MUST be scaled; leaving them raw
                # silently destroys the features.
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                else:
                    t = t.float()
                if t.dim() == 3 and t.shape[0] != 3:
                    t = t.permute(2, 0, 1)
                if t.shape[0] != 3:
                    raise ValueError(f"expected 3-channel image, got shape {tuple(t.shape)}")
                if t.shape[-2] != self.image_size or t.shape[-1] != self.image_size:
                    t = torch.nn.functional.interpolate(
                        t.unsqueeze(0), size=(self.image_size, self.image_size),
                        mode="bicubic", align_corners=False,
                    ).squeeze(0)
            else:
                if not isinstance(img, Image.Image):
                    arr_in = np.asarray(img)
                    if arr_in.ndim != 3 or arr_in.shape[2] != 3:
                        raise ValueError(f"expected HxWx3 image array, got shape {arr_in.shape}")
                    img = Image.fromarray(arr_in.astype("uint8"))
                img = img.convert("RGB")
                if img.size != (self.image_size, self.image_size):
                    img = img.resize((self.image_size, self.image_size), Image.BICUBIC)
                arr = np.asarray(img, dtype="float32") / 255.0
                t = torch.from_numpy(arr).permute(2, 0, 1)
            out.append(t)
        x = torch.stack(out, dim=0)
        # final H/W check — the assertion in extract_patches is not enough
        # because a bad resize would otherwise fail much later and less clearly
        if x.shape[-2] != self.image_size or x.shape[-1] != self.image_size:
            raise ValueError(f"preprocess produced {tuple(x.shape[-2:])}, expected {self.image_size}")
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (x - mean) / std

    # ------------------------------------------------------------------
    def extract_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[N, 3, S, S] -> [N, P, D]`` with CLS + register tokens removed.

        The result is **detached** (DINO is frozen; its output is a constant
        supervision target, not a differentiable branch).
        """
        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must be [N,3,S,S], got {tuple(pixel_values.shape)}")
        if pixel_values.shape[-1] != self.image_size or pixel_values.shape[-2] != self.image_size:
            raise ValueError(
                f"expected {self.image_size}x{self.image_size} input, got {tuple(pixel_values.shape[-2:])}"
            )
        try:
            body_device = next(self.body.parameters()).device
        except StopIteration:
            body_device = pixel_values.device
        with torch.no_grad():
            x = pixel_values.to(device=body_device, dtype=self._dtype)
            out = self.body(pixel_values=x)
            tokens = self._select_patch_tokens(out)
        # return on the INPUT device so callers can mix CPU data with a
        # device-placed encoder without surprise cross-device graphs
        return tokens.detach().to(device=pixel_values.device)

    def _select_patch_tokens(self, out) -> torch.Tensor:
        """Pull the patch-token matrix out of a HF DINO output and drop specials."""
        tokens = None
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            tokens = out.last_hidden_state
        elif isinstance(out, dict) or hasattr(out, "keys"):
            for key in ("last_hidden_state", "x_norm_patchtokens"):
                if key in out:
                    tokens = out[key]
                    break
        if tokens is None:
            raise RuntimeError(f"cannot read patch tokens from DINO output of type {type(out)!r}")

        # Some HF DINOv3 builds expose ``x_norm_patchtokens`` directly (already
        # stripped).  Detect that by sequence length.
        if tokens.shape[1] == self.num_patches:
            return tokens
        expected_total = self.num_patches + self.num_special_tokens
        if tokens.shape[1] != expected_total:
            raise ValueError(
                f"DINO token count {tokens.shape[1]} != {expected_total} "
                f"(= {self.num_patches} patches + {self.num_special_tokens} special). "
                "Refusing to guess which tokens are patches."
            )
        return tokens[:, self.num_special_tokens:, :]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, images: Sequence, device: torch.device | str = "cpu") -> torch.Tensor:
        """Convenience: PIL list -> ``[N, P, D]`` detached patch features.

        The **module is never moved** here.  One ``FrozenDINOv3`` instance is
        shared by the dataset (future-target extraction) and the model
        (current-frame conditioning), and the dataset asks for CPU features —
        a ``self.to(device)`` in this method dragged the model's frozen encoder
        back to CPU on every batch and forced the next forward to move it again.
        ``extract_patches`` already runs the body on its own device and returns
        the features on the caller's device, so moving only the input keeps the
        output contract (``device``) while leaving the encoder where the owner
        placed it (``model.to(...)``).
        """
        x = self.preprocess(images).to(device)
        return self.extract_patches(x)
