"""
Diffusion/LDM_dp.py  –  LatentDiffusionDP for DP-SGD Fine-Tuning
=================================================================
Extends LatentDiffusion (LDM.py) with DP-specific additions only:

1. __init__
   -------
   Accepts an extra `embedder` argument (BioBERTEmbedder) and delegates
   everything else to LatentDiffusion.__init__.

2. configure_dp_params(ablation_blocks, finetune_biobert)
   -------------------------------------------------------
   Freeze all parameters, then selectively unfreeze SpatialTransformer
   blocks (cross-attention) and optionally the BioBERT proj layer.
   Returns the list of trainable parameters to pass to AdamW.

3. get_input_dp(batch)  /  training_step_dp(batch)
   -------------------------------------------------
   Variant of get_input/training_step that accepts raw report strings in
   batch['reports'] and calls self.embedder internally.
   VAE encoding is always under torch.no_grad() (VAE is frozen).
   BioBERT embedding is in the computational graph when unfrozen.

4. init_from_ckpt (override)
   --------------------------
   Broader checkpoint key unwrapping than the parent version
   ('state_dict', 'model', 'model_state_dict') to handle varied ckpt formats.

All other methods (encode_first_stage, p_losses, training_step, forward, …)
are inherited unchanged from LatentDiffusion.
"""

import os
import sys
import inspect

import torch
import torch.nn as nn

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives at:  <proj_root>/Model/Diffusion/LDM_dp.py
_this_dir  = os.path.dirname(os.path.abspath(__file__))   # …/Model/Diffusion/
_model_dir = os.path.dirname(_this_dir)                    # …/Model/
_proj_root = os.path.dirname(_model_dir)                   # …/PrivaText-CXR/

for _d in [_proj_root, _model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Model.Diffusion.LDM  import LatentDiffusion            # noqa: E402
from Model.attention_module import SpatialTransformer        # noqa: E402  isinstance check
# UNetmodel.py uses Model.attention_module.SpatialTransformer;
# must import the same class or isinstance() always returns False.


# =============================================================================
# LatentDiffusionDP
# =============================================================================

class LatentDiffusionDP(LatentDiffusion):
    """
    LatentDiffusion with DP-SGD fine-tuning support.

    Extra arg vs LatentDiffusion:
        embedder : BioBERTEmbedder instance (required for training_step_dp /
                   configure_dp_params with finetune_biobert=True)

    Usage in LDM_dp_finetune.py:
        ldm = LatentDiffusionDP(unet, vae, embedder=biobert, ...)
        ldm.init_from_ckpt(args.pretrained_ckpt)
        attn_params = ldm.configure_dp_params(ablation_blocks=-1,
                                               finetune_biobert=True)
        # ... Opacus make_private() ...
        loss, _ = ldm.training_step_dp(batch)   # batch has 'reports' strings
    """

    def __init__(
        self,
        unet,
        first_stage_model,
        embedder          = None,
        cond_stage_key    = 'context',
        first_stage_key   = 'image',
        scale_factor      = 1.0,
        scale_by_std      = False,
        device            = None,
        *args, **kwargs
    ):
        # Some LatentDiffusion variants take `device` as an explicit __init__
        # arg, others expose it as a read-only @property (inferred from buffers).
        # Forward `device` only when the parent's signature accepts it, so this
        # class works against both implementations.
        parent_params = inspect.signature(LatentDiffusion.__init__).parameters
        if 'device' in parent_params:
            kwargs.setdefault('device', device)

        super().__init__(
            unet              = unet,
            first_stage_model = first_stage_model,
            cond_stage_key    = cond_stage_key,
            first_stage_key   = first_stage_key,
            scale_factor      = scale_factor,
            scale_by_std      = scale_by_std,
            *args, **kwargs,
        )
        self.embedder = embedder    # BioBERTEmbedder or None

    # =========================================================================
    # Checkpoint (override: broader key unwrapping than parent)
    # =========================================================================

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location='cpu')
        for key in ('state_dict', 'model', 'model_state_dict'):
            if isinstance(sd, dict) and key in sd:
                sd = sd[key]
                print(f'  [ckpt] using sd["{key}"]')
                break
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f'[LDM_dp] loaded  missing={len(missing)}  unexpected={len(unexpected)}')
        self._restarted_from_ckpt = True

    # =========================================================================
    # DP-specific: parameter freeze / unfreeze
    # =========================================================================

    def configure_dp_params(
        self,
        ablation_blocks : int  = -1,
        finetune_biobert: bool = True,
    ) -> list:
        """
        Freeze everything, then selectively unfreeze for DP training.

        Step 1 – Freeze all UNet parameters.
        Step 2 – Unfreeze SpatialTransformer blocks (cross-attention layers).
                 ablation_blocks=-1  : all blocks
                 ablation_blocks=N   : only blocks with index >= N-1
                 (mirrors DP-LDM ablation study logic)
        Step 3 – Optionally unfreeze BioBERT proj (linear layer).
        Step 4 – Return attn_params list to pass to AdamW.

        Args:
            ablation_blocks  : -1 = train all SpatialTransformer blocks;
                                N = train only blocks[N-1:]  (last N blocks)
            finetune_biobert : True = unfreeze self.embedder.proj (linear layer)

        Returns:
            list[nn.Parameter] – parameters to give to the optimizer
        """
        attn_params = []

        # 1. Freeze everything
        self.first_stage_model.requires_grad_(False)
        self.model.requires_grad_(False)
        if self.embedder is not None:
            self.embedder.requires_grad_(False)

        # 2. Selectively unfreeze SpatialTransformer blocks in UNet
        spatial_modules = [
            m for m in self.model.modules()
            if isinstance(m, SpatialTransformer)
        ]
        for i, m in enumerate(spatial_modules):
            m.requires_grad_(True)
            if ablation_blocks == -1 or (i + 1) >= ablation_blocks:
                attn_params.extend(list(m.parameters()))

        # 3. BioBERT fine-tuning (projection layer only)
        if finetune_biobert and self.embedder is not None:
            self.embedder.proj.requires_grad_(True)
            attn_params.extend(list(self.embedder.proj.parameters()))
            print(f'[configure_dp_params] BioBERT proj unfrozen '
                  f'({sum(p.numel() for p in self.embedder.proj.parameters()):,} params)')

        # Summary
        n_spatial   = len(spatial_modules)
        n_trainable = sum(p.numel() for p in attn_params)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f'[configure_dp_params] SpatialTransformer blocks: {n_spatial}  '
              f'(ablation={ablation_blocks})')
        print(f'[configure_dp_params] trainable: {n_trainable:,} / {n_total:,} '
              f'({100 * n_trainable / max(n_total, 1):.1f}%)')

        return attn_params

    # =========================================================================
    # DP-specific: training input / step
    # =========================================================================

    def get_input_dp(self, batch: dict):
        """
        Extract latent z (no_grad, frozen VAE) and context c (with grad if
        BioBERT proj is unfrozen) from a batch containing raw report strings.

        batch format:
            {
                'image'  : Tensor [B, 1, H, W]  – grayscale CXR
                'reports': list[str]             – raw report text per sample
            }

        Returns:
            z : Tensor [B, z_ch, h, w]       – scaled latent (no grad)
            c : Tensor [B, seq_len, out_dim] – BioBERT context
        """
        # Latent: frozen VAE, no gradient
        x = self._get_raw_image(batch).to(self.device)
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)
            z = self.get_first_stage_encoding(posterior)   # scale_factor applied

        # Context: gradient flows through proj when unfrozen
        assert self.embedder is not None, (
            "embedder is None. Pass a BioBERTEmbedder to LatentDiffusionDP.__init__."
        )
        c = self.embedder(batch['reports'])    # list[str] → [B, seq_len, output_dim]

        return z, c

    def training_step_dp(self, batch: dict):
        """
        DP-aware training step. Accepts raw strings in batch['reports'].
        VAE encoding is under no_grad; BioBERT proj is in the graph when unfrozen.

        Returns:
            loss     : scalar Tensor  (call .backward() externally)
            loss_dict: dict[str, Tensor]
        """
        z, c = self.get_input_dp(batch)
        t    = torch.randint(
            0, self.num_timesteps, (z.shape[0],), device=self.device
        ).long()
        return self.p_losses(z, c, t)
