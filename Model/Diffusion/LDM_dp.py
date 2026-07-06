"""
Diffusion/LDM_dp.py  ?  LatentDiffusionDP for DP-SGD Fine-Tuning
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

All other methods (encode_first_stage, p_losses, training_step, forward, ¡¦)
are inherited unchanged from LatentDiffusion.
"""

import os
import sys
import inspect

import torch
import torch.nn as nn

# ¦¡¦¡ Path setup ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_this_dir, '..', 'Model')
for _d in [_model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Model.Diffusion.LDM              import LatentDiffusion                 # noqa: E402
from Model.attention_module_dp import SpatialTransformer              # noqa: E402  isinstance check


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

        Step 1 ? Freeze all UNet parameters.
        Step 2 ? Unfreeze SpatialTransformer blocks (cross-attention layers).
                 ablation_blocks=-1  : all blocks
                 ablation_blocks=N   : only blocks with index >= N-1
                 (mirrors DP-LDM ablation study logic)
        Step 3 ? Optionally unfreeze BioBERT proj (linear layer).
        Step 4 ? Return attn_params list to pass to AdamW.

        Args:
            ablation_blocks  : -1 = train all SpatialTransformer blocks;
                                N = train only blocks[N-1:]  (last N blocks)
            finetune_biobert : True = unfreeze self.embedder.proj (linear layer)

        Returns:
            list[nn.Parameter] ? parameters to give to the optimizer
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
            if type(m).__name__ == 'SpatialTransformer'
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
    # DP-specific: LoRA parameter configuration
    # =========================================================================

    def configure_lora_params(
        self,
        rank            : int   = 4,
        alpha           : float = 4.0,
        dropout         : float = 0.0,
        finetune_biobert: bool  = True,
        targets         = None,
    ) -> list:
        """
        Freeze everything, inject LoRA into cross-attention Linear layers, and
        return ONLY the LoRA (A/B) parameters (+ optionally BioBERT proj) for
        the optimizer.  This is the LoRA counterpart of configure_dp_params.

        Args:
            rank             : LoRA rank r
            alpha            : LoRA scaling numerator (scale = alpha / rank)
            dropout          : LoRA-branch dropout
            finetune_biobert : also train self.embedder.proj (full, small layer)
            targets          : cross-attention attr names to adapt
                               (default to_q/to_k/to_v/to_out)

        Returns:
            list[nn.Parameter] – LoRA params (+ BioBERT proj) for AdamW
        """
        from Model.lora import (inject_lora_cross_attention, lora_parameters,
                                DEFAULT_TARGETS)
        targets = targets or DEFAULT_TARGETS

        # 1. Freeze everything
        self.first_stage_model.requires_grad_(False)
        self.model.requires_grad_(False)
        if self.embedder is not None:
            self.embedder.requires_grad_(False)

        # 2. Inject LoRA into UNet cross-attention; only A/B are trainable
        inject_lora_cross_attention(self.model, rank=rank, alpha=alpha,
                                    dropout=dropout, targets=targets)
        lora_params = lora_parameters(self.model)

        # 3. BioBERT projection (optional, full-trainable small layer)
        if finetune_biobert and self.embedder is not None:
            self.embedder.proj.requires_grad_(True)
            lora_params.extend(list(self.embedder.proj.parameters()))
            print(f'[configure_lora_params] BioBERT proj unfrozen '
                  f'({sum(p.numel() for p in self.embedder.proj.parameters()):,} params)')

        n_trainable = sum(p.numel() for p in lora_params)
        n_total     = sum(p.numel() for p in self.parameters())
        print(f'[configure_lora_params] rank={rank}  alpha={alpha}  '
              f'trainable: {n_trainable:,} / {n_total:,} '
              f'({100 * n_trainable / max(n_total, 1):.3f}%)')
        return lora_params

    # =========================================================================
    # DP-specific: training input / step
    # =========================================================================

    def get_input_dp(self, batch: dict):
        """
        Extract latent z (no_grad, frozen VAE) and context c (with grad if
        BioBERT proj is unfrozen) from a batch containing raw report strings.

        batch format:
            {
                'image'  : Tensor [B, 1, H, W]
                'reports': list[str]
            }

        Model parallelism: if self.offload_device != self.device,
        - images are sent to offload_device for VAE encoding
        - z is moved back to self.device for UNet forward pass
        - c is moved back to self.device after BioBERT embedding

        Returns:
            z, c  - both on self.device (UNet GPU).
        """
        # offload_device: GPU where frozen VAE + BioBERT live.
        # Equals self.device in single-GPU mode; differs in model-parallel mode.
        offload_dev = getattr(self, 'offload_device', self.device)

        # Latent: frozen VAE on offload_device, no gradient.
        # z must end up on self.device (UNet GPU) for the diffusion forward pass.
        x = batch[self.first_stage_key].to(offload_dev)
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)
            z = self.scale_factor * posterior.sample()
        z = z.detach().to(self.device)

        # Context: BioBERT on offload_device; c is moved to self.device.
        # Gradient flows through proj layer when finetune_biobert=True.
        assert self.embedder is not None, (
            "embedder is None. Pass a BioBERTEmbedder to LatentDiffusionDP.__init__."
        )
        c = self.embedder(batch['reports'])    # list[str] -> [B, seq_len, output_dim]
        c = c.to(self.device)

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