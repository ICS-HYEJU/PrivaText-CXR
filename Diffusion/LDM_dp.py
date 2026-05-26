"""
Diffusion/LDM_dp.py  –  LatentDiffusionDP for DP-SGD Fine-Tuning
=================================================================
Extends LatentDiffusion (LDM.py) with three DP-specific additions:

1. configure_dp_params(ablation_blocks, finetune_biobert)
   -------------------------------------------------------
   Freeze all parameters, then selectively unfreeze SpatialTransformer
   blocks (cross-attention) and optionally BioBERT proj for DP training.
   Returns the list of trainable parameters to pass to AdamW.

2. get_input_dp(batch)  /  training_step_dp(batch)
   -------------------------------------------------
   Variant of get_input/training_step that accepts raw report strings in
   batch['reports'] and calls self.embedder internally.
   VAE encoding is always under torch.no_grad() (VAE is frozen).
   BioBERT embedding is in the computational graph when unfrozen.

3. embedder attribute
   -------------------
   BioBERTEmbedder instance stored as self.embedder so configure_dp_params
   can selectively unfreeze it and training_step_dp can embed text.

All original LatentDiffusion methods are preserved unchanged.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_this_dir, '..', 'Model')
for _d in [_model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ddpm          import DDPM, extract_into_tensor, noise_like
from util_network  import exists, default, mean_flat
from encoder       import DiagonalGaussianDistribution
from attention_module import SpatialTransformer      # for isinstance check

try:
    from DDIMSampler import DDIMSampler
except ImportError:
    DDIMSampler = None


# =============================================================================
# Helpers  (same as LDM.py)
# =============================================================================

def disabled_train(self, mode=True):
    return self


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


# =============================================================================
# LatentDiffusionDP
# =============================================================================

class LatentDiffusionDP(DDPM):
    """
    LatentDiffusion with DP-SGD fine-tuning support.

    Extra args vs LatentDiffusion:
        embedder : BioBERTEmbedder instance (optional; required for
                   training_step_dp / configure_dp_params with finetune_biobert)

    Usage in LDM_dp_finetune.py:
        ldm = LatentDiffusionDP(unet, vae, embedder=biobert, ...)
        ldm.load_pretrained(args.pretrained_ckpt)
        attn_params = ldm.configure_dp_params(ablation_blocks=-1,
                                               finetune_biobert=True)
        # ... Opacus wrap ...
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
        *args, **kwargs
    ):
        kwargs.setdefault('conditioning_key', 'crossattn')
        super().__init__(unet=unet, *args, **kwargs)

        self.first_stage_key = first_stage_key
        self.cond_stage_key  = cond_stage_key
        self.scale_by_std    = scale_by_std
        self.clip_denoised   = False
        self.embedder        = embedder         # BioBERTEmbedder or None

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        # Freeze first-stage model
        self.first_stage_model = first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

        self._restarted_from_ckpt = False

    # =========================================================================
    # DP-specific methods
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
        Step 3 – Optionally unfreeze BioBERT proj (and full BERT if requested).
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

        # 3. BioBERT fine-tuning
        if finetune_biobert and self.embedder is not None:
            # Always unfreeze the projection layer (small, fast to train)
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

    def get_input_dp(self, batch: dict):
        """
        Extract latent z (no_grad, frozen VAE) and context c (with grad if
        BioBERT is unfrozen) from a batch containing raw report strings.

        batch format:
            {
                'image'  : Tensor [B, 1, H, W]  – grayscale CXR
                'reports': list[str]             – raw report text per sample
            }

        Returns:
            z : Tensor [B, z_ch, h, w]       – scaled latent (no grad)
            c : Tensor [B, seq_len, out_dim] – BioBERT context (grad if unfrozen)
        """
        # --- Latent (frozen VAE, no gradient needed) -------------------------
        x = self._get_raw_image(batch).to(self.device)
        with torch.no_grad():
            posterior = self.first_stage_model.encode(x)
            if isinstance(posterior, DiagonalGaussianDistribution):
                z = posterior.sample()
            else:
                z = posterior
            z = self.scale_factor * z          # [B, z_ch, h, w]

        # --- Context (BioBERT, grad flows if proj/bert not frozen) -----------
        assert self.embedder is not None, \
            "embedder is None. Pass a BioBERTEmbedder to LatentDiffusionDP.__init__."
        reports = batch['reports']             # list[str]
        c = self.embedder(reports)             # [B, seq_len, output_dim]

        return z, c

    def training_step_dp(self, batch: dict):
        """
        DP-aware training step. Accepts raw strings in batch['reports'].
        VAE encoding is under no_grad; BioBERT embedding is in the graph
        when its parameters have requires_grad=True.

        Returns:
            loss     : scalar Tensor  (call .backward() externally)
            loss_dict: dict[str, Tensor]
        """
        z, c = self.get_input_dp(batch)
        t    = torch.randint(
            0, self.num_timesteps, (z.shape[0],), device=self.device
        ).long()
        return self.p_losses(z, c, t)

    # =========================================================================
    # Checkpoint helpers
    # =========================================================================

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location='cpu')
        # unwrap common wrapper keys
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
    # Scale-factor initialisation  (call BEFORE Opacus make_private)
    # =========================================================================

    @torch.no_grad()
    def init_scale_factor(self, batch, is_first_batch=False):
        """
        Auto-compute scale_factor = 1/std(z) from first batch.
        MUST be called before privacy_engine.make_private() so the
        register_buffer call does not interact with Opacus grad hooks.
        """
        if not (self.scale_by_std and is_first_batch and not self._restarted_from_ckpt):
            return
        assert self.scale_factor == 1., \
            'Do not combine custom scale_factor with scale_by_std simultaneously.'
        x = self._get_raw_image(batch).to(self.device)
        posterior = self.first_stage_model.encode(x)
        z = posterior.sample() if isinstance(posterior, DiagonalGaussianDistribution) \
            else posterior
        del self.scale_factor
        self.register_buffer('scale_factor', 1. / z.flatten().std())
        print(f'  [init_scale_factor] scale_factor = {self.scale_factor.item():.6f}')

    # =========================================================================
    # All original LatentDiffusion methods below (unchanged)
    # =========================================================================

    @torch.no_grad()
    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, posterior):
        if isinstance(posterior, DiagonalGaussianDistribution):
            z = posterior.sample()
        elif isinstance(posterior, torch.Tensor):
            z = posterior
        else:
            raise TypeError(type(posterior))
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z):
        return self.first_stage_model.decode(z / self.scale_factor)

    def _get_raw_image(self, batch):
        x = batch[self.first_stage_key]
        if x.ndim == 3:
            x = x.unsqueeze(0)
        if x.shape[-1] < x.shape[-3]:
            x = rearrange(x, 'b h w c -> b c h w')
        return x.to(memory_format=torch.contiguous_format).float()

    @torch.no_grad()
    def get_input(self, batch):
        """Original get_input (pre-embedded context tensor in batch)."""
        x = self._get_raw_image(batch).to(self.device)
        posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(posterior)
        c = batch[self.cond_stage_key].to(self.device)
        return z, c

    def forward(self, z, c):
        t = torch.randint(
            0, self.num_timesteps, (z.shape[0],), device=self.device
        ).long()
        return self.p_losses(z, c, t)

    def apply_model(self, z_noisy, t, cond):
        if isinstance(cond, torch.Tensor):
            cond = {'c_crossattn': [cond]}
        elif isinstance(cond, list):
            cond = {'c_crossattn': cond}
        return self.model(z_noisy, t, **cond)

    def p_losses(self, x_start, cond, t, noise=None):
        noise   = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == 'x0':
            target = x_start
        elif self.parameterization == 'eps':
            target = noise
        else:
            raise NotImplementedError(self.parameterization)

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict[f'{prefix}/loss_simple'] = loss_simple.mean()

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict[f'{prefix}/loss_gamma'] = loss.mean()
            loss_dict['logvar'] = self.logvar.data.mean()

        loss = self.l_simple_weight * loss.mean()
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict[f'{prefix}/loss_vlb'] = loss_vlb
        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f'{prefix}/loss'] = loss

        return loss, loss_dict

    def training_step(self, batch):
        """Original training_step (uses pre-embedded context)."""
        z, c = self.get_input(batch)
        return self.forward(z, c)

    def build_optimizer(self, lr=None):
        lr     = lr or self.lr
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)
        return torch.optim.AdamW(params, lr=lr)
