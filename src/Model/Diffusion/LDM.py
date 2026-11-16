"""
Diffusion/LDM.py  ?  LatentDiffusion (pure PyTorch, no pytorch_lightning)
==========================================================================

2-Stage Pipeline
----------------
Stage 1  x ¦¡¦¡? AutoencoderKL.encode(x) ¦¡¦¡? DiagonalGaussianDistribution
               ¦¡¦¡? z = posterior.sample() * scale_factor          (frozen)

Stage 2  [z, c] ¦¡¦¡? DDPM diffusion in latent space
               c  : context tensor from pre-trained BioBERT
                    shape [B, seq_len, context_dim]
               UNetModel receives (z_noisy, t, context=c)
               via DiffusionWrapper(conditioning_key='crossattn')

Training loop (caller):
    model.init_scale_factor(first_batch, is_first_batch=True)  # once
    optimizer = model.build_optimizer()
    for batch in loader:
        # batch = {'image': Tensor[B,C,H,W], 'context': Tensor[B,seq,dim]}
        loss, loss_dict = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_ema()
"""

import os
import sys
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import tqdm
from Data.cls_emb import *

# ¦¡¦¡ Path setup ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_this_dir, '..', 'Model')
for _d in [_model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ddpm import DDPM, extract_into_tensor, noise_like   # noqa: E402
from Model.util_network import exists, default, mean_flat      # noqa: E402
from Model.VAE.Encoder import DiagonalGaussianDistribution         # noqa: E402

try:
    from DDIMSampler import DDIMSampler
except ImportError:
    DDIMSampler = None


# =============================================================================
# Helpers
# =============================================================================

def disabled_train(self, mode=True):
    """Prevent a frozen model from switching train/eval mode."""
    return self


def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 + logvar2 - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


# =============================================================================
# LatentDiffusion
# =============================================================================

class LatentDiffusion(DDPM):
    """
    Latent Diffusion Model operating in the latent space of a frozen AutoencoderKL.

    Args:
        unet              : UNetModel instance (from Diffusion/UNetModel.py).
        first_stage_model : AutoencoderKL instance (frozen after init).
        cond_stage_key    : Batch dict key for the conditioning tensor c.
                            Defaults to 'context' (pre-computed BioBERT embeddings).
        first_stage_key   : Batch dict key for the input image. Defaults to 'image'.
        scale_factor      : Latent scale applied after encoding. Defaults to 1.0.
        scale_by_std      : If True, auto-compute scale_factor from first batch std.
        All remaining kwargs are forwarded to DDPM (timesteps, image_size, ¡¦).
    """

    def __init__(
        self,
        device,
        unet,
        first_stage_model,
        cond_stage_key    = 'context',
        first_stage_key   = 'image',
        scale_factor      = 1.0,
        scale_by_std      = False,
        *args, **kwargs
    ):
        # conditioning_key must be 'crossattn' for BioBERT context
        kwargs.setdefault('conditioning_key', 'crossattn')

        super().__init__(unet=unet, *args, **kwargs)

        self.first_stage_key = first_stage_key
        self.cond_stage_key  = cond_stage_key
        self.scale_by_std    = scale_by_std
        self.clip_denoised   = False
        self.device = device
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

    # Checkpoint
    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                print(f'  [ckpt] removing key: {k}')
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f'[LDM] Loaded  missing={len(missing)}  unexpected={len(unexpected)}')
        self._restarted_from_ckpt = True

    # ¦¡¦¡ Scale-factor initialisation (call once before first training step) ¦¡¦¡¦¡¦¡

    @torch.no_grad()
    def init_scale_factor(self, batch, is_first_batch=False):
        """
        Auto-compute scale_factor = 1/std(z) so the latent distribution has
        unit variance. Only runs when scale_by_std=True and is_first_batch=True.

        Call once:
            model.init_scale_factor(first_batch, is_first_batch=True)
        """
        if not (self.scale_by_std and is_first_batch and not self._restarted_from_ckpt):
            return
        assert self.scale_factor == 1., \
            'Do not combine custom scale_factor with scale_by_std simultaneously.'
        print('### USING STD-RESCALING ###')
        x = self._get_raw_image(batch).to(self.device)
        posterior = self.model.diffusion_model.encode(x)
        del self.scale_factor
        self.register_buffer('scale_factor', 1. / z.flatten().std())
        print(f'  scale_factor set to {self.scale_factor.item():.6f}')

    # ¦¡¦¡ Stage 1: Encoding / Decoding ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    @torch.no_grad()
    def encode_first_stage(self, x):
        """x [B,C,H,W] ¡æ DiagonalGaussianDistribution."""
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, posterior) -> torch.Tensor:
        """
        Sample z from the posterior and apply scale_factor.

        posterior: DiagonalGaussianDistribution  OR  plain Tensor
        """
        if isinstance(posterior, DiagonalGaussianDistribution):
            z = posterior.sample()
        elif isinstance(posterior, torch.Tensor):
            z = posterior
        else:
            raise TypeError(f'Unsupported posterior type: {type(posterior)}')
        return self.scale_factor * z

    @torch.no_grad()
    def decode_first_stage(self, z) -> torch.Tensor:
        """z [B,C,h,w] ¡æ x_recon [B,C,H,W]."""
        return self.first_stage_model.decode(z / self.scale_factor)

    def differentiable_decode_first_stage(self, z) -> torch.Tensor:
        """Gradient-enabled decode (for perceptual losses etc.)."""
        return self.first_stage_model.decode(z / self.scale_factor)

    # ¦¡¦¡ Input pipeline ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def _get_raw_image(self, batch) -> torch.Tensor:
        """
        Extract and format the image tensor from the batch.

        Expects batch[first_stage_key] shaped [B, H, W, C] or [B, C, H, W].
        Returns [B, C, H, W] float32 contiguous.
        """
        x = batch[self.first_stage_key]
        if x.ndim == 3:                              # [H, W, C]
            x = x.unsqueeze(0)
        if x.shape[-1] < x.shape[-3]:               # [B, H, W, C] ¡æ [B, C, H, W]
            x = rearrange(x, 'b h w c -> b c h w')
        return x.to(memory_format=torch.contiguous_format).float()

    @torch.no_grad()
    def get_input(self, batch):
        """
        Extract latent z and conditioning context c from a batch.

        Returns:
            z : Tensor [B, z_channels, h, w]  -- scaled latent
            c : Tensor [B, seq_len, context_dim]  -- BioBERT embedding
        """
        # Stage 1: sampling z from trained VAE
        x = self._get_raw_image(batch).to(self.device)
        posterior = self.encode_first_stage(x)
        z = posterior.sample().detach().to(self.device) # [B, z_ch, h, w]
        # Stage 2 conditioning: pre-computed BioBERT embedding
        c = batch[self.cond_stage_key].to(self.device) # [B, seq, dim]

        return z, c

    # ¦¡¦¡ Forward & Loss ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def forward(self, z, c):
        """
        z : latent Tensor  [B, z_ch, h, w]
        c : context Tensor [B, seq_len, context_dim]
        """
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long() # [bs]
        return self.p_losses(z, c, t)

    def apply_model(self, z_noisy, t, cond):
        """
        Run the denoising UNet.

        cond may be:
          - Tensor [B, seq, dim]  (converted to c_crossattn list internally)
          - dict   {'c_crossattn': [Tensor]}  (passed through as-is)
        """
        if isinstance(cond, torch.Tensor):
            cond = {'c_crossattn': [cond]}
        elif isinstance(cond, list):
            cond = {'c_crossattn': cond}
        # dict case: passed straight to DiffusionWrapper
        return self.model(z_noisy, t, **cond)

    def p_losses(self, x_start, cond, t, noise=None):
        """
        Core diffusion loss.

        x_start : latent z  [B, z_ch, h, w]
        cond    : context c [B, seq, dim]
        t       : timesteps [B]
        """
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
            raise NotImplementedError(f'Unknown parameterization: {self.parameterization}')

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

    # ¦¡¦¡ Training / Validation steps ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def training_step(self, batch):
        """
        Returns (loss, loss_dict).

        Caller:
            loss, loss_dict = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()
        """
        z, c = self.get_input(batch)
        return self.forward(z, c)

    @torch.no_grad()
    def validation_step(self, batch):
        """Returns (loss_dict, loss_dict_ema)."""
        z, c = self.get_input(batch)
        _, loss_dict = self.forward(z, c)
        with self.ema_scope():
            _, loss_dict_ema = self.forward(z, c)
            loss_dict_ema = {k + '_ema': v for k, v in loss_dict_ema.items()}
        return loss_dict, loss_dict_ema

    # ¦¡¦¡ Reverse process (sampling) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def p_mean_variance(self, z, c, t, clip_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        model_out = self.apply_model(z, t, c)

        if score_corrector is not None:
            assert self.parameterization == 'eps'
            model_out = score_corrector.modify_score(self, model_out, z, t, c, **corrector_kwargs)

        if self.parameterization == 'eps':
            x_recon = self.predict_start_from_noise(z, t=t, noise=model_out)
        elif self.parameterization == 'x0':
            x_recon = model_out
        else:
            raise NotImplementedError

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_recon, x_t=z, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, z, c, t, clip_denoised=False, repeat_noise=False,
                 return_x0=False, temperature=1., noise_dropout=0.,
                 score_corrector=None, corrector_kwargs=None):
        b, device = z.shape[0], z.device
        outputs = self.p_mean_variance(
            z, c, t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(z.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = F.dropout(noise, p=noise_dropout)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(z.shape) - 1)))
        sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        return (sample, x0) if return_x0 else sample

    @torch.no_grad()
    def p_sample_loop(self, c, shape, return_intermediates=False,
                      x_T=None, verbose=True, timesteps=None,
                      log_every_t=None):
        """Ancestral sampling loop in latent space."""
        log_every_t = log_every_t or self.log_every_t
        device      = self.device
        b           = shape[0]
        z           = x_T if x_T is not None else torch.randn(shape, device=device)
        timesteps   = timesteps or self.num_timesteps
        intermediates = [z]

        iterator = tqdm(reversed(range(timesteps)), desc='Sampling', total=timesteps) \
                   if verbose else reversed(range(timesteps))
        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            z  = self.p_sample(z, c, ts, clip_denoised=self.clip_denoised)
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(z)

        return (z, intermediates) if return_intermediates else z

    @torch.no_grad()
    def sample(self, c, batch_size=1, return_intermediates=False,
               x_T=None, verbose=True, timesteps=None):
        """
        Generate samples conditioned on c (BioBERT embeddings).

        Args:
            c             : [B, seq_len, context_dim]
            batch_size    : number of samples to draw
            return_intermediates: also return denoising trajectory
        Returns:
            decoded image [B, C, H, W]
        """
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        z = self.p_sample_loop(
            c, shape,
            return_intermediates=return_intermediates,
            x_T=x_T, verbose=verbose, timesteps=timesteps,
        )
        if return_intermediates:
            z, intermediates = z
            return self.decode_first_stage(z), intermediates
        return self.decode_first_stage(z)

    @torch.no_grad()
    def sample_log(self, c, batch_size, ddim, ddim_steps, **kwargs):
        if ddim and DDIMSampler is not None:
            sampler = DDIMSampler(self)
            shape   = (self.channels, self.image_size, self.image_size)
            samples, intermediates = sampler.sample(
                ddim_steps, batch_size, shape, c, verbose=False, **kwargs
            )
        else:
            samples, intermediates = self.p_sample_loop(
                c, (batch_size, self.channels, self.image_size, self.image_size),
                return_intermediates=True, **kwargs,
            )
        return samples, intermediates

    # ¦¡¦¡ Prior BPD ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def _prior_bpd(self, x_start):
        b = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * b, device=x_start.device)
        qt_mean, _, qt_logvar = self.q_mean_variance(x_start, t)
        kl = normal_kl(qt_mean, qt_logvar, 0.0, 0.0)
        return mean_flat(kl) / np.log(2.0)

    # ¦¡¦¡ Colorize helper ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, 'colorize'):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.

    # ¦¡¦¡ Optimizer ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡

    def build_optimizer(self, lr=None):
        """
        Returns AdamW for the UNet (and logvar if learn_logvar=True).
        The first_stage_model (AutoencoderKL) is always frozen.
        """
        lr = lr or self.lr
        params = list(self.model.parameters())
        if self.learn_logvar:
            params.append(self.logvar)

        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in params if p.requires_grad)
        print(f'[LDM] trainable: {n_train:,} / {n_total:,}  ({n_train/n_total*100:.1f}%)')
        return torch.optim.AdamW(params, lr=lr)


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    from ddpm import DDPM, extract_into_tensor, noise_like  # noqa: E402
    from Model.util_network import exists, default, mean_flat  # noqa: E402
    from Model.VAE.Encoder import Encoder,DiagonalGaussianDistribution
    from Model.VAE.Autoencoder import VAE
    from UNetmodel import UNetModel
    from Data.nih import NIH
    from torch.utils.data import DataLoader, Dataset
    import argparse
    #
    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    #
    # =========================================================================
    # NIH ChestX-ray14 Dataset + DataLoader
    # =========================================================================
    nih_args = argparse.Namespace(
        root_path='/storage/hjchoi/archive/DATA',
        task='train',
        bs=8,
        image_size=256,
        image_show=False,
    )
    nih_dataset = NIH(nih_args)
    nih_loader = DataLoader(
        nih_dataset,
        batch_size=nih_args.bs,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    # =========================================================================
    # Class Label Embedder  (runs BEFORE LDM)
    # =========================================================================
    cls_embedder = ClassLabelEmbedder(
        embed_dim=256,
        output_dim=512,  # must match UNet context_dim
    ).to(device)

    def _infinite_batches(loader):
        """Yield batches indefinitely, restarting at epoch end."""
        while True:
            yield from loader


    _nih_gen = _infinite_batches(nih_loader)
    print(f'[NIH] dataset size: {len(nih_dataset)}  '
          f'| batch_size: {nih_args.bs}  | task: {nih_args.task}')


    def VAE_parse_args():
        parser = argparse.ArgumentParser()

        # Data
        parser.add_argument("--root_path", default="/storage/hjchoi/archive/DATA")
        parser.add_argument("--task", default="train", choices=["train", "val", "test"])
        parser.add_argument("--bs", default=2, type=int, help='batch size')
        parser.add_argument("--image_size", default=256, type=int, help='the value to resize')
        parser.add_argument("--image_show", default=True, type=bool)

        # Encoder
        parser.add_argument("--in_channels", default=1, type=int, help='Number of input img channels, NIH=gray-scale')
        parser.add_argument("--ch", default=128, type=int, help='Base channel')
        parser.add_argument("--ch_mult", default=[1, 1, 2, 2, 4], help='Channel multipliers per each level')
        parser.add_argument("--num_res_blocks", default=2, type=int, help='Number of residual blocks per each level')
        parser.add_argument("--attn_resolutions", default=[64, 32, 16], help='the resolution at which attention is applied')
        parser.add_argument("--dropout", default=0.0, type=float)
        parser.add_argument("--resamp_with_conv", default=True, type=bool,
                            help='Use strided conv for downsampling; False uses avg-pool')
        parser.add_argument("--resolution", default=256, type=int, help='Input spatial resolution (H = W)')
        parser.add_argument("--z_channels", default=1, type=int, help='Latent z-space channel dim')
        parser.add_argument("--double_z", default=True, type=bool,
                            help='Output 2*z_channels (mean + logvar) for VAE reparameterisation')
        parser.add_argument("--dims", default=2, type=int, help="Conv dim; N of ConvNd", choices=[1, 2, 3])
        parser.add_argument('--test_case', default=False, type=bool, help='True: not load real data, using rand values')

        # Decoder
        parser.add_argument("--out_channels", default=1, type=int, help='Number of output channels')

        # Print
        parser.add_argument('--verbose', default=False, type=bool,
                            help='True: print Block and output shape at each levels')
        return parser.parse_args()

    def Unet_parse_args():
        parser = argparse.ArgumentParser(description="UNetModel config")

        # Device
        parser.add_argument("--device_id", default=1)

        # Architecture
        parser.add_argument("--image_size", type=int, default=16)
        parser.add_argument("--in_channels", type=int, default=1)
        parser.add_argument("--out_channels", type=int, default=1)
        parser.add_argument("--model_channels", type=int, default=128)
        parser.add_argument("--num_res_blocks", type=int, default=2)
        parser.add_argument("--channel_mult", type=int, default=[1, 2, 2, 4])
        parser.add_argument("--attention_resolutions", type=int, default=[1, 2, 4])
        parser.add_argument("--dropout", type=float, default=0.0)
        parser.add_argument("--dims", type=int, default=2)
        parser.add_argument("--conv_resample", action="store_true", default=True)

        # Attention
        parser.add_argument("--num_heads", type=int, default=-1)
        parser.add_argument("--num_head_channels", type=int, default=8)
        parser.add_argument("--num_heads_upsample", type=int, default=-1)
        parser.add_argument("--use_spatial_transformer", action="store_true", default=True)
        parser.add_argument("--transformer_depth", type=int, default=1)
        parser.add_argument("--context_dim", type=int, default=512)
        parser.add_argument("--use_new_attention_order", action="store_true", default=False)
        parser.add_argument("--legacy", action="store_true", default=True)

        # ResBlock options
        parser.add_argument("--use_scale_shift_norm", action="store_true", default=False)
        parser.add_argument("--resblock_updown", action="store_true", default=False)

        # Misc
        parser.add_argument("--num_classes", type=int, default=None)
        parser.add_argument("--n_embed", type=int, default=None)
        parser.add_argument("--use_checkpoint", action="store_true", default=False)
        parser.add_argument("--use_fp16", action="store_true", default=False)

        # Debug
        parser.add_argument("--write_json", action="store_true", default=False)

        return parser.parse_args()

    unet_args = Unet_parse_args()
    vae_args = VAE_parse_args()

    # image [B,1,64,64] -> VAE -> z [B,1,16,16]
    # UNet in latent space: 4 channels, 16x16
    model = LatentDiffusion(
        device=device,
        unet              = UNetModel(unet_args),
        first_stage_model = VAE(vae_args),
        first_stage_key    = 'image',
        cond_stage_key     = 'context',
        conditioning_key  = 'crossattn',
        timesteps         = 100,
        beta_schedule     = 'linear',
        image_size        = 16,       # latent spatial size
        channels          = 1,        # latent channels
        use_ema           = True,
        lr                = 1e-4,
    ).to(device)

    # ¦¡¦¡ Fake batch ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    # 'image'  : raw CXR  [B, 1, 64, 64]
    # 'context': BioBERT  [B, seq=16, dim=64]

    def fake_batch(B=2, img_h=256, img_w=256, seq=16, ctx_dim=512):
        return {'image': torch.randn(B, 1, img_h, img_w).to(device),
                'context': torch.randn(B, seq, ctx_dim).to(device),}


    def make_batch() -> dict:
        """
        Pull one batch from the NIH DataLoader (via infinite generator).

        Pipeline:
            NIH.__getitem__  ¡æ  (img [1,256,256],  label_str)
            label_str        ¡æ  encode_label_str()
                             ¡æ  [BOS, label_1, SEP, label_2, ..., EOS, PAD...]
                             ¡æ  ClassLabelEmbedder
                             ¡æ  context [B, seq_len, 512]
        Returns:
            {'image': Tensor[B,1,256,256], 'context': Tensor[B,seq_len,512]}
        """
        imgs, label_strs = next(_nih_gen)  # closure over _nih_gen

        imgs = imgs.to(device)  # [B, 1, 256, 256]
        ctx = cls_embedder(list(label_strs)).detach()  # [B, seq_len, 512]

        return {'image': imgs, 'context': ctx}

    # ¦¡¦¡ scale_factor init ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    first_batch = make_batch()
    model.init_scale_factor(first_batch, is_first_batch=True)

    # ¦¡¦¡ Training loop ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    optimizer = model.build_optimizer(lr=1e-4)
    print('\n--- Training (3 steps) ---')
    model.train()
    for step in range(3):
        batch = make_batch()
        loss, loss_dict = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_ema()
        info = '  '.join(f'{k}={v.item():.4f}' for k, v in loss_dict.items())
        print(f'  step {step+1}/3 | {info}')

    # Validation
    print('\n--- Validation ---')
    model.eval()
    ld, ld_ema = model.validation_step(make_batch())
    print('  val:', {k: f'{v.item():.4f}' for k, v in ld.items()})
    print('  ema:', {k: f'{v.item():.4f}' for k, v in ld_ema.items()})

    # Sampling
    print('\n--- Sampling (5 steps, 2 samples) ---')
    c = torch.randn(2, 16, 512, device=device)
    x_gen = model.sample(c, batch_size=2, verbose=False, timesteps=5)
    print(f'  generated: {x_gen.shape}')   # [2, 1, 64, 64]

    print('\nDebug run completed successfully!')