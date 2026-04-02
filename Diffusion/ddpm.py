"""
Diffusion/ddpm.py  –  Standalone DDPM (pure PyTorch, no pytorch_lightning)
============================================================================

Dependencies replaced:
    ldm.modules.diffusionmodules.util  →  make_beta_schedule / extract_into_tensor / noise_like  (this file)
    ldm.modules.ema.LitEma             →  EMA  (this file)
    ldm.util.exists / default          →  Model/util_network.py
    instantiate_from_config            →  unet: nn.Module passed directly

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Model'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Swin_origin_Model'))

    from LDM import UNetModel
    from Diffusion.ddpm import DDPM

    unet  = UNetModel(image_size=16, in_channels=4, model_channels=128, ...)
    model = DDPM(unet=unet, timesteps=1000, channels=4, image_size=16).to(device)

    optimizer = model.build_optimizer()
    for batch in loader:
        loss, loss_dict = model.training_step(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.update_ema()
"""

import gc
import math
import os
import sys
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid
from tqdm import tqdm

# ── Model utilities from the existing codebase ────────────────────────────────
_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Model')
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

from util_network import exists, default  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Diffusion schedule utilities
# (replaces ldm.modules.diffusionmodules.util)
# ══════════════════════════════════════════════════════════════════════════════

def make_beta_schedule(schedule: str, n_timestep: int,
                       linear_start: float = 1e-4,
                       linear_end: float = 2e-2,
                       cosine_s: float = 8e-3) -> np.ndarray:
    """Return beta values for T timesteps under the requested schedule."""
    if schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)

    elif schedule == "cosine":
        steps = np.arange(n_timestep + 1, dtype=np.float64) / n_timestep + cosine_s
        alphas = np.cos(steps / (1 + cosine_s) * math.pi / 2) ** 2
        alphas /= alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, 0, 0.999)

    elif schedule == "sqrt_linear":
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2

    elif schedule == "sqrt":
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64) ** 0.5

    else:
        raise ValueError(f"Unknown beta schedule: '{schedule}'")

    return betas


def extract_into_tensor(a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
    """Gather values from 1-D tensor `a` at indices `t`, broadcast to `x_shape`."""
    out = a.gather(-1, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


def noise_like(shape: tuple, device: torch.device, repeat: bool = False) -> torch.Tensor:
    """Standard or repeated Gaussian noise."""
    if repeat:
        return torch.randn((1, *shape[1:]), device=device).repeat(
            shape[0], *((1,) * (len(shape) - 1))
        )
    return torch.randn(shape, device=device)


# ══════════════════════════════════════════════════════════════════════════════
# Exponential Moving Average
# (replaces ldm.modules.ema.LitEma)
# ══════════════════════════════════════════════════════════════════════════════

class EMA:
    """
    Maintains shadow (exponentially averaged) copies of model weights.

    Example:
        ema = EMA(model, decay=0.9999)
        # inside training loop, after optimizer.step():
        ema.update(model)
        # for validation / sampling:
        with ema.scope(model):
            output = model(x)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay   = decay
        self._shadow  = {n: p.data.detach().cpu().clone()
                         for n, p in model.named_parameters() if p.requires_grad}
        self._backup  = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        """shadow = decay·shadow + (1−decay)·param"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self._shadow[name].mul_(self.decay).add_(
                    param.data.cpu(), alpha=1.0 - self.decay
                )

    def store(self, model: nn.Module):
        """Save current model weights so they can be restored later."""
        self._backup = {n: p.data.clone()
                        for n, p in model.named_parameters() if p.requires_grad}

    def restore(self, model: nn.Module):
        """Restore previously saved model weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def copy_to(self, model: nn.Module):
        """Overwrite model weights with shadow (EMA) weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._shadow:
                param.data.copy_(self._shadow[name].to(param.device))

    @contextmanager
    def scope(self, model: nn.Module, label: str = None):
        """Context manager: temporarily use EMA weights for inference."""
        self.store(model)
        self.copy_to(model)
        if label:
            print(f"[EMA] {label}: switched to EMA weights")
        try:
            yield
        finally:
            self.restore(model)
            if label:
                print(f"[EMA] {label}: restored training weights")


# ══════════════════════════════════════════════════════════════════════════════
# DiffusionWrapper
# ══════════════════════════════════════════════════════════════════════════════

class DiffusionWrapper(nn.Module):
    """
    Thin wrapper around the UNet that dispatches conditioning inputs.

    conditioning_key:
        None        – unconditional  (x, t)
        'concat'    – channel-wise concatenation  (x ‖ c_concat, t)
        'crossattn' – cross-attention context     (x, t, context=c_crossattn)
        'hybrid'    – concat + crossattn
        'adm'       – class embedding (y=c_crossattn[0])
    """

    VALID_KEYS = {None, 'concat', 'crossattn', 'hybrid', 'adm'}

    def __init__(self, unet: nn.Module, conditioning_key=None):
        super().__init__()
        assert conditioning_key in self.VALID_KEYS, \
            f"conditioning_key must be one of {self.VALID_KEYS}, got '{conditioning_key}'"
        self.diffusion_model  = unet
        self.conditioning_key = conditioning_key

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                c_concat: list = None, c_crossattn: list = None) -> torch.Tensor:
        key = self.conditioning_key

        if key is None:
            return self.diffusion_model(x, t)

        if key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            return self.diffusion_model(xc, t)

        if key == 'crossattn':
            cc = torch.cat(c_crossattn, dim=1)
            return self.diffusion_model(x, t, context=cc)

        if key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, dim=1)
            return self.diffusion_model(xc, t, context=cc)

        if key == 'adm':
            return self.diffusion_model(x, t, y=c_crossattn[0])

        raise NotImplementedError(key)


# ══════════════════════════════════════════════════════════════════════════════
# DDPM
# ══════════════════════════════════════════════════════════════════════════════

class DDPM(nn.Module):
    """
    Classic DDPM with Gaussian diffusion, in image space.

    Args:
        unet:               Denoising UNet (nn.Module).  Passed directly — no config dict needed.
        conditioning_key:   Forwarded to DiffusionWrapper (None, 'concat', 'crossattn', …).
        timesteps:          Total diffusion steps T.
        beta_schedule:      "linear" | "cosine" | "sqrt_linear" | "sqrt".
        loss_type:          "l1" | "l2".
        parameterization:   "eps"  → model predicts noise ε.
                            "x0"   → model predicts clean image x₀.
        use_ema / ema_decay: Whether to maintain EMA shadow weights.
        lr:                 Default learning-rate used by build_optimizer().
    """

    def __init__(self,
                 unet: nn.Module,
                 conditioning_key=None,
                 timesteps: int = 1000,
                 beta_schedule: str = "linear",
                 loss_type: str = "l2",
                 ckpt_path: str = None,
                 ignore_keys: list = [],
                 use_ema: bool = True,
                 ema_decay: float = 0.9999,
                 first_stage_key: str = "image",
                 image_size: int = 256,
                 channels: int = 1,
                 log_every_t: int = 100,
                 clip_denoised: bool = True,
                 linear_start: float = 1e-4,
                 linear_end: float = 2e-2,
                 cosine_s: float = 8e-3,
                 given_betas=None,
                 original_elbo_weight: float = 0.,
                 v_posterior: float = 0.,
                 l_simple_weight: float = 1.,
                 parameterization: str = "eps",
                 learn_logvar: bool = False,
                 logvar_init: float = 0.,
                 lr: float = 1e-4,
                 ):
        super().__init__()
        assert parameterization in ("eps", "x0"), \
            'Only "eps" and "x0" parameterization are supported'

        self.parameterization      = parameterization
        self.first_stage_key       = first_stage_key
        self.image_size            = image_size
        self.channels              = channels
        self.clip_denoised         = clip_denoised
        self.log_every_t           = log_every_t
        self.loss_type             = loss_type
        self.lr                    = lr
        self.v_posterior           = v_posterior
        self.original_elbo_weight  = original_elbo_weight
        self.l_simple_weight       = l_simple_weight

        # UNet wrapped for conditioning dispatch
        self.model = DiffusionWrapper(unet, conditioning_key)
        n_params   = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[DDPM] {self.parameterization}-prediction  |  UNet params: {n_params:,}")

        # EMA
        self.use_ema = use_ema
        if self.use_ema:
            self.ema = EMA(self.model, decay=ema_decay)
            print(f"[DDPM] EMA enabled  (decay={ema_decay})")

        # Noise schedule buffers
        self.register_schedule(
            given_betas=given_betas, beta_schedule=beta_schedule,
            timesteps=timesteps, linear_start=linear_start,
            linear_end=linear_end, cosine_s=cosine_s,
        )

        # Learnable log-variance
        self.learn_logvar = learn_logvar
        self.logvar = torch.full((self.num_timesteps,), fill_value=logvar_init)
        if learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

        if ckpt_path is not None:
            self.load_from_ckpt(ckpt_path, ignore_keys)

    # ── Device property ────────────────────────────────────────────────────────

    @property
    def device(self) -> torch.device:
        """Current device inferred from the registered noise schedule buffers."""
        return self.betas.device

    # ── Noise schedule ─────────────────────────────────────────────────────────

    def register_schedule(self, given_betas=None, beta_schedule="linear",
                          timesteps=1000, linear_start=1e-4,
                          linear_end=2e-2, cosine_s=8e-3):
        betas = given_betas if given_betas is not None else make_beta_schedule(
            beta_schedule, timesteps,
            linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s,
        )
        alphas              = 1.0 - betas
        alphas_cumprod      = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.num_timesteps  = int(betas.shape[0])
        self.linear_start   = linear_start
        self.linear_end     = linear_end

        reg = partial(self.register_buffer)
        t   = partial(torch.tensor, dtype=torch.float32)

        reg('betas',                          t(betas))
        reg('alphas_cumprod',                 t(alphas_cumprod))
        reg('alphas_cumprod_prev',            t(alphas_cumprod_prev))
        reg('sqrt_alphas_cumprod',            t(np.sqrt(alphas_cumprod)))
        reg('sqrt_one_minus_alphas_cumprod',  t(np.sqrt(1.0 - alphas_cumprod)))
        reg('log_one_minus_alphas_cumprod',   t(np.log(1.0 - alphas_cumprod)))
        reg('sqrt_recip_alphas_cumprod',      t(np.sqrt(1.0 / alphas_cumprod)))
        reg('sqrt_recipm1_alphas_cumprod',    t(np.sqrt(1.0 / alphas_cumprod - 1)))

        posterior_variance = (
            (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            + self.v_posterior * betas
        )
        reg('posterior_variance',             t(posterior_variance))
        reg('posterior_log_variance_clipped', t(np.log(np.maximum(posterior_variance, 1e-20))))
        reg('posterior_mean_coef1',           t(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)))
        reg('posterior_mean_coef2',           t((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb = self.betas ** 2 / (
                2 * self.posterior_variance * t(alphas) * (1 - self.alphas_cumprod)
            )
        else:
            lvlb = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2.0 * 1 - torch.Tensor(alphas_cumprod))

        lvlb[0] = lvlb[1]
        self.register_buffer('lvlb_weights', lvlb, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    # ── EMA helpers ────────────────────────────────────────────────────────────

    def update_ema(self):
        """Call after every optimizer.step() to update the EMA shadow weights."""
        if self.use_ema:
            self.ema.update(self.model)

    @contextmanager
    def ema_scope(self, label: str = None):
        """Context manager: temporarily use EMA weights (for validation / sampling)."""
        if self.use_ema:
            with self.ema.scope(self.model, label=label):
                yield
        else:
            yield

    # ── Checkpoint ─────────────────────────────────────────────────────────────

    def load_from_ckpt(self, path: str, ignore_keys: list = []):
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("state_dict", sd)
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                print(f"  [ckpt] Removing key: {k}")
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"[DDPM] Loaded ckpt '{path}'  missing={len(missing)}  unexpected={len(unexpected)}")

    # ── Diffusion forward process ──────────────────────────────────────────────

    def q_mean_variance(self, x_start, t):
        mean         = extract_into_tensor(self.sqrt_alphas_cumprod,         t, x_start.shape) * x_start
        variance     = extract_into_tensor(1.0 - self.alphas_cumprod,        t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod,             t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ── Diffusion reverse process ──────────────────────────────────────────────

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod,   t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var     = extract_into_tensor(self.posterior_variance,             t, x_t.shape)
        log_var = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        else:
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        mean, var, log_var = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return mean, var, log_var

    # ── Loss ──────────────────────────────────────────────────────────────────

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            return loss.mean() if mean else loss
        elif self.loss_type == 'l2':
            return F.mse_loss(target, pred, reduction='mean' if mean else 'none')
        raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def p_losses(self, x_start, t, noise=None):
        noise   = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start, t, noise=noise)
        out     = self.model(x_noisy, t)

        target = noise if self.parameterization == "eps" else x_start
        loss   = self.get_loss(out, target, mean=False).mean(dim=[1, 2, 3])

        prefix = 'train' if self.training else 'val'
        loss_simple = loss.mean() * self.l_simple_weight
        loss_vlb    = (self.lvlb_weights[t] * loss).mean()
        loss_total  = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict = {
            f'{prefix}/loss_simple': loss.mean(),
            f'{prefix}/loss_vlb':    loss_vlb,
            f'{prefix}/loss':        loss_total,
        }
        gc.collect()
        return loss_total, loss_dict

    def forward(self, x, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    # ── Sampling ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b      = x.shape[0]
        mean, _, log_var = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise  = noise_like(x.shape, x.device, repeat_noise)
        mask   = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return mean + mask * (0.5 * log_var).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b      = shape[0]
        img    = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t_batch, clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        return (img, intermediates) if return_intermediates else img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        return self.p_sample_loop(
            (batch_size, self.channels, self.image_size, self.image_size),
            return_intermediates=return_intermediates,
        )

    # ── Training / validation steps ───────────────────────────────────────────

    def get_input(self, batch, k):
        x = batch[k]
        if x.ndim == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        return x.to(memory_format=torch.contiguous_format).float()

    def training_step(self, batch):
        """
        Returns (loss, loss_dict).

        Training loop skeleton:
            loss, loss_dict = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()          # keep EMA weights in sync
        """
        x = self.get_input(batch, self.first_stage_key)
        return self(x)

    @torch.no_grad()
    def validation_step(self, batch):
        """Returns (loss_dict, loss_dict_ema)."""
        x              = self.get_input(batch, self.first_stage_key)
        _, loss_dict   = self(x)

        with self.ema_scope():
            _, loss_dict_ema = self(x)
            loss_dict_ema = {k + '_ema': v for k, v in loss_dict_ema.items()}

        return loss_dict, loss_dict_ema

    # ── Optimizer ─────────────────────────────────────────────────────────────

    def build_optimizer(self, lr: float = None):
        """Create and return an AdamW optimizer for this model."""
        lr     = lr or self.lr
        params = list(self.model.parameters())
        if self.learn_logvar:
            params += [self.logvar]
        return torch.optim.AdamW(params, lr=lr)

    # ── Image logging ─────────────────────────────────────────────────────────

    def _rows_from_list(self, samples):
        grid = rearrange(torch.stack(samples), 'n b c h w -> (b n) c h w')
        return make_grid(grid, nrow=len(samples))

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None):
        device   = self.betas.device
        log      = {}
        x        = self.get_input(batch, self.first_stage_key).to(device)
        N, n_row = min(x.shape[0], N), min(x.shape[0], n_row)
        x        = x[:N]
        log["inputs"] = x

        # Noising row
        diffusion_row = []
        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t_b   = torch.full((n_row,), t, device=device, dtype=torch.long)
                noisy = self.q_sample(x[:n_row], t_b, noise=torch.randn_like(x[:n_row]))
                diffusion_row.append(noisy)
        log["diffusion_row"] = self._rows_from_list(diffusion_row)

        if sample:
            with self.ema_scope("log_images"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)
            log["samples"]     = samples
            log["denoise_row"] = self._rows_from_list(denoise_row)

        if return_keys:
            keys = [k for k in return_keys if k in log]
            return {k: log[k] for k in keys} if keys else log
        return log
