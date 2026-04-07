"""
Diffusion/LDM.py  -  LatentDiffusion (pure PyTorch, no pytorch_lightning)
==========================================================================
Changes from Swin_origin_Model/LattentDiffusion.py:
  - pytorch_lightning removed entirely
  - on_train_batch_start  -> init_scale_factor(batch, is_first_batch=False)
  - configure_optimizers  -> build_optimizer(lr=None)
  - optimizer_zero_grad / get_train_dataloader removed
  - self.learning_rate    -> self.lr
  - ConfigAttributeError  -> AttributeError
  - VQModelInterface      -> duck-typing: hasattr(model, 'quantize')
  - instantiate_from_config / disabled_train / normal_kl implemented locally
  - unet_config in kwargs auto-instantiated and passed as unet= to DDPM
"""
import copy, gc, importlib, math, os, sys
from contextlib import contextmanager
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torchvision.utils import make_grid
from tqdm import tqdm

_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_this_dir, '..', 'Model')
_swin_dir  = os.path.join(_this_dir, '..', 'Swin_origin_Model')
for _d in [_model_dir, _swin_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from ddpm import DDPM, extract_into_tensor, noise_like
from util_network import exists, default, mean_flat
from Encoder import DiagonalGaussianDistribution
from UNetBlock_module import ResBlock
from attention_module import SpatialTransformer

try:
    import opacus
except ImportError:
    opacus = None

try:
    from DDIMSampler import DDIMSampler
except ImportError:
    DDIMSampler = None

AttentionBlock = None  # not present in this codebase


def disabled_train(self, mode=True):
    return self

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2)
                  + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))

def instantiate_from_config(config):
    if isinstance(config, nn.Module):
        return config
    if isinstance(config, str):
        return None
    if not isinstance(config, dict):
        raise ValueError(f"config must be dict or nn.Module, got {type(config)}")
    if 'target' not in config:
        raise KeyError(f"Expected key 'target', got: {list(config.keys())}")
    module_path, cls_name = config['target'].rsplit('.', 1)
    return getattr(importlib.import_module(module_path), cls_name)(**config.get('params', {}))

def log_txt_as_img(wh, xc):
    b = len(xc) if hasattr(xc, '__len__') else 1
    return torch.zeros(b, 3, wh[0], wh[1])

def isimage(x):
    return isinstance(x, torch.Tensor) and x.ndim == 4

def ismap(x):
    return isinstance(x, torch.Tensor) and x.ndim == 4 and x.shape[1] > 3


class LatentDiffusion(DDPM):
    """
    Latent Diffusion Model - operates in the latent space of a frozen VAE/VQVAE.

    Training loop (caller's responsibility):
        model.init_scale_factor(batch, is_first_batch=True)
        optimizer = model.build_optimizer()
        for batch in loader:
            loss, loss_dict = model.training_step(batch)
            optimizer.zero_grad(); loss.backward()
            optimizer.step(); model.update_ema()
    """
    def __init__(self, first_stage_config, cond_stage_config,
                 num_timesteps_cond=None, cond_stage_key='image',
                 cond_stage_trainable=False, concat_mode=True,
                 cond_stage_forward=None, conditioning_key=None,
                 scale_factor=1.0, scale_by_std=False,
                 retrain_attention=False, train_attention_only=False,
                 attention_flag='spatial', condition_mask=0, diffusion_mask=0,
                 train_condition_only=False, train_input_blocks_only=False,
                 train_resblocks_only=False, ablation_blocks=-1,
                 dp_config=None, *args, **kwargs):

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']

        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None

        ckpt_path   = kwargs.pop('ckpt_path',  None)
        ignore_keys = kwargs.pop('ignore_keys', [])

        if 'unet' not in kwargs and 'unet_config' in kwargs:
            kwargs['unet'] = instantiate_from_config(kwargs.pop('unet_config'))
        elif 'unet_config' in kwargs:
            kwargs.pop('unet_config')

        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.concat_mode              = concat_mode
        self.cond_stage_trainable     = cond_stage_trainable
        self.cond_stage_key           = cond_stage_key
        self.use_positional_encodings = False

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except AttributeError:
            self.num_downs = 0

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.cond_stage_forward  = cond_stage_forward
        self.clip_denoised       = False
        self.bbox_tokenizer      = None
        self.retrain_attention       = retrain_attention
        self.train_attention_only    = train_attention_only
        self.attention_flag          = attention_flag
        self.condition_mask          = condition_mask
        self.diffuison_mask          = diffusion_mask
        self.train_condition_only    = train_condition_only
        self.train_input_blocks_only = train_input_blocks_only
        self.train_resblocks_only    = train_resblocks_only
        self.ablation_blocks         = ablation_blocks
        self.restarted_from_ckpt     = False

        if ckpt_path is not None:
            if self.retrain_attention:
                ignore_keys = self.init_attention(self.attention_flag)
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            self.restarted_from_ckpt = True

        self.dp_config = dp_config
        if dp_config and getattr(dp_config, 'enabled', False) and opacus is not None:
            self.privacy_engine = opacus.PrivacyEngine()

    def init_attention(self, attention_flag='spatial'):
        ignore_keys = []
        modules = self.model.named_modules()
        cls = SpatialTransformer if attention_flag == 'spatial' else AttentionBlock
        if cls is None:
            return ignore_keys
        for nm, m in modules:
            if isinstance(m, cls):
                ignore_keys.append(nm)
                ignore_keys.append(copy.copy(nm).replace('.', ''))
                print(f"init {nm}")
        return ignore_keys

    def make_cond_schedule(self):
        self.cond_ids = torch.full((self.num_timesteps,),
                                   fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1,
                                         self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def register_schedule(self, given_betas=None, beta_schedule='linear',
                          timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps,
                                  linear_start, linear_end, cosine_s)
        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    @torch.no_grad()
    def init_scale_factor(self, batch, is_first_batch=False):
        """Call once before the very first training step when scale_by_std=True."""
        if self.scale_by_std and is_first_batch and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'Do not combine custom + std rescaling.'
            print('### USING STD-RESCALING ###')
            x = super().get_input(batch, self.first_stage_key).to(self.device)
            z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f'setting self.scale_factor to {self.scale_factor}')
            print('### USING STD-RESCALING ###')

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for p in self.first_stage_model.parameters():
            p.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == '__is_first_stage__':
                print('Using first stage also as cond stage.')
                self.cond_stage_model = self.first_stage_model
            elif config == '__is_unconditional__':
                print(f'Training {self.__class__.__name__} as unconditional model.')
                self.cond_stage_model = None
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for p in self.cond_stage_model.parameters():
                    p.requires_grad = False
        else:
            assert config not in ('__is_first_stage__', '__is_unconditional__')
            self.cond_stage_model = instantiate_from_config(config)

    def init_from_ckpt(self, path, ignore_keys=None):
        ignore_keys = ignore_keys or []
        sd = torch.load(path, map_location='cpu')
        sd = sd.get('state_dict', sd)
        for k in list(sd.keys()):
            if any(k.startswith(ik) for ik in ignore_keys):
                del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f'[LDM] Loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}')


    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(type(encoder_posterior))
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)
        return torch.cat([y, x], dim=-1)

    def delta_border(self, h, w):
        arr = self.meshgrid(h, w) / torch.tensor([h-1, w-1]).view(1, 1, 2)
        dist_lu = torch.min(arr,     dim=-1, keepdim=True)[0]
        dist_rd = torch.min(1 - arr, dim=-1, keepdim=True)[0]
        return torch.min(torch.cat([dist_lu, dist_rd], dim=-1), dim=-1)[0]

    def get_weighting(self, h, w, Ly, Lx, device):
        sp = self.split_input_params
        weighting = torch.clip(self.delta_border(h, w), sp['clip_min_weight'], sp['clip_max_weight'])
        weighting = weighting.view(1, h*w, 1).repeat(1, 1, Ly*Lx).to(device)
        if sp['tie_braker']:
            L_w = torch.clip(self.delta_border(Ly, Lx), sp['clip_min_tie_weight'], sp['clip_max_tie_weight'])
            weighting = weighting * L_w.view(1, 1, Ly*Lx).to(device)
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):
        bs, nc, h, w = x.shape
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1
        fp = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
        unfold = torch.nn.Unfold(**fp)
        if uf == 1 and df == 1:
            fold = torch.nn.Fold(output_size=x.shape[2:], **fp)
            wt = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            norm = fold(wt).view(1, 1, h, w)
            wt   = wt.view(1, 1, kernel_size[0], kernel_size[1], Ly*Lx)
        elif uf > 1 and df == 1:
            fp2 = dict(kernel_size=(kernel_size[0]*uf, kernel_size[1]*uf), dilation=1, padding=0,
                       stride=(stride[0]*uf, stride[1]*uf))
            fold = torch.nn.Fold(output_size=(x.shape[2]*uf, x.shape[3]*uf), **fp2)
            wt   = self.get_weighting(kernel_size[0]*uf, kernel_size[1]*uf, Ly, Lx, x.device).to(x.dtype)
            norm = fold(wt).view(1, 1, h*uf, w*uf)
            wt   = wt.view(1, 1, kernel_size[0]*uf, kernel_size[1]*uf, Ly*Lx)
        elif df > 1 and uf == 1:
            fp2 = dict(kernel_size=(kernel_size[0]//df, kernel_size[1]//df), dilation=1, padding=0,
                       stride=(stride[0]//df, stride[1]//df))
            fold = torch.nn.Fold(output_size=(x.shape[2]//df, x.shape[3]//df), **fp2)
            wt   = self.get_weighting(kernel_size[0]//df, kernel_size[1]//df, Ly, Lx, x.device).to(x.dtype)
            norm = fold(wt).view(1, 1, h//df, w//df)
            wt   = wt.view(1, 1, kernel_size[0]//df, kernel_size[1]//df, Ly*Lx)
        else:
            raise NotImplementedError
        return fold, unfold, norm, wt

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, 'split_input_params') and self.split_input_params.get('patch_distributed_vq'):
            sp = self.split_input_params
            ks, stride, df = sp['ks'], sp['stride'], sp['vqf']
            sp['original_image_size'] = x.shape[-2:]
            bs, nc, h, w = x.shape
            ks = (min(ks[0], h), min(ks[1], w)); stride = (min(stride[0], h), min(stride[1], w))
            fold, unfold, norm, wt = self.get_fold_unfold(x, ks, stride, df=df)
            z = unfold(x).view(x.shape[0], -1, ks[0], ks[1], -1)
            o = torch.stack([self.first_stage_model.encode(z[:, :, :, :, i])
                             for i in range(z.shape[-1])], axis=-1) * wt
            return fold(o.view(o.shape[0], -1, o.shape[-1])) / norm
        return self.first_stage_model.encode(x)

    def _decode_z(self, z, predict_cids=False, force_not_quantize=False):
        is_vq = hasattr(self.first_stage_model, 'quantize') and callable(self.first_stage_model.quantize)
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()
        z = 1. / self.scale_factor * z
        if hasattr(self, 'split_input_params') and self.split_input_params.get('patch_distributed_vq'):
            sp = self.split_input_params
            ks, stride, uf = sp['ks'], sp['stride'], sp['vqf']
            bs, nc, h, w = z.shape
            ks = (min(ks[0], h), min(ks[1], w)); stride = (min(stride[0], h), min(stride[1], w))
            fold, unfold, norm, wt = self.get_fold_unfold(z, ks, stride, uf=uf)
            z = unfold(z).view(z.shape[0], -1, ks[0], ks[1], -1)
            fn = (lambda zi: self.first_stage_model.decode(zi,
                  force_not_quantize=predict_cids or force_not_quantize)) if is_vq \
                 else (lambda zi: self.first_stage_model.decode(zi))
            o  = torch.stack([fn(z[:, :, :, :, i]) for i in range(z.shape[-1])], axis=-1) * wt
            return fold(o.view(o.shape[0], -1, o.shape[-1])) / norm
        if is_vq:
            return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        return self._decode_z(z, predict_cids, force_not_quantize)

    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        return self._decode_z(z, predict_cids, force_not_quantize)

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        rows = [self.decode_first_stage(zd.to(self.device),
                force_not_quantize=force_no_decoder_quantization) for zd in tqdm(samples, desc=desc)]
        n    = len(rows)
        grid = rearrange(torch.stack(rows), 'n b c h w -> (b n) c h w')
        return make_grid(grid, nrow=n)


    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False,
                  force_c_encode=False, cond_key=None,
                  return_original_cond=False, bs=None):
        x = super().get_input(batch, k)
        if bs is not None: x = x[:bs]
        x = x.to(self.device)
        z = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()

        if self.model.conditioning_key is not None:
            cond_key = cond_key or self.cond_stage_key
            if cond_key != self.first_stage_key:
                xc = (batch[cond_key] if cond_key in ('caption', 'coordinates_bbox')
                      else batch if cond_key == 'class_label'
                      else super().get_input(batch, cond_key).to(self.device))
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                c = (self.get_learned_conditioning(xc) if isinstance(xc, (dict, list))
                     else self.get_learned_conditioning(xc.to(self.device)))
            else:
                c = xc
            if bs is not None: c = c[:bs]
        else:
            c = xc = None

        out = [z, c]
        if return_first_stage_outputs:
            out.extend([x, self.decode_first_stage(z)])
        if return_original_cond:
            out.append(xc)
        return out

    def shared_step(self, batch, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        return self(x, c)

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:
                tc = self.cond_ids[t].to(self.device)
                c  = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def training_step(self, batch):
        return self.shared_step(batch)

    @torch.no_grad()
    def validation_step(self, batch):
        _, ld = self.shared_step(batch)
        with self.ema_scope():
            _, ld_ema = self.shared_step(batch)
            ld_ema = {k + '_ema': v for k, v in ld_ema.items()}
        return ld, ld_ema

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if not isinstance(cond, dict):
            cond = [cond] if not isinstance(cond, list) else cond
            key  = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, 'split_input_params'):
            assert len(cond) == 1 and not return_ids
            ks, stride = self.split_input_params['ks'], self.split_input_params['stride']
            fold, unfold, norm, wt = self.get_fold_unfold(x_noisy, ks, stride)
            z = unfold(x_noisy).view(x_noisy.shape[0], -1, ks[0], ks[1], -1)
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]
            if self.cond_stage_key in ('image', 'LR_image', 'segmentation', 'bbox_img') \
                    and self.model.conditioning_key:
                c_key    = next(iter(cond.keys()))
                c_tensor = next(iter(cond.values()))[0]
                c_tensor = unfold(c_tensor).view(c_tensor.shape[0], -1, ks[0], ks[1], -1)
                cond_list = [{c_key: [c_tensor[:, :, :, :, i]]} for i in range(c_tensor.shape[-1])]
            else:
                cond_list = [cond] * z.shape[-1]
            outs = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(outs[0], tuple)
            o = torch.stack(outs, axis=-1) * wt
            x_recon = fold(o.view(o.shape[0], -1, o.shape[-1])) / norm
        else:
            x_recon = self.model(x_noisy, t, **cond)

        return x_recon[0] if isinstance(x_recon, tuple) and not return_ids else x_recon

    def p_losses(self, x_start, cond, t, noise=None):
        noise        = default(noise, lambda: torch.randn_like(x_start))
        x_noisy      = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)
        prefix       = 'train' if self.training else 'val'
        loss_dict    = {}
        target       = x_start if self.parameterization == 'x0' else noise

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict[f'{prefix}/loss_simple'] = loss_simple.mean()

        logvar_t = self.logvar[t.cpu()].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict[f'{prefix}/loss_gamma'] = loss.mean()
            loss_dict['logvar'] = self.logvar.data.mean()
        loss = self.l_simple_weight * loss.mean()

        loss_vlb = (self.lvlb_weights[t] * self.get_loss(model_output, target, mean=False).mean(dim=(1,2,3))).mean()
        loss_dict[f'{prefix}/loss_vlb'] = loss_vlb
        loss += self.original_elbo_weight * loss_vlb
        loss_dict[f'{prefix}/loss'] = loss
        return loss, loss_dict


    def p_mean_variance(self, x, c, t, clip_denoised=False, return_codebook_ids=False,
                        quantize_denoised=False, return_x0=False,
                        score_corrector=None, corrector_kwargs=None):
        model_out = self.apply_model(x, t, c, return_ids=return_codebook_ids)
        if score_corrector is not None:
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)
        if return_codebook_ids:
            model_out, logits = model_out
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out) \
                  if self.parameterization == 'eps' else model_out
        if clip_denoised: x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, _] = self.first_stage_model.quantize(x_recon)
        mm, pv, plv = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids: return mm, pv, plv, logits
        if return_x0:           return mm, pv, plv, x_recon
        return mm, pv, plv

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised, return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:   mm, _, mlv, logits = outputs
        elif return_x0:           mm, _, mlv, x0     = outputs
        else:                     mm, _, mlv          = outputs
        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.: noise = F.dropout(noise, p=noise_dropout)
        mask   = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        sample = mm + mask * (0.5 * mlv).exp() * noise
        return (sample, x0) if return_x0 else sample

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None,
                               quantize_denoised=False, img_callback=None, mask=None, x0=None,
                               temperature=1., noise_dropout=0., score_corrector=None,
                               corrector_kwargs=None, batch_size=None, x_T=None,
                               start_T=None, log_every_t=None):
        log_every_t = log_every_t or self.log_every_t
        timesteps   = self.num_timesteps
        b = batch_size if batch_size is not None else shape[0]
        if batch_size is not None: shape = [batch_size] + list(shape)
        img = x_T if x_T is not None else torch.randn(shape, device=self.device)
        if cond is not None:
            cond = ({k: (v[:b] if not isinstance(v, list) else [e[:b] for e in v])
                     for k, v in cond.items()} if isinstance(cond, dict)
                    else [c[:b] for c in cond] if isinstance(cond, list) else cond[:b])
        if start_T is not None: timesteps = min(timesteps, start_T)
        temp = [temperature] * timesteps if isinstance(temperature, float) else temperature
        it = tqdm(reversed(range(timesteps)), desc='Progressive Generation', total=timesteps) \
             if verbose else reversed(range(timesteps))
        intermediates = []
        for i in it:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                cond = self.q_sample(x_start=cond, t=self.cond_ids[ts].to(cond.device),
                                     noise=torch.randn_like(cond))
            img, x0p = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised,
                                     quantize_denoised=quantize_denoised, return_x0=True,
                                     temperature=temp[i], noise_dropout=noise_dropout,
                                     score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None: img = self.q_sample(x0, ts) * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1: intermediates.append(x0p)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None):
        log_every_t = log_every_t or self.log_every_t
        device = self.betas.device
        b = shape[0]
        img = x_T if x_T is not None else torch.randn(shape, device=device)
        intermediates = [img]
        timesteps = self.num_timesteps if timesteps is None else timesteps
        if start_T is not None: timesteps = min(timesteps, start_T)
        if mask is not None: assert x0 is not None and x0.shape[2:3] == mask.shape[2:3]
        it = tqdm(reversed(range(timesteps)), desc='Sampling t', total=timesteps) \
             if verbose else reversed(range(timesteps))
        for i in it:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                cond = self.q_sample(x_start=cond, t=self.cond_ids[ts].to(cond.device),
                                     noise=torch.randn_like(cond))
            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None: img = self.q_sample(x0, ts) * mask + (1. - mask) * img
            if i % log_every_t == 0 or i == timesteps - 1: intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return (img, intermediates) if return_intermediates else img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None, **kwargs):
        shape = shape or (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            cond = ({k: (v[:batch_size] if not isinstance(v, list) else [e[:batch_size] for e in v])
                     for k, v in cond.items()} if isinstance(cond, dict)
                    else [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size])
        return self.p_sample_loop(cond, shape, return_intermediates=return_intermediates,
                                  x_T=x_T, verbose=verbose, timesteps=timesteps,
                                  quantize_denoised=quantize_denoised, mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim and DDIMSampler is not None:
            sampler = DDIMSampler(self)
            samples, intermediates = sampler.sample(
                ddim_steps, batch_size, (self.channels, self.image_size, self.image_size),
                cond, verbose=False, **kwargs)
        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                  return_intermediates=True, **kwargs)
        return samples, intermediates

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        t = torch.tensor([self.num_timesteps - 1] * x_start.shape[0], device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        return mean_flat(normal_kl(qt_mean, qt_log_variance, 0., 0.)) / np.log(2.0)

    def _rescale_annotations(self, bboxes, crop_coordinates):
        def clamp(v): return min(max(v, 0), 1)
        def rb(bbox):
            x0 = clamp((bbox[0]-crop_coordinates[0])/crop_coordinates[2])
            y0 = clamp((bbox[1]-crop_coordinates[1])/crop_coordinates[3])
            return x0, y0, min(bbox[2]/crop_coordinates[2], 1-x0), min(bbox[3]/crop_coordinates[3], 1-y0)
        return [rb(b) for b in bboxes]

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, 'colorize'):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        return 2. * (x - x.min()) / (x.max() - x.min()) - 1.

    @torch.no_grad()
    def get_epsilon_spent(self, step):
        if self.dp_config is None or not getattr(self.dp_config, 'enabled', False):
            return 0
        if getattr(self.dp_config, 'poisson_sampling', False):
            return self.privacy_engine.get_epsilon(self.dp_config.delta)
        return 0

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1.,
                   return_keys=None, quantize_denoised=True, inpaint=False,
                   plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        use_ddim = ddim_steps is not None
        log = {}
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                             return_first_stage_outputs=True,
                                             force_c_encode=True, return_original_cond=True, bs=N)
        N, n_row = min(x.shape[0], N), min(x.shape[0], n_row)
        log['inputs'] = x; log['reconstruction'] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, 'decode'): log['conditioning'] = self.cond_stage_model.decode(c)
            elif self.cond_stage_key == 'caption': log['conditioning'] = log_txt_as_img((x.shape[2], x.shape[3]), batch['caption'])
            elif self.cond_stage_key == 'class_label': log['conditioning'] = log_txt_as_img((x.shape[2], x.shape[3]), batch['class_label'])
            elif isimage(xc): log['conditioning'] = xc
            if ismap(xc): log['original_conditioning'] = self.to_rgb(xc)
        if plot_diffusion_rows:
            drow, zs = [], z[:n_row]
            for ti in range(self.num_timesteps):
                if ti % self.log_every_t == 0 or ti == self.num_timesteps - 1:
                    tb = repeat(torch.tensor([ti]), '1 -> b', b=n_row).to(self.device).long()
                    drow.append(self.decode_first_stage(self.q_sample(zs, tb, torch.randn_like(zs))))
            dr = torch.stack(drow); dr = rearrange(dr, 'n b c h w -> (b n) c h w')
            log['diffusion_row'] = make_grid(dr, nrow=len(drow))
        if sample:
            with self.ema_scope('Plotting'):
                samples, z_dr = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                 ddim_steps=ddim_steps, eta=ddim_eta)
            log['samples'] = self.decode_first_stage(samples)
            if plot_denoise_rows: log['denoise_row'] = self._get_denoise_row_from_list(z_dr)
            is_vq = hasattr(self.first_stage_model, 'quantize') and callable(self.first_stage_model.quantize)
            if quantize_denoised and is_vq:
                with self.ema_scope('Quantized'):
                    sq, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                            ddim_steps=ddim_steps, eta=ddim_eta, quantize_denoised=True)
                log['samples_x0_quantized'] = self.decode_first_stage(sq.to(self.device))
            if inpaint:
                _, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w, device=self.device)
                mask[:, h//4:3*h//4, w//4:3*w//4] = 0.; mask = mask[:, None, ...]
                with self.ema_scope('Inpaint'):
                    si, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                log['samples_inpainting'] = self.decode_first_stage(si.to(self.device)); log['mask'] = mask
                with self.ema_scope('Outpaint'):
                    so, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim, eta=ddim_eta,
                                            ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                log['samples_outpainting'] = self.decode_first_stage(so.to(self.device))
        if plot_progressive_rows:
            with self.ema_scope('Progressives'):
                img, progs = self.progressive_denoising(
                    c, shape=(self.channels, self.image_size, self.image_size), batch_size=N)
            log['progressive_row'] = self._get_denoise_row_from_list(progs, desc='Progressive Generation')
        if return_keys:
            return {k: log[k] for k in return_keys if k in log} or log
        return log

    def build_optimizer(self, lr=None):
        """Replaces configure_optimizers. Returns AdamW optimizer."""
        print('#### build_optimizer ####')
        lr = lr or self.lr
        att_params = []

        if self.train_condition_only:
            self.model.requires_grad_(False)
            smods = [m for m in self.model.modules() if isinstance(m, SpatialTransformer)]
            for i, m in enumerate(smods):
                m.requires_grad_(True)
                if i + 1 >= self.ablation_blocks: att_params.extend(m.parameters())
            self.cond_stage_model.requires_grad_(True)
            params = att_params + list(self.cond_stage_model.parameters())
            print(f'  {len(smods)-max(self.ablation_blocks,0)}/{len(smods)} SpatialTransformers trained')
        elif self.train_attention_only:
            print('  Training AttentionBlocks only')
            self.model.requires_grad_(False)
            if AttentionBlock is not None:
                amods = [m for m in self.model.modules() if isinstance(m, AttentionBlock)]
                for i, m in enumerate(amods):
                    if i + 1 >= self.ablation_blocks: m.requires_grad_(True); att_params.extend(m.parameters())
                print(f'  {len(amods)-max(self.ablation_blocks,0)}/{len(amods)} AttentionBlocks trained')
            params = att_params
        elif self.train_input_blocks_only:
            print('  Training attention in input_blocks', end='')
            self.model.requires_grad_(True); params = []
            for m in self.model.diffusion_model.input_blocks.modules():
                if (AttentionBlock is not None and isinstance(m, AttentionBlock)) or isinstance(m, SpatialTransformer):
                    params.extend(m.parameters())
            if self.cond_stage_model:
                print(' and cond_stage_model', end='')
                self.cond_stage_model.requires_grad_(True); params.extend(self.cond_stage_model.parameters())
            print()
        elif self.train_resblocks_only:
            print('  Training ResBlocks', end='')
            self.model.requires_grad_(True); params = []
            for m in [m for m in self.model.modules() if isinstance(m, ResBlock)]:
                params.extend(m.parameters())
            if self.cond_stage_model:
                print(' and cond_stage_model', end='')
                self.cond_stage_model.requires_grad_(True); params.extend(self.cond_stage_model.parameters())
            print()
        else:
            self.model.requires_grad_(True)
            if self.ablation_blocks > 0:
                self.model.requires_grad_(False); cur = 0; flag = False; att_params = []
                for nm, m in self.model.named_modules():
                    if isinstance(m, SpatialTransformer):
                        cur += 1
                        if cur == self.ablation_blocks - 1: flag = True; print(f'Begin with Layer {cur+1}')
                    if flag: m.requires_grad_(True); att_params.extend(m.parameters())
                params = att_params
            else:
                params = list(self.model.parameters())
            if self.learn_logvar:
                print('  optimizing logvar'); params.append(self.logvar)

        nm = sum(p.numel() for p in self.model.parameters())
        nc = sum(p.numel() for p in self.cond_stage_model.parameters()) if self.cond_stage_model else 0
        ng = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ng += sum(p.numel() for p in self.cond_stage_model.parameters() if p.requires_grad) if self.cond_stage_model else 0
        nt = sum(p.numel() for p in params); tot = nm + nc
        print(f'  {ng}/{tot} ({ng/tot*100:.2f}%) compute gradients')
        print(f'  {nt}/{tot} ({nt/tot*100:.2f}%) will be trained')
        return torch.optim.AdamW(params, lr=lr)



# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':

    class TinyUNet(nn.Module):
        """Minimal UNet stub matching DiffusionWrapper call interface."""
        def __init__(self, in_channels=4, model_channels=32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, model_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(model_channels, in_channels, 3, padding=1),
            )
        def forward(self, x, t, context=None, **kwargs):
            return self.net(x)

    class DummyVAE(nn.Module):
        """
        Minimal VAE stub.
        encode: [B,C,H,W] -> Tensor [B,4,H/4,W/4]
        decode: [B,4,h,w] -> Tensor [B,1,h*4,w*4]
        (returns plain Tensor; LatentDiffusion handles both
         DiagonalGaussianDistribution and plain Tensor via get_first_stage_encoding)
        """
        def encode(self, x):
            B, C, H, W = x.shape
            return torch.randn(B, 4, H // 4, W // 4, device=x.device)

        def decode(self, z):
            B, C, H, W = z.shape
            return torch.randn(B, 1, H * 4, W * 4, device=z.device)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # image [B,1,64,64] -> encode -> [B,4,16,16], so image_size=16, channels=4
    model = LatentDiffusion(
        first_stage_config = DummyVAE(),
        cond_stage_config  = '__is_unconditional__',
        unet               = TinyUNet(in_channels=4, model_channels=32),
        timesteps          = 100,
        beta_schedule      = 'linear',
        image_size         = 16,
        channels           = 4,
        first_stage_key    = 'image',
        cond_stage_key     = 'image',
        use_ema            = True,
        lr                 = 1e-4,
        num_timesteps_cond = 1,
    ).to(device)

    def make_batch(bs=2):
        return {'image': torch.randn(bs, 1, 64, 64, device=device)}

    optimizer   = model.build_optimizer(lr=1e-4)
    first_batch = make_batch()
    model.init_scale_factor(first_batch, is_first_batch=True)

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

    print('\n--- Validation ---')
    model.eval()
    val_ld, val_ema = model.validation_step(make_batch())
    print('  val :', {k: f'{v.item():.4f}' for k, v in val_ld.items()})
    print('  ema :', {k: f'{v.item():.4f}' for k, v in val_ema.items()})

    print('\nDebug run completed successfully!')
