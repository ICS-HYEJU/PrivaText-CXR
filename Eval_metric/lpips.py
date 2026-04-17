"""
Eval_metric/lpips.py  –  LPIPS evaluation for VAE reconstruction
=================================================================

Pipeline
--------
1. Load trained VAE weights from checkpoint
2. Run inference on NIH test split (encode → posterior.mode() → decode)
3. Compute LPIPS between real and reconstructed images per sample
4. Report mean / std / min / max LPIPS

LPIPS is implemented using VGG16 multi-scale perceptual features
(no taming / external metric library required).

For grayscale images [B, 1, H, W], the single channel is replicated
to 3 channels before passing through VGG16.

Usage:
    python Eval_metric/lpips.py \\
        --ckpt_path  /path/to/vae_final.pt \\
        --root_path  /storage/hjchoi/archive \\
        --batch_size 16
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.normpath(os.path.join(_this_dir, '..'))
_model_dir = os.path.join(_proj_root, 'Model')
_data_dir  = os.path.join(_proj_root, 'Data')
for _d in [_model_dir, _data_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from autoencoder import AutoencoderKL as VAE   # Model/autoencoder.py
from dataset     import NIH                     # Data/dataset.py


# =============================================================================
# VGG16 perceptual feature extractor
# =============================================================================

class _VGG16Features(nn.Module):
    """
    Extract intermediate VGG16 feature maps at 5 scales.
    Output channels per scale: [64, 128, 256, 512, 512]
    """

    _SLICES = [
        (0,  4),   # relu1_2
        (4,  9),   # relu2_2
        (9,  16),  # relu3_3
        (16, 23),  # relu4_3
        (23, 30),  # relu5_3
    ]

    def __init__(self):
        super().__init__()
        feats = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleList([
            nn.Sequential(*list(feats.children())[s:e])
            for s, e in self._SLICES
        ])
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        """x : [B, 3, H, W]  ImageNet-normalised"""
        outs = []
        h = x
        for s in self.slices:
            h = s(h)
            outs.append(h)
        return outs  # list of 5 tensors


class _ScalingLayer(nn.Module):
    """ImageNet mean/std normalisation (applied to images in [-1, 1])."""

    def __init__(self):
        super().__init__()
        # ImageNet stats shifted to [-1, 1] input range
        self.register_buffer(
            'shift',
            torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            'scale',
            torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


def _normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """L2-normalise along the channel dimension."""
    norm = x.norm(dim=1, keepdim=True).clamp(min=eps)
    return x / norm


class LPIPSMetric(nn.Module):
    """
    VGG16-based perceptual distance (LPIPS-style, no learned linear layers).

    Uses L2 distance of channel-normalised feature maps, averaged spatially,
    then summed across 5 VGG scales.

    For single-channel inputs the channel is replicated to 3 before VGG16.
    """

    def __init__(self):
        super().__init__()
        self.scaling = _ScalingLayer()
        self.vgg     = _VGG16Features()

    def forward(self, img1: torch.Tensor,
                img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2 : [B, 1, H, W]  float in [-1, 1]

        Returns:
            Tensor [B]  – per-sample LPIPS score
        """
        # Replicate grayscale → 3-channel
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)

        # ImageNet normalisation
        x1 = self.scaling(img1)
        x2 = self.scaling(img2)

        feats1 = self.vgg(x1)
        feats2 = self.vgg(x2)

        dist = torch.zeros(img1.shape[0], device=img1.device)
        for f1, f2 in zip(feats1, feats2):
            n1   = _normalize_tensor(f1)
            n2   = _normalize_tensor(f2)
            diff = (n1 - n2) ** 2
            # mean over spatial dims (H, W) and channels → [B]
            dist = dist + diff.mean(dim=[1, 2, 3])

        return dist   # [B]


# =============================================================================
# VAE helpers  (shared with ssim.py / psnr.py)
# =============================================================================

def _extract_state_dict(raw: dict) -> dict:
    if not isinstance(raw, dict):
        return raw
    for key in ('state_dict', 'model', 'model_state_dict', 'net', 'weights'):
        if key in raw:
            print(f'  [ckpt] using raw["{key}"] as state_dict')
            return raw[key]
    return raw


def _auto_strip_prefix(state_dict: dict, model_state_dict: dict) -> dict:
    ckpt_keys  = list(state_dict.keys())
    model_keys = list(model_state_dict.keys())
    for ck in ckpt_keys:
        for mk in model_keys[:5]:
            if ck.endswith(mk):
                prefix = ck[: len(ck) - len(mk)]
                if not prefix:
                    break
                stripped = {k[len(prefix):]: v
                            for k, v in state_dict.items()
                            if k.startswith(prefix)}
                if len(stripped) >= len(model_keys) * 0.8:
                    print(f'  [ckpt] auto-stripped prefix "{prefix}"')
                    return stripped
    return state_dict


def build_vae(image_size: int) -> VAE:
    vae_args = argparse.Namespace(
        in_channels      = 1,
        out_channels     = 1,
        ch               = 128,
        ch_mult          = [1, 2, 4, 4, 4],
        num_res_blocks   = 2,
        attn_resolutions = [32, 16],
        dropout          = 0.0,
        resamp_with_conv = True,
        resolution       = image_size,
        z_channels       = 1,
        double_z         = True,
        dims             = 2,
    )
    return VAE(vae_args)


def load_vae(ckpt_path: str, device: str, image_size: int) -> VAE:
    vae        = build_vae(image_size).to(device)
    raw        = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(raw)

    ckpt_sample  = list(state_dict.keys())[:3]
    model_sample = list(vae.state_dict().keys())[:3]
    print(f'[VAE] checkpoint  : {ckpt_path}')
    print(f'  ckpt  keys (first 3): {ckpt_sample}')
    print(f'  model keys (first 3): {model_sample}')

    state_dict = _auto_strip_prefix(state_dict, vae.state_dict())
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        print(f'  missing keys   : {len(missing)}')
        print(f'  unexpected keys: {len(unexpected)}')
        if missing:
            print(f'    (first 3 missing)    {missing[:3]}')
        if unexpected:
            print(f'    (first 3 unexpected) {unexpected[:3]}')
        if len(missing) > len(vae.state_dict()) * 0.1:
            print('  [WARNING] >10 % keys missing – checkpoint may be incompatible')
    else:
        print('  all keys matched perfectly')

    vae.eval()
    return vae


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(vae: nn.Module,
             lpips_fn: LPIPSMetric,
             test_loader: DataLoader,
             device: str,
             save_dir: str = None) -> dict:
    """
    Run VAE reconstruction and compute per-sample LPIPS on the test set.

    Args:
        vae         : VAE in eval mode
        lpips_fn    : LPIPSMetric in eval mode
        test_loader : DataLoader  (NIH, task='test', shuffle=False)
        device      : 'cuda' | 'cpu'
        save_dir    : if set, saves real/ and recon/ PNGs

    Returns:
        dict with keys: mean, std, min, max, scores (np.ndarray [N])
    """
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'real'),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)

    all_scores = []
    img_idx    = 0

    for imgs, _ in tqdm(test_loader, desc='[LPIPS eval]'):
        imgs = imgs.to(device)              # [B, 1, H, W]  in [-1, 1]

        # ── VAE reconstruction (deterministic) ───────────────────────────────
        posterior = vae.encode(imgs)
        z         = posterior.mode()        # [B, 1, 16, 16]
        recon     = vae.decode(z)           # [B, 1, H, W]

        # ── Per-sample LPIPS ──────────────────────────────────────────────────
        scores = lpips_fn(imgs, recon)      # [B]
        all_scores.append(scores.cpu().numpy())

        # ── Optional: save images ─────────────────────────────────────────────
        if save_dir:
            from torchvision.utils import save_image
            for i in range(imgs.shape[0]):
                fname = f'{img_idx:05d}.png'
                save_image((imgs[i]  + 1.) / 2.,
                           os.path.join(save_dir, 'real',  fname))
                save_image((recon[i] + 1.) / 2.,
                           os.path.join(save_dir, 'recon', fname))
                img_idx += 1

    scores_all = np.concatenate(all_scores)   # [N]

    return {
        'mean'  : float(scores_all.mean()),
        'std'   : float(scores_all.std()),
        'min'   : float(scores_all.min()),
        'max'   : float(scores_all.max()),
        'scores': scores_all,
    }


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='VAE LPIPS evaluation on NIH test set')
    parser.add_argument('--ckpt_path',   required=True,
                        help='Trained VAE checkpoint (.pt / .pth)')
    parser.add_argument('--root_path',   default='/storage/hjchoi/archive',
                        help='Root directory; images/ and Data_Entry_2017.csv expected inside')
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--batch_size',  default=16,   type=int)
    parser.add_argument('--num_workers', default=4,    type=int)
    parser.add_argument('--output_dir',  default='./eval_outputs',
                        help='Base dir; sub-dir named after ckpt stem is auto-created')
    parser.add_argument('--save_images', action='store_true',
                        help='Save real and reconstructed images under output_dir/<ckpt_stem>/')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # ── Test DataLoader ───────────────────────────────────────────────────────
    nih_args = argparse.Namespace(
        root_path  = args.root_path,
        data_path  = os.path.join(args.root_path, 'images'),
        label_path = os.path.join(args.root_path, 'Data_Entry_2017.csv'),
        task       = 'test',
        image_size = args.image_size,
        image_show = False,
    )
    test_dataset = NIH(nih_args)
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        drop_last   = False,
        pin_memory  = (device == 'cuda'),
    )
    print(f'[NIH]  test samples : {len(test_dataset)}'
          f'  |  batch_size : {args.batch_size}')

    # ── Load VAE ──────────────────────────────────────────────────────────────
    vae = load_vae(args.ckpt_path, device, args.image_size)
    for p in vae.parameters():
        p.requires_grad = False

    # ── LPIPS metric ──────────────────────────────────────────────────────────
    lpips_fn = LPIPSMetric().to(device)
    lpips_fn.eval()
    print('[LPIPS] VGG16 perceptual metric loaded (no taming dependency)')

    # ── Output directory: output_dir / <ckpt_stem> / ─────────────────────────
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    save_dir  = os.path.join(args.output_dir, ckpt_stem) if args.save_images else None
    if save_dir:
        print(f'[save] {os.path.abspath(save_dir)}/')

    # ── Run evaluation ────────────────────────────────────────────────────────
    results = evaluate(vae, lpips_fn, test_loader, device, save_dir=save_dir)

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print('=' * 45)
    print(f'  Checkpoint  : {ckpt_stem}')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  LPIPS mean  : {results["mean"]:.6f}')
    print(f'  LPIPS std   : {results["std"]:.6f}')
    print(f'  LPIPS min   : {results["min"]:.6f}')
    print(f'  LPIPS max   : {results["max"]:.6f}')
    print('=' * 45)

    if save_dir:
        print(f'\n  Images saved to : {os.path.abspath(save_dir)}/')

    return results


if __name__ == '__main__':
    main()
