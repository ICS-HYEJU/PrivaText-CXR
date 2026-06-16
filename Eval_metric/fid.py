# """
# Diffusion/vae_test.py  ?  VAE evaluation on NIH test set + FID score
# =====================================================================
#
# Pipeline
# --------
# 1. Load trained AutoencoderKL weights from checkpoint
# 2. Run inference on NIH test split in eval mode (no gradient)
#        x  ¡æ  encode()  ¡æ  posterior.mode()  ¡æ  decode()  ¡æ  x_recon
#    (mode() = deterministic: uses posterior mean, no sampling noise)
# 3. Compute FID between real and reconstructed images
#    (uses torchmetrics.image.FrechetInceptionDistance with InceptionV3)
# 4. Report Avg MSE + FID
#
# Dependencies:
#     pip install torchmetrics[image]   (for FrechetInceptionDistance)
#
# Usage:
#     python vae_test.py \\
#         --ckpt_path  /path/to/vae_epoch50.pth \\
#         --data_path  /storage/hjchoi/archive/image_file \\
#         --label_path /storage/hjchoi/archive/Data_Entry_2017.csv \\
#         --batch_size 16 \\
#         --save_images                  # optional: save real/recon PNGs
# """
#
# import os
# import sys
# import argparse
#
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from tqdm import tqdm
#
# # ¦¡¦¡ Path setup ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
# _this_dir  = os.path.dirname(os.path.abspath(__file__))
# _model_dir = os.path.normpath(os.path.join(_this_dir, '..', 'Model'))
# _data_dir  = os.path.normpath(os.path.join(_this_dir, '..', 'Data'))
# for _d in [_model_dir, _data_dir, _this_dir]:
#     if _d not in sys.path:
#         sys.path.insert(0, _d)
#
# from Model.VAE.Autoencoder import VAE   # Model/autoencoder.py
# from Data.nih     import NIH             # Data/dataset.py
#
#
# # =============================================================================
# # Helpers
# # =============================================================================
#
# def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
#     """
#     [B, 1, H, W] float in [-1, 1]  ¡æ  [B, 3, H, W] uint8 [0, 255]
#
#     FrechetInceptionDistance (InceptionV3) requires 3-channel uint8 input.
#     Grayscale CXR is converted by repeating the single channel three times.
#     """
#     x = (x.clamp(-1., 1.) + 1.) / 2.   # [-1, 1] ¡æ [0, 1]
#     x = (x * 255.).to(torch.uint8)      # [0, 1]  ¡æ [0, 255]
#     return x.repeat(1, 3, 1, 1)         # [B, 1, H, W] ¡æ [B, 3, H, W]
#
#
# def build_vae(image_size: int) -> VAE:
#     """Construct AutoencoderKL with the same config used during training."""
#     vae_args = argparse.Namespace(
#         in_channels      = 1,
#         out_channels     = 1,
#         ch               = 128,
#         ch_mult          = [1, 2, 4, 4, 4],  # 4 downsamples (¡À16)
#         num_res_blocks   = 2,
#         attn_resolutions = [32, 16],
#         dropout          = 0.0,
#         resamp_with_conv = True,
#         resolution       = image_size,
#         z_channels       = 1,
#         double_z         = True,
#         dims             = 2,
#         verbose          = False
#     )
#     return VAE(vae_args)
#
#
# def load_vae(ckpt_path: str, device: str, image_size: int) -> VAE:
#     """
#     Load AutoencoderKL from a checkpoint.
#
#     Supports two checkpoint formats:
#         - plain state_dict  : torch.save(model.state_dict(), path)
#         - wrapped dict      : torch.save({'state_dict': ..., 'epoch': ...}, path)
#     """
#     vae = build_vae(image_size).to(device)
#     ckpt = torch.load(ckpt_path, map_location=device)
#     state_dict = ckpt.get('state_dict', ckpt)
#     missing, unexpected = vae.load_state_dict(state_dict, strict=False)
#     print(f'[VAE] loaded  {ckpt_path}')
#     if missing:
#         print(f'  missing keys   : {len(missing)}')
#     if unexpected:
#         print(f'  unexpected keys: {len(unexpected)}')
#     vae.eval()
#     return vae
#
#
# # =============================================================================
# # Evaluation
# # =============================================================================
#
# @torch.no_grad()
# def evaluate(vae, test_loader, device, fid_metric, save_dir=None):
#     """
#     Run VAE reconstruction on the test set.
#
#     Args:
#         vae         : AutoencoderKL in eval mode
#         test_loader : DataLoader (NIH, task='test')
#         device      : 'cuda' | 'cpu'
#         fid_metric  : FrechetInceptionDistance instance (already on device)
#         save_dir    : if not None, save real/recon PNGs to save_dir/real & /recon
#
#     Returns:
#         avg_mse : float
#     """
#     if save_dir:
#         os.makedirs(os.path.join(save_dir, 'real'),  exist_ok=True)
#         os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)
#
#     mse_total = 0.
#     n_batches = 0
#     img_idx   = 0
#
#     for imgs, _ in tqdm(test_loader, desc='[VAE test]'):
#         imgs = imgs.to(device)                  # [B, 1, 256, 256]  in [-1, 1]
#
#         # ¦¡¦¡ Test-mode inference: deterministic (no sampling noise) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#         posterior = vae.encode(imgs)            # DiagonalGaussianDistribution
#         z         = posterior.mode()            # [B, 1, 16, 16]  ¡ç mean, not sample
#         recon     = vae.decode(z)               # [B, 1, 256, 256]
#
#         # ¦¡¦¡ Reconstruction quality ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#         mse_total += F.mse_loss(recon, imgs).item()
#         n_batches += 1
#
#         # ¦¡¦¡ FID update  (needs uint8 RGB) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#         fid_metric.update(to_uint8_rgb(imgs),  real=True)
#         fid_metric.update(to_uint8_rgb(recon), real=False)
#
#         # ¦¡¦¡ Optional: save images ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#         if save_dir:
#             from torchvision.utils import save_image
#             for i in range(imgs.shape[0]):
#                 fname = f'{img_idx:05d}.png'
#                 save_image((imgs[i]  + 1.) / 2.,
#                            os.path.join(save_dir, 'real',  fname))
#                 save_image((recon[i] + 1.) / 2.,
#                            os.path.join(save_dir, 'recon', fname))
#                 img_idx += 1
#
#     return mse_total / n_batches
#
#
# # =============================================================================
# # Main
# # =============================================================================
#
# def parse_args():
#     parser = argparse.ArgumentParser(description='VAE test-set evaluation + FID')
#     parser.add_argument('--device_id', default=0)
#     parser.add_argument('--ckpt_path', default ='/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/vae/vae_final.pt',
#                         help='Trained VAE checkpoint (.pth)')
#     parser.add_argument('--root_path', default='/storage/hjchoi/archive/DATA',
#                         help='Root directory of CXR image files')
#     parser.add_argument('--image_size',  default=256,  type=int)
#     parser.add_argument('--batch_size',  default=16,   type=int)
#     parser.add_argument('--num_workers', default=4,    type=int)
#     parser.add_argument('--output_dir',  default='/home/hjchoi/PycharmProjects/PrivaText-CXR/Eval_metric/vae_test_outputs',
#                         help='Base directory; a sub-directory named after the '
#                              'checkpoint stem is created automatically '
#                              '(e.g. output_dir/vae_epoch50/real|recon/)')
#     parser.add_argument('--save_images', action='store_true',
#                         help='Save real and reconstructed images under '
#                              'output_dir/<ckpt_stem>/')
#     return parser.parse_args()
#
#
# def main():
#     args   = parse_args()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.cuda.set_device(args.device_id)
#
#     # ¦¡¦¡ Test DataLoader ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     nih_args = argparse.Namespace(
#         root_path  = args.root_path,
#         task       = 'test',            # ¡ç test split, no augmentation
#         image_size = args.image_size,
#         image_show = True,
#     )
#     test_dataset = NIH(nih_args)
#     test_loader  = DataLoader(
#         test_dataset,
#         batch_size  = args.batch_size,
#         shuffle     = False,            # keep deterministic order for test
#         num_workers = args.num_workers,
#         drop_last   = False,            # evaluate every sample
#         pin_memory  = (device == 'cuda'),
#     )
#     print(f'[NIH]  test samples : {len(test_dataset)}'
#           f'  |  batch_size : {args.batch_size}')
#
#     # ¦¡¦¡ Load VAE (eval mode, frozen) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     vae = load_vae(args.ckpt_path, device, args.image_size)
#     for p in vae.parameters():
#         p.requires_grad = False
#
#     # ¦¡¦¡ FID metric ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     try:
#         from torchmetrics.image.fid import FrechetInceptionDistance
#     except ImportError:
#         raise ImportError(
#             'torchmetrics is required for FID.\n'
#             '  pip install torchmetrics[image]'
#         )
#     fid_metric = FrechetInceptionDistance(
#         feature            = 2048,   # InceptionV3 pool3 features
#         reset_real_features= False,  # keep real features across batches
#     ).to(device)
#
#     # ¦¡¦¡ Output directory: output_dir / <ckpt_stem> / ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
#     save_dir  = os.path.join(args.output_dir, ckpt_stem) if args.save_images else None
#     if save_dir:
#         print(f'[save] {os.path.abspath(save_dir)}/')
#
#     # ¦¡¦¡ Run evaluation ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     avg_mse = evaluate(vae, test_loader, device, fid_metric, save_dir)
#
#     # ¦¡¦¡ Results ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
#     fid_score = fid_metric.compute().item()
#
#     print()
#     print('=' * 45)
#     print(f'  Checkpoint      : {ckpt_stem}')
#     print(f'  Test samples    : {len(test_dataset)}')
#     print(f'  Avg MSE (recon) : {avg_mse:.6f}')
#     print(f'  FID score       : {fid_score:.4f}')
#     print('=' * 45)
#
#     if save_dir:
#         print(f'\n  Images saved to : {os.path.abspath(save_dir)}/')
#         print(f'    real/  : original test images')
#         print(f'    recon/ : VAE reconstructions')
#
#     return {'fid': fid_score, 'mse': avg_mse}
#
#
# if __name__ == '__main__':
#     main()
"""
Diffusion/vae_test.py  ?  VAE evaluation on NIH test set + FID score
=====================================================================

Pipeline
--------
1. Load trained AutoencoderKL weights from checkpoint
2. Run inference on NIH test split in eval mode (no gradient)
       x  ¡æ  encode()  ¡æ  posterior.mode()  ¡æ  decode()  ¡æ  x_recon
   (mode() = deterministic: uses posterior mean, no sampling noise)
3. Compute FID between real and reconstructed images
   - InceptionV3 (torchvision built-in) ¡æ pool3 features [N, 2048]
   - Compute ¥ì, ¥Ò for real and recon sets
   - FID = ||¥ì_r - ¥ì_g||©÷ + Tr(¥Ò_r + ¥Ò_g - 2¡¤sqrt(¥Ò_r¡¤¥Ò_g))
4. Report Avg MSE + FID

Dependencies (no torchmetrics / torch-fidelity needed):
    torch, torchvision, numpy, scipy   ¡ç all standard packages

Usage:
    python vae_test.py \\
        --ckpt_path  /path/to/vae_epoch50.pth \\
        --data_path  /storage/hjchoi/archive/image_file \\
        --label_path /storage/hjchoi/archive/Data_Entry_2017.csv \\
        --batch_size 16 \\
        --save_images          # optional: save real/recon PNGs
"""

import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

# ¦¡¦¡ Path setup ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.normpath(os.path.join(_this_dir, '..', 'Model'))
_data_dir  = os.path.normpath(os.path.join(_this_dir, '..', 'Data'))
for _d in [_model_dir, _data_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Model.VAE.Autoencoder import VAE   # Model/autoencoder.py
from Data.nih     import NIH             # Data/dataset.py


# =============================================================================
# FID  (no external library)
# =============================================================================

class InceptionV3Features(nn.Module):
    """
    Extract 2048-dim pool3 features from a pretrained InceptionV3.

    Input  : [B, 3, H, W]  float32 in [0, 1]  (any spatial size ¡Ã 75)
    Output : [B, 2048]
    """

    def __init__(self):
        super().__init__()
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inc.eval()

        # Build feature extractor up to AdaptiveAvgPool (pool3)
        # Matches the standard FID feature space
        self.layers = nn.Sequential(
            inc.Conv2d_1a_3x3,
            inc.Conv2d_2a_3x3,
            inc.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inc.Conv2d_3b_1x1,
            inc.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inc.Mixed_5b,
            inc.Mixed_5c,
            inc.Mixed_5d,
            inc.Mixed_6a,
            inc.Mixed_6b,
            inc.Mixed_6c,
            inc.Mixed_6d,
            inc.Mixed_6e,
            inc.Mixed_7a,
            inc.Mixed_7b,
            inc.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 3, H, W]  float32 in [0, 1]
        Returns:
            [B, 2048]
        """
        # InceptionV3 expects 299¡¿299
        if x.shape[-2] != 299 or x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299),
                              mode='bilinear', align_corners=False)
        feats = self.layers(x)                  # [B, 2048, 1, 1]
        return feats.view(feats.shape[0], -1)   # [B, 2048]


def _compute_stats(features: np.ndarray):
    """Mean vector and covariance matrix of feature array [N, D]."""
    mu    = np.mean(features, axis=0)           # [D]
    sigma = np.cov(features, rowvar=False)      # [D, D]
    return mu, sigma


def compute_fid(feats_real: np.ndarray, feats_fake: np.ndarray,
                eps: float = 1e-6) -> float:
    """
    Frechet Inception Distance between two feature sets.

    FID = ||¥ì_r - ¥ì_g||©÷ + Tr(¥Ò_r + ¥Ò_g - 2¡¤sqrt(¥Ò_r¡¤¥Ò_g))

    Args:
        feats_real : [N, D]  InceptionV3 features of real images
        feats_fake : [N, D]  InceptionV3 features of generated/reconstructed images
        eps        : small offset added to diagonal for numerical stability
    Returns:
        fid_score : float
    """
    from scipy import linalg

    mu1, sigma1 = _compute_stats(feats_real)
    mu2, sigma2 = _compute_stats(feats_fake)

    diff = mu1 - mu2

    # Matrix square root of sigma1 @ sigma2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    # Numerical guard: if sqrtm fails (non-finite), add eps to diagonal
    if not np.isfinite(covmean).all():
        offset   = np.eye(sigma1.shape[0]) * eps
        covmean  = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))

    # sqrtm may return tiny imaginary parts due to floating-point errors
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-2):
            raise ValueError(
                f'Imaginary component in matrix sqrt: {np.max(np.abs(covmean.imag))}'
            )
        covmean = covmean.real

    fid = float(diff @ diff +
                np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return fid


# =============================================================================
# VAE helpers
# =============================================================================

def to_float_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    [B, 1, H, W] float in [-1, 1]  ¡æ  [B, 3, H, W] float in [0, 1]
    InceptionV3 expects 3-channel float input in [0, 1].
    """
    x = (x.clamp(-1., 1.) + 1.) / 2.   # [-1, 1] ¡æ [0, 1]
    return x.repeat(1, 3, 1, 1)         # 1ch ¡æ 3ch


def build_vae(image_size: int) -> VAE:
    """Construct AutoencoderKL with the same config used during training."""
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
        verbose          = False
    )
    return VAE(vae_args)


def load_vae(ckpt_path: str, device: str, image_size: int) -> VAE:
    """
    Load AutoencoderKL from checkpoint.

    Supports:
        - plain state_dict : torch.save(model.state_dict(), path)
        - wrapped dict     : torch.save({'state_dict': ..., 'epoch': ...}, path)
    """
    vae = build_vae(image_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = vae.load_state_dict(state_dict['model'], strict=False)
    print(f'[VAE] loaded  {ckpt_path}')
    if missing:
        print(f'  missing keys   : {len(missing)}')
    if unexpected:
        print(f'  unexpected keys: {len(unexpected)}')
    vae.eval()
    return vae


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(vae, inception, test_loader, device, save_dir=None):
    """
    Run VAE reconstruction and collect InceptionV3 features for FID.

    Returns:
        avg_mse    : float
        feats_real : np.ndarray [N, 2048]
        feats_recon: np.ndarray [N, 2048]
    """
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'real'),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)

    mse_total    = 0.
    n_batches    = 0
    img_idx      = 0
    all_real     = []
    all_recon    = []

    for imgs, _ in tqdm(test_loader, desc='[VAE test]'):
        imgs = imgs.to(device)                  # [B, 1, 256, 256]  in [-1, 1]

        # ¦¡¦¡ Test-mode inference (deterministic) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        posterior = vae.encode(imgs)            # DiagonalGaussianDistribution
        z         = posterior.mode()            # [B, 1, 16, 16]
        recon     = vae.decode(z)               # [B, 1, 256, 256]

        # ¦¡¦¡ MSE ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        mse_total += F.mse_loss(recon, imgs).item()
        n_batches += 1

        # ¦¡¦¡ InceptionV3 features  (float RGB [0,1]) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        real_rgb  = to_float_rgb(imgs)          # [B, 3, 256, 256]
        recon_rgb = to_float_rgb(recon)         # [B, 3, 256, 256]

        all_real .append(inception(real_rgb) .cpu().numpy())
        all_recon.append(inception(recon_rgb).cpu().numpy())

        # ¦¡¦¡ Optional: save images ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        if save_dir:
            from torchvision.utils import save_image
            for i in range(imgs.shape[0]):
                fname = f'{img_idx:05d}.png'
                save_image((imgs[i]  + 1.) / 2.,
                           os.path.join(save_dir, 'real',  fname))
                save_image((recon[i] + 1.) / 2.,
                           os.path.join(save_dir, 'recon', fname))
                img_idx += 1

    feats_real  = np.concatenate(all_real,  axis=0)  # [N, 2048]
    feats_recon = np.concatenate(all_recon, axis=0)  # [N, 2048]
    avg_mse     = mse_total / n_batches

    return avg_mse, feats_real, feats_recon


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='VAE test-set evaluation + FID')
    parser.add_argument('--device_id', default=0)
    parser.add_argument('--ckpt_path', default ='/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/vae/vae_ep0080.pt',
                        help='Trained VAE checkpoint (.pth)')
    parser.add_argument('--root_path', default='/storage/hjchoi/archive/DATA',
                        help='Root directory of CXR image files')
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--batch_size',  default=16,   type=int)
    parser.add_argument('--num_workers', default=4,    type=int)
    parser.add_argument('--output_dir',  default='/home/hjchoi/PycharmProjects/PrivaText-CXR/Eval_metric/vae_test_outputs',
                        help='Base directory; a sub-directory named after the '
                             'checkpoint stem is created automatically '
                             '(e.g. output_dir/vae_epoch50/real|recon/)')
    parser.add_argument('--save_images', default= False, action='store_true',
                        help='Save real and reconstructed images under '
                             'output_dir/<ckpt_stem>/')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # ¦¡¦¡ Test DataLoader ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    nih_args = argparse.Namespace(
        root_path  = args.root_path,
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

    # ¦¡¦¡ Load VAE ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    vae = load_vae(args.ckpt_path, device, args.image_size)
    for p in vae.parameters():
        p.requires_grad = False

    # ¦¡¦¡ InceptionV3 feature extractor ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    print('[InceptionV3] loading pretrained weights ...')
    inception = InceptionV3Features().to(device)

    # ¦¡¦¡ Output directory: output_dir / <ckpt_stem> / ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    save_dir  = os.path.join(args.output_dir, ckpt_stem) if args.save_images else None
    if save_dir:
        print(f'[save] {os.path.abspath(save_dir)}/')

    # ¦¡¦¡ Run evaluation ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    avg_mse, feats_real, feats_recon = evaluate(
        vae, inception, test_loader, device, save_dir
    )

    # ¦¡¦¡ FID ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    print('[FID] computing ...')
    fid_score = compute_fid(feats_real, feats_recon)

    # ¦¡¦¡ Results ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    print()
    print('=' * 45)
    print(f'  Checkpoint      : {ckpt_stem}')
    print(f'  Test samples    : {len(test_dataset)}')
    print(f'  Avg MSE (recon) : {avg_mse:.6f}')
    print(f'  FID score       : {fid_score:.4f}')
    print('=' * 45)

    if save_dir:
        print(f'\n  Images saved to : {os.path.abspath(save_dir)}/')
        print(f'    real/  : original test images')
        print(f'    recon/ : VAE reconstructions')

    return {'fid': fid_score, 'mse': avg_mse}


if __name__ == '__main__':
    main()