"""
Diffusion/vae_test.py  –  VAE evaluation on NIH test set + FID score
=====================================================================

Pipeline
--------
1. Load trained AutoencoderKL weights from checkpoint
2. Run inference on NIH test split in eval mode (no gradient)
       x  →  encode()  →  posterior.mode()  →  decode()  →  x_recon
   (mode() = deterministic: uses posterior mean, no sampling noise)
3. Compute FID between real and reconstructed images
   (uses torchmetrics.image.FrechetInceptionDistance with InceptionV3)
4. Report Avg MSE + FID

Dependencies:
    pip install torchmetrics[image]   (for FrechetInceptionDistance)

Usage:
    python vae_test.py \\
        --ckpt_path  /path/to/vae_epoch50.pth \\
        --data_path  /storage/hjchoi/archive/image_file \\
        --label_path /storage/hjchoi/archive/Data_Entry_2017.csv \\
        --batch_size 16 \\
        --save_images                  # optional: save real/recon PNGs
"""

import os
import sys
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.normpath(os.path.join(_this_dir, '..', 'Model'))
_data_dir  = os.path.normpath(os.path.join(_this_dir, '..', 'Data'))
for _d in [_model_dir, _data_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from autoencoder import AutoencoderKL   # Model/autoencoder.py
from dataset     import NIH             # Data/dataset.py


# =============================================================================
# Helpers
# =============================================================================

def to_uint8_rgb(x: torch.Tensor) -> torch.Tensor:
    """
    [B, 1, H, W] float in [-1, 1]  →  [B, 3, H, W] uint8 [0, 255]

    FrechetInceptionDistance (InceptionV3) requires 3-channel uint8 input.
    Grayscale CXR is converted by repeating the single channel three times.
    """
    x = (x.clamp(-1., 1.) + 1.) / 2.   # [-1, 1] → [0, 1]
    x = (x * 255.).to(torch.uint8)      # [0, 1]  → [0, 255]
    return x.repeat(1, 3, 1, 1)         # [B, 1, H, W] → [B, 3, H, W]


def build_vae(image_size: int) -> AutoencoderKL:
    """Construct AutoencoderKL with the same config used during training."""
    vae_args = argparse.Namespace(
        in_channels      = 1,
        out_channels     = 1,
        ch               = 128,
        ch_mult          = [1, 2, 4, 4, 4],  # 4 downsamples (÷16)
        num_res_blocks   = 2,
        attn_resolutions = [32, 16],
        dropout          = 0.0,
        resamp_with_conv = True,
        resolution       = image_size,
        z_channels       = 1,
        double_z         = True,
        dims             = 2,
    )
    return AutoencoderKL(vae_args)


def load_vae(ckpt_path: str, device: str, image_size: int) -> AutoencoderKL:
    """
    Load AutoencoderKL from a checkpoint.

    Supports two checkpoint formats:
        - plain state_dict  : torch.save(model.state_dict(), path)
        - wrapped dict      : torch.save({'state_dict': ..., 'epoch': ...}, path)
    """
    vae = build_vae(image_size).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = vae.load_state_dict(state_dict, strict=False)
    print(f'[VAE] loaded  {ckpt_path}')
    if missing:
        print(f'  missing keys   : {len(missing)}')
    if unexpected:
        print(f'  unexpected keys: {len(unexpected)}')
    vae.eval()
    return vae


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(vae, test_loader, device, fid_metric, save_dir=None):
    """
    Run VAE reconstruction on the test set.

    Args:
        vae         : AutoencoderKL in eval mode
        test_loader : DataLoader (NIH, task='test')
        device      : 'cuda' | 'cpu'
        fid_metric  : FrechetInceptionDistance instance (already on device)
        save_dir    : if not None, save real/recon PNGs to save_dir/real & /recon

    Returns:
        avg_mse : float
    """
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'real'),  exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'recon'), exist_ok=True)

    mse_total = 0.
    n_batches = 0
    img_idx   = 0

    for imgs, _ in tqdm(test_loader, desc='[VAE test]'):
        imgs = imgs.to(device)                  # [B, 1, 256, 256]  in [-1, 1]

        # ── Test-mode inference: deterministic (no sampling noise) ────────────
        posterior = vae.encode(imgs)            # DiagonalGaussianDistribution
        z         = posterior.mode()            # [B, 1, 16, 16]  ← mean, not sample
        recon     = vae.decode(z)               # [B, 1, 256, 256]

        # ── Reconstruction quality ────────────────────────────────────────────
        mse_total += F.mse_loss(recon, imgs).item()
        n_batches += 1

        # ── FID update  (needs uint8 RGB) ─────────────────────────────────────
        fid_metric.update(to_uint8_rgb(imgs),  real=True)
        fid_metric.update(to_uint8_rgb(recon), real=False)

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

    return mse_total / n_batches


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='VAE test-set evaluation + FID')
    parser.add_argument('--ckpt_path',   required=True,
                        help='Trained VAE checkpoint (.pth)')
    parser.add_argument('--data_path',   default='/storage/hjchoi/archive/image_file',
                        help='Root directory of CXR image files')
    parser.add_argument('--label_path',  default='/storage/hjchoi/archive/Data_Entry_2017.csv',
                        help='Path to NIH Data_Entry_2017.csv')
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--batch_size',  default=16,   type=int)
    parser.add_argument('--num_workers', default=4,    type=int)
    parser.add_argument('--output_dir',  default='./vae_test_outputs',
                        help='Root directory for saved images (used with --save_images)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save real and reconstructed images to output_dir')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device : {device}')

    # ── Test DataLoader ───────────────────────────────────────────────────────
    nih_args = argparse.Namespace(
        data_path  = args.data_path,
        label_path = args.label_path,
        task       = 'test',            # ← test split, no augmentation
        image_size = args.image_size,
        image_show = False,
    )
    test_dataset = NIH(nih_args)
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,            # keep deterministic order for test
        num_workers = args.num_workers,
        drop_last   = False,            # evaluate every sample
        pin_memory  = (device == 'cuda'),
    )
    print(f'[NIH]  test samples : {len(test_dataset)}'
          f'  |  batch_size : {args.batch_size}')

    # ── Load VAE (eval mode, frozen) ──────────────────────────────────────────
    vae = load_vae(args.ckpt_path, device, args.image_size)
    for p in vae.parameters():
        p.requires_grad = False

    # ── FID metric ────────────────────────────────────────────────────────────
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            'torchmetrics is required for FID.\n'
            '  pip install torchmetrics[image]'
        )
    fid_metric = FrechetInceptionDistance(
        feature            = 2048,   # InceptionV3 pool3 features
        reset_real_features= False,  # keep real features across batches
    ).to(device)

    # ── Run evaluation ────────────────────────────────────────────────────────
    save_dir = args.output_dir if args.save_images else None
    avg_mse  = evaluate(vae, test_loader, device, fid_metric, save_dir)

    # ── Results ───────────────────────────────────────────────────────────────
    fid_score = fid_metric.compute().item()

    print()
    print('=' * 45)
    print(f'  Test samples  : {len(test_dataset)}')
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
