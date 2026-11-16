"""
Eval_metric/psnr.py  ?  PSNR evaluation for VAE reconstruction
===============================================================

Pipeline
--------
1. Load trained AutoencoderKL weights from checkpoint
2. Run inference on NIH test split (encode ¡æ posterior.mode() ¡æ decode)
3. Compute PSNR between real and reconstructed images per sample
4. Report mean / std / min / max PSNR

PSNR formula:
    MSE  = mean((x - x_recon)©÷)
    PSNR = 10 * log10(data_range©÷ / MSE)

    data_range = 2.0  for images in [-1, 1]

Dependencies: torch, torchvision, numpy  (all standard)

Usage:
    python Eval_metric/psnr.py \\
        --ckpt_path  /path/to/vae_final.pt \\
        --data_path  /storage/hjchoi/archive/image_file \\
        --label_path /storage/hjchoi/archive/Data_Entry_2017.csv \\
        --batch_size 16
"""

import os
import sys
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ¦¡¦¡ Path setup ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.normpath(os.path.join(_this_dir, '..'))
_model_dir = os.path.join(_proj_root, 'Model')
_data_dir  = os.path.join(_proj_root, 'Data')
for _d in [_model_dir, _data_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Model.VAE.Autoencoder import VAE # Model/autoencoder.py
from Data.nih     import NIH             # Data/dataset.py


# =============================================================================
# PSNR  (pure PyTorch)
# =============================================================================

def compute_psnr(img1: torch.Tensor, img2: torch.Tensor,
                 data_range: float = 2.0) -> torch.Tensor:
    """
    Peak Signal-to-Noise Ratio per sample in the batch.

    Args:
        img1, img2   : [B, 1, H, W]  float in [-1, 1]
        data_range   : pixel value range (2.0 for [-1,1], 1.0 for [0,1])

    Returns:
        Tensor [B]  ? PSNR in dB for each image pair
        (returns inf when MSE == 0, i.e. identical images)
    """
    mse = F.mse_loss(img1, img2, reduction='none')  # [B, 1, H, W]
    mse = mse.mean(dim=[1, 2, 3])                   # [B]

    psnr = 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-10))
    return psnr                                      # [B]


# =============================================================================
# VAE helpers  (shared with ssim.py / vae_test.py)
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
        verbose          = False,
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
            print('  [WARNING] >10 % keys missing ? checkpoint may be incompatible')
    else:
        print('  all keys matched perfectly')

    vae.eval()
    return vae


# =============================================================================
# Evaluation loop
# =============================================================================

@torch.no_grad()
def evaluate(vae: nn.Module,
             test_loader: DataLoader,
             device: str,
             save_dir: str = None) -> dict:
    """
    Run VAE reconstruction and compute per-sample PSNR on the test set.

    Args:
        vae         : AutoencoderKL in eval mode
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

    for imgs, _ in tqdm(test_loader, desc='[PSNR eval]'):
        imgs = imgs.to(device)              # [B, 1, H, W]  in [-1, 1]

        # ¦¡¦¡ VAE reconstruction (deterministic) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        posterior = vae.encode(imgs)
        z         = posterior.mode()        # [B, 1, 16, 16]
        recon     = vae.decode(z)           # [B, 1, H, W]

        # ¦¡¦¡ Per-sample PSNR ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
        scores = compute_psnr(imgs, recon, data_range=2.0)   # [-1,1] ¡æ range=2
        all_scores.append(scores.cpu().numpy())

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
    parser = argparse.ArgumentParser(description='VAE PSNR evaluation on NIH test set')
    parser.add_argument('--device_id', default=0)
    parser.add_argument('--ckpt_path', default='/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/vae/2026-03-31_10-58-39/resume/vae_final.pt')
    parser.add_argument('--root_path', default='/storage/hjchoi/archive/DATA')
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--batch_size',  default=16,   type=int)
    parser.add_argument('--num_workers', default=4,    type=int)
    parser.add_argument('--output_dir',  default='/home/hjchoi/PycharmProjects/PrivaText-CXR/Eval_metric/psnr_vae_test',
                        help='Base dir; sub-dir named after ckpt stem is auto-created')
    parser.add_argument('--save_images', default=True, action='store_true',
                        help='Save real and reconstructed images under output_dir/<ckpt_stem>/')
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device_id)
    print(f'Device : {device}')

    # ¦¡¦¡ Test DataLoader ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    nih_args = argparse.Namespace(
        root_path=args.root_path,
        task='test',
        image_size=args.image_size,
        image_show=False,
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

    # ¦¡¦¡ Output directory: output_dir / <ckpt_stem> / ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    ckpt_stem = os.path.splitext(os.path.basename(args.ckpt_path))[0]
    save_dir  = os.path.join(args.output_dir, ckpt_stem) if args.save_images else None
    if save_dir:
        print(f'[save] {os.path.abspath(save_dir)}/')

    # ¦¡¦¡ Run evaluation ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    results = evaluate(vae, test_loader, device, save_dir=save_dir)

    # ¦¡¦¡ Results ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    print()
    print('=' * 45)
    print(f'  Checkpoint  : {ckpt_stem}')
    print(f'  Test samples: {len(test_dataset)}')
    print(f'  PSNR  mean  : {results["mean"]:.4f} dB')
    print(f'  PSNR  std   : {results["std"]:.4f} dB')
    print(f'  PSNR  min   : {results["min"]:.4f} dB')
    print(f'  PSNR  max   : {results["max"]:.4f} dB')
    print('=' * 45)

    if save_dir:
        print(f'\n  Images saved to : {os.path.abspath(save_dir)}/')

    return results


if __name__ == '__main__':
    main()