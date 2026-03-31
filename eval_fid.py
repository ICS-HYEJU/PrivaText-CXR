"""
eval_fid.py  –  FID evaluation for trained VAE
================================================

Usage:
    python eval_fid.py --checkpoint ./checkpoints/vae/vae_final.pt \
                       --root_path  /storage/hjchoi/archive/DATA  \
                       --n_samples  2048 \
                       --bs         16

FID 계산 방식:
    - 저장된 그리드 이미지에서 원본을 분리하지 않습니다.
    - Checkpoint에서 모델을 로드하고 DataLoader를 통해 직접 추론합니다.
    - 실제 이미지와 재구성 이미지를 Inception v3 feature space에서 비교합니다.
    - 그레이스케일(1ch) → RGB(3ch) 변환 후 Inception에 입력합니다.

Dependencies:
    pip install torchmetrics[image]
    (or: pip install torch-fidelity)
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model.autoencoder import AutoencoderKL
from Data.dataset import NIH


# ──────────────────────────────────────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VAE FID Evaluation")

    # ── Required ──────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to .pt checkpoint (e.g. ./checkpoints/vae/vae_final.pt)")
    parser.add_argument("--root_path",  default="/storage/hjchoi/archive/DATA",
                        help="Root data directory (same as training)")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--task",       default="test", choices=["train", "val", "test"],
                        help="Which split to evaluate on")
    parser.add_argument("--bs",         default=16,  type=int,  help="Batch size")
    parser.add_argument("--image_size", default=256, type=int)
    parser.add_argument("--n_samples",  default=2048, type=int,
                        help="Max number of images to use for FID (0 = full split)")
    parser.add_argument("--num_workers", default=4,  type=int)

    # ── Model (must match checkpoint) ─────────────────────────────────────────
    parser.add_argument("--in_channels",      default=1,   type=int)
    parser.add_argument("--ch",               default=128, type=int)
    parser.add_argument("--ch_mult",          default=[1, 2, 4, 4, 4])
    parser.add_argument("--num_res_blocks",   default=2,   type=int)
    parser.add_argument("--attn_resolutions", default=[32, 16])
    parser.add_argument("--dropout",          default=0.0, type=float)
    parser.add_argument("--resamp_with_conv", default=True, type=bool)
    parser.add_argument("--resolution",       default=256, type=int)
    parser.add_argument("--z_channels",       default=1,   type=int)
    parser.add_argument("--double_z",         default=True, type=bool)
    parser.add_argument("--dims",             default=2,   type=int)
    parser.add_argument("--out_channels",     default=1,   type=int)
    parser.add_argument("--verbose",          default=False, type=bool)
    parser.add_argument("--test_case",        default=False, type=bool)
    parser.add_argument("--image_show",       default=False, type=bool)

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def gray_to_uint8_rgb(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert grayscale tensor [-1, 1] → uint8 RGB [0, 255] for Inception.

    Args:
        tensor: (B, 1, H, W) float in [-1, 1]

    Returns:
        (B, 3, H, W) uint8 in [0, 255]
    """
    # [-1, 1] → [0, 1]
    t = (tensor.clamp(-1.0, 1.0) + 1.0) / 2.0
    # (B, 1, H, W) → (B, 3, H, W)
    t = t.repeat(1, 3, 1, 1)
    # [0, 1] → [0, 255] uint8
    return (t * 255).to(torch.uint8)


def resize_for_inception(tensor: torch.Tensor, size: int = 299) -> torch.Tensor:
    """Resize to 299×299 for Inception v3 if needed."""
    if tensor.shape[-1] != size:
        tensor = F.interpolate(tensor.float(), size=(size, size),
                               mode="bilinear", align_corners=False)
        tensor = tensor.to(torch.uint8)
    return tensor


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FID Eval] device={device}  checkpoint={args.checkpoint}")

    # ── torchmetrics FID ──────────────────────────────────────────────────────
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError:
        raise ImportError(
            "torchmetrics[image] is required.\n"
            "  pip install torchmetrics[image]"
        )

    fid_metric = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)

    # If the checkpoint stored its own args, use those (safer than CLI defaults).
    if "args" in ckpt:
        saved = ckpt["args"]
        for k, v in saved.items():
            if not hasattr(args, k):
                setattr(args, k, v)

    model = AutoencoderKL(args).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[FID Eval] Model loaded from epoch {ckpt.get('epoch', '?')}")

    # ── DataLoader ────────────────────────────────────────────────────────────
    args.data_path  = os.path.join(args.root_path, "images")
    args.label_path = os.path.join(args.root_path, "Data_Entry_2017.csv")

    dataset = NIH(args)
    total   = len(dataset) if args.n_samples == 0 else min(args.n_samples, len(dataset))

    # Take a fixed subset without shuffling for reproducibility.
    subset  = torch.utils.data.Subset(dataset, range(total))
    loader  = DataLoader(subset, batch_size=args.bs, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)
    print(f"[FID Eval] Evaluating on {total} images  (split={args.task})")

    # ── Inference loop ────────────────────────────────────────────────────────
    n_processed = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)                   # (B, 1, H, W), [-1, 1]

            posterior = model.encode(x)
            z         = posterior.mode()        # deterministic: use mean, not sample
            x_hat     = model.decode(z)         # (B, 1, H, W), [-1, 1]

            # Convert to uint8 RGB for Inception
            real_rgb  = resize_for_inception(gray_to_uint8_rgb(x)).to(device)
            fake_rgb  = resize_for_inception(gray_to_uint8_rgb(x_hat)).to(device)

            fid_metric.update(real_rgb, real=True)
            fid_metric.update(fake_rgb, real=False)

            n_processed += x.size(0)
            print(f"  processed {n_processed}/{total}", end="\r")

    print()

    # ── Compute & report ──────────────────────────────────────────────────────
    fid_score = fid_metric.compute().item()
    print(f"\n{'='*50}")
    print(f"  FID Score : {fid_score:.4f}")
    print(f"  Samples   : {n_processed}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*50}\n")

    return fid_score


if __name__ == "__main__":
    main()
