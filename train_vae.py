"""
VAE Training Script
===================
Trains AutoencoderKL using:
    L_total = λ_rec * L_rec  +  λ_ssim * L_ssim  +  λ_kl * L_kl  +  λ_mmd * L_mmd

Usage
-----
    python train_vae.py \
        --data_path  /path/to/images \
        --label_path /path/to/Data_Entry_2017.csv \
        --n_epochs   100 \
        --batch_size 8
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model.autoencoder import AutoencoderKL
from Loss.vae_loss import VAELoss
from Data.dataset import NIH


# ──────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Training")

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_path",   default="/storage/hjchoi/archive/image_file")
    parser.add_argument("--label_path",  default="/storage/hjchoi/archive/Data_Entry_2017.csv")
    parser.add_argument("--image_size",  default=256,   type=int)
    parser.add_argument("--image_show",  default=False, type=bool)
    parser.add_argument("--num_workers", default=4,     type=int)

    # ── VAE model ─────────────────────────────────────────────────────────────
    parser.add_argument("--in_channels",      default=1,   type=int)
    parser.add_argument("--out_channels",     default=1,   type=int)
    parser.add_argument("--ch",               default=128, type=int)
    parser.add_argument("--ch_mult",          default=[1, 2, 4, 4, 4])
    parser.add_argument("--num_res_blocks",   default=2,   type=int)
    parser.add_argument("--attn_resolutions", default=[32, 16])
    parser.add_argument("--dropout",          default=0.0, type=float)
    parser.add_argument("--resamp_with_conv", default=True, type=bool)
    parser.add_argument("--resolution",       default=256, type=int)
    parser.add_argument("--z_channels",       default=3,   type=int)
    parser.add_argument("--double_z",         default=True, type=bool)
    parser.add_argument("--dims",             default=2,   type=int)

    # ── Loss weights ──────────────────────────────────────────────────────────
    parser.add_argument("--lambda_rec",  default=1.0,  type=float,
                        help="Weight for L1 reconstruction loss")
    parser.add_argument("--lambda_ssim", default=1.0,  type=float,
                        help="Weight for SSIM loss")
    parser.add_argument("--lambda_kl",   default=1e-4, type=float,
                        help="Weight for KL divergence loss")
    parser.add_argument("--lambda_mmd",  default=1e-3, type=float,
                        help="Weight for MMD loss")
    parser.add_argument("--mmd_sigma",   default=1.0,  type=float,
                        help="Bandwidth σ_k for Gaussian MMD kernel")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    parser.add_argument("--lr",           default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size",   default=8,    type=int)
    parser.add_argument("--n_epochs",     default=100,  type=int)

    # ── Checkpoints ───────────────────────────────────────────────────────────
    parser.add_argument("--save_dir",   default="./checkpoints/vae")
    parser.add_argument("--save_every", default=10,   type=int,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume",     default=None, type=str,
                        help="Path to checkpoint to resume from")

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--log_every",  default=50, type=int,
                        help="Print step log every N steps")

    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Training loop (one epoch)
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, log_every):
    model.train()

    running = {k: 0.0 for k in ("loss_total", "loss_rec", "loss_ssim", "loss_kl", "loss_mmd")}

    for step, (x, _) in enumerate(loader):
        x = x.to(device)

        # ── Forward ───────────────────────────────────────────────────────────
        posterior = model.encode(x)           # DiagonalGaussianDistribution
        z         = posterior.sample()        # reparameterisation trick
        x_hat     = model.decode(z)           # reconstructed image

        # ── Loss ──────────────────────────────────────────────────────────────
        loss, loss_dict = criterion(x, x_hat, posterior, z)

        # ── Backward ──────────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Accumulate for epoch average ──────────────────────────────────────
        for k in running:
            running[k] += loss_dict[k]

        # ── Step-level log ────────────────────────────────────────────────────
        if (step + 1) % log_every == 0:
            print(
                f"  [Epoch {epoch:4d} | Step {step+1:5d}/{len(loader)}]"
                f"  total={loss_dict['loss_total']:.4f}"
                f"  rec={loss_dict['loss_rec']:.4f}"
                f"  ssim={loss_dict['loss_ssim']:.4f}"
                f"  kl={loss_dict['loss_kl']:.6f}"
                f"  mmd={loss_dict['loss_mmd']:.6f}"
            )

    n = len(loader)
    return {k: v / n for k, v in running.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────────────────
    args.task = "train"
    dataset = NIH(args)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoencoderKL(args).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = VAELoss(
        lambda_rec  = args.lambda_rec,
        lambda_ssim = args.lambda_ssim,
        lambda_kl   = args.lambda_kl,
        lambda_mmd  = args.lambda_mmd,
        mmd_sigma   = args.mmd_sigma,
    )

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Loaded checkpoint from epoch {ckpt['epoch']}")

    print(
        f"\n[Config]"
        f"  device={device}"
        f"  dataset={len(dataset)}"
        f"  batch={args.batch_size}"
        f"  epochs={args.n_epochs}"
        f"  lr={args.lr}"
        f"\n  λ_rec={args.lambda_rec}"
        f"  λ_ssim={args.lambda_ssim}"
        f"  λ_kl={args.lambda_kl}"
        f"  λ_mmd={args.lambda_mmd}\n"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.n_epochs + 1):

        avg = train_one_epoch(
            model, loader, criterion, optimizer, device,
            epoch=epoch, log_every=args.log_every,
        )

        print(
            f"[Epoch {epoch:4d}/{args.n_epochs}]"
            f"  total={avg['loss_total']:.4f}"
            f"  rec={avg['loss_rec']:.4f}"
            f"  ssim={avg['loss_ssim']:.4f}"
            f"  kl={avg['loss_kl']:.6f}"
            f"  mmd={avg['loss_mmd']:.6f}"
        )

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"vae_epoch{epoch:04d}.pt")
            torch.save({
                "epoch"    : epoch,
                "model"    : model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args"     : vars(args),
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, "vae_final.pt")
    torch.save({
        "epoch": args.n_epochs,
        "model": model.state_dict(),
        "args" : vars(args),
    }, final_path)
    print(f"\nTraining complete. Final model: {final_path}")


if __name__ == "__main__":
    main()
