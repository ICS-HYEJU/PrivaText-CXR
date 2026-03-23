"""
train_vae.py  ?  VAE Training Entry Point
==========================================

Normal training:
    python train_vae.py --root_path /storage/hjchoi/archive/DATA

Debug without real data (random tensors):
    python train_vae.py --test_case True --debug True --n_epochs 2

Resume from checkpoint:
    python train_vae.py --resume ./checkpoints/vae/vae_ep0010.pt
"""

import argparse
import os
import sys

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Model.Autoencoder import *
from Loss.vae_loss import VAELoss
from Data.dataset import NIH


# ===========================================================================================================================
# Args
# ===========================================================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="VAE Training")

    # Device
    parser.add_argument("--device_id", type=int, default=1)
    # Date
    from datetime import datetime
    parser.add_argument("--run_date",default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    # ------------------------------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--root_path", default="/storage/hjchoi/archive/DATA")
    parser.add_argument("--task", default="train", choices=["train", "val", "test"])
    parser.add_argument("--bs", default=2, type=int, help="batch size")
    parser.add_argument("--image_size", default=256, type=int, help="resize image H,W")
    parser.add_argument("--image_show", default=True, type=bool)

    # ------------------------------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--in_channels", default=1, type=int, help='Number of input img channels, NIH=gray-scale')
    parser.add_argument("--ch", default=128, type=int, help='Base channel')
    parser.add_argument("--ch_mult", default=[1, 2, 4, 4, 4], help='Channel multipliers per each level')
    parser.add_argument("--num_res_blocks", default=2, type=int, help='Number of residual blocks per each level')
    parser.add_argument("--attn_resolutions", default=[32, 16], help='the resolution at which attention is applied')
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--resamp_with_conv", default=True, type=bool,
                        help='Use strided conv for downsampling; False uses avg-pool')
    parser.add_argument("--resolution", default=256, type=int, help='Input spatial resolution (H = W)')
    parser.add_argument("--z_channels", default=1, type=int, help='Latent z-space channel dim')
    parser.add_argument("--double_z", default=True, type=bool,
                        help='Output 2*z_channels (mean + logvar) for VAE reparameterisation')
    parser.add_argument("--dims", default=2, type=int, help="Conv dim; N of ConvNd", choices=[1, 2, 3])
    parser.add_argument('--test_case', default=False, type=bool, help='True: not load real data, using rand values')

    # ------------------------------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--out_channels", default=1, type=int, help='Number of output channels')

    # ------------------------------------------------------------------------------------------
    # Loss(VAELoss)
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--lambda_rec", default=1.0, type=float)
    parser.add_argument("--lambda_ssim", default=1.0, type=float)
    parser.add_argument("--lambda_kl", default=1e-4, type=float)
    parser.add_argument("--lambda_mmd", default=1e-3, type=float)
    parser.add_argument("--mmd_sigma", default=1.0, type=float, help="Bandwidth σ_k for Gaussian MMD kernel")
    parser.add_argument("--data_range",default=2.0, help='value range of images (2.0 for [-1, 1] normalised input)')

    # ------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--verbose", default=False, type=bool, help="Print block shapes at each encoder/decoder level")
    parser.add_argument("--debug", default=True, type=bool, help="Print detailed loss table + save recon images "
                             "for the first step of every epoch")
    parser.add_argument("--log_every", default=1000, type=int, help="Print step-level log every N steps")
    parser.add_argument("--num_workers", default=0, type=int)

    # ------------------------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)

    # ------------------------------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------------------------------
    parser.add_argument("--save_dir", default="./checkpoints/vae")
    parser.add_argument("--save_every", default=10, type=int,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", default=None, type=str,
                        help="Path to a .pt checkpoint to resume from")

    return parser.parse_args()


# ============================================================================================
# Fake dataset (test_case=True)
# ============================================================================================

class _FakeDataset(Dataset):
    """Returns random tensors so the full pipeline can be verified without data."""

    def __init__(self, args, length: int = 200):
        self.shape = (args.in_channels, args.image_size, args.image_size)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Mimic normalised image in [-1, 1]
        x = torch.randn(*self.shape)
        return x, "fake"


# ===========================================================================================================================
# Builders
# ===========================================================================================================================

def build_loader(args) -> DataLoader:
    if args.test_case:
        dataset = _FakeDataset(args)
        dataloader = DataLoader(
            dataset,
            batch_size=args.bs,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        print("[DataLoader] test_case=True, using random (fake) data")
    else:
        dataset = NIH(args)
        dataloader = DataLoader(
                    dataset,
                    batch_size=args.bs,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    )
    return dataloader


def build_model(args, device) -> VAE:
    model = VAE(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] AutoencoderKL  trainable params={n_params:,}")
    return model


def build_criterion(args) -> VAELoss:
    return VAELoss(args)


# ==============================================================================================================
# Debug image saver
# ==============================================================================================================

def _save_recon_images(x, x_hat, save_dir, epoch, step, n: int = 4):
    """Save a side-by-side grid of originals vs reconstructions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import torchvision.utils as vutils

        os.makedirs(save_dir, exist_ok=True)
        n = min(n, x.size(0))

        # Denormalise from [-1, 1] to [0, 1]
        to_grid = lambda t: vutils.make_grid(
            (t[:n].detach().cpu().clamp(-1, 1) + 1) / 2,
            nrow=n, padding=2,
        )
        grid_in = to_grid(x).permute(1, 2, 0).squeeze()
        grid_out = to_grid(x_hat).permute(1, 2, 0).squeeze()

        fig, axes = plt.subplots(2, 1, figsize=(n * 3, 6))
        axes[0].imshow(grid_in, cmap="gray");
        axes[0].set_title("Original");
        axes[0].axis("off")
        axes[1].imshow(grid_out, cmap="gray");
        axes[1].set_title("Reconstruction");
        axes[1].axis("off")

        path = os.path.join(save_dir, f"recon_ep{epoch:04d}_step{step:05d}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close(fig)
        print(f"  [Debug] Saved recon image �� {path}")

    except Exception as e:
        print(f"  [Debug] Could not save images: {e}")


# ==============================================================================================================
# Training loop
# ==============================================================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, args, epoch):
    model.train()
    running = {k: 0.0 for k in ("loss_total", "loss_rec", "loss_ssim", "loss_kl", "loss_mmd")}
    debug_img_dir = os.path.join(args.save_dir, f"debug_imgs ({args.run_date})")

    for step, (x, _) in enumerate(loader):
        x = x.to(device)

        # Forward
        posterior = model.encode(x)  #DiagonalGaussianDistribution
        z = posterior.sample()  # reparameterisation trick  [B, z_ch, h, w]
        x_hat = model.decode(z)  # reconstructed image       [B, C, H, W]

        # Loss
        if args.debug and step == 0:
            # First step of every epoch: detailed table + save images
            loss, loss_dict = criterion.debug_forward(x, x_hat, posterior, z)
            _save_recon_images(x, x_hat, debug_img_dir, epoch, step)
        else:
            loss, loss_dict = criterion(x, x_hat, posterior, z)

        # Backpropa
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate
        for k in running:
            running[k] += loss_dict[k]

        # Step log
        if (step + 1) % args.log_every == 0:
            print(
                f"  [Ep {epoch:4d} | Step {step + 1:5d}/{len(loader)}]"
                f"  total={loss_dict['loss_total']:.4f}"
                f"  rec={loss_dict['loss_rec']:.4f}"
                f"  ssim={loss_dict['loss_ssim']:.4f}"
                f"  kl={loss_dict['loss_kl']:.6f}"
                f"  mmd={loss_dict['loss_mmd']:.6f}"
            )

    n = len(loader)
    return {k: v / n for k, v in running.items()}


# =================================================================================
# Main
# ================================================================================

def main():
    args = parse_args()

    # Setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device_id)
    os.makedirs(args.save_dir, exist_ok=True)

    # Modeling
    loader = build_loader(args)
    model = build_model(args, device)
    criterion = build_criterion(args)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume
    # If training was interrupted (e.g. server crash, time limit), resume from
    # a saved checkpoint instead of restarting from scratch.
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[Resume] Loaded epoch {ckpt['epoch']}")

    # Config summary
    print(f"\n{'=' * 80}")
    print(f"  device     : {device, args.device_id}")
    print(f"  data       : {'FakeDataset' if args.test_case else 'NIH'}")
    print(f"  batch size : {args.bs}   epochs : {args.n_epochs}   lr : {args.lr}")
    print(f"  z_channels : {args.z_channels}   resolution : {args.resolution}")
    print(f"  λ_rec={args.lambda_rec}  λ_ssim={args.lambda_ssim}"
          f"  λ_kl={args.lambda_kl}    λ_mmd={args.lambda_mmd}")
    print(f"  debug mode : {args.debug}   save_every : {args.save_every} epoch")
    print(f"{'=' * 80}\n")

    # Training Loop
    for epoch in range(start_epoch, args.n_epochs + 1):
        avg = train_one_epoch(model, loader, criterion, optimizer, device, args, epoch)

        print(
            f"[Epoch {epoch:4d}/{args.n_epochs}]"
            f"  total={avg['loss_total']:.4f}"
            f"  rec={avg['loss_rec']:.4f}"
            f"  ssim={avg['loss_ssim']:.4f}"
            f"  kl={avg['loss_kl']:.6f}"
            f"  mmd={avg['loss_mmd']:.6f}"
        )

        # Save checkpoint (Periodic)
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"vae_ep{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> Checkpoint saved: {ckpt_path}")

    # Final save
    final = os.path.join(args.save_dir, "vae_final.pt")
    torch.save({"epoch": args.n_epochs, "model": model.state_dict(), "args": vars(args)}, final)
    print(f"\nTraining complete. Final model: {final}")


if __name__ == "__main__":
    main()