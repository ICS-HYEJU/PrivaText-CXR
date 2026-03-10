import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encoder import Encoder, DiagonalGaussianDistribution
from decoder import Decoder


# ============================================================================
# AutoencoderKL
# ============================================================================

class AutoencoderKL(nn.Module):
    """
    VAE combining Encoder and Decoder.

    Typical usage
    -------------
    # Training
    posterior = model.encode(x)          # DiagonalGaussianDistribution
    z         = posterior.sample()       # reparameterisation trick
    recon     = model.decode(z)          # reconstructed image
    kl_loss   = posterior.kl().mean()

    # Inference (deterministic)
    z    = model.encode(x).mode()        # use mean instead of sampling
    recon = model.decode(z)
    """

    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    # ─────────────────────────────────────────────────────────────────────────
    def encode(self, x) -> DiagonalGaussianDistribution:
        """
        Encode an input image into a Gaussian posterior.

        Args:
            x: input image tensor  [B, in_channels, H, W]

        Returns:
            DiagonalGaussianDistribution
              .sample() → sampled latent z  [B, z_channels, H/16, W/16]
              .mode()   → mean of posterior [B, z_channels, H/16, W/16]
              .kl()     → KL divergence from N(0, I) per sample  [B]
        """
        # Encoder outputs concatenated [mean | logvar] along channel dim
        h = self.encoder(x)
        return DiagonalGaussianDistribution(h)

    # ─────────────────────────────────────────────────────────────────────────
    def decode(self, z) -> torch.Tensor:
        """
        Decode a latent vector back into image space.

        Args:
            z: latent tensor  [B, z_channels, H/16, W/16]

        Returns:
            reconstructed image  [B, out_channels, H, W]
        """
        return self.decoder(z)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, x, sample_posterior=True):
        """
        Full encode → (sample or mode) → decode pass.

        Args:
            x               : input image  [B, in_channels, H, W]
            sample_posterior: if True, sample z ~ q(z|x);
                              if False, use the posterior mean (deterministic)

        Returns:
            recon    : reconstructed image  [B, out_channels, H, W]
            posterior: DiagonalGaussianDistribution (for KL loss computation)
        """
        posterior = self.encode(x)
        z         = posterior.sample() if sample_posterior else posterior.mode()
        recon     = self.decode(z)
        return recon, posterior


# ============================================================================
# Main – quick sanity check
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_channels",      default=1,            type=int)
    parser.add_argument("--out_channels",     default=1,            type=int)
    parser.add_argument("--ch",               default=128,          type=int)
    parser.add_argument("--ch_mult",          default=[1, 2, 4, 4, 4])
    parser.add_argument("--num_res_blocks",   default=2,            type=int)
    parser.add_argument("--attn_resolutions", default=[32, 16])
    parser.add_argument("--dropout",          default=0.0,          type=float)
    parser.add_argument("--resamp_with_conv", default=True,         type=bool)
    parser.add_argument("--resolution",       default=256,          type=int)
    parser.add_argument("--z_channels",       default=3,            type=int)
    parser.add_argument("--double_z",         default=True,         type=bool)
    parser.add_argument("--dims",             default=2,            type=int)

    args = parser.parse_args()

    B     = 2
    model = AutoencoderKL(args)
    dummy = torch.zeros(B, args.in_channels, args.resolution, args.resolution)

    print("─── encode ──────────────────────────────────────────────────")
    posterior = model.encode(dummy)
    z         = posterior.sample()
    kl        = posterior.kl()
    print(f"  posterior mean  : {posterior.mean.shape}")
    print(f"  posterior logvar: {posterior.logvar.shape}")
    print(f"  sampled z       : {z.shape}")
    print(f"  KL mean         : {kl.mean().item():.4f}")

    print()
    print("─── decode ──────────────────────────────────────────────────")
    recon = model.decode(z)
    print(f"  recon shape     : {recon.shape}")
    print(f"  expected        : [{B}, {args.out_channels}, {args.resolution}, {args.resolution}]")

    print()
    print("─── forward (sample_posterior=True) ─────────────────────────")
    recon, posterior = model(dummy, sample_posterior=True)
    print(f"  recon shape     : {recon.shape}")
    print(f"  KL mean         : {posterior.kl().mean().item():.4f}")

    print()
    print("─── forward (sample_posterior=False, deterministic) ─────────")
    recon, posterior = model(dummy, sample_posterior=False)
    print(f"  recon shape     : {recon.shape}")
