import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Encoder import Encoder, DiagonalGaussianDistribution
from Decoder import Decoder
from Data.dataset import NIH
from util_network import *

# ============================================================================
# VAE
# ============================================================================

class VAE(nn.Module):
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
        self.post_quant_conv = conv_nd(args.dims, args.z_channels, args.z_channels,kernel_size=1,stride=1)

    def encode(self, x) -> DiagonalGaussianDistribution:
        """
        Encode an input image into a Gaussian posterior.

        Args:
            x: input image tensor  [B, in_channels, H, W]

        Returns:
            DiagonalGaussianDistribution
              .sample() : sampled latent z  [B, z_channels, H/16, W/16]
              .mode()   : mean of posterior [B, z_channels, H/16, W/16]
              .kl()     : KL divergence from N(0, I) per sample  [B]
        """
        # Encoder outputs concatenated [mean | logvar] along channel dim
        h = self.encoder(x)
        return DiagonalGaussianDistribution(h)

    def decode(self, z) -> torch.Tensor:
        """
        Decode a latent vector back into image space.

        Args:
            z: latent tensor  [B, z_channels, H/16, W/16]

        Returns:
            reconstructed image  [B, out_channels, H, W]
        """
        # Remap z in channel space before passing to the Decoder.
        # The 1x1 conv learns a linear mixing across channels, allowing
        # the decoder to operate in its own optimal feature space.
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        """
        Full encode -> sampling z -> decode pass.

        Args:
            x               : input image  [B, in_channels, H, W]
            sample_posterior: if True, sample z ~ q(z|x);
                              if False, use the posterior mean (deterministic)

        Returns:
            recon    : reconstructed image  [B, out_channels, H, W]
            posterior: DiagonalGaussianDistribution (for KL loss computation)
        """
        posterior = self.encode(x)
        z = posterior.sample() if sample_posterior else posterior.mode()
        recon = self.decode(z)
        return recon, posterior


# ============================================================================
# Main - quick sanity check
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--root_path", default="/storage/hjchoi/archive/DATA")
    parser.add_argument("--task", default="train", choices=["train", "val", "test"])
    parser.add_argument("--bs", default=2, type=int, help='batch size')
    parser.add_argument("--image_size", default=256, type=int, help='the value to resize')
    parser.add_argument("--image_show", default=True, type=bool)

    # Encoder
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

    # Decoder
    parser.add_argument("--out_channels", default=1, type=int, help='Number of output channels')

    # Print
    parser.add_argument('--verbose', default=False, type=bool,help='True: print Block and output shape at each levels')
    args = parser.parse_args()

    if args.test_case:
        B = 2
        model = VAE(args)
        dummy = torch.zeros(B, args.in_channels, args.resolution, args.resolution)

        print(
            "====== encode ========================================================================")
        posterior = model.encode(dummy)
        z = posterior.sample()
        kl = posterior.kl()
        print(f"  posterior mean  : {posterior.mean.shape}")
        print(f"  posterior logvar: {posterior.logvar.shape}")
        print(f"  sampled z       : {z.shape}")
        print(f"  KL mean         : {kl.mean().item():.4f}")

        print()
        print("====== decode ========================================================================")
        recon = model.decode(z)
        print(f"  recon shape     : {recon.shape}")
        print(f"  expected        : [{B}, {args.out_channels}, {args.resolution}, {args.resolution}]")

        print()
        print("====== forward ========================================================================")
        recon, posterior = model(dummy, sample_posterior=True)
        print(f"  recon shape     : {recon.shape}")
        print(f"  KL mean         : {posterior.kl().mean().item():.4f}")
    else:
        try:
            dataset = NIH(args)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)
            for batch_id, data in enumerate(dataloader):
                if batch_id == 1:
                    break
                image, label = data[0], data[1]
                print(f"image shape : {image.shape}")
                # ==============================================
                model= VAE(args)
                posterior = model.encode(image)
                z = posterior.sample()
                kl = posterior.kl()
                print(f"  posterior mean  : {posterior.mean.shape}")
                print(f"  posterior logvar: {posterior.logvar.shape}")
                print(f"  sampled z       : {z.shape}")
                print(f"  KL mean         : {kl.mean().item():.4f}")
                recon = model.decode(z)
                print(f"  recon shape     : {recon.shape}")
                print(f"  expected        : [{args.bs}, {args.out_channels}, {args.resolution}, {args.resolution}]")
                # ===== Using model.forward ==============================================
                recon, posterior = model(image, sample_posterior=True)
        except Exception as e:
            print(f"error: {e}")
