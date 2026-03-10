import torch
import torch.nn as nn
import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from swin_attention import SwinTransformerBlock
from UNetBlock_module import ResBlock, Downsample
from util_network import conv_nd, normalization


# ============================================================================
# Helper modules
# ============================================================================
class SwinWrapper2D(nn.Module):
    """
    Adapts SwinTransformerBlock (expects NHWC) for NCHW feature maps.

    Permutes NCHW -> NHWC before Swin, then NHWC -> NCHW after.
    Mask is always None (no cyclic-shift mask needed here).
    """

    def __init__(self, dim, num_heads, window_size, shift_size, **kwargs):
        super().__init__()
        self.swin = SwinTransformerBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            **kwargs,
        )

    def forward(self, x):
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        x = self.swin(x, mask_matrix=None)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return x

# ============================================================================
# Encoder
# ============================================================================

class Encoder(nn.Module):
    """
    VAE Encoder supporting the following architecture:

    Input : [B, in_channels, H, W]

    ConvIn : in_channels -> ch

    Level 0 : ResBlock -> num_res_blocks -> Downsample
    Level 1 : ResBlock -> num_res_blocks -> Downsample
    Level 2 : ResBlock -> num_res_blocks -> Downsample
    Level 3 : (ResBlock + SwinBlock) -> num_res_blocks -> Downsample   # attn_res=32
    Level 4 : (ResBlock + SwinBlock) -> num_res_blocks                 # attn_res=16

    Middle  : ResBlock -> SwinBlock -> ResBlock

    Output  : Norm -> SiLU -> ConvOut (block_in -> z_channels*2 if double_z)

    Default args (from argparse):
        ch=128, ch_mult=[1,2,4,4,4], num_res_blocks=2,
        attn_resolutions=[32,16], resolution=256,
        z_channels=256, double_z=True, dims=2
    """

    # Swin hyper-params (fixed; tune if needed)
    SWIN_NUM_HEADS = 8
    SWIN_WINDOW_SIZE = (8, 8)  # 2-D window
    SWIN_SHIFT_SIZE = (4, 4)  # half of window_size (for alternating shift)

    def __init__(self, args):
        super().__init__()

        self.in_channels = args.in_channels
        self.ch = args.ch
        self.ch_mult = list(args.ch_mult)
        self.num_res_blocks = args.num_res_blocks
        self.attn_resolutions = list(args.attn_resolutions)
        self.dropout = args.dropout
        self.resamp_with_conv = args.resamp_with_conv
        self.resolution = args.resolution
        self.z_channels = args.z_channels
        self.double_z = args.double_z
        self.dims = args.dims

        self.num_resolutions = len(self.ch_mult)
        self.temb_ch = 0  # no timestep embedding for Encoder

        # ---- input projection -----------------------------------------------
        self.conv_in = conv_nd(
            self.dims, self.in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        # ---- downsampling levels --------------------------------------------
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(self.ch_mult)  # prepend 1 for i_level=0 input
        self.down = nn.ModuleList()

        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResBlock(
                        channels=block_in,
                        emb_channels=0,
                        out_channels=block_out,
                        dropout=self.dropout,
                        dims=self.dims,
                    )
                )
                block_in = block_out  # subsequent blocks: in == out

                # SwinTransformerBlock at attention resolutions
                if curr_res in self.attn_resolutions:
                    # Alternate: even i_block �� no shift, odd �� shifted
                    shift = (0, 0) if i_block % 2 == 0 else self.SWIN_SHIFT_SIZE
                    attn.append(
                        SwinWrapper2D(
                            dim=block_out,
                            num_heads=self.SWIN_NUM_HEADS,
                            window_size=self.SWIN_WINDOW_SIZE,
                            shift_size=shift,
                        )
                    )

            down = nn.Module()
            down.block = block
            down.attn = attn

            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(
                    block_in,
                    self.resamp_with_conv,
                    dims=self.dims,
                )
                curr_res = curr_res // 2

            self.down.append(down)

        # ---- middle ---------------------------------------------------------
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(
            channels=block_in,
            emb_channels=0,
            out_channels=block_in,
            dropout=self.dropout,
            dims=self.dims,
        )
        self.mid.attn_1 = SwinWrapper2D(
            dim=block_in,
            num_heads=self.SWIN_NUM_HEADS,
            window_size=self.SWIN_WINDOW_SIZE,
            shift_size=(0, 0),
        )
        self.mid.block_2 = ResBlock(
            channels=block_in,
            emb_channels=0,
            out_channels=block_in,
            dropout=self.dropout,
            dims=self.dims,
        )

        # ---- output ---------------------------------------------------------
        self.norm_out = normalization(block_in)
        self.conv_out = conv_nd(
            self.dims,
            block_in,
            2 * self.z_channels if self.double_z else self.z_channels,
            kernel_size=3,
            padding=1,
        )

    # -------------------------------------------------------------------------
    def forward(self, x, verbose=False):
        """
        :param x:       [B, in_channels, H, W]
        :param verbose: print intermediate shapes for debugging
        :return:        [B, z_channels*2, H_out, W_out]  (double_z=True)
        """
        emb = None  # no timestep embedding

        h = self.conv_in(x)
        if verbose:
            print(f"  conv_in  : {x.shape} -> {h.shape}")

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h, emb)
                if verbose:
                    print(f"  L{i_level} block[{i_block}]  : ResBlock -> {h.shape}")

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                    if verbose:
                        shift = self.down[i_level].attn[i_block].swin.shift_size
                        print(f"  L{i_level} attn [{i_block}]  : Swin(shift={shift}) -> {h.shape}")

            if i_level != self.num_resolutions - 1:
                h = self.down[i_level].downsample(h)
                if verbose:
                    print(f"  L{i_level} downsample      : -> {h.shape}")

        h = self.mid.block_1(h, emb)
        if verbose:
            print(f"  mid block_1 : ResBlock  -> {h.shape}")
        h = self.mid.attn_1(h)
        if verbose:
            print(f"  mid attn_1  : Swin      -> {h.shape}")
        h = self.mid.block_2(h, emb)
        if verbose:
            print(f"  mid block_2 : ResBlock  -> {h.shape}")

        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        if verbose:
            print(f"  conv_out    :            -> {h.shape}")

        return h

    # -------------------------------------------------------------------------
    def print_architecture(self):
        """Pretty-print the full encoder block layout."""
        print("=" * 60)
        print("Encoder Architecture")
        print("=" * 60)
        print(f"  conv_in : {self.in_channels} -> {self.ch}  (3x3 conv)")
        print()

        for i_level in range(self.num_resolutions):
            block_in = self.ch * ((1,) + tuple(self.ch_mult))[i_level]
            block_out = self.ch * self.ch_mult[i_level]
            has_attn = len(self.down[i_level].attn) > 0
            print(
                f"  ====== Level {i_level} {'(attn)' if has_attn else '      '} ===========================================================")

            for i_block in range(self.num_res_blocks):
                ch_in = block_in if i_block == 0 else block_out
                print(f"     block[{i_block}] : ResBlockEncoder  {ch_in} -> {block_out}")
                if has_attn:
                    shift = self.down[i_level].attn[i_block].swin.shift_size
                    print(f"      attn[{i_block}] : SwinWrapper2D    dim={block_out}"
                          f"  heads={self.SWIN_NUM_HEADS}"
                          f"  win={self.SWIN_WINDOW_SIZE}"
                          f"  shift={shift}")

            if i_level != self.num_resolutions - 1:
                print(f"     downsample : Downsample({block_out})")
            print()

        print("  ====== Middle ===========================================================")
        mid_ch = self.ch * self.ch_mult[-1]
        print(f"     block_1 : ResBlockEncoder  {mid_ch} -> {mid_ch}")
        print(f"     attn_1  : SwinWrapper2D    dim={mid_ch}"
              f"  heads={self.SWIN_NUM_HEADS}"
              f"  win={self.SWIN_WINDOW_SIZE}"
              f"  shift=(0, 0)")
        print(f"     block_2 : ResBlockEncoder  {mid_ch} -> {mid_ch}")
        print()

        out_ch = 2 * self.z_channels if self.double_z else self.z_channels
        print("  ====== Output ===========================================================")
        print(f"     norm_out  : GroupNorm32")
        print(f"     SiLU")
        print(f"     conv_out  : {mid_ch} -> {out_ch}  (3x3 conv)")
        print("=" * 60)


# ============================================================================
# DiagonalGaussianDistribution
# ============================================================================

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters  # [B, z_channels*2, H, W]
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def kl(self, other=None):
        if self.deterministic:
            return torch.tensor([0.])
        if other is None:
            # KL divergence between the encoder's posterior q(z|x) and the standard normal prior N(0, I)
            # KL(N(mean, var)||N(0,1) = 1/2 * (mean**2 + var**2 -1 -log(var**2))
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        return 0.5 * torch.sum(
            torch.pow(self.mean - other.mean, 2) / other.var
            + self.var / other.var - 1.0 - self.logvar + other.logvar,
            dim=[1, 2, 3],
        )

    def nll(self, sample, dims=(1, 2, 3)):
        if self.deterministic:
            return torch.tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    from Data.dataset import NIH

    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--root_path", default="/storage/hjchoi/archive/DATA")
    parser.add_argument("--task", default="train", choices=["train", "val", "test"])
    parser.add_argument("--bs", default=2, type=int, help='batch size')
    parser.add_argument("--image_size", default=256, type=int,help='the value to resize')
    parser.add_argument("--image_show", default=True, type=bool)

    # Encoder
    parser.add_argument("--in_channels", default=1, type=int, help='Number of input img channels, NIH=gray-scale')
    parser.add_argument("--ch", default=128, type=int, help='Base channel')
    parser.add_argument("--ch_mult", default=[1, 2, 4, 4, 4], help='Channel multipliers per each level')
    parser.add_argument("--num_res_blocks", default=2, type=int, help='Number of residual blocks per each level')
    parser.add_argument("--attn_resolutions", default=[32, 16], help='the resolution at which attention is applied')
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--resamp_with_conv", default=True, type=bool, help='Use strided conv for downsampling; False uses avg-pool')
    parser.add_argument("--resolution", default=256, type=int, help='Input spatial resolution (H = W)')
    parser.add_argument("--z_channels", default=3, type=int, help= 'Latent z-space channel dim')
    parser.add_argument("--double_z", default=True, type=bool, help='Output 2*z_channels (mean + logvar) for VAE reparameterisation')
    parser.add_argument("--dims", default=2, type=int, help="Conv dim; N of ConvNd", choices=[1, 2, 3])
    parser.add_argument('--test_case',default=False, type=bool, help='True: not load real data, using rand values' )

    # Decoder
    parser.add_argument("--out_channels", default=1, type=int, help='Number of output channels')
    args = parser.parse_args()

    encoder = Encoder(args)
    encoder.print_architecture() # Print architecture
    print()

    if args.test_case:
        # Dummy-tensor forward (no dataset needed)
        B = 2
        dummy = torch.zeros(B, args.in_channels, args.resolution, args.resolution)

        print("Forward pass (verbose=True):")
        with torch.no_grad():
            h = encoder(dummy, verbose=True)

        print()
        print(f"Encoder output shape : {h.shape}")
        print(f"Expected             : [{B}, {2 * args.z_channels}, 16, 16]")
        print()

        posterior = DiagonalGaussianDistribution(h)
        z = posterior.sample()
        kl = posterior.kl()

        print(f"mean shape  : {posterior.mean.shape}")
        print(f"logvar shape: {posterior.logvar.shape}")
        print(f"z shape     : {z.shape}   (expected [{B}, {args.z_channels}, 16, 16])")
        print(f"KL mean     : {kl.mean().item():.4f}")
        print()
    else:
        # NIH dataset
        try:
            dataset = NIH(args)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.bs, shuffle=True)

            for batch_id, data in enumerate(dataloader):
                if batch_id == 1:
                    break
                image, label = data[0], data[1]
                print(f"Real image shape : {image.shape}")
                with torch.no_grad():
                    h = encoder(image, verbose=True)
                print(f"Encoder output   : {h.shape}")
                posterior = DiagonalGaussianDistribution(h)
                z = posterior.sample()
                kl = posterior.kl()
                print(f"Sampled z        : {z.shape}")
                print(f"KL mean        : {kl.mean().item():.4f}")
                latent_dim  = z.shape[1]*z.shape[2]*z.shape[3]
                print(f"KL per dim     : {kl.mean().item()/latent_dim:.6f}")
        except Exception as e:
            print(f"[Dataset skipped] {e}")