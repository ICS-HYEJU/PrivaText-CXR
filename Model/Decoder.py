import torch
import torch.nn as nn
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from UNetBlock_module import ResBlock, Upsample
from util_network import conv_nd, normalization
from Encoder import SwinWrapper2D, Encoder, DiagonalGaussianDistribution


# ============================================================================
# Decoder
# ============================================================================

class Decoder(nn.Module):
    """
    VAE Decoder - mirrors the Encoder in reversed order.
    No skip connections; pure generative decoder for VAE reconstruction.

    Input : [B, z_channels, H/16, W/16]

    ConvIn  : z_channels -> block_in  (= ch -> ch_mult[-1])

    Middle  : ResBlock -> SwinBlock(shift=(0,0)) -> ResBlock

    Level 4 : (ResBlock + SwinBlock) -> num_res_blocks -> Upsample   # attn_res=16
    Level 3 : (ResBlock + SwinBlock) -> num_res_blocks -> Upsample   # attn_res=32
    Level 2 : ResBlock -> num_res_blocks               -> Upsample
    Level 1 : ResBlock -> num_res_blocks               -> Upsample
    Level 0 : ResBlock -> num_res_blocks

    Output  : Norm -> SiLU -> ConvOut (block_out -> out_channels)

    ResBlock is used with emb_channels=0 (no timestep embedding).
    """

    SWIN_NUM_HEADS = 8
    SWIN_WINDOW_SIZE = (8, 8)
    SWIN_SHIFT_SIZE = (4, 4)

    def __init__(self, args):
        super().__init__()

        self.ch = args.ch
        self.ch_mult = list(args.ch_mult)
        self.num_res_blocks = args.num_res_blocks
        self.attn_resolutions = list(args.attn_resolutions)
        self.dropout = args.dropout
        self.resamp_with_conv = args.resamp_with_conv
        self.resolution = args.resolution
        self.z_channels = args.z_channels
        self.out_channels = args.out_channels
        self.dims = args.dims

        self.num_resolutions = len(self.ch_mult)

        # Deepest channel count ? matches the encoder's final feature channels
        block_in = self.ch * self.ch_mult[-1]

        # ==== input projection (z -> block_in) =================================================
        self.conv_in = conv_nd(
            self.dims, self.z_channels, block_in,
            kernel_size=3, stride=1, padding=1,
        )

        # ==== middle ===========================================================================
        self.mid = nn.Module()
        self.mid.block_1 = ResBlock(block_in, emb_channels=0, dropout=self.dropout, dims=self.dims)
        self.mid.attn_1 = SwinWrapper2D(
            dim=block_in,
            num_heads=self.SWIN_NUM_HEADS,
            window_size=self.SWIN_WINDOW_SIZE,
            shift_size=(0, 0),
        )
        self.mid.block_2 = ResBlock(block_in, emb_channels=0, dropout=self.dropout, dims=self.dims)

        # ==== upsampling levels ===========================================================================
            # curr_res starts at the minimum spatial resolution (after all encoder downsamples)
            # e.g. resolution=256 with 5 levels (4 downsamples): 256 // 2^4 = 16
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))
        self.up = nn.ModuleList()

        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.ch * self.ch_mult[i_level]

            for i_block in range(self.num_res_blocks):
                block.append(
                    ResBlock(
                        channels=block_in,
                        emb_channels=0,  # Decoder: no timestep embedding
                        dropout=self.dropout,
                        out_channels=block_out,
                        dims=self.dims,
                    )
                )
                block_in = block_out

                if curr_res in self.attn_resolutions:
                    # Alternate shift to enable shifted-window attention
                    shift = (0, 0) if i_block % 2 == 0 else self.SWIN_SHIFT_SIZE
                    attn.append(
                        SwinWrapper2D(
                            dim=block_out,
                            num_heads=self.SWIN_NUM_HEADS,
                            window_size=self.SWIN_WINDOW_SIZE,
                            shift_size=shift,
                        )
                    )

            up = nn.Module()
            up.block = block
            up.attn = attn

            if i_level != 0:
                # Upsample at every level except the last (finest resolution)
                up.upsample = Upsample(block_in, self.resamp_with_conv, dims=self.dims)
                curr_res *= 2

            # Insert at index 0 so self.up[i] corresponds to encoder level i
            self.up.insert(0, up)

        # ==== output ===========================================================================
        self.norm_out = normalization(block_in)
        self.conv_out = conv_nd(
            self.dims, block_in, self.out_channels, kernel_size=3, padding=1,
        )

    def forward(self, z, verbose=False):
        # Project z into feature space
        h = self.conv_in(z)
        if verbose:
            print(f"  conv_in     : {z.shape} -> {h.shape}")

        # Middle block (same resolution as z)
        h = self.mid.block_1(h)
        if verbose:
            print(f"  mid block_1 : ResBlock -> {h.shape}")
        h = self.mid.attn_1(h)
        if verbose:
            print(f"  mid attn_1  : Swin     -> {h.shape}")
        h = self.mid.block_2(h)
        if verbose:
            print(f"  mid block_2 : ResBlock -> {h.shape}")

        # Upsampling levels (coarse -> fine: level 4 -> level 0)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks):
                h = self.up[i_level].block[i_block](h)
                if verbose:
                    print(f"  L{i_level} block[{i_block}] : ResBlock -> {h.shape}")

                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                    if verbose:
                        shift = self.up[i_level].attn[i_block].swin.shift_size
                        print(f"  L{i_level}  attn[{i_block}] : Swin(shift={shift}) -> {h.shape}")

            if i_level != 0:
                h = self.up[i_level].upsample(h)
                if verbose:
                    print(f"  L{i_level} upsample      : _-> {h.shape}")

        # Output projection
        h = self.norm_out(h)
        h = nn.SiLU()(h)
        h = self.conv_out(h)
        if verbose:
            print(f"  conv_out    :           -> {h.shape}")

        return h


    def print_architecture(self):
        """Print a human-readable summary of every block and its channel sizes."""
        print("=" * 60)
        print("Decoder Architecture")
        print("=" * 60)

        block_in = self.ch * self.ch_mult[-1]
        curr_res = self.resolution // (2 ** (self.num_resolutions - 1))

        print(f"  conv_in : {self.z_channels} -> {block_in}  (3x3 conv)")
        print()

        print("  ==== Middle ===========================================================================")
        mid_ch = block_in
        print(f"     block_1 : ResBlock      {mid_ch} -> {mid_ch}  (emb_channels=0)")
        print(f"     attn_1  : SwinWrapper2D dim={mid_ch}"
              f"  heads={self.SWIN_NUM_HEADS}"
              f"  win={self.SWIN_WINDOW_SIZE}"
              f"  shift=(0, 0)")
        print(f"     block_2 : ResBlock      {mid_ch} -> {mid_ch}  (emb_channels=0)")
        print()

        for i_level in reversed(range(self.num_resolutions)):
            block_out = self.ch * self.ch_mult[i_level]
            has_attn = curr_res in self.attn_resolutions
            print(
                f"  ==== Level {i_level} {'(attn)' if has_attn else '      '} ===========================================================================")

            for i_block in range(self.num_res_blocks):
                ch_in = block_in if i_block == 0 else block_out
                print(f"     block[{i_block}] : ResBlock  {ch_in} -> {block_out}  (emb_channels=0)")
                if has_attn:
                    shift = (0, 0) if i_block % 2 == 0 else self.SWIN_SHIFT_SIZE
                    print(f"      attn[{i_block}] : SwinWrapper2D  dim={block_out}"
                          f"  heads={self.SWIN_NUM_HEADS}"
                          f"  win={self.SWIN_WINDOW_SIZE}"
                          f"  shift={shift}")

            if i_level != 0:
                print(f"     upsample : Upsample({block_out})  {curr_res} -> {curr_res * 2}")
                curr_res *= 2

            block_in = block_out
            print()

        print("  ==== Output ===========================================================================")
        print(f"     norm_out  : GroupNorm32")
        print(f"     SiLU")
        print(f"     conv_out  : {block_in} -> {self.out_channels}  (3x3 conv)")
        print("=" * 60)


# ============================================================================
# print_level_shapes ? standalone shape-inspection utility
# ============================================================================

def print_level_shapes(encoder: Encoder, decoder: "Decoder", batch_size: int = 1) -> None:
    """
    Run a full encode -> sample z -> decode forward pass and print the output
    shape at every block across both the Encoder and Decoder.

    Args:
        encoder   : Encoder instance (must match decoder config).
        decoder   : Decoder instance.
        batch_size: Number of samples in the dummy batch (default 1).
    """
    device = next(encoder.parameters()).device

    # Create a dummy input image matching the encoder's expected resolution
    dummy_img = torch.zeros(
        batch_size, encoder.in_channels,
        encoder.resolution, encoder.resolution,
        device=device,
    )

    print("=" * 60)
    print("Encoder - forward pass shapes")
    print("=" * 60)
    with torch.no_grad():
        enc_out = encoder(dummy_img, verbose=True)

    print()
    print(f"  Encoder output : {enc_out.shape}")
    print(f"  (mean + logvar, will be split into 2 x z_channels along dim=1)")
    print()

    # Sample z via the reparameterisation trick
    posterior = DiagonalGaussianDistribution(enc_out)
    z = posterior.sample()
    print(f"  Sampled z      : {z.shape}")
    print()

    print("=" * 60)
    print("Decoder - forward pass shapes")
    print("=" * 60)
    with torch.no_grad():
        recon = decoder(z, verbose=True)

    print()
    print(f"  Decoder output : {recon.shape}")
    print(f"  Expected       : [{batch_size}, {decoder.out_channels},"
          f" {decoder.resolution}, {decoder.resolution}]")


# ============================================================================
# Main - architecture summary and shape verification
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Shared encoder / decoder hyperparameters
    parser.add_argument("--in_channels", default=1, type=int)
    parser.add_argument("--out_channels", default=1, type=int,
                        help="Reconstructed image channels (usually == in_channels)")
    parser.add_argument("--ch", default=128, type=int)
    parser.add_argument("--ch_mult", default=[1, 2, 4, 4, 4])
    parser.add_argument("--num_res_blocks", default=2, type=int)
    parser.add_argument("--attn_resolutions", default=[32, 16])
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--resamp_with_conv", default=True, type=bool)
    parser.add_argument("--resolution", default=256, type=int)
    parser.add_argument("--z_channels", default=3, type=int)
    parser.add_argument("--double_z", default=True, type=bool)
    parser.add_argument("--dims", default=2, type=int)

    args = parser.parse_args()

    decoder = Decoder(args)
    decoder.print_architecture()
    print()


    encoder = Encoder(args)

    B = 2
    print_level_shapes(encoder, decoder, batch_size=B)