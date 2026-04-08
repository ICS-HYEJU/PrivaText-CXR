"""
Diffusion/UNetModel.py  –  UNet with timestep + cross-attention conditioning
=============================================================================

Building blocks from Model/:
    ResBlock, Upsample, Downsample  ←  Model/UNetBlock_module.py
    SpatialTransformer              ←  Model/attention_module.py
    TimestepBlock,
    TimestepEmbedSequential         ←  Model/timestep_block.py
    util helpers                    ←  Model/util_network.py

AttentionBlock (self-attention fallback when use_spatial_transformer=False)
is implemented locally.
"""

import argparse
import json
import math
import os
import sys

import torch as th
import torch.nn as nn
import torch.nn.functional as F

# ── Path setup ─────────────────────────────────────────────────────────────
# Supports both:
#   python3 Diffusion/UNetModel.py          (project root as cwd)
#   python3 UNetModel.py                    (Diffusion/ as cwd)
_this_dir   = os.path.dirname(os.path.abspath(__file__))
_proj_root  = os.path.join(_this_dir, '..')
_model_dir  = os.path.join(_proj_root, 'Model')
for _d in [_proj_root, _model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from util_network import (                              # noqa: E402
    conv_nd, linear, normalization,
    zero_module, checkpoint, timestep_embedding,
)
from UNetBlock_module import ResBlock, Upsample, Downsample    # noqa: E402
from timestep_block import TimestepBlock, TimestepEmbedSequential  # noqa: E402
from attention_module import SpatialTransformer         # noqa: E402


# =============================================================================
# AttentionBlock  (self-attention; used when use_spatial_transformer=False)
# =============================================================================

class QKVAttentionLegacy(nn.Module):
    """Multi-head QKV self-attention."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale  = 1.0 / math.sqrt(ch)
        weight = th.einsum('bct,bcs->bts', q * scale, k * scale)
        weight = th.softmax(weight.float(), dim=-1).to(weight.dtype)
        out    = th.einsum('bts,bcs->bct', weight, v)
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """Plain spatial self-attention (no cross-attention context)."""

    def __init__(self, channels: int, num_heads: int = 1,
                 num_head_channels: int = -1,
                 use_checkpoint: bool = False,
                 use_new_attention_order: bool = False):
        super().__init__()
        self.channels       = channels
        self.use_checkpoint = use_checkpoint

        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm     = normalization(channels)
        self.qkv      = conv_nd(1, channels, channels * 3, 1)
        self.attn     = QKVAttentionLegacy(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x: th.Tensor, emb=None) -> th.Tensor:
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x: th.Tensor) -> th.Tensor:
        b, c, *spatial = x.shape
        x_flat = x.reshape(b, c, -1)
        h = self.norm(x_flat)
        h = self.qkv(h)
        h = self.attn(h)
        h = self.proj_out(h)
        return (x_flat + h).reshape(b, c, *spatial)


# =============================================================================
# UNetModel
# =============================================================================

class UNetModel(nn.Module):
    """
    Full UNet with timestep embedding and optional cross-attention conditioning.
    Accepts an argparse.Namespace (args) as configuration.

    Key args:
        image_size, in_channels, model_channels, out_channels
        num_res_blocks, attention_resolutions, channel_mult
        num_heads or num_head_channels  (exactly one must be != -1)
        use_spatial_transformer, transformer_depth, context_dim
        dropout, conv_resample, dims
        use_checkpoint, use_fp16, use_scale_shift_norm
        resblock_updown, use_new_attention_order, legacy
        num_classes, n_embed
        write_json  (debug: dump forward shapes to forward_log.json)
    """

    def __init__(self, args):
        super().__init__()

        # ── Validate ───────────────────────────────────────────────────────
        if args.use_spatial_transformer:
            assert args.context_dim is not None, \
                "context_dim must be set when use_spatial_transformer=True"
        if args.context_dim is not None:
            assert args.use_spatial_transformer, \
                "use_spatial_transformer must be True when context_dim is set"

        assert not (args.num_heads == -1 and args.num_head_channels == -1), \
            "Set either --num_heads or --num_head_channels"

        # ── Store config ───────────────────────────────────────────────────
        self.image_size           = args.image_size
        self.in_channels          = args.in_channels
        self.model_channels       = args.model_channels
        self.out_channels         = args.out_channels
        self.num_res_blocks       = args.num_res_blocks
        self.attention_resolutions = args.attention_resolutions
        self.dropout              = args.dropout
        self.channel_mult         = args.channel_mult
        self.conv_resample        = args.conv_resample
        self.num_classes          = args.num_classes
        self.use_checkpoint       = args.use_checkpoint
        self.dtype                = th.float16 if args.use_fp16 else th.float32
        self.num_heads            = args.num_heads
        self.num_head_channels    = args.num_head_channels
        self.num_heads_upsample   = args.num_heads if args.num_heads_upsample == -1 \
                                    else args.num_heads_upsample
        self.predict_codebook_ids = args.n_embed is not None
        self.write_json           = args.write_json   # ← stored as attribute

        # ── Time embedding ─────────────────────────────────────────────────
        time_embed_dim = args.model_channels * 4
        self.time_embed = nn.Sequential(
            linear(args.model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(args.num_classes, time_embed_dim)

        # ── Internal helpers ───────────────────────────────────────────────
        def _head_params(ch):
            """Return (num_heads, dim_head) for a given channel count."""
            if args.num_head_channels == -1:
                h = args.num_heads
                d = ch // h
            else:
                h = ch // args.num_head_channels
                d = args.num_head_channels
            if args.legacy:
                d = ch // h if args.use_spatial_transformer else args.num_head_channels
            return h, d

        def _attn(ch):
            h, d = _head_params(ch)
            if args.use_spatial_transformer:
                return SpatialTransformer(
                    ch, h, d,
                    depth=args.transformer_depth,
                    context_dim=args.context_dim,
                )
            else:
                return AttentionBlock(
                    ch,
                    num_heads=h,
                    num_head_channels=d,
                    use_checkpoint=args.use_checkpoint,
                    use_new_attention_order=args.use_new_attention_order,
                )

        def _resblock(ch_in, ch_out, **kw):
            return ResBlock(
                ch_in, time_embed_dim, args.dropout,
                out_channels=ch_out,
                dims=args.dims,
                use_checkpoint=args.use_checkpoint,
                use_scale_shift_norm=args.use_scale_shift_norm,
                **kw,
            )

        # ── Input (encoder) blocks ─────────────────────────────────────────
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(args.dims, args.in_channels, args.model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [args.model_channels]
        ch = args.model_channels
        ds = 1

        for level, mult in enumerate(args.channel_mult):
            for _ in range(args.num_res_blocks):
                layers = [_resblock(ch, mult * args.model_channels)]
                ch = mult * args.model_channels
                if ds in args.attention_resolutions:
                    layers.append(_attn(ch))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(args.channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    _resblock(ch, ch, down=True) if args.resblock_updown else
                    Downsample(ch, args.conv_resample, dims=args.dims, out_channels=ch)
                ))
                input_block_chans.append(ch)
                ds *= 2

        # ── Middle block ───────────────────────────────────────────────────
        self.middle_block = TimestepEmbedSequential(
            _resblock(ch, ch),
            _attn(ch),
            _resblock(ch, ch),
        )

        # ── Output (decoder) blocks ────────────────────────────────────────
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(args.channel_mult))[::-1]:
            for i in range(args.num_res_blocks + 1):
                ich    = input_block_chans.pop()
                layers = [_resblock(ch + ich, args.model_channels * mult)]
                ch     = args.model_channels * mult
                if ds in args.attention_resolutions:
                    layers.append(_attn(ch))
                if level and i == args.num_res_blocks:
                    layers.append(
                        _resblock(ch, ch, up=True) if args.resblock_updown else
                        Upsample(ch, args.conv_resample, dims=args.dims, out_channels=ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # ── Output head ────────────────────────────────────────────────────
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(args.dims, args.model_channels, args.out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(args.dims, args.model_channels, args.n_embed, 1),
            )

    # ── Utility ────────────────────────────────────────────────────────────

    def convert_to_fp16(self):
        for m in [self.input_blocks, self.middle_block, self.output_blocks]:
            m.apply(lambda x: x.half() if hasattr(x, 'half') else x)

    def convert_to_fp32(self):
        for m in [self.input_blocks, self.middle_block, self.output_blocks]:
            m.apply(lambda x: x.float() if hasattr(x, 'float') else x)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Args:
            x         : [B, in_channels, H, W]
            timesteps : [B]  diffusion timesteps
            context   : [B, seq_len, context_dim]  (None → self-attention)
            y         : [B]  class labels (only when num_classes is set)
        Returns:
            [B, out_channels, H, W]
        """
        assert (y is not None) == (self.num_classes is not None), \
            "y must be provided iff model is class-conditional"

        def _shape(t):
            if th.is_tensor(t): return list(t.shape)
            if isinstance(t, (list, tuple)): return [_shape(v) for v in t]
            return str(type(t))

        # Timestep embedding
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb   = self.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        log = {"input_x": _shape(x), "t_emb": _shape(t_emb), "emb": _shape(emb),
               "input_blocks": [], "middle_block": None,
               "output_blocks": [], "final": None} if self.write_json else None

        # Encoder path
        hs = []
        h  = x.type(self.dtype)
        for i, module in enumerate(self.input_blocks):
            h = module(h, emb, context)
            hs.append(h)
            if log is not None:
                log["input_blocks"].append({"idx": i, "shape": _shape(h)})

        # Bottleneck
        h = self.middle_block(h, emb, context)
        if log is not None:
            log["middle_block"] = {"shape": _shape(h)}

        # Decoder path
        for i, module in enumerate(self.output_blocks):
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)
            if log is not None:
                log["output_blocks"].append({"idx": i, "shape": _shape(h)})

        h   = h.type(x.dtype)
        out = self.id_predictor(h) if self.predict_codebook_ids else self.out(h)

        if log is not None:
            log["final"] = _shape(out)
            with open("forward_log.json", "w") as f:
                json.dump(log, f, indent=2)
            print("[UNetModel] forward_log.json written")

        return out


# =============================================================================
# ArgParser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="UNetModel config")

    # Device
    parser.add_argument("--device_id", type=int, default=0)

    # Architecture
    parser.add_argument("--image_size",            type=int,   default=16)
    parser.add_argument("--in_channels",           type=int,   default=1)
    parser.add_argument("--out_channels",          type=int,   default=1)
    parser.add_argument("--model_channels",        type=int,   default=128)
    parser.add_argument("--num_res_blocks",        type=int,   default=2)
    parser.add_argument("--channel_mult",          type=int,   nargs='+', default=[1, 2, 2, 4])
    parser.add_argument("--attention_resolutions", type=int,   nargs='+', default=[1, 2, 4])
    parser.add_argument("--dropout",               type=float, default=0.0)
    parser.add_argument("--dims",                  type=int,   default=2)
    parser.add_argument("--conv_resample",         action="store_true", default=True)

    # Attention
    parser.add_argument("--num_heads",              type=int,  default=-1)
    parser.add_argument("--num_head_channels",      type=int,  default=8)
    parser.add_argument("--num_heads_upsample",     type=int,  default=-1)
    parser.add_argument("--use_spatial_transformer",action="store_true", default=True)
    parser.add_argument("--transformer_depth",      type=int,  default=1)
    parser.add_argument("--context_dim",            type=int,  default=512)
    parser.add_argument("--use_new_attention_order",action="store_true", default=False)
    parser.add_argument("--legacy",                 action="store_true", default=True)

    # ResBlock options
    parser.add_argument("--use_scale_shift_norm",   action="store_true", default=False)
    parser.add_argument("--resblock_updown",        action="store_true", default=False)

    # Misc
    parser.add_argument("--num_classes",  type=int,  default=None)
    parser.add_argument("--n_embed",      type=int,  default=None)
    parser.add_argument("--use_checkpoint",action="store_true", default=False)
    parser.add_argument("--use_fp16",     action="store_true", default=False)

    # Debug
    parser.add_argument("--write_json",   action="store_true", default=False)

    return parser.parse_args()


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    import torch

    args = parse_args()

    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Build model
    model = UNetModel(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params : {n_params:,}")

    # ── Cross-attention test (with BioBERT-style context) ─────────────────
    B   = 2
    x   = torch.randn(B, args.in_channels, args.image_size, args.image_size, device=device)
    t   = torch.randint(0, 1000, (B,), device=device)
    ctx = torch.randn(B, 16, args.context_dim, device=device)   # [B, seq, context_dim]

    out = model(x, timesteps=t, context=ctx)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    assert out.shape == x.shape, f"Shape mismatch: {x.shape} vs {out.shape}"

    # ── Unconditional test (only valid when use_spatial_transformer=False) ─
    # When use_spatial_transformer=True + context_dim is set, context=None
    # causes a shape error in SpatialTransformer's k/v projection.
    # Pass a dummy zero context instead.
    ctx_null = torch.zeros(B, 1, args.context_dim, device=device)
    out2 = model(x, timesteps=t, context=ctx_null)
    print(f"Null-ctx output: {out2.shape}")

    print("\nDebug run completed successfully!")
