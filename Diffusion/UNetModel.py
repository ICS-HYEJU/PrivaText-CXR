"""
Diffusion/UNetModel.py  –  UNet with timestep + cross-attention conditioning
=============================================================================

All building blocks are reused from the existing Model/ directory:
    ResBlock, Upsample, Downsample  ←  Model/UNetBlock_module.py
    SpatialTransformer              ←  Model/attention_module.py
    TimestepBlock,
    TimestepEmbedSequential         ←  Model/timestep_block.py
    util helpers                    ←  Model/util_network.py

AttentionBlock (plain self-attention, used when use_spatial_transformer=False)
is implemented locally since it is not part of the project's block library.

Config example (from YAML):
    image_size: 16
    in_channels: 3
    out_channels: 3
    model_channels: 128
    attention_resolutions: [1, 2, 4]
    num_res_blocks: 2
    channel_mult: [1, 2, 2, 4]
    num_head_channels: 8
    use_spatial_transformer: true
    transformer_depth: 1
    context_dim: 512
"""

import os
import sys
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_model_dir = os.path.join(_this_dir, '..', 'Model')
for _d in [_model_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from util_network import (                       # noqa: E402
    conv_nd, linear, normalization,
    zero_module, checkpoint, timestep_embedding,
)
from UNetBlock_module import ResBlock, Upsample, Downsample   # noqa: E402
from timestep_block import TimestepBlock, TimestepEmbedSequential  # noqa: E402
from attention_module import SpatialTransformer  # noqa: E402


# =============================================================================
# AttentionBlock  (plain self-attention; fallback when use_spatial_transformer=False)
# =============================================================================

class QKVAttentionLegacy(nn.Module):
    """Multi-head QKV self-attention (legacy split-before-heads layout)."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: th.Tensor) -> th.Tensor:
        """
        qkv: [B, 3*C, T]  (C = n_heads * head_dim)
        returns: [B, C, T]
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale   = 1.0 / math.sqrt(ch)
        weight  = th.einsum('bct,bcs->bts', q * scale, k * scale)
        weight  = th.softmax(weight.float(), dim=-1).to(weight.dtype)
        out     = th.einsum('bts,bcs->bct', weight, v)
        return out.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    Spatial self-attention block (no cross-attention context).
    Used as a drop-in replacement for SpatialTransformer when
    use_spatial_transformer=False.
    """

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
            assert channels % num_head_channels == 0, \
                f"channels ({channels}) must be divisible by num_head_channels ({num_head_channels})"
            self.num_heads = channels // num_head_channels

        self.norm  = normalization(channels)
        self.qkv   = conv_nd(1, channels, channels * 3, 1)
        self.attn  = QKVAttentionLegacy(self.num_heads)
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

    Args:
        image_size:              Spatial size of latent input (e.g. 16).
        in_channels:             Input channels (e.g. 3 or 4 for latent space).
        model_channels:          Base channel count. (e.g. 128)
        out_channels:            Output channels. Usually == in_channels.
        num_res_blocks:          ResBlocks per resolution level.
        attention_resolutions:   Downsampling factors at which attention is added.
        dropout:                 Dropout probability.
        channel_mult:            Per-level channel multiplier tuple.
        conv_resample:           Use learned conv for up/downsampling.
        dims:                    1 / 2 / 3-D convolutions.
        num_classes:             If set, enables class-conditional generation (y label).
        use_checkpoint:          Gradient checkpointing to save memory.
        use_fp16:                Store activations as float16.
        num_heads:               Attention heads (-1 → derive from num_head_channels).
        num_head_channels:       Fixed head width (-1 → derive from num_heads).
        num_heads_upsample:      Attention heads in the decoder (-1 → same as num_heads).
        use_scale_shift_norm:    FiLM-style scale+shift conditioning in ResBlocks.
        resblock_updown:         Use ResBlock (not conv) for up/downsampling.
        use_spatial_transformer: Use SpatialTransformer instead of AttentionBlock.
        transformer_depth:       Transformer layers per SpatialTransformer block.
        context_dim:             Cross-attention context dimension.
        n_embed:                 If set, adds a codebook-id prediction head.
        legacy:                  Legacy head-dimension calculation (default True).
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout                 = 0,
        channel_mult            = (1, 2, 4, 8),
        conv_resample           = True,
        dims                    = 2,
        num_classes             = None,
        use_checkpoint          = False,
        use_fp16                = False,
        num_heads               = -1,
        num_head_channels       = -1,
        num_heads_upsample      = -1,
        use_scale_shift_norm    = False,
        resblock_updown         = False,
        use_new_attention_order = False,
        use_spatial_transformer = False,
        transformer_depth       = 1,
        context_dim             = None,
        n_embed                 = None,
        legacy                  = True,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert context_dim is not None, \
                "context_dim must be set when use_spatial_transformer=True"
        if context_dim is not None:
            assert use_spatial_transformer, \
                "use_spatial_transformer must be True when context_dim is set"
            # omegaconf ListConfig → plain list
            try:
                from omegaconf.listconfig import ListConfig
                if isinstance(context_dim, ListConfig):
                    context_dim = list(context_dim)
            except ImportError:
                pass

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        assert not (num_heads == -1 and num_head_channels == -1), \
            "Set either num_heads or num_head_channels"

        self.image_size          = image_size
        self.in_channels         = in_channels
        self.model_channels      = model_channels
        self.out_channels        = out_channels
        self.num_res_blocks      = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout             = dropout
        self.channel_mult        = channel_mult
        self.conv_resample       = conv_resample
        self.num_classes         = num_classes
        self.use_checkpoint      = use_checkpoint
        self.dtype               = th.float16 if use_fp16 else th.float32
        self.num_heads           = num_heads
        self.num_head_channels   = num_head_channels
        self.num_heads_upsample  = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ── helper: build one attention layer ──────────────────────────────
        def _attn(ch, heads, dim_head):
            if use_spatial_transformer:
                return SpatialTransformer(
                    ch, heads, dim_head,
                    depth=transformer_depth, context_dim=context_dim,
                )
            else:
                return AttentionBlock(
                    ch,
                    num_heads=heads,
                    num_head_channels=dim_head,
                    use_checkpoint=use_checkpoint,
                    use_new_attention_order=use_new_attention_order,
                )

        def _head_params(ch, _num_heads, _num_head_channels):
            """Return (num_heads, dim_head) given channel count."""
            if _num_head_channels == -1:
                return _num_heads, ch // _num_heads
            else:
                h = ch // _num_head_channels
                d = _num_head_channels
                if legacy:
                    d = ch // h if use_spatial_transformer else _num_head_channels
                return h, d

        # ── Input (encoder) blocks ──────────────────────────────────────────
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(
                conv_nd(dims, in_channels, model_channels, 3, padding=1)
            )
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    heads, dim_head = _head_params(ch, num_heads, num_head_channels)
                    layers.append(_attn(ch, heads, dim_head))

                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            if level != len(channel_mult) - 1:        # add downsampler
                out_ch = ch
                self.input_blocks.append(TimestepEmbedSequential(
                    ResBlock(ch, time_embed_dim, dropout,
                             out_channels=out_ch, dims=dims,
                             use_checkpoint=use_checkpoint,
                             use_scale_shift_norm=use_scale_shift_norm,
                             down=True)
                    if resblock_updown else
                    Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                ))
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # ── Middle block ────────────────────────────────────────────────────
        heads, dim_head = _head_params(ch, num_heads, num_head_channels)
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            _attn(ch, heads, dim_head),
            ResBlock(ch, time_embed_dim, dropout, dims=dims,
                     use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        # ── Output (decoder) blocks ─────────────────────────────────────────
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    heads, dim_head = _head_params(ch, num_heads, num_head_channels)
                    layers.append(_attn(ch, heads, dim_head))

                if level and i == num_res_blocks:    # add upsampler
                    out_ch = ch
                    layers.append(
                        ResBlock(ch, time_embed_dim, dropout,
                                 out_channels=out_ch, dims=dims,
                                 use_checkpoint=use_checkpoint,
                                 use_scale_shift_norm=use_scale_shift_norm,
                                 up=True)
                        if resblock_updown else
                        Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # ── Output head ─────────────────────────────────────────────────────
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1),
            )

    # ── Utility ──────────────────────────────────────────────────────────────

    def convert_to_fp16(self):
        self.input_blocks.apply(lambda m: m.half() if hasattr(m, 'half') else m)
        self.middle_block.apply(lambda m: m.half() if hasattr(m, 'half') else m)
        self.output_blocks.apply(lambda m: m.half() if hasattr(m, 'half') else m)

    def convert_to_fp32(self):
        self.input_blocks.apply(lambda m: m.float() if hasattr(m, 'float') else m)
        self.middle_block.apply(lambda m: m.float() if hasattr(m, 'float') else m)
        self.output_blocks.apply(lambda m: m.float() if hasattr(m, 'float') else m)

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        """
        Args:
            x:          [B, in_channels, H, W]
            timesteps:  [B]  long tensor of diffusion timesteps
            context:    [B, seq_len, context_dim]  cross-attention condition (optional)
            y:          [B]  class labels (only when num_classes is set)
        Returns:
            [B, out_channels, H, W]
        """
        assert (y is not None) == (self.num_classes is not None), \
            "y must be provided iff model is class-conditional"

        # Timestep embedding
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb   = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        # Encoder path
        hs = []
        h  = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)

        # Bottleneck
        h = self.middle_block(h, emb, context)

        # Decoder path
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.id_predictor(h) if self.predict_codebook_ids else self.out(h)


# =============================================================================
# Debug / __main__
# =============================================================================

if __name__ == '__main__':
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Config from YAML (use_spatial_transformer=True)
    model = UNetModel(
        image_size              = 16,
        in_channels             = 3,
        out_channels            = 3,
        model_channels          = 128,
        attention_resolutions   = [1, 2, 4],
        num_res_blocks          = 2,
        channel_mult            = [1, 2, 2, 4],
        num_head_channels       = 8,
        use_spatial_transformer = True,
        transformer_depth       = 1,
        context_dim             = 512,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {n_params:,}')

    B = 2
    x   = torch.randn(B, 3, 16, 16, device=device)
    t   = torch.randint(0, 1000, (B,), device=device)
    ctx = torch.randn(B, 77, 512, device=device)   # e.g. text/class embedding

    out = model(x, timesteps=t, context=ctx)
    print(f'Input : {x.shape}')
    print(f'Output: {out.shape}')
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    # Unconditional (AttentionBlock path)
    model_self_attn = UNetModel(
        image_size            = 16,
        in_channels           = 3,
        out_channels          = 3,
        model_channels        = 64,
        attention_resolutions = [1, 2],
        num_res_blocks        = 1,
        channel_mult          = [1, 2, 4],
        num_heads             = 4,
    ).to(device)

    out2 = model_self_attn(x, timesteps=t)
    print(f'Self-attn output: {out2.shape}')

    print('\nDebug run completed successfully!')
