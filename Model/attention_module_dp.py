"""
Model/attention_module_dp.py  ?  DP-Compatible Attention Modules
=================================================================
Identical to attention_module.py with one critical change for Opacus DP-SGD:

    BasicTransformerBlock.__init__: checkpoint=False  (was True)

WHY gradient checkpointing must be disabled for DP-SGD:
    Opacus computes per-sample gradients by attaching forward hooks to each
    nn.Module. These hooks record the per-sample input activations, which are
    then used during backward to compute grad_sample[B, ...] for each param.

    Gradient checkpointing (util_network.CheckpointFunction) re-computes
    activations during backward via a custom autograd.Function. This re-run
    bypasses Opacus's forward hooks ¡æ the recorded activations are stale/wrong
    ¡æ per-sample gradients are computed incorrectly ¡æ privacy guarantee broken.

    Setting checkpoint=False makes BasicTransformerBlock call _forward()
    directly, keeping all activations in the normal autograd tape so Opacus
    hooks fire correctly.

Usage options:
    A) Build new UNet with DP-compatible SpatialTransformer:
           from attention_module_dp import SpatialTransformer   # drop-in
    B) Patch existing UNet built with attention_module.py:
           from attention_module_dp import disable_checkpointing
           n = disable_checkpointing(ldm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

from Model.util_network import default, exists, normalization, zero_module, checkpoint


# =============================================================================
# Building blocks  (unchanged from attention_module.py)
# =============================================================================

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim  = int(dim * mult)
        dim_out    = default(dim_out, dim)
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu else GEGLU(dim, inner_dim)
        )
        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim   = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale  = dim_head ** -0.5
        self.heads  = heads
        self.to_q   = nn.Linear(query_dim,   inner_dim, bias=False)
        self.to_k   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v   = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h   = self.heads
        q   = self.to_q(x)
        ctx = default(context, x)
        k   = self.to_k(ctx)
        v   = self.to_v(ctx)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v)
        )
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask    = rearrange(mask, 'b ... -> b (...)')
            max_neg = -torch.finfo(sim.dtype).max
            mask    = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg)
        attn = sim.softmax(dim=-1)
        out  = einsum('b i j, b j d -> b i d', attn, v)
        out  = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


# =============================================================================
# BasicTransformerBlock  ?  checkpoint=False by default
# =============================================================================

class BasicTransformerBlock(nn.Module):
    """
    Self-attention (attn1) + Cross-attention (attn2) + FFN.

    DP change vs attention_module.py:
        checkpoint default changed False -> disables CheckpointFunction which
        is incompatible with Opacus per-sample gradient hooks.
    """

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=False):          #  False (was True)
        super().__init__()
        self.attn1 = CrossAttention(
            query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.attn2 = CrossAttention(
            query_dim=dim, context_dim=context_dim,
            heads=n_heads, dim_head=d_head, dropout=dropout
        )
        self.ff    = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint # bool type

    def forward(self, x, context=None):
        if self.checkpoint:
            # gradient checkpointing path -- NOT compatible with Opacus
            return checkpoint(self._forward, (x, context), self.parameters(), True) # call util_network.checkpoint()
        return self._forward(x, context)       # direct call for DP-SGD

    def _forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


# =============================================================================
# SpatialTransformer  ?  instantiates BasicTransformerBlock with checkpoint=False
# =============================================================================

class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data (B, C, H, W).
    Projects to inner_dim, applies BasicTransformerBlock(s), projects back.
    Uses checkpoint=False in all BasicTransformerBlock for Opacus compatibility.
    """

    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim        = n_heads * d_head
        self.norm        = normalization(in_channels)
        self.proj_in     = nn.Conv2d(in_channels, inner_dim,
                                     kernel_size=1, stride=1, padding=0)
        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(
                inner_dim, n_heads, d_head,
                dropout=dropout, context_dim=context_dim,
                checkpoint=False,                          # ¡ç explicit False
            )
            for _ in range(depth)
        ])
        self.proj_out = zero_module(
            nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        x_in = x
        x    = self.norm(x)
        x    = self.proj_in(x)
        x    = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


# =============================================================================
# Runtime patch helper
# =============================================================================

def disable_checkpointing(model: nn.Module) -> int:
    """
    Set BasicTransformerBlock.checkpoint=False on every block in model.

    Use this when the UNet was already built with the original
    attention_module.py (which defaults checkpoint=True) and you need to make
    it Opacus-compatible without rebuilding.

    Args:
        model : any nn.Module (typically LatentDiffusionDP or UNetModel)
    Returns:
        Number of BasicTransformerBlock instances patched.
    """
    count = 0
    for module in model.modules():
        if type(module).__name__ == 'BasicTransformerBlock':
            if getattr(module, 'checkpoint', False):
                module.checkpoint = False # --> BasicTransformerBlock.checkpoint = False -> don't call ckpt utils
                count += 1
    print(f'[disable_checkpointing] patched {count} BasicTransformerBlock(s) '
          f'(checkpoint -> False)')
    return count