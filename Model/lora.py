"""
Model/lora.py  –  LoRA adapters for DP-SGD fine-tuning of cross-attention
=========================================================================
Low-Rank Adaptation (LoRA) for the cross-attention Linear layers of the UNet.
Only the small A/B matrices are trained under DP-SGD; the frozen base weight
W0 is untouched.  This drastically reduces the number of trainable parameters,
which improves the DP privacy–utility trade-off (less noise spread over fewer
dimensions) and lets us store one small adapter file per privacy budget.

Design (docs/DP_LDM_FRAMEWORK_PLAN.md §5):
    LoRALinear(x) = W0(x) + (alpha / r) * B(A(x))
    - W0 : original nn.Linear, frozen (requires_grad=False)
    - A  : Linear(in_features, r, bias=False)   – init: kaiming/normal
    - B  : Linear(r, out_features, bias=False)   – init: ZERO  (so ΔW=0 at start)
    - scaling = alpha / r

Opacus compatibility:
    A and B are plain nn.Linear, so GradSampleModule computes their per-sample
    gradients correctly.  W0 has requires_grad=False, so no hook is attached.

Typical use:
    from Model.lora import inject_lora_cross_attention, lora_parameters
    n = inject_lora_cross_attention(ldm.model, rank=4, alpha=4.0)
    params = lora_parameters(ldm)            # only A/B tensors
    # ... Opacus make_private(module=ldm, optimizer=AdamW(params), ...)
    sd = lora_state_dict(ldm)                # save only adapters
    load_lora_state_dict(ldm, sd)            # load into an injected model
"""

import math

import torch
import torch.nn as nn


# Default cross-attention Linear attribute names to adapt.
DEFAULT_TARGETS = ('to_q', 'to_k', 'to_v', 'to_out')


# =============================================================================
# LoRALinear
# =============================================================================

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds a trainable low-rank update.

    Args:
        base  : the original nn.Linear to wrap (its weights are frozen)
        rank  : LoRA rank r (>0)
        alpha : LoRA scaling numerator; effective scale = alpha / rank
        dropout : dropout on the LoRA branch input (0 = none)
    """

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 4.0,
                 dropout: float = 0.0):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f'LoRALinear expects nn.Linear, got {type(base)}')
        if rank <= 0:
            raise ValueError(f'rank must be > 0, got {rank}')

        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False        # freeze W0 (and bias if present)

        in_features  = base.in_features
        out_features = base.out_features
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank

        # A: in→r, B: r→out.  B zero-init ⇒ ΔW = 0 at initialization.
        self.lora_A  = nn.Linear(in_features, rank, bias=False)
        self.lora_B  = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # Place the adapters on the SAME device/dtype as the frozen base layer.
        # The base UNet is already on its device when LoRA is injected; newly
        # created Linear layers default to CPU/float32 and would otherwise cause
        # a device mismatch (especially under model parallelism, where each base
        # layer may live on a different GPU).
        self.lora_A.to(base.weight.device, base.weight.dtype)
        self.lora_B.to(base.weight.device, base.weight.dtype)

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + self.scaling * self.lora_B(self.lora_A(self.lora_dropout(x)))

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """
        Return a plain nn.Linear whose weight = W0 + scaling * B @ A (for
        inference without the LoRA branch).  Bias is copied from the base.
        """
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        merged.to(self.base.weight.device, self.base.weight.dtype)
        delta = self.scaling * (self.lora_B.weight @ self.lora_A.weight)
        merged.weight.copy_(self.base.weight + delta)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged

    def extra_repr(self):
        return (f'in={self.base.in_features}, out={self.base.out_features}, '
                f'rank={self.rank}, alpha={self.alpha}')


# =============================================================================
# Injection
# =============================================================================

def _iter_cross_attention(model: nn.Module):
    """Yield modules whose class name is 'CrossAttention'."""
    for m in model.modules():
        if type(m).__name__ == 'CrossAttention':
            yield m


def _wrap_attr(module: nn.Module, attr: str, rank, alpha, dropout) -> int:
    """
    Replace module.<attr> (an nn.Linear, or a Sequential whose [0] is Linear,
    e.g. to_out) with a LoRALinear.  Returns number of layers wrapped (0/1).
    """
    target = getattr(module, attr, None)
    if isinstance(target, nn.Linear):
        setattr(module, attr, LoRALinear(target, rank, alpha, dropout))
        return 1
    # to_out is nn.Sequential(Linear, Dropout)
    if isinstance(target, nn.Sequential) and len(target) > 0 \
            and isinstance(target[0], nn.Linear):
        target[0] = LoRALinear(target[0], rank, alpha, dropout)
        return 1
    return 0


def inject_lora_cross_attention(model: nn.Module,
                                rank: int = 4,
                                alpha: float = 4.0,
                                dropout: float = 0.0,
                                targets=DEFAULT_TARGETS) -> int:
    """
    Inject LoRA into every CrossAttention module's target Linear layers.

    Args:
        model   : root module (e.g. ldm.model, the UNet)
        rank    : LoRA rank
        alpha   : LoRA scaling numerator (scale = alpha / rank)
        dropout : LoRA-branch dropout
        targets : attribute names to adapt (default to_q/to_k/to_v/to_out)

    Returns:
        number of Linear layers wrapped
    """
    count = 0
    n_attn = 0
    for attn in _iter_cross_attention(model):
        n_attn += 1
        for attr in targets:
            count += _wrap_attr(attn, attr, rank, alpha, dropout)
    print(f'[lora] injected into {count} Linear layer(s) across {n_attn} '
          f'CrossAttention module(s)  (rank={rank}, alpha={alpha})')
    return count


# =============================================================================
# Parameter / state helpers
# =============================================================================

def lora_parameters(model: nn.Module) -> list:
    """Return only the LoRA A/B parameters (for the optimizer)."""
    params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            params += list(m.lora_A.parameters())
            params += list(m.lora_B.parameters())
    return params


def lora_state_dict(model: nn.Module) -> dict:
    """
    Return a state_dict containing ONLY LoRA parameters, keyed by the full
    module path so it can be reloaded into an identically-injected model.
    """
    full = model.state_dict()
    return {k: v for k, v in full.items()
            if '.lora_A.' in k or '.lora_B.' in k
            or k.endswith('lora_A.weight') or k.endswith('lora_B.weight')}


def load_lora_state_dict(model: nn.Module, sd: dict, strict: bool = False):
    """
    Load a LoRA-only state_dict into an already-injected model.
    Returns (missing, unexpected) like nn.Module.load_state_dict.
    """
    return model.load_state_dict(sd, strict=strict)


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """
    Replace every LoRALinear with its merged plain nn.Linear (W0 + ΔW) for
    fast inference.  Returns the number of layers merged.

    Note: this mutates the model in place; keep a copy if you still need the
    separated adapters.
    """
    count = 0
    # Collect (parent, attr, lora) triples first to avoid mutating during walk.
    to_merge = []
    for parent in model.modules():
        for attr, child in list(parent._modules.items()):
            if isinstance(child, LoRALinear):
                to_merge.append((parent, attr, child))
    for parent, attr, lora in to_merge:
        parent._modules[attr] = lora.merged_linear()
        count += 1
    print(f'[lora] merged {count} LoRALinear layer(s) into plain Linear')
    return count
