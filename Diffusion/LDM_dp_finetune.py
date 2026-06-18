"""
LDM_dp_finetune.py  –  DP-SGD Fine-Tuning of LDM on MIMIC-CXR
================================================================
Fine-tunes only the cross-attention (SpatialTransformer) blocks of a
pre-trained LatentDiffusion model under (ε, δ)-differential privacy using
Opacus DP-SGD.

Expected directory layout (produced by Data/mimic_cxr.py prepare_split_dirs):
    <root_path>/
    ├── train/
    │   └── p10/ └── p10000032/ └── s50414267/ └── <id>.dcm
    │                            └── s50414267.txt
    ├── validate/
    └── test/

Key design notes:
  1. BioBERT lives INSIDE LatentDiffusionDP (not in collate_fn) so Opacus
     computes per-sample grads for its projection layer.
  2. Loader yields raw strings; embedding happens inside training_step_dp.
  3. Only SpatialTransformer blocks (+ optionally BioBERT proj) are trained;
     VAE is fully frozen.
  4. Gradient checkpointing is disabled (incompatible with Opacus hooks).
  5. scale_factor initialised BEFORE make_private() to avoid buffer issues.
  6. Checkpoint save/load uses model._module.state_dict() (GradSampleModule).
  7. Gradient accumulation via VirtualBatch chunk splitting:
       logical_batch → n_chunks = ceil(B / physical_batch) physical chunks
     Each chunk accumulates per-sample grads; one optimizer.step() per logical batch.

Usage (run from project root):
    python LDM_dp_finetune.py \\
        --pretrained_ckpt ./checkpoints/ldm_epoch0100.pt \\
        --root_path /storage/hjchoi/mimic/split \\
        --split train \\
        --target_epsilon 10.0 \\
        --target_delta 1e-5 \\
        --max_grad_norm 1.0 \\
        --logical_batch 256 \\
        --physical_batch 8 \\
        --epochs 30 \\
        --lr 1e-4 \\
        --save_dir ./checkpoints_dp
"""

import os
import sys
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ── Path setup ────────────────────────────────────────────────────────────────
# File lives at project root: <proj_root>/LDM_dp_finetune.py
_proj_root = os.path.dirname(os.path.abspath(__file__))

if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from Model.Diffusion.LDM_dp    import LatentDiffusionDP
from Model.Diffusion.UNetmodel  import UNetModel
from Model.VAE.Autoencoder      import VAE
from Model.attention_module_dp  import disable_checkpointing

from Modules.BioBERT_embedder   import BioBERTEmbedder

from Data.mimic_cxr import MIMICCXRDataset
from privacy.privacy_analysis import (
    compute_noise_multiplier,
    print_privacy_summary,
)


# =============================================================================
# Dataset helper
# =============================================================================

def _make_dataset(root_path: str, split: str,
                  image_size: int = 256,
                  max_length: int = 512) -> MIMICCXRDataset:
    """
    Construct a MIMICCXRDataset for the given split.

    Args:
        root_path  : root containing train/ validate/ test/ subdirectories
                     e.g. /storage/hjchoi/mimic/split
        split      : one of 'train', 'validate', 'test'
        image_size : resize target
        max_length : max report character length
    """
    valid_splits = ('train', 'validate', 'test')
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    ds_args = argparse.Namespace(
        prebuilt_split_dir = root_path,
        split              = split,
        image_size         = image_size,
        max_length         = max_length,
    )
    return MIMICCXRDataset(ds_args)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DP-SGD Fine-Tuning of LDM on MIMIC-CXR (pre-split dirs)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DEVICE -------------------------------------------------------------------
    parser.add_argument('--device_id', default='0',
                        help='GPU id (e.g. "0", "1")')

    # DATA (pre-split MIMIC-CXR) -----------------------------------------------
    parser.add_argument('--root_path', default='/storage/hjchoi/mimic/split',
                        help='Root containing train/ validate/ test/ subdirectories')
    parser.add_argument('--split',     default='train',
                        choices=['train', 'validate', 'test'],
                        help='Which split to use as training data')
    parser.add_argument('--do_validation', default=True,
                        type=lambda x: x.lower() != 'false',
                        help='Run validation loop after each epoch')
    parser.add_argument('--val_batches', default=-1, type=int,
                        help='Max validation batches per epoch (-1 = all)')
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--max_length',  default=512,  type=int,
                        help='Report text character limit')
    parser.add_argument('--num_workers', default=4,    type=int)
    parser.add_argument('--pin_memory',  default=True,
                        type=lambda x: x.lower() != 'false')

    # VAE: Encoder / Decoder ---------------------------------------------------
    parser.add_argument('--vae_img_size',         default=256,           type=int)
    parser.add_argument('--vae_in_ch',            default=1,             type=int)
    parser.add_argument('--vae_ch',               default=128,           type=int)
    parser.add_argument('--vae_ch_mult',          default=[1, 2, 4, 4, 4],
                        nargs='+', type=int)
    parser.add_argument('--vae_num_res_blocks',   default=2,             type=int)
    parser.add_argument('--vae_attn_resolutions', default=[32, 16],
                        nargs='+', type=int)
    parser.add_argument('--vae_dropout',          default=0.0,           type=float)
    parser.add_argument('--resamp_with_conv',     default=True,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--z_channels',           default=1,             type=int)
    parser.add_argument('--double_z',             default=True,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--vae_out_ch',           default=1,             type=int)
    parser.add_argument('--vae_ckpt',             default=None,
                        help='Path to pretrained VAE checkpoint (.pt / .pth)')

    # UNet ---------------------------------------------------------------------
    parser.add_argument('--unet_image_size',         default=16,         type=int,
                        help='Latent spatial size after VAE downsampling')
    parser.add_argument('--unet_in_ch',              default=1,          type=int)
    parser.add_argument('--unet_out_ch',             default=1,          type=int)
    parser.add_argument('--unet_ch',                 default=128,        type=int)
    parser.add_argument('--unet_num_res_blocks',     default=2,          type=int)
    parser.add_argument('--unet_ch_mult',            default=[1, 2, 4],
                        nargs='+', type=int)
    parser.add_argument('--unet_attn_resolutions',   default=[1, 2, 4],
                        nargs='+', type=int)
    parser.add_argument('--unet_dropout',            default=0.0,        type=float)
    parser.add_argument('--unet_num_heads',          default=-1,         type=int)
    parser.add_argument('--unet_num_head_channels',  default=8,          type=int)
    parser.add_argument('--use_spatial_transformer', default=True,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--transformer_depth',       default=1,          type=int)
    parser.add_argument('--context_dim',             default=512,        type=int)
    parser.add_argument('--legacy',                  default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--use_scale_shift_norm',    default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--resblock_updown',         default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--num_classes',  default=None, type=int)
    parser.add_argument('--n_embed',      default=None, type=int)
    parser.add_argument('--use_fp16',    default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--write_json',  default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--conv_dims',   default=2, type=int, choices=[1, 2, 3])

    # Diffusion / LDM ----------------------------------------------------------
    parser.add_argument('--timesteps',     default=1000,  type=int)
    parser.add_argument('--beta_schedule', default='linear')
    parser.add_argument('--scale_factor',  default=1.0,   type=float)
    parser.add_argument('--scale_by_std',  default=False, action='store_true',
                        help='Auto-compute scale_factor from first-batch latent std')
    parser.add_argument('--use_ema',       default=False,
                        type=lambda x: x.lower() != 'false',
                        help='EMA disabled by default for DP fine-tuning')

    # Pretrained LDM checkpoint ------------------------------------------------
    parser.add_argument('--pretrained_ckpt', default=None,
                        help='Pretrained LDM checkpoint to fine-tune from')

    # BioBERT ------------------------------------------------------------------
    parser.add_argument('--biobert_path', default='/storage/hjchoi',
                        help='HuggingFace model ID or local path for BioBERT')

    # DP-SGD -------------------------------------------------------------------
    parser.add_argument('--use_dp',           default=True,
                        type=lambda x: x.lower() != 'false',
                        help='Enable DP-SGD. Set --use_dp false for non-DP baseline.')
    parser.add_argument('--target_epsilon',   default=10.0,  type=float,
                        help='Target ε privacy budget')
    parser.add_argument('--target_delta',     default=1e-5,  type=float,
                        help='Target δ (recommend 1/dataset_size)')
    parser.add_argument('--max_grad_norm',    default=1.0,   type=float,
                        help='Per-sample gradient clipping norm C')
    parser.add_argument('--noise_multiplier', default=None,  type=float,
                        help='Override auto-computed σ (skips compute_noise_multiplier)')
    parser.add_argument('--logical_batch',    default=256,   type=int,
                        help='Logical batch size for privacy accounting')
    parser.add_argument('--physical_batch',   default=8,     type=int,
                        help='Physical batch size that fits in GPU memory')
    parser.add_argument('--ablation_blocks',  default=-1,    type=int,
                        help='-1 = all SpatialTransformer blocks; '
                             'N = only last N blocks (DP-LDM ablation study)')
    parser.add_argument('--finetune_biobert', default=True,
                        type=lambda x: x.lower() != 'false',
                        help='Unfreeze BioBERT projection layer under DP')

    # Training -----------------------------------------------------------------
    parser.add_argument('--epochs',         default=30,              type=int)
    parser.add_argument('--lr',             default=2e-5,            type=float,
                        help='Peak learning rate. DP-SGD typically needs 2x-5x lower '
                             'than non-private training (e.g. non-DP: 1e-4 → DP: 2e-5~5e-5)')
    parser.add_argument('--warmup_steps',   default=500,             type=int,
                        help='Linear warmup steps (0 = no warmup)')
    parser.add_argument('--scheduler_type', default='cosine_warmup',
                        choices=['cosine_warmup', 'linear_warmup', 'none'],
                        help='LR scheduler: cosine_warmup = warmup + cosine decay, '
                             'linear_warmup = warmup + linear decay, '
                             'none = constant LR')
    parser.add_argument('--save_dir',    default='./finetune_dp')
    parser.add_argument('--save_every',  default=5,     type=int,
                        help='Save DP checkpoint every N epochs')
    parser.add_argument('--log_every',   default=50,    type=int,
                        help='Print training loss every N logical steps')
    parser.add_argument('--resume_ckpt', default=None,
                        help='DP checkpoint to resume from (saved by this script)')

    return parser.parse_args()


# =============================================================================
# Model build helpers
# =============================================================================

def build_vae(args) -> VAE:
    vae_cfg = argparse.Namespace(
        in_channels      = args.vae_in_ch,
        out_channels     = args.vae_out_ch,
        ch               = args.vae_ch,
        ch_mult          = args.vae_ch_mult,
        num_res_blocks   = args.vae_num_res_blocks,
        attn_resolutions = args.vae_attn_resolutions,
        dropout          = args.vae_dropout,
        resamp_with_conv = args.resamp_with_conv,
        resolution       = args.vae_img_size,
        z_channels       = args.z_channels,
        double_z         = args.double_z,
        dims             = args.conv_dims,
    )
    return VAE(vae_cfg)


def build_unet(args) -> UNetModel:
    unet_cfg = argparse.Namespace(
        image_size             = args.unet_image_size,
        in_channels            = args.unet_in_ch,
        out_channels           = args.unet_out_ch,
        model_channels         = args.unet_ch,
        num_res_blocks         = args.unet_num_res_blocks,
        channel_mult           = args.unet_ch_mult,
        attention_resolutions  = args.unet_attn_resolutions,
        dropout                = args.unet_dropout,
        dims                   = args.conv_dims,
        conv_resample          = True,
        num_heads              = args.unet_num_heads,
        num_head_channels      = args.unet_num_head_channels,
        num_heads_upsample     = -1,
        use_spatial_transformer= args.use_spatial_transformer,
        transformer_depth      = args.transformer_depth,
        context_dim            = args.context_dim,
        use_new_attention_order= False,
        legacy                 = args.legacy,
        use_scale_shift_norm   = args.use_scale_shift_norm,
        resblock_updown        = args.resblock_updown,
        num_classes            = args.num_classes,
        n_embed                = args.n_embed,
        use_checkpoint         = False,     # must be False for Opacus
        use_fp16               = args.use_fp16,
        write_json             = args.write_json,
    )
    return UNetModel(unet_cfg)


def _load_vae_ckpt(vae: VAE, ckpt_path: str, device):
    raw = torch.load(ckpt_path, map_location=device)
    for key in ('state_dict', 'model', 'model_state_dict', 'net', 'weights'):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            print(f'  [VAE ckpt] using raw["{key}"]')
            break
    model_keys = list(vae.state_dict().keys())
    for ck in list(raw.keys()):
        for mk in model_keys[:5]:
            if ck.endswith(mk):
                prefix = ck[: len(ck) - len(mk)]
                if prefix:
                    stripped = {k[len(prefix):]: v
                                for k, v in raw.items() if k.startswith(prefix)}
                    if len(stripped) >= len(model_keys) * 0.8:
                        raw = stripped
                        print(f'  [VAE ckpt] stripped prefix "{prefix}"')
                break
    missing, unexpected = vae.load_state_dict(raw, strict=False)
    print(f'  [VAE ckpt] missing={len(missing)}  unexpected={len(unexpected)}')
    if len(missing) > len(model_keys) * 0.1:
        print('  [WARNING] >10% VAE keys missing – checkpoint may be incompatible')


# =============================================================================
# DataLoaders  (raw strings – NO pre-embedding, BioBERT runs inside model)
# =============================================================================

def _raw_collate_fn(samples):
    """Collate into {'image': Tensor[B,1,H,W], 'reports': list[str]}."""
    images, reports = zip(*samples)
    return {
        'image'  : torch.stack(images),
        'reports': list(reports),
    }


def build_dp_loader(dataset: Dataset, logical_batch: int,
                    num_workers: int = 4, pin_memory: bool = True) -> DataLoader:
    """
    Training DataLoader passed to Opacus make_private().

    batch_size = logical_batch (e.g. 256) so Opacus Poisson-samples at
    q = logical_batch / N.  This must match the q used in compute_noise_multiplier.
    Physical splitting into smaller GPU-sized chunks is done in the training loop.

    Mirrors DP-LDM's VirtualBatchWrapper approach:
        config batch_size=256 (accounting unit) + max_batch_size=3 (GPU unit)
    """
    return DataLoader(
        dataset,
        batch_size  = logical_batch,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = True,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        collate_fn  = _raw_collate_fn,
    )


def build_val_loader(dataset: Dataset, batch_size: int,
                     num_workers: int = 4) -> DataLoader:
    """Validation DataLoader (no DP, shuffle=False, no drop_last)."""
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = num_workers,
        drop_last   = False,
        pin_memory  = False,
        collate_fn  = _raw_collate_fn,
    )


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def evaluate(ldm, val_loader: DataLoader, device, max_batches: int = -1) -> float:
    inner = ldm._module if hasattr(ldm, '_module') else ldm
    inner.eval()

    losses = []
    for i, batch in enumerate(val_loader):
        if max_batches > 0 and i >= max_batches:
            break
        batch = {'image': batch['image'].to(device), 'reports': batch['reports']}
        loss, _ = inner.training_step_dp(batch)
        losses.append(loss.item())

    inner.train()
    return float(np.mean(losses)) if losses else float('nan')


# =============================================================================
# Checkpoint helpers  (GradSampleModule-aware)
# =============================================================================

def save_dp_checkpoint(save_path, model, optimizer, privacy_engine,
                       epoch, global_step, args, target_delta):
    eps_spent  = (privacy_engine.get_epsilon(target_delta)
                  if privacy_engine is not None else None)
    model_sd   = (model._module.state_dict()
                  if hasattr(model, '_module') else model.state_dict())
    opt_sd     = (optimizer.original_optimizer.state_dict()
                  if hasattr(optimizer, 'original_optimizer') else optimizer.state_dict())
    torch.save({
        'epoch'         : epoch,
        'global_step'   : global_step,
        'model'         : model_sd,
        'optimizer'     : opt_sd,
        'epsilon_spent' : eps_spent,
        'args'          : vars(args),
    }, save_path)
    eps_str = f'{eps_spent:.4f}' if eps_spent is not None else 'N/A (no DP)'
    print(f'  [ckpt] saved → {save_path}  (ε_spent={eps_str})')


def load_dp_checkpoint(load_path, model, optimizer=None, device='cpu'):
    ckpt   = torch.load(load_path, map_location=device)
    sd     = ckpt.get('model', ckpt)
    target = model._module if hasattr(model, '_module') else model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f'[load_dp_checkpoint] missing={len(missing)}  unexpected={len(unexpected)}  '
          f'ε_spent_at_save={ckpt.get("epsilon_spent", "N/A")}')

    if optimizer is not None and 'optimizer' in ckpt:
        opt_target = (optimizer.original_optimizer
                      if hasattr(optimizer, 'original_optimizer') else optimizer)
        opt_target.load_state_dict(ckpt['optimizer'])

    return ckpt.get('epoch', 0), ckpt.get('global_step', 0)


# =============================================================================
# LR Scheduler
# =============================================================================

def build_scheduler(optimizer, warmup_steps: int, total_steps: int,
                    scheduler_type: str = 'cosine_warmup'):
    """
    cosine_warmup : linear warmup → cosine decay to 0
    linear_warmup : linear warmup → linear decay to 0
    Both are suitable for DP-SGD; warmup stabilises early training
    before the clipping/noise regime settles.
    """
    from torch.optim.lr_scheduler import LambdaLR

    def _cosine(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _linear(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    fn = _cosine if scheduler_type == 'cosine_warmup' else _linear
    print(f'[Scheduler] type={scheduler_type}  warmup_steps={warmup_steps}  '
          f'total_steps={total_steps}')
    return LambdaLR(optimizer, fn)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device : {device}')
    print(f'Split root : {args.root_path}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = _make_dataset(
        root_path  = args.root_path,
        split      = args.split,
        image_size = args.image_size,
        max_length = args.max_length,
    )
    n_train = len(train_dataset)
    if n_train == 0:
        raise RuntimeError(
            f"No samples found in {os.path.join(args.root_path, args.split)}. "
            "Run Data/mimic_cxr.py --make_split_dir first."
        )

    val_dataset = None
    if args.do_validation:
        try:
            val_dataset = _make_dataset(
                root_path  = args.root_path,
                split      = 'validate',
                image_size = args.image_size,
                max_length = args.max_length,
            )
            print(f'[Validation] validate samples: {len(val_dataset)}')
        except FileNotFoundError as e:
            print(f'[Validation] WARNING: {e}  – validation disabled')
            val_dataset = None

    print(f'[{args.split}] samples={n_train}')

    # ── BioBERT Embedder ──────────────────────────────────────────────────────
    embedder = BioBERTEmbedder(
        model_path = args.biobert_path,
        max_length = args.max_length,
    ).to(device)
    print(f'[BioBERT] loaded: {args.biobert_path}')

    # ── VAE (always frozen) ───────────────────────────────────────────────────
    vae = build_vae(args).to(device)
    if args.vae_ckpt:
        print(f'[VAE] loading: {args.vae_ckpt}')
        _load_vae_ckpt(vae, args.vae_ckpt, device)
    else:
        print('[VAE] WARNING: no --vae_ckpt; using random weights')
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ── UNet ──────────────────────────────────────────────────────────────────
    unet = build_unet(args).to(device)
    print(f'[UNet] params: {sum(p.numel() for p in unet.parameters()):,}')

    # ── LatentDiffusionDP ─────────────────────────────────────────────────────
    ldm = LatentDiffusionDP(
        unet              = unet,
        first_stage_model = vae,
        embedder          = embedder,
        cond_stage_key    = 'context',
        first_stage_key   = 'image',
        conditioning_key  = 'crossattn',
        timesteps         = args.timesteps,
        beta_schedule     = args.beta_schedule,
        image_size        = args.unet_image_size,
        channels          = args.z_channels,
        scale_factor      = args.scale_factor,
        scale_by_std      = args.scale_by_std,
        use_ema           = args.use_ema,
        lr                = args.lr,
        use_dp            = args.use_dp,
    ).to(device)

    if args.pretrained_ckpt:
        print(f'[LDM_dp] loading pretrained: {args.pretrained_ckpt}')
        ldm.init_from_ckpt(args.pretrained_ckpt)
    else:
        print('[LDM_dp] WARNING: no --pretrained_ckpt; fine-tuning from scratch')

    n_patched = disable_checkpointing(ldm)
    print(f'[DP] gradient checkpointing disabled on {n_patched} block(s)')

    # ── DataLoaders ───────────────────────────────────────────────────────────
    # base_loader uses logical_batch so Opacus accounts at q = logical_batch/N
    # (same as DP-LDM config batch_size=256 passed to make_private)
    base_loader = build_dp_loader(
        train_dataset,
        logical_batch = args.logical_batch,
        num_workers   = args.num_workers,
        pin_memory    = args.pin_memory,
    )
    val_loader = (
        build_val_loader(val_dataset, args.physical_batch, args.num_workers)
        if val_dataset is not None else None
    )

    # DP-LDM 방식: sample_rate = 1 / len(dataloader) = logical_batch / N
    # steps = int(1/sample_rate) = len(dataloader) per epoch
    logical_steps_per_epoch = len(base_loader)                # = floor(N / logical_batch)
    total_logical_steps     = logical_steps_per_epoch * args.epochs
    sample_rate             = 1.0 / logical_steps_per_epoch   # q ≈ logical_batch / N
    approx_chunks           = math.ceil(args.logical_batch / args.physical_batch)
    print(f'[Loader] N={n_train}  logical_steps/epoch={logical_steps_per_epoch}  '
          f'sample_rate=1/{logical_steps_per_epoch}={sample_rate:.6f}  '
          f'physical_batch={args.physical_batch}  chunks/logical≈{approx_chunks}')

    # ── scale_factor init BEFORE make_private ─────────────────────────────────
    if args.scale_by_std:
        print('[LDM_dp] computing scale_factor from first batch ...')
        first_raw = next(iter(base_loader))
        # Use first physical chunk only (saves GPU memory)
        first_chunk = {
            'image'  : first_raw['image'][:args.physical_batch].to(device),
            'reports': first_raw['reports'][:args.physical_batch],
        }
        ldm.init_scale_factor(first_chunk, is_first_batch=True)

    # ── Selective parameter unfreeze ──────────────────────────────────────────
    attn_params = ldm.configure_dp_params(
        ablation_blocks  = args.ablation_blocks,
        finetune_biobert = args.finetune_biobert,
    )
    if not attn_params:
        raise RuntimeError(
            'configure_dp_params returned empty param list. '
            'Check UNet has SpatialTransformer blocks and --use_spatial_transformer=True.'
        )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(attn_params, lr=args.lr)

    # ── DP-SGD setup (skipped when --use_dp false) ────────────────────────────
    if args.use_dp:
        if args.noise_multiplier is not None:
            sigma = args.noise_multiplier
            print(f'[DP] using provided noise_multiplier σ={sigma:.6f}')
        else:
            sigma = compute_noise_multiplier(
                target_epsilon = args.target_epsilon,
                target_delta   = args.target_delta,
                sample_rate    = sample_rate,
                epochs         = args.epochs,
            )

        print_privacy_summary(
            noise_multiplier = sigma,
            max_grad_norm    = args.max_grad_norm,
            sample_rate      = sample_rate,
            epochs           = args.epochs,
            target_epsilon   = args.target_epsilon,
            target_delta     = args.target_delta,
        )

        try:
            from opacus import PrivacyEngine
            from opacus.grad_sample import GradSampleModule
        except ImportError:
            raise ImportError('opacus is required. Install with: pip install opacus')

        # Unwrap if model was already wrapped by a previous make_private call
        # (happens when PyCharm reuses the Python process between runs)
        if isinstance(ldm, GradSampleModule):
            print('[DP] WARNING: model already wrapped – unwrapping before make_private')
            ldm = ldm._module

        privacy_engine = PrivacyEngine()
        ldm, optimizer, dp_loader = privacy_engine.make_private(
            module           = ldm,
            optimizer        = optimizer,
            data_loader      = base_loader,
            noise_multiplier = sigma,
            max_grad_norm    = args.max_grad_norm,
            poisson_sampling = True,
        )
        print(f'[Opacus] GradSampleModule ready  σ={sigma:.6f}  C={args.max_grad_norm}')
    else:
        sigma          = None
        privacy_engine = None
        dp_loader      = base_loader
        print('[DP] DP disabled – running standard fine-tuning (no Opacus)')

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    if args.scheduler_type == 'none':
        scheduler = None
        print('[Scheduler] disabled (constant LR)')
    else:
        scheduler = build_scheduler(
            optimizer,
            warmup_steps   = args.warmup_steps,
            total_steps    = total_logical_steps,
            scheduler_type = args.scheduler_type,
        )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume_ckpt:
        print(f'[resume] {args.resume_ckpt}')
        start_epoch, global_step = load_dp_checkpoint(
            args.resume_ckpt, ldm, optimizer, device=str(device)
        )
        print(f'  resumed: epoch={start_epoch}  global_step={global_step}')

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Inner model reference (unwrap GradSampleModule when DP is on) ─────────
    inner_model = ldm._module if args.use_dp else ldm

    # =========================================================================
    # Training Loop
    # =========================================================================
    sigma_str = f'σ={sigma:.4f}' if sigma is not None else 'no-DP'
    print(f'\n[DP-Train] split={args.split}  epochs={args.epochs}  lr={args.lr}  '
          f'logical_batch={args.logical_batch}  physical_batch={args.physical_batch}  '
          f'chunks/step≈{approx_chunks}  {sigma_str}')
    print('=' * 60)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        ldm.train()
        epoch_losses = []
        t0           = time.time()

        for logical_batch in dp_loader:
            images  = logical_batch['image'].to(device)   # [B, 1, H, W]
            reports = logical_batch['reports']             # list[str], len ≈ B

            B        = images.shape[0]
            n_chunks = max(1, math.ceil(B / args.physical_batch))

            optimizer.zero_grad()
            step_loss      = 0.0
            last_loss_dict = {}

            # ── VirtualBatch: split logical batch into physical chunks ─────────
            for i in range(n_chunks):
                start = i * args.physical_batch
                end   = min(start + args.physical_batch, B)
                chunk = {
                    'image'  : images[start:end],
                    'reports': reports[start:end],
                }
                loss, loss_dict = inner_model.training_step_dp(chunk)
                (loss / n_chunks).backward()
                step_loss     += loss.item() / n_chunks
                last_loss_dict = loss_dict

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_losses.append(step_loss)
            global_step += 1

            if global_step % args.log_every == 0:
                info = '  '.join(f'{k}={v.item():.4f}'
                                 for k, v in last_loss_dict.items())
                if args.use_dp:
                    eps_now = privacy_engine.get_epsilon(args.target_delta)
                    print(f'  ep {epoch+1:04d}  step {global_step:06d} | '
                          f'{info}  ε={eps_now:.4f}')
                else:
                    print(f'  ep {epoch+1:04d}  step {global_step:06d} | {info}')

        # ── Epoch summary ─────────────────────────────────────────────────────
        elapsed   = time.time() - t0
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')

        val_str = ''
        if val_loader is not None:
            val_loss = evaluate(ldm, val_loader, device, args.val_batches)
            val_str  = f'  val_loss={val_loss:.4f}'

        if args.use_dp:
            eps_now  = privacy_engine.get_epsilon(args.target_delta)
            dp_str   = f'  ε={eps_now:.4f}  δ={args.target_delta}'
            if eps_now > args.target_epsilon * 1.05:
                print(f'[WARNING] ε_spent={eps_now:.4f} exceeds target '
                      f'ε={args.target_epsilon}. Consider early stopping.')
        else:
            dp_str = ''

        print(f'[Epoch {epoch+1:04d}/{start_epoch+args.epochs}] '
              f'loss={mean_loss:.4f}{val_str}{dp_str}  '
              f'time={elapsed:.1f}s')

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f'ldm_dp_epoch{epoch+1:04d}.pt')
            save_dp_checkpoint(
                save_path      = ckpt_path,
                model          = ldm,
                optimizer      = optimizer,
                privacy_engine = privacy_engine,
                epoch          = epoch + 1,
                global_step    = global_step,
                args           = args,
                target_delta   = args.target_delta,
            )

    # ── Final checkpoint ──────────────────────────────────────────────────────
    final_path = os.path.join(args.save_dir, 'ldm_dp_final.pt')
    save_dp_checkpoint(
        save_path      = final_path,
        model          = ldm,
        optimizer      = optimizer,
        privacy_engine = privacy_engine,
        epoch          = start_epoch + args.epochs,
        global_step    = global_step,
        args           = args,
        target_delta   = args.target_delta,
    )
    eps_final = privacy_engine.get_epsilon(args.target_delta)
    print(f'\nTraining complete.  Final ε={eps_final:.4f}  δ={args.target_delta}')


if __name__ == '__main__':
    main()
