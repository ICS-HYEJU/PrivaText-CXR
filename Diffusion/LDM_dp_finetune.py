"""
Diffusion/LDM_dp_finetune.py  –  DP-SGD Fine-Tuning of LDM on MIMIC-CXR
=========================================================================
Fine-tunes only the cross-attention (SpatialTransformer) blocks of a
pre-trained LatentDiffusion model under (ε, δ)-differential privacy using
Opacus DP-SGD.

Key differences vs LDM_train.py:
  1. BioBERT embedder lives INSIDE LatentDiffusionDP (not in collate_fn)
     so Opacus computes per-sample grads for its projection layer.
  2. Loader yields raw strings; embedding happens inside training_step_dp.
  3. Only SpatialTransformer blocks (+ optionally BioBERT proj) are trained;
     VAE is fully frozen.
  4. Gradient checkpointing is disabled (incompatible with Opacus hooks).
  5. scale_factor initialised BEFORE make_private() to avoid buffer issues.
  6. Checkpoint save/load uses model._module.state_dict() (GradSampleModule).
  7. Gradient accumulation via signal_skip_step:
       logical_batch = physical_batch × grad_accum_steps
     Privacy accounting is done at the logical-batch level.

Usage:
    python Diffusion/LDM_dp_finetune.py \\
        --pretrained_ckpt ./checkpoints/ldm_epoch0100.pt \\
        --root_path /data/mimic-cxr \\
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.normpath(os.path.join(_this_dir, '..'))
_model_dir = os.path.join(_proj_root, 'Model')
_data_dir  = os.path.join(_proj_root, 'Data')
for _d in [_proj_root, _model_dir, _data_dir, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from LDM_dp        import LatentDiffusionDP             # Diffusion/LDM_dp.py
from UNetModel     import UNetModel                     # Diffusion/UNetModel.py
from autoencoder   import AutoencoderKL                 # Model/autoencoder.py
from attention_module_dp import disable_checkpointing   # Model/attention_module_dp.py
from mimic_cxr     import MIMICCXRDataset               # Data/mimic_cxr.py
from privacy.privacy_analysis import (                  # Diffusion/privacy/
    compute_noise_multiplier,
    get_epsilon_spent,
    print_privacy_summary,
)

try:
    from Modules.BioBERT_embedder import BioBERTEmbedder
except ImportError:
    sys.path.insert(0, _proj_root)
    try:
        from Modules.BioBERT_embedder import BioBERTEmbedder
    except ImportError:
        BioBERTEmbedder = None


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DP-SGD Fine-Tuning of LDM on MIMIC-CXR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DEVICE -------------------------------------------------------------------
    parser.add_argument('--device_id', default='0',
                        help='GPU id (e.g. "0", "1")')

    # DATA (MIMIC-CXR) ---------------------------------------------------------
    parser.add_argument('--root_path',    required=True,
                        help='MIMIC-CXR dataset root directory')
    parser.add_argument('--split_csv',    default='mimic-cxr-2.0.0-split.csv')
    parser.add_argument('--meta_csv',     default='mimic-cxr-2.0.0-metadata.csv')
    parser.add_argument('--view_filter',  nargs='+', default=['PA', 'AP'],
                        help='ViewPosition filter for frontal CXRs')
    parser.add_argument('--image_size',   default=256,  type=int)
    parser.add_argument('--max_length',   default=512,  type=int,
                        help='BioBERT tokeniser max length')
    parser.add_argument('--num_workers',  default=4,    type=int)
    parser.add_argument('--pin_memory',   default=True,
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

    # VAE checkpoint (required – VAE must be pretrained) -----------------------
    parser.add_argument('--vae_ckpt', default=None,
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
    parser.add_argument('--conv_dims',   default=2,   type=int, choices=[1, 2, 3])

    # Diffusion / LDM ----------------------------------------------------------
    parser.add_argument('--timesteps',     default=1000,  type=int)
    parser.add_argument('--beta_schedule', default='linear')
    parser.add_argument('--scale_factor',  default=1.0,   type=float)
    parser.add_argument('--scale_by_std',  default=False, action='store_true',
                        help='Auto-compute scale_factor from first-batch latent std')
    parser.add_argument('--use_ema',       default=False,
                        type=lambda x: x.lower() != 'false',
                        help='EMA is disabled by default for DP fine-tuning')

    # Pretrained LDM checkpoint (required) -------------------------------------
    parser.add_argument('--pretrained_ckpt', default=None,
                        help='Pretrained LDM checkpoint to fine-tune from')

    # BioBERT ------------------------------------------------------------------
    parser.add_argument('--biobert_path', default='dmis-lab/biobert-base-cased-v1.2',
                        help='HuggingFace model ID or local path for BioBERT')

    # DP-SGD -------------------------------------------------------------------
    parser.add_argument('--target_epsilon',  default=10.0,  type=float,
                        help='Target ε privacy budget')
    parser.add_argument('--target_delta',    default=1e-5,  type=float,
                        help='Target δ (recommend 1/dataset_size)')
    parser.add_argument('--max_grad_norm',   default=1.0,   type=float,
                        help='Per-sample gradient clipping norm')
    parser.add_argument('--noise_multiplier', default=None, type=float,
                        help='Override auto-computed σ (skips compute_noise_multiplier)')
    parser.add_argument('--logical_batch',   default=256,   type=int,
                        help='Logical batch size for privacy accounting')
    parser.add_argument('--physical_batch',  default=8,     type=int,
                        help='Physical batch size (must divide logical_batch)')
    parser.add_argument('--ablation_blocks', default=-1,    type=int,
                        help='-1 = all SpatialTransformer blocks; '
                             'N = only last N blocks (DP-LDM ablation)')
    parser.add_argument('--finetune_biobert', default=True,
                        type=lambda x: x.lower() != 'false',
                        help='Unfreeze BioBERT projection layer under DP')

    # Training -----------------------------------------------------------------
    parser.add_argument('--epochs',      default=30,     type=int)
    parser.add_argument('--lr',          default=1e-4,   type=float)
    parser.add_argument('--warmup_steps',default=500,    type=int,
                        help='Linear warmup steps (0 = disabled)')
    parser.add_argument('--save_dir',    default='./checkpoints_dp')
    parser.add_argument('--save_every',  default=5,      type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every',   default=50,     type=int,
                        help='Log training loss every N logical steps')
    parser.add_argument('--resume_ckpt', default=None,
                        help='DP checkpoint to resume from (saved by this script)')

    return parser.parse_args()


# =============================================================================
# Build helpers
# =============================================================================

def build_vae(args) -> AutoencoderKL:
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
    return AutoencoderKL(vae_cfg)


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
        use_checkpoint         = False,     # always False for DP
        use_fp16               = args.use_fp16,
        write_json             = args.write_json,
    )
    return UNetModel(unet_cfg)


def _load_vae_ckpt(vae: AutoencoderKL, ckpt_path: str, device):
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
# DataLoader  (raw strings – NO pre-embedding)
# =============================================================================

def build_dp_loader(args, dataset: MIMICCXRDataset, batch_size: int) -> DataLoader:
    """
    Build a DataLoader whose batches are:
        {'image': Tensor[B,1,H,W], 'reports': list[str]}

    BioBERT embedding is intentionally NOT done here; it must happen inside
    LatentDiffusionDP.training_step_dp so Opacus can track per-sample grads
    for the projection layer.
    """
    def collate_fn(samples):
        images, reports = zip(*samples)
        return {
            'image'  : torch.stack(images),   # [B, 1, H, W]
            'reports': list(reports),          # list[str]
        }

    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        drop_last   = True,
        pin_memory  = args.pin_memory and torch.cuda.is_available(),
        collate_fn  = collate_fn,
    )


# =============================================================================
# Checkpoint helpers  (GradSampleModule-aware)
# =============================================================================

def save_dp_checkpoint(
    save_path  : str,
    model,
    optimizer,
    privacy_engine,
    epoch      : int,
    global_step: int,
    args,
    target_delta: float,
):
    """
    Save a DP training checkpoint.

    After make_private(), `model` is a GradSampleModule.  Use
    `model._module.state_dict()` to get the unwrapped weights so they can be
    loaded back into a plain LatentDiffusionDP without Opacus.
    """
    eps_spent = privacy_engine.get_epsilon(target_delta)
    state = {
        'epoch'       : epoch,
        'global_step' : global_step,
        'model'       : model._module.state_dict(),
        'optimizer'   : optimizer.original_optimizer.state_dict(),
        'epsilon_spent': eps_spent,
        'args'        : vars(args),
    }
    torch.save(state, save_path)
    print(f'  [ckpt] saved → {save_path}  (ε_spent={eps_spent:.4f})')


def load_dp_checkpoint(load_path: str, model, optimizer=None, device='cpu'):
    """
    Resume from a checkpoint saved by save_dp_checkpoint.

    Works whether `model` is a plain LatentDiffusionDP or a GradSampleModule
    (i.e. can be called before or after make_private).
    """
    ckpt = torch.load(load_path, map_location=device)
    sd   = ckpt.get('model', ckpt)  # fallback if raw state_dict was saved

    # If wrapped by Opacus, target the inner module
    target = model._module if hasattr(model, '_module') else model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f'[load_dp_checkpoint] missing={len(missing)}  unexpected={len(unexpected)}  '
          f'ε_spent_at_save={ckpt.get("epsilon_spent", "N/A")}')

    if optimizer is not None and 'optimizer' in ckpt:
        opt_target = (optimizer.original_optimizer
                      if hasattr(optimizer, 'original_optimizer')
                      else optimizer)
        opt_target.load_state_dict(ckpt['optimizer'])

    return ckpt.get('epoch', 0), ckpt.get('global_step', 0)


# =============================================================================
# LR Scheduler helpers
# =============================================================================

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay."""
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    # Validate gradient accumulation
    assert args.logical_batch % args.physical_batch == 0, (
        f'logical_batch ({args.logical_batch}) must be divisible by '
        f'physical_batch ({args.physical_batch})'
    )
    grad_accum_steps = args.logical_batch // args.physical_batch

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    )
    print(f'Device : {device}')

    # ── Dataset ───────────────────────────────────────────────────────────────
    mimic_args = argparse.Namespace(
        root_path    = args.root_path,
        split_csv    = args.split_csv,
        meta_csv     = args.meta_csv,
        split        = 'train',
        image_size   = args.image_size,
        max_length   = args.max_length,
        view_filter  = args.view_filter,
        biobert_path = args.biobert_path,
    )
    dataset = MIMICCXRDataset(mimic_args)
    n_samples = len(dataset)
    print(f'[MIMIC-CXR] train samples: {n_samples}')
    if n_samples == 0:
        raise RuntimeError('No samples found. Check --root_path and CSV files.')

    sample_rate = args.logical_batch / n_samples
    print(f'[MIMIC-CXR] sample_rate q = {args.logical_batch}/{n_samples} = {sample_rate:.6f}')

    # ── BioBERT Embedder ──────────────────────────────────────────────────────
    assert BioBERTEmbedder is not None, (
        'BioBERTEmbedder could not be imported. '
        'Ensure Modules/BioBERT_embedder.py is accessible.'
    )
    embedder = BioBERTEmbedder(
        model_path = args.biobert_path,
        max_length = args.max_length,
    ).to(device)
    print(f'[BioBERT] loaded: {args.biobert_path}')

    # ── VAE ───────────────────────────────────────────────────────────────────
    vae = build_vae(args).to(device)
    if args.vae_ckpt:
        print(f'[VAE] loading checkpoint: {args.vae_ckpt}')
        _load_vae_ckpt(vae, args.vae_ckpt, device)
    else:
        print('[VAE] WARNING: no --vae_ckpt provided; using random weights')
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # ── UNet (use_checkpoint=False enforced in build_unet) ────────────────────
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
    ).to(device)

    # ── Load pretrained weights ───────────────────────────────────────────────
    if args.pretrained_ckpt:
        print(f'[LDM_dp] loading pretrained: {args.pretrained_ckpt}')
        ldm.init_from_ckpt(args.pretrained_ckpt)
    else:
        print('[LDM_dp] WARNING: no --pretrained_ckpt; fine-tuning from scratch')

    # ── Disable gradient checkpointing (Opacus incompatible) ──────────────────
    n_patched = disable_checkpointing(ldm)
    print(f'[DP] gradient checkpointing disabled on {n_patched} block(s)')

    # ── DataLoader (pre-Opacus, Opacus will re-wrap with Poisson sampler) ─────
    base_loader = build_dp_loader(args, dataset, batch_size=args.physical_batch)
    steps_per_epoch   = len(base_loader)
    logical_steps_per_epoch = steps_per_epoch // grad_accum_steps
    total_logical_steps = logical_steps_per_epoch * args.epochs
    print(f'[Loader] physical steps/epoch={steps_per_epoch}  '
          f'logical steps/epoch={logical_steps_per_epoch}  '
          f'grad_accum={grad_accum_steps}')

    # ── scale_factor init BEFORE make_private (register_buffer restriction) ───
    if args.scale_by_std:
        print('[LDM_dp] computing scale_factor from first batch ...')
        first_raw = next(iter(base_loader))
        first_raw = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in first_raw.items()}
        ldm.init_scale_factor(first_raw, is_first_batch=True)

    # ── Selective parameter unfreeze ──────────────────────────────────────────
    attn_params = ldm.configure_dp_params(
        ablation_blocks  = args.ablation_blocks,
        finetune_biobert = args.finetune_biobert,
    )
    if not attn_params:
        raise RuntimeError(
            'configure_dp_params returned empty param list. '
            'Check UNet has SpatialTransformer blocks and use_spatial_transformer=True.'
        )

    # ── Noise multiplier ──────────────────────────────────────────────────────
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

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(attn_params, lr=args.lr)

    # ── Opacus: make_private ──────────────────────────────────────────────────
    try:
        from opacus import PrivacyEngine
    except ImportError:
        raise ImportError('opacus is required. Install with: pip install opacus')

    privacy_engine = PrivacyEngine()
    ldm, optimizer, dp_loader = privacy_engine.make_private(
        module           = ldm,
        optimizer        = optimizer,
        data_loader      = base_loader,
        noise_multiplier = sigma,
        max_grad_norm    = args.max_grad_norm,
        poisson_sampling = True,
    )
    print(f'[Opacus] model wrapped with GradSampleModule')
    print(f'[Opacus] noise_multiplier={sigma:.6f}  max_grad_norm={args.max_grad_norm}')

    # ── LR Scheduler ──────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer    = optimizer,
        warmup_steps = args.warmup_steps,
        total_steps  = total_logical_steps,
    ) if args.warmup_steps > 0 else None

    # ── Resume DP checkpoint ──────────────────────────────────────────────────
    start_epoch = 0
    global_step = 0
    if args.resume_ckpt:
        print(f'[resume] loading DP checkpoint: {args.resume_ckpt}')
        start_epoch, global_step = load_dp_checkpoint(
            args.resume_ckpt, ldm, optimizer, device=str(device)
        )
        print(f'  resumed: epoch={start_epoch}  global_step={global_step}')

    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f'\n[DP-Train] epochs={args.epochs}  lr={args.lr}  '
          f'logical_batch={args.logical_batch}  physical_batch={args.physical_batch}  '
          f'grad_accum={grad_accum_steps}  σ={sigma:.4f}')
    print('=' * 60)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        ldm.train()
        epoch_losses  = []
        phys_step_idx = 0
        t0            = time.time()

        for batch in dp_loader:
            # Move image tensor to device; leave reports as list[str]
            batch = {
                'image'  : batch['image'].to(device),
                'reports': batch['reports'],
            }

            # ── Forward + Backward ────────────────────────────────────────────
            loss, loss_dict = ldm._module.training_step_dp(batch)
            (loss / grad_accum_steps).backward()

            phys_step_idx += 1
            is_last_accum = (phys_step_idx % grad_accum_steps == 0)

            if not is_last_accum:
                # Accumulate gradients: skip DP mechanism (no clip / no noise)
                optimizer.signal_skip_step(do_skip=True)
                optimizer.step()
            else:
                # Actual DP step: clip per-sample grads, add Gaussian noise
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

                epoch_losses.append(loss.item())
                global_step += 1

                if global_step % args.log_every == 0:
                    info = '  '.join(
                        f'{k}={v.item():.4f}' for k, v in loss_dict.items()
                    )
                    eps_now = privacy_engine.get_epsilon(args.target_delta)
                    print(f'  ep {epoch+1:04d}  step {global_step:06d} | '
                          f'{info}  ε={eps_now:.4f}')

            # Reset phys counter at end of epoch
            if phys_step_idx >= steps_per_epoch:
                break

        # ── Epoch summary ─────────────────────────────────────────────────────
        elapsed  = time.time() - t0
        eps_now  = privacy_engine.get_epsilon(args.target_delta)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')
        print(f'[Epoch {epoch+1:04d}/{start_epoch+args.epochs}] '
              f'loss={mean_loss:.4f}  ε={eps_now:.4f}  δ={args.target_delta}  '
              f'time={elapsed:.1f}s')

        if eps_now > args.target_epsilon * 1.05:
            print(f'[WARNING] ε_spent={eps_now:.4f} exceeds target ε={args.target_epsilon}. '
                  f'Consider stopping.')

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.save_dir, f'ldm_dp_epoch{epoch+1:04d}.pt'
            )
            save_dp_checkpoint(
                save_path     = ckpt_path,
                model         = ldm,
                optimizer     = optimizer,
                privacy_engine= privacy_engine,
                epoch         = epoch + 1,
                global_step   = global_step,
                args          = args,
                target_delta  = args.target_delta,
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
