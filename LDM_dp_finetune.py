"""
LDM_dp_finetune.py  -  DP-SGD Fine-Tuning of LDM on MIMIC-CXR
=========================================================================
Fine-tunes only the cross-attention (SpatialTransformer) blocks of a
pre-trained LatentDiffusion model under (epsilon, delta)-differential privacy
using Opacus DP-SGD.

Reads directly from the original MIMIC-CXR PhysioNet directory using the
official split CSV (mimic-cxr-2.0.0-split.csv).  No pre-built split
directories are required.

Expected directory layout:
    <root_path>/
    ├── files/
    │   ├── p10/ ... p19/
    │   │   └── p<subject_id>/
    │   │       ├── s<study_id>/
    │   │       │   └── <dicom_id>.dcm
    │   │       └── s<study_id>.txt
    └── mimic-cxr-2.0.0-split.csv

Key design notes:
  1. BioBERT lives INSIDE LatentDiffusionDP (not in collate_fn) so Opacus
     computes per-sample grads for its projection layer.
  2. Loader yields raw strings; embedding happens inside training_step_dp.
  3. Only SpatialTransformer blocks (+ optionally BioBERT proj) are trained;
     VAE is fully frozen.
  4. Gradient checkpointing is disabled (incompatible with Opacus hooks).
  5. scale_factor initialised BEFORE make_private() to avoid buffer issues.
  6. Checkpoint save/load uses model._module.state_dict() (GradSampleModule).
  7. Physical-batch splitting via Opacus BatchMemoryManager:
       logical_batch -> N physical batches of size physical_batch
     optimizer.step() is called after every physical forward/backward pass
     (required for Poisson sampling compatibility).  The scheduler advances
     only at logical-batch boundaries (when _is_last_step_skipped is False).
  8. Model parallelism: VAE/BioBERT on --offload_device_id, UNet on --device_id.
     Buffers are re-pinned after Opacus make_private() which can shuffle them.

Usage:
    python LDM_dp_finetune.py \\
        --pretrained_ckpt ./checkpoints/ldm/ldm_epoch0100.pt \\
        --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \\
        --split train \\
        --device_id 0 \\
        --offload_device_id 1 \\
        --target_epsilon 10.0 \\
        --target_delta 1e-5 \\
        --max_grad_norm 0.001 \\
        --logical_batch 256 \\
        --physical_batch 1 \\
        --epochs 10 \\
        --lr 2e-5 \\
        --save_dir ./finetune_dp
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
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_proj_root = os.path.normpath(os.path.join(_this_dir, '..'))

for _d in [_proj_root, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from Model.Diffusion.LDM_dp    import LatentDiffusionDP
from Model.Diffusion.UNetmodel import UNetModel
from Model.VAE.Autoencoder     import VAE
from Model.attention_module_dp import disable_checkpointing

from Data.mimic_cxr import MIMICCXRDataset
from privacy.privacy_analysis import (
    compute_noise_multiplier,
    print_privacy_summary,
)

try:
    from Modules.BioBERT_embedder import BioBERTEmbedder
except ImportError:
    BioBERTEmbedder = None


# =============================================================================
# Dataset helper
# =============================================================================

def _load_patient_whitelist(dp_split_json, group):
    """
    Load the patient list for a group ('search'/'train') from a dp_splits.json
    manifest produced by Data/make_dp_splits.py.  Returns None when no manifest
    is given (→ use the whole split).
    """
    if not dp_split_json:
        return None
    import json
    with open(dp_split_json, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    assignment = manifest.get('assignment', manifest)
    patients = [p for p, g in assignment.items() if g == group]
    if not patients:
        raise ValueError(
            f"dp_split '{dp_split_json}' has no patients for group='{group}'. "
            f"Available groups: {sorted(set(assignment.values()))}")
    print(f"[dp_split] group='{group}'  patients={len(patients)}  "
          f"(from {dp_split_json})")
    return patients


def _make_dataset(root_path: str, split: str,
                  split_csv: str = 'mimic-cxr-2.0.0-split.csv',
                  image_size: int = 256,
                  max_length: int = 512,
                  patient_whitelist=None) -> MIMICCXRDataset:
    """
    Construct a MIMICCXRDataset for the given split.

    Args:
        root_path         : root of original MIMIC-CXR PhysioNet download
                            (contains files/ and mimic-cxr-2.0.0-split.csv)
        split             : one of 'train', 'validate', 'test'
        split_csv         : CSV filename relative to root_path (or absolute path)
        image_size        : resize target
        max_length        : max report character length
        patient_whitelist : optional list of patient_ids (e.g. only p10 subset
                            for D_search); None = use the whole split
    """
    valid_splits = ('train', 'validate', 'test')
    if split not in valid_splits:
        raise ValueError(f"split must be one of {valid_splits}, got '{split}'")

    ds_args = argparse.Namespace(
        root_path         = root_path,
        split_csv         = split_csv,
        split             = split,
        image_size        = image_size,
        max_length        = max_length,
        patient_whitelist = patient_whitelist,
    )
    return MIMICCXRDataset(ds_args)


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='DP-SGD Fine-Tuning of LDM on MIMIC-CXR (original data + split CSV)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # DEVICE -------------------------------------------------------------------
    parser.add_argument('--device_id', default='1',
                        help='Main GPU id for UNet + Opacus')
    parser.add_argument('--offload_device_id', default=None,
                        help='GPU id for frozen VAE + BioBERT. '
                             'None = same as --device_id (single-GPU). '
                             'Different GPU = model parallelism.')

    # DATA (original MIMIC-CXR) ------------------------------------------------
    parser.add_argument('--root_path',
                        default='/storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0',
                        help='Root of original MIMIC-CXR PhysioNet download '
                             '(contains files/ and mimic-cxr-2.0.0-split.csv)')
    parser.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv',
                        help='Split CSV filename (relative to root_path, or absolute path)')
    parser.add_argument('--split',     default='train',
                        choices=['train', 'validate', 'test'])
    parser.add_argument('--dp_split_json', default=None,
                        help='dp_splits.json from Data/make_dp_splits.py. When '
                             'set, only patients assigned to --dp_split_group '
                             'are used (e.g. the p10 subset for D_search). This '
                             'isolates the DP privacy budget across search/train.')
    parser.add_argument('--dp_split_group', default='search',
                        choices=['search', 'train'],
                        help='Which group of --dp_split_json to train on')
    parser.add_argument('--do_validation', default=True,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--val_batches', default=-1, type=int)
    parser.add_argument('--image_size',  default=256,  type=int)
    parser.add_argument('--max_length',  default=512,  type=int)
    parser.add_argument('--num_workers', default=0,    type=int)
    parser.add_argument('--pin_memory',  default=False)

    # VAE ----------------------------------------------------------------------
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
    parser.add_argument('--vae_ckpt',
                        default='/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/vae/vae_ep0070.pt')
    parser.add_argument('--verbose', default=False)

    # UNet ---------------------------------------------------------------------
    parser.add_argument('--unet_image_size',         default=16,         type=int)
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
    parser.add_argument('--scale_by_std',  default=False, action='store_true')
    parser.add_argument('--use_ema',       default=False,
                        type=lambda x: x.lower() != 'false')

    # Pretrained checkpoint ----------------------------------------------------
    parser.add_argument('--pretrained_ckpt',
                        default='/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/ldm/ldm_epoch0100.pt')

    # BioBERT ------------------------------------------------------------------
    parser.add_argument('--biobert_path', default='/storage/hjchoi')

    # DP-SGD -------------------------------------------------------------------
    parser.add_argument('--use_dp', default=True)
    parser.add_argument('--target_epsilon',   default=10.0,  type=float)
    parser.add_argument('--target_delta',     default=1e-5,  type=float)
    parser.add_argument('--max_grad_norm',    default=0.001, type=float)
    parser.add_argument('--noise_multiplier', default=None,  type=float)
    parser.add_argument('--logical_batch',    default=256,   type=int)
    parser.add_argument('--physical_batch',   default=1,     type=int)
    parser.add_argument('--ablation_blocks',  default=-1,    type=int)
    parser.add_argument('--finetune_biobert', default=False)

    # LoRA (adapt-lora) --------------------------------------------------------
    parser.add_argument('--use_lora', default=False,
                        type=lambda x: str(x).lower() != 'false',
                        help='Train LoRA adapters on cross-attention instead of '
                             'full SpatialTransformer blocks')
    parser.add_argument('--lora_rank',    default=4,   type=int)
    parser.add_argument('--lora_alpha',   default=4.0, type=float)
    parser.add_argument('--lora_dropout', default=0.0, type=float)
    parser.add_argument('--eps_milestones', default=[1, 3, 5, 10],
                        nargs='+', type=float,
                        help='Save a LoRA adapter file each time epsilon crosses '
                             'one of these budgets (LoRA mode only)')

    # Training -----------------------------------------------------------------
    parser.add_argument('--epochs',       default=10,    type=int)
    parser.add_argument('--lr',           default=2e-5,  type=float)
    parser.add_argument('--warmup_steps', default=500,   type=int)
    parser.add_argument('--scheduler_type', default='linear_warmup',
                        choices=['cosine_warmup', 'linear_warmup', 'none'])
    parser.add_argument('--save_dir',    default='./finetune_dp')
    parser.add_argument('--save_every',  default=5,     type=int)
    parser.add_argument('--log_every',   default=50,    type=int)
    parser.add_argument('--resume_ckpt', default=None)

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
        verbose          = args.verbose,
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
        use_checkpoint         = False,
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
        print('  [WARNING] >10% VAE keys missing - checkpoint may be incompatible')


# =============================================================================
# Buffer pinning (model parallelism)
# =============================================================================

def _pin_ldm_buffers_to_device(ldm, device, offload_device, verbose=False):
    """
    Ensure LDM-level buffers (sqrt_alphas_cumprod, betas, ...) stay on the
    main device while VAE/BioBERT buffers remain on offload_device.
    Recognises Opacus's '_module.' prefix when the model is GradSampleModule-wrapped.
    """
    offload_prefixes = (
        'first_stage_model.', 'embedder.',
        '_module.first_stage_model.', '_module.embedder.',
    )
    moved = []
    for name, buf in ldm.named_buffers():
        if any(name.startswith(p) for p in offload_prefixes):
            continue
        if buf.device != device:
            buf.data = buf.data.to(device)
            moved.append(name)
    if verbose:
        print(f'[pin_buffers] moved {len(moved)} buffer(s) -> {device}')
        for n in moved[:5]:
            print(f'              - {n}')


# =============================================================================
# DataLoaders
# =============================================================================

def _raw_collate_fn(samples):
    """Collate into {'image': Tensor[B,1,H,W], 'reports': list[str]}."""
    images, reports = zip(*samples)
    return {
        'image'  : torch.stack(images),
        'reports': list(reports),
    }


def _dp_raw_collate_fn(samples):
    """Collate into (Tensor[B,1,H,W], list[str]) tuple.

    Returns a tuple instead of a dict so that Opacus BatchMemoryManager can
    split each element by index: tensor[start:end] for images and
    list[start:end] for reports.
    """
    images, reports = zip(*samples)
    return torch.stack(images), list(reports)


def build_dp_loader(dataset, logical_batch, num_workers=0, pin_memory=False):
    return DataLoader(
        dataset,
        batch_size  = logical_batch,
        shuffle     = True,
        num_workers = num_workers,
        drop_last   = True,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        collate_fn  = _dp_raw_collate_fn,
    )


def build_val_loader(dataset, batch_size, num_workers=0):
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
def evaluate(ldm, val_loader, device, max_batches=-1):
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
# Checkpoint helpers
# =============================================================================

def save_dp_checkpoint(save_path, model, optimizer, privacy_engine,
                       epoch, global_step, args, target_delta):
    eps_spent = privacy_engine.get_epsilon(target_delta)
    torch.save({
        'epoch'         : epoch,
        'global_step'   : global_step,
        'model'         : model._module.state_dict(),
        'optimizer'     : optimizer.original_optimizer.state_dict(),
        'epsilon_spent' : eps_spent,
        'args'          : vars(args),
    }, save_path)
    print(f'  [ckpt] saved -> {save_path}  (eps_spent={eps_spent:.4f})')


def save_lora_checkpoint(save_path, model, privacy_engine, epoch, global_step,
                         args, target_delta):
    """
    Save ONLY the LoRA adapters (small file), tagged with the privacy budget
    spent so far.  `model` is the Opacus GradSampleModule; the inner module is
    model._module.
    """
    from Model.lora import lora_state_dict
    inner     = model._module if hasattr(model, '_module') else model
    eps_spent = privacy_engine.get_epsilon(target_delta) if privacy_engine else None
    torch.save({
        'epoch'         : epoch,
        'global_step'   : global_step,
        'lora'          : lora_state_dict(inner),
        'lora_rank'     : args.lora_rank,
        'lora_alpha'    : args.lora_alpha,
        'epsilon_spent' : eps_spent,
        'args'          : vars(args),
    }, save_path)
    eps_str = f'{eps_spent:.4f}' if eps_spent is not None else 'N/A'
    print(f'  [lora] saved -> {save_path}  (eps_spent={eps_str})')


def load_dp_checkpoint(load_path, model, optimizer=None, device='cpu'):
    ckpt   = torch.load(load_path, map_location=device)
    sd     = ckpt.get('model', ckpt)
    target = model._module if hasattr(model, '_module') else model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f'[load_dp_checkpoint] missing={len(missing)}  unexpected={len(unexpected)}  '
          f'eps_spent_at_save={ckpt.get("epsilon_spent", "N/A")}')

    if optimizer is not None and 'optimizer' in ckpt:
        opt_target = (optimizer.original_optimizer
                      if hasattr(optimizer, 'original_optimizer') else optimizer)
        opt_target.load_state_dict(ckpt['optimizer'])

    return ckpt.get('epoch', 0), ckpt.get('global_step', 0)


# =============================================================================
# Loss logging / plotting
# =============================================================================

def _write_loss_csvs(save_dir, step_hist, step_loss_hist,
                     epoch_rows):
    """Write per-step and per-epoch loss CSVs (overwritten each call)."""
    import csv
    with open(os.path.join(save_dir, 'loss_steps.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['global_step', 'train_loss'])
        w.writerows(zip(step_hist, step_loss_hist))
    with open(os.path.join(save_dir, 'loss_epochs.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['epoch', 'global_step', 'train_loss', 'val_loss', 'epsilon'])
        w.writerows(epoch_rows)


def save_loss_plot(save_dir, step_hist, step_loss_hist, epoch_rows):
    """
    Save loss_curve.png overlaying per-step train loss with per-epoch means
    (and val loss when available).  Silently skips if matplotlib is absent.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plot] matplotlib not available – skipped loss_curve.png')
        return None

    fig, ax = plt.subplots(figsize=(9, 5))
    if step_hist:
        ax.plot(step_hist, step_loss_hist, lw=0.7, alpha=0.5,
                color='tab:blue', label='train (per step)')
    if epoch_rows:
        e_steps = [r[1] for r in epoch_rows]
        e_train = [r[2] for r in epoch_rows]
        e_val   = [r[3] for r in epoch_rows]
        ax.plot(e_steps, e_train, 'o-', color='tab:blue', lw=1.6,
                label='train (epoch mean)')
        if any(v == v for v in e_val):        # any non-NaN
            ax.plot(e_steps, e_val, 's--', color='tab:orange', lw=1.6,
                    label='val (epoch)')
    ax.set_xlabel('global step')
    ax.set_ylabel('loss')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(save_dir, 'loss_curve.png')
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# =============================================================================
# LR Scheduler
# =============================================================================

def build_scheduler(optimizer, warmup_steps, total_steps,
                    scheduler_type='cosine_warmup'):
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

    # ── Devices ───────────────────────────────────────────────────────────────
    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    if args.offload_device_id is not None and torch.cuda.is_available():
        offload_device = torch.device(f'cuda:{args.offload_device_id}')
    else:
        offload_device = device

    model_parallel = (offload_device != device)
    print(f'Device (UNet/Opacus)     : {device}')
    if model_parallel:
        print(f'Offload device (VAE/BioBERT) : {offload_device}  [model parallelism ON]')
    print(f'Root path : {args.root_path}')
    print(f'Split CSV : {args.split_csv}')

    # ── Datasets ──────────────────────────────────────────────────────────────
    # DP budget isolation: restrict to the patient subset (e.g. p10 part) for
    # D_search / D_train when a dp_splits.json manifest is provided.
    train_whitelist = _load_patient_whitelist(args.dp_split_json,
                                              args.dp_split_group)
    train_dataset = _make_dataset(
        root_path         = args.root_path,
        split             = args.split,
        split_csv         = args.split_csv,
        image_size        = args.image_size,
        max_length        = args.max_length,
        patient_whitelist = train_whitelist,
    )
    n_train = len(train_dataset)
    print(f'[Train] training samples: {n_train}')
    if n_train == 0:
        raise RuntimeError(
            f"No samples found for split='{args.split}' in {args.root_path}. "
            "Check --root_path and --split_csv are correct."
        )

    val_dataset = None
    if args.do_validation:
        try:
            val_dataset = _make_dataset(
                root_path  = args.root_path,
                split      = 'validate',
                split_csv  = args.split_csv,
                image_size = args.image_size,
                max_length = args.max_length,
            )
            print(f'[Validation] validate samples: {len(val_dataset)}')
        except FileNotFoundError as e:
            print(f'[Validation] WARNING: {e}  - validation disabled')
            val_dataset = None

    # ── BioBERT Embedder -> offload_device ────────────────────────────────────
    assert BioBERTEmbedder is not None, (
        'BioBERTEmbedder could not be imported. '
        'Check Modules/BioBERT_embedder.py is accessible.'
    )
    embedder = BioBERTEmbedder(
        model_path = args.biobert_path,
        max_length = args.max_length,
    ).to(offload_device)
    print(f'[BioBERT] loaded: {args.biobert_path}  -> {offload_device}')

    # ── VAE -> offload_device (always frozen) ─────────────────────────────────
    vae = build_vae(args).to(offload_device)
    if args.vae_ckpt:
        print(f'[VAE] ckpt loading: {args.vae_ckpt}')
        _load_vae_ckpt(vae, args.vae_ckpt, offload_device)
    else:
        print('[VAE] WARNING: no --vae_ckpt; using random weights')
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    print(f'[VAE] frozen  -> {offload_device}')

    # ── UNet -> main device ───────────────────────────────────────────────────
    unet = build_unet(args).to(device)
    print(f'[UNet] params: {sum(p.numel() for p in unet.parameters()):,}  -> {device}')

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
        device            = device,
        use_dp            = args.use_dp,
    )
    ldm = ldm.to(device)
    if model_parallel:
        ldm.first_stage_model.to(offload_device)
        ldm.embedder.to(offload_device)
        _pin_ldm_buffers_to_device(ldm, device, offload_device)

    ldm.offload_device = offload_device

    if args.pretrained_ckpt:
        print(f'[LDM_dp] loading pretrained: {args.pretrained_ckpt}')
        ldm.init_from_ckpt(args.pretrained_ckpt)
        if model_parallel:
            ldm.first_stage_model.to(offload_device)
            ldm.embedder.to(offload_device)
            _pin_ldm_buffers_to_device(ldm, device, offload_device)
    else:
        print('[LDM_dp] WARNING: no --pretrained_ckpt; fine-tuning from scratch')

    n_patched = disable_checkpointing(ldm)
    print(f'[DP] gradient checkpointing disabled on {n_patched} block(s)')

    # ── DataLoaders ───────────────────────────────────────────────────────────
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

    logical_steps_per_epoch = len(base_loader)
    total_logical_steps     = logical_steps_per_epoch * args.epochs
    sample_rate             = 1.0 / logical_steps_per_epoch
    approx_chunks           = math.ceil(args.logical_batch / args.physical_batch)
    print(f'[Loader] N={n_train}  logical_steps/epoch={logical_steps_per_epoch}  '
          f'sample_rate=1/{logical_steps_per_epoch}={sample_rate:.6f}  '
          f'physical_batch={args.physical_batch}  chunks/logical~{approx_chunks}')

    if args.scale_by_std:
        print('[LDM_dp] computing scale_factor from first batch ...')
        first_images, first_reports = next(iter(base_loader))
        first_chunk = {
            'image'  : first_images[:args.physical_batch].to(device),
            'reports': first_reports[:args.physical_batch],
        }
        ldm.init_scale_factor(first_chunk, is_first_batch=True)

    if args.use_lora:
        attn_params = ldm.configure_lora_params(
            rank             = args.lora_rank,
            alpha            = args.lora_alpha,
            dropout          = args.lora_dropout,
            finetune_biobert = args.finetune_biobert,
        )
        if not attn_params:
            raise RuntimeError('configure_lora_params returned empty param list.')
    else:
        attn_params = ldm.configure_dp_params(
            ablation_blocks  = args.ablation_blocks,
            finetune_biobert = args.finetune_biobert,
        )
        if not attn_params:
            raise RuntimeError('configure_dp_params returned empty param list.')

    if args.noise_multiplier is not None:
        sigma = args.noise_multiplier
        print(f'[DP] using provided noise_multiplier sigma={sigma:.6f}')
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

    optimizer = torch.optim.AdamW(attn_params, lr=args.lr)

    # ── Opacus ────────────────────────────────────────────────────────────────
    try:
        from opacus import PrivacyEngine
        from opacus.grad_sample import GradSampleModule
        from opacus.utils.batch_memory_manager import BatchMemoryManager
    except ImportError:
        raise ImportError('opacus is required. Install with: pip install opacus')

    if isinstance(ldm, GradSampleModule):
        print('[DP] WARNING: model already wrapped - unwrapping before make_private')
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
    print(f'[Opacus] GradSampleModule ready  sigma={sigma:.6f}  C(clip)={args.max_grad_norm}')

    # ── Re-pin after Opacus wrapping (GradSampleModule can shuffle buffers) ──
    if model_parallel:
        inner = ldm._module if hasattr(ldm, '_module') else ldm
        inner.first_stage_model.to(offload_device)
        inner.embedder.to(offload_device)
        inner.offload_device = offload_device
        _pin_ldm_buffers_to_device(ldm, device, offload_device, verbose=True)
        sa = inner.sqrt_alphas_cumprod
        print(f'[pin_buffers] sqrt_alphas_cumprod.device = {sa.device} '
              f'(expected {device})')
        assert sa.device == device, (
            f'sqrt_alphas_cumprod on {sa.device}, expected {device}. '
            f'Buffer pinning failed.'
        )

    scheduler = (
        build_scheduler(optimizer, args.warmup_steps, total_logical_steps,
                        args.scheduler_type)
        if args.warmup_steps > 0 and args.scheduler_type != 'none' else None
    )

    start_epoch = 0
    global_step = 0
    if args.resume_ckpt:
        print(f'[resume] {args.resume_ckpt}')
        start_epoch, global_step = load_dp_checkpoint(
            args.resume_ckpt, ldm, optimizer, device=str(device)
        )
        print(f'  resumed: epoch={start_epoch}  global_step={global_step}')

    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Training Loop
    # =========================================================================
    print(f'\n[DP-Train] split={args.split}  epochs={args.epochs}  lr={args.lr}  '
          f'logical_batch={args.logical_batch}  physical_batch={args.physical_batch}  '
          f'chunks/step~{approx_chunks}  sigma={sigma:.4f}')
    print('=' * 60)

    # LoRA: track which epsilon milestones have been saved (LoRA mode only)
    saved_milestones = set()

    # Loss history for CSV logging + loss_curve.png
    step_hist, step_loss_hist = [], []
    epoch_rows = []   # (epoch, global_step, train_loss, val_loss, epsilon)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        ldm.train()
        epoch_losses = []
        t0           = time.time()

        # BatchMemoryManager splits each Poisson-sampled logical batch into
        # physical batches and calls optimizer.step() after every one,
        # which is required for Poisson-sampling compatibility.
        # _is_last_step_skipped is False only at the logical-batch boundary
        # (when the actual DP noise-and-update step is performed).
        physical_losses = []
        last_loss_dict  = {}

        with BatchMemoryManager(
            data_loader             = dp_loader,
            max_physical_batch_size = args.physical_batch,
            optimizer               = optimizer,
        ) as memory_safe_loader:
            for images, reports in memory_safe_loader:
                optimizer.zero_grad()

                batch = {
                    'image'  : images.to(device),
                    'reports': list(reports),
                }
                loss, loss_dict = ldm._module.training_step_dp(batch)
                loss.backward()
                optimizer.step()

                physical_losses.append(loss.item())
                last_loss_dict = loss_dict

                # _is_last_step_skipped is True for intermediate physical
                # batches; False when the logical-batch DP update fired.
                if not getattr(optimizer, '_is_last_step_skipped', False):
                    step_loss       = float(np.mean(physical_losses))
                    physical_losses = []

                    if scheduler is not None:
                        scheduler.step()

                    epoch_losses.append(step_loss)
                    global_step += 1

                    step_hist.append(global_step)
                    step_loss_hist.append(step_loss)

                    if global_step % args.log_every == 0:
                        info    = '  '.join(f'{k}={v.item():.4f}'
                                            for k, v in last_loss_dict.items())
                        eps_now = privacy_engine.get_epsilon(args.target_delta)
                        print(f'  ep {epoch+1:04d}  step {global_step:06d} | '
                              f'{info}  eps={eps_now:.4f}')

        elapsed   = time.time() - t0
        eps_now   = privacy_engine.get_epsilon(args.target_delta)
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float('nan')

        val_str  = ''
        val_loss = float('nan')
        if val_loader is not None:
            val_loss = evaluate(ldm, val_loader, device, args.val_batches)
            val_str  = f'  val_loss={val_loss:.4f}'

        print(f'[Epoch {epoch+1:04d}/{start_epoch+args.epochs}] '
              f'loss={mean_loss:.4f}{val_str}  '
              f'eps={eps_now:.4f}  delta={args.target_delta}  '
              f'time={elapsed:.1f}s')

        # Record epoch summary, dump CSVs, and refresh loss_curve.png
        epoch_rows.append((epoch + 1, global_step, mean_loss, val_loss, eps_now))
        _write_loss_csvs(args.save_dir, step_hist, step_loss_hist, epoch_rows)
        plot_path = save_loss_plot(args.save_dir, step_hist, step_loss_hist,
                                   epoch_rows)
        if plot_path:
            print(f'  [plot] loss curve -> {plot_path}')

        if eps_now > args.target_epsilon * 1.05:
            print(f'[WARNING] eps_spent={eps_now:.4f} exceeds target '
                  f'eps={args.target_epsilon}. Consider early stopping.')

        # LoRA: save a small adapter file each time we cross a budget milestone
        if args.use_lora:
            for m in sorted(args.eps_milestones):
                if m not in saved_milestones and eps_now >= m:
                    saved_milestones.add(m)
                    lora_path = os.path.join(
                        args.save_dir, f'ldm_lora_eps{m:g}.pt')
                    save_lora_checkpoint(
                        save_path      = lora_path,
                        model          = ldm,
                        privacy_engine = privacy_engine,
                        epoch          = epoch + 1,
                        global_step    = global_step,
                        args           = args,
                        target_delta   = args.target_delta,
                    )

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
    print(f'\nTraining complete.  Final eps={eps_final:.4f}  delta={args.target_delta}')


if __name__ == '__main__':
    main()
