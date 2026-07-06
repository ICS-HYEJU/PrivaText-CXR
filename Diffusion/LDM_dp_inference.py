"""
LDM_dp_inference.py  –  End-to-End Generation from a DP-finetuned LDM
=====================================================================
Loads a DP-finetuned LatentDiffusionDP checkpoint and generates CXR images
conditioned on text descriptions, then saves an annotated grid (image +
description) plus individual images and a CSV mapping.

This is the S3 "end-to-end verification" pipeline of docs/DP_LDM_FRAMEWORK_PLAN.md.
It reuses the EXISTING full-attention DP checkpoint (NOT LoRA) to confirm the
full loop works:

    description (str)
      └─ BioBERT encoder (mode='description') → context [B,1,512]
           └─ LatentDiffusionDP.sample(c)  (DDPM reverse diffusion)
                └─ latent z → decode_first_stage(z) → VAE.decode
                     └─ image [B,1,256,256]
                          └─ grid PNG + individual PNGs + CSV

Usage (run from project root):
    python Diffusion/LDM_dp_inference.py \\
        --dp_ckpt      ./finetune_dp/ldm_dp_final.pt \\
        --vae_ckpt     ./checkpoints/vae/vae_ep0020.pt \\
        --biobert_path /storage/hjchoi \\
        --descriptions ./prompts.txt \\
        --n_samples    2 \\
        --output_dir   ./generated

--descriptions may be a .txt file (one description per line) OR a single
inline string.  --n_samples draws N independent samples per description.
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))       # Diffusion/
_proj_root = os.path.normpath(os.path.join(_this_dir, '..'))  # PrivaText-CXR/
for _d in [_proj_root, _this_dir]:
    if _d not in sys.path:
        sys.path.insert(0, _d)


# The DP finetune script and LDM_dp live under either Model/Diffusion/ or
# Diffusion/ depending on the working copy; try both import layouts.
def _import_symbols():
    """Return (LatentDiffusionDP, UNetModel, VAE, build_embedder)."""
    ldm_dp = unet = vae = None
    errors = []
    for ldm_dp_path in ('Model.Diffusion.LDM_dp', 'Diffusion.LDM_dp', 'LDM_dp'):
        try:
            mod = __import__(ldm_dp_path, fromlist=['LatentDiffusionDP'])
            ldm_dp = mod.LatentDiffusionDP
            break
        except ImportError as e:
            errors.append(f'{ldm_dp_path}: {e}')
    for unet_path in ('Model.Diffusion.UNetModel', 'Diffusion.UNetModel',
                      'Model.Diffusion.UNetmodel', 'UNetModel'):
        try:
            mod = __import__(unet_path, fromlist=['UNetModel'])
            unet = mod.UNetModel
            break
        except ImportError as e:
            errors.append(f'{unet_path}: {e}')
    for vae_path in ('Model.VAE.Autoencoder', 'Model.autoencoder', 'autoencoder'):
        try:
            mod = __import__(vae_path, fromlist=['VAE', 'AutoencoderKL'])
            vae = getattr(mod, 'VAE', None) or getattr(mod, 'AutoencoderKL')
            break
        except (ImportError, AttributeError) as e:
            errors.append(f'{vae_path}: {e}')
    if ldm_dp is None or unet is None or vae is None:
        raise ImportError('Could not import required symbols:\n  ' +
                          '\n  '.join(errors))
    return ldm_dp, unet, vae


def _build_embedder(biobert_path, max_length, device):
    """
    Build the text encoder, matching whichever class the training used.
    Prefers BioBERTEmbedder (training), falls back to BioBERTContextEncoder.
    Returns (embedder, encode_fn) where encode_fn(texts) -> context tensor.
    """
    # Try the training-time embedder first (so checkpoint keys match).
    for path, kwarg in (('Modules.BioBERT_embedder', 'model_path'),
                        ('BioBERT_embedder', 'model_path')):
        try:
            mod = __import__(path, fromlist=['BioBERTEmbedder'])
            emb = mod.BioBERTEmbedder(**{kwarg: biobert_path},
                                      max_length=max_length).to(device)
            return emb, _make_encode_fn(emb)
        except (ImportError, AttributeError, TypeError):
            continue

    # Fallback: BioBERTContextEncoder (this branch).
    for path in ('Diffusion.context_encoder', 'context_encoder'):
        try:
            mod = __import__(path, fromlist=['BioBERTContextEncoder'])
            emb = mod.BioBERTContextEncoder(output_dim=512,
                                            max_length=max_length,
                                            freeze=True).to(device)
            return emb, _make_encode_fn(emb)
        except (ImportError, AttributeError):
            continue

    raise ImportError('Could not import a BioBERT text encoder '
                      '(BioBERTEmbedder or BioBERTContextEncoder).')


def _make_encode_fn(embedder):
    """
    Wrap an embedder so descriptions are encoded with mode='description'
    when supported (see docs 1.1 mode bug), else plain call.
    """
    def encode(texts):
        try:
            return embedder(texts, mode='description')
        except TypeError:
            return embedder(texts)
    return encode


# =============================================================================
# Argument Parser  (mirrors LDM_dp_finetune.py model args)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='End-to-end generation from a DP-finetuned LDM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument('--device_id', default='0')

    # Checkpoints / conditioning
    p.add_argument('--dp_ckpt', required=True,
                   help='DP-finetuned checkpoint (saved by LDM_dp_finetune.py)')
    p.add_argument('--vae_ckpt', default=None,
                   help='VAE checkpoint (decoder must match training). If the '
                        'DP ckpt already contains first_stage_model weights, '
                        'this is optional.')
    p.add_argument('--biobert_path', default='/storage/hjchoi')

    # Descriptions
    p.add_argument('--descriptions', required=True,
                   help='Path to a .txt file (one description per line) OR an '
                        'inline description string')
    p.add_argument('--n_samples', default=1, type=int,
                   help='Independent samples to draw per description')
    p.add_argument('--sample_timesteps', default=None, type=int,
                   help='Override number of DDPM sampling steps (default: all)')
    p.add_argument('--seed', default=0, type=int)

    # Output
    p.add_argument('--output_dir', default='./generated')
    p.add_argument('--max_length', default=512, type=int)

    # ── VAE args (must match training) ────────────────────────────────────────
    p.add_argument('--vae_img_size',         default=256, type=int)
    p.add_argument('--vae_in_ch',            default=1,   type=int)
    p.add_argument('--vae_ch',               default=128, type=int)
    p.add_argument('--vae_ch_mult',          default=[1, 2, 4, 4, 4], nargs='+', type=int)
    p.add_argument('--vae_num_res_blocks',   default=2,   type=int)
    p.add_argument('--vae_attn_resolutions', default=[32, 16], nargs='+', type=int)
    p.add_argument('--vae_dropout',          default=0.0, type=float)
    p.add_argument('--resamp_with_conv',     default=True,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--z_channels',           default=1,   type=int)
    p.add_argument('--double_z',             default=True,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--vae_out_ch',           default=1,   type=int)
    p.add_argument('--conv_dims',            default=2,   type=int, choices=[1, 2, 3])

    # ── UNet args (must match training) ───────────────────────────────────────
    p.add_argument('--unet_image_size',        default=16,  type=int)
    p.add_argument('--unet_in_ch',             default=1,   type=int)
    p.add_argument('--unet_out_ch',            default=1,   type=int)
    p.add_argument('--unet_ch',                default=128, type=int)
    p.add_argument('--unet_num_res_blocks',    default=2,   type=int)
    p.add_argument('--unet_ch_mult',           default=[1, 2, 4], nargs='+', type=int)
    p.add_argument('--unet_attn_resolutions',  default=[1, 2, 4], nargs='+', type=int)
    p.add_argument('--unet_dropout',           default=0.0, type=float)
    p.add_argument('--unet_num_heads',         default=-1,  type=int)
    p.add_argument('--unet_num_head_channels', default=8,   type=int)
    p.add_argument('--use_spatial_transformer', default=True,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--transformer_depth',      default=1,   type=int)
    p.add_argument('--context_dim',            default=512, type=int)
    p.add_argument('--legacy',                 default=False,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--use_scale_shift_norm',   default=False,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--resblock_updown',        default=False,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--num_classes', default=None, type=int)
    p.add_argument('--n_embed',     default=None, type=int)
    p.add_argument('--use_fp16',    default=False,
                   type=lambda x: x.lower() != 'false')
    p.add_argument('--write_json',  default=False,
                   type=lambda x: x.lower() != 'false')

    # ── Diffusion args ────────────────────────────────────────────────────────
    p.add_argument('--timesteps',     default=1000, type=int)
    p.add_argument('--beta_schedule', default='linear')
    p.add_argument('--scale_factor',  default=1.0,  type=float)

    return p.parse_args()


# =============================================================================
# Model build (mirrors LDM_dp_finetune.py)
# =============================================================================

def build_vae(args, VAE):
    cfg = argparse.Namespace(
        in_channels=args.vae_in_ch, out_channels=args.vae_out_ch,
        ch=args.vae_ch, ch_mult=args.vae_ch_mult,
        num_res_blocks=args.vae_num_res_blocks,
        attn_resolutions=args.vae_attn_resolutions,
        dropout=args.vae_dropout, resamp_with_conv=args.resamp_with_conv,
        resolution=args.vae_img_size, z_channels=args.z_channels,
        double_z=args.double_z, dims=args.conv_dims,
    )
    return VAE(cfg)


def build_unet(args, UNetModel):
    cfg = argparse.Namespace(
        image_size=args.unet_image_size, in_channels=args.unet_in_ch,
        out_channels=args.unet_out_ch, model_channels=args.unet_ch,
        num_res_blocks=args.unet_num_res_blocks, channel_mult=args.unet_ch_mult,
        attention_resolutions=args.unet_attn_resolutions, dropout=args.unet_dropout,
        dims=args.conv_dims, conv_resample=True, num_heads=args.unet_num_heads,
        num_head_channels=args.unet_num_head_channels, num_heads_upsample=-1,
        use_spatial_transformer=args.use_spatial_transformer,
        transformer_depth=args.transformer_depth, context_dim=args.context_dim,
        use_new_attention_order=False, legacy=args.legacy,
        use_scale_shift_norm=args.use_scale_shift_norm,
        resblock_updown=args.resblock_updown, num_classes=args.num_classes,
        n_embed=args.n_embed, use_checkpoint=False, use_fp16=args.use_fp16,
        write_json=args.write_json,
    )
    return UNetModel(cfg)


def _load_state_into(model, ckpt_path, device):
    """Load a DP-finetune checkpoint (saved by save_dp_checkpoint) into model."""
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get('model', ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'[ckpt] loaded {ckpt_path}  missing={len(missing)}  '
          f'unexpected={len(unexpected)}')
    eps = ckpt.get('epsilon_spent') if isinstance(ckpt, dict) else None
    if eps is not None:
        print(f'[ckpt] epsilon_spent_at_save = {eps}')
    return missing, unexpected


# =============================================================================
# Description loading
# =============================================================================

def load_descriptions(spec: str) -> list:
    """spec is a path to a .txt (one per line) or an inline string."""
    if os.path.isfile(spec):
        with open(spec, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            raise ValueError(f'No non-empty lines in {spec}')
        return lines
    return [spec]


# =============================================================================
# Saving  (grid with captions + individual PNGs + CSV)
# =============================================================================

def _to_uint8(img: torch.Tensor) -> np.ndarray:
    """[1,H,W] or [H,W] in [-1,1] → uint8 [H,W] in [0,255]."""
    x = img.detach().float().cpu()
    if x.dim() == 3:
        x = x[0]
    x = (x.clamp(-1, 1) + 1.0) / 2.0            # [-1,1] → [0,1]
    return (x.numpy() * 255.0).round().astype(np.uint8)


def save_outputs(images, descriptions, sample_idx, output_dir):
    """
    images       : Tensor [N, 1, H, W]  in [-1, 1]
    descriptions : list[str]  length N  (per-image caption)
    sample_idx   : list[int]  length N  (which description each came from)
    """
    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    # ── Individual PNGs + CSV ────────────────────────────────────────────────
    from PIL import Image
    rows = []
    for i in range(images.shape[0]):
        arr   = _to_uint8(images[i])
        fname = f'{sample_idx[i]:03d}_sample{i:03d}.png'
        Image.fromarray(arr, mode='L').save(os.path.join(samples_dir, fname))
        rows.append({'index': sample_idx[i],
                     'description': descriptions[i],
                     'file': os.path.join('samples', fname)})

    with open(os.path.join(output_dir, 'descriptions.csv'), 'w',
              newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['index', 'description', 'file'])
        writer.writeheader()
        writer.writerows(rows)

    # ── Annotated grid (best-effort; skipped if matplotlib absent) ───────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n    = images.shape[0]
        cols = min(4, n)
        rows_n = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows_n, cols,
                                 figsize=(4 * cols, 4.6 * rows_n),
                                 squeeze=False)
        for i in range(rows_n * cols):
            ax = axes[i // cols][i % cols]
            ax.axis('off')
            if i < n:
                ax.imshow(_to_uint8(images[i]), cmap='gray', vmin=0, vmax=255)
                cap = descriptions[i]
                cap = (cap[:60] + '…') if len(cap) > 60 else cap
                ax.set_title(cap, fontsize=8, wrap=True)
        fig.tight_layout()
        grid_path = os.path.join(output_dir, 'grid_all.png')
        fig.savefig(grid_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'[save] grid → {grid_path}')
    except ImportError:
        print('[save] matplotlib not available – skipped grid_all.png '
              '(individual PNGs + CSV still written)')

    print(f'[save] {images.shape[0]} image(s) → {samples_dir}')
    print(f'[save] mapping → {os.path.join(output_dir, "descriptions.csv")}')


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    LatentDiffusionDP_, UNetModel_, VAE_ = _import_symbols()

    # ── Descriptions ──────────────────────────────────────────────────────────
    descriptions = load_descriptions(args.descriptions)
    print(f'[input] {len(descriptions)} description(s), '
          f'{args.n_samples} sample(s) each')

    # ── Build modules ─────────────────────────────────────────────────────────
    embedder, encode_fn = _build_embedder(args.biobert_path, args.max_length, device)
    vae  = build_vae(args, VAE_).to(device)
    unet = build_unet(args, UNetModel_).to(device)

    ldm = LatentDiffusionDP_(
        unet=unet, first_stage_model=vae, embedder=embedder,
        cond_stage_key='context', first_stage_key='image',
        conditioning_key='crossattn', timesteps=args.timesteps,
        beta_schedule=args.beta_schedule, image_size=args.unet_image_size,
        channels=args.z_channels, scale_factor=args.scale_factor,
        scale_by_std=False, use_ema=False, use_dp=False,
    ).to(device)

    # ── Load weights ──────────────────────────────────────────────────────────
    # Optional standalone VAE ckpt first (in case DP ckpt lacks first_stage_model)
    if args.vae_ckpt:
        raw = torch.load(args.vae_ckpt, map_location=device)
        for key in ('state_dict', 'model', 'model_state_dict', 'net', 'weights'):
            if isinstance(raw, dict) and key in raw:
                raw = raw[key]
                break
        m, u = vae.load_state_dict(raw, strict=False)
        print(f'[vae ckpt] {args.vae_ckpt}  missing={len(m)}  unexpected={len(u)}')

    # DP-finetuned weights (UNet attn + BioBERT proj + buffers, possibly VAE)
    _load_state_into(ldm, args.dp_ckpt, device)

    ldm.eval()

    # ── Generate ──────────────────────────────────────────────────────────────
    all_images, all_caps, all_idx = [], [], []
    for d_idx, desc in enumerate(descriptions):
        # context for this description, repeated n_samples times
        c1 = encode_fn([desc]).to(device)                # [1, seq, 512]
        c  = c1.repeat(args.n_samples, 1, 1)             # [n, seq, 512]

        with torch.no_grad():
            imgs = ldm.sample(c, batch_size=args.n_samples,
                              verbose=False, timesteps=args.sample_timesteps)
        imgs = imgs.detach().cpu()                       # [n, 1, H, W]
        all_images.append(imgs)
        all_caps.extend([desc] * args.n_samples)
        all_idx.extend([d_idx] * args.n_samples)
        print(f'  [{d_idx+1}/{len(descriptions)}] generated '
              f'{args.n_samples} image(s): "{desc[:50]}"')

    images = torch.cat(all_images, dim=0)                # [N, 1, H, W]
    save_outputs(images, all_caps, all_idx, args.output_dir)
    print('\nDone.')


if __name__ == '__main__':
    main()
