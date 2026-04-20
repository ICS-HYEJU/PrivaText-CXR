"""
Diffusion/LDM_train.py  –  LDM training with real NIH CXR dataset
==================================================================

Usage:
    python Diffusion/LDM_train.py \\
        --data_name nih \\
        --bs 8 \\
        --epochs 100 \\
        --lr 1e-4 \\
        --save_dir ./checkpoints \\
        --save_every 10

Single-argparser design: every hyper-parameter (data, VAE, UNet, training)
is controlled through one parse_args() call.
"""

import os
import sys
import argparse
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

from LDM           import LatentDiffusion           # Diffusion/LDM.py
from UNetModel     import UNetModel                 # Diffusion/UNetModel.py
from autoencoder   import AutoencoderKL             # Model/autoencoder.py
from dataset       import NIH                       # Data/dataset.py
from class_label   import ClassLabelEmbedder        # Data/class_label.py


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='LDM Training on NIH CXR')

    # DEVICE -------------------------------------------------------------------
    parser.add_argument('--device_id', default='0',
                        help='GPU id (e.g. "0", "1")')

    # DATA ---------------------------------------------------------------------
    parser.add_argument('--data_name', default='nih', choices=['nih', 'mimic'])
    parser.add_argument('--root_path', default=None,
                        help='Dataset root; auto-set by --data_name if omitted')
    parser.add_argument('--task',      default='train', choices=['train', 'test'])
    parser.add_argument('--bs',        default=8,   type=int)
    parser.add_argument('--image_size',default=256, type=int)
    parser.add_argument('--image_show',default=False, action='store_true',
                        help='Show first dataset image on startup')
    # DataLoader
    parser.add_argument('--shuffle',     default=True,  type=lambda x: x.lower() != 'false')
    parser.add_argument('--num_workers', default=0,     type=int,
                        help='0 for debug; 4+ for training')
    parser.add_argument('--pin_memory',  default=True,  type=lambda x: x.lower() != 'false',
                        help='Enable pinned memory for faster GPU transfer')

    # Embedder -----------------------------------------------------------------
    parser.add_argument('--emb_target',  default='cls', choices=['cls', 'descript'])
    parser.add_argument('--emb_dims',    default=256,   type=int,
                        help='nn.Embedding output dim')
    parser.add_argument('--emb_out_dim', default=512,   type=int,
                        help='Projection output dim; must match UNet context_dim')

    # VAE: Encoder / Decoder ---------------------------------------------------
    parser.add_argument('--vae_img_size',         default=256,         type=int)
    parser.add_argument('--vae_in_ch',            default=1,           type=int)
    parser.add_argument('--vae_ch',               default=128,         type=int)
    parser.add_argument('--vae_ch_mult',          default=[1, 2, 4, 4, 4],
                        nargs='+', type=int)
    parser.add_argument('--vae_num_res_blocks',   default=2,           type=int)
    parser.add_argument('--vae_attn_resolutions', default=[32, 16],
                        nargs='+', type=int)
    parser.add_argument('--vae_dropout',          default=0.0,         type=float)
    parser.add_argument('--resamp_with_conv',     default=True,        type=lambda x: x.lower() != 'false')
    parser.add_argument('--z_channels',           default=1,           type=int)
    parser.add_argument('--double_z',             default=True,        type=lambda x: x.lower() != 'false')
    parser.add_argument('--vae_out_ch',           default=1,           type=int)
    parser.add_argument('--verbose',              default=False, action='store_true',
                        help='Print block shapes at each level')

    # VAE: pretrained checkpoint (optional) ------------------------------------
    parser.add_argument('--vae_ckpt', default=None,
                        help='Path to pretrained VAE checkpoint (.pt / .pth)')

    # UNet ---------------------------------------------------------------------
    parser.add_argument('--unet_image_size',        default=16,         type=int,
                        help='Latent spatial size (image_size / 2^num_downsamples)')
    parser.add_argument('--unet_in_ch',             default=1,          type=int)
    parser.add_argument('--unet_out_ch',            default=1,          type=int)
    parser.add_argument('--unet_ch',                default=128,        type=int)
    parser.add_argument('--unet_num_res_blocks',    default=2,          type=int)
    parser.add_argument('--unet_ch_mult',           default=[1, 2, 4],
                        nargs='+', type=int)
    parser.add_argument('--unet_attn_resolutions',  default=[1, 2, 4],
                        nargs='+', type=int)
    parser.add_argument('--unet_dropout',           default=0.0,        type=float)
    # UNet: Attention
    parser.add_argument('--unet_num_heads',         default=-1,         type=int,
                        help='Fixed head count; -1 → use num_head_channels')
    parser.add_argument('--unet_num_head_channels', default=8,          type=int)
    parser.add_argument('--use_spatial_transformer',default=True,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--transformer_depth',      default=1,          type=int)
    parser.add_argument('--context_dim',            default=512,        type=int)
    parser.add_argument('--legacy',                 default=False,
                        type=lambda x: x.lower() != 'false')
    # UNet: ResBlock options
    parser.add_argument('--use_scale_shift_norm',   default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--resblock_updown',        default=False,
                        type=lambda x: x.lower() != 'false')
    # UNet: misc
    parser.add_argument('--num_classes',  default=None, type=int)
    parser.add_argument('--n_embed',      default=None, type=int)
    parser.add_argument('--use_checkpoint',default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--use_fp16',    default=False,
                        type=lambda x: x.lower() != 'false')
    parser.add_argument('--write_json',  default=False,
                        type=lambda x: x.lower() != 'false')

    # Common -------------------------------------------------------------------
    parser.add_argument('--conv_dims', default=2, type=int, choices=[1, 2, 3])

    # Diffusion / LDM ----------------------------------------------------------
    parser.add_argument('--timesteps',      default=1000,     type=int)
    parser.add_argument('--beta_schedule',  default='linear')
    parser.add_argument('--scale_factor',   default=1.0,      type=float)
    parser.add_argument('--scale_by_std',   default=False, action='store_true',
                        help='Auto-compute scale_factor from first-batch latent std')
    parser.add_argument('--use_ema',        default=True,
                        type=lambda x: x.lower() != 'false')

    # Training -----------------------------------------------------------------
    parser.add_argument('--epochs',     default=100, type=int)
    parser.add_argument('--lr',         default=1e-4, type=float)
    parser.add_argument('--save_dir',   default='./checkpoints')
    parser.add_argument('--save_every', default=10,  type=int,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_every',  default=50,  type=int,
                        help='Print training loss every N steps')
    parser.add_argument('--resume_ckpt',default=None,
                        help='LDM checkpoint to resume from (.pt / .pth)')

    return parser.parse_args()


# =============================================================================
# Build helpers
# =============================================================================

def _load_vae_ckpt(vae: AutoencoderKL, ckpt_path: str, device):
    raw = torch.load(ckpt_path, map_location=device)
    # unwrap wrapper keys
    for key in ('state_dict', 'model', 'model_state_dict', 'net', 'weights'):
        if isinstance(raw, dict) and key in raw:
            raw = raw[key]
            print(f'  [VAE ckpt] using raw["{key}"]')
            break
    # auto-strip common prefix
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
        use_checkpoint         = args.use_checkpoint,
        use_fp16               = args.use_fp16,
        write_json             = args.write_json,
    )
    return UNetModel(unet_cfg)


# =============================================================================
# Infinite batch generator
# =============================================================================

def _infinite_batches(loader: DataLoader):
    while True:
        yield from loader


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

    # ── Root path ─────────────────────────────────────────────────────────────
    if args.root_path is None:
        if args.data_name == 'nih':
            args.root_path = '/storage/hjchoi/archive/DATA'
        else:
            raise ValueError('--root_path must be set for non-NIH datasets')

    data_path  = os.path.join(args.root_path, 'images')
    label_path = os.path.join(args.root_path, 'Data_Entry_2017.csv')

    # ── Dataset & DataLoader ──────────────────────────────────────────────────
    nih_args = argparse.Namespace(
        root_path  = args.root_path,
        data_path  = data_path,
        label_path = label_path,
        task       = args.task,
        image_size = args.image_size,
        image_show = args.image_show,
    )
    dataset = NIH(nih_args)
    loader  = DataLoader(
        dataset,
        batch_size  = args.bs,
        shuffle     = args.shuffle,
        num_workers = args.num_workers,
        drop_last   = True,
        pin_memory  = args.pin_memory and torch.cuda.is_available(),
    )
    print(f'[NIH] samples={len(dataset)}  bs={args.bs}  '
          f'steps_per_epoch={len(loader)}')

    # ── Embedder ──────────────────────────────────────────────────────────────
    if args.emb_target == 'cls':
        embedder = ClassLabelEmbedder(
            embed_dim  = args.emb_dims,
            output_dim = args.emb_out_dim,
        ).to(device)
    else:
        # TODO: load BioBERT description embedder
        raise NotImplementedError('Description embedder not yet implemented')

    # ── VAE ───────────────────────────────────────────────────────────────────
    vae = build_vae(args).to(device)
    if args.vae_ckpt:
        print(f'[VAE] loading checkpoint: {args.vae_ckpt}')
        _load_vae_ckpt(vae, args.vae_ckpt, device)
    else:
        print('[VAE] no checkpoint provided – using random weights')
    vae.eval()

    if args.verbose:
        from encoder import Encoder
        from decoder import Decoder
        Encoder(argparse.Namespace(
            in_channels=args.vae_in_ch, ch=args.vae_ch,
            ch_mult=args.vae_ch_mult, num_res_blocks=args.vae_num_res_blocks,
            attn_resolutions=args.vae_attn_resolutions, dropout=args.vae_dropout,
            resamp_with_conv=args.resamp_with_conv, resolution=args.vae_img_size,
            z_channels=args.z_channels, double_z=args.double_z, dims=args.conv_dims,
        )).print_architecture()

    # ── UNet ──────────────────────────────────────────────────────────────────
    unet = build_unet(args).to(device)
    n_unet = sum(p.numel() for p in unet.parameters())
    print(f'[UNet] params: {n_unet:,}')

    # ── LatentDiffusion ───────────────────────────────────────────────────────
    model = LatentDiffusion(
        unet              = unet,
        first_stage_model = vae,
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

    # ── Resume checkpoint ─────────────────────────────────────────────────────
    start_epoch = 0
    if args.resume_ckpt:
        print(f'[LDM] resuming from {args.resume_ckpt}')
        model.init_from_ckpt(args.resume_ckpt)
        stem = os.path.splitext(os.path.basename(args.resume_ckpt))[0]
        try:
            start_epoch = int(stem.split('epoch')[-1])
            print(f'  resumed at epoch {start_epoch}')
        except ValueError:
            pass

    # ── Batch factory ─────────────────────────────────────────────────────────
    _gen = _infinite_batches(loader)

    def make_batch() -> dict:
        imgs, label_strs = next(_gen)
        imgs = imgs.to(device)
        ctx  = embedder(list(label_strs)).detach()
        return {'image': imgs, 'context': ctx}

    # ── Scale-factor init (once, before first step) ───────────────────────────
    print('[LDM] initialising scale_factor ...')
    first_batch = make_batch()
    model.init_scale_factor(first_batch, is_first_batch=True)

    # ── Optimizer & save dir ──────────────────────────────────────────────────
    optimizer = model.build_optimizer(lr=args.lr)
    os.makedirs(args.save_dir, exist_ok=True)

    # =========================================================================
    # Training Loop
    # =========================================================================
    steps_per_epoch = len(loader)
    global_step     = start_epoch * steps_per_epoch
    print(f'\n[Train] epochs={args.epochs}  lr={args.lr}  '
          f'timesteps={args.timesteps}  scale_factor={model.scale_factor}')
    print('=' * 60)

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        epoch_losses = []
        t0 = time.time()

        for local_step in range(steps_per_epoch):
            batch = make_batch()

            loss, loss_dict = model.training_step(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.update_ema()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % args.log_every == 0:
                info = '  '.join(
                    f'{k}={v.item():.4f}' for k, v in loss_dict.items()
                )
                print(f'  ep {epoch+1:04d}  step {global_step:06d} | {info}')

        # ── Epoch summary ─────────────────────────────────────────────────────
        elapsed = time.time() - t0
        mean_loss = np.mean(epoch_losses)
        print(f'[Epoch {epoch+1:04d}/{start_epoch+args.epochs}] '
              f'loss={mean_loss:.4f}  time={elapsed:.1f}s')

        # ── Checkpoint ────────────────────────────────────────────────────────
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.save_dir, f'ldm_epoch{epoch+1:04d}.pt'
            )
            torch.save({
                'epoch'     : epoch + 1,
                'global_step': global_step,
                'model'     : model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'args'      : vars(args),
            }, ckpt_path)
            print(f'  [ckpt] saved → {ckpt_path}')

    print('\nTraining complete.')


if __name__ == '__main__':
    main()
