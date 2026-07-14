"""
LDM_dp_eval.py  ?  Paired evaluation: generated vs held-out real images
=======================================================================
Feeds descriptions from data NOT used in training (a held-out split, e.g. the
official 'test'/'validate' split, or D_test from dp_splits.json) to the model,
generates images, and measures how well each generated image matches its
PAIRED real image.

    (real_image, report)  ¦¡ report ¦¡?  LDM.sample  ¦¡?  generated_image
                ¦¦¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡ metric(real, generated) ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¥

IMPORTANT ? generation is NOT reconstruction:
    A description ("bilateral pleural effusion ...") maps to MANY possible real
    images (different patients/pose/anatomy).  So pixel-aligned metrics
    (SSIM / PSNR) are expected to be LOW and are only a rough proxy.  For a
    clinically meaningful "match", prefer:
      - LPIPS            : perceptual distance (lower = closer)
      - FID (set-level)  : distribution distance real-set vs generated-set
      - (future S8) a CXR classifier's label agreement between real & generated
    SSIM/PSNR are still reported for completeness.

Usage (run from project root):
    python LDM_dp_eval.py \\
        --dp_ckpt      ./checkpoints/ldm/ldm_epoch0100.pt \\
        --lora_ckpt    ./finetune_dp/ldm_lora_eps5.pt \\
        --vae_ckpt     ./checkpoints/vae/vae_ep0070.pt \\
        --root_path    /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \\
        --eval_split   test \\
        --max_eval     100 \\
        --metrics ssim psnr lpips --compute_fid true \\
        --output_dir   ./eval_out

To evaluate on the held-out D_test patients of a dp_splits.json instead of the
official split, add:  --dp_split_json ./dp_splits.json --dp_split_group train
(train-group patients NOT used for D_search) ? or just use --eval_split test.
"""

import os
import sys
import csv
import argparse

import numpy as np
import torch
import torch.nn.functional as F

_proj_root = os.path.dirname(os.path.abspath(__file__))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

# Reuse the inference model builder/loader and arg set.
from LDM_dp_inference import add_model_args, build_and_load_ldm, _to_uint8
from Data.mimic_cxr import MIMICCXRDataset
from Eval_metric.ssim import compute_ssim
from Eval_metric.psnr import compute_psnr


# =============================================================================
# Args
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Paired eval: generated vs held-out real images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_model_args(p)   # ckpts + vae/unet/diffusion args (shared with inference)

    # Held-out data (must NOT be the training data)
    p.add_argument('--root_path', required=True,
                   help='Root of original MIMIC-CXR download (contains files/ '
                        'and the split CSV)')
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test',
                   choices=['train', 'validate', 'test'],
                   help="Official split to evaluate on. Use 'test'/'validate' "
                        "(never seen in training).")
    p.add_argument('--dp_split_json', default=None,
                   help='Optional dp_splits.json to further restrict patients '
                        '(e.g. --dp_split_group train = patients NOT used for '
                        'D_search).')
    p.add_argument('--dp_split_group', default='train',
                   choices=['search', 'train'])
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--max_eval', default=100, type=int,
                   help='Max (image, report) pairs to evaluate (-1 = all)')

    # Sampling
    p.add_argument('--n_samples', default=1, type=int,
                   help='Samples generated per description; per-pair metric is '
                        'reported as best and mean over these samples')
    p.add_argument('--sample_timesteps', default=None, type=int)
    p.add_argument('--seed', default=0, type=int)

    # Metrics
    p.add_argument('--metrics', nargs='+', default=['ssim', 'psnr'],
                   choices=['ssim', 'psnr', 'lpips'],
                   help='Per-pair metrics to compute')
    p.add_argument('--compute_fid', default=False,
                   type=lambda x: str(x).lower() != 'false',
                   help='Also compute set-level FID (real set vs generated set)')

    # Output
    p.add_argument('--output_dir', default='./eval_out')
    p.add_argument('--save_pairs', default=True,
                   type=lambda x: str(x).lower() != 'false',
                   help='Save real|generated side-by-side PNGs')
    return p.parse_args()


# =============================================================================
# Dataset (held-out)
# =============================================================================

def _load_patient_whitelist(dp_split_json, group):
    if not dp_split_json:
        return None
    import json
    with open(dp_split_json, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    assignment = manifest.get('assignment', manifest)
    patients = [pid for pid, g in assignment.items() if g == group]
    print(f"[dp_split] group='{group}'  patients={len(patients)}")
    return patients or None


def build_eval_dataset(args):
    ds_args = argparse.Namespace(
        root_path         = args.root_path,
        split_csv         = args.split_csv,
        split             = args.eval_split,
        image_size        = args.image_size,
        max_length        = args.max_length,
        patient_whitelist = _load_patient_whitelist(args.dp_split_json,
                                                    args.dp_split_group),
    )
    return MIMICCXRDataset(ds_args)


# =============================================================================
# Metrics
# =============================================================================

def _load_lpips(device):
    """Return an LPIPS callable(real, gen)->[B] or None if unavailable."""
    try:
        from Loss.lpips import LPIPS
        net = LPIPS().to(device).eval()
        try:
            net.load_from_pretrained()          # taming vgg_lpips weights
        except Exception:
            pass                                # vgg16 backbone still usable
        for p_ in net.parameters():
            p_.requires_grad = False

        def _lpips(real, gen):
            # LPIPS expects 3-channel [-1,1]
            r = real.repeat(1, 3, 1, 1) if real.shape[1] == 1 else real
            g = gen.repeat(1, 3, 1, 1) if gen.shape[1] == 1 else gen
            with torch.no_grad():
                d = net(r, g)
            return d.view(-1)
        return _lpips
    except Exception as e:
        print(f'[lpips] unavailable ({e}) ? skipping LPIPS')
        return None


def _resize_to(gen, size):
    if gen.shape[-1] != size or gen.shape[-2] != size:
        gen = F.interpolate(gen, size=(size, size), mode='bilinear',
                            align_corners=False)
    return gen


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ldm, embedder = build_and_load_ldm(args, device)
    dataset = build_eval_dataset(args)

    n_eval = len(dataset) if args.max_eval < 0 else min(args.max_eval, len(dataset))
    print(f'[eval] pairs={n_eval}  n_samples/desc={args.n_samples}  '
          f'metrics={args.metrics}  fid={args.compute_fid}')

    lpips_fn = _load_lpips(device) if 'lpips' in args.metrics else None

    os.makedirs(args.output_dir, exist_ok=True)
    pairs_dir = os.path.join(args.output_dir, 'pairs')
    if args.save_pairs:
        os.makedirs(pairs_dir, exist_ok=True)

    rows = []
    fid_real, fid_gen = [], []

    for i in range(n_eval):
        real_img, report = dataset[i]                       # [1,H,W], str
        real = real_img.unsqueeze(0).to(device)             # [1,1,H,W] in [-1,1]

        with torch.no_grad():
            c = embedder([report]).to(device).repeat(args.n_samples, 1, 1)
            gen = ldm.sample(c, batch_size=args.n_samples,
                             verbose=False, timesteps=args.sample_timesteps)
        gen = _resize_to(gen, args.image_size)              # [n,1,H,W] in [-1,1]
        real_rep = real.repeat(args.n_samples, 1, 1, 1)     # [n,1,H,W]

        rec = {'index': i, 'description': report}

        if 'ssim' in args.metrics:
            s = compute_ssim(real_rep, gen)                 # [n]
            rec['ssim_best'] = float(s.max()); rec['ssim_mean'] = float(s.mean())
        if 'psnr' in args.metrics:
            ps = compute_psnr(real_rep, gen)                # [n]
            rec['psnr_best'] = float(ps.max()); rec['psnr_mean'] = float(ps.mean())
        if lpips_fn is not None:
            lp = lpips_fn(real_rep, gen)                    # [n]  (lower better)
            rec['lpips_best'] = float(lp.min()); rec['lpips_mean'] = float(lp.mean())

        # pick the representative sample (best ssim if available, else first)
        if 'ssim' in args.metrics:
            best_k = int(compute_ssim(real_rep, gen).argmax())
        else:
            best_k = 0
        best_gen = gen[best_k:best_k + 1]

        if args.compute_fid:
            fid_real.append(real.cpu())
            fid_gen.append(best_gen.cpu())

        if args.save_pairs:
            from PIL import Image
            pair = np.concatenate([_to_uint8(real[0]), _to_uint8(best_gen[0])], axis=1)
            Image.fromarray(pair, mode='L').save(
                os.path.join(pairs_dir, f'{i:04d}_real_gen.png'))

        rows.append(rec)
        if (i + 1) % 10 == 0 or i == n_eval - 1:
            print(f'  [{i+1}/{n_eval}] {report[:45]!r}')

    # ¦¡¦¡ Per-pair CSV ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    fieldnames = ['index', 'description']
    for m in ('ssim', 'psnr', 'lpips'):
        if any(f'{m}_best' in r for r in rows):
            fieldnames += [f'{m}_best', f'{m}_mean']
    csv_path = os.path.join(args.output_dir, 'eval_pairs.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)

    # ¦¡¦¡ Aggregate summary ¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡¦¡
    print('\n' + '=' * 52)
    print(f'Evaluation summary  (n={len(rows)}, held-out split={args.eval_split})')
    print('=' * 52)
    summary = {'n': len(rows), 'eval_split': args.eval_split}
    for key in fieldnames:
        if key in ('index', 'description'):
            continue
        vals = np.array([r[key] for r in rows if key in r], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            m, sd = float(vals.mean()), float(vals.std())
            summary[key] = m
            print(f'  {key:12s}: {m:.4f} ¡¾ {sd:.4f}')

    if args.compute_fid and len(fid_real) >= 2:
        try:
            from Eval_metric.fid import InceptionV3Features, compute_fid, to_float_rgb
            inception = InceptionV3Features().to(device).eval()

            def _feats(img_list):
                feats = []
                with torch.no_grad():
                    for im in img_list:
                        x = to_float_rgb(im.to(device))          # [1,3,H,W] in [0,1]
                        x = F.interpolate(x, size=(299, 299), mode='bilinear',
                                          align_corners=False)   # InceptionV3 input
                        feats.append(inception(x).cpu().numpy())
                return np.concatenate(feats, axis=0)

            fid = compute_fid(_feats(fid_real), _feats(fid_gen))
            summary['fid'] = float(fid)
            print(f'  {"fid":12s}: {fid:.4f}  (real-set vs generated-set)')
        except Exception as e:
            print(f'  [fid] skipped ({e})')

    import json
    with open(os.path.join(args.output_dir, 'eval_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'\n[save] per-pair  -> {csv_path}')
    print(f'[save] summary   -> {os.path.join(args.output_dir, "eval_summary.json")}')
    if args.save_pairs:
        print(f'[save] pairs     -> {pairs_dir}')
    print('\nNote: SSIM/PSNR are pixel-aligned proxies and read low because '
          'generation != reconstruction. LPIPS/FID are more meaningful.')


if __name__ == '__main__':
    main()