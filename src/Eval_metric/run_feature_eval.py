"""
Eval_metric/run_feature_eval.py  —  Feature-based eval driver (Eval steps 1-3)
==============================================================================
Extracts features ONCE (cached) from a real held-out image folder and a
generated image folder, then computes every feature-based metric from that SAME
cache:
    - FDS  (Feature KL divergence)          [Eval_metric/fds.py]
    - T-SNE feature-space visualization      [Eval_metric/tsne_viz.py]
    - FID  (optional, reuses Eval_metric/fid.compute_fid)

Output layout (matches the requested convention)
------------------------------------------------
    ./eval/eps<N>/
        eval_summary.json     # merged metrics (fds_*, fid, ...); created/updated
        fds.json              # standalone FDS result
        tsne.png              # real-vs-generated scatter
        tsne_embedding.npz    # 2-D embedding for re-plotting
        ckpt_info.json        # args/epsilon/lora the ckpt was produced with (#3)
        _features/            # cached real_<model>.npz, gen_<model>.npz

Real reference source (pick one)
--------------------------------
  (a) --real_dir <folder>  : a folder of real PNGs.
  (b) dataset mode         : omit --real_dir and pass --root_path <MIMIC root>;
      real images are loaded straight from the dataset by --eval_split
      (train/validate/test) — no separate test_pngs folder needed.

Usage
-----
    # dataset mode (real loaded by split=test)
    python Eval_metric/run_feature_eval.py \\
        --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \\
        --eval_split test --max_real 500 \\
        --gen_dir   ./generated_eps5/samples \\
        --out_dir   ./eval/eps5 \\
        --eval_model xrv --image_size 256 \\
        --device cuda:0 \\
        --ckpt      ./finetune_dp/ldm_lora_eps5.pt \\
        --metrics fds tsne fid

    # or folder mode
    python Eval_metric/run_feature_eval.py \\
        --real_dir ./real_test_pngs --gen_dir ./generated_eps5/samples \\
        --out_dir ./eval/eps5 --eval_model xrv --metrics fds tsne fid
"""

import os
import sys
import csv
import json
import argparse

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # allow `python Eval_metric/run_feature_eval.py`


def dump_ckpt_info(ckpt_path, out_dir):
    """Record which training args produced this checkpoint (requirement #3)."""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f'[ckpt_info] skip (no ckpt at {ckpt_path})')
        return None
    import torch
    # Our own checkpoints store non-tensor objects (args dict, numpy scalar
    # epsilon_spent). PyTorch >=2.6 defaults weights_only=True and rejects them,
    # so load with weights_only=False (trusted: produced by our training script).
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location='cpu')   # older PyTorch
    info = {'ckpt_path': os.path.abspath(ckpt_path)}
    for k in ('epoch', 'global_step', 'epsilon_spent', 'lora_rank', 'lora_alpha'):
        if isinstance(ckpt, dict) and k in ckpt:
            info[k] = ckpt[k]
    if isinstance(ckpt, dict) and 'args' in ckpt:
        info['args'] = ckpt['args']            # vars(args) saved at train time
    path = os.path.join(out_dir, 'ckpt_info.json')
    with open(path, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    print(f'[ckpt_info] -> {path}')
    return info


def _load_lpips(device):
    """LPIPS callable(real, gen)->[B], or None if weights/lib unavailable."""
    try:
        import torch
        from Loss.lpips import LPIPS
        net = LPIPS().to(device).eval()
        try:
            net.load_from_pretrained()
        except Exception:
            pass
        for p_ in net.parameters():
            p_.requires_grad = False

        def _lpips(real, gen):
            r = real.repeat(1, 3, 1, 1) if real.shape[1] == 1 else real
            g = gen.repeat(1, 3, 1, 1) if gen.shape[1] == 1 else gen
            with torch.no_grad():
                return net(r, g).view(-1)
        return _lpips
    except Exception as e:
        print(f'[lpips] unavailable ({e})')
        return None


def _load_gen_index_pairs(gen_dir):
    """[(abs_image_path, prompt_index), ...] from descriptions.csv (gen_dir or parent)."""
    for base in (gen_dir, os.path.dirname(os.path.abspath(gen_dir))):
        csv_path = os.path.join(base, 'descriptions.csv')
        if os.path.isfile(csv_path):
            pairs = []
            with open(csv_path, newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    fp = os.path.join(base, row['file'])
                    if os.path.isfile(fp) and row.get('index', '').strip() != '':
                        pairs.append((fp, int(row['index'])))
            if pairs:
                return pairs
    return None


def compute_paired_pixel_metrics(args, which):
    """
    Paired SSIM/PSNR/LPIPS: generated image i vs the real image of report i.
    Requires (a) dataset mode (--root_path) for the real images and (b) the
    generation prompts to correspond to the eval split BY INDEX
    (--paired_from_split, auto-set when generate_and_eval uses --prompt_source
    split). Otherwise the index->real mapping is meaningless, so we skip.
    """
    if not getattr(args, 'paired_from_split', False):
        print('[paired] skipped (need --paired_from_split: gen index must map to '
              'the eval split; only true when prompts came from the split)')
        return {}
    if not args.root_path:
        print('[paired] skipped (need --root_path dataset mode for the paired real image)')
        return {}
    pairs = _load_gen_index_pairs(args.gen_dir)
    if not pairs:
        print('[paired] skipped (descriptions.csv with index column not found)')
        return {}
    import torch
    from Data.mimic_cxr import MIMICCXRDataset
    from Eval_metric.features import load_image_as_tensor
    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
        image_size=args.image_size, max_length=args.max_length, patient_whitelist=None))
    valid = [(fp, idx) for fp, idx in pairs if 0 <= idx < len(ds)]
    if not valid:
        print('[paired] skipped (no gen index falls within the split range)')
        return {}
    ssim_fn = psnr_fn = lpips_fn = None
    if 'ssim' in which:
        from Eval_metric.ssim import compute_ssim as ssim_fn
    if 'psnr' in which:
        from Eval_metric.psnr import compute_psnr as psnr_fn
    if 'lpips' in which:
        lpips_fn = _load_lpips(args.device)
    acc = {'ssim': [], 'psnr': [], 'lpips': []}
    bs = max(1, args.batch_size)
    for i in range(0, len(valid), bs):
        chunk = valid[i:i + bs]
        real = torch.cat([ds[idx][0].unsqueeze(0) for _, idx in chunk], 0).to(args.device)
        gen = torch.cat([load_image_as_tensor(fp, args.image_size) for fp, _ in chunk], 0).to(args.device)
        if ssim_fn is not None:
            acc['ssim'].append(ssim_fn(real, gen).detach().cpu())
        if psnr_fn is not None:
            acc['psnr'].append(psnr_fn(real, gen).detach().cpu())
        if lpips_fn is not None:
            acc['lpips'].append(lpips_fn(real, gen).detach().cpu())
    out = {'n_paired': len(valid)}
    for k in ('ssim', 'psnr', 'lpips'):
        if acc[k]:
            t = torch.cat(acc[k])
            t = t[torch.isfinite(t)]
            out[f'{k}_mean'] = float(t.mean())
            out[f'{k}_std'] = float(t.std())
    print('[paired] ' + '  '.join(f'{k}={out.get(k + "_mean"):.4f}'
                                   for k in ('ssim', 'psnr', 'lpips') if f'{k}_mean' in out)
          + f'  (n={out["n_paired"]})')
    return out


def _merge_summary(out_dir, new_fields):
    """Create or update eval_summary.json in place (keeps existing metrics)."""
    path = os.path.join(out_dir, 'eval_summary.json')
    summary = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                summary = json.load(f)
        except Exception:
            summary = {}
    summary.update(new_fields)
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'[summary] -> {path}')
    return summary


def load_real_features(args, cache_path):
    """
    Real reference features from EITHER:
      (a) --real_dir : a folder of real PNGs, OR
      (b) the MIMIC dataset, loaded by split (train/val/test) — no PNG dump
          needed; images come straight from the dataset via args.eval_split.
    Returns np.ndarray [N, D].
    """
    from Eval_metric.features import extract_features, extract_features_from_tensors
    if args.real_dir:
        feats, _ = extract_features(args.real_dir, args.eval_model, args.device,
                                    args.image_size, args.batch_size, cache_path=cache_path)
        return feats

    # dataset mode — mirror LDM_dp_eval.build_eval_dataset
    from Data.mimic_cxr import MIMICCXRDataset
    whitelist = None
    if args.dp_split_json:
        with open(args.dp_split_json, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        whitelist = manifest.get(args.dp_split_group, manifest.get('groups', {}).get(args.dp_split_group))
    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
        image_size=args.image_size, max_length=args.max_length,
        patient_whitelist=whitelist))
    n = len(ds) if args.max_real in (None, 0) else min(args.max_real, len(ds))
    print(f'[real] dataset split={args.eval_split}  using {n}/{len(ds)} images')
    images = [ds[i][0] for i in range(n)]          # (image, report) -> image [1,H,W]
    cid = f'{args.eval_split}:{n}:{args.image_size}'
    return extract_features_from_tensors(images, args.eval_model, args.device,
                                         args.batch_size, cache_path=cache_path, cache_id=cid)


def parse_args():
    p = argparse.ArgumentParser(description='Feature-based eval driver (steps 1-3)')
    # --- real reference source: EITHER --real_dir OR dataset (--root_path + --eval_split) ---
    p.add_argument('--real_dir', default=None,
                   help='folder of held-out REAL pngs. Omit to load from the '
                        'MIMIC dataset by split (--root_path/--eval_split) instead.')
    p.add_argument('--root_path', default=None,
                   help='MIMIC root (dataset mode, when --real_dir is not given)')
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test', help='train/validate/test')
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--max_real', default=None, type=int,
                   help='cap number of real images (default: all in the split)')
    p.add_argument('--dp_split_json', default=None,
                   help='optional dp_splits.json to restrict real patients')
    p.add_argument('--dp_split_group', default='train')
    p.add_argument('--gen_dir', required=True, help='folder of GENERATED pngs (e.g. ./generated_eps5/samples)')
    p.add_argument('--out_dir', required=True, help='e.g. ./eval/eps5')
    p.add_argument('--eval_model', default='xrv', choices=['inception', 'xrv'])
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--metrics', nargs='+', default=['fds', 'tsne'],
                   choices=['fds', 'tsne', 'fid', 'clip', 'ssim', 'psnr', 'lpips',
                            'label', 'clip_diagnose'])
    # clip_diagnose (opt-in): CLIP evaluator validity checks on REAL images
    p.add_argument('--clip_prompt_template', default='{label}',
                   help="zero-shot prompt template, e.g. 'findings consistent with {label}'")
    p.add_argument('--clip_bootstrap', default=2000, type=int)
    p.add_argument('--clip_permutations', default=5000, type=int)
    p.add_argument('--paired_from_split', action='store_true',
                   help='enable paired ssim/psnr/lpips AND label-agreement: gen image '
                        'index maps to the eval-split real image / study (only valid '
                        'when prompts came from the split)')
    # label-agreement (metric 'label'): XRV DenseNet AUROC vs CheXpert GT
    p.add_argument('--xrv_weights', nargs='+',
                   default=['densenet121-res224-all', 'densenet121-res224-nih'])
    p.add_argument('--label_min_pos', default=10, type=int,
                   help='min positives+negatives for a pathology to enter label macro AUROC')
    p.add_argument('--chexpert_csv', default=None, help='override CheXpert csv path')
    # CLIPScore (metric 'clip'): text-image alignment via a domain encoder
    p.add_argument('--clip_backend', default='medclip',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    p.add_argument('--clip_model', default='ViT-B-32', help='open_clip model (cxr-clip/openclip)')
    p.add_argument('--clip_pretrained', default='openai', help='open_clip weights or ckpt path')
    p.add_argument('--clip_w', default=2.5, type=float)
    from Eval_metric.text_utils import SECTION_MODES
    p.add_argument('--clip_text_mode', default='FINDINGS/IMPRESSION', choices=SECTION_MODES,
                   help='CLIP text sections (match generation --text_mode)')
    p.add_argument('--clip_retrieval_ks', nargs='+', type=int, default=[1, 5, 10],
                   help='CLIP retrieval R@k cutoffs (empty to disable)')
    p.add_argument('--clip_dedup', action='store_true',
                   help='CLIP retrieval on unique-prompt subset (fairer with repeats)')
    p.add_argument('--clip_batch_size', default=32, type=int)
    p.add_argument('--reg', default=1e-6, type=float)
    p.add_argument('--fds_cov', default='lw', choices=['lw', 'empirical'],
                   help="FDS covariance estimator (lw=Ledoit-Wolf, robust when n<dim)")
    p.add_argument('--pca_dim', default=None, type=int,
                   help='FDS: reduce features to this dim (PCA on real) first, e.g. 64')
    p.add_argument('--perplexity', default=30.0, type=float)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--ckpt', default=None, help='ckpt to record args from (#3)')
    return p.parse_args()


def run(args):
    """Run feature-based eval (steps 1-3) for a config namespace.

    Reused by both the CLI (main) and the generate+eval orchestrator, so the
    two entry points share one implementation. `args` needs the fields set by
    parse_args() below.
    """
    if not args.real_dir and not args.root_path:
        raise SystemExit('provide a real reference: either --real_dir <folder> '
                         'or --root_path <MIMIC root> (dataset mode, uses --eval_split)')
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = os.path.join(args.out_dir, '_features')
    real_cache = os.path.join(cache_dir, f'real_{args.eval_model}.npz')
    gen_cache = os.path.join(cache_dir, f'gen_{args.eval_model}.npz')

    from Eval_metric.features import extract_features
    # Step 1: extract features ONCE (cached); every metric below reuses these.
    feats_real = load_real_features(args, real_cache)
    feats_gen, _ = extract_features(args.gen_dir, args.eval_model, args.device,
                                    args.image_size, args.batch_size, cache_path=gen_cache)
    print(f'[driver] real={feats_real.shape}  gen={feats_gen.shape}  model={args.eval_model}')

    merged = {'eval_model': args.eval_model,
              'n_real': int(len(feats_real)), 'n_gen': int(len(feats_gen))}

    if 'fds' in args.metrics:
        from Eval_metric.fds import compute_fds
        pca_dim = getattr(args, 'pca_dim', None)
        fds = compute_fds(feats_real, feats_gen, reg=args.reg,
                          cov=getattr(args, 'fds_cov', 'lw'), pca_dim=pca_dim)
        fds.update({'eval_model': args.eval_model,
                    'n_real': int(len(feats_real)), 'n_gen': int(len(feats_gen)),
                    'feature_dim': int(feats_real.shape[1]),
                    'pca_dim': int(pca_dim) if pca_dim else None})
        with open(os.path.join(args.out_dir, 'fds.json'), 'w') as f:
            json.dump(fds, f, indent=2)
        print(f'[fds] gen||real={fds["fds_gen_given_real"]:.4f}  '
              f'real||gen={fds["fds_real_given_gen"]:.4f}  sym={fds["fds_symmetric"]:.4f}')
        merged.update({k: fds[k] for k in
                       ('fds_gen_given_real', 'fds_real_given_gen', 'fds_symmetric')})

    if 'fid' in args.metrics:
        try:
            from Eval_metric.fid import compute_fid
            fid = float(compute_fid(feats_real, feats_gen))
            merged['fid'] = fid
            merged['fid_backbone'] = args.eval_model
            print(f'[fid] {fid:.4f}  (backbone={args.eval_model})')
        except Exception as e:
            print(f'[fid] skipped ({e})')

    if 'tsne' in args.metrics:
        import numpy as np
        from Eval_metric.tsne_viz import run_tsne, plot_source
        feats = np.concatenate([feats_real, feats_gen], 0).astype(np.float64)
        emb, perp = run_tsne(feats, args.perplexity, args.seed)
        out_png = os.path.join(args.out_dir, 'tsne.png')
        plot_source(emb, len(feats_real), out_png,
                    f't-SNE ({args.eval_model}, perp={perp:.0f}) real vs generated')
        np.savez(os.path.join(args.out_dir, 'tsne_embedding.npz'),
                 emb=emb, n_real=len(feats_real), eval_model=args.eval_model, perplexity=perp)
        merged['tsne_perplexity'] = float(perp)
        print(f'[tsne] -> {out_png}')

    pixel_wanted = [m for m in ('ssim', 'psnr', 'lpips') if m in args.metrics]
    if pixel_wanted:
        try:
            merged.update(compute_paired_pixel_metrics(args, pixel_wanted))
        except Exception as e:
            print(f'[paired] skipped ({type(e).__name__}: {e})')

    if 'label' in args.metrics:
        try:
            from Eval_metric.downstream_cls import compute_label_agreement
            la = compute_label_agreement(
                args, roc_png=os.path.join(args.out_dir, 'label_auroc_roc.png'))
            if la:
                with open(os.path.join(args.out_dir, 'label_agreement.json'), 'w') as f:
                    json.dump(la, f, indent=2)
                for short, r in la.items():   # macro summary only (per-pathology in json)
                    for k in ('auroc_gen_macro', 'auroc_real_macro',
                              'auroc_gap_macro', 'auroc_ratio_macro'):
                        merged[f'label_{short}_{k}'] = r.get(k)
        except Exception as e:
            print(f'[label] skipped ({type(e).__name__}: {e})')

    if 'clip' in args.metrics:
        try:
            from Eval_metric.clipscore import (load_encoder, clipscore,
                                               load_pairs_from_csv, real_baseline)
            ks = tuple(getattr(args, 'clip_retrieval_ks', None) or ())
            tmode = getattr(args, 'clip_text_mode', 'FINDINGS/IMPRESSION')
            dedup = getattr(args, 'clip_dedup', False)
            enc = load_encoder(args.clip_backend, args.device,
                               getattr(args, 'clip_model', 'ViT-B-32'),
                               getattr(args, 'clip_pretrained', 'openai'),
                               batch_size=getattr(args, 'clip_batch_size', 32))
            paths, prompts = load_pairs_from_csv(args.gen_dir)
            clip_res = {'clip_backend': enc.name, 'clip_text_mode': tmode}
            gen = clipscore(paths, prompts, enc, w=args.clip_w, text_mode=tmode,
                            retrieval_ks=ks, dedup=dedup)
            clip_res.update({f'clip_gen_{k}': v for k, v in gen.items()})
            print(f'[clip] gen cos_mean={gen["cos_mean"]:.4f} (primary)  '
                  f'shuffled={gen["cos_shuffled_mean"]:.4f}  signal={gen["cos_signal"]:.4f}  n={gen["n"]}')
            if args.root_path:                 # dataset mode -> real ceiling + gap
                real = real_baseline(args.root_path, args.split_csv, args.eval_split,
                                     args.image_size, args.max_length, enc,
                                     max_real=args.max_real, w=args.clip_w,
                                     tmp_dir=os.path.join(args.out_dir, '_clip_real_tmp'),
                                     text_mode=tmode, retrieval_ks=ks, dedup=dedup)
                clip_res.update({f'clip_real_{k}': v for k, v in real.items()})
                clip_res['clip_gap_cos'] = float(real['cos_mean'] - gen['cos_mean'])
                clip_res['clip_gap_clipscore'] = float(real['clipscore_mean'] - gen['clipscore_mean'])
                print(f'[clip] real cos_mean={real["cos_mean"]:.4f}  '
                      f'gap_cos={clip_res["clip_gap_cos"]:.4f}')
            with open(os.path.join(args.out_dir, 'clipscore.json'), 'w') as f:
                json.dump(clip_res, f, indent=2)
            merged.update(clip_res)
        except Exception as e:
            print(f'[clip] skipped ({type(e).__name__}: {e})')

    if 'clip_diagnose' in args.metrics:
        try:
            from Eval_metric.clip_diagnose import run_diagnose
            dcfg = argparse.Namespace(
                backend=args.clip_backend, clip_model=getattr(args, 'clip_model', 'ViT-B-32'),
                clip_pretrained=getattr(args, 'clip_pretrained', 'openai'),
                batch_size=getattr(args, 'clip_batch_size', 32), device=args.device,
                root_path=args.root_path, split_csv=args.split_csv, eval_split=args.eval_split,
                chexpert_csv=getattr(args, 'chexpert_csv', None), image_size=args.image_size,
                max_length=args.max_length, max_real=args.max_real,
                text_mode=getattr(args, 'clip_text_mode', 'FINDINGS/IMPRESSION'),
                prompt_template=getattr(args, 'clip_prompt_template', '{label}'),
                retrieval_ks=(getattr(args, 'clip_retrieval_ks', None) or [1, 5, 10]),
                bootstrap=getattr(args, 'clip_bootstrap', 2000),
                permutations=getattr(args, 'clip_permutations', 5000),
                output=os.path.join(args.out_dir, 'clip_diagnose.json'))
            diag = run_diagnose(dcfg)
            with open(os.path.join(args.out_dir, 'clip_diagnose.json'), 'w') as f:
                json.dump(diag, f, indent=2)
            for k in ('signal_mean', 'signal_ci_low', 'signal_ci_high',
                      'signal_ci_excludes_zero', 'signal_perm_pvalue',
                      'zeroshot_label_auroc_macro'):
                merged[f'clipdiag_{k}'] = diag.get(k)
            if diag.get('label_retrieval_i2t'):
                merged['clipdiag_label_retrieval_mAP_i2t'] = diag['label_retrieval_i2t'].get('mAP')
        except Exception as e:
            print(f'[clip_diagnose] skipped ({type(e).__name__}: {e})')

    summary = _merge_summary(args.out_dir, merged)
    dump_ckpt_info(args.ckpt, args.out_dir)
    return summary


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
