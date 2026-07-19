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
    ckpt = torch.load(ckpt_path, map_location='cpu')
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
                   choices=['fds', 'tsne', 'fid'])
    p.add_argument('--reg', default=1e-6, type=float)
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
        fds = compute_fds(feats_real, feats_gen, reg=args.reg)
        fds.update({'eval_model': args.eval_model,
                    'n_real': int(len(feats_real)), 'n_gen': int(len(feats_gen)),
                    'feature_dim': int(feats_real.shape[1])})
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

    summary = _merge_summary(args.out_dir, merged)
    dump_ckpt_info(args.ckpt, args.out_dir)
    return summary


def main():
    run(parse_args())


if __name__ == '__main__':
    main()
