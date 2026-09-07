"""
Eval_metric/build_synth_utility_pool.py
========================================
Builds the fixed real/synthetic "pool" for the from-scratch DenseNet-121
downstream-classification utility experiment (see feature_list.json's
`experiment-synth-utility-densenet-scratch`).

Concept
-------
Draw a single seeded superset pool B (default n=5,500) of train-split studies
(text_mode='LABEL+IMPRESSION', same filtering as LDM training). The first
n_real (default 1,100) rows of B ARE pool A -- reused as the "real" half of
every real-containing training arm (R, R+S1.1k, R+S5.5k), so those arms share
the exact same 1,100 real images. Writes:

  <out_dir>/pool_manifest.csv.gz   -- pool_index,in_pool_a,subject_id,study_id,
                                       real_image_path,prompt_text,<CheXpert 14 cols>
  <out_dir>/label_impression.txt   -- one prompt per line, in pool_index order
                                       (fed straight into LDM_dp_inference.py
                                       --descriptions; its descriptions.csv
                                       'index' column then equals pool_index)

Reuses downstream_cls.py's find_chexpert_csv/load_chexpert_gt (no duplicate
CheXpert-parsing logic) and MIMICCXRDataset's existing LABEL+IMPRESSION
filtering + private _build_text (prompt only, no image decode needed here).
"""

import os
import sys
import argparse

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)   # allow `python Eval_metric/build_synth_utility_pool.py` (namespace-package merge, see CLAUDE.md pitfall #4 -- also needs PYTHONPATH=<repo>/src for Data.*)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest_csv', default='data_manifests/mimic_p10_p12/train_balanced.csv.gz')
    p.add_argument('--root_path', required=True, help='MIMIC root (for CheXpert csv fallback)')
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--chexpert_csv', default=None)
    p.add_argument('--split', default='train')
    p.add_argument('--text_mode', default='LABEL+IMPRESSION')
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--n_pool', default=5500, type=int, help='size of superset pool B')
    p.add_argument('--n_real', default=1100, type=int, help='size of the real subset A (prefix of B)')
    p.add_argument('--seed', default=42, type=int)
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def main():
    args = parse_args()

    import pandas as pd
    from Data.mimic_cxr import MIMICCXRDataset
    from Eval_metric.downstream_cls import find_chexpert_csv, load_chexpert_gt

    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.split,
        manifest_csv=args.manifest_csv, text_mode=args.text_mode,
        image_size=256, max_length=args.max_length, patient_whitelist=None))
    n_total = len(ds)
    if args.n_pool > n_total:
        raise ValueError(f'--n_pool={args.n_pool} exceeds dataset size {n_total}')
    if args.n_real > args.n_pool:
        raise ValueError('--n_real must be <= --n_pool')

    rng = np.random.default_rng(args.seed)
    order = rng.choice(n_total, size=args.n_pool, replace=False)   # pool B, order = generation order

    gt = load_chexpert_gt(find_chexpert_csv(args.root_path, args.chexpert_csv))
    label_cols = sorted({c for row in gt.values() for c in row})

    rows, prompts = [], []
    for pool_idx, ds_idx in enumerate(order):
        ds_idx = int(ds_idx)
        meta = ds.samples[ds_idx]
        subj = int(str(meta['patient_id']).lstrip('p'))
        stud = int(str(meta['study_id']).lstrip('s'))
        prompt = ds._build_text(meta)
        row = {
            'pool_index': pool_idx, 'in_pool_a': pool_idx < args.n_real,
            'ds_index': ds_idx, 'subject_id': subj, 'study_id': stud,
            'real_image_path': meta.get('image_path', meta.get('dcm_path')),
            'prompt_text': prompt,
        }
        gt_row = gt.get((subj, stud), {})
        for c in label_cols:
            row[c] = gt_row.get(c, float('nan'))
        rows.append(row)
        prompts.append(prompt.replace('\n', ' ').replace('\r', ' '))

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'pool_manifest.csv.gz')
    df.to_csv(manifest_path, index=False, compression='gzip')
    prompts_path = os.path.join(args.out_dir, 'label_impression.txt')
    with open(prompts_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(prompts) + '\n')

    n_a = int(df['in_pool_a'].sum())
    print(f'[pool] B={len(df)} (A=first {n_a}) written -> {manifest_path}')
    print(f'[pool] prompts -> {prompts_path}')
    print(f'[pool] label columns ({len(label_cols)}): {label_cols}')
    for c in label_cols:
        col = df[c]
        pos = int((col == 1.0).sum())
        unc = int((col == -1.0).sum())
        neg = int((col == 0.0).sum())
        print(f'  {c:28s} pos={pos:5d} neg={neg:5d} uncertain={unc:5d} '
              f'(pool A: pos={int((df.loc[df.in_pool_a, c]==1.0).sum())})')


if __name__ == '__main__':
    main()
