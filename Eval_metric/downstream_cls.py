"""
Eval_metric/downstream_cls.py  —  Downstream label-agreement (Eval step 5a)
===========================================================================
Concept
-------
"Does a generated image actually contain the pathology its prompt implied?"
We run a PRE-TRAINED CXR classifier (TorchXRayVision DenseNet-121, no training
needed) on the generated images and score its predictions against the study's
ground-truth CheXpert labels (mimic-cxr-2.0.0-chexpert.csv). The SAME classifier
is run on the paired REAL images to get an empirical ceiling; the per-classifier
gap is the fair, backbone-agnostic signal.

    auroc_gen  = AUROC( classifier(generated) , CheXpert GT )
    auroc_real = AUROC( classifier(real)      , CheXpert GT )   # ceiling
    gap = real - gen   (down = better) ;  ratio = gen / real

Two DenseNet weight sets cover both training domains of the project:
    densenet121-res224-nih : NIH (the VAE/LDM PRE-TRAIN domain) — out-of-domain
                             for MIMIC, so a stricter, more independent probe.
    densenet121-res224-all : NIH+CheXpert+MIMIC+… — in-domain, higher ceiling.
NOTE: 'all' shares its backbone with the FID feature extractor, so treat it as
less independent than 'nih'. Absolute AUROC is NOT comparable across weight sets
— compare each classifier's gen against its OWN real baseline.

Design choices (per user):
- Uncertain CheXpert labels (-1) are DROPPED (excluded from that pathology).
- MedCLIP zero-shot classifier is intentionally excluded for now.

Label harmonization is pure evaluation-side name mapping (no retraining):
XRV pathology name -> CheXpert column. A pathology is scored only if the weight
set actually predicts it (op_threshs not NaN) AND it maps to a CheXpert column
AND both classes are present after dropping uncertain labels.
"""

import os
import csv
import json
import argparse

import numpy as np


# XRV default pathology name -> CheXpert CSV column name
XRV_TO_CHEXPERT = {
    'Atelectasis': 'Atelectasis',
    'Cardiomegaly': 'Cardiomegaly',
    'Consolidation': 'Consolidation',
    'Edema': 'Edema',
    'Effusion': 'Pleural Effusion',
    'Pneumonia': 'Pneumonia',
    'Pneumothorax': 'Pneumothorax',
    'Lung Opacity': 'Lung Opacity',
    'Lung Lesion': 'Lung Lesion',
    'Fracture': 'Fracture',
    'Enlarged Cardiomediastinum': 'Enlarged Cardiomediastinum',
}

DEFAULT_XRV_WEIGHTS = ['densenet121-res224-all', 'densenet121-res224-nih']


# ── ground truth ─────────────────────────────────────────────────────────────
def find_chexpert_csv(root_path, override=None):
    if override:
        return override
    for name in ('mimic-cxr-2.0.0-chexpert.csv.gz', 'mimic-cxr-2.0.0-chexpert.csv'):
        p = os.path.join(root_path, name)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f'CheXpert CSV not found under {root_path} '
        '(mimic-cxr-2.0.0-chexpert.csv[.gz]); pass --chexpert_csv.')


def load_chexpert_gt(csv_path):
    """dict[(subject_id:int, study_id:int)] -> {chexpert_col: float|nan}."""
    import pandas as pd
    df = pd.read_csv(csv_path)                     # pandas reads .gz transparently
    label_cols = [c for c in df.columns if c not in ('subject_id', 'study_id')]
    # Use numpy arrays keyed by ORIGINAL column names — itertuples()._asdict()
    # mangles names with spaces (e.g. 'Enlarged Cardiomediastinum').
    subj = df['subject_id'].astype(int).to_numpy()
    stud = df['study_id'].astype(int).to_numpy()
    vals = df[label_cols].to_numpy(dtype=float)    # blanks -> NaN
    gt = {}
    for i in range(len(df)):
        gt[(int(subj[i]), int(stud[i]))] = {
            label_cols[j]: vals[i, j] for j in range(len(label_cols))}
    return gt


# ── classifier ───────────────────────────────────────────────────────────────
def load_xrv_classifier(weights, device):
    import torchxrayvision as xrv
    model = xrv.models.DenseNet(weights=weights).to(device).eval()
    return model


def _resize_224(t):
    """[1,H,W] or [1,1,H,W] in [-1,1] -> [1,1,224,224]."""
    import torch.nn.functional as F
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return F.interpolate(t, size=(224, 224), mode='bilinear', align_corners=False)


def classify(model, tensors, device, batch_size=16):
    """tensors: list of [1,1,224,224] in [-1,1]. Returns probs [N, 18]."""
    import torch
    probs = []
    with torch.no_grad():
        for i in range(0, len(tensors), batch_size):
            x = torch.cat(tensors[i:i + batch_size], 0).to(device)
            x = x.clamp(-1., 1.) * 1024.           # xrv normalize convention
            probs.append(model(x).cpu().numpy())
    return np.concatenate(probs, 0)


def _valid_pathology(model, idx):
    """True if this weight set actually predicts pathology idx (op_threshs set)."""
    ot = getattr(model, 'op_threshs', None)
    if ot is None:
        return True
    try:
        return not bool(np.isnan(float(ot[idx])))
    except Exception:
        return True


def auroc_per_pathology(probs, model, gt, keys, min_pos=10):
    """
    probs [N,18], keys [(subject_id, study_id)] per row.
    Returns (per_pathology_auroc, macro, support, macro_pathologies).
    Uncertain (-1) and missing (NaN) GT are dropped per pathology.

    per-pathology AUROC is reported for EVERY scorable pathology, but `macro`
    only averages the RELIABLE ones — those with at least `min_pos` positives AND
    `min_pos` negatives — since AUROC on 1-2 positives is pure noise (e.g. 1.0 /
    0.5). `macro_pathologies` lists which were included.
    """
    from sklearn.metrics import roc_auc_score
    pathologies = list(model.pathologies)
    per, support = {}, {}
    for i, pname in enumerate(pathologies):
        if pname not in XRV_TO_CHEXPERT or not _valid_pathology(model, i):
            continue
        col = XRV_TO_CHEXPERT[pname]
        y_true, y_score = [], []
        for r, key in enumerate(keys):
            row = gt.get(key)
            if row is None or col not in row:
                continue
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue                            # no label
            if float(v) == -1.0:
                continue                            # uncertain -> drop
            y_true.append(1 if float(v) == 1.0 else 0)
            y_score.append(float(probs[r, i]))
        if len(set(y_true)) < 2:                    # need both classes for AUROC
            continue
        n_pos = int(sum(y_true))
        per[pname] = float(roc_auc_score(y_true, y_score))
        support[pname] = {'n': len(y_true), 'n_pos': n_pos, 'n_neg': len(y_true) - n_pos}
    included = [p for p in per
               if min(support[p]['n_pos'], support[p]['n_neg']) >= min_pos]
    macro = float(np.mean([per[p] for p in included])) if included else None
    return per, macro, support, included


# ── gen/real image collection (paired by descriptions.csv index) ─────────────
def _load_gen_index_pairs(gen_dir):
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


def _key_of_sample(ds, idx):
    """(subject_id:int, study_id:int) from the dataset sample metadata."""
    s = ds.samples[idx]
    subj = int(str(s['patient_id']).lstrip('p'))
    stid = int(str(s['study_id']).lstrip('s'))
    return (subj, stid)


def compute_label_agreement(args):
    """
    Returns {short_weights: {auroc_gen_macro, auroc_real_macro, auroc_gap_macro,
    auroc_ratio_macro, per_pathology_gen, per_pathology_real, n, weights}}.
    Requires dataset mode + gen index mapping to the split (paired_from_split).
    """
    if not getattr(args, 'paired_from_split', False):
        print('[label] skipped (need --paired_from_split: gen index must map to '
              'the eval split, i.e. prompts came from the split)')
        return {}
    if not args.root_path:
        print('[label] skipped (need --root_path dataset mode)')
        return {}
    pairs = _load_gen_index_pairs(args.gen_dir)
    if not pairs:
        print('[label] skipped (descriptions.csv with index column not found)')
        return {}

    import torch  # noqa: F401
    from Data.mimic_cxr import MIMICCXRDataset
    from Eval_metric.features import load_image_as_tensor
    ds = MIMICCXRDataset(argparse.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
        image_size=args.image_size, max_length=args.max_length, patient_whitelist=None))
    valid = [(fp, idx) for fp, idx in pairs if 0 <= idx < len(ds)]
    if not valid:
        print('[label] skipped (no gen index falls within the split range)')
        return {}

    gt = load_chexpert_gt(find_chexpert_csv(args.root_path, getattr(args, 'chexpert_csv', None)))
    keys = [_key_of_sample(ds, idx) for _, idx in valid]
    gen_imgs = [load_image_as_tensor(fp, 224) for fp, _ in valid]        # [1,1,224,224]
    real_imgs = [_resize_224(ds[idx][0]) for _, idx in valid]

    weights_list = getattr(args, 'xrv_weights', None) or DEFAULT_XRV_WEIGHTS
    bs = max(1, getattr(args, 'batch_size', 16))
    results = {}
    for weights in weights_list:
        try:
            model = load_xrv_classifier(weights, args.device)
        except Exception as e:
            print(f'[label] {weights} skipped ({type(e).__name__}: {e})')
            continue
        p_gen = classify(model, gen_imgs, args.device, bs)
        p_real = classify(model, real_imgs, args.device, bs)
        min_pos = getattr(args, 'label_min_pos', 10)
        gen_per, gen_macro, gen_sup, included = auroc_per_pathology(p_gen, model, gt, keys, min_pos)
        real_per, real_macro, _, _ = auroc_per_pathology(p_real, model, gt, keys, min_pos)
        short = weights.split('-')[-1]              # 'all' / 'nih'
        gap = (real_macro - gen_macro) if (gen_macro is not None and real_macro is not None) else None
        ratio = (gen_macro / real_macro) if (gen_macro and real_macro) else None
        results[short] = {
            'weights': weights, 'min_pos': min_pos,
            'auroc_gen_macro': gen_macro, 'auroc_real_macro': real_macro,
            'auroc_gap_macro': gap, 'auroc_ratio_macro': ratio,
            'macro_pathologies': included,          # reliable set the macro averages
            'per_pathology_gen': gen_per, 'per_pathology_real': real_per,
            'support': gen_sup, 'n': len(valid),
        }
        gm = f'{gen_macro:.4f}' if gen_macro is not None else 'NA'
        rm = f'{real_macro:.4f}' if real_macro is not None else 'NA'
        gp = f'{gap:.4f}' if gap is not None else 'NA'
        print(f'[label] {short:4s} gen_macro={gm}  real_macro={rm}  gap={gp}  '
              f'n={len(valid)}  macro over {len(included)} pathology(ies) '
              f'(min_pos={min_pos}): {included}')
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Downstream label-agreement (XRV DenseNet AUROC)')
    p.add_argument('--gen_dir', required=True, help='generated pngs (samples/) with descriptions.csv')
    p.add_argument('--root_path', required=True, help='MIMIC root (real images + chexpert csv)')
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--chexpert_csv', default=None, help='override CheXpert csv path')
    p.add_argument('--xrv_weights', nargs='+', default=DEFAULT_XRV_WEIGHTS)
    p.add_argument('--label_min_pos', default=10, type=int,
                   help='min positives AND negatives for a pathology to enter macro')
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default=None)
    p.set_defaults(paired_from_split=True)          # standalone assumes split prompts
    return p.parse_args()


def main():
    args = parse_args()
    res = compute_label_agreement(args)
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
