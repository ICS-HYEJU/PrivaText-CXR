"""
Eval_metric/clip_labelret.py  —  Label-level CLIP alignment (primary CLIP metric)
=================================================================================
Report-level exact-text cosine failed (matched ~= shuffled), but the diagnostic
showed MedCLIP IS functional at the PATHOLOGY level (zero-shot label AUROC macro
~0.68; label-level retrieval i2t R@1 ~0.60). So we evaluate CLIP the way it works:

Setup (per project decision):
  - Keep ONLY single-abnormality studies: exactly ONE positive CheXpert
    abnormality (excludes "No Finding" and "Support Devices", and any study with
    2+ findings). Each kept study therefore has ONE clean label.
  - Text side = the IMPRESSION section of that report.
  - Zero-shot prompt = the label name only (template "{label}").

Two metrics, each computed for GENERATED and paired REAL images:
  1. zero-shot label AUROC — score_i = cos(image, "{label}"); AUROC vs the study's
     single label. macro over labels with enough support. gen vs real -> gap/ratio.
  2. label-level retrieval — relevance = same single label; text = IMPRESSION
     embeddings. R@k / P@k / mAP (i2t).

Requires the generation prompts to map to the eval split (--paired_from_split),
since gen image i is scored with study i's label. Opt-in metric 'clip_label'.
"""

import os
import json
import argparse

import numpy as np

# CheXpert abnormalities (exclude "No Finding" and "Support Devices").
ABNORMALITY_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
]


def single_abnormality(row):
    """Return the single positive abnormality label, or None if 0 or >=2."""
    if not row:
        return None
    pos = []
    for L in ABNORMALITY_LABELS:
        v = row.get(L)
        if v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) == 1.0:
            pos.append(L)
    return pos[0] if len(pos) == 1 else None


def _zeroshot_auroc(img_emb, prompt_emb, scored_labels, labels):
    """Per-label AUROC of cos(image, label-prompt) vs the study's single label."""
    from sklearn.metrics import roc_auc_score
    y_label = np.asarray(labels, dtype=object)
    per = {}
    for k, L in enumerate(scored_labels):
        y = (y_label == L).astype(int)
        if len(set(y.tolist())) < 2:
            continue
        score = img_emb @ prompt_emb[k]
        per[L] = float(roc_auc_score(y, score))
    macro = float(np.mean(list(per.values()))) if per else None
    return per, macro


def run_clip_label(args):
    import argparse as _a
    from PIL import Image
    from Eval_metric.clipscore import load_encoder
    from Eval_metric.clip_diagnose import retrieval_with_relevance
    from Eval_metric.text_utils import extract_report_sections
    from Eval_metric.downstream_cls import (find_chexpert_csv, load_chexpert_gt,
                                            _key_of_sample, _load_gen_index_pairs)
    from Data.mimic_cxr import MIMICCXRDataset

    if not getattr(args, 'paired_from_split', False):
        print('[clip_label] skipped (need --paired_from_split: gen index -> study label)')
        return {}
    pairs = _load_gen_index_pairs(args.gen_dir)
    if not pairs:
        print('[clip_label] skipped (descriptions.csv with index not found)')
        return {}

    ds = MIMICCXRDataset(_a.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
        image_size=args.image_size, max_length=args.max_length, patient_whitelist=None))
    gt = load_chexpert_gt(find_chexpert_csv(args.root_path, getattr(args, 'chexpert_csv', None)))

    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.output or '.')), '_clip_label_tmp')
    os.makedirs(tmp_dir, exist_ok=True)
    gen_paths, real_paths, texts, labels = [], [], [], []
    for gp, idx in pairs:
        if not (0 <= idx < len(ds)):
            continue
        label = single_abnormality(gt.get(_key_of_sample(ds, idx)))
        if label is None:                                # keep single-abnormality only
            continue
        img, report = ds[idx]
        arr = ((img.squeeze(0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype('uint8')
        rp = os.path.join(tmp_dir, f'real_{idx:05d}.png')
        Image.fromarray(arr, mode='L').save(rp)
        gen_paths.append(gp); real_paths.append(rp)
        texts.append(extract_report_sections(report, 'IMPRESSION'))
        labels.append(label)

    n = len(labels)
    if n < 10:
        print(f'[clip_label] skipped (only {n} single-abnormality studies)')
        return {}

    from collections import Counter
    cnt = Counter(labels)
    min_pos = getattr(args, 'clip_label_min_pos', 10)
    scored = [L for L in ABNORMALITY_LABELS if cnt[L] >= min_pos]
    print(f'[clip_label] single-abnormality N={n}  scored labels (>= {min_pos}): '
          f'{[(L, cnt[L]) for L in scored]}')
    if not scored:
        print('[clip_label] skipped (no label meets min support)')
        return {}

    tmpl = getattr(args, 'clip_prompt_template', '{label}')
    enc = load_encoder(args.clip_backend, args.device,
                       getattr(args, 'clip_model', 'ViT-B-32'),
                       getattr(args, 'clip_pretrained', 'openai'),
                       batch_size=getattr(args, 'clip_batch_size', 32))
    gen_img = enc.encode_image(gen_paths)
    real_img = enc.encode_image(real_paths)
    txt_emb = enc.encode_text(texts)                                  # IMPRESSION texts
    prompt_emb = enc.encode_text([tmpl.format(label=L) for L in scored])  # label prompts

    ks = tuple(getattr(args, 'clip_retrieval_ks', None) or [1, 5, 10])
    lab = np.asarray(labels, dtype=object)
    scored_arr = np.asarray(scored, dtype=object)

    # 1. zero-shot label classification: AUROC + retrieval over the K label prompts.
    #    Candidates = the K label prompts; the ONE correct label is relevant, so
    #    R@1 = top-1 accuracy, mAP = mean reciprocal rank. (R@k trivially 1 once k>=K.)
    gen_per, gen_macro = _zeroshot_auroc(gen_img, prompt_emb, scored, labels)
    real_per, real_macro = _zeroshot_auroc(real_img, prompt_emb, scored, labels)
    zs_rel = (lab[:, None] == scored_arr[None, :])          # [N, K] correct label
    gen_zs_ret = retrieval_with_relevance(gen_img @ prompt_emb.T, zs_rel, ks)
    real_zs_ret = retrieval_with_relevance(real_img @ prompt_emb.T, zs_rel, ks)

    # 2. label-level retrieval over IMPRESSION texts (relevance = same single label)
    rel = (lab[:, None] == lab[None, :])
    gen_ret = retrieval_with_relevance(gen_img @ txt_emb.T, rel, ks)
    real_ret = retrieval_with_relevance(real_img @ txt_emb.T, rel, ks)

    gap = (real_macro - gen_macro) if (gen_macro is not None and real_macro is not None) else None
    ratio = (gen_macro / real_macro) if (gen_macro and real_macro) else None
    res = {
        'clip_backend': enc.name, 'text_mode': 'IMPRESSION', 'prompt_template': tmpl,
        'n_single_abnormality': n, 'labels_scored': scored,
        'label_counts': dict(cnt),
        'clipzs_gen_macro': gen_macro, 'clipzs_real_macro': real_macro,
        'clipzs_gap': gap, 'clipzs_ratio': ratio,
        'clipzs_gen_per': gen_per, 'clipzs_real_per': real_per,
        'clipzs_gen_ret': gen_zs_ret, 'clipzs_real_ret': real_zs_ret,   # R@k/P@k/mAP over labels
        'labelret_gen_i2t': gen_ret, 'labelret_real_i2t': real_ret,
    }
    gm = f'{gen_macro:.4f}' if gen_macro is not None else 'NA'
    rm = f'{real_macro:.4f}' if real_macro is not None else 'NA'
    print(f'[clip_label] zero-shot AUROC  gen={gm}  real={rm}  '
          f'gap={gap:.4f}' if gap is not None else
          f'[clip_label] zero-shot AUROC  gen={gm}  real={rm}')
    print(f'[clip_label] zero-shot retrieval (over labels)  '
          f'gen R@1={gen_zs_ret.get("R@1"):.4f} mAP={gen_zs_ret.get("mAP"):.4f}  '
          f'real R@1={real_zs_ret.get("R@1"):.4f} mAP={real_zs_ret.get("mAP"):.4f}')
    print(f'[clip_label] label-retrieval i2t (over IMPRESSION)  gen mAP={gen_ret.get("mAP"):.4f}  '
          f'real mAP={real_ret.get("mAP"):.4f}')
    return res


def parse_args():
    p = argparse.ArgumentParser(description='Label-level CLIP alignment (zero-shot AUROC + retrieval)')
    p.add_argument('--gen_dir', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--chexpert_csv', default=None)
    p.add_argument('--clip_backend', dest='clip_backend', default='medclip',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    p.add_argument('--clip_model', default='ViT-B-32')
    p.add_argument('--clip_pretrained', default='openai')
    p.add_argument('--clip_batch_size', default=32, type=int)
    p.add_argument('--clip_prompt_template', default='{label}')
    p.add_argument('--clip_retrieval_ks', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--clip_label_min_pos', default=10, type=int)
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default=None)
    p.set_defaults(paired_from_split=True)
    return p.parse_args()


def main():
    args = parse_args()
    res = run_clip_label(args)
    print(json.dumps(res, indent=2))
    if args.output and res:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
