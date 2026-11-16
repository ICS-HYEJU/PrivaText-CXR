"""
Eval_metric/clip_diagnose.py  —  CLIP evaluator validity diagnostics (opt-in)
=============================================================================
The question this answers: is the MedCLIP (or other) cross-modal space actually
FUNCTIONAL on REAL MIMIC image-report pairs? Because report-level cosine came out
matched≈shuffled (signal≈0), we must separate "evaluator/encoder broken" from
"report-level exact matching is just hard".

Runs on REAL images only (that is where validity must hold first) and reports:

  1. sanity      — encoder loaded & deterministic:
                     logit_scale (if any), embedding repeatability (cos≈1).
  2. matched vs shuffled — per-sample d = cos(img,matched) - cos(img,shuffled),
                     with 95% bootstrap CI, sign-flip permutation p-value,
                     effect size, P(matched>shuffled).  Tells if the tiny signal
                     is statistically real.
  3. zero-shot label AUROC (DECISIVE) — short pathology prompts vs CheXpert GT.
                     If these are >>0.5 the encoder works and the report-level
                     failure is a text-formulation issue; if ≈0.5 the encoder /
                     preprocessing is broken.
  4. label-level retrieval — relevance = sharing a positive CheXpert label
                     (lenient) vs the current exact-text relevance. If exact≈0 but
                     label-level is meaningful, CLIP is usable as an AUXILIARY
                     semantic measure.

Opt-in only: `--metrics clip_diagnose` (driver) or run this module directly.
Writes clip_diagnose.json; does not change any existing metric/default.
"""

import os
import json
import argparse

import numpy as np

# 14 MIMIC-CheXpert label columns.
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'No Finding', 'Pleural Effusion', 'Pleural Other', 'Pneumonia',
    'Pneumothorax', 'Support Devices',
]


# ── pure-numpy stats (testable without torch) ────────────────────────────────
def bootstrap_ci(d, B=2000, seed=0, alpha=0.05):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def signflip_pvalue(d, B=5000, seed=0):
    """Paired sign-flip permutation test of H0: mean(d)=0 (two-sided)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    n = len(d)
    ge = 0
    for _ in range(B):
        s = rng.choice([-1.0, 1.0], size=n)
        if abs((d * s).mean()) >= obs:
            ge += 1
    return float((ge + 1) / (B + 1))


def label_matrix(label_rows, labels=CHEXPERT_LABELS):
    """Binary label matrix Y [N, L]: 1 if GT==1 else 0 (uncertain/NaN -> 0)."""
    Y = np.zeros((len(label_rows), len(labels)), dtype=float)
    for i, row in enumerate(label_rows):
        if not row:
            continue
        for j, L in enumerate(labels):
            v = row.get(L)
            if v is not None and not (isinstance(v, float) and np.isnan(v)) and float(v) == 1.0:
                Y[i, j] = 1.0
    return Y


def retrieval_with_relevance(sim, relevant, ks=(1, 5, 10)):
    """
    sim [Nq, Nc] similarity; relevant [Nq, Nc] bool. Query is scored only if it
    has >=1 relevant candidate. Returns R@k, P@k, mAP (relevance-based).
    """
    Nq = sim.shape[0]
    Rhit = {k: 0 for k in ks}
    Psum = {k: 0.0 for k in ks}
    aps, used = [], 0
    for i in range(Nq):
        rel = relevant[i]
        if rel.sum() == 0:
            continue
        used += 1
        order = np.argsort(-sim[i])
        rel_ord = rel[order]
        hits = np.where(rel_ord)[0]
        for k in ks:
            if rel_ord[:k].any():
                Rhit[k] += 1
            Psum[k] += float(rel_ord[:k].sum()) / k
        prec = np.cumsum(rel_ord)[hits] / (hits + 1.0)
        aps.append(float(prec.mean()))
    if used == 0:
        return {}
    out = {f'R@{k}': Rhit[k] / used for k in ks}
    out.update({f'P@{k}': Psum[k] / used for k in ks})
    out['mAP'] = float(np.mean(aps))
    out['n_queries'] = used
    return out


# ── data loading (real images + reports + CheXpert GT) ───────────────────────
def _load_real(args, tmp_dir):
    import argparse as _a
    from PIL import Image
    from Data.mimic_cxr import MIMICCXRDataset
    from Eval_metric.downstream_cls import (find_chexpert_csv, load_chexpert_gt,
                                            _key_of_sample)
    ds = MIMICCXRDataset(_a.Namespace(
        root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
        image_size=args.image_size, max_length=args.max_length, patient_whitelist=None))
    n = len(ds) if args.max_real in (None, 0) else min(args.max_real, len(ds))
    gt = load_chexpert_gt(find_chexpert_csv(args.root_path, args.chexpert_csv))
    os.makedirs(tmp_dir, exist_ok=True)
    paths, reports, rows = [], [], []
    for i in range(n):
        img, report = ds[i]
        arr = ((img.squeeze(0).numpy() * 0.5 + 0.5) * 255).clip(0, 255).astype('uint8')
        fp = os.path.join(tmp_dir, f'real_{i:05d}.png')
        Image.fromarray(arr, mode='L').save(fp)
        paths.append(fp); reports.append(report); rows.append(gt.get(_key_of_sample(ds, i)))
    return paths, reports, rows


# ── the diagnostic ───────────────────────────────────────────────────────────
def run_diagnose(args):
    from Eval_metric.clipscore import (load_encoder, _paired_cos, _shuffled_cos,
                                        retrieval_metrics)
    from Eval_metric.text_utils import extract_report_sections
    from sklearn.metrics import roc_auc_score

    enc = load_encoder(args.backend, args.device, args.clip_model,
                       args.clip_pretrained, batch_size=args.batch_size)
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(args.output or '.')), '_clip_diag_tmp')
    paths, reports, rows = _load_real(args, tmp_dir)
    texts = [extract_report_sections(r, args.text_mode) for r in reports]
    N = len(paths)
    print(f'[diag] backend={enc.name}  real N={N}  text_mode={args.text_mode}')

    img_emb = enc.encode_image(paths)
    txt_emb = enc.encode_text(texts)
    res = {'backend': enc.name, 'text_mode': args.text_mode, 'n_real': N,
           'retrieval_relevance': 'exact_text + chexpert_label'}

    # 1. sanity: logit_scale + repeatability
    import numpy as _np
    ls = getattr(getattr(enc, 'model', None), 'logit_scale', None)
    if ls is not None:
        try:
            res['logit_scale'] = float(ls.detach().exp().cpu())
        except Exception:
            res['logit_scale'] = None
    rep = enc.encode_image(paths[:min(16, N)])
    res['img_embed_repeatability_cos'] = float(
        _np.mean(_np.sum(rep * img_emb[:len(rep)], axis=1)))   # ~1.0 if deterministic

    # 2. matched vs shuffled: statistical test on the tiny signal
    cos_m = _paired_cos(img_emb, txt_emb)
    cos_s = _shuffled_cos(img_emb, txt_emb, texts, seed=0)
    d = cos_m - cos_s
    lo, hi = bootstrap_ci(d, B=args.bootstrap, seed=0)
    res.update({
        'matched_cos_mean': float(cos_m.mean()),
        'shuffled_cos_mean': float(cos_s.mean()),
        'signal_mean': float(d.mean()),
        'signal_ci_low': lo, 'signal_ci_high': hi,
        'signal_ci_excludes_zero': bool(lo > 0 or hi < 0),
        'signal_perm_pvalue': signflip_pvalue(d, B=args.permutations, seed=0),
        'signal_effect_size': float(d.mean() / (d.std() + 1e-8)),
        'frac_matched_gt_shuffled': float((cos_m > cos_s).mean()),
    })

    # 3. zero-shot label AUROC (DECISIVE): short prompt cos vs CheXpert GT
    Y = label_matrix(rows)
    zs = {}
    for j, L in enumerate(CHEXPERT_LABELS):
        y = Y[:, j]
        valid = []
        for i, row in enumerate(rows):                # drop missing / uncertain
            v = row.get(L) if row else None
            valid.append(v is not None and not (isinstance(v, float) and np.isnan(v))
                         and float(v) in (0.0, 1.0))
        valid = np.array(valid)
        if valid.sum() < 20 or len(set(y[valid])) < 2:
            continue
        prompt = args.prompt_template.format(label=L) if args.prompt_template else L
        pv = enc.encode_text([prompt])[0]
        score = img_emb @ pv
        zs[L] = float(roc_auc_score(y[valid], score[valid]))
    res['zeroshot_label_auroc'] = zs
    res['zeroshot_label_auroc_macro'] = float(np.mean(list(zs.values()))) if zs else None

    # 4. retrieval: exact-text (current) vs chexpert-label relevance
    exact = retrieval_metrics(img_emb, txt_emb, texts, tuple(args.retrieval_ks))
    res['exact_text_retrieval'] = {k: exact[k] for k in exact
                                   if k.startswith(('R@', 'P@', 'mAP'))}
    S = img_emb @ txt_emb.T
    rel = (Y @ Y.T) > 0                                # share >=1 positive label
    res['label_retrieval_i2t'] = retrieval_with_relevance(S, rel, tuple(args.retrieval_ks))
    res['label_retrieval_t2i'] = retrieval_with_relevance(S.T, rel.T, tuple(args.retrieval_ks))

    _print_verdict(res)
    return res


def _print_verdict(res):
    print(f"[diag] repeatability cos = {res.get('img_embed_repeatability_cos'):.4f} "
          f"(≈1 = deterministic)")
    if res.get('logit_scale') is not None:
        print(f"[diag] logit_scale = {res['logit_scale']:.2f} "
              "(trained CLIP ~50-100; ~14/1 = maybe unloaded)")
    print(f"[diag] signal = {res['signal_mean']:.5f}  CI=[{res['signal_ci_low']:.5f}, "
          f"{res['signal_ci_high']:.5f}]  excl.0={res['signal_ci_excludes_zero']}  "
          f"p={res['signal_perm_pvalue']:.4f}  P(m>s)={res['frac_matched_gt_shuffled']:.3f}")
    zm = res.get('zeroshot_label_auroc_macro')
    print(f"[diag] zero-shot label AUROC macro = {zm if zm is None else round(zm, 4)}  "
          "(>>0.5 = encoder works -> report-text is the problem; ≈0.5 = encoder/preproc broken)")
    lr = res.get('label_retrieval_i2t', {})
    if lr:
        print(f"[diag] label-level retrieval i2t: R@10={lr.get('R@10'):.4f}  mAP={lr.get('mAP'):.4f}  "
              f"(vs exact R@10={res['exact_text_retrieval'].get('R@10_i2t'):.4f})")


def parse_args():
    from Eval_metric.text_utils import SECTION_MODES
    p = argparse.ArgumentParser(description='CLIP evaluator validity diagnostics (real images)')
    p.add_argument('--backend', default='medclip',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    p.add_argument('--clip_model', default='ViT-B-32')
    p.add_argument('--clip_pretrained', default='openai')
    p.add_argument('--batch_size', default=32, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--root_path', required=True)
    p.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    p.add_argument('--eval_split', default='test')
    p.add_argument('--chexpert_csv', default=None)
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--max_length', default=512, type=int)
    p.add_argument('--max_real', default=361, type=int)
    p.add_argument('--text_mode', default='FINDINGS/IMPRESSION', choices=SECTION_MODES)
    p.add_argument('--prompt_template', default='{label}',
                   help="zero-shot prompt template, e.g. 'findings consistent with {label}'")
    p.add_argument('--retrieval_ks', nargs='+', type=int, default=[1, 5, 10])
    p.add_argument('--bootstrap', default=2000, type=int)
    p.add_argument('--permutations', default=5000, type=int)
    p.add_argument('--output', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    res = run_diagnose(args)
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
