#!/usr/bin/env python
"""Aggregate per-epsilon evaluation results into a cross-model comparison.

This is a *self-contained analysis driver*: it does NOT run any model. It reads
the JSON produced by the evaluation scripts (`eval_summary.json`, `fds.json`,
`clipscore.json`, `ckpt_info.json`) under one result root and turns them into

    (1) a per-epsilon comparison table (console + CSV),
    (2) a privacy-utility tradeoff report (monotonicity check per EVALUATION_GUIDE §6.2),
    (3) an FDS direction diagnosis (mode-collapse vs hallucination) per model,
    (4) a fair-comparison sanity check (§6.4: same real set / backbone / pca_dim),
    (5) optional metric-vs-epsilon plots (matplotlib, skipped if unavailable).

Directory layout it expects (see EVALUATION_GUIDE.md §3):

    <eval_root>/
        eps1/   eval_summary.json  fds.json  clipscore.json  ckpt_info.json
        eps3/   ...
        eps5/   ...
        eps10/  ...

Each `eps<N>` folder is one model (one privacy budget). The folder name gives a
fallback epsilon; `ckpt_info.json.epsilon_spent` (the *actually spent* budget) is
preferred when present.

Usage
-----
    python Eval_metric/analyze_eval.py --eval_root ./eval
    python Eval_metric/analyze_eval.py --dirs ./eval/eps1 ./eval/eps10
    python Eval_metric/analyze_eval.py --eval_root ./eval --out_dir ./eval/analysis --no_plot

Dependencies: standard library only for the core report. matplotlib is imported
lazily and only for the optional plots; if it is missing, plotting is skipped
with a warning and every other output is still produced.
"""

import argparse
import csv
import glob
import json
import os
import re


# --------------------------------------------------------------------------- #
# Metric registry: (key, human label, direction, group)
#   direction : 'up'   -> higher is better (↑)
#               'down' -> lower  is better (↓)
#               None   -> descriptive / not a quality score
#   The `trend_wrt_eps` for a quality metric is derived from `direction`:
#   weaker privacy (higher ε) should improve utility, so a 'down' (lower-better)
#   metric is expected to DECREASE with ε, and an 'up' metric to INCREASE.
#   See EVALUATION_GUIDE.md §6.2.
# --------------------------------------------------------------------------- #
METRICS = [
    # key                        label                 dir     group
    ('fid',                      'FID',                'down', 'distribution'),
    ('fds_symmetric',            'FDS(sym)',           'down', 'distribution'),
    ('fds_gen_given_real',       'FDS gen||real',      'down', 'distribution'),
    ('fds_real_given_gen',       'FDS real||gen',      'down', 'distribution'),
    ('lpips_mean',               'LPIPS(mean)',        'down', 'perceptual'),
    ('lpips_best',               'LPIPS(best)',        'down', 'perceptual'),
    ('ssim_mean',                'SSIM(mean)',         'up',   'pixel'),
    ('ssim_best',                'SSIM(best)',         'up',   'pixel'),
    ('psnr_mean',                'PSNR(mean)',         'up',   'pixel'),
    ('psnr_best',                'PSNR(best)',         'up',   'pixel'),
    ('clip_gap',                 'CLIP gap',           'down', 'alignment'),
    ('clip_gen_clipscore_mean',  'CLIPScore(gen)',     'up',   'alignment'),
    ('clip_gen_cos_mean',        'CLIP cos(gen)',      'up',   'alignment'),
]

# Reliability order (EVALUATION_GUIDE §6.3): distribution/perceptual/alignment
# are trusted over the pixel proxies (SSIM/PSNR). Headline metrics reported first.
HEADLINE = ['fid', 'fds_symmetric', 'lpips_mean', 'clip_gap']

# Fields that MUST match across models for a fair comparison (§6.4).
FAIRNESS_KEYS = ['eval_model', 'fid_backbone', 'n_real', 'n_gen',
                 'pca_dim', 'feature_dim', 'eval_split', 'tsne_perplexity',
                 'clip_backend']


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _load_json(path):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f'[warn] failed to read {path}: {e}')
        return {}


def _parse_eps_from_name(name):
    """Extract the epsilon hint from a folder name like 'eps10' / 'eps_3' / 'eps0.5'."""
    m = re.search(r'eps[_-]?([0-9]+(?:\.[0-9]+)?)', name, re.IGNORECASE)
    return float(m.group(1)) if m else None


def load_run(run_dir):
    """Merge every JSON in one result folder into a single flat record.

    Precedence: eval_summary.json is the merged main file; fds.json / clipscore.json
    fill any key it is missing (e.g. feature_dim, pca_dim live in fds.json). Paired
    metrics (ssim/psnr/lpips) written by LDM_dp_eval.py land in eval_summary too.
    ckpt_info.json contributes epsilon_spent / lora_rank / lora_alpha.
    """
    rec = {}
    # fds/clip first (lower precedence), then summary overwrites shared keys.
    for fname in ('fds.json', 'clipscore.json', 'eval_summary.json'):
        rec.update(_load_json(os.path.join(run_dir, fname)))

    ckpt = _load_json(os.path.join(run_dir, 'ckpt_info.json'))
    for k in ('epsilon_spent', 'lora_rank', 'lora_alpha', 'epoch',
              'global_step', 'ckpt_path'):
        if k in ckpt:
            rec[k] = ckpt[k]

    name = os.path.basename(os.path.normpath(run_dir))
    rec['_dir'] = run_dir
    rec['_name'] = name

    # Epsilon: prefer the spent budget from the checkpoint, fall back to folder name.
    eps = rec.get('epsilon_spent')
    if eps is None:
        eps = _parse_eps_from_name(name)
    rec['_epsilon'] = float(eps) if eps is not None else None
    return rec


def discover_runs(eval_root):
    """Return eps* sub-directories that contain at least one result JSON."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(eval_root, '*'))):
        if not os.path.isdir(d):
            continue
        if any(os.path.isfile(os.path.join(d, f))
               for f in ('eval_summary.json', 'fds.json', 'clipscore.json')):
            dirs.append(d)
    return dirs


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def _fmt(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.4f}'
    return str(v)


def kendall_sign(xs, ys):
    """Sign of the monotonic association between xs and ys (Kendall-tau sign).

    Returns +1 (ys tends to rise with xs), -1 (falls), 0 (flat/mixed/insufficient).
    Pure-stdlib, robust to few points; ties are ignored.
    """
    concordant = discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            if xs[i] == xs[j] or ys[i] == ys[j]:
                continue
            same = (xs[i] < xs[j]) == (ys[i] < ys[j])
            concordant += same
            discordant += not same
        # /loop guard: nothing
    if concordant == discordant:
        return 0
    return 1 if concordant > discordant else -1


def diagnose_fds(rec):
    """mode-collapse vs hallucination from the asymmetric KL (§6.5)."""
    g_r = rec.get('fds_gen_given_real')
    r_g = rec.get('fds_real_given_gen')
    if g_r is None or r_g is None or g_r <= 0 or r_g <= 0:
        return '—'
    ratio = r_g / g_r
    if ratio >= 1.3:
        return f'mode-collapse (real||gen {ratio:.2f}× gen||real)'
    if ratio <= 1 / 1.3:
        return f'hallucination (gen||real {1/ratio:.2f}× real||gen)'
    return f'balanced (ratio {ratio:.2f})'


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def build_table(records, keys):
    """Return (header, rows) of strings for the metrics present in `records`."""
    present = [k for (k, *_ ) in keys
               if any(r.get(k) is not None for r in records)]
    label = {k: lbl for (k, lbl, *_ ) in keys}
    header = ['epsilon'] + [label[k] for k in present]
    rows = []
    for r in records:
        eps = r.get('_epsilon')
        eps_s = f'{eps:g}' if eps is not None else r['_name']
        rows.append([eps_s] + [_fmt(r.get(k)) for k in present])
    return header, rows, present


def render_ascii_table(header, rows):
    cols = list(zip(header, *rows)) if rows else [(h,) for h in header]
    widths = [max(len(str(c)) for c in col) for col in cols]
    line = lambda cells: ' | '.join(str(c).rjust(w) for c, w in zip(cells, widths))
    sep = '-+-'.join('-' * w for w in widths)
    out = [line(header), sep]
    out += [line(row) for row in rows]
    return '\n'.join(out)


def fairness_report(records):
    """Warn about columns that differ across models (unfair comparison)."""
    lines = []
    for k in FAIRNESS_KEYS:
        vals = {r.get(k) for r in records if r.get(k) is not None}
        if len(vals) > 1:
            detail = ', '.join(f'{r["_name"]}={_fmt(r.get(k))}' for r in records)
            lines.append(f'  ✗ `{k}` differs across models: {detail}')
    if not lines:
        return '  ✓ real set / backbone / pca_dim / split consistent across models.'
    return 'Fair-comparison check (§6.4):\n' + '\n'.join(lines)


def tradeoff_report(records, present):
    """Monotonicity of each metric vs epsilon against the expected direction (§6.2)."""
    direction = {k: d for (k, _lbl, d, _g) in METRICS}
    eps_vals = [(r['_epsilon'], r) for r in records if r['_epsilon'] is not None]
    if len(eps_vals) < 2:
        return ('Tradeoff check: need ≥2 models with a known epsilon to assess '
                'monotonicity — skipped.')
    lines = ['Privacy-utility tradeoff (expected as ε↓: quality worsens; §6.2):']
    for k in present:
        d = direction.get(k)
        if d is None:
            continue
        pts = [(e, r.get(k)) for (e, r) in eps_vals if r.get(k) is not None]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        obs = kendall_sign(list(xs), list(ys))          # trend of metric vs ε
        expect = 1 if d == 'up' else -1                 # up-metric rises with ε
        label = dict((kk, lbl) for (kk, lbl, *_ ) in METRICS)[k]
        if obs == 0:
            verdict = '~ flat/mixed'
        elif obs == expect:
            verdict = '✓ as expected'
        else:
            verdict = '✗ UNEXPECTED (framework may be misbehaving or noise-limited)'
        arrow = {1: 'rises with ε', -1: 'falls with ε', 0: 'flat'}[obs]
        lines.append(f'  {label:<16} {arrow:<16} {verdict}')
    return '\n'.join(lines)


def diagnosis_report(records):
    lines = ['FDS direction diagnosis (mode-collapse vs hallucination; §6.5):']
    for r in records:
        eps = r.get('_epsilon')
        tag = f'ε={eps:g}' if eps is not None else r['_name']
        lines.append(f'  {tag:<10} {diagnose_fds(r)}')
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Plotting (optional)
# --------------------------------------------------------------------------- #
def make_plots(records, present, out_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f'[plot] skipped (matplotlib unavailable: {e})')
        return None

    label = {k: lbl for (k, lbl, *_ ) in METRICS}
    plot_keys = [k for k in HEADLINE + ['ssim_mean', 'psnr_mean',
                                        'clip_gen_clipscore_mean'] if k in present]
    pts = [(r['_epsilon'], r) for r in records if r['_epsilon'] is not None]
    pts.sort(key=lambda t: t[0])
    if len(pts) < 2 or not plot_keys:
        print('[plot] skipped (need ≥2 epsilons and ≥1 plottable metric)')
        return None

    n = len(plot_keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.4 * rows),
                             squeeze=False)
    for idx, k in enumerate(plot_keys):
        ax = axes[idx // cols][idx % cols]
        xy = [(e, r.get(k)) for (e, r) in pts if r.get(k) is not None]
        if len(xy) >= 1:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, 'o-', color='#2563eb')
        ax.set_title(label.get(k, k))
        ax.set_xlabel('epsilon (ε)')
        ax.grid(True, alpha=0.3)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis('off')
    fig.suptitle('Privacy–Utility tradeoff (metric vs ε)')
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_png = os.path.join(out_dir, 'tradeoff.png')
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f'[plot] -> {out_png}')
    return out_png


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def write_csv(records, present, out_dir):
    path = os.path.join(out_dir, 'comparison.csv')
    extra = ['epsilon_spent', 'lora_rank', 'lora_alpha', 'n_real', 'n_gen',
             'eval_model', 'fid_backbone', 'pca_dim', 'eval_split']
    cols = ['name', 'epsilon'] + present + [c for c in extra if c not in present]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            row = [r['_name'], r.get('_epsilon')]
            row += [r.get(k) for k in present]
            row += [r.get(c) for c in extra if c not in present]
            w.writerow(row)
    print(f'[csv]  -> {path}')
    return path


def write_report(text, out_dir):
    path = os.path.join(out_dir, 'analysis_report.md')
    with open(path, 'w') as f:
        f.write(text)
    print(f'[report] -> {path}')
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description='Aggregate per-epsilon eval JSON into a comparison report.')
    p.add_argument('--eval_root', default='./eval',
                   help='root holding eps*/ result folders (default ./eval)')
    p.add_argument('--dirs', nargs='*', default=None,
                   help='explicit result folders (overrides --eval_root discovery)')
    p.add_argument('--out_dir', default=None,
                   help='where to write comparison.csv / report / plot '
                        '(default <eval_root>/analysis)')
    p.add_argument('--no_plot', action='store_true', help='skip matplotlib plots')
    args = p.parse_args()

    run_dirs = args.dirs if args.dirs else discover_runs(args.eval_root)
    if not run_dirs:
        raise SystemExit(f'No result folders with eval JSON found under '
                         f'{args.dirs or args.eval_root!r}.')

    records = [load_run(d) for d in run_dirs]
    # Sort by epsilon ascending (strongest privacy first); unknown eps go last.
    records.sort(key=lambda r: (r['_epsilon'] is None, r['_epsilon'] or 0.0))

    out_dir = args.out_dir or os.path.join(args.eval_root, 'analysis')
    os.makedirs(out_dir, exist_ok=True)

    header, rows, present = build_table(records, METRICS)
    table = render_ascii_table(header, rows)
    fairness = fairness_report(records)
    tradeoff = tradeoff_report(records, present)
    diagnosis = diagnosis_report(records)

    # LoRA / budget provenance line per model.
    prov = ['Model provenance (from ckpt_info.json):']
    for r in records:
        eps = r.get('_epsilon')
        tag = f'ε={eps:g}' if eps is not None else r['_name']
        prov.append(f'  {tag:<10} spent={_fmt(r.get("epsilon_spent"))} '
                    f'lora_rank={_fmt(r.get("lora_rank"))} '
                    f'lora_alpha={_fmt(r.get("lora_alpha"))} '
                    f'dir={r["_name"]}')
    provenance = '\n'.join(prov)

    report = (
        '# Per-epsilon evaluation comparison\n\n'
        f'Result folders: {", ".join(r["_name"] for r in records)}\n\n'
        '## Comparison table\n\n```\n' + table + '\n```\n\n'
        '> Direction: FID / FDS / LPIPS / CLIP gap are ↓ (lower better); '
        'SSIM / PSNR / CLIPScore are ↑ (higher better).\n'
        '> Reliability order (§6.3): FID, FDS, LPIPS, CLIP gap > SSIM/PSNR '
        '(pixel proxies; generation ≠ reconstruction).\n\n'
        '## Model provenance\n\n```\n' + provenance + '\n```\n\n'
        '## Privacy–utility tradeoff\n\n```\n' + tradeoff + '\n```\n\n'
        '## FDS direction diagnosis\n\n```\n' + diagnosis + '\n```\n\n'
        '## Fair-comparison check\n\n```\n' + fairness + '\n```\n'
    )

    # Console output
    print('\n' + '=' * 72)
    print('PER-EPSILON EVALUATION COMPARISON')
    print('=' * 72)
    print(table)
    print('\n' + provenance)
    print('\n' + tradeoff)
    print('\n' + diagnosis)
    print('\n' + fairness)
    print('=' * 72 + '\n')

    write_csv(records, present, out_dir)
    write_report(report, out_dir)
    if not args.no_plot:
        make_plots(records, present, out_dir)


if __name__ == '__main__':
    main()
