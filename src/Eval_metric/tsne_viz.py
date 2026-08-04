"""
Eval_metric/tsne_viz.py  —  T-SNE feature-space visualization (Eval step 3)
===========================================================================
Concept
-------
T-SNE embeds high-dim features (2048-d Inception / 1024-d XRV DenseNet) into 2-D
by preserving LOCAL neighbourhoods: neighbour relations are modelled as a
Gaussian distribution in the high-dim space and a Student-t distribution in 2-D,
and the two are matched by MINIMISING their KL divergence.  It is a QUALITATIVE
diagnostic — axes are meaningless, global distances/densities are distorted, and
results depend on `perplexity` and the random seed (both fixed here).

What it shows for this project
------------------------------
1. real vs generated colouring  -> mode coverage vs mode collapse: does the
   generated cloud overlap the real cloud, or sit in its own corner?
2. finding-label colouring (optional --label_csv) -> do generated images land in
   the correct per-class real cluster, i.e. is the text condition reflected in
   feature space?

Usage
-----
    python Eval_metric/tsne_viz.py \\
        --real_dir ./real_test_pngs \\
        --gen_dir  ./generated_eps5/samples \\
        --eval_model xrv --image_size 256 \\
        --perplexity 30 --seed 0 \\
        --output ./eval/eps5/tsne.png
"""

import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE


def run_tsne(feats, perplexity=30.0, seed=0):
    n = feats.shape[0]
    perp = float(min(perplexity, max(5.0, (n - 1) / 3.0)))   # keep perplexity < n
    tsne = TSNE(n_components=2, perplexity=perp, init='pca',
                learning_rate='auto', random_state=seed)
    return tsne.fit_transform(feats), perp


def plot_source(emb, n_real, out_png, title):
    plt.figure(figsize=(7, 6))
    plt.scatter(emb[:n_real, 0], emb[:n_real, 1], s=10, alpha=0.6,
                label=f'real (n={n_real})', c='#2563eb')
    plt.scatter(emb[n_real:, 0], emb[n_real:, 1], s=10, alpha=0.6,
                label=f'generated (n={len(emb) - n_real})', c='#dc2626')
    plt.legend(loc='best'); plt.title(title)
    plt.xlabel('t-SNE 1'); plt.ylabel('t-SNE 2'); plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()


def visualize_from_dirs(real_dir, gen_dir, eval_model='xrv', image_size=256,
                        device='cpu', batch_size=16, perplexity=30.0, seed=0,
                        output='tsne.png', real_cache=None, gen_cache=None):
    from Eval_metric.features import extract_features
    fr, _ = extract_features(real_dir, eval_model, device, image_size,
                             batch_size, cache_path=real_cache)
    fg, _ = extract_features(gen_dir, eval_model, device, image_size,
                             batch_size, cache_path=gen_cache)
    feats = np.concatenate([fr, fg], 0).astype(np.float64)
    emb, perp = run_tsne(feats, perplexity, seed)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    title = f't-SNE ({eval_model}, perplexity={perp:.0f})  real vs generated'
    plot_source(emb, len(fr), output, title)
    # persist the 2-D embedding for re-plotting / label overlays later
    np.savez(os.path.splitext(output)[0] + '_embedding.npz',
             emb=emb, n_real=len(fr), eval_model=eval_model, perplexity=perp)
    print(f'[save] -> {output}')
    return {'eval_model': eval_model, 'n_real': int(len(fr)),
            'n_gen': int(len(fg)), 'perplexity': float(perp), 'output': output}


def parse_args():
    p = argparse.ArgumentParser(description='T-SNE feature-space visualization')
    p.add_argument('--real_dir', required=True)
    p.add_argument('--gen_dir', required=True)
    p.add_argument('--eval_model', default='xrv', choices=['inception', 'xrv'])
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--perplexity', default=30.0, type=float)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument('--output', default='./tsne.png')
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.join(os.path.dirname(args.output), '_features')
    visualize_from_dirs(
        args.real_dir, args.gen_dir, args.eval_model, args.image_size,
        args.device, args.batch_size, args.perplexity, args.seed, args.output,
        real_cache=os.path.join(cache_dir, f'real_{args.eval_model}.npz'),
        gen_cache=os.path.join(cache_dir, f'gen_{args.eval_model}.npz'))


if __name__ == '__main__':
    main()
