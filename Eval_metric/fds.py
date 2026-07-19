"""
Eval_metric/fds.py  —  FDS: Feature Distribution Similarity via KL (Eval step 2)
================================================================================
Concept
-------
Like FID, we model the real and generated feature sets each as a multivariate
Gaussian N(mu, Sigma) in a pretrained network's feature space.  FID reports the
Frechet (2-Wasserstein) DISTANCE between the two Gaussians.  FDS instead reports
the KL DIVERGENCE between them (lower = more similar, hence FDS(FeatureKL)(down)).

Closed-form KL between two k-dim Gaussians:
    D_KL(N0 || N1) = 0.5 * [ tr(S1^-1 S0)
                             + (mu1-mu0)^T S1^-1 (mu1-mu0)
                             - k + ln(det S1 / det S0) ]

Because KL is ASYMMETRIC, the direction carries meaning:
    gen||real : mass the generator puts OUTSIDE the real support
                -> sensitive to hallucinated / invented modes.
    real||gen : real modes the generator FAILS to cover
                -> sensitive to mode collapse (a real DP failure mode).
We report both directions and their symmetric average.

Covariances are shrinkage-regularised (Sigma + reg*I) and all determinants /
solves go through Cholesky for numerical stability in high dimensions.

Usage
-----
    python Eval_metric/fds.py \\
        --real_dir  ./real_test_pngs \\
        --gen_dir   ./generated_eps5/samples \\
        --eval_model xrv \\
        --image_size 256 \\
        --output    ./eval/eps5/fds.json
"""

import os
import json
import argparse

import numpy as np


def _gaussian_stats(feats, reg):
    mu = feats.mean(0)
    sigma = np.cov(feats, rowvar=False)
    sigma = sigma + reg * np.eye(sigma.shape[0])
    return mu, sigma


def _kl_gaussian(mu0, s0, mu1, s1):
    """D_KL( N(mu0,s0) || N(mu1,s1) ) via Cholesky (stable log-det & solve)."""
    k = mu0.shape[0]
    L1 = np.linalg.cholesky(s1)                        # s1 = L1 L1^T
    # tr(s1^-1 s0): solve L1 Y = s0 twice -> s1^-1 s0
    sol = np.linalg.solve(L1, s0)
    sol = np.linalg.solve(L1.T, sol)
    tr_term = np.trace(sol)
    # (mu1-mu0)^T s1^-1 (mu1-mu0)
    d = (mu1 - mu0)
    y = np.linalg.solve(L1, d)
    maha = float(y @ y)
    # ln(det s1 / det s0) = 2*(sum log diag L1 - sum log diag L0)
    L0 = np.linalg.cholesky(s0)
    logdet_ratio = 2.0 * (np.sum(np.log(np.diag(L1))) - np.sum(np.log(np.diag(L0))))
    return 0.5 * (tr_term + maha - k + logdet_ratio)


def compute_fds(feats_real, feats_gen, reg=1e-6):
    """
    Returns a dict with both KL directions and their symmetric average.
        fds_gen_given_real : D_KL(gen || real)  (hallucination-sensitive)
        fds_real_given_gen : D_KL(real || gen)  (mode-collapse-sensitive)
        fds_symmetric      : mean of the two
    """
    feats_real = np.asarray(feats_real, dtype=np.float64)
    feats_gen = np.asarray(feats_gen, dtype=np.float64)
    mu_r, s_r = _gaussian_stats(feats_real, reg)
    mu_g, s_g = _gaussian_stats(feats_gen, reg)
    kl_g_r = _kl_gaussian(mu_g, s_g, mu_r, s_r)         # gen || real
    kl_r_g = _kl_gaussian(mu_r, s_r, mu_g, s_g)         # real || gen
    return {
        'fds_gen_given_real': float(kl_g_r),
        'fds_real_given_gen': float(kl_r_g),
        'fds_symmetric': float(0.5 * (kl_g_r + kl_r_g)),
    }


def compute_fds_from_dirs(real_dir, gen_dir, eval_model='xrv', image_size=256,
                          device='cpu', batch_size=16, reg=1e-6,
                          real_cache=None, gen_cache=None):
    from Eval_metric.features import extract_features
    fr, _ = extract_features(real_dir, eval_model, device, image_size,
                             batch_size, cache_path=real_cache)
    fg, _ = extract_features(gen_dir, eval_model, device, image_size,
                             batch_size, cache_path=gen_cache)
    out = compute_fds(fr, fg, reg=reg)
    out.update({'eval_model': eval_model, 'n_real': int(len(fr)),
                'n_gen': int(len(fg)), 'feature_dim': int(fr.shape[1])})
    return out


def parse_args():
    p = argparse.ArgumentParser(description='FDS (Feature KL divergence)')
    p.add_argument('--real_dir', required=True)
    p.add_argument('--gen_dir', required=True)
    p.add_argument('--eval_model', default='xrv', choices=['inception', 'xrv'])
    p.add_argument('--image_size', default=256, type=int)
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch_size', default=16, type=int)
    p.add_argument('--reg', default=1e-6, type=float,
                   help='covariance shrinkage Sigma + reg*I')
    p.add_argument('--output', default=None, help='write result JSON here')
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = os.path.join(os.path.dirname(args.output or '.'), '_features')
    res = compute_fds_from_dirs(
        args.real_dir, args.gen_dir, args.eval_model, args.image_size,
        args.device, args.batch_size, args.reg,
        real_cache=os.path.join(cache_dir, f'real_{args.eval_model}.npz'),
        gen_cache=os.path.join(cache_dir, f'gen_{args.eval_model}.npz'))
    print(json.dumps(res, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'[save] -> {args.output}')


if __name__ == '__main__':
    main()
