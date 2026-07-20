"""
Eval_metric/generate_and_eval.py  —  Generate images, then evaluate (one shot)
==============================================================================
Runs DP-LDM inference to produce images from text descriptions, saves them in
the standard layout (<gen_output_dir>/samples/*.png + descriptions.csv +
grid_all.png), then runs the feature-based eval (FDS / T-SNE / FID) on those
freshly generated images — all in a single command.

This is the "one shot" counterpart to run_feature_eval.py, which evaluates
images ALREADY on disk (gen_dir mode).  Both share the SAME eval implementation
(run_feature_eval.run) and the SAME generation code (LDM_dp_inference), so the
two paths never diverge.

Usage
-----
    python Eval_metric/generate_and_eval.py \\
        --dp_ckpt    ./checkpoints/ldm/ldm_epoch0100.pt \\
        --lora_ckpt  ./finetune_dp/ldm_lora_eps5.pt \\
        --vae_ckpt   ./checkpoints/vae/vae_ep0070.pt \\
        --descriptions ./prompts_example.txt --n_samples 8 \\
        --device_id  0 \\
        --gen_output_dir ./generated_eps5 \\
        --out_dir    ./eval/eps5 \\
        --root_path  /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \\
        --eval_split test --max_real 500 \\
        --eval_model xrv --metrics fds tsne fid

The real reference works exactly as in run_feature_eval.py: dataset mode
(--root_path/--eval_split) or folder mode (--real_dir).
"""

import os
import sys
import argparse

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def parse_args():
    from LDM_dp_inference import add_model_args
    p = argparse.ArgumentParser(
        description='Generate DP-LDM images and evaluate them in one shot',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        # add_model_args already defines some names shared with the eval group
        # (e.g. --max_length used for report tokenisation); 'resolve' lets the
        # later (eval) definition win instead of raising a conflict error.
        conflict_handler='resolve')

    # ── model / checkpoint args (shared with inference: adds --device_id,
    #    --vae_img_size, vae_*/unet_*/diffusion args, --dp_ckpt/--lora_ckpt/…) ──
    add_model_args(p)

    # ── generation args (mirror LDM_dp_inference.parse_args) ──────────────────
    g = p.add_argument_group('generation')
    g.add_argument('--prompt_source', default='file', choices=['file', 'split'],
                   help="'file': prompts from --descriptions. 'split': prompts "
                        'are real reports pulled from the MIMIC --eval_split, so '
                        'the generated distribution matches the real reference.')
    g.add_argument('--descriptions', default=None,
                   help='.txt (one per line) OR inline string. Required when '
                        '--prompt_source file.')
    g.add_argument('--n_prompts', default=None, type=int,
                   help='prompt_source=split: number of report prompts to use '
                        '(default: same as --max_real, else all in the split).')
    g.add_argument('--n_samples', default=1, type=int,
                   help='independent samples per description')
    g.add_argument('--sample_timesteps', default=None, type=int)
    g.add_argument('--seed', default=0, type=int,
                   help='generation seed AND t-SNE seed')
    g.add_argument('--gen_output_dir', default='./generated',
                   help='where generated images are saved '
                        '(<dir>/samples/*.png). Eval reads <dir>/samples.')

    # ── eval args (mirror run_feature_eval; gen_dir is derived automatically) ──
    e = p.add_argument_group('evaluation')
    e.add_argument('--out_dir', required=True, help='e.g. ./eval/eps5')
    e.add_argument('--eval_model', default='xrv', choices=['inception', 'xrv'])
    e.add_argument('--eval_batch_size', default=16, type=int,
                   help='batch size for feature extraction')
    e.add_argument('--metrics', nargs='+', default=['fds', 'tsne'],
                   choices=['fds', 'tsne', 'fid', 'clip'])
    e.add_argument('--clip_backend', default='medclip',
                   choices=['biovil-t', 'medclip', 'cxr-clip', 'openclip'])
    e.add_argument('--clip_model', default='ViT-B-32')
    e.add_argument('--clip_pretrained', default='openai')
    e.add_argument('--clip_w', default=2.5, type=float)
    e.add_argument('--reg', default=1e-6, type=float)
    e.add_argument('--fds_cov', default='lw', choices=['lw', 'empirical'],
                   help="FDS covariance estimator (lw=Ledoit-Wolf, robust when n<dim)")
    e.add_argument('--pca_dim', default=None, type=int,
                   help='FDS: reduce features to this dim (PCA on real) first, e.g. 64')
    e.add_argument('--perplexity', default=30.0, type=float)
    # real reference source (either --real_dir OR dataset mode)
    e.add_argument('--real_dir', default=None)
    e.add_argument('--root_path', default=None)
    e.add_argument('--split_csv', default='mimic-cxr-2.0.0-split.csv')
    e.add_argument('--eval_split', default='test')
    e.add_argument('--max_length', default=512, type=int)
    e.add_argument('--max_real', default=None, type=int)
    e.add_argument('--dp_split_json', default=None)
    e.add_argument('--dp_split_group', default='train')
    e.add_argument('--eval_ckpt', default=None,
                   help='ckpt to record args from (default: lora_ckpt or dp_ckpt)')
    return p.parse_args()


def resolve_descriptions(args):
    """Prompts either from --descriptions (file) or from the eval split reports."""
    if args.prompt_source == 'split':
        # (A) pull real reports from the dataset so generated prompts — and thus
        # the generated distribution — match the real reference split.
        if not args.root_path:
            raise SystemExit('--prompt_source split needs --root_path (MIMIC root)')
        from Data.mimic_cxr import MIMICCXRDataset
        ds = MIMICCXRDataset(argparse.Namespace(
            root_path=args.root_path, split_csv=args.split_csv, split=args.eval_split,
            image_size=args.vae_img_size, max_length=args.max_length,
            patient_whitelist=None))
        cap = args.n_prompts if args.n_prompts not in (None, 0) else \
            (args.max_real if args.max_real not in (None, 0) else len(ds))
        n = min(cap, len(ds))
        prompts = [ds[i][1] for i in range(n)]        # (image, report) -> report
        print(f'[gen] prompts from split={args.eval_split}: {n} report(s)')
        return prompts
    from LDM_dp_inference import load_descriptions
    if not args.descriptions:
        raise SystemExit('--prompt_source file needs --descriptions')
    return load_descriptions(args.descriptions)


def generate(args, device):
    """Run inference; save to <gen_output_dir>/samples. Returns that samples dir."""
    import torch
    from LDM_dp_inference import build_and_load_ldm, save_outputs

    torch.manual_seed(args.seed)
    descriptions = resolve_descriptions(args)
    print(f'[gen] {len(descriptions)} description(s) x {args.n_samples} sample(s)')

    ldm, embedder = build_and_load_ldm(args, device)
    all_images, all_caps, all_idx = [], [], []
    for d_idx, desc in enumerate(descriptions):
        with torch.no_grad():
            c1 = embedder([desc]).to(device)
            c = c1.repeat(args.n_samples, 1, 1)
            imgs = ldm.sample(c, batch_size=args.n_samples,
                              verbose=False, timesteps=args.sample_timesteps)
        all_images.append(imgs.detach().cpu())
        all_caps.extend([desc] * args.n_samples)
        all_idx.extend([d_idx] * args.n_samples)
        print(f'  [{d_idx + 1}/{len(descriptions)}] "{desc[:50]}"')

    images = torch.cat(all_images, dim=0)
    save_outputs(images, all_caps, all_idx, args.gen_output_dir)
    return os.path.join(args.gen_output_dir, 'samples')


def build_eval_cfg(args, gen_dir):
    """Map the combined args to the namespace run_feature_eval.run expects."""
    import torch
    device = f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu'
    ckpt = args.eval_ckpt or args.lora_ckpt or args.dp_ckpt
    return argparse.Namespace(
        real_dir=args.real_dir, root_path=args.root_path, split_csv=args.split_csv,
        eval_split=args.eval_split, max_length=args.max_length, max_real=args.max_real,
        dp_split_json=args.dp_split_json, dp_split_group=args.dp_split_group,
        gen_dir=gen_dir, out_dir=args.out_dir, eval_model=args.eval_model,
        image_size=args.vae_img_size, device=device, batch_size=args.eval_batch_size,
        metrics=args.metrics, reg=args.reg, fds_cov=args.fds_cov,
        pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed, ckpt=ckpt,
        clip_backend=args.clip_backend, clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained, clip_w=args.clip_w)


def main():
    args = parse_args()
    if not args.real_dir and not args.root_path:
        raise SystemExit('provide a real reference: --real_dir <folder> or '
                         '--root_path <MIMIC root> (dataset mode)')
    import torch
    device = torch.device(
        f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 1) generate  ->  <gen_output_dir>/samples
    gen_dir = generate(args, device)
    print(f'[gen] images saved -> {gen_dir}')

    # 2) evaluate the freshly generated images (same code path as gen_dir mode)
    from Eval_metric.run_feature_eval import run
    run(build_eval_cfg(args, gen_dir))
    print('\nDone (generate + eval).')


if __name__ == '__main__':
    main()
