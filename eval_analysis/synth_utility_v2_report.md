# Synth-utility v2: pretrained backbone + full real pool (2026-09-07)

**Session goal:** macro AUROC ≥ 0.8; classifiers trained on synthetic images alone must
not degrade toward chance; mixing real+synthetic must beat real-only by ≥1% (relative
macro AUROC).

**Verdict: none of the three goals were met.** Real-data *scale* was the dominant lever
(0.646→0.763), but synthetic images never helped and sometimes hurt, regardless of
backbone, epoch count, or CLAHE preprocessing. Full detail below.

## Result table (macro AUROC, 12/14 reliable CheXpert labels, min_pos=10, same fixed
5,000-image p12 test set as v1)

| Arm | Backbone | n_train | Real source | Synth source | AUROC macro | vs. best real-only |
|---|---|---:|---|---|---:|---:|
| R (v1, prior session) | from-scratch | 1,100 | pool A | – | 0.6461 | −15.3% |
| R1100_pretrained | ImageNet | 1,100 | pool A | – | 0.6720 | −11.9% |
| **Rfull** | ImageNet | **49,907** | full train | – | **0.7631** | baseline (best) |
| Rfull_ep20 (20 epochs) | ImageNet | 49,907 | full train | – | 0.7524 | −1.4% (overfit) |
| RfullS_eps10 | ImageNet | 55,407 | full train | eps10, 5,500 | 0.7559 | −0.9% |
| RfullS_eps1 | ImageNet | 55,407 | full train | eps1, 5,500 | 0.7533 | −1.3% |
| S5500_eps1_pretrained | ImageNet | 5,500 | – (synth only) | eps1, 5,500 | 0.5010 | chance |
| S5500_eps10_pretrained | ImageNet | 5,500 | – (synth only) | eps10, 5,500 | 0.4781 | below chance |
| S5500_eps1_clahe_pretrained | ImageNet | 5,500 | – (synth only) | eps1 + CLAHE | 0.5184 | chance |

Full per-pathology breakdown: `synth_utility_v2_comparison.csv`. v1 (from-scratch,
matched-pool) results for reference: `synth_utility_comparison.csv` /
`synth_utility_report.md`.

## Goal-by-goal

**① Macro AUROC ≥ 0.8 — not met.** Best result is `Rfull` at 0.7631 (real-only, ImageNet
backbone, full 49,907-image real pool, 10 epochs). Gap to target: −0.037 absolute
(−4.8% relative). More training did not close the gap — `Rfull_ep20` (20 epochs)
scored *lower* (0.7524): train loss fell from 0.159→0.083, i.e. it overfit rather than
generalized further. 10 epochs on this real-pool size is already close to the point of
diminishing/negative returns for this setup; closing the remaining gap to 0.8 likely
needs either more real training data, stronger augmentation/regularization, or an
architecture/pretraining change (e.g. a CXR-specific pretrained backbone such as the
XRV weights `downstream_cls.py` already uses as a *frozen scorer*, rather than
ImageNet), not more epochs on the same pool.

**② No degradation on synthetic-only training — not met, and not close.** Every
synthetic-only arm scored at or below chance (0.478–0.518), regardless of:
- backbone (from-scratch [v1]: 0.501–0.521; ImageNet-pretrained: 0.478–0.501),
- epsilon (eps1 vs eps10 — eps1 is consistently a bit higher but both are ~chance),
- CLAHE contrast normalization on the synthetic images (0.518, barely different from
  the non-CLAHE 0.501).

This is the clearest finding of the round: the synthetic-only failure is **not** a
downstream-classifier-training artifact (data-starved from-scratch training, backbone
choice, or a fixable contrast/intensity mismatch) — every lever tried on the
classifier/preprocessing side failed to move the needle. That points at the LDM
checkpoints' own generative fidelity (label-conditional structure, not just overall
image statistics) as the real bottleneck, consistent with this repo's separate finding
that these same checkpoints' FID/CLIP-zeroshot generative-quality metrics do not predict
downstream-training utility.

**③ Real+synthetic mixing beats real-only by ≥1% — not met, for either epsilon.**
Adding the SAME already-generated 5,500-image synthetic pool on top of the full
49,907-image real pool made results slightly *worse*, not better: eps10 mixing −0.9%,
eps1 mixing −1.3% (relative to `Rfull` alone). Note this also breaks the repo's
previously-observed "eps1 > eps10" downstream-utility pattern (CLIP-zeroshot, RoentGen,
and the v1 synth-utility round all showed eps1 winning) — here eps10 mixing scored
*higher* than eps1 mixing, though both still lost to real-only. With synthetic images
already failing on their own (goal ②), this result is consistent, not surprising: at an
~11% blend ratio (5,500 synth / 49,907 real), synthetic images close to chance-quality
act as label noise diluting an otherwise-strong real signal.

## What this round ruled in / ruled out as the next lever

- **Ruled out** (tried, no effect): from-scratch → pretrained backbone (helps a little
  alone, 0.646→0.672, but is dominated by data scale); more training epochs on the
  full-real arm (hurts via overfitting); CLAHE contrast normalization on synthetic
  images (no meaningful change, 0.501→0.518).
- **Ruled in** (confirmed to matter): real training-pool size is the dominant lever for
  goal ① (0.672→0.763, +13.5 points, just from 1,100→49,907 real images at fixed
  backbone/epoch settings).
- **Still open**: whether the synthetic-only/mixing failure (goals ②③) is fixable at
  all with the *current* eps1/eps10 LABEL+IMPRESSION checkpoints, or whether it
  requires better checkpoints (e.g. more training epochs on the LDM side, or a
  different `ablation_blocks` setting per `experiment-ablation-blocks-sweep`) before
  downstream-classifier-side tuning can help further.

## Notes on method

- Same fixed 5,000-image p12 test set (`eval_subsample_test_n5000_seed42.csv.gz`) as
  v1, so all 9+8=17 arms across both rounds are directly comparable.
- `Rfull`/`RfullS_*` use `--real_manifest` (new in this round) to draw the full
  49,907-row LABEL+IMPRESSION train pool instead of v1's matched 1,100-image pool A;
  `RfullS_*`'s synthetic half is the SAME already-generated 5,500-image eps1/eps10 pool
  used in v1 (no new ~22h generation needed).
- ImageNet-pretrained runs use `torchvision.models.densenet121(weights=IMAGENET1K_V1)`
  with proper ImageNet mean/std normalization (`to_model_input()`); from-scratch runs
  (v1, and any left unlabeled `_pretrained` here) keep the original plain [-1,1]
  3-channel-repeat preprocessing for exact reproducibility of v1's numbers.
- CLAHE preprocessing via `Eval_metric/apply_clahe.py` (clip_limit=2.0, tile=8×8,
  OpenCV `createCLAHE`), applied once to the full eps1 5,500-image pool
  (`EVAL/gen_out/synth_utility/eps1_clahe/`), reused as-is for the CLAHE arm.
