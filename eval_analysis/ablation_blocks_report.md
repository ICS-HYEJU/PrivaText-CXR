# ablation_blocks sweep — how many attention blocks does DP-LoRA utility need?

All three models share the exact `EVAL/metric/LABEL_IMPRESSION/eps10/ckpt_info.json`
config (`text_mode=LABEL+IMPRESSION`, `target_epsilon=10`, `seed=42`, `lr=0.002`,
`lora_rank=4`, `lora_alpha=4`, `epochs=10`) and land at **identical** `epsilon_spent`
(8.5913, matching to 4 decimals) — DP-SGD's noise multiplier is calibrated from
`target_epsilon`/`epochs`/`batch`, not from how many parameters are trainable, so this
sweep isolates a single variable cleanly: **how many of the UNet's 16 SpatialTransformer
blocks get a LoRA adapter.**

| model | blocks trained | epsilon_spent |
|---|---|---|
| `LI_eps10` (anchor, `ablation_blocks=-1`) | 16/16 (all) | 8.5913 |
| `ablation8` (`ablation_blocks=8`) | 9/16 (last 9) | 8.5913 |
| `ablation16` (`ablation_blocks=16`) | 1/16 (last 1 only) | 8.5913 |

## Core metrics

| metric | dir | 16 blocks | 9 blocks | 1 block |
|---|---|---|---|---|
| FID | ↓ | **2.271** | 2.452 | 2.914 |
| FDS (sym) | ↓ | **27.203** | 29.391 | 33.921 |
| FDS gen‖real (hallucination) | ↓ | **21.816** | 23.087 | 27.855 |
| FDS real‖gen (mode collapse) | ↓ | **32.589** | 35.695 | 39.987 |
| LPIPS | ↓ | 0.5269 | 0.5255 | **0.5233** |
| SSIM | ↑ | 0.2108 | 0.2125 | **0.2144** |
| Label AUROC ratio (all, 7-path) | ↑ | **0.7146** | 0.7055 | 0.6796 |
| Label AUROC ratio (nih, 6-path) | ↑ | 0.7893 | **0.8014** | 0.7710 |
| CLIP zero-shot label AUROC ratio | ↑ | 0.6051 | 0.6110 | **0.6701** |
| CLIP zero-shot label AUROC gap | ↓ | 0.3328 | 0.3278 | **0.2780** |

## Two opposite trends — reported honestly, not forced into one winner

**Distribution metrics (FID, all 3 FDS variants) monotonically favor MORE trainable
blocks.** More LoRA-adaptable capacity lets the model match the real feature
distribution more closely in aggregate — expected, and the cleanest, most reliable
signal in this sweep per this repo's own metric-reliability convention.

**CLIP zero-shot label AUROC reverses this — it monotonically favors FEWER trainable
blocks**, and by a large margin (ratio 0.605→0.611→**0.670** going from 16→9→1 block).
Fewer trainable blocks under DP-SGD means less DP-noised parameter surface touching the
network's fine-grained content-generation machinery; the working hypothesis is that this
acts as an implicit regularizer — the 1-block model deviates less from the (non-private)
pretrained base, so individual generated images stay more semantically coherent/
diagnosable even though the *population* they form matches the real distribution less
well in aggregate. Pixel proxies (LPIPS, SSIM) point the same direction as CLIP here,
consistent with "less deviation from a good pretrained base ≈ more locally
faithful-looking images," though per this repo's convention pixel proxies are secondary
evidence, not decisive on their own.

**Label-AUROC splits the difference**: the broader 7-pathology set follows the FID/FDS
direction (more blocks better), while the better-supported NIH 6-pathology set actually
peaks at 9 blocks (middle setting) — plausibly the genuine "sweet spot" once the
noisiest, low-support pathologies are excluded from the broader set.

**No single ablation_blocks setting wins on every metric family.** This is a real
utility-composition tradeoff, not a bug or noise artifact:
- Want the closest match to the real image *distribution* → 16 blocks (current best model, unchanged).
- Want the most reliably *individually diagnosable* synthetic images → 1 block.
- 9 blocks is the closest thing to a compromise, and wins outright on the
  best-supported label subset (NIH-6).

## Real-vs-generated degradation, (real−gen)/real

| task | 16 blocks | 9 blocks | 1 block |
|---|---|---|---|
| Label AUROC (all, 7-path macro) | 28.5% | 29.4% | 32.0% |
| Label AUROC (nih, 6-path macro) | 21.1% | **19.9%** | 22.9% |
| CLIP zero-shot label AUROC | 39.5% | 38.9% | **33.0%** |

Consistent with the finding above: CLIP-diagnosis degradation shrinks as trainable
blocks decrease (39.5%→33.0%), while label-AUROC degradation is roughly flat to
slightly worse.

## Caveat

`ablation8`/`ablation16` carry one extra field (`MT_DDPM_FDS`) that the `LI_eps10`
anchor summary lacks — `src/Eval_metric/fds.py` gained this metric as a local,
uncommitted change sometime between the anchor's computation and this sweep. It doesn't
affect `fds_symmetric` or any other field's definition, so the core comparison above is
unaffected; the anchor's `eval_summary.json` was simply computed before this field
existed and was not backfilled.
