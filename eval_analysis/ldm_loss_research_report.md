# LDM loss research: min-SNR + classifier-guidance loss (2026-09-07)

**Goal:** analyze the LDM's training objective from multiple angles and test whether an
added loss term (e.g. a classifier term, or a changed prediction target) can improve
downstream-classification-relevant fidelity, as a follow-up to
`experiment-synth-utility-pretrained-fullreal`'s finding that every downstream-classifier-
SIDE lever failed to rescue synthetic-only training.

**Outcome: inconclusive, for a specific and important reason.** The implementation works
correctly (verified via smoke tests under real Opacus DP-SGD wrapping) and adds zero DP-
budget cost (verified empirically, not just by code inspection). But at the pilot's DP-SGD
training scale, **DP noise dominates the loss function's influence on the trained weights
by a factor of over 4,000x** -- the pilot as designed cannot detect whether the new loss
objective helps or hurts, because virtually nothing about the training objective survives
the noise floor into the final model.

## What was implemented

Two opt-in, off-by-default, fully backward-compatible levers in
`Eval_metric/train_downstream_cls.py`... no -- in `LDM_dp_finetune.py` / `Model/Diffusion/{ddpm,LDM,LDM_dp}.py`:

- **`--min_snr_gamma`**: Hang et al. 2023 min-SNR-gamma loss reweighting (per-timestep
  weight `min(SNR(t), gamma)/SNR(t)`), eps-parameterization only.
- **`--cls_loss_weight` / `--cls_loss_max_t` / `--cls_loss_xrv_weights`**: an auxiliary BCE
  loss between a frozen TorchXRayVision classifier's prediction on a VAE-decoded x0
  estimate and the study's real CheXpert label, gated to low/mid timesteps (`t <
  cls_loss_max_t`). Reuses `Eval_metric/downstream_cls.py`'s `load_xrv_classifier`/
  `XRV_TO_CHEXPERT` (no duplicate label-mapping logic); defaults to the `-nih` XRV weight
  set to avoid any dependency on a classifier that may have seen this private dataset in
  its own (`-all`-style) training data.

Both are folded into the SAME per-sample backward pass Opacus already clips and noises.
**Opacus's privacy accounting depends only on `(target_epsilon, target_delta, sample_rate,
epochs)` -- not on the loss function's shape** -- so this costs zero extra DP budget. This
was verified empirically: two smoke-test runs (all-new-flags-off vs. both-flags-on),
identical `(target_epsilon, epochs, steps)`, produced byte-identical `eps_spent`/`sigma`.

Also extended `Data/mimic_cxr.py` with an opt-in `return_chexpert_vector` flag (3-tuple
`__getitem__` instead of 2-tuple, default off) to expose the raw CheXpert label vector the
auxiliary loss needs, and extended `run_feature_eval.py`/`downstream_cls.py`'s `label`
metric with `--manifest_csv`/`--label_text_mode` support (it previously only worked in
DICOM/`--root_path` mode, which has no local files for the p12 test split).

## The pilot and what it found

Two matched DP-SGD LoRA fine-tunes from the same `--pretrained_ckpt`, same 1,858-image
train subset (seed=42-sampled from the p10/p11 manifest) + full 748-image validate subset,
same `target_epsilon=10`/`epochs=8` -- **baseline** (all new flags off) vs. **clsloss**
(`min_snr_gamma=5.0`, `cls_loss_weight=0.2`, `cls_loss_max_t=300`).

**First result (both runs `--seed 42`):** generated 200 images each from the resulting
LoRA adapters (same p12-test LABEL+IMPRESSION prompts, index-aligned real reference via
the newly-added `--label_text_mode`) and ran the `label` metric (XRV DenseNet AUROC).
Macro AUROC was **identical to 4+ decimal places for every individual pathology**, not
just the macro average:

| weight set | gen_macro (baseline) | gen_macro (clsloss) | difference |
|---|---:|---:|---:|
| `-all` | 0.46636 | 0.46636 | 0.00000 |
| `-nih` | 0.45108 | 0.45108 | 0.00000 |

**Root cause, confirmed by directly diffing the saved LoRA weight tensors:**

| comparison | ||diff|| | ||diff|| / ||weights|| |
|---|---:|---:|
| baseline_seed42 vs clsloss_seed42 (loss differs, seed same) | 0.0003 | **0.0000** |
| baseline_seed43 vs clsloss_seed43 (loss differs, seed same) | 0.0003 | **0.0000** |
| baseline_seed42 vs baseline_seed43 (loss same, seed differs) | 18.4429 | **1.4105** |
| clsloss_seed42 vs clsloss_seed43 (loss same, seed differs) | 18.4429 | **1.4105** |

Changing the training seed moves the final LoRA weights by **>4,000x more** than changing
the loss function does -- under *either* of the two seeds tested. This rules out the
initial hypothesis that the two seed-42 runs merely happened to draw correlated DP noise
(true, but incomplete): re-running with an independent seed=43 realization shows the SAME
pattern. At this training scale (`max_grad_norm=0.001`, ~1,858 images x 8 epochs, ~14,864
physical per-sample steps), **DP noise is not just the dominant factor -- it appears to be
essentially the ONLY factor** determining the final model; the loss function's gradient
signal does not survive the noise floor into the trained weights in any detectable way.

## Why this happened, and what it means

`LDM_dp_finetune.py`'s `set_seed()` fixes the global torch RNG that Opacus's own per-step
Gaussian noise injection also draws from (documented in that function's own docstring).
That's a real methodological pitfall for same-seed A/B testing under DP-SGD in general --
but the seed=43 follow-up shows the deeper issue isn't really about seed-sharing between
arms; it's that **the loss-function signal is simply too small relative to the DP clip
norm and noise scale at this pilot's training volume**, regardless of which noise
realization is used.

**Project-level implication (beyond this one experiment):** other single-seed DP-SGD
comparisons already recorded in this repo (the `ablation_blocks` sweep, eps1-vs-eps10
rounds, epochs=10-vs-30) compare more structurally different configs -- different epsilon
changes the noise multiplier itself, different block counts change the trainable-parameter
count -- which are more likely to produce a real, seed-robust difference than this round's
loss-function-only change did. But it's worth treating any *small*-effect-size DP-SGD
comparison in this repo with caution unless it's been checked across multiple seeds.

## Status and recommended next step

The engineering deliverable is complete: both loss levers are implemented, verified
correct, verified DP-budget-free, and given a real (if inconclusive-by-design-limitation)
pilot test. The research question -- does either loss term actually improve downstream-
classification-relevant fidelity -- remains **open**, not answered "no": this pilot could
not have detected an effect even if one exists.

Two ways to get a decisive answer, neither committed to here given the GPU-hours already
spent this session:

1. **Full-scale run** (~49,907 images, more epochs, matching the scale of this repo's
   already-evaluated eps1/eps10 checkpoints) -- more accumulated gradient steps should
   improve the signal-to-noise ratio, at the cost of many more GPU-hours (multi-day scale,
   matching this repo's other full-scale training rounds).
2. **Multi-seed variance study at pilot scale** -- several seeds x 2 arms, comparing
   distributions of the resulting downstream AUROC rather than single point estimates --
   cheaper per-run than option 1, but needs enough seeds (a handful per arm at minimum) to
   say anything statistically meaningful, so the total cost is comparable.

## Files

- Implementation: `src/Model/Diffusion/ddpm.py`, `src/Model/Diffusion/LDM.py`,
  `src/Model/Diffusion/LDM_dp.py`, `src/LDM_dp_finetune.py`, `src/Data/mimic_cxr.py`
- Eval-pipeline fix: `Eval_metric/run_feature_eval.py`, `src/Eval_metric/downstream_cls.py`
- Pilot manifests: `data_manifests/mimic_p10_p12/{pilot_lossresearch_manifest,
  smoke_tiny_manifest,pilot_eval_prompts}.{csv.gz,txt}`
- Pilot outputs (gitignored, local only): `finetune_dp/pilot_lossresearch/{baseline,
  clsloss,baseline_seed43,clsloss_seed43}/`, `EVAL/gen_out/pilot_lossresearch/{baseline,
  clsloss}/`, `EVAL/metric/pilot_lossresearch/{baseline,clsloss}/`
- Tracked as `feature_list.json` -> `experiment-ldm-loss-research` (passing, with the
  research question explicitly left open -- see its `evidence` array for full detail)
