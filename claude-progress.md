# Progress Log — PrivaText-CXR

## Current verified state (as of 2026-08-20)

- Docker container `CXR_dp_medclip` is the only place this pipeline runs. Two Python
  envs inside it: `/opt/conda/bin/python` (train/generate) and
  `/opt/medclip_env/bin/python` (evaluate). See `CLAUDE.md` for why they can't be swapped.
- Local MIMIC-CXR copy only has `p10`/`p11` patient-prefix DICOMs downloaded.
  `train`=31,553 (legacy text) / 29,888 (LABEL+IMPRESSION-eligible), `test`=361.
  Every eval in this repo is against that same 361-image test reference.
- Three checkpoints now have a complete metric suite (first time ever evaluated for
  these specific files — see `feature_list.json` `eval-*` entries and
  `eval_analysis/label_impression_analysis_report.md`):
  - `finetune_dp/LABEL+IMPRESSION/eps1/ldm_dp_final.pt` (ε_spent=0.9086)
  - `finetune_dp/LABEL+IMPRESSION/eps10/ldm_dp_final.pt` (ε_spent=8.5913) — **best model**
  - `finetune_dp/lr2e-3_eps10/ldm_dp_final.pt` (ε_spent=8.6117, legacy text_mode)
- `init.sh` runs a fast smoke check only (~30s): container/path/compileall/eval-imports
  + a 1-image, 50-step generation using a known-good LoRA checkpoint. It does **not**
  run a real generate+evaluate cycle — those take 30–90 min per checkpoint on GPU and
  are invoked explicitly via `feature_list.json`'s per-feature `verification` commands.
- `--ablation_blocks` now works under `--use_lora` too (`pipeline-007`, 2026-08-11):
  restricts LoRA injection to `blocks[N-1:]` of the 16 SpatialTransformer blocks,
  mirroring the pre-existing full-DP-SGD `configure_dp_params` semantics exactly.
  Previously it was silently ignored whenever `use_lora=True` — true for every
  checkpoint trained so far, including all three `eval-*` entries above (they all
  trained with the equivalent of `ablation_blocks=-1`, i.e. all 16 blocks).
- `experiment-ablation-blocks-sweep` (2026-08-11/12): two more checkpoints trained and
  evaluated, `ablation_blocks=8` (9/16 blocks) and `ablation_blocks=16` (1/16 blocks),
  same config as the eps10 anchor otherwise. Result is a genuine tradeoff, not a single
  winner — FID/FDS favor more blocks, CLIP zero-shot label AUROC favors fewer (by a wide
  margin). Full writeup: `eval_analysis/ablation_blocks_report.md` +
  `ablation_blocks_dashboard.html`.
- **DONE (2026-08-20 through 2026-08-26):** eps1/eps10 LABEL+IMPRESSION retrained from
  scratch on the new p10/p11 JPG manifest (`data_manifests/mimic_p10_p12/
  ldm_dp_manifest.csv.gz`, see `pipeline-008`), at BOTH epochs=10 and epochs=30 (4
  checkpoints total), fully generated+evaluated against a fixed 10,240-image p12-test
  subsample. Do not treat the DICOM-era `eval-LI-*` numbers above as reflecting this new
  data -- different test set entirely. Two dashboards published:
  epochs=10-only (https://claude.ai/code/artifact/4309eb50-a9c2-4dee-b618-0e3f857507a1)
  and the full 4-way epoch×epsilon comparison
  (https://claude.ai/code/artifact/527395cc-7624-422b-bcd5-aa04cd91eddd). See
  `feature_list.json`'s `experiment-jpg-retrain-eps1-eps10` /
  `experiment-jpg-retrain-eps1-eps10-epochs30` entries for full evidence.

## Known pitfalls (read before touching generation/eval code)

1. **LoRA checkpoints need `--lora_ckpt` even when it's the same file as `--dp_ckpt`.**
   `--dp_ckpt` alone loads via `strict=False`, which silently drops all 256 LoRA-delta
   tensors and falls back to the pretrained base. Symptom: two different checkpoints
   produce byte-identical (same md5) generated images. This actually happened on
   2026-08-10 and cost a full re-generation cycle to catch and fix.
2. **`run_feature_eval.py` merges into `eval_summary.json` by read-modify-write.**
   Never run two invocations against the same `--out_dir` concurrently (race condition,
   caught and killed mid-session before it corrupted anything).
3. **CLIP-based metrics (`clip`, `clip_label`, `clip_diagnose`) must use `--device cuda:0`.**
   MedCLIP appears to hardcode device 0 internally; any other device raises
   "Expected all tensors to be on the same device" and the metric silently skips.
4. **Two `Eval_metric` directories merge as a namespace package.** Top-level `Eval_metric/`
   (has `clip_label`, newest) + `src/Eval_metric/` (most other metric modules) only work
   together with `PYTHONPATH=<repo>/src` set AND the driver invoked as
   `Eval_metric/run_feature_eval.py` from the repo root (not `src/Eval_metric/run_feature_eval.py`,
   which is a stale copy missing `clip_label`).
5. ~~**`roent_clip_eval.py` does not exist anywhere in git history**~~ — **CORRECTED
   2026-08-21**: the `.py` was indeed never committed, but its compiled bytecode
   survived at `src/Eval_metric/__pycache__/roent_clip_eval.cpython-311.pyc` and was
   fully recoverable via `dis`/`marshal` disassembly (docstrings, signatures, constants,
   and control flow all preserved in bytecode). Reimplemented, verified, `pipeline-006`
   now `passing`. **Lesson: before declaring an uncommitted file "permanently lost",
   check `__pycache__` for a stale `.pyc` — bytecode survives even when the source
   was never `git add`ed.**
6. **`--ablation_blocks` used to be silently ignored under `--use_lora`** (it only fed
   `configure_dp_params`, the non-LoRA path). Fixed 2026-08-11 (`pipeline-007`) — now
   also honored by `configure_lora_params`/`inject_lora_cross_attention`, and by
   inference-time LoRA loading in `LDM_dp_inference.py`. Every checkpoint trained
   *before* 2026-08-11 has `ablation_blocks=-1` recorded in its saved `args` regardless
   of what was passed at the CLI, since it had no effect either way under LoRA — don't
   read a pre-2026-08-11 checkpoint's `ablation_blocks` value as meaningful.
7. **Three distinct "zero-shot" CLIP metric families exist — don't conflate them or
   call any of them "accuracy".** `clip_label_clipzs_{gen,real}_macro/gap/ratio` is
   zero-shot label **AUROC** (single-abnormality studies, `cos(image, "{label}")` vs
   ground truth, real-vs-gen comparable). `clip_label_clipzs_*_R@k/mAP` is a *different*
   zero-shot label **retrieval** metric off the same scores. `clipdiag_zeroshot_label_
   auroc_macro` is a real-images-only encoder validity check (no gen counterpart,
   identical across every model). Also don't confuse any of these with
   `label_all_auroc_*`/`label_nih_auroc_*`, which are AUROC from the XRV DenseNet-121
   classifier, not CLIP at all. Full detail in the `clip-metric-terminology` memory.
8. **`LDM_dp_finetune.py`'s `--vae_ckpt`/`--pretrained_ckpt` argparse defaults are
   hardcoded HOST-absolute paths** (`/home/hjchoi/PycharmProjects/PrivaText-CXR/
   checkpoints/...`), which do not exist inside the `CXR_dp_medclip` container
   (project path is `/workspace/PrivaText-CXR` there). Omitting these flags and
   relying on the default crashes immediately with `FileNotFoundError`. Always
   pass both explicitly (e.g. `--vae_ckpt ./checkpoints/vae/vae_ep0070.pt
   --pretrained_ckpt ./checkpoints/ldm/ldm_epoch0100.pt`, relative to the
   container's cwd at `/workspace/PrivaText-CXR`). Caught 2026-08-20 when both
   the JPG-manifest eps1 and eps10 retraining jobs died instantly on launch;
   the defaults themselves were left unfixed (not this task's scope).
9. **Two "FDS" metrics exist in `eval_summary.json` and can rank checkpoints in
   OPPOSITE order — never call a single "FDS" value without saying which one.**
   `fds_symmetric`/`fds_gen_given_real`/`fds_real_given_gen` (Ledoit-Wolf
   full-covariance KL, `Eval_metric/fds.py::compute_fds`) agrees with FID.
   `MT_DDPM_FDS` (t-SNE 2-D joint embedding, symmetric Gaussian KL,
   `compute_mt_ddpm_fds`) is a DIFFERENT, much lower-dimensional estimate — in the
   2026-08-26 epoch×epsilon round it ranked all four checkpoints backwards from
   FID/Ledoit-Wolf-FDS (best became worst and vice versa) and moved in the
   opposite direction with epoch count. Neither is "wrong"; they answer different
   questions (full-covariance overlap vs. coarse 2-D-embedding overlap). Always
   label which FDS you're reporting.

## Session log

### 2026-09-07 (2) — LDM loss research: min-SNR + classifier-guidance loss, DP-noise dominates at pilot scale

**Goal:** analyze the LDM's training objective (loss target, conditioning, architecture)
from multiple angles per user request, implement the most promising lever(s), and test
whether they improve downstream-classification-relevant fidelity -- follow-up to the same-
day synth-utility v2 finding that every downstream-classifier-SIDE lever had already
failed, shifting suspicion to the LDM checkpoints' own generative fidelity.

**Completed:**
- Grounded multi-angle analysis of the actual DP-SGD fine-tuning code (not literature
  alone): current setup is plain eps-parameterization simple-MSE loss, uniform timestep
  weighting, cross-attention-only trainable (ResBlock backbone + output layer stay frozen
  even under full DP-SGD), text-only conditioning. Key grounding fact used throughout:
  Opacus's privacy accounting depends only on `(target_epsilon, target_delta, sample_rate,
  epochs)`, not the loss function's shape -- so a new loss term folded into the same
  per-sample backward pass is DP-budget-free.
- Implemented `--min_snr_gamma` (ddpm.py/LDM.py, Hang et al. 2023 min-SNR-gamma
  reweighting) and `--cls_loss_weight`/`--cls_loss_max_t`/`--cls_loss_xrv_weights`
  (LDM_dp.py, an auxiliary BCE loss between a frozen TorchXRayVision classifier's
  prediction on a VAE-decoded x0 estimate and the study's real CheXpert label, timestep-
  gated). Refactored LDM.py's `p_losses` into `_diffusion_forward`+
  `_loss_from_model_output` so the DP subclass reuses the SAME UNet forward for the
  auxiliary term (no duplicate forward, no risk to Opacus's per-sample-grad hooks).
  Extended `Data/mimic_cxr.py` with an opt-in `return_chexpert_vector` flag (3-tuple
  `__getitem__`, default off) for the raw label vector the auxiliary loss needs.
- Verified on real GPU under real Opacus wrapping: a regression smoke test (all new flags
  off) and a new-path smoke test (both on) both complete cleanly; the timestep gate
  correctly fires the classifier loss only on samples with `t < cls_loss_max_t`; both
  smoke tests produce IDENTICAL `eps_spent`/`sigma` given identical
  `(target_epsilon, epochs, steps)` -- empirical confirmation of the zero-DP-budget claim.
- Extended `run_feature_eval.py`/`downstream_cls.py`'s `label` metric with
  `--manifest_csv`/`--label_text_mode` support (previously DICOM-only, unusable for the
  p12 test split which has no local DICOMs).
- Ran a real pilot A/B: baseline vs (min-SNR + classifier loss), same 1,858-image train
  subset + full validate split, `target_epsilon=10`/`epochs=8`, `--seed 42` both --
  finished in ~35min each (much faster than the ~5h/round precedent this repo's older
  `search`-split rounds used, since this pilot's images/epochs count is far smaller).
  Generated 200 images each, ran the `label` metric (XRV AUROC) -- **result was IDENTICAL
  to 4+ decimals for every pathology between the two arms.**
- Root-caused via a direct weight-diff on the saved LoRA adapters:
  `||baseline_seed42 - clsloss_seed42|| / ||baseline_seed42|| = 0.0000`. Re-ran BOTH arms
  with `--seed 43` to test whether it was just correlated noise from sharing a seed
  (`set_seed()` fixes the global RNG that Opacus's own noise injection also draws from):
  same result under the independent seed too (`0.0000`), while cross-seed diffs (same loss
  function, different seed) are **>4,000x larger** (`1.4105` relative). **Corrected
  conclusion: at this training scale (~1,858 images x 8 epochs, `max_grad_norm=0.001`), DP
  noise doesn't just dominate the loss-function signal -- it appears to be essentially the
  ONLY determinant of the final weights.** The loss function's gradient signal does not
  survive the noise floor into the trained model in any numerically detectable way, under
  either seed tested.
- Published `eval_analysis/ldm_loss_research_report.md` with full detail + a new Claude
  Artifact HTML dashboard: (see below, published after this log entry).

**Known unresolved / left for later:**
- The underlying research question (does either loss term help?) is genuinely open, not
  answered "no" -- this pilot could not have detected an effect at its scale even if one
  exists. Two paths to a decisive answer, neither committed to this session given the
  GPU-hours already spent: (a) a full ~49,907-image scale run (many more accumulated
  steps -> better signal-to-noise ratio, multi-day cost matching this repo's other
  full-scale rounds), or (b) a multi-seed variance study at pilot scale (several seeds x 2
  arms, compare distributions not single points).
- Project-level flag for future sessions: other single-seed DP-SGD comparisons already in
  this repo (`ablation_blocks` sweep, eps1-vs-eps10, epochs=10-vs-30) compare more
  structurally different configs (different epsilon changes the noise multiplier itself;
  different block counts change trainable-parameter count) which are more likely to show
  a real, seed-robust difference than this round's loss-function-only change did -- but
  worth treating any *small*-effect-size single-seed DP-SGD comparison in this repo with
  caution unless checked across multiple seeds.
- Did not re-run generate+eval for the seed=43 pair (the weight-level evidence already
  answers the question conclusively).

### 2026-09-07 — synth-utility v2: pretrained backbone + full real pool, all 3 session goals tested (none met)

**Goal (user-set for this session):** push the synth-utility downstream classifier to
macro AUROC>=0.8, stop synthetic-only training from degrading toward chance, and get
real+synthetic mixing to beat real-only by >=1%. Follow-up to `experiment-synth-utility-
densenet-scratch` (2026-08-27/28), whose from-scratch/small-pool design landed far
short (best 0.646, synthetic-only ~chance, mixing always worse than real-only).

**Completed:**
- Worked in git worktree `.claude/worktrees/synth-utility-improve` (branch
  `worktree-synth-utility-improve`) per background-job isolation rules. Since worktrees
  only carry committed history and `EVAL/`/generated images are gitignored, copied over
  the (still-uncommitted-in-main-checkout) v1 artifacts needed to build on
  (`train_downstream_cls.py`, `build_synth_utility_pool.py`, `apply_clahe.py`,
  `pool_manifest.csv.gz`, `eval_subsample_test_n5000_seed42.csv.gz`) and referenced the
  already-generated `EVAL/gen_out/synth_utility/{eps1,eps10}` (5,500 images each) by
  their absolute container path rather than copying — no new ~22h generation needed.
- Extended `Eval_metric/train_downstream_cls.py`: `--pretrained` (ImageNet DenseNet-121
  backbone + proper ImageNet mean/std normalization via new `to_model_input()`, vs the
  original plain [-1,1] 3ch-repeat) and `--real_manifest` (lets arms R/RS5500 draw the
  real half from an arbitrary pool_manifest.csv.gz-schema csv instead of pool A, so a
  much larger real pool can be used without touching the RS1100/S5500 code paths).
- Built `data_manifests/mimic_p10_p12/real_full/pool_manifest.csv.gz` — ALL 49,907
  LABEL+IMPRESSION-eligible train rows (not just the 1,100-image pool A), same
  find_chexpert_csv/load_chexpert_gt labeling path as v1.
- Ran 8 arms total across both GPUs (see `eval_analysis/synth_utility_v2_report.md` for
  full per-pathology numbers): `R1100_pretrained`, `S5500_eps1_pretrained`,
  `S5500_eps10_pretrained`, `S5500_eps1_clahe_pretrained` (CLAHE via the previously-built
  but unused `apply_clahe.py`, clip_limit=2.0/tile=8×8), `Rfull`, `Rfull_ep20`,
  `RfullS_eps10`, `RfullS_eps1`.
- **Results (macro AUROC, 12/14 reliable labels, min_pos=10, same fixed 5,000-image test
  set as v1):** R1100_pretrained=0.6720 (vs v1 from-scratch 0.6461, backbone alone only
  +4.0%) < **Rfull=0.7631 (BEST, full 49,907-image real pool, ImageNet backbone, 10
  epochs)** > Rfull_ep20=0.7524 (20 epochs — WORSE, overfit: train loss 0.159→0.083) >
  RfullS_eps10=0.7559 (full real + eps10 synth, −0.94% vs Rfull) > RfullS_eps1=0.7533
  (full real + eps1 synth, −1.28% vs Rfull) >> S5500_eps1_clahe_pretrained=0.5184 >
  S5500_eps1_pretrained=0.5010 > S5500_eps10_pretrained=0.4781 (all three synthetic-only
  arms at/below chance).
- **Goal verdict — all 3 NOT MET:** (1) AUROC 0.8: best 0.763, −0.037 short; more
  training does not close the gap (it overfits instead). (2) No synthetic-only
  degradation: fails robustly — every lever tried (from-scratch→pretrained backbone,
  CLAHE contrast normalization) left synthetic-only training at/below chance; this
  points at the LDM checkpoints' own generative fidelity as the bottleneck, not the
  downstream-classifier setup. (3) Mixing beats real-only by >=1%: fails for BOTH
  epsilons (eps10 −0.9%, eps1 −1.3%) — note this breaks the repo's previously-observed
  "eps1>eps10" downstream-utility pattern (CLIP-zeroshot/RoentGen/v1 synth-utility all
  favored eps1); here eps10-mixing scored higher than eps1-mixing, though both lost to
  real-only.
- **Ruled in vs. ruled out** (for whoever picks this up next): real-pool SIZE is the
  dominant lever tested (1,100→49,907 real images, +13.6 points at fixed
  backbone/epochs) — much bigger than the pretrained-backbone effect alone (+4.0%).
  Backbone choice, more epochs, and CLAHE preprocessing were all tried as fixes for the
  synthetic-image problem specifically and none worked.
- Published `eval_analysis/synth_utility_v2_report.md` + `synth_utility_v2_comparison.csv`
  + a NEW Claude Artifact HTML dashboard:
  https://claude.ai/code/artifact/b6f5ce86-07e6-4c38-849d-21bd224f44a2

**Known unresolved / left for later:**
- Still open whether the synthetic-only/mixing failure is fixable at all with the
  *current* eps1/eps10 LABEL+IMPRESSION checkpoints, or whether it needs better LDM
  checkpoints first (more LDM training epochs, or a different `ablation_blocks` setting
  per `experiment-ablation-blocks-sweep`) before downstream-classifier-side tuning can
  help further — this round ruled out every classifier-side lever tried, which shifts
  suspicion toward the generative side.
- Single seed/run per arm, same caveat as v1 — no variance estimate.
- The main checkout (non-worktree) still has its own uncommitted local changes unrelated
  to this feature (`apply_clahe.py`, `roent_clip_eval.py`, `--manifest_csv` threading
  through several eval scripts, JPG comparison dashboards, `taming/`, `analysis/`) from
  a different session's work-in-progress — untouched by this session, still sitting
  uncommitted there; worth reconciling/committing in a future session if still relevant.

### 2026-08-27 — synth-utility DenseNet-from-scratch experiment: pool built, generation launched

**Goal:** Replicate the user-referenced paper's downstream-classification utility table
(DenseNet-121 trained FROM SCRATCH on real/synthetic mixes, multi-label AUROC on a held-out
real test set) using this repo's data. See `feature_list.json`'s
`experiment-synth-utility-densenet-scratch` and the approved plan
(`/home/hjchoi/.claude/plans/recursive-purring-wirth.md`) for full design/assumptions.

**Completed so far:**
- Clarified scope with the user (AskUserQuestion): synthetic images from BOTH
  `JPG_LABEL_IMPRESSION` eps1_ep30 and eps10_ep30 checkpoints (7 classifier-training runs
  total: shared R1.1k + {eps1,eps10} x {R+S1.1k, S5.5k, R+S5.5k}); labels = full CheXpert-14;
  test set = fresh 5,000-row subsample of the existing 10,240-row p12 test subsample; real
  training pool sampled from `train_balanced.csv.gz`.
- Built two new scripts (neither existed before -- `downstream_cls.py` only *scores* a frozen
  pretrained XRV classifier, it never trained one):
  - `Eval_metric/build_synth_utility_pool.py` -- draws a seeded 5,500-row pool B from the
    LABEL+IMPRESSION-filtered train split (`MIMICCXRDataset`), first 1,100 rows = pool A
    (reused as the real half of every real-containing arm). Writes
    `pool_manifest.csv.gz` (labels via `downstream_cls.find_chexpert_csv`/`load_chexpert_gt`,
    no duplicate CheXpert-parsing logic) + `label_impression.txt` prompts for generation.
  - `Eval_metric/train_downstream_cls.py` -- assembles one arm (R/RS1100/S5500/RS5500),
    trains a `torchvision.models.densenet121(weights=None)` (true from-scratch, 1ch->3ch
    replication, 14-way head) with masked multi-label BCE (uncertain/missing labels dropped
    from the loss, not just eval) + inverse-prevalence pos_weight, evaluates AUROC on the
    fixed test set reusing the same `auroc_per_pathology`-style min_pos=10 reliability
    convention as `downstream_cls.py`.
  - Both `py_compile` + namespace-merge import clean; smoke-tested ALL FOUR arm code paths
    end-to-end (tiny n=20 pool, 2 dummy epochs, mock synth_dir built from real images
    standing in for generated ones) before committing to the real run -- confirmed correct
    item counts per arm (R=4, RS1100=8, S5500=20, RS5500=24) and that masked-BCE/AUROC/ROC
    plotting all execute without error.
- Built the real pool: `data_manifests/mimic_p10_p12/synth_utility/pool_manifest.csv.gz`
  (5,500 rows, seed=42, pool A = first 1,100) + `label_impression.txt`. Label prevalence
  printed and looks sane across all 14 CheXpert columns (e.g. No Finding pos=950/5500,
  Fracture pos=187/5500).
- Built the fixed test set: `data_manifests/mimic_p10_p12/eval_subsample_test_n5000_seed42.csv.gz`
  (seed=42 subsample of the existing 10,240-row p12 subsample, 5,000/5,000 kept, all
  `dataset_split=='test'`).
- **Launched generation** (both on GPU0/GPU1, `/opt/conda/bin/python src/LDM_dp_inference.py`,
  `--lora_ckpt` = same file as `--dp_ckpt` per the LoRA-loading pitfall):
  eps1_ep30 -> `EVAL/gen_out/synth_utility/eps1` (cuda:0, pid 35522),
  eps10_ep30 -> `EVAL/gen_out/synth_utility/eps10` (cuda:1, pid 35579). 5,500 images each.
  A persistent background Monitor polls both logs every 30 min for progress/errors/completion.

**Important discovered detail**: `LDM_dp_inference.py`'s `main()` holds ALL generated
images in memory (`all_images` list) and only calls `save_outputs` (individual
PNGs+descriptions.csv, then the grid_all.png that has OOM-killed prior large rounds) ONCE
at the very end -- so no incremental per-image PNGs appear on disk during the run, only
after all samples finish. Confirmed the same benign "OOM-killed at grid_all.png, PNGs+csv
already written" pattern recurred here (both processes exit-137'd, both had 5,500/5,500
PNGs + complete descriptions.csv already on disk -- verified index range 0-5499
complete/unique and md5(index 0) differs between eps1/eps10 before proceeding).

**Completed (generation + classifier training + report, same day):**
- Generation finished both sides after ~22h wall-clock (matched the ~21-22h estimate).
  Verified before proceeding (see above).
- Ran all 7 classifier-training arms immediately (GPU0: R -> eps1_RS1100 -> eps1_S5500 ->
  eps1_RS5500; GPU1: eps10_RS1100 -> eps10_S5500 -> eps10_RS5500), ~16 min wall-clock total
  -- vastly faster than generation, as expected (from-scratch DenseNet on 1.1k-6.6k images,
  30 epochs, is cheap compared to 1000-step DDPM sampling).
- **Result (macro AUROC, 12/14 reliable CheXpert labels, min_pos=10):**
  R(real-only,n=1100)=**0.6461** (best overall) > eps1_RS1100=0.6369 > eps10_RS1100=0.6275 >
  eps1_RS5500=0.6014 > eps10_RS5500=0.5903 > eps1_S5500=0.5211 > eps10_S5500=0.5006 (~chance).
  Four findings, all reported in the dashboard: (1) real-only beats every synthetic-containing
  arm at both privacy levels -- synthetic never helped this from-scratch classifier; (2)
  synthetic-ONLY training collapses toward chance broadly across pathologies (not one outlier),
  even though these same checkpoints score well on FID/CLIP-zeroshot as *generative* metrics --
  fidelity and downstream-training utility are different questions; (3) 5x more synthetic
  volume (5.5k vs 1.1k) does not recover the real-only gap and costs MORE, not less; (4)
  **eps1 beats eps10 in all 3 matched comparisons** (+1.5% to +4.1% relative) -- the SAME
  direction as this repo's earlier-documented eps1>eps10 finding on CLIP-zeroshot label AUROC
  and RoentGen retrieval (2026-08-23/26 rounds), even though FID/Ledoit-Wolf FDS favor eps10.
  This is now the THIRD independent metric family (after CLIP-zeroshot, RoentGen) where
  eps1 > eps10 despite FID/FDS saying the opposite -- worth treating as a real, recurring
  project-level pattern, not a one-off (saved to memory).
- Published `eval_analysis/synth_utility_report.md` + `synth_utility_comparison.csv`
  (full per-pathology AUROC x 7 arms) + a NEW Claude Artifact HTML dashboard:
  https://claude.ai/code/artifact/bc5f68c8-5153-47e8-8aa9-3422a1397d89

**Known unresolved / left for later:**
- Single seed/run per arm -- no variance estimate; treat deltas under ~2-3 points as
  directional given how small these training pools are (1.1k-6.6k images).
- `eval-LI-milestones` (still not_started, unrelated).

### 2026-08-26 — epochs=30 round complete: 4-way (eps1/eps10 × epochs 10/30) comparison

**Goal:** Finish `experiment-jpg-retrain-eps1-eps10-epochs30` -- generate, evaluate, and
build a 4-way comparison HTML that includes BOTH the epochs=30 results and the
already-complete epochs=10 results, per the user's explicit standing instruction
(don't just compare eps1_ep30 vs eps10_ep30 in isolation).

**Completed:**
- Generation completed both sides (10,240/10,240 each), same benign grid_all.png
  OOM-kill pattern as the epochs=10 round -- verified no data loss both times before
  proceeding (PNG count/uniqueness, descriptions.csv row count).
- Eval Stage A+B ran clean both sides, including `roent_clip` and the new
  `clip_zeroshot_auroc_roc.png` (both came for free since that code was already
  wired in from earlier this session).
- Built the 4-way comparison (eps1_ep10, eps10_ep10, eps1_ep30, eps10_ep30) as two
  separated axes rather than one flat table:
  - **Epoch axis (new finding this round):** more epochs (10->30) monotonically
    improves every distribution/pixel metric (FID/FDS/SSIM/PSNR/LPIPS/XRV
    label-AUROC ratio) at BOTH privacy levels. Gain is larger at eps10 (FID
    3.36->2.84, ~15%) than eps1 (3.77->3.40, ~10%) -- plausibly eps1's noisier DP
    gradient limits how much extra training can extract. CLIP zero-shot label
    AUROC ratio is essentially flat across epoch count at both eps levels.
    RoentGen retrieval trends flat-to-slightly-worse with more epochs, most
    visible at eps1.
  - **Privacy axis:** the same eps1-vs-eps10 mixed signal from the epochs=10 round
    (distribution metrics favor eps10, RoentGen/CLIP-zeroshot favor eps1) holds at
    BOTH epoch counts -- confirms it's a real property of this trade space, not an
    artifact of one training length.
- Published `eval_analysis/jpg_4way_report.md` + `.csv` + `jpg_4way_dashboard.html`
  (small-multiple slope charts for the epoch axis, grouped table for both axes), as
  a new Claude Artifact: https://claude.ai/code/artifact/527395cc-7624-422b-bcd5-aa04cd91eddd
- Also added, mid-session, per a direct user request: per-pathology ROC curve
  plots for the CLIP zero-shot label AUROC metric (`clip_zeroshot_auroc_roc.png`),
  reusing `downstream_cls.py`'s existing `_plot_roc` helper rather than duplicating
  plotting code (`pipeline-010`). Generated immediately for the epochs=10 round on
  request, and it now runs automatically as part of every `--metrics clip_label`
  pass going forward (confirmed it fired for free during the epochs=30 Stage B).

**Known unresolved / left for later:**
- `eval-LI-milestones` (still not_started, unrelated).
- The RoentGen eps1>eps10 finding (both epoch counts) is still from a single
  200-query/400-candidate stratified sample per checkpoint -- worth a re-run with a
  different seed before treating it as more than suggestive.
- No further epoch counts (e.g. 20, 50) planned unless requested.

### 2026-08-23 — JPG-manifest eps1 vs eps10: generation, full eval, HTML (round complete)

**Goal:** Finish the loop kicked off 2026-08-20/21 -- generate from both JPG-retrained
checkpoints, evaluate with the full metric suite (base + CLIP family, including the
newly-recovered `roent_clip`), and publish an eps1-vs-eps10 comparison dashboard.

**Completed:**
- Generation finished both sides (10,240/10,240 images each). Both processes were
  OOM-killed (exit 137) AFTER writing all PNGs + `descriptions.csv` -- the crash was in
  `LDM_dp_inference.py`'s optional `grid_all.png` matplotlib step, sized for the old
  n=361 case and never tested at n=10,240 (a ~2,560-row subplot grid). Verified no data
  loss: PNG count/index-uniqueness/openability and `descriptions.csv` row count all
  checked before proceeding. md5-cross-checked index-0 images differ between checkpoints
  and match the earlier 5-image sanity check exactly (seed=0 reproducibility holds).
- Eval Stage A (fds/tsne/fid/ssim/psnr/lpips/label, cuda:1, both checkpoints
  concurrently) and Stage B (clip/clip_label/clip_diagnose/roent_clip, cuda:0) both ran
  clean on the first try -- 115-key `eval_summary.json` each, n_real=n_gen=n_paired=10240.
- Result: distribution-fidelity metrics (FID 3.36 vs 3.77, both FDS variants, XRV label
  AUROC ratio) mildly favor eps10 -- the expected privacy-utility direction, matching the
  original DICOM-era finding. **RoentGen retrieval (first real run of the recovered
  metric) and CLIP zero-shot label AUROC instead favor eps1**, consistently across every
  cutoff (i2i/i2t x k=5/10). Reported as a genuine mixed signal, not resolved into one
  winner -- same honest-reporting posture as the 2026-08-12 ablation_blocks sweep.
- Published `eval_analysis/jpg_eps1_vs_eps10_report.md` + `.csv` +
  `jpg_eps1_vs_eps10_dashboard.html`, and as a new Claude Artifact:
  https://claude.ai/code/artifact/4309eb50-a9c2-4dee-b618-0e3f857507a1
- Per standing instructions, launched `experiment-jpg-retrain-eps1-eps10-epochs30`
  (same config, `--epochs 30`) the moment generation freed GPU0/GPU1, without waiting for
  this eval to finish -- it ran concurrently sharing GPU0/GPU1 with this eval's two
  stages (accepted slowdown, not incorrect, per established precedent).

**Known unresolved / left for later:**
- `experiment-jpg-retrain-eps1-eps10-epochs30` still training as of this log entry
  (~1 day/checkpoint expected, 3x the epochs=10 run's 7h52m).
- The RoentGen eps1>eps10 finding is from a single 200-query/400-candidate stratified
  sample per checkpoint -- worth a re-run with a different seed or larger sample before
  treating it as more than suggestive (see the dashboard's own caveat).

### 2026-08-21 — roent_clip recovered from bytecode + reimplemented (pipeline-006 unblocked)

**Goal:** Add RoentGen-style i2i/i2t retrieval to the metrics used once the current
10,240-image generation finishes. User initially wanted ConVIRT (i2i) + CXR-RePaiR
(i2t), matching the actual RoentGen paper's protocol.

**Completed:**
- Investigated ConVIRT and CXR-RePaiR checkpoint availability via WebSearch/WebFetch:
  **neither has a usable public checkpoint.** ConVIRT has no official release ever
  (only unofficial training code, e.g. `fbrynpk/ConVIRT`). CXR-RePaiR's own repo
  README links a Stanford Box download that now 404s, and no mirror exists on
  HuggingFace or elsewhere. User chose medclip as the substitute for BOTH directions
  after seeing this evidence.
- Separately, discovered `roent_clip_eval.py` (thought permanently lost, see old
  pitfall #5) is actually recoverable: its compiled `.pyc` survived in
  `src/Eval_metric/__pycache__/`. Disassembled it (`dis`/`marshal`) to recover exact
  docstrings/signatures/constants/control-flow, cross-checked field-for-field against
  the historical `EVAL/metric/lr2e-3/impression/eps10/roent_clip.json` this repo still
  has on disk (protocol name, threshold values, filter-stat key names all matched).
- Reimplemented as `src/Eval_metric/roent_clip_eval.py` (faithful-intent rewrite of the
  recovered spec, manifest_csv-aware per the pipeline-009 pattern) and wired into
  `Eval_metric/run_feature_eval.py` as `--metrics roent_clip`.
- Verified end-to-end with a self-retrieval smoke test (300 real p12-test JPGs fed in
  as a mock `--gen_dir`) since no real generated images exist yet for this experiment
  round: manifest_csv loading, single-target-label filtering, IMPRESSION extraction +
  token-length filtering, encoder loading, and JSON schema all confirmed working.
  Prec@50=1.0 both directions (expected for self-retrieval), Prec@5/10 non-degenerate.
- **Also discovered while testing**: the legacy DICOM path (`--root_path` only, no
  `--manifest_csv`) is now broken for verification purposes -- p10/p11 DICOM files
  have since been deleted from `/storage` (superseded by the JPG pipeline). Not a bug
  in this feature; a pre-existing environment state change. Anything needing the old
  DICOM `test`=361 reference set no longer works; use `--manifest_csv` going forward.

**Known unresolved / left for later:**
- `roent_clip` has NOT been run against real generated images yet -- queue it into
  `experiment-jpg-retrain-eps1-eps10`'s eval stage (add to `--metrics` alongside
  clip/clip_label/clip_diagnose) once the current 10,240-image generation finishes.

### 2026-08-20 (2) — Retrain eps1/eps10 on JPG manifest: kicked off, in progress

**Goal:** Repeat the eps1-vs-eps10 train→eval→compare→HTML loop, this time on the
new `data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz` (p10/p11 JPG train, p12
JPG custom test) instead of the old DICOM-only path. See `pipeline-008` for the data
side and `experiment-jpg-retrain-eps1-eps10` for this experiment's full status.

**Completed so far:**
- Sized the p12-test generation/eval set: benchmarked actual throughput first
  (~14 sec/image at full 1000-step DDPM sampling, timed against the existing DICOM
  eps10 checkpoint) before committing GPU time -- flagged that the user's initial
  10,240-image choice implies ~40h/checkpoint generation alone; user explicitly
  re-confirmed 10,240 after seeing the estimate.
- Built a fixed, seeded (seed=42) 10,240-row subsample of the 35,107-row
  LABEL+IMPRESSION-eligible p12 test pool:
  `data_manifests/mimic_p10_p12/eval_subsample_test_n10240_seed42.csv.gz`. Same
  subsample is reused for both eps1 and eps10 (not rebuilt per-checkpoint) so the
  comparison is fair. Reload-verified 10,240/10,240 kept.
- Built `EVAL/prompts/jpg_test_n10240/label_impression.txt` from that subsample in
  matching dataset order (index i == prompt line i == real-reference index i, so
  `--paired_from_split` pixel/label pairing in `run_feature_eval.py` stays correct).
  8,621/10,240 have a real IMPRESSION section; 1,619 are LABEL-only.
- Extended the eval pipeline with `--manifest_csv` support (`pipeline-009`) so
  dataset-mode real-reference construction can read the JPG manifest instead of
  DICOM `root_path` scanning, across every metric family: `Eval_metric/
  run_feature_eval.py` (fds/paired pixel), `clipscore.py::real_baseline` (clip),
  `downstream_cls.py::compute_label_agreement` (label), `clip_labelret.py::
  run_clip_label` (clip_label), `clip_diagnose.py::_load_real` (clip_diagnose).
  `--root_path` is still required alongside `--manifest_csv` (fallback for the
  default CheXpert csv location); omitting `--manifest_csv` leaves every existing
  DICOM eval run byte-identical.
- Launched training: eps1 on GPU0 (`target_epsilon=1.0`), eps10 on GPU1
  (`target_epsilon=10.0`), both `text_mode=LABEL+IMPRESSION`, otherwise identical
  hyperparameters to the original `EVAL/metric/LABEL_IMPRESSION/eps{1,10}/
  ckpt_info.json`. `total_steps=1940` each (49,907 train samples, vs the original
  DICOM run's 1160 from 29,888 samples -- consistent with the ~1.67x larger JPG
  train set).

**Caught mid-session:** first launch attempt crashed instantly on both GPUs --
`--vae_ckpt`/`--pretrained_ckpt` argparse defaults in `src/LDM_dp_finetune.py:
290-291,334-335` are hardcoded **host**-absolute paths
(`/home/hjchoi/PycharmProjects/PrivaText-CXR/checkpoints/...`) that don't exist
inside the container. Worked around by passing both explicitly at the CLI
(relative paths, matching what the original eps10 ckpt_info.json recorded) --
did NOT change the argparse defaults themselves (flagged to the user, left as a
known footgun for whoever next omits these flags relying on the default). Added
as pitfall #8 below.

**Known unresolved / left for later:**
- Training is running in the background (many hours expected; the original
  DICOM eps10 run's own duration wasn't recorded, and this run has ~1.67x more
  steps). Not yet done as of this log entry.
- Generation (10,240 images/checkpoint, ~40h/checkpoint) and the full 2-stage
  eval pass have NOT started -- blocked on training finishing.
- HTML dashboard (new artifact URL) not yet built -- blocked on eval finishing.

### 2026-08-20 (1) — p10/p11 JPG training manifest + p12 custom test holdout

**Goal:** Replace the DICOM-only input path with audited JPG/report pairing, train on
p10/p11 without their official-test rows, reserve all p12 patients for testing, control
single-positive No Finding imbalance, and verify at least 10,240 test images.

**Completed:**
- Added `src/Data/prepare_mimic_jpg.py` and generated reproducible manifests under
  `data_manifests/mimic_p10_p12/` from record-list, official split, CheXpert, JPG,
  and report files. All 112,413 p10-p12 candidate rows were checked: 1,403 JPGs
  missing, zero present JPG decode failures, and zero report misses among readable JPGs.
- Enforced patient-disjoint custom split: p10/p11 official train=72,249 usable,
  validation=802, p10/p11 official test excluded, and all usable p12=37,190 test.
  Train/validation/test patient intersections are all zero; test>=10,240 passes.
- Deterministically capped single-positive No Finding at 6,230 (seed 42, twice the
  largest abnormal single-positive class), reducing train to 53,857 without changing
  validation/test. Added image/study/patient class counts and sufficiency labels.
- Added manifest/JPG support to `MIMICCXRDataset` and `LDM_dp_finetune.py`. Verified
  real JPG tensors and all manifest splits; LABEL+IMPRESSION usable counts are
  train=49,907, validation=748, test=35,107.

**Artifacts:** `data_manifests/mimic_p10_p12/{README.md,split_summary.json,`
`ldm_dp_manifest.csv.gz,class_distribution.csv}`.

### 2026-08-12 — ablation_blocks=8/16 sweep: train, generate, evaluate, report

**Goal:** Run the actual sweep enabled by `pipeline-007` — train `ablation_blocks=8`
and `=16` LoRA checkpoints (same config as the `eps10` anchor), generate+evaluate both,
and report results as an HTML dashboard (new standing instruction from user, see below).

**Completed:**
- Trained both on GPU0/GPU1 in parallel: `ablation16` (1/16 blocks) finished in 4h45m,
  `ablation8` (9/16 blocks) in 5h53m. Both reached `epsilon_spent=8.591346449555628`,
  identical to the anchor to 4 decimals — confirmed DP-SGD's noise calibration doesn't
  depend on trainable-parameter count, so the sweep isolates block-count cleanly.
- Generated 361 images each (reused the existing `LABEL+IMPRESSION` prompt file),
  md5-cross-checked against the anchor and each other (all distinct).
- Ran the full 2-stage metric suite (base + CLIP) on both, sequentially per out_dir to
  avoid the known race-condition pitfall — but ran the two checkpoints' CLIP stages
  *concurrently* on cuda:0 against each other (different out_dirs, no race risk there,
  just shared GPU compute).
- Key finding: **FID/FDS and CLIP zero-shot label AUROC disagree** on which block count
  is best. FID/FDS improve monotonically with more trained blocks (16 best). CLIP
  zero-shot label AUROC ratio (`clip_label_clipzs_ratio` -- an AUROC, not an accuracy;
  see below) improves monotonically with *fewer* trained blocks (1 block best,
  0.670 vs 16-block's 0.605) — a real, reported-honestly tradeoff, not resolved into a
  single winner. Label-AUROC splits: the broad 7-pathology set follows FID/FDS, the
  better-supported NIH-6 set peaks at 9 blocks.
- New user instruction (durable, saved to memory `html-result-summaries`): all future
  result summaries must be HTML dashboards, each as its own new artifact (new URL), not
  a reuse/update of a prior one. Built `eval_analysis/ablation_blocks_dashboard.html`
  (separate scratchpad file, separate Artifact URL from the first dashboard) alongside
  the usual markdown report + CSV.
- Caught (not a bug of mine, but worth recording): `src/Eval_metric/fds.py` gained an
  extra `MT_DDPM_FDS` field as a local uncommitted change sometime between the anchor's
  computation and this sweep — present in ablation8/16's summaries, absent from the
  anchor's. Confirmed via `git status`/`git log` that it's a pre-existing local
  modification, not something this session introduced; doesn't affect any field used in
  the comparison.

**Known unresolved / left for later:**
- No further ablation_blocks values planned unless requested (e.g. 4, 12 for finer
  resolution between the three tested points).
- `eval-LI-milestones` (still `not_started`, unrelated to this sweep).

### 2026-08-11 — ablation_blocks support under LoRA

**Goal:** Enable sweeping `--ablation_blocks` (how many of the 16 SpatialTransformer
blocks get fine-tuned) while keeping the rest of the config identical to the best
model (`eval-LI-eps10`), to isolate "how many attention blocks does DP-utility
actually need" as a single clean variable.

**Completed:**
- Diagnosed that `--ablation_blocks` had zero effect for every checkpoint trained so
  far, since all of them used `--use_lora` and the flag was only wired into the
  non-LoRA `configure_dp_params` path.
- Considered switching to full DP-SGD (`use_lora=False`) to make `ablation_blocks`
  active, but rejected it: that would also change the trainable-parameter count by
  ~2 orders of magnitude (LoRA rank-4 adapters vs. full block weights, 44.3% of the
  101M-param UNet), confounding "block count" with "DP noise-to-signal ratio" — two
  different variables moving at once.
- Implemented `ablation_blocks` support directly in the LoRA path instead (see
  `feature_list.json` `pipeline-007`): `inject_lora_cross_attention` now walks
  SpatialTransformer blocks and only wraps CrossAttention Linears inside
  `blocks[N-1:]`, matching `configure_dp_params`'s existing `-1`/`N` convention
  exactly. Wired through `configure_lora_params`, `LDM_dp_finetune.py`'s training
  call, and both LoRA-loading paths in `LDM_dp_inference.py` (adapter-only and
  full-checkpoint-with-embedded-LoRA).
- Verified: unit-level block/param counts match the `blocks[N-1:]` rule exactly for
  N ∈ {-1,1,8,16,20} (out-of-range N=20 correctly degrades to 0 blocks, which fails
  loudly via the existing `RuntimeError('...returned empty param list')` guard rather
  than silently training nothing); `./init.sh` regression-passed post-change with a
  pre-existing `ablation_blocks=-1` checkpoint, confirming no behavior change for
  anything trained before today.

**Known unresolved / left for later:**
- `experiment-ablation-blocks-sweep` (`feature_list.json`, `not_started`): the actual
  sweep hasn't been run yet — this session only built and verified the capability.
  `eval-LI-eps10` already IS the `ablation_blocks=-1` (all 16 blocks) anchor point;
  don't retrain it.

### 2026-08-10 — Evaluate LABEL+IMPRESSION vs legacy-text checkpoints; harness setup

**Goal:** Score three never-before-evaluated checkpoints for privacy-utility tradeoff
(Goal 2), text-conditioning ablation, and real-vs-generated degradation ratio (Goal 3),
select the best model (Goal 1); then set up harness-engineering files for continuity.

**Completed:**
- Built an index-aligned `LABEL+IMPRESSION` prompt generator (matches the training-time
  text format, falls back to legacy text for the 24/361 studies with no positive
  CheXpert label, never drops/reindexes so real-image pairing stays valid).
- Generated 361 images each for the three checkpoints (see pitfall #1 above for the
  detour this took).
- Ran the full metric suite (FID/FDS/SSIM/PSNR/LPIPS/label-AUROC/CLIP/clip_label/
  clip_diagnose) on all three (see pitfall #2/#3 for the detour this took).
- Aggregated into `eval_analysis/label_impression_analysis_report.md` +
  3 CSVs + `label_impression_dashboard.html` (published as a Claude Artifact).
- Verdict: `LABEL+IMPRESSION eps10` is the best model (best FID/FDS/CLIP zero-shot
  label AUROC); the expected privacy-utility tradeoff direction holds on every metric
  this repo's own evaluation guide ranks as reliable; CLIP zero-shot label AUROC
  degrades ~40% real→gen across all three models (the dominant, model-invariant
  utility cost).
- Created harness-engineering files (`CLAUDE.md` rewritten from generic template,
  `AGENTS.md` derived, `init.sh` upgraded to a real smoke test, `feature_list.json`,
  this file, `session-handoff.md`, `clean-state-checklist.md`) per user request,
  adapted from `walkinglabs/learn-harness-engineering`'s templates.

**Known unresolved / left for later:**
- `eval-LI-milestones` (not_started): `ldm_lora_eps{1,3,5}.pt` adapters saved mid-run
  inside both the `LABEL+IMPRESSION/eps10` and `lr2e-3_eps10` training jobs are
  unevaluated — would give a second, within-run privacy-utility curve.
- `roent_clip` reimplementation intentionally left `blocked` (user decision).
- `evaluator-rubric.md` / `quality-document.md` intentionally not created — low value
  for a single-session-at-a-time research pipeline right now; revisit if multiple
  worktrees start running in parallel routinely.
