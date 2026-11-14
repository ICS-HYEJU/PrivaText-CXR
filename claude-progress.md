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
5. **`roent_clip_eval.py` does not exist anywhere in git history**, on any branch, despite
   being referenced by code in the `ldm-dp-text-prompt3` worktree and having produced
   output in old `EVAL/metric/lr2e-3/impression/*` runs. Treat as permanently lost unless
   reimplemented from the JSON schema (see `feature_list.json` `pipeline-006`, currently
   `blocked` by user decision).
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

## Session log

### 2026-08-20 — p10/p11 JPG training manifest + p12 custom test holdout

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
