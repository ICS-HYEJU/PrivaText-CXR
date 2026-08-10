# Progress Log — PrivaText-CXR

## Current verified state (as of 2026-08-10)

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

## Session log

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
- Verdict: `LABEL+IMPRESSION eps10` is the best model (best FID/FDS/CLIP zero-shot);
  the expected privacy-utility tradeoff direction holds on every metric this repo's own
  evaluation guide ranks as reliable; CLIP zero-shot diagnosis degrades ~40% real→gen
  across all three models (the dominant, model-invariant utility cost).
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
