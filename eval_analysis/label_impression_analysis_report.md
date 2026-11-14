# LABEL+IMPRESSION vs. plain conditioning — model comparison

Newly generated + evaluated for this analysis (no prior eval existed for these checkpoints).
Real reference set: MIMIC-CXR `test` split, 361 images (only `p10`/`p11` patient-prefix DICOMs
are downloaded locally; train used 31,553 images legacy / 29,888 images for LABEL+IMPRESSION
after dropping studies with no positive CheXpert label — see §5).

## 1. Models compared

| id | checkpoint | text_mode (training) | epoch/step | epsilon_spent |
|---|---|---|---|---|
| `LI_eps1` | `finetune_dp/LABEL+IMPRESSION/eps1/ldm_dp_final.pt` | LABEL+IMPRESSION | 10 / 1160 | 0.9086 |
| `LI_eps10` | `finetune_dp/LABEL+IMPRESSION/eps10/ldm_dp_final.pt` | LABEL+IMPRESSION | 10 / 1160 | 8.5913 |
| `plain_eps10` | `finetune_dp/lr2e-3_eps10/ldm_dp_final.pt` | legacy FINDINGS+IMPRESSION | 10 / 1220 | 8.6117 |

All three are DP-SGD + LoRA (rank=4, alpha=4) fine-tunes of the same pretrained LDM
(`checkpoints/ldm/ldm_epoch0100.pt`), lr=2e-3, `max_grad_norm=0.001`, `target_delta=1e-5`.
`LI_eps1`/`LI_eps10` share every training setting except `target_epsilon` (1 vs 10) — a clean
privacy-utility pair. `LI_eps10`/`plain_eps10` land at nearly the same spent epsilon (8.59 vs
8.61) but differ in conditioning text — a clean text-mode ablation.

`LABEL+IMPRESSION` text = `"LABEL: <positive CheXpert pathologies>" + " IMPRESSION: <section>"`;
legacy text = `"FINDINGS: <section> IMPRESSION: <section>"`. Generation prompts for evaluation
were built to mirror each model's own training-time text format, per test-split item, in the
same index order as the real reference set (337/361 items had a usable CheXpert row for the
LABEL+IMPRESSION prompts; 24/361 fell back to legacy text — logged, not dropped, so all three
models were scored against the identical 361-image real set).

## 2. Full metric comparison

Direction: ↓ lower better, ↑ higher better. Reliability order (established convention in this
repo, see `docs/EVALUATION_GUIDE.md` §6.3): **FID, FDS, LPIPS, CLIP gap > label-AUROC > SSIM/PSNR**
(pixel proxies — generation ≠ reconstruction, treat as secondary).

| metric | dir | LI_eps1 | LI_eps10 | plain_eps10 |
|---|---|---|---|---|
| FID | ↓ | 2.4626 | **2.2710** | 2.3527 |
| FDS (sym) | ↓ | 29.751 | **27.203** | 28.141 |
| FDS gen‖real (hallucination) | ↓ | 23.578 | **21.816** | 22.234 |
| FDS real‖gen (mode collapse) | ↓ | 35.924 | **32.589** | 34.048 |
| SSIM | ↑ | 0.2109 | 0.2108 | 0.2103 |
| PSNR | ↑ | 11.250 | 11.269 | **11.302** |
| LPIPS | ↓ | 0.5275 | 0.5269 | **0.5265** |
| Label AUROC gap (all, 7 pathologies) | ↓ | **0.1933** | 0.2016 | 0.1967 |
| Label AUROC ratio (all) | ↑ | **0.7264** | 0.7146 | 0.7216 |
| Label AUROC gap (nih, 6 pathologies) | ↓ | 0.1466 | 0.1269 | **0.1251** |
| Label AUROC ratio (nih) | ↑ | 0.7566 | 0.7893 | **0.7922** |
| CLIP zero-shot diagnosis gap | ↓ | 0.3396 | **0.3328** | 0.3607 |
| CLIP zero-shot diagnosis ratio | ↑ | 0.5969 | **0.6051** | 0.5720 |
| CLIPScore gap (cos) | ↓ | -0.0065 | **-0.0053** | -0.0063 |

(Full machine-readable table: `label_impression_comparison.csv`.)

## 3. Goal 2 — Privacy-utility tradeoff (`LI_eps1` vs `LI_eps10`, same config)

```
Expected as epsilon rises (less DP noise): FID/FDS/LPIPS/CLIP-gap fall, SSIM/PSNR rise.
  FID                    2.463 -> 2.271   (-7.8%)   eps10 better  [OK, expected]
  FDS(sym)               29.75 -> 27.20   (-8.6%)   eps10 better  [OK, expected]
  FDS gen||real          23.58 -> 21.82   (-7.5%)   eps10 better  [OK, expected]
  FDS real||gen          35.92 -> 32.59   (-9.3%)   eps10 better  [OK, expected]
  LPIPS                  0.5275 -> 0.5269 (-0.1%)   eps10 better  [OK, expected, tiny]
  CLIP zero-shot gap     0.340 -> 0.333   (-2.0%)   eps10 better  [OK, expected]
  Label AUROC gap (all)  0.193 -> 0.202   (+4.3%)   eps1 better   [reversed - noisy metric, see below]
  Label AUROC gap (nih)  0.147 -> 0.127   (-13.4%)  eps10 better  [OK, expected]
```

**Verdict: the expected privacy-utility tradeoff holds cleanly** — every distribution/perceptual
metric (FID, all three FDS variants, LPIPS, CLIP zero-shot gap), which this repo's own guide
ranks as most reliable, moves in the theoretically-expected direction as epsilon rises from 0.91
to 8.59. The one exception, `label_all_auroc_gap` (7-pathology macro), reverses — but its `nih`
counterpart (6-pathology macro, different classifier weights) agrees with the expected direction,
and per §6.3 of the guide, per-pathology AUROC is only trustworthy when support ≥10 positive/
negative — several of the 7 "all" pathologies (e.g. Lung Lesion n=6, Fracture n=3) sit near or
below that floor, so this single reversal is most plausibly small-sample noise rather than a real
effect. All magnitude changes are modest (≤10% on the reliable metrics) — over this short
10-epoch / ~1160-step training run, tightening the privacy budget by ~9.5x costs a small,
consistent, but not dramatic amount of utility.

## 4. Bonus — text-mode ablation at matched epsilon (`LI_eps10` vs `plain_eps10`, eps≈8.6)

```
  FID                    2.271 vs 2.353   LI_eps10 better (-3.6%)
  FDS(sym)               27.20 vs 28.14   LI_eps10 better (-3.5%)
  FDS gen||real          21.82 vs 22.23   LI_eps10 better (-1.9%)
  FDS real||gen          32.59 vs 34.05   LI_eps10 better (-4.5%)
  CLIP zero-shot gap     0.333 vs 0.361   LI_eps10 better (-7.7pp gap narrower)
  CLIP zero-shot ratio   0.605 vs 0.572   LI_eps10 better
  Label AUROC ratio(all) 0.715 vs 0.722   plain_eps10 slightly better
  Label AUROC ratio(nih) 0.789 vs 0.792   plain_eps10 slightly better (~tie)
  SSIM/PSNR/LPIPS                          essentially tied either way (<0.5% apart)
```

**Verdict: `LABEL+IMPRESSION` conditioning wins at matched privacy budget.** It's ahead on every
distribution metric (FID, all FDS variants) and clearly ahead on CLIP zero-shot diagnosis
(narrower gap, higher ratio) — the metrics this repo's guide weights most heavily. `plain_eps10`
edges ahead only on label-AUROC ratio and the pixel proxies, and only by ~1% or less. Adding
explicit CheXpert-label text to the conditioning prompt appears to help the model generate
images that are both closer to the real distribution and more correctly diagnosable by a
CLIP zero-shot reader, without costing anything on the more reliable metrics.

## 5. Goal 3 — real-vs-generated performance degradation, (real−gen)/real

Computed for all three models on the tasks that have both a real-image and generated-image
score in the same eval run (downstream classifier AUROC, CLIP zero-shot diagnosis accuracy).

| model | Label AUROC (all, 7-path macro) | Label AUROC (nih, 6-path macro) | CLIP zero-shot diagnosis accuracy |
|---|---|---|---|
| `LI_eps1` | real 0.7065 → gen 0.5132 (**+27.4%** degradation) | real 0.6020 → gen 0.4555 (**+24.3%**) | real 0.8427 → gen 0.5030 (**+40.3%**) |
| `LI_eps10` | real 0.7065 → gen 0.5048 (**+28.5%**) | real 0.6020 → gen 0.4752 (**+21.1%**) | real 0.8427 → gen 0.5099 (**+39.5%**) |
| `plain_eps10` | real 0.7065 → gen 0.5098 (**+27.8%**) | real 0.6020 → gen 0.4770 (**+20.8%**) | real 0.8427 → gen 0.4820 (**+42.8%**) |

**Reading**: degradation is remarkably consistent across all three configurations (~27-29% for
the broader label set, ~21-24% for the NIH-backbone label set, ~40-43% for CLIP zero-shot
diagnosis) — neither the privacy budget nor the text-conditioning choice explored here moves this
number much. The largest, most consistent gap is in **CLIP zero-shot diagnosability**: a
classifier/CLIP reader loses roughly 40% of its diagnostic accuracy when handed this pipeline's
synthetic images instead of real ones, regardless of model variant. This is the dominant
utility cost of using generated images as a substitute for real data in this pipeline, well
above what privacy-budget or prompt-format tuning can currently close.

## 6. Goal 1 — best model

**`LABEL+IMPRESSION eps10` (epsilon_spent=8.591) is the best of the three.** It has the best FID,
best FDS on all three variants, and the best CLIP zero-shot diagnosis gap/ratio — the four
metric families this repo's own evaluation guide ranks as most trustworthy — while being
statistically tied (within ~1%) with the other two models on the less-reliable label-AUROC and
pixel-proxy metrics. `LI_eps1` and `plain_eps10` are both viable at their respective use cases
(tighter privacy budget, or legacy-format compatibility) but neither beats `LI_eps10` on the
metrics that matter most for judging generation quality.

## 7. Methodology notes & caveats

- **Bug caught and fixed during this analysis**: the first generation pass for all three
  checkpoints used `--dp_ckpt` only. Since these are LoRA fine-tunes (256 LoRA-delta tensors per
  checkpoint), omitting `--lora_ckpt` causes `load_state_dict(..., strict=False)` to silently
  drop every LoRA key, so the generator fell back to the un-fine-tuned pretrained base for all
  three — confirmed by byte-identical generated images across different checkpoints. Fixed by
  passing the same file as both `--dp_ckpt` and `--lora_ckpt` (triggers the "full LoRA checkpoint"
  load path, which injects and merges the LoRA delta). Re-verified post-fix that outputs differ
  and that `[lora] full ckpt ... lora_keys=256 missing=0 unexpected=0` appears in every run's log.
- **`roent_clip` (RoentGen-style retrieval)**, present in the older `EVAL/metric/lr2e-3/impression/*`
  reference runs, is **not included here** — its implementing module
  (`Eval_metric/roent_clip_eval.py`) does not exist in git history on any branch (an orphaned,
  never-committed file from a past session); reimplementing it was explicitly descoped for this
  analysis.
- **Real reference set is capped at 361 images** (test split) because only `p10`/`p11`
  patient-prefix DICOMs are downloaded locally (train: 31,553 legacy-text / 29,888
  LABEL+IMPRESSION-eligible images, same p10/p11-only constraint).
- CLIP-based metrics (`clip`, `clip_label`, `clip_diagnose`) were run pinned to `cuda:0` — the
  MedCLIP backend appears to hardcode device 0 internally regardless of `--device`, causing a
  device-mismatch crash when pointed at `cuda:1`.
- CLIP text mode for scoring was fixed to `FINDINGS/IMPRESSION` (real report content) for all
  three models, independent of each model's own training-time conditioning format, so the CLIP
  comparison is apples-to-apples across models.
