# Session Handoff

## Verified Now

**What is currently working:**
- `init.sh` passes end-to-end (container, path, compileall, eval-imports, 1-image
  generation smoke) in ~30s.
- Three checkpoints have complete, cross-checked `eval_summary.json`s: `LI_eps1`,
  `LI_eps10` (best model), `plain_eps10`. See `feature_list.json` `eval-*` entries.
- Report + dashboard generation pipeline (`aggregate.py` → `build_reports.py` →
  `gen_html.py` pattern) works from `eval_summary.json`/`ckpt_info.json` inputs.

**What verification actually ran:**
- `./init.sh` (2026-08-10, passed).
- Full generate→evaluate cycle for all three checkpoints above, each with an explicit
  md5 cross-check that outputs differ (see `feature_list.json` `pipeline-002`).

## Changed This Session

**Code or behavior added:**
- A prompt-builder script (currently a scratch file, not yet committed to the repo
  proper — see Next Best Step) that reconstructs `LABEL+IMPRESSION` training-time text
  format for arbitrary eval-split indices.

**Infrastructure or harness changes:**
- `CLAUDE.md` rewritten from the generic template to project-specific rules.
- `AGENTS.md` added (derived from `CLAUDE.md`, same content).
- `init.sh` upgraded from a 3-step path/compile check to a 5-step smoke test including
  an actual 1-image generation.
- `feature_list.json`, this file, and `clean-state-checklist.md` created for the first time.
- Removed `AGENTS.md.template.bak` / `init.sh.template.bak` (superseded by the real files).

## Broken Or Unverified

**Known defect:** none currently open — the LoRA-loading bug (see `claude-progress.md`
pitfall #1) was caught and fixed within this same session, with re-verification.

**Unverified path:** the prompt-builder script referenced above lives only at
`/tmp/claude-.../scratchpad/build_prompts.py` from this session — it was `docker cp`'d
into the container to run, but was never committed into the repo. If it's needed again
(e.g. for `eval-LI-milestones`), it will need to be recreated or retrieved.

**Risk for the next session:** none blocking. `roent_clip` reimplementation is
deliberately `blocked`, not a live risk.

## Next Best Step

**Highest-priority unfinished feature:** `eval-LI-milestones` (`feature_list.json`,
priority 4, `not_started`).

**Reasoning:** `finetune_dp/LABEL+IMPRESSION/eps10/ldm_lora_eps{1,3,5}.pt` and
`finetune_dp/lr2e-3_eps10/ldm_lora_eps{1,3,5}.pt` are LoRA-adapter-only checkpoints
saved mid-run as epsilon crossed those milestones, inside the single run that reached
eps10. Evaluating them gives a *within-run* privacy-utility curve to compare against
this session's `eval-LI-eps1` (a separately-trained standalone run) — worth knowing
whether milestone-based and standalone-trained checkpoints at the same nominal ε
actually behave the same.

**Success criteria:** each adapter has a complete `eval_summary.json` (104 keys,
matching this session's three), with md5-distinct generated images from its siblings.

**Constraints:** these are LoRA-adapter-only files (`{'lora': ..., 'lora_rank': ...}`,
not `{'model': ...}`), so they load via `build_and_load_ldm`'s path (a) —
`--dp_ckpt <base or eps10 ldm_dp_final.pt> --lora_ckpt ldm_lora_eps<N>.pt` — not the
"same file for both" trick used this session for the full-checkpoint case. Verify the
`[lora] adapter ... loaded=... eps_at_save=...` log line to confirm it took the
adapter-only path.

## Commands

**Startup:**
```
./init.sh
```

**Verification (per checkpoint, ~30-90 min on GPU — run inside CXR_dp_medclip):**
```
# 1. build prompts (once per text_mode, reused across checkpoints)
PYTHONPATH=/workspace/PrivaText-CXR/src /opt/conda/bin/python build_prompts.py \
  --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
  --eval_split test --out_dir /workspace/PrivaText-CXR/EVAL/prompts/test

# 2. generate (LoRA checkpoints need --lora_ckpt even if same file as --dp_ckpt!)
/opt/conda/bin/python -u src/LDM_dp_inference.py \
  --dp_ckpt <ckpt.pt> --lora_ckpt <ckpt.pt> \
  --vae_ckpt ./checkpoints/vae/vae_ep0070.pt --biobert_path /storage/hjchoi \
  --device_id 0 --descriptions ./EVAL/prompts/test/<label_impression|legacy>.txt \
  --n_samples 1 --seed 0 --output_dir ./EVAL/gen_out/<name>

# 3a. base metrics (any free GPU)
PYTHONPATH=/workspace/PrivaText-CXR/src /opt/medclip_env/bin/python -u \
  Eval_metric/run_feature_eval.py \
  --gen_dir ./EVAL/gen_out/<name>/samples --out_dir ./EVAL/metric/<name> \
  --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
  --eval_split test --max_real 361 --paired_from_split --device cuda:1 \
  --eval_model xrv --pca_dim 64 --metrics fds tsne fid ssim psnr lpips label \
  --ckpt <ckpt.pt>
# WAIT for 3a to fully exit before running 3b (same out_dir, race condition otherwise)

# 3b. CLIP metrics (must be cuda:0)
PYTHONPATH=/workspace/PrivaText-CXR/src /opt/medclip_env/bin/python -u \
  Eval_metric/run_feature_eval.py \
  --gen_dir ./EVAL/gen_out/<name>/samples --out_dir ./EVAL/metric/<name> \
  --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
  --eval_split test --max_real 361 --paired_from_split --device cuda:0 \
  --eval_model xrv --pca_dim 64 --clip_text_mode 'FINDINGS/IMPRESSION' \
  --metrics clip clip_label clip_diagnose --ckpt <ckpt.pt>
```

**Focused debug command** (confirm a checkpoint's LoRA delta actually loads, without
generating anything):
```
/opt/conda/bin/python -c "
import torch
ck = torch.load('<ckpt.pt>', map_location='cpu', weights_only=False)
print('use_lora:', ck['args'].get('use_lora'))
print('lora keys in model dict:', sum('.lora_A.' in k or '.lora_B.' in k for k in ck.get('model', {})))
"
```
