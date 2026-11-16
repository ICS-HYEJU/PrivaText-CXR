#!/usr/bin/env bash
set -euo pipefail

# Fast baseline smoke check only. This is NOT full pipeline verification —
# a real generate+evaluate run takes 30-90 min per checkpoint on GPU and is
# invoked explicitly via the `verification` commands in feature_list.json,
# not automatically on every session start.

CONTAINER_NAME="CXR_dp_medclip"
PROJECT_DIR="/workspace/PrivaText-CXR"
TRAIN_PYTHON="/opt/conda/bin/python"          # training/generation env (Opacus, no MedCLIP)
EVAL_PYTHON="/opt/medclip_env/bin/python"     # evaluation env (MedCLIP, transformers 4.24.0)
SMOKE_CKPT="./finetune_dp/lr2e-3_eps10/ldm_dp_final.pt"
SMOKE_VAE="./checkpoints/vae/vae_ep0070.pt"

echo "[1/5] Docker container state CHECK ($CONTAINER_NAME)"
docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" | grep -qx true

echo "[2/5] Project path CHECK"
docker exec "$CONTAINER_NAME" sh -lc \
  "test -d '$PROJECT_DIR/src'"

echo "[3/5] Python source compile CHECK (train env)"
docker exec "$CONTAINER_NAME" sh -lc \
  "cd '$PROJECT_DIR' && '$TRAIN_PYTHON' -m compileall -q src"

echo "[4/5] Eval import CHECK (eval env, namespace-package merge across Eval_metric/ + src/Eval_metric/)"
docker exec "$CONTAINER_NAME" sh -lc \
  "cd '$PROJECT_DIR' && PYTHONPATH='$PROJECT_DIR/src' '$EVAL_PYTHON' -c \
    'from Eval_metric.text_utils import extract_report_sections; \
     from Eval_metric.clip_labelret import run_clip_label; \
     from Eval_metric.features import extract_features; \
     from Data.mimic_cxr import MIMICCXRDataset; \
     print(\"eval imports OK\")'"

echo "[5/5] Generation smoke test (1 image, reduced timesteps, known-good LoRA checkpoint)"
docker exec "$CONTAINER_NAME" sh -lc \
  "cd '$PROJECT_DIR' && \
   printf 'FINDINGS: init.sh smoke test.\n' > /tmp/init_smoke_prompt.txt && \
   rm -rf /tmp/init_smoke_out && \
   '$TRAIN_PYTHON' src/LDM_dp_inference.py \
     --dp_ckpt   '$SMOKE_CKPT' \
     --lora_ckpt '$SMOKE_CKPT' \
     --vae_ckpt  '$SMOKE_VAE' \
     --biobert_path /storage/hjchoi \
     --device_id 0 \
     --descriptions /tmp/init_smoke_prompt.txt \
     --n_samples 1 --sample_timesteps 50 --seed 0 \
     --output_dir /tmp/init_smoke_out \
     2>&1 | grep -E '\[lora\]|\[save\]|Traceback' && \
   test -f /tmp/init_smoke_out/samples/000_sample000.png"

echo "INIT & CHECK — baseline smoke passed. Full generate+evaluate is NOT run here;
see feature_list.json's per-feature 'verification' commands for that."
