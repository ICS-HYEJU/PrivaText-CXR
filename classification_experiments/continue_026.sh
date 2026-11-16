#!/usr/bin/env bash
set -euo pipefail
cd /home/hjchoi/PycharmProjects/PrivaText-CXR
train_log="classification_experiments/artifacts/xrv_aux_stratified_train.log"
while ! grep -q "Training complete" "$train_log" 2>/dev/null; do
  if grep -Eq "Traceback|RuntimeError|CUDA out of memory" "$train_log" 2>/dev/null; then
    echo "E026 training failure detected" >&2
    exit 1
  fi
  sleep 30
done
test -s classification_experiments/artifacts/ablation/xrv_aux_stratified/ldm_dp_final.pt
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python -u \
  classification_experiments/ablation_eval.py generate --arm xrv_aux_stratified --device 1 --steps 50
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/ablation_eval.py evaluate --arm xrv_aux_stratified
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/ablation_eval.py compare --arm projection \
  --other xrv_aux_stratified --number 026_xrv_auxiliary_stratified_timestep --json-only
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python -u \
  classification_experiments/026_prompt_response.py generate --arm xrv_aux_stratified \
  --device 1 --seeds 32 --steps 50
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/026_prompt_response.py evaluate --arms xrv_aux_stratified
echo "E026_AUTOMATED_EVALUATION_COMPLETE"
