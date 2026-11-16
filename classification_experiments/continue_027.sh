#!/usr/bin/env bash
set -euo pipefail
cd /home/hjchoi/PycharmProjects/PrivaText-CXR
train_log="classification_experiments/artifacts/counterfactual_train.log"
while ! grep -q "Training complete" "$train_log" 2>/dev/null; do
  if grep -Eq "Traceback|RuntimeError|CUDA out of memory" "$train_log" 2>/dev/null; then
    echo "E027 training failure detected" >&2
    exit 1
  fi
  sleep 30
done
test -s classification_experiments/artifacts/ablation/counterfactual/ldm_dp_final.pt
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python -u \
  classification_experiments/ablation_eval.py generate --arm counterfactual --device 1 --steps 50
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/ablation_eval.py evaluate --arm counterfactual
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/ablation_eval.py compare --arm projection \
  --other counterfactual --number 027_counterfactual_conditioning --json-only
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python -u \
  classification_experiments/027_context_shuffle.py --arm counterfactual --device 1
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python -u \
  classification_experiments/027_prompt_response.py generate --arm counterfactual \
  --device 1 --seeds 32 --steps 50
docker exec -w /workspace/PrivaText-CXR -e PYTHONPATH=/workspace/PrivaText-CXR/src \
  CXR_dp_medclip /opt/medclip_env/bin/python -u \
  classification_experiments/027_prompt_response.py evaluate --arms counterfactual
echo "E027_AUTOMATED_EVALUATION_COMPLETE"
