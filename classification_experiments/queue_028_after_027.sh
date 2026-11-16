#!/usr/bin/env bash
set -euo pipefail
cd /home/hjchoi/PycharmProjects/PrivaText-CXR
while ! grep -q "E027_AUTOMATED_EVALUATION_COMPLETE" /tmp/e027_continue.log 2>/dev/null; do
  if ! pgrep -f 'counterfactual_training.py|continue_027.sh' >/dev/null; then
    echo "E027 stopped without completing automated evaluation" >&2
    exit 1
  fi
  sleep 30
done
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip /opt/conda/bin/python \
  classification_experiments/adjudicate_027.py

# E028 begins only after E027 has been fully adjudicated. Any smoke failure stops the queue.
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip bash \
  classification_experiments/run_label_flip_clipping.sh smoke label_flip_clip 1
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip bash \
  classification_experiments/run_label_flip_clipping.sh smoke structured_clip_control 0

docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip bash -lc \
  'bash classification_experiments/run_label_flip_clipping.sh train structured_clip_control 0 2>&1 | tee classification_experiments/artifacts/structured_clip_control_train.log' &
control_pid=$!
docker exec -w /workspace/PrivaText-CXR CXR_dp_medclip bash -lc \
  'bash classification_experiments/run_label_flip_clipping.sh train label_flip_clip 1 2>&1 | tee classification_experiments/artifacts/label_flip_clip_train.log' &
candidate_pid=$!
wait "$control_pid"
wait "$candidate_pid"
echo "E028_TRAINING_COMPLETE"
