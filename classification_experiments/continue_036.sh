#!/usr/bin/env bash
set -euo pipefail

cd /workspace/PrivaText-CXR
result=RESULT_20261110_OFFICIAL_TRAIN_FILTERED_UTILITY
mkdir -p "$result/logs"
export PYTHONPATH=/workspace/PrivaText-CXR/.claude/worktrees/synth-utility-improve/src:/workspace/PrivaText-CXR

/opt/conda/bin/python -u classification_experiments/036_official_train_filtered_utility.py \
  generate --epsilon 1 --device 0 --steps 50 > "$result/logs/generate_eps1.log" 2>&1 &
p1=$!
/opt/conda/bin/python -u classification_experiments/036_official_train_filtered_utility.py \
  generate --epsilon 10 --device 1 --steps 50 > "$result/logs/generate_eps10.log" 2>&1 &
p10=$!
echo "$p1" > "$result/logs/generate_eps1.pid"
echo "$p10" > "$result/logs/generate_eps10.pid"

wait "$p1"
wait "$p10"

/opt/medclip_env/bin/python -u classification_experiments/036_official_train_filtered_utility.py \
  filter --epsilon 1 > "$result/logs/filter_eps1.log" 2>&1
/opt/medclip_env/bin/python -u classification_experiments/036_official_train_filtered_utility.py \
  filter --epsilon 10 > "$result/logs/filter_eps10.log" 2>&1

/opt/medclip_env/bin/python -u classification_experiments/036_official_train_filtered_utility.py \
  train > "$result/logs/classification.log" 2>&1

echo complete
