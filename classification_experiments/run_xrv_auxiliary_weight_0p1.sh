#!/usr/bin/env bash
set -euo pipefail
cd /workspace/PrivaText-CXR
mode="${1:-train}"
device="${2:-1}"
/opt/conda/bin/python -u classification_experiments/xrv_auxiliary_training.py "$mode" \
 --experiment-id 025 \
 --smoke-save-dir classification_experiments/artifacts/xrv_aux_lambda0p1_smoke \
 --aux-weight 0.1 --aux-timestep-max 250 \
 --root_path /storage/hjchoi/physionet.org/files/mimic-cxr/2.1.0 \
 --manifest_csv data_manifests/mimic_p10_p12/ldm_dp_manifest.csv.gz \
 --split train --text_mode LABEL+IMPRESSION --image_size 256 --max_length 512 \
 --vae_ckpt checkpoints/vae/vae_ep0070.pt --pretrained_ckpt checkpoints/ldm/ldm_epoch0100.pt \
 --biobert_path /storage/hjchoi --device_id "$device" --seed 42 \
 --use_lora true --lora_rank 4 --lora_alpha 4 --logical_batch 256 --physical_batch 4 \
 --epochs 3 --target_epsilon 10 --target_delta 1e-5 --max_grad_norm 0.001 \
 --lr 0.002 --warmup_steps 100 --scheduler_type linear_warmup \
 --num_workers 0 --do_validation false --save_every 3 --log_every 25 \
 --context_masking --frozen_bert_eval --finetune_biobert true \
 --save_dir classification_experiments/artifacts/ablation/xrv_aux_lambda0p1
