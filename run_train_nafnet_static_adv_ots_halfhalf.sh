#!/usr/bin/env bash
set -euo pipefail

cd /home/huhao/adv_ir/PromptIR

# 1:1 mix between default dehaze train set and static adversarial dataset.
# The static dataset can be smaller; loader cycles through it and applies simple augmentations.
conda run -n promptir --no-capture-output \
  python /home/huhao/adv_ir/source/promptir_static_adv_training.py \
  --model_arch nafnet \
  --de_type dehaze \
  --epochs "${EPOCHS:-128}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --accumulate_grad_batches "${ACCUMULATE_GRAD_BATCHES:-2}" \
  --lr "${LR:-2e-4}" \
  --patch_size "${PATCH_SIZE:-128}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --num_gpus "${NUM_GPUS:-3}" \
  --ckpt_dir "${CKPT_DIR:-/home/huhao/adv_ir/exp/train_ckpt_nafnet_random_adv_ots_s1_64_1024_d8_halfhalf}" \
  --adv_ratio 0.5 \
  --adv_samples_per_resample "${ADV_SAMPLES_PER_RESAMPLE:-16384}" \
  --adv_cache_root "${ADV_CACHE_ROOT:-/home/huhao/adv_ir/dataset_ours/random_adv_ots_s1_64_1024_d8}" \
  --degradation_size "${DEGRADATION_SIZE:-16384}" \
  --auto_resume
  # --wblogger "${WBLOGGER:-none}"

