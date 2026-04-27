#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${1:-/home/huhao/adv_ir/exp/haze/train_ckpt_nafnet_adv50_gaussian}"
DATASET_ROOT="${2:-/home/huhao/adv_ir/dataset_ours/haze_dataset_builder}"
DEVICE="${3:-cuda}"
OUTPUT_DIR="${4:-/home/huhao/adv_ir/tmp_demo/promptir_paired_dataset_test_latest_nafnet_adv_haze50_gaussian_haze_dataset_builder}"

EXTRA_ARGS=()
if [ "$#" -gt 4 ]; then
  EXTRA_ARGS=("${@:5}")
fi

if [ ! -d "$CKPT_DIR" ]; then
  echo "[Error] ckpt_dir not found: $CKPT_DIR" >&2
  exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
  echo "[Error] dataset_root not found: $DATASET_ROOT" >&2
  exit 1
fi

mapfile -t CKPT_CANDIDATES < <(
  find "$CKPT_DIR" -maxdepth 1 -type f \( -name "*.ckpt" -o -name "*.pth" \) -printf "%T@|%p\n" | sort -nr
)

if [ "${#CKPT_CANDIDATES[@]}" -eq 0 ]; then
  echo "[Error] no checkpoint found in: $CKPT_DIR" >&2
  exit 1
fi

LATEST_CKPT="${CKPT_CANDIDATES[0]#*|}"

echo "[Latest CKPT] $LATEST_CKPT"
echo "[Dataset] $DATASET_ROOT"
echo "[Device] $DEVICE"
echo "[Output] $OUTPUT_DIR"

conda run -n advir --no-capture-output python /home/huhao/adv_ir/source/promptir_paired_dataset_test.py \
  --model_path "$LATEST_CKPT" \
  --dataset_root "$DATASET_ROOT" \
  --device "$DEVICE" \
  --model_arch "nafnet" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
