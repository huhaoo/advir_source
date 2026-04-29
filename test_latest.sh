#!/usr/bin/env bash
set -euo pipefail

DEFAULT_CKPT_DIR="/home/huhao/adv_ir/exp/haze/train_ckpt_nafnet_adv50_gaussian"
DEFAULT_DATASET_ROOT="/home/huhao/adv_ir/dataset_ours/haze_dataset_builder"
DEFAULT_DEVICE="cuda"
DEFAULT_OUTPUT_DIR="/home/huhao/adv_ir/tmp_demo/promptir_paired_dataset_test_latest_nafnet_adv_haze50_gaussian_haze_dataset_builder"

usage() {
  cat <<'EOF'
Usage:
  test_latest.sh [CKPT_DIR] [DATASET_ROOT] [DEVICE] [OUTPUT_DIR] [extra args...]
  test_latest.sh -s PATH [CKPT_DIR] [DATASET_ROOT] [DEVICE] [OUTPUT_DIR] [extra args...]

Options:
  -s, --s PATH   Keep restored images under PATH/restored (auto-add --save_restored).
  -h, --help     Show this help message.

Notes:
  - Positional defaults stay the same as before.
  - When -s/--s is set, OUTPUT_DIR will be overridden to PATH.
EOF
}

SAVE_OUTPUT_DIR=""
POSITIONAL_ARGS=()
EXTRA_ARGS=()
IN_EXTRA_ARGS=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -s|--s)
      if [ "$#" -lt 2 ]; then
        echo "[Error] -s/--s requires a path argument" >&2
        exit 1
      fi
      SAVE_OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      if [ "$IN_EXTRA_ARGS" -eq 1 ]; then
        EXTRA_ARGS+=("$1")
      elif [ "${#POSITIONAL_ARGS[@]}" -lt 4 ] && [[ "$1" != -* ]]; then
        POSITIONAL_ARGS+=("$1")
      else
        EXTRA_ARGS+=("$1")
        if [[ "$1" == -* ]]; then
          IN_EXTRA_ARGS=1
        fi
      fi
      shift
      ;;
  esac
done

CKPT_DIR="${POSITIONAL_ARGS[0]:-$DEFAULT_CKPT_DIR}"
DATASET_ROOT="${POSITIONAL_ARGS[1]:-$DEFAULT_DATASET_ROOT}"
DEVICE="${POSITIONAL_ARGS[2]:-$DEFAULT_DEVICE}"
OUTPUT_DIR="${POSITIONAL_ARGS[3]:-$DEFAULT_OUTPUT_DIR}"

if [ -n "$SAVE_OUTPUT_DIR" ]; then
  OUTPUT_DIR="$SAVE_OUTPUT_DIR"
  EXTRA_ARGS+=("--save_restored")
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
if [ -n "$SAVE_OUTPUT_DIR" ]; then
  echo "[Save Restored] $OUTPUT_DIR/restored"
fi

conda run -n advir --no-capture-output python /home/huhao/adv_ir/source/promptir_paired_dataset_test.py \
  --model_path "$LATEST_CKPT" \
  --dataset_root "$DATASET_ROOT" \
  --device "$DEVICE" \
  --model_arch "nafnet" \
  --output_dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"
