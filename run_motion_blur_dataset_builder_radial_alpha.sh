#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/huhao/adv_ir"
BUILDER_PY="${PROJECT_ROOT}/source/motion_blur_dataset_builder_radial_alpha.py"
MERGE_PY="${PROJECT_ROOT}/source/merge_motion_blur_split_shards.py"

SPLITS_JSON="${SPLITS_JSON:-${PROJECT_ROOT}/dataset_path/promptir_clear_depth_sets_only.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/dataset_ours/motion_blur_rotation0.02}"

TRAIN_COUNT="${TRAIN_COUNT:-16384}"
VAL_COUNT="${VAL_COUNT:-1024}"
TEST_COUNT="${TEST_COUNT:-1024}"

ALPHA_MIN="${ALPHA_MIN:-0.01}"
ALPHA_MAX="${ALPHA_MAX:-0.02}"

NUM_STEPS="${NUM_STEPS:-16}"
MODE="${MODE:-bilinear}"
PADDING_MODE="${PADDING_MODE:-border}"
ALIGN_CORNERS_FLAG="${ALIGN_CORNERS_FLAG:---align_corners}"
BATCHIFY_STEPS_FLAG="${BATCHIFY_STEPS_FLAG:---batchify_steps}"

PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-128}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
SAVE_VIZ="${SAVE_VIZ:-1}"
VIZ_COUNT="${VIZ_COUNT:-24}"
VIZ_STRIDE="${VIZ_STRIDE:-48}"
VIZ_MAX_POINTS="${VIZ_MAX_POINTS:-2048}"
VIZ_ARROW_SCALE="${VIZ_ARROW_SCALE:-5.0}"

TRAIN_USE_SHARDS="${TRAIN_USE_SHARDS:-1}"
TRAIN_SHARD0_COUNT="${TRAIN_SHARD0_COUNT:-5462}"
TRAIN_SHARD1_COUNT="${TRAIN_SHARD1_COUNT:-5461}"
TRAIN_SHARD2_COUNT="${TRAIN_SHARD2_COUNT:-5461}"
TRAIN_SHARD0_SEED="${TRAIN_SHARD0_SEED:-123}"
TRAIN_SHARD1_SEED="${TRAIN_SHARD1_SEED:-124}"
TRAIN_SHARD2_SEED="${TRAIN_SHARD2_SEED:-125}"
TRAIN_SHARD0_DEVICE="${TRAIN_SHARD0_DEVICE:-cuda:0}"
TRAIN_SHARD1_DEVICE="${TRAIN_SHARD1_DEVICE:-cuda:1}"
TRAIN_SHARD2_DEVICE="${TRAIN_SHARD2_DEVICE:-cuda:2}"

VAL_SEED="${VAL_SEED:-223}"
TEST_SEED="${TEST_SEED:-323}"
VAL_DEVICE="${VAL_DEVICE:-cuda:0}"
TEST_DEVICE="${TEST_DEVICE:-cuda:0}"

if [[ ! -f "${BUILDER_PY}" ]]; then
  echo "builder script not found: ${BUILDER_PY}" >&2
  exit 1
fi

if [[ ! -f "${MERGE_PY}" ]]; then
  echo "merge script not found: ${MERGE_PY}" >&2
  exit 1
fi

if [[ "${CLEAN_OUTPUT}" == "1" ]]; then
  rm -rf "${OUTPUT_ROOT}/train" "${OUTPUT_ROOT}/val" "${OUTPUT_ROOT}/test"
fi
mkdir -p "${OUTPUT_ROOT}"

echo "[radial_alpha] output_root=${OUTPUT_ROOT}"
echo "[radial_alpha] alpha_range=[${ALPHA_MIN}, ${ALPHA_MAX}]"

VECTOR_VIZ_FLAGS=(
  --viz_count "${VIZ_COUNT}"
  --viz_stride "${VIZ_STRIDE}"
  --viz_max_points "${VIZ_MAX_POINTS}"
  --viz_arrow_scale "${VIZ_ARROW_SCALE}"
)
if [[ "${SAVE_VIZ}" == "1" ]]; then
  VECTOR_VIZ_FLAGS+=(--save_viz)
fi

if [[ "${TRAIN_USE_SHARDS}" == "1" ]]; then
  conda run -n advir --no-capture-output python "${BUILDER_PY}" \
    --split train --count "${TRAIN_SHARD0_COUNT}" \
    --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
    --seed "${TRAIN_SHARD0_SEED}" --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
    --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
    --device "${TRAIN_SHARD0_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
    "${VECTOR_VIZ_FLAGS[@]}" --index_offset 0 --artifact_suffix shard0 &
  pid0=$!

  conda run -n advir --no-capture-output python "${BUILDER_PY}" \
    --split train --count "${TRAIN_SHARD1_COUNT}" \
    --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
    --seed "${TRAIN_SHARD1_SEED}" --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
    --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
    --device "${TRAIN_SHARD1_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
    "${VECTOR_VIZ_FLAGS[@]}" --index_offset 5462 --artifact_suffix shard1 &
  pid1=$!

  conda run -n advir --no-capture-output python "${BUILDER_PY}" \
    --split train --count "${TRAIN_SHARD2_COUNT}" \
    --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
    --seed "${TRAIN_SHARD2_SEED}" --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
    --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
    --device "${TRAIN_SHARD2_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
    "${VECTOR_VIZ_FLAGS[@]}" --index_offset 10923 --artifact_suffix shard2 &
  pid2=$!

  wait "${pid0}" "${pid1}" "${pid2}"

  conda run -n advir --no-capture-output python "${MERGE_PY}" \
    --split_root "${OUTPUT_ROOT}/train" \
    --manifest_glob "manifest_shard*.json" \
    --summary_glob "summary_shard*.json" \
    --output_manifest "manifest.json" \
    --output_summary "summary.json"
else
  conda run -n advir --no-capture-output python "${BUILDER_PY}" \
    --split train --count "${TRAIN_COUNT}" \
    --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
    --seed 123 --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
    --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
    --device "${TRAIN_SHARD0_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
    "${VECTOR_VIZ_FLAGS[@]}"
fi

conda run -n advir --no-capture-output python "${BUILDER_PY}" \
  --split val --count "${VAL_COUNT}" \
  --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
  --seed "${VAL_SEED}" --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
  --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
  --device "${VAL_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
  "${VECTOR_VIZ_FLAGS[@]}"

conda run -n advir --no-capture-output python "${BUILDER_PY}" \
  --split test --count "${TEST_COUNT}" \
  --splits_json "${SPLITS_JSON}" --output_root "${OUTPUT_ROOT}" \
  --seed "${TEST_SEED}" --alpha_min "${ALPHA_MIN}" --alpha_max "${ALPHA_MAX}" \
  --num_steps "${NUM_STEPS}" ${BATCHIFY_STEPS_FLAG} --mode "${MODE}" --padding_mode "${PADDING_MODE}" ${ALIGN_CORNERS_FLAG} \
  --device "${TEST_DEVICE}" --progress_interval "${PROGRESS_INTERVAL}" \
  "${VECTOR_VIZ_FLAGS[@]}"

echo "[radial_alpha] dataset generation completed: ${OUTPUT_ROOT}"
