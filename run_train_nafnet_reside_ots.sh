#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash /home/huhao/adv_ir/source/run_train_nafnet_reside_ots.sh [options] [-- <extra train args>]

Options:
  --nafnet_root PATH   NAFNet repository root.
                       Default: /home/huhao/adv_ir/NAFNet
  --train_opt PATH     Training YAML option file.
                       Default: <nafnet_root>/options/train/RESIDE_OTS/NAFNet-width32.yml
  --num_gpu N          GPU count to use for training.
                       Default: 3
  --launcher TYPE      Launcher: auto|none|pytorch|slurm.
                       Default: auto (none for 1 GPU, pytorch for >1 GPU)
  --master_port PORT   Port for torch.distributed.run.
                       Default: 29501
  --wandb_project STR  W&B project name.
                       Default: ${WANDB_PROJECT:-adv_ir_nafnet_reside_ots}
  --wandb_resume_id ID Optional W&B resume id.
  --run_name STR       Optional run name. Default auto-generated with timestamp.
  --wandb_offline      Disable online sync and use offline mode.
  --output_dir PATH    Directory to save launcher metadata.
                       Default: /home/huhao/adv_ir/tmp_demo/nafnet_reside_ots_train_launcher
  --dry_run            Print and save command only, do not start training.
  -h, --help           Show this help message.

Examples:
  bash /home/huhao/adv_ir/source/run_train_nafnet_reside_ots.sh
  bash /home/huhao/adv_ir/source/run_train_nafnet_reside_ots.sh --num_gpu 3 --wandb_project adv_ir_nafnet
  bash /home/huhao/adv_ir/source/run_train_nafnet_reside_ots.sh --dry_run --wandb_project adv_ir_nafnet
EOF
}

NAFNET_ROOT="/home/huhao/adv_ir/NAFNet"
TRAIN_OPT=""
NUM_GPU=3
LAUNCHER="auto"
MASTER_PORT=29501
WANDB_PROJECT="${WANDB_PROJECT:-adv_ir_nafnet_reside_ots}"
WANDB_RESUME_ID=""
RUN_NAME=""
WANDB_OFFLINE=0
OUTPUT_DIR="/home/huhao/adv_ir/tmp_demo/nafnet_reside_ots_train_launcher"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nafnet_root)
      NAFNET_ROOT="$2"
      shift 2
      ;;
    --train_opt)
      TRAIN_OPT="$2"
      shift 2
      ;;
    --num_gpu)
      NUM_GPU="$2"
      shift 2
      ;;
    --launcher)
      LAUNCHER="$2"
      shift 2
      ;;
    --master_port)
      MASTER_PORT="$2"
      shift 2
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb_resume_id)
      WANDB_RESUME_ID="$2"
      shift 2
      ;;
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --wandb_offline)
      WANDB_OFFLINE=1
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        EXTRA_ARGS+=("$1")
        shift
      done
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TRAIN_OPT" ]]; then
  TRAIN_OPT="$NAFNET_ROOT/options/train/RESIDE_OTS/NAFNet-width32.yml"
fi

if ! [[ "$NUM_GPU" =~ ^[0-9]+$ ]] || [[ "$NUM_GPU" -le 0 ]]; then
  echo "Invalid --num_gpu: $NUM_GPU"
  exit 1
fi

if ! [[ "$MASTER_PORT" =~ ^[0-9]+$ ]] || [[ "$MASTER_PORT" -le 0 ]]; then
  echo "Invalid --master_port: $MASTER_PORT"
  exit 1
fi

if [[ ! -d "$NAFNET_ROOT" ]]; then
  echo "NAFNet root not found: $NAFNET_ROOT"
  exit 1
fi

if [[ ! -f "$TRAIN_OPT" ]]; then
  echo "Train option file not found: $TRAIN_OPT"
  exit 1
fi

if [[ "$LAUNCHER" != "auto" && "$LAUNCHER" != "none" && "$LAUNCHER" != "pytorch" && "$LAUNCHER" != "slurm" ]]; then
  echo "Invalid launcher: $LAUNCHER"
  exit 1
fi

if [[ -z "$WANDB_PROJECT" ]]; then
  echo "--wandb_project must be non-empty"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
COMMAND_FILE="$OUTPUT_DIR/train_command.txt"
RUNTIME_OPT="$OUTPUT_DIR/runtime_train_opt.yml"
RUNTIME_META="$OUTPUT_DIR/runtime_meta.json"

if [[ "$LAUNCHER" == "auto" ]]; then
  if [[ "$NUM_GPU" -gt 1 ]]; then
    RESOLVED_LAUNCHER="pytorch"
  else
    RESOLVED_LAUNCHER="none"
  fi
else
  RESOLVED_LAUNCHER="$LAUNCHER"
fi

if [[ "$NUM_GPU" -gt 1 && "$RESOLVED_LAUNCHER" == "none" ]]; then
  echo "For num_gpu > 1, launcher cannot be none. Use --launcher auto or pytorch."
  exit 1
fi

if [[ "$NUM_GPU" -eq 1 && "$RESOLVED_LAUNCHER" == "pytorch" ]]; then
  echo "[Warn] num_gpu=1 with launcher=pytorch is unusual but allowed."
fi

conda run -n advir --no-capture-output python - <<'PY' "$TRAIN_OPT" "$RUNTIME_OPT" "$WANDB_PROJECT" "$WANDB_RESUME_ID" "$RUN_NAME" "$NUM_GPU" "$WANDB_OFFLINE" "$RESOLVED_LAUNCHER"
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

train_opt = Path(sys.argv[1])
runtime_opt = Path(sys.argv[2])
wandb_project = sys.argv[3]
wandb_resume_id = sys.argv[4]
run_name = sys.argv[5]
num_gpu = int(sys.argv[6])
wandb_offline = int(sys.argv[7])
resolved_launcher = sys.argv[8]

with train_opt.open('r', encoding='utf-8') as f:
    opt = yaml.safe_load(f)

if not isinstance(opt, dict):
    raise RuntimeError('Invalid yaml content: root is not dict')

default_name = str(opt.get('name', 'NAFNet-RESIDE-OTS-width32'))
if not run_name:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{default_name}-g{num_gpu}-{timestamp}"

opt['name'] = run_name
opt['num_gpu'] = int(num_gpu)

logger = opt.setdefault('logger', {})
if not isinstance(logger, dict):
    raise RuntimeError('Invalid yaml content: logger is not dict')

logger['use_tb_logger'] = True
wandb_cfg = logger.setdefault('wandb', {})
if wandb_cfg is None or not isinstance(wandb_cfg, dict):
    wandb_cfg = {}
    logger['wandb'] = wandb_cfg

wandb_cfg['project'] = wandb_project
wandb_cfg['resume_id'] = wandb_resume_id if wandb_resume_id else None

runtime_opt.parent.mkdir(parents=True, exist_ok=True)
with runtime_opt.open('w', encoding='utf-8') as f:
    yaml.safe_dump(opt, f, sort_keys=False)

meta = {
    'train_opt_source': str(train_opt),
    'train_opt_runtime': str(runtime_opt),
    'resolved_launcher': resolved_launcher,
    'num_gpu': int(num_gpu),
    'run_name': run_name,
    'wandb_project': wandb_project,
    'wandb_resume_id': wandb_cfg['resume_id'],
    'wandb_mode': 'offline' if wandb_offline == 1 else 'online',
}
print(json.dumps(meta, indent=2))
PY

conda run -n advir --no-capture-output python - <<'PY' "$RUNTIME_META" "$TRAIN_OPT" "$RUNTIME_OPT" "$RESOLVED_LAUNCHER" "$NUM_GPU" "$WANDB_PROJECT" "$WANDB_RESUME_ID" "$RUN_NAME" "$WANDB_OFFLINE"
import json
import sys
from datetime import datetime
from pathlib import Path

runtime_meta = Path(sys.argv[1])
train_opt = sys.argv[2]
runtime_opt = sys.argv[3]
resolved_launcher = sys.argv[4]
num_gpu = int(sys.argv[5])
wandb_project = sys.argv[6]
wandb_resume_id = sys.argv[7]
run_name = sys.argv[8]
wandb_offline = int(sys.argv[9])

meta = {
    'timestamp': datetime.now().isoformat(),
    'train_opt_source': train_opt,
    'train_opt_runtime': runtime_opt,
    'resolved_launcher': resolved_launcher,
    'num_gpu': num_gpu,
    'wandb_project': wandb_project,
    'wandb_resume_id': wandb_resume_id if wandb_resume_id else None,
    'run_name_arg': run_name if run_name else None,
    'wandb_mode': 'offline' if wandb_offline == 1 else 'online',
}
runtime_meta.write_text(json.dumps(meta, indent=2), encoding='utf-8')
PY

if [[ "$RESOLVED_LAUNCHER" == "pytorch" ]]; then
  TRAIN_CMD=(
    conda run -n advir --no-capture-output
    python -m torch.distributed.run
    --nproc_per_node "$NUM_GPU"
    --master_port "$MASTER_PORT"
    basicsr/train.py
    -opt "$RUNTIME_OPT"
    --launcher pytorch
  )
elif [[ "$RESOLVED_LAUNCHER" == "none" ]]; then
  TRAIN_CMD=(
    conda run -n advir --no-capture-output
    python basicsr/train.py
    -opt "$RUNTIME_OPT"
    --launcher none
  )
else
  TRAIN_CMD=(
    conda run -n advir --no-capture-output
    python basicsr/train.py
    -opt "$RUNTIME_OPT"
    --launcher "$RESOLVED_LAUNCHER"
  )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

CMD_STR="cd $NAFNET_ROOT && $(printf '%q ' "${TRAIN_CMD[@]}")"
CMD_STR="${CMD_STR% }"
printf '%s\n' "$CMD_STR" > "$COMMAND_FILE"

if [[ "$WANDB_OFFLINE" -eq 1 ]]; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=online
fi
export WANDB_CONSOLE=wrap

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[DryRun] Command saved to $COMMAND_FILE"
  echo "[DryRun] Runtime opt saved to $RUNTIME_OPT"
  echo "[DryRun] Runtime meta saved to $RUNTIME_META"
  echo "[DryRun] W&B mode: $WANDB_MODE"
  echo "[DryRun] $CMD_STR"
  exit 0
fi

echo "[Run] Starting NAFNet RESIDE-OTS training"
echo "[Run] Working directory: $NAFNET_ROOT"
echo "[Run] Launcher: $RESOLVED_LAUNCHER"
echo "[Run] num_gpu: $NUM_GPU"
echo "[Run] W&B project: $WANDB_PROJECT"
echo "[Run] W&B mode: $WANDB_MODE"
echo "[Run] Runtime opt: $RUNTIME_OPT"
echo "[Run] Command: $CMD_STR"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[Warn] WANDB_API_KEY is not set in environment. If wandb login is not cached, upload may fail."
fi

pushd "$NAFNET_ROOT" >/dev/null
"${TRAIN_CMD[@]}"
popd >/dev/null

echo "[Done] Training command finished"
echo "[Saved] $COMMAND_FILE"
echo "[Saved] $RUNTIME_OPT"
echo "[Saved] $RUNTIME_META"
