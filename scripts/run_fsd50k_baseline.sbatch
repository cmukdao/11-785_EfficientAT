#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 08:00:00
#SBATCH -A cis260147p
#SBATCH -o /ocean/projects/cis260147p/shared/11-785_EfficientAT/logs/slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT"
RUN_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs"

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch --gres=gpu:<gpu-type>:1 $0 <save_name> [training args...]"
  exit 2
fi

SAVE_NAME="$1"
shift

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RUN_ROOT"

source ~/.bashrc
conda activate efficientat

RESUME_PATH="$RUN_ROOT/$SAVE_NAME/checkpoints/last.pt"
RESUME_ARGS=()
if [ -f "$RESUME_PATH" ]; then
  RESUME_ARGS=(--resume "$RESUME_PATH")
fi

set +e
python ex_fsd50k.py \
  --train \
  --save-name "$SAVE_NAME" \
  --output-dir "$RUN_ROOT" \
  --max-wall-minutes 450 \
  "${RESUME_ARGS[@]}" \
  "$@"
STATUS=$?
set -e

if [ "$STATUS" -eq 89 ]; then
  echo "Baseline reached wall-time budget. Preparing resume."
  if [ -n "${GPU_TYPE:-}" ]; then
    echo "Resubmitting baseline run $SAVE_NAME with GPU_TYPE=$GPU_TYPE."
    if sbatch --gres="gpu:${GPU_TYPE}:1" --export=ALL,GPU_TYPE="$GPU_TYPE" --job-name "fsd50k_${SAVE_NAME}" "$0" "$SAVE_NAME" "$@"; then
      exit 0
    fi
    echo "Automatic sbatch failed."
  else
    echo "GPU_TYPE is not set, so automatic resubmission was skipped."
  fi
  echo "Manual resume command:"
  echo "sbatch --gres=gpu:<gpu-type>:1 --export=ALL,GPU_TYPE=<gpu-type> --job-name fsd50k_${SAVE_NAME} $0 $SAVE_NAME $*"
  exit 1
fi

if [ "$STATUS" -eq 0 ]; then
  echo "Baseline run complete."
  exit 0
fi

echo "Baseline run failed with status $STATUS. Not resubmitting."
exit "$STATUS"
