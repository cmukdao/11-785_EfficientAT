#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 08:00:00
#SBATCH -A cis260147p
#SBATCH -o /ocean/projects/cis260147p/shared/11-785_EfficientAT/logs/slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT"
RUN_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs"
MAX_PHASES="${MAX_PHASES:-10}"

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch --gres=gpu:<gpu-type>:1 $0 <save_name> [phase] [training args...]"
  exit 2
fi

SAVE_NAME="$1"
if [ "$#" -ge 2 ]; then
  PHASE="$2"
  shift 2
else
  PHASE="0"
  shift 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RUN_ROOT"

set +u
source ~/.bashrc
conda activate efficientat
set -u

RESUME_ARGS=()
if [ "$PHASE" -gt 0 ]; then
  RESUME_ARGS=(--resume-pai --pai-resume-tag latest)
fi

set +e
python ex_fsd50k_perforatedai.py \
  --save-name "$SAVE_NAME" \
  --output-dir "$RUN_ROOT" \
  --exit-on-pai-restructure \
  --max-wall-minutes 450 \
  --sync-pai-saves \
  --phase-index "$PHASE" \
  "${RESUME_ARGS[@]}" \
  "$@"
STATUS=$?
set -e

if [ "$STATUS" -eq 88 ] || [ "$STATUS" -eq 89 ]; then
  NEXT_PHASE=$((PHASE + 1))
  if [ "$NEXT_PHASE" -ge "$MAX_PHASES" ]; then
    echo "Reached MAX_PHASES=$MAX_PHASES. Not resubmitting."
    exit 1
  fi

  echo "PAI phase $PHASE exited with $STATUS. Next phase is $NEXT_PHASE."
  if [ -n "${GPU_TYPE:-}" ]; then
    echo "Submitting next PAI phase for $SAVE_NAME with GPU_TYPE=$GPU_TYPE."
    if sbatch --gres="gpu:${GPU_TYPE}:1" --export=ALL,GPU_TYPE="$GPU_TYPE",MAX_PHASES="$MAX_PHASES" --job-name "fsd50k_${SAVE_NAME}" "$0" "$SAVE_NAME" "$NEXT_PHASE" "$@"; then
      exit 0
    fi
    echo "Automatic sbatch failed."
  else
    echo "GPU_TYPE is not set, so automatic resubmission was skipped."
  fi
  echo "Manual resume command:"
  echo "sbatch --gres=gpu:<gpu-type>:1 --export=ALL,GPU_TYPE=<gpu-type>,MAX_PHASES=$MAX_PHASES --job-name fsd50k_${SAVE_NAME} $0 $SAVE_NAME $NEXT_PHASE $*"
  exit 1
fi

if [ "$STATUS" -eq 0 ]; then
  echo "PAI run complete."
  exit 0
fi

echo "PAI run failed with status $STATUS. Not resubmitting."
exit "$STATUS"
