#!/bin/bash
#SBATCH -p GPU-shared
#SBATCH -t 08:00:00
#SBATCH -A cis260147p
#SBATCH -o /ocean/projects/cis260147p/shared/11-785_EfficientAT/logs/slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT"
RUN_ROOT="/ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs"

cd "$PROJECT_ROOT"
mkdir -p "$PROJECT_ROOT/logs"
mkdir -p "$RUN_ROOT"

source ~/.bashrc
conda activate efficientat

echo "Smoke test ran on $(date)" > "$RUN_ROOT/smoke_bridge2_fsd50k.txt"

python - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

if [ "${TEST_RECURSIVE_SBATCH:-0}" = "1" ] && [ "${SMOKE_RECURSED:-0}" = "0" ]; then
  if [ -z "${GPU_TYPE:-}" ]; then
    echo "GPU_TYPE is not set, so recursive smoke submission was skipped."
  else
    sbatch --gres="gpu:${GPU_TYPE}:1" --export=ALL,GPU_TYPE="$GPU_TYPE",SMOKE_RECURSED=1 "$0"
  fi
fi
