# FSD50K Bridge2 Baseline and PerforatedAI Batches

This is the runbook for launching FSD50K baseline batches and FSD50K
PerforatedAI phase batches on PSC Bridge2.

## One-Time Login-Node Setup

Run this before the first `sbatch`. Slurm may fail to open `#SBATCH -o` if the
log directory does not already exist before the script body starts.

```bash
mkdir -p /ocean/projects/cis260147p/shared/11-785_EfficientAT/logs
mkdir -p /ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs
```

Then work from the Bridge2 project root:

```bash
cd /ocean/projects/cis260147p/shared/11-785_EfficientAT
```

## Scripts

- `scripts/smoke_bridge2_fsd50k.sbatch`: verifies Slurm, conda, CUDA, and run-root writes.
- `scripts/run_fsd50k_baseline.sbatch`: runs `ex_fsd50k.py`.
- `scripts/run_fsd50k_pai_phase.sbatch`: runs `ex_fsd50k_perforatedai.py` by PAI phase.
- `scripts/submit_fsd50k_batch.py`: submits a JSON batch manifest and chooses GPU type.

The sbatch scripts intentionally do not contain `#SBATCH --gres`. GPU requests
are passed at submit time.

## Smoke Test

Submit this first:

```bash
sbatch --gres=gpu:v100-16:1 scripts/smoke_bridge2_fsd50k.sbatch
```

Check the Slurm log under:

```text
/ocean/projects/cis260147p/shared/11-785_EfficientAT/logs/
```

Expected:

- `efficientat` conda env activates.
- Python path is printed.
- CUDA is visible.
- A test file appears in the run root.

## Baseline Batch Manifest

Use `"type": "baseline"` for runs that should call `ex_fsd50k.py`.
Each baseline entry must include `--model_name`.

Example `batch_fsd50k_baselines.json`:

```json
[
  {
    "save_name": "FSD50K-mn04-baseline-s0",
    "type": "baseline",
    "args": [
      "--cuda",
      "--pretrained",
      "--model_name", "mn04_as",
      "--seed", "0"
    ]
  },
  {
    "save_name": "FSD50K-mn10-baseline-s0",
    "type": "baseline",
    "args": [
      "--cuda",
      "--pretrained",
      "--model_name", "mn10_as",
      "--seed", "0"
    ]
  }
]
```

Submit the baseline batch:

```bash
python scripts/submit_fsd50k_batch.py batch_fsd50k_baselines.json --dry-run
python scripts/submit_fsd50k_batch.py batch_fsd50k_baselines.json
```

Baseline behavior:

- The wrapper calls `ex_fsd50k.py --train`.
- Checkpoints are normal PyTorch checkpoints.
- If `$RUN_ROOT/<save_name>/checkpoints/last.pt` exists, the wrapper resumes from it.
- Baseline jobs auto-resubmit only on exit code `89`.
- Crashes do not auto-resubmit.

## PerforatedAI Batch Manifest

Use `"type": "pai"` for runs that should call `ex_fsd50k_perforatedai.py`.
Each PAI entry must include `--model`.

Example `batch_fsd50k_pai.json`:

```json
[
  {
    "save_name": "FSD50K-mn04-pai-probe-late-s0",
    "type": "pai",
    "args": [
      "--model", "mn04_as",
      "--preset", "probe_late_backbone_head"
    ]
  },
  {
    "save_name": "FSD50K-mn05-pai-head-late-se-s0",
    "type": "pai",
    "args": [
      "--model", "mn05_as",
      "--preset", "classifier_head_plus_late_se_linears"
    ]
  }
]
```

Submit the PerforatedAI batch:

```bash
python scripts/submit_fsd50k_batch.py batch_fsd50k_pai.json --dry-run
python scripts/submit_fsd50k_batch.py batch_fsd50k_pai.json
```

PerforatedAI behavior:

- The wrapper calls `ex_fsd50k_perforatedai.py`.
- Phase `0` starts from scratch.
- Phases greater than `0` add `--resume-pai --pai-resume-tag latest`.
- PAI system saves live under `$RUN_ROOT/<save_name>/pai/system/`.
- Fallback `last.pt` and `best.pt` are for inspection only.
- PAI jobs auto-resubmit on exit code `88` or `89`, while under `MAX_PHASES`.
- Crashes do not auto-resubmit.

## Mixed Batch Manifest

You can also mix baseline and PAI entries in one manifest:

```json
[
  {
    "save_name": "FSD50K-mn04-baseline-s0",
    "type": "baseline",
    "args": [
      "--cuda",
      "--pretrained",
      "--model_name", "mn04_as",
      "--seed", "0"
    ]
  },
  {
    "save_name": "FSD50K-mn04-pai-probe-late-s0",
    "type": "pai",
    "args": [
      "--model", "mn04_as",
      "--preset", "probe_late_backbone_head"
    ]
  }
]
```

Baseline runs must include `--model_name`. PAI runs must include `--model`.

Submit the mixed batch:

```bash
python scripts/submit_fsd50k_batch.py batch_fsd50k_runs.json --dry-run
python scripts/submit_fsd50k_batch.py batch_fsd50k_runs.json
```

## Batch Submission Rules

The submitter skips runs with:

```text
/ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs/<save_name>/complete.json
```

It also checks `squeue` to avoid duplicate active jobs with the same generated
job name.

## GPU Mapping

`submit_fsd50k_batch.py` uses this exact model mapping:

```text
mn04_as   -> v100-16
mn05_as   -> v100-16
mn10_as   -> v100-32
dymn04_as -> v100-16
dymn10_as -> v100-32
```

Unmapped or missing model names fail before submission.

## Direct Manual Submission

Baseline:

```bash
sbatch \
  --gres=gpu:v100-16:1 \
  --export=ALL,GPU_TYPE=v100-16 \
  scripts/run_fsd50k_baseline.sbatch \
  FSD50K-mn04-baseline-s0 \
  --cuda --pretrained --model_name mn04_as --seed 0
```

PAI phase 0:

```bash
sbatch \
  --gres=gpu:v100-16:1 \
  --export=ALL,GPU_TYPE=v100-16 \
  scripts/run_fsd50k_pai_phase.sbatch \
  FSD50K-mn04-pai-probe-late-s0 \
  0 \
  --model mn04_as --preset probe_late_backbone_head
```

Set `MAX_PHASES` if you want a non-default cap:

```bash
sbatch \
  --gres=gpu:v100-16:1 \
  --export=ALL,GPU_TYPE=v100-16,MAX_PHASES=6 \
  scripts/run_fsd50k_pai_phase.sbatch \
  FSD50K-mn04-pai-probe-late-s0 \
  0 \
  --model mn04_as --preset probe_late_backbone_head
```

## Outputs

All canonical outputs go under:

```text
/ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs/<save_name>/
```

Baseline:

```text
checkpoints/last.pt
checkpoints/best.pt
metrics/metrics.csv
config/args.json
complete.json
```

PAI:

```text
checkpoints/last.pt          # fallback only
checkpoints/best.pt          # fallback only
pai/system/                  # canonical PAI system folder
metrics/metrics.csv
config/config.json
resume_next_command.txt
resume_state.json
complete.json
```

## Resume Behavior

Baseline resumes from normal PyTorch checkpoints:

```bash
--resume "$RUN_ROOT/<save_name>/checkpoints/last.pt"
```

The baseline sbatch wrapper adds this automatically when `last.pt` exists.
Baseline jobs auto-resubmit only on exit code `89`.

PAI resumes with PAI loading, not normal `last.pt`:

```bash
--resume-pai --pai-resume-tag latest
```

The PAI wrapper adds this automatically for phases greater than `0`. PAI jobs
auto-resubmit on exit code `88` or `89`, while `PHASE + 1 < MAX_PHASES`.

Exit codes:

```text
0  = complete
88 = PAI restructured, submit next phase
89 = wall-time budget reached, resume later
other = failure, do not auto-resubmit
```

Python writes the source-of-truth resume files for exit `88` and `89`:

```text
resume_next_command.txt
resume_state.json
```

If recursive `sbatch` fails, inspect `resume_next_command.txt` and the Slurm
log for the exact recovery state.
