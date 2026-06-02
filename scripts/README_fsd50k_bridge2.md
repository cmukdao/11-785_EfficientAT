# FSD50K Bridge2 Batch Runs

Before the first `sbatch`, create the persistent log and run directories on the
Bridge2 login node:

```bash
mkdir -p /ocean/projects/cis260147p/shared/11-785_EfficientAT/logs
mkdir -p /ocean/projects/cis260147p/shared/11-785_EfficientAT/checkpoints/fsd50k_runs
```

Run the smoke test with an explicit GPU request:

```bash
sbatch --gres=gpu:v100-16:1 scripts/smoke_bridge2_fsd50k.sbatch
```

Submit a batch manifest:

```bash
python scripts/submit_fsd50k_batch.py batch_fsd50k_runs.json
```

The generic sbatch scripts intentionally do not contain `#SBATCH --gres`.
`submit_fsd50k_batch.py` maps model names to `v100-16` or `v100-32` and passes
the GPU request at submit time.
