#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path("/ocean/projects/cis260147p/shared/11-785_EfficientAT")
RUN_ROOT = PROJECT_ROOT / "checkpoints" / "fsd50k_runs"
LOG_ROOT = PROJECT_ROOT / "logs"

GPU_BY_MODEL = {
    "mn04_as": "v100-16",
    "mn05_as": "v100-16",
    "mn10_as": "v100-32",
    "dymn04_as": "v100-16",
    "dymn10_as": "v100-32",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit FSD50K Bridge2 batch runs.")
    parser.add_argument("manifest", nargs="?", default="batch_fsd50k_runs.json")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize_job_name(save_name):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", save_name)
    return f"fsd50k_{safe}"[:128]


def extract_flag_value(args, flag):
    for index, value in enumerate(args):
        if value == flag:
            if index + 1 >= len(args):
                raise ValueError(f"{flag} was provided without a value")
            return args[index + 1]
        if value.startswith(flag + "="):
            return value.split("=", 1)[1]
    return None


def model_for_run(run):
    run_type = run["type"]
    run_args = run.get("args", [])
    if run_type == "baseline":
        model = extract_flag_value(run_args, "--model_name")
    elif run_type == "pai":
        model = extract_flag_value(run_args, "--model")
    else:
        raise ValueError(f"Unknown run type: {run_type!r}")
    if not model:
        raise ValueError(
            f"Run {run.get('save_name')!r} is missing the required model flag "
            f"({'--model_name' if run_type == 'baseline' else '--model'})."
        )
    if model not in GPU_BY_MODEL:
        choices = ", ".join(sorted(GPU_BY_MODEL))
        raise ValueError(f"Unmapped model {model!r}. Expected one of: {choices}")
    return model


def run_is_complete(save_name):
    return (RUN_ROOT / save_name / "complete.json").exists()


def active_job_exists(job_name):
    user = os.environ.get("USER")
    if not user:
        return False
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", user, "-n", job_name],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        print("Warning: squeue not found; duplicate active-job check was skipped.")
        return False
    if result.returncode != 0:
        print(f"Warning: squeue failed while checking {job_name}: {result.stderr.strip()}")
        return False
    return bool(result.stdout.strip())


def submit_run(run, dry_run=False):
    save_name = run["save_name"]
    run_type = run["type"]
    run_args = list(run.get("args", []))
    model = model_for_run(run)
    gpu = GPU_BY_MODEL[model]
    job_name = sanitize_job_name(save_name)

    if run_is_complete(save_name):
        print(f"Skipping complete run: {save_name}")
        return
    if active_job_exists(job_name):
        print(f"Skipping active run: {save_name} ({job_name})")
        return

    if run_type == "baseline":
        script = PROJECT_ROOT / "scripts" / "run_fsd50k_baseline.sbatch"
        cmd = [
            "sbatch",
            f"--gres=gpu:{gpu}:1",
            f"--export=ALL,GPU_TYPE={gpu}",
            "--job-name",
            job_name,
            str(script),
            save_name,
            *run_args,
        ]
    elif run_type == "pai":
        script = PROJECT_ROOT / "scripts" / "run_fsd50k_pai_phase.sbatch"
        cmd = [
            "sbatch",
            f"--gres=gpu:{gpu}:1",
            f"--export=ALL,GPU_TYPE={gpu}",
            "--job-name",
            job_name,
            str(script),
            save_name,
            "0",
            *run_args,
        ]
    else:
        raise ValueError(f"Unknown run type: {run_type!r}")

    print("Submitting:", " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with open(args.manifest) as f:
        runs = json.load(f)
    for run in runs:
        submit_run(run, dry_run=args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"submit_fsd50k_batch.py: {exc}", file=sys.stderr)
        sys.exit(1)
