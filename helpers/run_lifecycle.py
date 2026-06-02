import csv
import json
import shlex
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import torch


EXIT_PAI_RESTRUCTURED = 88
EXIT_WALL_TIME = 89


def _json_default(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    return str(value)


def create_run_dirs(
    save_name,
    output_dir,
    allow_overwrite=False,
    resume=False,
    resume_pai=False,
):
    run_dir = Path(output_dir) / save_name
    complete_file = run_dir / "complete.json"
    if complete_file.exists() and not allow_overwrite:
        raise RuntimeError(
            f"Run is already complete: {run_dir}. "
            "Use a new --save-name or pass --allow-overwrite."
        )
    if run_dir.exists() and not (allow_overwrite or resume or resume_pai):
        has_contents = any(run_dir.iterdir())
        if has_contents:
            raise RuntimeError(
                f"Run directory already exists: {run_dir}. "
                "Use a new --save-name or pass --resume/--resume-pai."
            )

    paths = {
        "run": run_dir,
        "checkpoints": run_dir / "checkpoints",
        "pai": run_dir / "pai",
        "pai_system": run_dir / "pai" / "system",
        "logs": run_dir / "logs",
        "config": run_dir / "config",
        "metrics": run_dir / "metrics",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def save_args_config(args, run_paths):
    save_json(vars(args), run_paths["config"] / "args.json")
    with open(run_paths["config"] / "command.txt", "w") as f:
        f.write(" ".join(shlex.quote(part) for part in sys.argv) + "\n")


def save_config(config, run_paths):
    payload = asdict(config) if is_dataclass(config) else config
    save_json(payload, run_paths["config"] / "config.json")
    with open(run_paths["config"] / "command.txt", "w") as f:
        f.write(" ".join(shlex.quote(part) for part in sys.argv) + "\n")


def save_json(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def append_metrics_row(run_paths, row):
    metrics_path = run_paths["metrics"] / "metrics.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    write_header = not metrics_path.exists()
    with open(metrics_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def wall_time_exceeded(max_wall_minutes, run_start_time):
    if max_wall_minutes is None:
        return False
    elapsed_minutes = (time.time() - run_start_time) / 60.0
    return elapsed_minutes >= max_wall_minutes


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_metric,
    best_epoch,
    schedule_epoch,
    extra_state,
    args_or_config,
    run_paths,
    filename,
    is_perforated=False,
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "schedule_epoch": schedule_epoch,
        "extra_state": extra_state or {},
        "args": vars(args_or_config)
        if hasattr(args_or_config, "__dict__") and not is_dataclass(args_or_config)
        else asdict(args_or_config)
        if is_dataclass(args_or_config)
        else args_or_config,
        "is_perforated": is_perforated,
    }
    if is_perforated:
        checkpoint["note"] = (
            "Fallback checkpoint only. Full PAI resume should use "
            "--resume-pai / UPA.load_system(...)."
        )
    torch.save(checkpoint, run_paths["checkpoints"] / filename)


def load_training_checkpoint(model, optimizer, scheduler, resume_path, device):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def write_resume_files(
    run_paths,
    save_name,
    reason,
    exit_code,
    resume_command,
    is_perforated,
    phase_index=0,
):
    with open(run_paths["run"] / "resume_next_command.txt", "w") as f:
        f.write(resume_command + "\n")
    save_json(
        {
            "save_name": save_name,
            "reason": reason,
            "exit_code": exit_code,
            "resume_command": resume_command,
            "is_perforated": is_perforated,
            "phase_index": phase_index,
            "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        run_paths["run"] / "resume_state.json",
    )


def write_complete_file(
    run_paths,
    save_name,
    best_metric,
    best_epoch,
    is_perforated=False,
    phase_index=0,
):
    save_json(
        {
            "save_name": save_name,
            "status": "complete",
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "is_perforated": is_perforated,
            "phase_index": phase_index,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        run_paths["run"] / "complete.json",
    )


def python_resume_command(args, updates):
    parts = [sys.executable or "python", sys.argv[0]]
    skip_next = False
    update_flags = {}
    for key, value in updates.items():
        update_flags[f"--{key.replace('_', '-')}"] = value
        update_flags[f"--{key}"] = value
    update_keys = set(update_flags)
    for index, item in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if item in update_keys:
            if update_flags[item] is not True and index + 2 <= len(sys.argv[1:]):
                skip_next = True
            continue
        if any(item.startswith(key + "=") for key in update_keys):
            continue
        parts.append(item)
    for key, value in updates.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            parts.append(flag)
        elif value is False or value is None:
            continue
        else:
            parts.extend([flag, str(value)])
    return " ".join(shlex.quote(part) for part in parts)
