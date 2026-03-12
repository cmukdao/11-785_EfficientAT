import os
from datetime import datetime
from types import SimpleNamespace


class OfflineWandb:
    def __init__(self, import_error):
        self.import_error = import_error
        self.run = None
        self._did_print_notice = False

    def init(self, project=None, name=None, dir="wandb", **kwargs):
        os.makedirs(dir, exist_ok=True)
        run_name = name or project or "offline-run"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.abspath(os.path.join(dir, f"{run_name}_{timestamp}"))
        suffix = 1
        while os.path.exists(run_dir):
            run_dir = os.path.abspath(os.path.join(dir, f"{run_name}_{timestamp}_{suffix}"))
            suffix += 1
        os.makedirs(run_dir, exist_ok=True)
        self.run = SimpleNamespace(dir=run_dir, name=run_name, config=kwargs.get("config"))
        if not self._did_print_notice:
            print(
                f"wandb unavailable ({self.import_error}). "
                f"Falling back to local logging only; checkpoints will be saved to {run_dir}."
            )
            self._did_print_notice = True
        return self.run

    def log(self, *_args, **_kwargs):
        return None

    def finish(self, *_args, **_kwargs):
        return None


def get_wandb():
    try:
        import wandb
        return wandb
    except Exception as exc:
        return OfflineWandb(exc)
