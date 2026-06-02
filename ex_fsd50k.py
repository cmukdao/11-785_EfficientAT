import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.fsd50k import get_eval_set, get_valid_set, get_training_set
from helpers.init import make_generator, seed_everything, worker_init_fn
from helpers.run_lifecycle import (
    EXIT_WALL_TIME,
    append_metrics_row,
    create_run_dirs,
    load_training_checkpoint,
    python_resume_command,
    save_args_config,
    save_training_checkpoint,
    wall_time_exceeded,
    write_complete_file,
    write_resume_files,
)
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from models.dymn.model import get_model as get_dymn
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT


NUM_CLASSES = 200
TRAIN_GENERATOR_OFFSET = 0
VAL_GENERATOR_OFFSET = 1
EVAL_GENERATOR_OFFSET = 2


def build_mel(args, device):
    mel = AugmentMelSTFT(
        n_mels=args.n_mels,
        sr=args.resample_rate,
        win_length=args.window_size,
        hopsize=args.hop_size,
        n_fft=args.n_fft,
        freqm=args.freqm,
        timem=args.timem,
        fmin=args.fmin,
        fmax=args.fmax,
        fmin_aug_range=args.fmin_aug_range,
        fmax_aug_range=args.fmax_aug_range,
    )
    mel.to(device)
    return mel


def build_model(args, device, pretrained_name=None):
    model_name = args.model_name
    if pretrained_name is None and args.pretrained:
        pretrained_name = model_name
    width = NAME_TO_WIDTH(model_name) if model_name and pretrained_name else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(
            width_mult=width,
            pretrained_name=pretrained_name,
            pretrain_final_temp=args.pretrain_final_temp,
            num_classes=NUM_CLASSES,
        )
    else:
        model = get_mobilenet(
            width_mult=width,
            pretrained_name=pretrained_name,
            head_type=args.head_type,
            se_dims=args.se_dims,
            num_classes=NUM_CLASSES,
        )
    model.to(device)
    return model, width


def build_train_loader(args):
    return DataLoader(
        dataset=get_training_set(
            resample_rate=args.resample_rate,
            roll=False if args.no_roll else True,
            wavmix=False if args.no_wavmix else True,
            gain_augment=args.gain_augment,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        generator=make_generator(args.seed, TRAIN_GENERATOR_OFFSET),
    )


def build_val_loader(args):
    return DataLoader(
        dataset=get_valid_set(
            resample_rate=args.resample_rate,
            variable_eval=args.variable_eval_length,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=1 if args.variable_eval_length else args.batch_size,
        generator=make_generator(args.seed, VAL_GENERATOR_OFFSET),
    )


def build_eval_loader(args):
    return DataLoader(
        dataset=get_eval_set(
            resample_rate=args.resample_rate,
            variable_eval=args.variable_eval_length,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=1 if args.variable_eval_length else args.batch_size,
        generator=make_generator(args.seed, EVAL_GENERATOR_OFFSET),
    )


def log_epoch_metrics(train_loss, train_mAP, train_ROC, lr_now, mAP, ROC, val_loss):
    wandb.log({
        "train_loss": train_loss,
        "train_mAP": train_mAP,
        "train_ROC": train_ROC,
        "learning_rate": lr_now,
        "mAP": mAP,
        "ROC": ROC,
        "val_loss": val_loss,
    })


def save_checkpoint_if_best(model, improved):
    latest_path = os.path.join(wandb.run.dir, "latest.pt")
    torch.save(model.state_dict(), latest_path)
    if improved:
        best_ckpt = os.path.join(wandb.run.dir, "best_val_mAP.pt")
        torch.save(model.state_dict(), best_ckpt)


def train(args):
    seed_everything(args.seed)

    if not args.save_name:
        args.save_name = args.experiment_name
    run_start_time = time.time()
    run_paths = create_run_dirs(
        args.save_name,
        args.output_dir,
        allow_overwrite=args.allow_overwrite,
        resume=bool(args.resume),
    )
    save_args_config(args, run_paths)

    wandb.init(
        entity="11-785_perforated_ai",
        project="FSD50K",
        group=f"FSD50K-{args.model_name}",
        notes="Fine-tune Models on FSD50K.",
        tags=["FSDK50K", "Audio Tagging"],
        config=args,
        name=args.experiment_name
    )

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    mel = build_mel(args, device)
    model, _ = build_model(args, device)
    dl = build_train_loader(args)
    valid_dl = build_val_loader(args)

    # optimizer & scheduler (aligned with ex_fsd50k_perforatedai.py / OptimConfig)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    lr_lambda = None
    scheduler = None
    if args.scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=args.scheduler_mode,
            patience=args.scheduler_patience,
            factor=args.scheduler_factor,
            cooldown=args.scheduler_cooldown,
            min_lr=args.scheduler_min_lr,
        )
    elif args.scheduler_name == "lambda_warmup":
        lr_lambda = exp_warmup_linear_down(
            args.warm_up_len,
            args.ramp_down_len,
            args.ramp_down_start,
            args.last_lr_value,
        )
    else:
        raise ValueError(
            f"Unknown --scheduler_name={args.scheduler_name!r}; "
            "expected 'plateau' or 'lambda_warmup'."
        )

    mAP, ROC, val_loss = float('NaN'), float('NaN'), float('NaN')

    # Early stopping on validation mAP (after min_epochs).
    best_mAP = float('-inf')
    best_epoch = -1
    epochs_since_improvement = 0

    max_epochs = args.n_epochs if args.n_epochs and args.n_epochs > 0 else int(1e9)
    schedule_epoch = 0  # drives lambda_warmup when lambda_warmup_per_cycle is True

    start_epoch = 0
    if args.resume:
        checkpoint = load_training_checkpoint(
            model,
            optimizer,
            scheduler,
            args.resume,
            device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        best_mAP = checkpoint.get("best_metric", best_mAP)
        best_epoch = checkpoint.get("best_epoch", best_epoch)
        schedule_epoch = checkpoint.get("schedule_epoch", schedule_epoch)
        extra_state = checkpoint.get("extra_state", {})
        epochs_since_improvement = extra_state.get(
            "epochs_since_improvement",
            epochs_since_improvement,
        )
        print(f"Resumed baseline checkpoint from {args.resume} at epoch {start_epoch}.")

    epoch = start_epoch
    while epoch < max_epochs:
        # Manual LR for lambda_warmup (epoch-start, same order as perforated script).
        if lr_lambda is not None:
            sched_idx = schedule_epoch if args.lambda_warmup_per_cycle else epoch
            scale = float(lr_lambda(sched_idx))
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * scale
            schedule_epoch += 1

        mel.train()
        model.train()
        train_stats = dict(train_loss=list())
        train_targets, train_outputs = [], []
        pbar = tqdm(dl)
        pbar.set_description(
            "Epoch {} (best val mAP: {:.4f} @ ep {}, no-improve: {}/{}; min_ep {})"
            .format(
                epoch + 1,
                best_mAP if best_mAP > float("-inf") else 0.0,
                best_epoch + 1,
                epochs_since_improvement,
                args.patience,
                args.min_epochs,
            )
        )
        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x, y = x.to(device), y.to(device)
            x = _mel_forward(x, mel)

            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                y_mix = y * lam.reshape(bs, 1) + y[rn_indices] * (1. - lam.reshape(bs, 1))
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y_mix, reduction="none")
            else:
                y_hat, _ = model(x)
                samples_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")

            loss = samples_loss.mean()

            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # This is a noisy in-epoch fit proxy because weights change between batches.
            train_targets.append(y.detach().cpu().numpy())
            train_outputs.append(y_hat.detach().float().cpu().numpy())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_mAP, train_ROC = _score(train_targets, train_outputs)
        mAP, ROC, val_loss = _test(model, mel, valid_dl, device)

        if scheduler is not None:
            scheduler.step(mAP)

        lr_now = optimizer.param_groups[0]["lr"]

        # Early-stopping bookkeeping based on validation mAP.
        improved = mAP > best_mAP + args.min_delta
        if improved:
            best_mAP = mAP
            best_epoch = epoch
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        log_epoch_metrics(
            np.mean(train_stats['train_loss']),
            train_mAP,
            train_ROC,
            lr_now,
            mAP,
            ROC,
            val_loss,
        )
        save_checkpoint_if_best(model, improved)
        extra_state = {
            "epochs_since_improvement": epochs_since_improvement,
        }
        save_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            best_mAP,
            best_epoch,
            schedule_epoch,
            extra_state,
            args,
            run_paths,
            "last.pt",
            is_perforated=False,
        )
        if improved:
            save_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                best_mAP,
                best_epoch,
                schedule_epoch,
                extra_state,
                args,
                run_paths,
                "best.pt",
                is_perforated=False,
            )
        append_metrics_row(
            run_paths,
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_stats["train_loss"])),
                "train_mAP": train_mAP,
                "train_ROC": train_ROC,
                "learning_rate": lr_now,
                "mAP": mAP,
                "ROC": ROC,
                "val_loss": val_loss,
                "best_mAP": best_mAP,
                "best_epoch": best_epoch,
            },
        )

        if wall_time_exceeded(args.max_wall_minutes, run_start_time):
            resume_command = python_resume_command(
                args,
                {
                    "save_name": args.save_name,
                    "output_dir": args.output_dir,
                    "resume": str(run_paths["checkpoints"] / "last.pt"),
                },
            )
            write_resume_files(
                run_paths,
                args.save_name,
                "wall_time_budget",
                EXIT_WALL_TIME,
                resume_command,
                is_perforated=False,
            )
            print("Exiting after wall-time budget.")
            sys.exit(EXIT_WALL_TIME)

        # Early-stopping: only after min_epochs; metric = validation mAP.
        if (
            (epoch + 1) >= args.min_epochs
            and epochs_since_improvement >= args.patience
        ):
            print(
                f"Early stopping at epoch {epoch + 1}: validation mAP has not "
                f"improved for {args.patience} epochs (min_epochs={args.min_epochs}; "
                f"best val mAP {best_mAP:.4f} at epoch {best_epoch + 1})."
            )
            break

        epoch += 1

    wandb.run.summary["best_val_mAP"] = best_mAP
    wandb.run.summary["best_mAP"] = best_mAP  # alias for older dashboards
    wandb.run.summary["best_epoch"] = best_epoch + 1
    # After a normal max-epochs finish, ``epoch`` has been incremented to
    # ``max_epochs``; after early stopping it is the last completed 0-based index.
    wandb.run.summary["stopped_epoch"] = (
        epoch if epoch >= max_epochs else epoch + 1
    )
    write_complete_file(
        run_paths,
        args.save_name,
        best_mAP,
        best_epoch,
        is_perforated=False,
    )


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _score(targets_list, outputs_list):
    """Macro mAP and ROC-AUC from accumulated per-batch target / logit arrays.

    Shared by training and validation. Two boundary fixes:

    1. Binarize targets with `>= 0.5`. When `wavmix=True` (default), the
       training set is wrapped in `MixupDataset`, which returns
       DATASET-LEVEL blended targets like `y1 * l + y2 * (1 - l)` with
       `l in [0.5, 1]` -- i.e. continuous floats. sklearn's
       `average_precision_score` rejects those as
       "continuous-multioutput". Thresholding at 0.5 recovers the
       dominant sample's label vector, the natural binary ground truth.
       No-op on validation's already-binary `{0.0, 1.0}` targets.
    2. `nan_to_num` on outputs guards against any non-finite logits from
       numerical instability early in training.
    """
    targets = (np.concatenate(targets_list) >= 0.5).astype(np.int8)
    outputs = np.nan_to_num(np.concatenate(outputs_list))
    mAP = metrics.average_precision_score(targets, outputs, average=None).mean()
    ROC = metrics.roc_auc_score(targets, outputs, average=None).mean()
    return float(mAP), float(ROC)


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets, outputs, losses = [], [], []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.binary_cross_entropy_with_logits(y_hat, y).cpu().numpy())

    mAP, ROC = _score(targets, outputs)
    return mAP, ROC, float(np.stack(losses).mean())


def evaluate(args):
    seed_everything(args.seed)

    model_name = args.model_name
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    model, _ = build_model(args, device, pretrained_name=model_name)
    model.eval()

    mel = build_mel(args, device)
    mel.eval()

    dl = build_eval_loader(args)

    print(f"Running FSD50K evaluation for model '{model_name}' on device '{device}'")
    targets = []
    outputs = []
    for batch in tqdm(dl):
        x, _, y = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)

    print(f"Results on FSD50K evaluation split for loaded model: {model_name}")
    print("  mAP: {:.3f}".format(mAP.mean()))
    print("  ROC: {:.3f}".format(ROC.mean()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')

    # general
    parser.add_argument('--experiment_name', type=str, default="FSD50K-mn05_as-baseline")
    parser.add_argument('--save-name', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default="runs")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-wall-minutes', type=float, default=None)
    parser.add_argument('--allow-overwrite', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)

    # validation & evaluation
    # if true, requires setting validation and evaluation batch size to 1
    parser.add_argument('--variable_eval_length', action='store_true', default=False)

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn05_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=200,
                        help="Hard upper bound on training epochs (baseline default 200). "
                             "Set to 0 or negative to train until early stopping only.")
    parser.add_argument('--min_epochs', type=int, default=25,
                        help="Minimum epochs before early stopping can trigger (baseline 25).")
    parser.add_argument('--patience', type=int, default=25,
                        help="Early stopping patience on validation mAP: stop after this "
                             "many consecutive epochs without improvement (after min_epochs).")
    parser.add_argument('--min_delta', type=float, default=0,
                        help="Minimum mAP improvement to reset the early-stopping counter.")
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=7e-5)
    parser.add_argument(
        '--scheduler_name',
        type=str,
        default='plateau',
        choices=('plateau', 'lambda_warmup'),
        help="plateau: ReduceLROnPlateau stepped on val mAP; "
        "lambda_warmup: exp warmup + linear rampdown applied each epoch.",
    )
    parser.add_argument('--scheduler_mode', type=str, default='max')
    parser.add_argument('--scheduler_patience', type=int, default=5)
    parser.add_argument('--scheduler_factor', type=float, default=0.5)
    parser.add_argument('--scheduler_cooldown', type=int, default=3)
    parser.add_argument('--scheduler_min_lr', type=float, default=1e-7)
    parser.add_argument(
        '--lambda_warmup_per_cycle',
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For lambda_warmup: use a per-run schedule counter (same as single PAI cycle).",
    )
    parser.add_argument('--warm_up_len', type=int, default=3)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=25)
    parser.add_argument('--last_lr_value', type=float, default=0.01)

    # preprocessing
    parser.add_argument('--resample_rate', type=int, default=32000)
    parser.add_argument('--window_size', type=int, default=800)
    parser.add_argument('--hop_size', type=int, default=320)
    parser.add_argument('--n_fft', type=int, default=1024)
    parser.add_argument('--n_mels', type=int, default=128)
    parser.add_argument('--freqm', type=int, default=0)
    parser.add_argument('--timem', type=int, default=0)
    parser.add_argument('--fmin', type=int, default=0)
    parser.add_argument('--fmax', type=int, default=None)
    parser.add_argument('--fmin_aug_range', type=int, default=10)
    parser.add_argument('--fmax_aug_range', type=int, default=2000)

    args = parser.parse_args()
    if args.train:
        train(args)
    else:
        evaluate(args)
