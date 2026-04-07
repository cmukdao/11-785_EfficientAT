import os

# ============================================================================
# PERFORATED AI LICENSE CREDENTIALS
# Must be set BEFORE importing perforatedai
# ============================================================================
os.environ["PAIEMAIL"] = "PAIUser3.11.2026@perforatedai.com"
os.environ["PAITOKEN"] = "g3ZDrJNAmBlh/tAdFUkkedY+mUGqaueCPQXAtkVyNq405Fc9+20MhmEIDttx285EhfFqDHFtMRA20BWKpQaqai3wNxaOeCNvsNsF7Nn2nTmpFUmkHGvmVWGD5JI1uxdPdW6jeUmaQBKNUNcNKOumr1iDaQpFnCvFDDcSi3yUmZ5JoCd0c/lAyGmRXWQ+dInOXX6MdE4XVmv7DI8jW626pNLemX7ZMo4dEGikNuhyuiAD1IJYNYaJxUK0zaizx/Avq6QbmwQviPjXDNyBsTdVzJQzhAG6Zf2DlU9j29RnEaj0XGd4j3MJeqsrq0FeXKqEMShnBKO0oL69CP4icVTOpQ=="

# ============================================================================
# DISABLE PDB DEBUGGER
# PAI can trigger pdb breakpoints on warnings/errors — disable them entirely.
# ============================================================================
# os.environ["PYTHONBREAKPOINT"] = "0"
# import bdb
# bdb.Bdb.set_trace = lambda self, frame=None: None
# import sys
# sys.breakpointhook = lambda *args, **kwargs: None
# import pdb
# pdb.set_trace = lambda: None

import wandb
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn as nn
import torch.nn.functional as F

from datasets.esc50 import get_test_set, get_training_set
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, mixup

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import perforatedbp

def _configure_pai(args):
    """Configure all PAI global settings before model initialization."""

    # Capacity test: rapidly adds 3 dendrite sets to confirm GPU can handle the
    # enlarged network. Run once, then pass --no_dendrite_test for real training.
    GPA.pc.set_testing_dendrite_capacity(not args.no_dendrite_test)

    # Perforated Backpropagation (proprietary — requires valid credentials above).
    # When False, falls back to open-source gradient descent dendrite learning.
    if hasattr(GPA.pc, 'set_perforated_backpropagation'):
        GPA.pc.set_perforated_backpropagation(args.perforated_bp)
        print(f"  Perforated Backpropagation: {'ENABLED' if args.perforated_bp else 'DISABLED'}")

    GPA.pc.set_verbose(args.pai_verbose)
    GPA.pc.set_dendrite_update_mode(True)

    if hasattr(GPA.pc, 'set_initial_correlation_batches'):
        GPA.pc.set_initial_correlation_batches(args.initial_correlation_batches)

    # Silence noisy dimension-shape messages during forward passes
    if hasattr(GPA.pc, 'set_debugging_output_dimensions'):
        GPA.pc.set_debugging_output_dimensions(0)

    # Fix .item() errors that PB can trigger inside PAI's internal type conversions
    if hasattr(GPA.pc, 'set_using_safe_tensors'):
        GPA.pc.set_using_safe_tensors(True)

    # Silence warnings about BatchNorm/unwrapped modules and weight_decay
    if hasattr(GPA.pc, 'set_unwrapped_modules_confirmed'):
        GPA.pc.set_unwrapped_modules_confirmed(False)
    if hasattr(GPA.pc, 'set_weight_decay_accepted'):
        GPA.pc.set_weight_decay_accepted(True)

    # Switch mode: fixed epoch intervals vs adaptive history
    if args.pai_switch_mode == "fixed":
        if hasattr(GPA.pc, "DOING_FIXED_SWITCH"):
            GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_fixed_switch_num(args.pai_fixed_switch_num)
        GPA.pc.set_first_fixed_switch_num(args.pai_first_fixed_switch_num)
    else:
        if hasattr(GPA.pc, "DOING_HISTORY"):
            GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
        GPA.pc.set_n_epochs_to_switch(args.n_epochs_to_switch)
        GPA.pc.set_p_epochs_to_switch(args.p_epochs_to_switch)

    # History lookback window for plateau detection
    if hasattr(GPA.pc, 'set_history_lookback'):
        GPA.pc.set_history_lookback(args.history_lookback)

    GPA.pc.set_max_dendrites(args.max_dendrites)

    # PAI's automatic LR search after each dendrite addition — recommended
    if hasattr(GPA.pc, 'set_find_best_lr'):
        GPA.pc.set_find_best_lr(True)

    # Progressive improvement threshold (from WandB sweep best config):
    # strict → very strict → permissive across successive dendrite sets
    if hasattr(GPA.pc, 'set_improvement_threshold'):
        GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])

    # Small dendrite weight init reduces disruption to already-trained weights
    if hasattr(GPA.pc, 'set_candidate_weight_initialization_multiplier'):
        GPA.pc.set_candidate_weight_initialization_multiplier(0.005)

    # Clip dendrite candidate gradients to prevent NaN/inf in early dendrite epochs
    if hasattr(GPA.pc, 'set_candidate_grad_clipping'):
        GPA.pc.set_candidate_grad_clipping(1.0)

    if hasattr(GPA.pc, 'set_drawing_extra_graphs'):
        GPA.pc.set_drawing_extra_graphs(True)

def _register_mobilenet_pai_blocks(model):
    convnorm_type = type(model.features[0])   # Conv2dNormActivation
    invres_type = type(model.features[1])     # InvertedResidual

    GPA.pc.append_modules_to_convert([convnorm_type, invres_type])

    return model

def train(args):
    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    # Perforated Backprop + GradScaler breaks on PAI optimizers ("No inf checks…"); use fp32.
    amp_enabled = (
        args.cuda and device.type == "cuda" and not args.perforated_bp
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

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

    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(
            width_mult=width,
            pretrained_name=pretrained_name,
            pretrain_final_temp=args.pretrain_final_temp,
            num_classes=50,
        )
    else:
        model = get_mobilenet(
            width_mult=width,
            pretrained_name=pretrained_name,
            head_type=args.head_type,
            se_dims=args.se_dims,
            num_classes=50,
        )
    model = _register_mobilenet_pai_blocks(model)

    # Configure PAI globals, then initialize
    _configure_pai(args)
    pai_output_dir = os.path.join("pai", args.experiment_name)
    os.makedirs(pai_output_dir, exist_ok=True)
    os.makedirs(os.path.join(pai_output_dir, "pai"), exist_ok=True)
    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name=pai_output_dir,
        maximizing_score=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"param_count: {total_params}")
    print(f"trainable_param_count: {trainable_params}")
    for name, p in model.named_parameters():
        if not hasattr(p, "parameter_type"):
            print("missing parameter_type:", name, tuple(p.shape))
    print(UPA.find_param_name_by_id(model, 135364621016320))
    print(UPA.find_param_name_by_id(model, 135364621016480))

    dl = DataLoader(
        dataset=get_training_set(
            resample_rate=args.resample_rate,
            roll=not args.no_roll,
            wavmix=not args.no_wavmix,
            gain_augment=args.gain_augment,
            fold=args.fold,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.num_workers // 2 if args.num_workers > 0 else None,
    )

    eval_dl = DataLoader(
        dataset=get_test_set(resample_rate=args.resample_rate, fold=args.fold),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=args.num_workers // 2 if args.num_workers > 0 else None,
    )

    # PAI-managed optimizer + scheduler.
    # patience must be < n_epochs_to_switch so plateau is detected before switching.
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
    schedArgs = {'mode': 'max', 'patience': 5}
    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')
    train_accuracy = float('NaN')

    # Initialize WandB
    wandb.init(
        entity="11-785_perforated_ai",
        project="ESC50",
        group="ESC50-mn05_as-pai",
        notes="Fine-tune Models on ESC50 with Perforated AI Dendrites + Perforated BP.",
        tags=["Environmental Sound Classification", "Fine-Tuning", "Dendrites", "PerforatedBP"],
        config=args,
        name=args.experiment_name
    )


    # PAI determines convergence; use while-True instead of a fixed epoch count
    epoch = -1
    while True:
        epoch += 1
        mel.train()
        model.train()
        train_loss_list = []
        correct = 0
        total = 0

        pbar = tqdm(dl)
        pbar.set_description(
            f"Epoch {epoch + 1} | acc={accuracy:.4f} val_loss={val_loss:.4f} train_acc={train_accuracy:.4f}"
        )

        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                x = _mel_forward(x, mel)

                if args.mixup_alpha:
                    rn_indices, lam = mixup(bs, args.mixup_alpha)
                    lam = lam.to(x.device)
                    x = x * lam.reshape(bs, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                    y_hat, _ = model(x)
                    samples_loss = (
                        F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs)
                        + F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(bs))
                    )
                else:
                    y_hat, _ = model(x)
                    samples_loss = F.cross_entropy(y_hat, y, reduction="none")

                loss = samples_loss.mean()

            train_loss_list.append(loss.detach().cpu().numpy())

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            preds = y_hat.detach().argmax(dim=1)
            targets_idx = y.argmax(dim=1) if y.dim() > 1 else y
            correct += preds.eq(targets_idx).sum().item()
            total += bs

        train_accuracy = correct / total if total > 0 else float('NaN')

        # Extra score feeds PAI's graphs and best_arch_scores.csv
        GPA.pai_tracker.add_extra_score(train_accuracy * 100.0, "Train")
        model.to(device)

        accuracy, val_loss = _test(model, mel, eval_dl, device)

        # Core PAI call: may restructure model (add/incorporate dendrites) and
        # signals training_complete when no further improvement is possible
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            accuracy * 100.0, model
        )
        model.to(device)

        wandb.log({
            "train_loss": np.mean(train_loss_list),
            "train_accuracy": train_accuracy,
            "accuracy": accuracy,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if name is not None:
            try:
                os.remove(os.path.join(wandb.run.dir, name))
            except FileNotFoundError:
                pass
        name = f"{model_name}_esc50_epoch_{epoch}_acc_{int(round(accuracy * 1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))

        if training_complete:
            print(f"PAI training complete at epoch {epoch}. Best accuracy: {accuracy:.4f}")
            break
        elif restructured:
            # Model topology changed — optimizer state is stale, must reinitialize
            optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
            schedArgs = {'mode': 'max', 'patience': 5}
            optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)


def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets = []
    outputs = []
    losses = []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, f, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    accuracy = metrics.accuracy_score(targets.argmax(axis=1), outputs.argmax(axis=1))
    return accuracy, losses.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESC50 fine-tuning with Perforated AI Dendrites + Perforated BP. Reduced batch size to 64.')

    # general
    parser.add_argument('--experiment_name', type=str, default="ESC50-mn05_as-pai-fixed-switch-2")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fold', type=int, default=1)

    # model / training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=6e-5)

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

    # PAI settings (PB is default — use --no_perforated_bp for OSS dendrites only)
    parser.add_argument(
        '--no_perforated_bp',
        dest='perforated_bp',
        action='store_false',
        help='Disable Perforated Backpropagation (open-source dendrite GD only)',
    )
    parser.set_defaults(perforated_bp=True)
    parser.add_argument('--no_dendrite_test', action='store_true', default=False,
                        help='Skip dendrite capacity test (set after first successful run)')
    parser.add_argument('--pai_verbose', action='store_true', default=False,
                        help='Enable verbose PAI logging')
    parser.add_argument(
        '--initial_correlation_batches',
        type=int,
        default=8,
        help='PB initial correlation batches (GPA.pc.set_initial_correlation_batches, if available)',
    )
    parser.add_argument(
        '--pai_switch_mode',
        type=str,
        choices=('fixed', 'history'),
        default='fixed',
        help='fixed: dendrite switches on a fixed epoch schedule; history: adaptive (uses n/p_epochs_to_switch)',
    )
    parser.add_argument(
        '--pai_fixed_switch_num',
        type=int,
        default=5,
        help='With --pai_switch_mode=fixed: epochs between dendrite switches after the first',
    )
    parser.add_argument(
        '--pai_first_fixed_switch_num',
        type=int,
        default=15,
        help='With --pai_switch_mode=fixed: epochs before the first dendrite switch',
    )
    parser.add_argument('--max_dendrites', type=int, default=3,
                        help='Maximum number of dendrite sets to add')
    parser.add_argument('--n_epochs_to_switch', type=int, default=10,
                        help='Normal-learning epochs before plateau check (must be > scheduler patience)')
    parser.add_argument('--p_epochs_to_switch', type=int, default=10,
                        help='PAI-learning epochs before plateau check')
    parser.add_argument('--history_lookback', type=int, default=5,
                        help='Number of epochs to look back for plateau detection')

    args = parser.parse_args()
    train(args)
