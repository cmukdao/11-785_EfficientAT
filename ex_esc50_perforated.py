import sys
import os
'''
USAGE: Pick ONE of the two training options, then eval.

[Option A] Dendrite search with regular backprop:
EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --model_name=mn10_as --fold=1 \
    --finetune_checkpoint=/home/xinyiy/11-785_EfficientAT/wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt
    --experiment_name=ESC50_PAI_regularbp

[Option B] Dendrite search with Perforated Backpropagation (requires API token, better results):
EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --model_name=mn10_as --fold=1 \
    --finetune_checkpoint=/home/xinyiy/11-785_EfficientAT/wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt \
    --perforated_bp \
    --checkpoint_path=mn10_dendrite_pbp/final_clean_pai.pt

EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --model_name=mn10_as --fold=1 \
    --finetune_checkpoint=/home/xinyiy/11-785_EfficientAT/wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt \
    --perforated_bp \
    --experiment_name=ESC50_PAI_pbp

EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --model_name=mn10_as --fold=1 \
    --finetune_checkpoint=/home/xinyiy/11-785_EfficientAT/wandb/ESC50_20260311_204237/mn10_esc50_epoch_79_acc_952.pt \
    --perforated_bp \
    --experiment_name=ESC50_PAI_pbp_force_switchP \

  


[Eval] 
option A:
EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --eval_only --model_name=mn10_as --fold=1 \
    --checkpoint_path=ESC50_PAI/backup/best_model.pt


After training completes, evaluate PAI/final_clean_pai.pt:
EFFICIENTAT_ESC50_DIR=/home/xinyiy/datasets/ESC-50 \
python ex_esc50_perforated.py \
    --cuda --eval_only --model_name=mn10_as --fold=1 \
    --checkpoint_path=PAI/final_clean_pai.pt
'''
# ── PAI / Perforated BP setup ─────────────────────────────────────────────────
# env vars MUST be set before importing perforatedai / perforatedbp
_perforated_bp = '--perforated_bp' in sys.argv
# Always set credentials so perforatedbp can be imported (it checks license on import).
# Whether PBP is actually used during training is controlled by --perforated_bp flag.
os.environ["PAIEMAIL"] = "PAIUser3.11.2026@perforatedai.com"
os.environ["PAITOKEN"] = ("g3ZDrJNAmBlh/tAdFUkkedY+mUGqaueCPQXAtkVyNq405Fc9+20MhmEIDttx285Eh"
                           "fFqDHFtMRA20BWKpQaqai3wNxaOeCNvsNsF7Nn2nTmpFUmkHGvmVWGD5JI1uxdPd"
                           "W6jeUmaQBKNUNcNKOumr1iDaQpFnCvFDDcSi3yUmZ5JoCd0c/lAyGmRXWQ+dInO"
                           "XX6MdE4XVmv7DI8jW626pNLemX7ZMo4dEGikNuhyuiAD1IJYNYaJxUK0zaizx/A"
                           "vq6QbmwQviPjXDNyBsTdVzJQzhAG6Zf2DlU9j29RnEaj0XGd4j3MJeqsrq0FeXK"
                           "qEMShnBKO0oL69CP4icVTOpQ==")
if _perforated_bp:
    print("Perforated Backpropagation enabled.")
else:
    print("Running dendrite search without Perforated Backpropagation.")

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import argparse
from sklearn import metrics
import torch.nn.functional as F

from datasets.esc50 import get_test_set, get_training_set
from models.mn.model import get_model as get_mobilenet
from models.dymn.model import get_model as get_dymn
from models.preprocess import AugmentMelSTFT
from helpers.init import worker_init_fn
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from helpers.wandb import get_wandb


wandb = get_wandb()


def _get_device(args):
    return torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')


def _build_mel(args, device):
    mel = AugmentMelSTFT(n_mels=args.n_mels,
                         sr=args.resample_rate,
                         win_length=args.window_size,
                         hopsize=args.hop_size,
                         n_fft=args.n_fft,
                         freqm=args.freqm,
                         timem=args.timem,
                         fmin=args.fmin,
                         fmax=args.fmax,
                         fmin_aug_range=args.fmin_aug_range,
                         fmax_aug_range=args.fmax_aug_range
                         )
    mel.to(device)
    return mel


def _resolve_width(args):
    if args.model_name:
        return NAME_TO_WIDTH(args.model_name)
    return args.model_width


def _build_model(args, device, load_pretrained=True):
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained and load_pretrained else None
    width = _resolve_width(args)
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=50)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=50)
    # Note: initialize_pai is called in train() before model.to(device)
    return model


def _build_eval_loader(args):
    return DataLoader(dataset=get_test_set(resample_rate=args.resample_rate, fold=args.fold),
                      worker_init_fn=worker_init_fn,
                      num_workers=args.num_workers,
                      batch_size=args.batch_size)


def _targets_to_indices(targets):
    if targets.ndim == 1:
        return targets.long()
    return targets.argmax(dim=1)


def _cross_entropy(logits, targets, reduction="mean"):
    if targets.is_floating_point():
        targets = targets.to(dtype=logits.dtype)
    return F.cross_entropy(logits, targets, reduction=reduction)


def _checkpoint_name(args, epoch, accuracy):
    model_tag = args.model_name.split("_", 1)[0] if args.model_name else "model"
    return f"{model_tag}_esc50_epoch_{epoch}_acc_{int(round(accuracy * 1000))}.pt"


def train(args):
    # Train Models for Acoustic Scene Classification

    # ── PAI config ────────────────────────────────────────────────────────────
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(args.test_dendrites)
    GPA.pc.set_weight_decay_accepted(True)
    # If not using PBP, disable it even if perforatedbp is installed
    if not _perforated_bp:
        GPA.pc.set_perforated_backpropagation(False)
    else:
        # ESC-50 has only ~13 batches/epoch; PBP default (100) would never complete
        GPA.pc.set_initial_correlation_batches(10)
    # ─────────────────────────────────────────────────────────────────────────

    # logging is done using wandb
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        notes="Fine-tune Models on ESC50 with PerforatedAI dendrites.",
        tags=["Environmental Sound Classification", "Fine-Tuning", "PerforatedAI"],
        config=args,
        name=args.experiment_name
    )

    device = _get_device(args)
    mel = _build_mel(args, device)

    # ── PAI model init (before .to(device)) ───────────────────────────────────
    # If a fine-tuned checkpoint is provided, use that as the starting point.
    # Otherwise fall back to --pretrained (AudioSet weights) or random init.
    if args.finetune_checkpoint:
        model = _build_model(args, device=device, load_pretrained=False)
        state_dict = torch.load(args.finetune_checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded fine-tuned checkpoint: {args.finetune_checkpoint}")
    else:
        model = _build_model(args, device=device, load_pretrained=True)
    # initialize_pai must happen BEFORE model.to(device)
    model = UPA.initialize_pai(model, save_name=args.experiment_name)
    # PAI only wraps Conv2d/Linear; tag remaining params (e.g. BatchNorm) as neuron
    # so PBP's filter_params doesn't hit pdb.set_trace() on untagged params
    for p in model.parameters():
        if not hasattr(p, 'parameter_type'):
            p.parameter_type = 'n'
    model.to(device)
    # ─────────────────────────────────────────────────────────────────────────

    # dataloader
    dl = DataLoader(dataset=get_training_set(resample_rate=args.resample_rate,
                                             roll=False if args.no_roll else True,
                                             wavmix=False if args.no_wavmix else True,
                                             gain_augment=args.gain_augment,
                                             fold=args.fold),
                    worker_init_fn=worker_init_fn,
                    num_workers=args.num_workers,
                    batch_size=args.batch_size,
                    shuffle=True)

    # evaluation loader
    eval_dl = _build_eval_loader(args)

    # optimizer & scheduler
    # ── PAI optimizer hook ────────────────────────────────────────────────────
    # PBP requires setup_optimizer (not set_optimizer_instance) to tag params correctly
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.LambdaLR)
    optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
    schedArgs = {'lr_lambda': schedule_lambda}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    # ─────────────────────────────────────────────────────────────────────────

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')

    # ── while True instead of for-loop so PAI can add dendrites past n_epochs ─
    epoch = -1
    while True:
        epoch += 1
    # ─────────────────────────────────────────────────────────────────────────
        mel.train()
        model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description("Epoch {}: accuracy: {:.4f}, val_loss: {:.4f}"
                             .format(epoch + 1, accuracy, val_loss))
        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x = x.to(device)
            y = y.to(device)
            x = _mel_forward(x, mel)

            if args.mixup_alpha:
                rn_indices, lam = mixup(bs, args.mixup_alpha)
                lam = lam.to(x.device)
                x = x * lam.reshape(bs, 1, 1, 1) + \
                    x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                y_hat, _ = model(x)
                samples_loss = (_cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                                _cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                            1. - lam.reshape(bs)))

            else:
                y_hat, _ = model(x)
                samples_loss = _cross_entropy(y_hat, y, reduction="none")

            # loss
            loss = samples_loss.mean()

            # append training statistics
            train_stats['train_loss'].append(loss.detach().cpu().numpy())

            # Update Model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Update learning rate
        scheduler.step()

        # evaluate
        accuracy, val_loss = _test(model, mel, eval_dl, device)

        # ── PAI validation hook ───────────────────────────────────────────────
        # PBP p-mode: correlation always improves slightly so DOING_HISTORY never
        # switches back. Force switch after 10 epochs in p mode, then restore DOING_HISTORY.
        if _perforated_bp and GPA.pai_tracker.member_vars.get("mode") == "p":
            if GPA.pai_tracker.steps_after_switch() >= 10:
                GPA.pai_tracker.member_vars["switch_mode"] = GPA.pc.DOING_SWITCH_EVERY_TIME
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(accuracy, model)
        if _perforated_bp and GPA.pai_tracker.member_vars.get("mode") == "n":
            GPA.pai_tracker.member_vars["switch_mode"] = GPA.pc.DOING_HISTORY
        model = model.to(device)
        if training_complete:
            print("PAI training complete. Best model loaded.")
            break
        elif restructured:
            optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}
            schedArgs = {'lr_lambda': schedule_lambda}
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        # ─────────────────────────────────────────────────────────────────────

        # log train and validation statistics
        wandb.log({"train_loss": np.mean(train_stats['train_loss']),
                   "accuracy": accuracy,
                   "val_loss": val_loss
                   })

        # remove previous model and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = _checkpoint_name(args, epoch, accuracy)
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))


def evaluate(args):
    assert args.checkpoint_path is not None, "--checkpoint_path is required with --eval_only"

    device = _get_device(args)
    mel = _build_mel(args, device)

    # PAI saves in safetensors format - use load_system for PAI checkpoints,
    # fall back to torch.load for plain state_dict checkpoints (e.g. wandb saves).
    checkpoint_path = args.checkpoint_path
    folder = os.path.dirname(checkpoint_path) or "."
    name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    try:
        model = _build_model(args, device=device, load_pretrained=False)
        model = UPA.initialize_pai(model, save_name=folder)
        UPA.load_system(model, folder, name)
        model.to(device)
    except Exception:
        # Fallback: plain torch checkpoint (state_dict or pickled model)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict):
            model = _build_model(args, device, load_pretrained=False)
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        model.to(device)

    eval_dl = _build_eval_loader(args)
    accuracy, val_loss = _test(model, mel, eval_dl, device)
    print(f"ESC-50 fold {args.fold} evaluation")
    print(f"  checkpoint: {args.checkpoint_path}")
    print(f"  accuracy: {accuracy:.4f}")
    print(f"  val_loss: {val_loss:.4f}")


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
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(_targets_to_indices(y).cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(_cross_entropy(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    accuracy = metrics.accuracy_score(targets, outputs.argmax(axis=1))
    return accuracy, losses.mean()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESC-50 fine-tuning with PerforatedAI dendrites.')

    # general
    parser.add_argument('--experiment_name', type=str, default="ESC50_PAI")
    parser.add_argument('--wandb_project', type=str, default="11-785_perforated_ai")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='wandb team/entity name (e.g. your group shared project entity)')
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--eval_only', action='store_true', default=False)
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--finetune_checkpoint', type=str, default=None,
                        help='Path to your own fine-tuned ESC-50 checkpoint (.pt state_dict). '
                             'Used as starting point for PAI dendrite training.')

    # PAI flags
    parser.add_argument('--perforated_bp', action='store_true', default=False,
                        help='Enable Perforated Backpropagation (requires API token).')
    parser.add_argument('--test_dendrites', action='store_true', default=False,
                        help='Run dendrite capacity test (adds 3 dendrite sets to verify setup).')

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument('--n_epochs', type=int, default=80)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--no_roll', action='store_true', default=False)
    parser.add_argument('--no_wavmix', action='store_true', default=False)
    parser.add_argument('--gain_augment', type=int, default=12)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    # lr schedule
    parser.add_argument('--lr', type=float, default=6e-5)
    parser.add_argument('--warm_up_len', type=int, default=10)
    parser.add_argument('--ramp_down_start', type=int, default=10)
    parser.add_argument('--ramp_down_len', type=int, default=65)
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
    if args.eval_only:
        evaluate(args)
    else:
        train(args)
