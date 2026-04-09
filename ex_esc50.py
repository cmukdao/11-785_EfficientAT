import wandb
import numpy as np
import os
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


def train(args):
    # Train Models for Acoustic Scene Classification

    # logging is done using wandb
    wandb.init(
        entity="11-785_perforated_ai",
        project="ESC50",
        group=f"ESC50-{args.model_name}",
        notes="Fine-tune Models on ESC50.",
        tags=["Environmental Sound Classification", "Fine-Tuning"],
        config=args,
        name=args.experiment_name
    )

    device = torch.device('cuda') if args.cuda and torch.cuda.is_available() else torch.device('cpu')

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # model to preprocess waveform into mel spectrograms
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

    # load prediction model
    model_name = args.model_name
    pretrained_name = model_name if args.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and args.pretrained else args.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(width_mult=width, pretrained_name=pretrained_name,
                         pretrain_final_temp=args.pretrain_final_temp,
                         num_classes=50)
    else:
        model = get_mobilenet(width_mult=width, pretrained_name=pretrained_name,
                              head_type=args.head_type, se_dims=args.se_dims,
                              num_classes=50)
    model.to(device)

    # parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("param_count: ", total_params)
    print("trainable_param_count: ", trainable_params)
    
    # dataloader
    dl = DataLoader(
        dataset=get_training_set(
            resample_rate=args.resample_rate,
            roll=False if args.no_roll else True,
            wavmix=False if args.no_wavmix else True,
            gain_augment=args.gain_augment,
            fold=args.fold
        ),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=args.num_workers//2 if args.num_workers > 0 else None,
    )

    eval_dl = DataLoader(
        dataset=get_test_set(resample_rate=args.resample_rate, fold=args.fold),
        worker_init_fn=worker_init_fn,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=args.num_workers//2 if args.num_workers > 0 else None
    )

    # optimizer & scheduler
    lr = args.lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # phases of lr schedule: exponential increase, constant lr, linear decrease, fine-tune
    schedule_lambda = \
        exp_warmup_linear_down(args.warm_up_len, args.ramp_down_len, args.ramp_down_start, args.last_lr_value)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule_lambda)

    name = None
    accuracy, val_loss = float('NaN'), float('NaN')
    best_accuracy = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(args.max_epochs):
        mel.train()
        model.train()
        train_stats = dict(train_loss=list())
        pbar = tqdm(dl)
        pbar.set_description(
            "Epoch {}/{} (best {:.4f}, stall {}/{}): acc {:.4f}, val_loss {:.4f}".format(
                epoch + 1,
                args.max_epochs,
                best_accuracy if best_state is not None else float("nan"),
                epochs_no_improve,
                args.early_stop_patience,
                accuracy,
                val_loss,
            )
        )
        
        for batch in pbar:
            x, f, y = batch
            bs = x.size(0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(scaler is not None)):
                x = _mel_forward(x, mel)

                if args.mixup_alpha:
                    rn_indices, lam = mixup(bs, args.mixup_alpha)
                    lam = lam.to(x.device)
                    x = x * lam.reshape(bs, 1, 1, 1) + \
                        x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                    y_hat, _ = model(x)
                    samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(bs) +
                                    F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (
                                                1. - lam.reshape(bs)))

                else:
                    y_hat, _ = model(x)
                    samples_loss = F.cross_entropy(y_hat, y, reduction="none")

                # loss
                loss = samples_loss.mean()

                # append training statistics
                train_stats['train_loss'].append(loss.detach().cpu().numpy())

                # Update Model
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        # Update learning rate
        scheduler.step()

        # evaluate
        accuracy, val_loss = _test(model, mel, eval_dl, device)

        if accuracy > best_accuracy + args.early_stop_min_delta:
            best_accuracy = accuracy
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # log train and validation statistics
        wandb.log({
            "train_loss": np.mean(train_stats['train_loss']),
            "accuracy": accuracy,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "epochs_without_improve": epochs_no_improve,
        })

        # remove previous model (we try to not flood your hard disk) and save latest model
        if name is not None:
            os.remove(os.path.join(wandb.run.dir, name))
        name = f"{model_name}_esc50_epoch_{epoch}_acc_{int(round(accuracy*1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))

        if (
            epoch + 1 >= args.min_epochs
            and epochs_no_improve >= args.early_stop_patience
        ):
            print(
                "Early stopping: validation accuracy did not improve by more than "
                f"{args.early_stop_min_delta} for {args.early_stop_patience} epochs "
                f"(best accuracy {best_accuracy:.4f} at an earlier epoch)."
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        best_name = f"{model_name}_esc50_best_acc_{int(round(best_accuracy * 1000))}.pt"
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, best_name))
        print(f"Restored best weights (accuracy {best_accuracy:.4f}), saved as {best_name}")


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
    parser = argparse.ArgumentParser(description='Fold 1. Reduced batch size to 64.')

    # general
    parser.add_argument('--experiment_name', type=str, default="ESC50-mn05_as-64bs")
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--fold', type=int, default=1)

    # training
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--model_name', type=str, default="mn10_as")
    parser.add_argument('--pretrain_final_temp', type=float, default=1.0)  # for DyMN
    parser.add_argument('--model_width', type=float, default=1.0)
    parser.add_argument('--head_type', type=str, default="mlp")
    parser.add_argument('--se_dims', type=str, default="c")
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=500,
        help='Upper bound on epochs (training usually stops earlier via early stopping).',
    )
    parser.add_argument(
        '--early_stop_patience',
        type=int,
        default=20,
        help='Stop if validation accuracy does not improve for this many epochs.',
    )
    parser.add_argument(
        '--early_stop_min_delta',
        type=float,
        default=0.0,
        help='Minimum accuracy gain to count as improvement (e.g. 0.001 for 0.1%%).',
    )
    parser.add_argument(
        '--min_epochs',
        type=int,
        default=10,
        help='Do not early-stop before this many epochs (e.g. allow LR warm-up).',
    )
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
    train(args)
