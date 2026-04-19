"""
FSD50K fine-tuning with Perforated AI Dendrites.

Usage:
    python ex_fsd50k_perforatedai.py

Architecture target (mn10_as, width_mult=1.0):
  Input  (1 channel mel spectrogram)
    |
  features[0]            Conv2dNormActivation (stem, 1->8)       [tracked only]
    |
  features[1..3]         InvertedResidual  (early: 8 -> 16 -> 24) [perforated]
  features[4..6]         InvertedResidual  (early: 24 -> 24)      [perforated]
  features[7..10]        InvertedResidual  (middle: 24 -> 40)     [perforated]
  features[11..12]       InvertedResidual  (later: 40 -> 56)      [perforated]
  features[13..15]       InvertedResidual  (later: 56 -> 80)      [perforated]
  features[16]           Conv2dNormActivation (final 1x1, 80->480)[perforated]
    |
  classifier.0..1        AvgPool2d + Flatten                      [no params]
  classifier.2 (Linear)  480 -> 640                               [perforated]
  classifier.3..4        Hardswish + Dropout                      [no params]
  classifier.5 (Linear)  640 -> 200 (logits)                      [perforated]

Dendrite conversion happens at the InvertedResidual-block level (skill
step 7.2): the 1x1 expand, depthwise conv, SE attention, and 1x1 project
inside each block are wrapped together rather than individually. For a
parameter-efficient run restricted to the "Later blocks + Final conv +
Classifier" region, set `config.pai.perforate_module_ids = list(
TOP_ONLY_MODULE_IDS)` in the config module (covers features[11..16] and
both classifier Linears).

PAI reference: https://www.perforatedai.com/docs
"""
import os

from configs.fsd50k_pai_config import (
    Config,
    PAI_EMAIL,
    PAI_TOKEN,
    config as default_config,
    make_fsd50k_config,
)

# ============================================================================
# Perforated AI license credentials -- must be set BEFORE importing perforatedai
# ============================================================================
os.environ["PAIEMAIL"] = PAI_EMAIL
os.environ["PAITOKEN"] = PAI_TOKEN

import wandb  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from sklearn import metrics  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from datasets.fsd50k import get_valid_set, get_training_set  # noqa: E402
from models.mn.model import get_model as get_mobilenet  # noqa: E402
from models.mn.block_types import InvertedResidual  # noqa: E402
from models.dymn.model import get_model as get_dymn  # noqa: E402
from models.preprocess import AugmentMelSTFT  # noqa: E402
from helpers.init import worker_init_fn  # noqa: E402
from helpers.utils import NAME_TO_WIDTH, mixup  # noqa: E402

from torchvision.ops.misc import Conv2dNormActivation  # noqa: E402

from perforatedai import globals_perforatedai as GPA  # noqa: E402
from perforatedai import utils_perforatedai as UPA  # noqa: E402
import perforatedbp  # noqa: F401,E402  (imported for side effects / license check)


# ----------------------------------------------------------------------------
# PAI API compatibility shim
# ----------------------------------------------------------------------------
# PAI renamed three config vars between releases:
#   old name                       ->  new name
#   modules_to_convert             ->  modules_to_perforate
#   module_names_to_convert        ->  module_names_to_perforate
#   module_ids_to_convert          ->  module_ids_to_perforate
#
# `PAIConfig.__getattr__` silently swallows unknown setters with an
# "Ignoring append attempt" log, so calling the wrong name is a no-op -- not
# an error. We alias the missing direction onto the existing one exactly
# once, then the rest of the code can call the canonical new-API names
# regardless of which library version is installed.
def _install_pai_name_aliases() -> None:
    pc = GPA.pc
    renames = {
        "modules_to_perforate": "modules_to_convert",
        "module_names_to_perforate": "module_names_to_convert",
        "module_ids_to_perforate": "module_ids_to_convert",
    }
    for new, old in renames.items():
        have_new = hasattr(pc, f"_{new}")
        have_old = hasattr(pc, f"_{old}")
        src, dst = (old, new) if have_old and not have_new else \
                   (new, old) if have_new and not have_old else (None, None)
        if src is None:
            continue
        for prefix in ("get_", "set_", "append_"):
            fn = getattr(pc, f"{prefix}{src}", None)
            if callable(fn):
                setattr(pc, f"{prefix}{dst}", fn)
    print(f"[PAI] installed name aliases (api={'new' if hasattr(pc, '_modules_to_perforate') else 'old'})")


_install_pai_name_aliases()


# ----------------------------------------------------------------------------
# PAI configuration
# ----------------------------------------------------------------------------
def _configure_pai(cfg: Config, save_name: str) -> None:
    """Apply every PAI setting BEFORE `UPA.perforate_model` runs.

    Per the skill, all `GPA.pc.set_*`/`append_*` calls must happen before
    `perforate_model` -- otherwise the tracker reads stale defaults when it
    walks the module tree.
    """
    pai = cfg.pai

    # --- Save name first so auto-save/auto-load points at THIS experiment. --
    # (PAIConfig.__init__ auto-loads `{cwd}/{save_name}/{save_name}_config.json`
    #  at import time; without this, stale JSON from a different run can
    #  silently override Python defaults.)
    GPA.pc.set_save_name(save_name)

    # --- Master switches ----------------------------------------------------
    GPA.pc.set_testing_dendrite_capacity(pai.testing_dendrite_capacity)
    GPA.pc.set_perforated_backpropagation(pai.perforated_bp)
    GPA.pc.set_verbose(pai.verbose)

    # --- Tensor layout for NCHW conv features (channel dim = index 1) ------
    GPA.pc.set_output_dimensions(list(pai.output_dimensions))

    # --- File + correlation knobs ------------------------------------------
    GPA.pc.set_using_safe_tensors(pai.using_safe_tensors)
    # `set_initial_correlation_batches` is a PBP-only variable; when PBP
    # isn't imported it's swallowed as a no-op by PAIConfig.__getattr__.
    GPA.pc.set_initial_correlation_batches(pai.initial_correlation_batches)

    # --- Module selection ---------------------------------------------------
    # Perforate every InvertedResidual block as a single unit, plus the final
    # 1x1 Conv2dNormActivation. The library-default name list already
    # contains PAISequential/Conv1d/Conv2d/Conv3d/Linear, which covers the
    # classifier Linear layers.
    #
    # We register by class object AND by class-name string: the class-identity
    # check in PAI can miss when the same class is imported via two paths
    # (e.g. torchvision re-exports), but the short-name string match still hits.
    GPA.pc.append_modules_to_perforate([InvertedResidual, Conv2dNormActivation])
    GPA.pc.append_module_names_to_perforate(["InvertedResidual", "Conv2dNormActivation"])

    # Optional extras by class name
    if pai.extra_module_names_to_perforate:
        GPA.pc.append_module_names_to_perforate(list(pai.extra_module_names_to_perforate))

    # Optional: restrict conversion to specific dotted module ids only.
    # When set, PAI only perforates these exact ids.
    if pai.perforate_module_ids:
        GPA.pc.append_module_ids_to_perforate(list(pai.perforate_module_ids))

    # Skip the stem conv (and any other tracked-only ids) -- tracked means
    # "don't add dendrites here, but do keep gradients flowing through".
    if pai.track_module_ids:
        GPA.pc.append_module_ids_to_track(list(pai.track_module_ids))

    # --- Switch mode --------------------------------------------------------
    if pai.switch_mode == "fixed":
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_fixed_switch_num(pai.fixed_switch_num)
        GPA.pc.set_first_fixed_switch_num(pai.first_fixed_switch_num)
    else:
        GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
        GPA.pc.set_n_epochs_to_switch(pai.n_epochs_to_switch)
        GPA.pc.set_p_epochs_to_switch(pai.p_epochs_to_switch)

    GPA.pc.set_history_lookback(pai.history_lookback)
    GPA.pc.set_cap_at_n(pai.cap_at_n)
    GPA.pc.set_max_dendrites(pai.max_dendrites)

    # --- Validation smoothing ----------------------------------------------
    GPA.pc.set_running_average_pb(pai.running_average_pb)

    # --- Improvement thresholds --------------------------------------------
    GPA.pc.set_pai_improvement_threshold(pai.pai_improvement_threshold)
    GPA.pc.set_pai_improvement_threshold_raw(pai.pai_improvement_threshold_raw)
    GPA.pc.set_improvement_threshold(pai.improvement_threshold)
    GPA.pc.set_improvement_threshold_raw(pai.improvement_threshold_raw)

    # --- Candidate dendrite init + stability -------------------------------
    GPA.pc.set_candidate_weight_initialization_multiplier(
        pai.candidate_weight_initialization_multiplier
    )
    GPA.pc.set_candidate_grad_clipping(pai.candidate_grad_clipping)
    GPA.pc.set_drawing_extra_graphs(pai.drawing_extra_graphs)


def _reapply_validation_thresholds(cfg: Config) -> None:
    """`perforate_model` overwrites some thresholds; re-assert them.

    Observed PAI quirk: running_average_pb and the four improvement
    thresholds are reset to library defaults during tracker initialization.
    They must be re-applied AFTER `perforate_model` to stick for training.
    """
    pai = cfg.pai
    GPA.pc.set_running_average_pb(pai.running_average_pb)
    GPA.pc.set_improvement_threshold(pai.improvement_threshold)
    GPA.pc.set_improvement_threshold_raw(pai.improvement_threshold_raw)
    GPA.pc.set_pai_improvement_threshold(pai.pai_improvement_threshold)
    GPA.pc.set_pai_improvement_threshold_raw(pai.pai_improvement_threshold_raw)


def _perforate_model(model, save_name: str, maximizing_score: bool):
    """Call `UPA.perforate_model` (canonical name, per skill + source).

    Falls back to the legacy `initialize_pai` alias only if `perforate_model`
    is not present on this install.
    """
    if hasattr(UPA, "perforate_model"):
        return UPA.perforate_model(
            model,
            doing_pai=True,
            save_name=save_name,
            maximizing_score=maximizing_score,
        )
    return UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name=save_name,
        maximizing_score=maximizing_score,
    )


def _report_dendrite_targets(model) -> None:
    """Print the modules PAI wrapped so the user can visually confirm."""
    try:
        pai_modules = UPA.get_pai_modules(model, 0)
    except Exception:
        return
    if not pai_modules:
        print("[PAI] WARNING: no PAINeuronModules were registered. "
              "Check `module_names_to_perforate` / `modules_to_perforate` / "
              "`module_ids_to_perforate`.")
        return
    print(f"[PAI] Registered {len(pai_modules)} dendrite-eligible modules:")
    for m in pai_modules:
        main_type = type(getattr(m, "main_module", m)).__name__
        print(f"  - {m.name:40s}  ({main_type})")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(cfg: Config = default_config):
    pai_save_dir = os.path.join("pai", cfg.experiment_name)
    os.makedirs(pai_save_dir, exist_ok=True)
    os.makedirs(os.path.join(pai_save_dir, "pai"), exist_ok=True)

    wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        group=cfg.wandb.group,
        notes=cfg.wandb.notes,
        tags=list(cfg.wandb.tags),
        config=_config_as_dict(cfg),
    )

    device = torch.device('cuda') if cfg.cuda and torch.cuda.is_available() else torch.device('cpu')

    # AMP is disabled under Perforated BP (its custom autograd isn't autocast-safe).
    amp_enabled = bool(not cfg.pai.perforated_bp and cfg.cuda and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    pp = cfg.preprocess
    mel = AugmentMelSTFT(
        n_mels=pp.n_mels, sr=pp.resample_rate,
        win_length=pp.window_size, hopsize=pp.hop_size, n_fft=pp.n_fft,
        freqm=pp.freqm, timem=pp.timem,
        fmin=pp.fmin, fmax=pp.fmax,
        fmin_aug_range=pp.fmin_aug_range, fmax_aug_range=pp.fmax_aug_range,
    ).to(device)

    # ---------------- Build the (un-perforated) model first ---------------
    mc = cfg.model
    model_name = mc.model_name
    pretrained_name = model_name if mc.pretrained else None
    width = NAME_TO_WIDTH(model_name) if model_name and mc.pretrained else mc.model_width
    if model_name.startswith("dymn"):
        model = get_dymn(
            width_mult=width,
            pretrained_name=pretrained_name,
            pretrain_final_temp=mc.pretrain_final_temp,
            num_classes=mc.num_classes,
        )
    else:
        model = get_mobilenet(
            width_mult=width,
            pretrained_name=pretrained_name,
            head_type=mc.head_type,
            se_dims=mc.se_dims,
            num_classes=mc.num_classes,
        )

    # ---------------- Configure PAI, THEN perforate the model --------------
    _configure_pai(cfg, save_name=pai_save_dir)
    model = _perforate_model(
        model,
        save_name=pai_save_dir,
        maximizing_score=True,  # mAP is "higher is better"
    )
    _reapply_validation_thresholds(cfg)
    _report_dendrite_targets(model)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"param_count: {total_params}")
    print(f"trainable_param_count: {trainable_params}")

    # ---------------- Data loaders -----------------------------------------
    dc = cfg.data
    dl = DataLoader(
        dataset=get_training_set(
            resample_rate=pp.resample_rate,
            roll=dc.roll,
            wavmix=dc.wavmix,
            gain_augment=dc.gain_augment,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.num_workers // 2 if cfg.num_workers > 0 else None,
    )

    valid_dl = DataLoader(
        dataset=get_valid_set(
            resample_rate=pp.resample_rate,
            variable_eval=dc.variable_eval_length,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=cfg.num_workers,
        batch_size=1 if dc.variable_eval_length else cfg.batch_size,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=cfg.num_workers // 2 if cfg.num_workers > 0 else None,
    )

    # ---------------- PAI-managed optimizer + scheduler --------------------
    oc = cfg.optim
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optim_args = {'params': model.parameters(), 'lr': oc.lr, 'weight_decay': oc.weight_decay}
    sched_args = {'mode': oc.scheduler_mode, 'patience': oc.scheduler_patience}
    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)

    # ---------------- Training loop ----------------------------------------
    name = None
    mAP, ROC, val_loss = float('NaN'), float('NaN'), float('NaN')
    train_loss_epoch = float('NaN')
    best_val_mAP, best_val_epoch = float("-inf"), -1

    epoch = -1
    while True:
        epoch += 1
        mel.train()
        model.train()
        train_loss_list = []

        pbar = tqdm(dl)
        pbar.set_description(
            f"Epoch {epoch + 1} | mAP={mAP:.4f} val_loss={val_loss:.4f} "
            f"train_loss={train_loss_epoch:.4f}"
        )

        for batch in pbar:
            x, _, y = batch
            bs = x.size(0)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                x = _mel_forward(x, mel)

                if oc.mixup_alpha:
                    rn_indices, lam = mixup(bs, oc.mixup_alpha)
                    lam = lam.to(x.device)
                    x = (
                        x * lam.reshape(bs, 1, 1, 1)
                        + x[rn_indices] * (1. - lam.reshape(bs, 1, 1, 1))
                    )
                    y_hat, _ = model(x)
                    y_mix = y * lam.reshape(bs, 1) + y[rn_indices] * (1. - lam.reshape(bs, 1))
                    samples_loss = F.binary_cross_entropy_with_logits(
                        y_hat, y_mix, reduction="none"
                    )
                else:
                    y_hat, _ = model(x)
                    samples_loss = F.binary_cross_entropy_with_logits(
                        y_hat, y, reduction="none"
                    )

                loss = samples_loss.mean()

            train_loss_list.append(loss.detach().cpu().numpy())

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        train_loss_epoch = float(np.mean(train_loss_list)) if train_loss_list else float('NaN')

        # Train-loss trace for PAI graphs. Negated so "higher = better" still
        # holds on the extra-score axis (PAI treats these as informational).
        GPA.pai_tracker.add_extra_score(-train_loss_epoch, "NegTrainLoss")
        model.to(device)

        mAP, ROC, val_loss, _ = _test(model, mel, valid_dl, device)

        if mAP > best_val_mAP:
            best_val_mAP = mAP
            best_val_epoch = epoch

        # PAI drives switching + "training complete" off mAP on the 0-100 scale.
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            mAP * 100.0, model
        )
        model.to(device)

        wandb.log({
            "train_loss": train_loss_epoch,
            "mAP": mAP,
            "ROC": ROC,
            "val_loss": val_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })

        if name is not None:
            try:
                os.remove(os.path.join(wandb.run.dir, name))
            except FileNotFoundError:
                pass
        name = (
            f"mn{str(width).replace('.', '')}_fsd50k_pai_"
            f"epoch_{epoch}_mAP_{int(round(mAP * 1000))}.pt"
        )
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, name))

        if training_complete:
            print(
                f"PAI training complete at epoch {epoch}. "
                f"Best validation mAP (this run): {best_val_mAP:.4f} "
                f"(epoch {best_val_epoch + 1}); "
                f"last epoch mAP: {mAP:.4f}"
            )
            break
        elif restructured:
            # Topology changed: optimizer state is stale, rebuild with the
            # EXACT SAME args (skill Step 7 requirement).
            optim_args = {'params': model.parameters(), 'lr': oc.lr, 'weight_decay': oc.weight_decay}
            sched_args = {'mode': oc.scheduler_mode, 'patience': oc.scheduler_patience}
            optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
            scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _mel_forward(x, mel):
    old_shape = x.size()
    x = x.reshape(-1, old_shape[2])
    x = mel(x)
    x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
    return x


def _test(model, mel, eval_loader, device):
    model.eval()
    mel.eval()

    targets, outputs, losses = [], [], []
    pbar = tqdm(eval_loader)
    pbar.set_description("Validating")
    for batch in pbar:
        x, _, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.binary_cross_entropy_with_logits(y_hat, y).cpu().numpy())

    targets = np.concatenate(targets)
    outputs = np.concatenate(outputs)
    losses = np.stack(losses)
    mAP = metrics.average_precision_score(targets, outputs, average=None)
    ROC = metrics.roc_auc_score(targets, outputs, average=None)
    return float(mAP.mean()), float(ROC.mean()), float(losses.mean()), int(targets.shape[0])


def _config_as_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == '__main__':
    train(default_config)
