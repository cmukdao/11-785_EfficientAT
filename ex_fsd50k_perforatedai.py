"""
FSD50K fine-tuning with Perforated AI Dendrites.

Usage:
    python ex_fsd50k_perforatedai.py

PERFORATION STRATEGY (per backbone)
-----------------------------------
MobileNet (`mn*_as`): each `InvertedResidual` block + the final
`Conv2dNormActivation` (80->480) are wrapped as `PAINeuronModule` units,
along with both classifier `Linear`s (skill step 7.2, block-level wrap).
The stem conv is excluded via `track_module_ids`.

DyMN (`dymn*_as`): all 15 `DY_Block`s + the two classifier `Linear`s are
perforated. `DY_Block.forward(x, g=None)` is a multi-input module (feature
map + upstream global context), so PAI's default single-tensor dendrite
machinery silently drops `g`. We fix this with `DYBlockProcessor`
registered via `modules_with_processing` (customization.md section 2.2) so
the dendrite clone receives the same `(x, g)` pair as the neuron.

`Conv2dNormActivation` is still track-only on DyMN: empirically the `out_c`
head sat at PBScore ~0 for the entire prior run (wide 64->384 projection
near the pooled output -- no residual error left to model), and the
`in_c` stem is excluded by the skill's "no dendrites on low-level
features" guidance anyway. The walker stops at both as atomic units.

To override per-experiment, set `config.pai.perforate_module_ids` in the
config module. `TOP_ONLY_MODULE_IDS` is a convenience preset for
parameter-efficient MN runs (features[11..16] + both classifier Linears).

PAI reference: https://www.perforatedai.com/docs
"""
import os

from configs.fsd50k_pai_config import (
    Config,
    PAI_EMAIL,
    PAI_TOKEN,
    config as default_config,
)

# ============================================================================
# Perforated AI license credentials -- must be set BEFORE importing perforatedai
# ============================================================================
os.environ["PAIEMAIL"] = PAI_EMAIL
os.environ["PAITOKEN"] = PAI_TOKEN

import wandb  
import numpy as np  
from tqdm import tqdm  
import torch  
from torch.utils.data import DataLoader  
from sklearn import metrics  
import torch.nn.functional as F  

from datasets.fsd50k import get_valid_set, get_training_set  
from models.mn.model import get_model as get_mobilenet  
from models.mn.block_types import InvertedResidual  
from models.dymn.model import get_model as get_dymn  
from models.dymn.dy_block import DY_Block  
from models.preprocess import AugmentMelSTFT  
from helpers.init import worker_init_fn  
from helpers.utils import NAME_TO_WIDTH, mixup  

from torchvision.ops.misc import Conv2dNormActivation  

from perforatedai import globals_perforatedai as GPA  
from perforatedai import utils_perforatedai as UPA  
import perforatedbp  # noqa: F401,E402  (imported for side effects / license check)


# ----------------------------------------------------------------------------
# PAI API compatibility shim
# ----------------------------------------------------------------------------
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
# DY_Block dendrite processor (customization.md section 2.2)
# ----------------------------------------------------------------------------
# NOTE: processor classes MUST live at module scope (not nested inside another
# class/function). PAI copies them into its internal registry by reference and
# instantiates one per wrapped module at `perforate_model` time; bound methods
# on a local class would capture `self` of the enclosing scope and break.
class DYBlockProcessor:
    """Dendrite processor for ``DY_Block`` (multi-input, single-tensor output).

    ``DY_Block.forward(self, x, g=None) -> Tensor``
      * ``x`` is the feature map (B, C, H, W).
      * ``g`` is the upstream global-context vector produced by the
        preceding ``DY_Block.context_gen`` (``None`` on block 0).
      * Output is a single tensor after projection + internal residual.

    Without a processor, PAI's default dendrite clone only receives ``x`` and
    calls ``DY_Block(x)`` -- i.e. ``g`` defaults to ``None`` inside the clone,
    ``context_gen(x, None)`` runs on the degenerate path, every
    ``DynamicConv`` produces near-constant kernels and the PBScore sits at
    ~0. That's exactly what the first DyMN run's "Best PBScores" plot showed
    for all 15 blocks.

    Since the module is single-tensor-out, three of the four hooks are pure
    pass-throughs; only ``pre_d`` has non-trivial work (forwarding *all*
    args + kwargs to the dendrite so it gets the same ``g`` as the neuron).
    """

    def pre_d(self, *args, **kwargs):
        # Dendrite input must match the main module's signature -- PAI will
        # splat the returned (args, kwargs) into `dendrite_module(*args, **kwargs)`.
        return args, kwargs

    def post_d(self, dendrite_out, *args, **kwargs):
        # Single-tensor output -- nothing to extract.
        return dendrite_out

    def post_n1(self, neuron_out, *args, **kwargs):
        # Single-tensor output -- nothing to split off / save.
        return neuron_out

    def post_n2(self, combined_out, *args, **kwargs):
        # No side state was saved in post_n1, so no re-tupling needed.
        return combined_out

    def clear_processor(self):
        # Called whenever PAI saves/copies the network; we hold no state.
        pass


# ----------------------------------------------------------------------------
# Per-model block registration
# ----------------------------------------------------------------------------
def _select_pai_blocks_for_model(model_name: str):
    """Return ``(perforate_classes, perforate_names, track_classes, track_names)``.

    ``perforate_*``: classes wrapped as ``PAINeuronModule`` (get dendrites).
    ``track_*``:     classes wrapped as ``TrackedNeuronModule`` (no dendrites,
                     walker stops here so interior layers are untouched).
    Both are registered by class object AND by short class name to be robust
    against duplicate imports (identity check misses, name check hits).

    MobileNet (`mn*`): `InvertedResidual` blocks and the head
    `Conv2dNormActivation` are the canonical perforation targets -- they all
    have clean `forward(x) -> y` semantics so PAI's default dendrite machinery
    works out of the box. The stem is excluded via ``track_module_ids``.

    DyMN (`dymn*`): `DY_Block` is perforated as a single unit via the
    ``DYBlockProcessor`` above -- the processor routes the multi-tensor
    ``(x, g)`` inputs through the dendrite clone so its internal
    ``DynamicConv / DyReLUB / CoordAtt`` see the real upstream global context
    instead of a degenerate ``g=None``. ``Conv2dNormActivation`` stays
    track-only: prior-run PBScore on ``out_c`` (64->384 head) was ~0 (wide
    projection near the pooled output -- no residual error left to model),
    and the stem ``in_c`` is excluded by the skill's "no dendrites on
    low-level features" guidance via ``track_module_ids``.
    """
    family = model_name.lower()
    if family.startswith("dymn"):
        perforate_classes = [DY_Block]
        perforate_names = ["DY_Block"]
        track_classes = [Conv2dNormActivation]
        track_names = ["Conv2dNormActivation"]
    else:
        perforate_classes = [InvertedResidual, Conv2dNormActivation]
        perforate_names = ["InvertedResidual", "Conv2dNormActivation"]
        track_classes = []
        track_names = []
    return perforate_classes, perforate_names, track_classes, track_names


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
    # Block-level perforation: wrap each residual block as a single
    # PAINeuronModule (rather than dendriting its inner Conv2d/Linear layers
    # individually). The library-default name list already contains
    # PAISequential/Conv1d/Conv2d/Conv3d/Linear, which covers the classifier
    # Linears in both backbones.
    #
    # We register by class object AND by class-name string: the class-identity
    # check in PAI can miss when the same class is imported via two paths
    # (e.g. torchvision re-exports), but the short-name string match still hits.
    perforate_classes, perforate_names, track_classes, track_names = (
        _select_pai_blocks_for_model(cfg.model.model_name)
    )
    if perforate_classes:
        GPA.pc.append_modules_to_perforate(perforate_classes)
    if perforate_names:
        GPA.pc.append_module_names_to_perforate(perforate_names)

    # Optional extras by class name
    if pai.extra_module_names_to_perforate:
        GPA.pc.append_module_names_to_perforate(list(pai.extra_module_names_to_perforate))

    # Optional: restrict conversion to specific dotted module ids only.
    # When set, PAI only perforates these exact ids.
    if pai.perforate_module_ids:
        GPA.pc.append_module_ids_to_perforate(list(pai.perforate_module_ids))

    # Skip specific dotted module ids (stem etc.) -- tracked means
    # "don't add dendrites here, but do keep gradients flowing through".
    if pai.track_module_ids:
        GPA.pc.append_module_ids_to_track(list(pai.track_module_ids))

    # Track-only classes: PAI wraps these as `TrackedNeuronModule` (no
    # dendrites added, and the walker STOPS at them so the inner layers are
    # never flagged as "unwrapped norm"). We register both by class object
    # and short name for robustness against duplicate imports.
    if track_classes:
        GPA.pc.append_modules_to_track(track_classes)
    all_track_names = list(track_names) + list(pai.track_module_names)
    if all_track_names:
        GPA.pc.append_module_names_to_track(all_track_names)

    # Per-class processors for modules whose forward takes >1 tensor or
    # returns >1 tensor (customization.md section 2.2). DY_Block is the only
    # such module in this codebase; MobileNet's InvertedResidual has a clean
    # `forward(x) -> y` signature and uses PAI's default machinery.
    # IMPORTANT: the two processor arrays are paired by position -- for every
    # entry in `module_names_with_processing` there must be a matching entry
    # in `module_by_name_processing_classes`. If you ever add a MobileNet
    # processor, append in BOTH lists in lockstep.
    if cfg.model.model_name.lower().startswith("dymn"):
        GPA.pc.append_module_names_with_processing(["DY_Block"])
        GPA.pc.append_module_by_name_processing_classes([DYBlockProcessor])

    # Silence the "unwrapped modules" pdb break once the user has audited
    # their registration and confirmed all unwrapped params are intentional.
    if pai.unwrapped_modules_confirmed:
        GPA.pc.set_unwrapped_modules_confirmed(True)

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
    # `improvement_threshold` must be a `list` (not tuple): PAIConfig's
    # getter checks `type(...) is list` exactly, and a tuple silently
    # defeats the per-dendrite-count indexing.
    GPA.pc.set_pai_improvement_threshold(pai.pai_improvement_threshold)
    GPA.pc.set_pai_improvement_threshold_raw(pai.pai_improvement_threshold_raw)
    GPA.pc.set_improvement_threshold(list(pai.improvement_threshold))
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
    GPA.pc.set_improvement_threshold(list(pai.improvement_threshold))
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

    # Full post-perforation module tree: lets the user visually confirm
    # which blocks became `PAINeuronModule` / `TrackedNeuronModule` and
    # which stayed as plain `nn.Module`s.
    print("[PAI] Model structure after perforation:")
    print(model)

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
        num_workers=dc.num_workers,
        batch_size=dc.batch_size,
        shuffle=True,
        pin_memory=True,
        persistent_workers=dc.num_workers > 0,
        prefetch_factor=dc.prefetch_factor,
    )

    valid_dl = DataLoader(
        dataset=get_valid_set(
            resample_rate=pp.resample_rate,
            variable_eval=dc.variable_eval_length,
        ),
        worker_init_fn=worker_init_fn,
        num_workers=dc.num_workers,
        batch_size=1 if dc.variable_eval_length else dc.batch_size,
        pin_memory=True,
        persistent_workers=dc.num_workers > 0,
        prefetch_factor=dc.prefetch_factor,
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
    train_mAP, train_ROC = float('NaN'), float('NaN')
    best_val_mAP, best_val_epoch = float("-inf"), -1

    epoch = -1
    while True:
        epoch += 1
        mel.train()
        model.train()
        train_loss_list = []
        train_targets, train_outputs = [], []

        pbar = tqdm(dl)
        pbar.set_description(
            f"Epoch {epoch + 1} | mAP={mAP:.4f} val_loss={val_loss:.4f} "
            f"train_loss={train_loss_epoch:.4f} train_mAP={train_mAP:.4f}"
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
            # Accumulate predictions vs. ORIGINAL (pre-mixup) targets so we
            # can compute an epoch-level training mAP/ROC without an extra
            # forward pass. This is a noisy running estimate (model weights
            # are updating through the epoch, and under mixup the logits
            # correspond to mixed inputs), but it tracks train-set fit and
            # makes the train/val generalization gap visible in wandb.
            train_targets.append(y.detach().cpu().numpy())
            train_outputs.append(y_hat.detach().float().cpu().numpy())

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        train_loss_epoch = float(np.mean(train_loss_list)) if train_loss_list else float('NaN')
        train_mAP, train_ROC = _score(train_targets, train_outputs)

        # Canonical PAI extra-score registration (SKILL.md 7.1). The label
        # "train" is what `perforatedai-analyze` keys off for overfitting /
        # train-val-gap diagnostics, so don't rename it. Value is on the
        # 0-100 scale to match `add_validation_score(mAP * 100, ...)` below
        # so both traces share the y-axis on PAI's auto-generated graphs.
        GPA.pai_tracker.add_extra_score(train_mAP * 100.0, "train")
        # Auxiliary trace: negated train loss so "higher = better" still
        # holds on the extra-score axis (PAI treats extras as informational).
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
            "train_mAP": train_mAP,
            "train_ROC": train_ROC,
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


def _score(targets_list, outputs_list):
    """Macro mAP and ROC-AUC from accumulated per-batch target / logit arrays.

    Shared by training and validation. Two boundary fixes:

    1. Binarize targets with `>= 0.5`. When `wavmix=True`, the training
       set is wrapped in `MixupDataset`, which returns DATASET-LEVEL
       blended targets like `y1 * l + y2 * (1 - l)` with `l in [0.5, 1]`
       -- i.e. continuous floats (0.73, 0.27, ...). sklearn's
       `average_precision_score` rejects those with
       "continuous-multioutput format is not supported". Thresholding at
       0.5 recovers the DOMINANT sample's label vector, which is the
       natural binary ground truth to score against. This is a no-op on
       validation's already-binary `{0.0, 1.0}` targets.
    2. `nan_to_num` on outputs. Under AMP, freshly-initialized dendrite
       candidates occasionally push fp16 logits to inf/nan on the first
       batches; `GradScaler` rejects those optimizer steps but the
       logits were already captured. Scrub to finite sentinels so
       sklearn's internal sort stays stable.
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
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.no_grad():
            x = _mel_forward(x, mel)
            y_hat, _ = model(x)
        targets.append(y.cpu().numpy())
        outputs.append(y_hat.float().cpu().numpy())
        losses.append(F.binary_cross_entropy_with_logits(y_hat, y).cpu().numpy())

    mAP, ROC = _score(targets, outputs)
    return mAP, ROC, float(np.stack(losses).mean()), int(np.concatenate(targets).shape[0])


def _config_as_dict(cfg: Config) -> dict:
    from dataclasses import asdict
    return asdict(cfg)


if __name__ == '__main__':
    train(default_config)
