"""
FSD50K fine-tuning with Perforated AI Dendrites.

Usage:
    python ex_fsd50k_perforatedai.py

PERFORATION STRATEGY (per backbone)
-----------------------------------
MobileNet (`mn*_as`): probe-validated "SE + classifier Linears only"
strategy. `probe_grad_magnitude.py` (see `probes/mn04_as_plateau.csv`)
showed SE-block `Linear`s and the two classifier `Linear`s carry >=26x
the gradient magnitude of any IR/C2NA block wrapper at the pre-switch
plateau. We therefore:

  1. `set_module_names_to_perforate(["Linear"])` -- REPLACE PAI's default
     `["Conv2d", "Linear", ...]` so the walker does NOT wrap Conv2d
     leaves inside `InvertedResidual` / `Conv2dNormActivation`.
  2. No class entries in `modules_to_perforate` for MN (empty list).
     With nothing in the class filter except "Linear", the walker
     descends through every IR / C2NA and only wraps the Linears it
     finds on the way down.
  3. Track each ``Conv2dNormActivation`` block as a whole (class name) so
     PAI's walker stops at the C2NA boundary: inner ``Conv2d`` / ``BatchNorm2d``
     are not wrapped separately (``TrackedNeuronModule`` on the block only).

Effective perforation set for mn*_as:
  - 15 x 2 SE Linears inside `features[N].block[2].conc_se_layers[i]`
    (fc1 squeeze + fc2 excite)
  - classifier.2 (pre-logit)
  - classifier.5 (logit)
Effective parameter overhead per dendrite set:  ~(SE + classifier) params,
much smaller than the previous block-level wrap (~backbone params).

DyMN (`dymn*_as`): unchanged. All 15 `DY_Block`s + the two classifier
`Linear`s are perforated. `DY_Block.forward(x, g=None)` is a multi-input
module (feature map + upstream global context), so PAI's default single-
tensor dendrite machinery silently drops `g`. We fix this with
`DYBlockProcessor` registered via `modules_with_processing`
(customization.md section 2.2) so the dendrite clone receives the same
`(x, g)` pair as the neuron. `Conv2dNormActivation` remains track-only
(empirical PBScore ~0 on `out_c`, stem excluded by `.in_c` skip).

PAI reference: https://www.perforatedai.com/docs
"""
import argparse
import copy
import os
import shutil
import sys
import time
from contextlib import contextmanager

from configs.fsd50k_pai_config import (
    Config,
    PAI_EMAIL,
    PAI_TOKEN,
    apply_runtime_overrides,
    config as default_config,
    get_pai_preset_names,
)


def _print_available_presets() -> None:
    canonical, aliases = get_pai_preset_names()
    print("Canonical PAI presets:")
    for name in canonical:
        print(f"  {name}")
    if aliases:
        print("\nLegacy aliases:")
        for old, new in aliases.items():
            print(f"  {old} -> {new}")


if __name__ == "__main__" and "--list-presets" in sys.argv:
    _print_available_presets()
    sys.exit(0)

# ============================================================================
# Perforated AI license credentials -- must be set BEFORE importing perforatedai
# ============================================================================
os.environ["PAIEMAIL"] = PAI_EMAIL
os.environ["PAITOKEN"] = PAI_TOKEN

import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as torch_mp
import wandb
from sklearn import metrics
from torch.utils.data import DataLoader
from torchvision.ops.misc import Conv2dNormActivation
from tqdm import tqdm

try:
    torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass

from datasets.fsd50k import get_valid_set, get_training_set
from helpers.init import make_generator, seed_everything, worker_init_fn
from helpers.run_lifecycle import (
    EXIT_PAI_RESTRUCTURED,
    EXIT_WALL_TIME,
    append_metrics_row,
    create_run_dirs,
    python_resume_command,
    save_config,
    save_training_checkpoint,
    wall_time_exceeded,
    write_complete_file,
    write_resume_files,
)
from helpers.utils import NAME_TO_WIDTH, exp_warmup_linear_down, mixup
from models.dymn.dy_block import DY_Block
from models.dymn.model import get_model as get_dymn
from models.mn.block_types import InvertedResidual
from models.mn.model import get_model as get_mobilenet
from models.preprocess import AugmentMelSTFT
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
import perforatedbp


TRAIN_GENERATOR_OFFSET = 0
VAL_GENERATOR_OFFSET = 1


@contextmanager
def pai_working_directory(run_dir):
    """Run PAI filesystem operations from the canonical run directory.

    PAI treats `save_name` as a run-name/path fragment rather than a plain
    output directory. Keeping cwd at the run root and passing `pai/system`
    prevents PAI from appending absolute paths back under its own save folder.
    """
    previous_cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


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
    """Return ``(perforate_classes, perforate_names, track_classes,
    track_names, perforate_names_override)``.

    ``perforate_*``: classes wrapped as ``PAINeuronModule`` (get dendrites).
    ``track_*``:     classes wrapped as ``TrackedNeuronModule`` (no dendrites,
                     walker stops here so interior layers are untouched).
    ``perforate_names_override``: if non-None, REPLACES PAI's default
    class-name perforate list (``Conv2d`` + ``Linear`` + ...) before
    anything else is appended. Used on MobileNet to restrict wrapping to
    ``Linear`` only so the walker descends through every IR/C2NA block.

    Both perforate/track are registered by class object AND by short class
    name to be robust against duplicate imports (identity check misses,
    name check hits).

    MobileNet (`mn*`): probe-driven "Linear-only" strategy. The class list
    is intentionally empty so the walker descends through every
    ``InvertedResidual`` and ``Conv2dNormActivation``. The override list
    ``["Linear"]`` is what actually pulls the trigger: PAI's walker finds
    the SE ``fc1``/``fc2`` Linears inside each IR's ``ConcurrentSEBlock``,
    plus both classifier Linears, and wraps those as ``PAINeuronModule``.
    Each ``Conv2dNormActivation`` is tracked by class name so PAI's walker
    wraps the whole block (inner conv/BN are not visited separately). See
    probes/mn04_as_plateau.csv for the gradient-magnitude evidence supporting
    this targeting.

    DyMN (`dymn*`): unchanged block-level strategy. ``DY_Block`` is
    perforated via ``DYBlockProcessor`` (multi-input forward needs
    `pre_d`/`post_*` routing, customization.md section 2.2).
    ``Conv2dNormActivation`` stays track-only.
    """
    family = model_name.lower()
    if family.startswith("dymn"):
        perforate_classes = [DY_Block]
        perforate_names = ["DY_Block"]
        track_classes = [Conv2dNormActivation]
        track_names = ["Conv2dNormActivation"]
        perforate_names_override = None  # keep PAI default + DY_Block
    else:
        # MobileNet: empty class list; the override below is what selects.
        perforate_classes = []
        perforate_names = []
        track_classes = [Conv2dNormActivation]
        track_names = ["Conv2dNormActivation"]
        perforate_names_override = ("Linear",)
    return (
        perforate_classes,
        perforate_names,
        track_classes,
        track_names,
        perforate_names_override,
    )


def _set_pai_config_list(field_name: str, values) -> None:
    """Set a PAI config list exactly, falling back to append on old APIs."""
    setter = getattr(GPA.pc, f"set_{field_name}", None)
    if callable(setter):
        setter(list(values))
        return
    getter = getattr(GPA.pc, f"get_{field_name}", None)
    if callable(getter):
        current = getter()
        if isinstance(current, list):
            current.clear()
            current.extend(values)
            return
    appender = getattr(GPA.pc, f"append_{field_name}", None)
    if callable(appender):
        appender(list(values))


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
    # Perforation strategy differs by backbone:
    #   - MobileNet:   REPLACE default class-name list with ["Linear"] so the
    #                  walker descends through IRs (C2NA blocks are track-
    #                  wrapped as a whole) and only wraps the SE fc1/fc2
    #                  Linears and classifier Linears it finds.
    #   - DyMN:        append `DY_Block` to the default list for block-level
    #                  dendrites on the dynamic backbone.
    #
    # We register classes both by object AND by class-name string: the
    # class-identity check in PAI can miss when the same class is imported
    # via two paths (e.g. torchvision re-exports), but the short-name string
    # match still hits.
    (
        perforate_classes,
        perforate_names,
        track_classes,
        track_names,
        perforate_names_override,
    ) = _select_pai_blocks_for_model(cfg.model.model_name)

    # The REPLACE step (if any) must come first so subsequent appends stack
    # on top of a clean slate. A config-level override beats the per-family
    # default.
    effective_override = (
        pai.perforate_names_override
        if pai.perforate_names_override is not None
        else perforate_names_override
    )
    if effective_override is not None:
        GPA.pc.set_module_names_to_perforate(list(effective_override))
    if perforate_classes:
        GPA.pc.append_modules_to_perforate(perforate_classes)
    if perforate_names:
        GPA.pc.append_module_names_to_perforate(perforate_names)

    # Optional extras by class name
    if pai.extra_module_names_to_perforate:
        GPA.pc.append_module_names_to_perforate(list(pai.extra_module_names_to_perforate))

    # Optional: restrict conversion to specific dotted module ids only.
    # When set, PAI only perforates these exact ids.
    _set_pai_config_list("module_ids_to_perforate", pai.perforate_module_ids or [])

    # Skip specific dotted module ids (stem etc.) -- tracked means
    # "don't add dendrites here, but do keep gradients flowing through".
    _set_pai_config_list("module_ids_to_track", pai.track_module_ids or [])

    # Track-only classes: PAI wraps these as `TrackedNeuronModule` (no
    # dendrites added, and the walker STOPS at them so the inner layers are
    # never flagged as "unwrapped norm"). We register both by class object
    # and short name for robustness against duplicate imports.
    _set_pai_config_list("modules_to_track", track_classes)
    all_track_names = list(track_names) + list(pai.track_module_names)
    _set_pai_config_list("module_names_to_track", all_track_names)

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
    GPA.pc.set_improvement_threshold(list(pai.improvement_threshold))
    GPA.pc.set_pai_improvement_threshold(pai.pai_improvement_threshold)
    GPA.pc.set_pai_improvement_threshold_raw(pai.pai_improvement_threshold_raw)
    # Apply validation raw last (same ordering as `_reapply_validation_thresholds`).
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
    Re-apply after `perforate_model` and after optional `load_system` (resume)
    so training always sees the values from ``cfg.pai``.
    """
    pai = cfg.pai
    GPA.pc.set_improvement_threshold(list(pai.improvement_threshold))
    GPA.pc.set_pai_improvement_threshold(pai.pai_improvement_threshold)
    GPA.pc.set_pai_improvement_threshold_raw(pai.pai_improvement_threshold_raw)
    GPA.pc.set_improvement_threshold_raw(pai.improvement_threshold_raw)


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


def _maybe_resume_from_checkpoint(model, cfg: Config):
    """Load a prior PAI save on top of the freshly-perforated model.

    Called AFTER ``_perforate_model`` so the wrapper tree is already in
    place -- ``UPA.load_system`` only populates weights + tracker state and
    requires an identical wrapper layout.

    Why this works with a *different* P-training-mode setting
    --------------------------------------------------------
    ``load_system`` restores:
      - network ``state_dict`` (neuron weights + BN stats)
      - PAI tracker ``member_vars`` via the serialized ``tracker_string``
        buffer: score history, ``switch_epochs``, ``num_epochs_run``,
        ``mode``, ``num_dendrites_added``, ...
    ``load_system`` does NOT restore any field on ``GPA.pc``. Everything
    configured by ``_configure_pai`` (switch_mode, p_epochs_to_switch,
    thresholds, ...) stays in effect. So: set the new dendrite-phase
    schedule via ``PAIConfig``, perforate, load, and the new schedule
    drives the next cycle.

    Flags
    -----
    ``cfg.pai.resume_from_folder`` is the directory containing
    ``{resume_checkpoint}.pt`` (and the companion ``_pai.pt`` that
    ``save_system`` emits for ``save_pai_net``). ``None`` disables the
    whole codepath.

    ``load_from_manual_save=True`` suppresses PAI's internal
    ``start_epoch`` call inside ``load_system`` so the epoch counter
    doesn't tick before the first training iteration (PAI's own
    ``add_validation_score`` will advance it naturally).

    Returns the model (possibly a new instance after load).
    """
    pai = cfg.pai
    folder = pai.resume_from_folder
    if not folder:
        return model

    name = pai.resume_checkpoint
    ckpt_path = os.path.join(folder, f"{name}.pt")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"[PAI resume] Checkpoint not found: {ckpt_path}. "
            f"Set cfg.pai.resume_from_folder to a directory containing "
            f"'{name}.pt' (PAI save_system format) or clear resume_from_folder."
        )

    mv_before = dict(GPA.pai_tracker.member_vars) if hasattr(GPA.pai_tracker, "member_vars") else {}
    print(f"[PAI resume] Loading '{name}.pt' from '{folder}' "
          f"(load_from_manual_save=True)")
    model = UPA.load_system(
        model,
        folder,
        name,
        load_from_restart=False,
        switch_call=False,
        load_from_manual_save=True,
    )

    # PAI tracker serialized state is now in place. Apply any requested
    # post-load surgery so the NEW PAI schedule can take over cleanly.
    mv = GPA.pai_tracker.member_vars
    if pai.resume_force_neuron_mode:
        # Force "n" so the new switch_mode / p_epochs_to_switch / plateau
        # thresholds evaluate against a fresh neuron phase starting now.
        # load_system already zeroed current_best_validation_score and
        # set epoch_last_improved = num_epochs_run; we additionally slam
        # mode back to "n" in case the saved tracker was already flipped.
        mv["mode"] = "n"
        mv["epoch_last_improved"] = mv.get("num_epochs_run", 0)
        mv["current_best_validation_score"] = 0
        print("[PAI resume] Forced tracker.mode='n' (resume_force_neuron_mode=True)")

    for key, value in (pai.resume_tracker_overrides or {}).items():
        mv[key] = value
        print(f"[PAI resume] tracker.member_vars[{key!r}] <- {value!r}")

    # Short diff of a few headline tracker fields so the log shows the
    # *effective* starting point of training after the load.
    headline_keys = (
        "mode",
        "num_epochs_run",
        "num_dendrites_added",
        "epoch_last_improved",
        "current_best_validation_score",
        "switch_epochs",
    )
    print("[PAI resume] Tracker state after load:")
    for k in headline_keys:
        print(f"  {k:35s} before={mv_before.get(k, '<absent>')!r:>20}  "
              f"after={mv.get(k, '<absent>')!r}")

    return model


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


def configure_pai(cfg: Config, save_name: str) -> None:
    _configure_pai(cfg, save_name)


def build_mel(cfg: Config, device):
    pp = cfg.preprocess
    return AugmentMelSTFT(
        n_mels=pp.n_mels,
        sr=pp.resample_rate,
        win_length=pp.window_size,
        hopsize=pp.hop_size,
        n_fft=pp.n_fft,
        freqm=pp.freqm,
        timem=pp.timem,
        fmin=pp.fmin,
        fmax=pp.fmax,
        fmin_aug_range=pp.fmin_aug_range,
        fmax_aug_range=pp.fmax_aug_range,
    ).to(device)


def build_model(cfg: Config):
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
    return model, width


def initialize_pai_model(model, cfg: Config, pai_save_name: str):
    configure_pai(cfg, save_name=pai_save_name)
    model = _perforate_model(
        model,
        save_name=pai_save_name,
        maximizing_score=True,  # mAP is higher-is-better.
    )
    model = _maybe_resume_from_checkpoint(model, cfg)
    _report_dendrite_targets(model)
    _reapply_validation_thresholds(cfg)
    return model


def build_train_loader(cfg: Config):
    pp = cfg.preprocess
    dc = cfg.data
    return DataLoader(
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
        generator=make_generator(cfg.seed, TRAIN_GENERATOR_OFFSET),
    )


def build_val_loader(cfg: Config):
    pp = cfg.preprocess
    dc = cfg.data
    return DataLoader(
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
        generator=make_generator(cfg.seed, VAL_GENERATOR_OFFSET),
    )


def _pai_scheduler_args(cfg: Config, register_scheduler: bool = True):
    oc = cfg.optim
    lr_lambda = None
    if oc.scheduler_name == "plateau":
        if register_scheduler:
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        sched_args = {
            'mode': oc.scheduler_mode,
            'patience': oc.scheduler_patience,
            'factor': oc.scheduler_factor,
            'cooldown': oc.scheduler_cooldown,
            'min_lr': oc.scheduler_min_lr,
        }
    elif oc.scheduler_name == "lambda_warmup":
        if register_scheduler:
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.LambdaLR)
        # PAI may rebuild this scheduler; the real warmup is applied manually.
        sched_args = {'lr_lambda': (lambda _e: 1.0)}
        lr_lambda = exp_warmup_linear_down(
            oc.warm_up_len,
            oc.ramp_down_len,
            oc.ramp_down_start,
            oc.last_lr_value,
        )
    else:
        raise ValueError(
            f"Unknown cfg.optim.scheduler_name={oc.scheduler_name!r}; "
            "expected 'plateau' or 'lambda_warmup'."
        )
    return sched_args, lr_lambda


def setup_pai_optimizer(model, cfg: Config, register_with_tracker: bool = True):
    oc = cfg.optim
    if register_with_tracker:
        GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    optim_args = {'params': model.parameters(), 'lr': oc.lr, 'weight_decay': oc.weight_decay}
    sched_args, lr_lambda = _pai_scheduler_args(
        cfg,
        register_scheduler=register_with_tracker,
    )
    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
    return optimizer, sched_args, lr_lambda


def log_epoch_metrics(train_loss, train_mAP, train_ROC, mAP, ROC, val_loss, learning_rate):
    wandb.log({
        "train_loss": train_loss,
        "train_mAP": train_mAP,
        "train_ROC": train_ROC,
        "mAP": mAP,
        "ROC": ROC,
        "val_loss": val_loss,
        "learning_rate": learning_rate,
    })


def save_rolling_checkpoint(model, name, width, epoch, mAP):
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
    return name


def save_pai_system_checkpoint(model, pai_save_dir: str, tag: str = "latest") -> None:
    """Best-effort manual PAI save for phase boundaries.

    PAI versions have had slightly different `save_system` signatures. The
    existing resume path is authoritative, so try the common call shapes and
    leave a clear warning if this install relies only on PAI's automatic saves.
    """
    save_fn = getattr(UPA, "save_system", None)
    if save_fn is None:
        print("[PAI save] UPA.save_system is unavailable; relying on PAI auto-saves.")
        return
    attempts = (
        lambda: save_fn(model, pai_save_dir, tag, True),
        lambda: save_fn(model, pai_save_dir, tag),
        lambda: save_fn(model, tag, pai_save_dir),
        lambda: save_fn(model, tag),
    )
    last_error = None
    for attempt in attempts:
        try:
            attempt()
            print(f"[PAI save] Wrote PAI system checkpoint '{tag}' in {pai_save_dir}.")
            return
        except TypeError as exc:
            last_error = exc
        except Exception as exc:
            print(f"[PAI save] Warning: failed to save '{tag}': {exc}")
            return
    print(f"[PAI save] Warning: could not call UPA.save_system: {last_error}")


def sync_perforated_outputs(cfg: Config, run_paths) -> None:
    if not cfg.runtime.sync_pai_saves:
        return
    run_dir = run_paths["run"].resolve()
    mirror_dir = run_paths["pai"] / "mirrored_pai_outputs"
    mirror_dir.mkdir(parents=True, exist_ok=True)
    possible_sources = [
        os.path.join("pai", cfg.experiment_name),
        f"{cfg.experiment_name}_graphs",
        f"{cfg.experiment_name}_models",
        "graphs",
    ]
    for source_name in possible_sources:
        source = os.path.abspath(source_name)
        source_path = os.path.realpath(source)
        try:
            source_obj = os.path.abspath(source_path)
            source_path_obj = os.path.realpath(source_obj)
            if not os.path.exists(source_path_obj):
                continue
            source_resolved = os.path.realpath(source_path_obj)
            if str(source_resolved).startswith(str(run_dir)):
                continue
            dest = mirror_dir / os.path.basename(source_resolved)
            if os.path.isdir(source_resolved):
                shutil.copytree(source_resolved, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(source_resolved, dest)
        except Exception as exc:
            print(f"[PAI sync] Warning: failed to mirror {source_name}: {exc}")


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
def train(cfg: Config = default_config):
    seed_everything(cfg.seed)

    runtime = cfg.runtime
    save_name = runtime.save_name or cfg.experiment_name
    runtime.save_name = save_name
    run_start_time = time.time()
    run_paths = create_run_dirs(
        save_name,
        runtime.output_dir,
        allow_overwrite=runtime.allow_overwrite,
        resume_pai=runtime.resume_pai or bool(cfg.pai.resume_from_folder),
    )
    pai_save_dir = str(run_paths["pai_system"])
    pai_save_name = os.path.join("pai", "system")
    if runtime.resume_pai:
        cfg.pai.resume_from_folder = pai_save_name
        cfg.pai.resume_checkpoint = runtime.pai_resume_tag
    save_config(cfg, run_paths)
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

    mel = build_mel(cfg, device)
    model, width = build_model(cfg)
    with pai_working_directory(run_paths["run"]):
        model = initialize_pai_model(model, cfg, pai_save_name)

    # Optional full post-perforation module tree: lets the user visually
    # confirm which blocks became `PAINeuronModule` / `TrackedNeuronModule`
    # and which stayed as plain `nn.Module`s.
    if cfg.pai.print_model_layout:
        print("[PAI] Model structure after perforation:")
        print(model)

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"param_count: {total_params}")
    print(f"trainable_param_count: {trainable_params}")

    dl = build_train_loader(cfg)
    valid_dl = build_val_loader(cfg)

    # ---------------- PAI-managed optimizer + scheduler --------------------
    # Two supported schedules (selected via cfg.optim.scheduler_name):
    #
    #   "plateau"       -> PAI-managed ReduceLROnPlateau. PAI's
    #                      add_validation_score() steps it with the mAP
    #                      metric each epoch. Used by the original PAI
    #                      baseline and works with switch_mode="fixed".
    #
    #   "lambda_warmup" -> mirror ex_fsd50k.py: exp warmup + linear ramp-down
    #                      + constant floor. Implemented by manually writing
    #                      lr * lr_lambda(epoch) into optimizer.param_groups
    #                      at the START of every epoch (see training loop).
    #                      A no-op LambdaLR(lr_lambda=1.0) is still registered
    #                      with the PAI tracker so optimizer-rebuild-on-
    #                      restructure (which re-instantiates the scheduler
    #                      class) keeps working without surprising PAI's
    #                      internal scheduler.step() calls -- LambdaLR.step()
    #                      is a no-op on a constant lambda, and PAI won't
    #                      pass a metric value to a non-plateau scheduler
    #                      class.
    oc = cfg.optim
    optimizer, _, lr_lambda = setup_pai_optimizer(model, cfg)

    # ---------------- Training loop ----------------------------------------
    name = None
    mAP, ROC, val_loss = float('NaN'), float('NaN'), float('NaN')
    train_loss_epoch = float('NaN')
    train_mAP, train_ROC = float('NaN'), float('NaN')
    best_val_mAP, best_val_epoch = float("-inf"), -1

    epoch = -1
    # schedule_epoch drives lr_lambda. In per-cycle mode it is reset to 0
    # every time PAI restructures (see restructured-branch below), so each
    # PAI cycle gets its own warmup+rampdown. In global mode it follows the
    # absolute epoch counter.
    schedule_epoch = 0
    consecutive_epochs_in_p = 0
    while True:
        epoch += 1
        mel.train()
        model.train()
        train_loss_list = []
        train_targets, train_outputs = [], []

        # Apply the warmup+linear-down schedule directly to the optimizer
        # each epoch, mirroring ex_fsd50k.py's LambdaLR(exp_warmup_linear_down).
        # Done at epoch-start so it also overrides any LR change PAI made
        # during the prior epoch's add_validation_score() / restructure.
        if lr_lambda is not None:
            sched_idx = schedule_epoch if oc.lambda_warmup_per_cycle else epoch
            scale = float(lr_lambda(sched_idx))
            for pg in optimizer.param_groups:
                pg['lr'] = oc.lr * scale
        schedule_epoch += 1

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
        # train-val-gap diagnostics, so don't rename it. Keep this on the
        # same 0-1 scale as add_validation_score(...) below.
        GPA.pai_tracker.add_extra_score(train_mAP, "train")
        # Auxiliary trace: negated train loss so "higher = better" still
        # holds on the extra-score axis (PAI treats extras as informational).
        GPA.pai_tracker.add_extra_score(-train_loss_epoch, "NegTrainLoss")
        model.to(device)

        mAP, ROC, val_loss, _ = _test(model, mel, valid_dl, device)

        if mAP > best_val_mAP:
            best_val_mAP = mAP
            best_val_epoch = epoch
            improved = True
        else:
            improved = False

        # PAI prints node-improvement / switch diagnostics inside add_validation_score.
        # Raise verbosity only for this block, then restore the configured default.
        GPA.pc.set_verbose(cfg.pai.verbose)
        try:
            # PAI drives switching + "training_complete" off validation mAP.
            with pai_working_directory(run_paths["run"]):
                model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                    mAP, model
                )
        finally:
            GPA.pc.set_verbose(cfg.pai.verbose)
        model.to(device)
        mv = GPA.pai_tracker.member_vars
        pai_mode = mv.get("mode")
        num_dendrites_added = mv.get("num_dendrites_added")
        if mv.get("mode") == "p":
            consecutive_epochs_in_p += 1
        else:
            consecutive_epochs_in_p = 0
        cap_p = cfg.pai.max_consecutive_epochs_in_p_mode
        if cap_p is not None and consecutive_epochs_in_p >= cap_p:
            print(
                f"[PAI] Stopping: max_consecutive_epochs_in_p_mode={cap_p} reached "
                f"(tracker mode={mv.get('mode')!r}, script epoch={epoch + 1})."
            )
            break

        log_epoch_metrics(
            train_loss_epoch,
            train_mAP,
            train_ROC,
            mAP,
            ROC,
            val_loss,
            optimizer.param_groups[0]["lr"],
        )
        name = save_rolling_checkpoint(model, name, width, epoch, mAP)
        save_training_checkpoint(
            model,
            optimizer,
            None,
            epoch,
            best_val_mAP,
            best_val_epoch,
            schedule_epoch,
            {
                "train_mAP": train_mAP,
                "train_ROC": train_ROC,
                "mAP": mAP,
                "ROC": ROC,
                "val_loss": val_loss,
            },
            cfg,
            run_paths,
            "last.pt",
            is_perforated=True,
        )
        if improved:
            save_training_checkpoint(
                model,
                optimizer,
                None,
                epoch,
                best_val_mAP,
                best_val_epoch,
                schedule_epoch,
                {
                    "train_mAP": train_mAP,
                    "train_ROC": train_ROC,
                    "mAP": mAP,
                    "ROC": ROC,
                    "val_loss": val_loss,
                },
                cfg,
                run_paths,
                "best.pt",
                is_perforated=True,
            )
        append_metrics_row(
            run_paths,
            {
                "epoch": epoch,
                "train_loss": train_loss_epoch,
                "train_mAP": train_mAP,
                "train_ROC": train_ROC,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "mAP": mAP,
                "ROC": ROC,
                "val_loss": val_loss,
                "best_val_mAP": best_val_mAP,
                "best_val_epoch": best_val_epoch,
                "pai_mode": pai_mode,
                "num_dendrites_added": num_dendrites_added,
                "restructured": restructured,
                "training_complete": training_complete,
            },
        )

        if training_complete:
            with pai_working_directory(run_paths["run"]):
                save_pai_system_checkpoint(model, pai_save_name, "latest")
            sync_perforated_outputs(cfg, run_paths)
            write_complete_file(
                run_paths,
                save_name,
                best_val_mAP,
                best_val_epoch,
                is_perforated=True,
                phase_index=runtime.phase_index,
            )
            print(
                f"PAI training complete at epoch {epoch}. "
                f"Best validation mAP (this run): {best_val_mAP:.4f} "
                f"(epoch {best_val_epoch + 1}); "
                f"last epoch mAP: {mAP:.4f}"
            )
            break
        elif restructured:
            with pai_working_directory(run_paths["run"]):
                save_pai_system_checkpoint(model, pai_save_name, "latest")
            sync_perforated_outputs(cfg, run_paths)
            if runtime.exit_on_pai_restructure:
                resume_command = python_resume_command(
                    cfg,
                    {
                        "save_name": save_name,
                        "output_dir": runtime.output_dir,
                        "resume_pai": True,
                        "pai_resume_tag": "latest",
                        "exit_on_pai_restructure": True,
                        "max_wall_minutes": runtime.max_wall_minutes,
                        "sync_pai_saves": runtime.sync_pai_saves,
                        "phase_index": runtime.phase_index + 1,
                    },
                )
                write_resume_files(
                    run_paths,
                    save_name,
                    "pai_restructured",
                    EXIT_PAI_RESTRUCTURED,
                    resume_command,
                    is_perforated=True,
                    phase_index=runtime.phase_index,
                )
                print("Exiting after PAI restructure.")
                sys.exit(EXIT_PAI_RESTRUCTURED)
            # Topology changed: optimizer state is stale, rebuild with the
            # same scheduler family we configured at startup.
            optimizer, _, _ = setup_pai_optimizer(model, cfg, register_with_tracker=False)
            scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
            # Fresh PAI cycle on a new architecture: restart the warmup+
            # rampdown curve from 0 so the new (larger) model gets the same
            # careful warm start the initial neuron cycle did.
            if lr_lambda is not None and oc.lambda_warmup_per_cycle:
                schedule_epoch = 0

        if wall_time_exceeded(runtime.max_wall_minutes, run_start_time):
            with pai_working_directory(run_paths["run"]):
                save_pai_system_checkpoint(model, pai_save_name, "latest")
            sync_perforated_outputs(cfg, run_paths)
            resume_command = python_resume_command(
                cfg,
                {
                    "save_name": save_name,
                    "output_dir": runtime.output_dir,
                    "resume_pai": True,
                    "pai_resume_tag": "latest",
                    "exit_on_pai_restructure": runtime.exit_on_pai_restructure,
                    "max_wall_minutes": runtime.max_wall_minutes,
                    "sync_pai_saves": runtime.sync_pai_saves,
                    "phase_index": runtime.phase_index + 1,
                },
            )
            write_resume_files(
                run_paths,
                save_name,
                "wall_time_budget",
                EXIT_WALL_TIME,
                resume_command,
                is_perforated=True,
                phase_index=runtime.phase_index,
            )
            print("Exiting after wall-time budget.")
            sys.exit(EXIT_WALL_TIME)


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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune FSD50K with Perforated AI using config defaults.",
    )
    parser.add_argument(
        "--model",
        "-model",
        dest="model_name",
        default=None,
        help="Override cfg.model.model_name, e.g. mn10_as or dymn04_as.",
    )
    parser.add_argument(
        "--preset",
        "-preset",
        dest="pai_preset",
        default=None,
        help="Override cfg.pai_preset. Use --list-presets to see valid names.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Print canonical PAI presets and legacy aliases, then exit.",
    )
    parser.add_argument("--save-name", default=None)
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--resume-pai", action="store_true")
    parser.add_argument("--pai-resume-tag", default="latest")
    parser.add_argument("--exit-on-pai-restructure", action="store_true")
    parser.add_argument("--max-wall-minutes", type=float, default=None)
    parser.add_argument("--sync-pai-saves", action="store_true")
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.list_presets:
        _print_available_presets()
        return

    cfg = copy.deepcopy(default_config)
    apply_runtime_overrides(
        cfg,
        model_name=args.model_name,
        pai_preset=args.pai_preset,
        save_name=args.save_name,
        output_dir=args.output_dir,
        resume_pai=args.resume_pai,
        pai_resume_tag=args.pai_resume_tag,
        exit_on_pai_restructure=args.exit_on_pai_restructure,
        max_wall_minutes=args.max_wall_minutes,
        sync_pai_saves=args.sync_pai_saves,
        phase_index=args.phase_index,
        allow_overwrite=args.allow_overwrite,
    )
    train(cfg)


if __name__ == '__main__':
    main()
