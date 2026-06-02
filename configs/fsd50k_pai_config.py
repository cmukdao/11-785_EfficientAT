"""
Configuration for FSD50K fine-tuning with Perforated AI Dendrites.

Usage:

    from configs.fsd50k_pai_config import config
    train(config)                     # defaults (Linear-only, S0)

    # or customise a sub-config:
    cfg = Config(model=ModelConfig(model_name="mn10_as"))
    cfg.data.batch_size = 32
    train(cfg)

    # or set a named PAI strategy on the root config (no separate factory call):
    cfg = Config(pai_preset="all_c2na_plus_logit")
    train(cfg)

    cfg = Config(pai_preset="late_se_blocks_only")
    train(cfg)

    cfg = Config(pai_preset="prelogit_plus_late_se_blocks")
    train(cfg)

    cfg = Config(pai_preset="probe_c2na_stages_7_13")
    train(cfg)

    cfg = Config(pai_preset="probe_late_backbone_head")
    train(cfg)

    # or RESUME an existing run's pre-first-switch snapshot with a new
    # dendrite-phase schedule (keeps perforation layout, swaps switch_mode
    # / p_epochs_to_switch):
    cfg = Config(
        pai_preset="classifier_head_plus_late_se_linears",
        pai=PAIConfig(
            switch_mode="history",
            p_epochs_to_switch=8,   # new value -- drives next dendrite phase
            resume_from_folder=(
                "pai/FSD50K-mn04_as-pai-fixed-classifier-head-plus-late-se-linears"
            ),
            resume_checkpoint="beforeSwitch_0",   # default; omit to use it
            resume_force_neuron_mode=True,        # restart plateau clock fresh
        ),
        experiment_name=(
            "FSD50K-mn04_as-pai-fixed-classifier-head-plus-late-se-linears-resume-p8"
        ),
    )
    train(cfg)

Derived fields (`experiment_name`, `wandb.name/group/notes`) are filled
in by `Config.__post_init__` from `model.model_name` + `pai.switch_mode`
unless you pass explicit values. Non-default ``pai_preset`` patches
``PAIConfig`` and picks a default ``experiment_name`` slug unless you set
it yourself.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple

# PAI strategy presets; edit ``Config.pai_preset`` or instantiate ``Config(...)``.
PAI_PRESET_DEFAULT = "default"
PAI_PRESET_ALL_C2NA_PLUS_LOGIT = "all_c2na_plus_logit"
PAI_PRESET_LOGIT_ONLY = "logit_only"
PAI_PRESET_PRELOGIT_ONLY = "prelogit_only"
PAI_PRESET_CLASSIFIER_HEAD_ONLY = "classifier_head_only"
PAI_PRESET_CLASSIFIER_HEAD_LATE_SE_LINEARS = "classifier_head_plus_late_se_linears"
PAI_PRESET_PRELOGIT_LATE_SE_LINEARS = "prelogit_plus_late_se_linears"
PAI_PRESET_LATE_SE_BLOCKS_ONLY = "late_se_blocks_only"
PAI_PRESET_PRELOGIT_LATE_SE_BLOCKS = "prelogit_plus_late_se_blocks"
PAI_PRESET_PROBE_C2NA_STAGES_7_13 = "probe_c2na_stages_7_13"
PAI_PRESET_PROBE_LATE_BACKBONE_HEAD = "probe_late_backbone_head"

# Legacy aliases remain valid for old notebooks/scripts.
PAI_PRESET_C2NA_PLUS_LOGIT = PAI_PRESET_ALL_C2NA_PLUS_LOGIT
PAI_PRESET_CLASSIFIER5_ONLY = PAI_PRESET_LOGIT_ONLY
PAI_PRESET_CLASSIFIER2_ONLY = PAI_PRESET_PRELOGIT_ONLY
PAI_PRESET_CLASSIFIERS_ONLY = PAI_PRESET_CLASSIFIER_HEAD_ONLY
PAI_PRESET_HEAD_LATE_SE = PAI_PRESET_CLASSIFIER_HEAD_LATE_SE_LINEARS
PAI_PRESET_CLASSIFIER2_LATE_SE = PAI_PRESET_PRELOGIT_LATE_SE_LINEARS
PAI_PRESET_LATE_SE_BLOCK = PAI_PRESET_LATE_SE_BLOCKS_ONLY
PAI_PRESET_CLASSIFIER2_LATE_SE_BLOCK = PAI_PRESET_PRELOGIT_LATE_SE_BLOCKS
PAI_PRESET_FEATURES_7_13_C2NA = PAI_PRESET_PROBE_C2NA_STAGES_7_13
PAI_PRESET_PROBE_BACKBONE_HEAD = PAI_PRESET_PROBE_LATE_BACKBONE_HEAD

_PROBE_BACKBONE_HEAD_MODULE_IDS: Tuple[str, ...] = (
    ".features.14.block.2",
    ".features.14.block.3",
    ".features.15.block.2",
    ".features.15.block.3",
    ".features.16",
)

_MID_LATE_SE_BLOCK2_TRACK_IDS: Tuple[str, ...] = tuple(
    f".features.{b}.block.2" for b in (11, 12, 13)
)

_FEATURES_7_13_C2NA_MODULE_IDS: Tuple[str, ...] = (
    ".features.7.block.1",
    ".features.7.block.2",
    ".features.13.block.1",
    ".features.13.block.3",
)

_HEAD_LATE_SE_MODULE_IDS: Tuple[str, ...] = (
    ".classifier.2",
    ".classifier.5",
) + tuple(
    f".features.{b}.block.2.conc_se_layers.0.fc1"
    for b in (11, 12, 13, 14, 15)
) + tuple(
    f".features.{b}.block.2.conc_se_layers.0.fc2"
    for b in (12, 14, 15)
)

_LATE_SE_ONLY_MODULE_IDS: Tuple[str, ...] = tuple(
    f".features.{b}.block.2.conc_se_layers.0.fc1"
    for b in (11, 12, 13, 14, 15)
) + tuple(
    f".features.{b}.block.2.conc_se_layers.0.fc2"
    for b in (12, 14, 15)
)

_LATE_CONCURRENT_SE_BLOCK_IDS: Tuple[str, ...] = tuple(
    f".features.{b}.block.2" for b in (11, 12, 13, 14, 15)
)

_EARLY_SE_BLOCK2_TRACK_IDS: Tuple[str, ...] = (
    ".features.4.block.2",
    ".features.5.block.2",
    ".features.6.block.2",
    ".features.10.block.2",
)

_PRESET_PAI_PATCHES: Dict[str, dict] = {
    PAI_PRESET_ALL_C2NA_PLUS_LOGIT: {
        "perforate_names_override": ("Conv2dNormActivation",),
        "perforate_module_ids": [".classifier.5"],
        "track_module_names": ("Linear",),
        "track_module_ids": (),
    },
    PAI_PRESET_PROBE_C2NA_STAGES_7_13: {
        "perforate_names_override": ("Conv2dNormActivation",),
        "perforate_module_ids": list(_FEATURES_7_13_C2NA_MODULE_IDS),
        "track_module_names": ("Linear",),
        "track_module_ids": (),
    },
    PAI_PRESET_PROBE_LATE_BACKBONE_HEAD: {
        # Whole ``.block.2`` SE modules by id; ``.block.3`` + ``features.16`` C2NA.
        "perforate_names_override": ("Linear", "Conv2dNormActivation"),
        "perforate_module_ids": list(_PROBE_BACKBONE_HEAD_MODULE_IDS),
        "track_module_names": ("Linear",),
        "track_module_ids": (
            ".features.0",
            ".in_c",
            *_EARLY_SE_BLOCK2_TRACK_IDS,
            *_MID_LATE_SE_BLOCK2_TRACK_IDS,
        ),
    },
    PAI_PRESET_LOGIT_ONLY: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": [".classifier.5"],
        "track_module_names": ("Linear",),
        "track_module_ids": (".features.0", ".in_c"),
    },
    PAI_PRESET_PRELOGIT_ONLY: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": [".classifier.2"],
        "track_module_names": ("Linear",),
        "track_module_ids": (".features.0", ".in_c"),
    },
    PAI_PRESET_CLASSIFIER_HEAD_ONLY: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": [".classifier.2", ".classifier.5"],
        "track_module_names": ("Linear",),
        "track_module_ids": (".features.0", ".in_c"),
    },
    PAI_PRESET_CLASSIFIER_HEAD_LATE_SE_LINEARS: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": list(_HEAD_LATE_SE_MODULE_IDS),
        "track_module_names": ("Linear",),
        "track_module_ids": (".features.0", ".in_c"),
    },
    PAI_PRESET_PRELOGIT_LATE_SE_LINEARS: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": [".classifier.2", *list(_LATE_SE_ONLY_MODULE_IDS)],
        "track_module_names": ("Linear",),
        "track_module_ids": (".features.0", ".in_c"),
    },
    PAI_PRESET_LATE_SE_BLOCKS_ONLY: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": list(_LATE_CONCURRENT_SE_BLOCK_IDS),
        "track_module_names": ("Linear",),
        "track_module_ids": (
            ".features.0",
            ".in_c",
            *_EARLY_SE_BLOCK2_TRACK_IDS,
        ),
    },
    PAI_PRESET_PRELOGIT_LATE_SE_BLOCKS: {
        "perforate_names_override": ("Linear",),
        "perforate_module_ids": [
            ".classifier.2",
            *list(_LATE_CONCURRENT_SE_BLOCK_IDS),
        ],
        "track_module_names": ("Linear",),
        "track_module_ids": (
            ".features.0",
            ".in_c",
            *_EARLY_SE_BLOCK2_TRACK_IDS,
        ),
    },
}

# Wandb / save slug fragment when ``experiment_name`` is left empty.
_PRESET_EXPERIMENT_SUFFIX: Dict[str, str] = {
    PAI_PRESET_ALL_C2NA_PLUS_LOGIT: "all-c2na-plus-logit",
    PAI_PRESET_PROBE_C2NA_STAGES_7_13: "probe-c2na-stages-7-13",
    PAI_PRESET_PROBE_LATE_BACKBONE_HEAD: "probe-late-backbone-head",
    PAI_PRESET_LOGIT_ONLY: "logit-only",
    PAI_PRESET_PRELOGIT_ONLY: "prelogit-only",
    PAI_PRESET_CLASSIFIER_HEAD_ONLY: "classifier-head-only",
    PAI_PRESET_CLASSIFIER_HEAD_LATE_SE_LINEARS: "classifier-head-plus-late-se-linears",
    PAI_PRESET_PRELOGIT_LATE_SE_LINEARS: "prelogit-plus-late-se-linears",
    PAI_PRESET_LATE_SE_BLOCKS_ONLY: "late-se-blocks-only",
    PAI_PRESET_PRELOGIT_LATE_SE_BLOCKS: "prelogit-plus-late-se-blocks",
}

_PRESET_ALIASES: Dict[str, str] = {
    "c2na_plus_logit": PAI_PRESET_ALL_C2NA_PLUS_LOGIT,
    "classifier5_only": PAI_PRESET_LOGIT_ONLY,
    "classifier2_only": PAI_PRESET_PRELOGIT_ONLY,
    "classifiers_only": PAI_PRESET_CLASSIFIER_HEAD_ONLY,
    "head_late_se": PAI_PRESET_CLASSIFIER_HEAD_LATE_SE_LINEARS,
    "classifier2_late_se": PAI_PRESET_PRELOGIT_LATE_SE_LINEARS,
    "late_se_block": PAI_PRESET_LATE_SE_BLOCKS_ONLY,
    "classifier2_late_se_block": PAI_PRESET_PRELOGIT_LATE_SE_BLOCKS,
    "features_7_13_c2na": PAI_PRESET_PROBE_C2NA_STAGES_7_13,
    "probe_backbone_head": PAI_PRESET_PROBE_LATE_BACKBONE_HEAD,
}


def _normalize_pai_preset(preset: str) -> str:
    key = (preset or PAI_PRESET_DEFAULT).strip()
    return _PRESET_ALIASES.get(key, key)


def get_pai_preset_names() -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """Return canonical preset names and legacy aliases for CLI display."""
    return tuple(sorted(_PRESET_PAI_PATCHES)), dict(sorted(_PRESET_ALIASES.items()))


def _apply_pai_preset(cfg: "Config") -> None:
    """Merge preset-specific ``PAIConfig`` fields onto ``cfg.pai``."""
    key = _normalize_pai_preset(cfg.pai_preset)
    cfg.pai_preset = key
    if key == PAI_PRESET_DEFAULT:
        return
    patches = _PRESET_PAI_PATCHES.get(key)
    if patches is None:
        choices = sorted({PAI_PRESET_DEFAULT, *_PRESET_PAI_PATCHES, *_PRESET_ALIASES})
        raise ValueError(
            f"Unknown Config.pai_preset={key!r}. Valid choices: {choices}"
        )
    cfg.pai = replace(cfg.pai, **patches)


def _preset_suffix(pai_preset: str) -> str:
    return _PRESET_EXPERIMENT_SUFFIX.get(
        pai_preset,
        pai_preset.replace("_", "-"),
    )


def _derive_run_identity(cfg: "Config", force: bool = False) -> None:
    if force:
        cfg.experiment_name = ""
        cfg.wandb.name = ""
        cfg.wandb.group = ""
        cfg.wandb.notes = ""

    if not cfg.experiment_name:
        if cfg.pai_preset != PAI_PRESET_DEFAULT:
            cfg.experiment_name = (
                f"FSD50K-{cfg.model.model_name}-pai-"
                f"{cfg.pai.switch_mode}-{_preset_suffix(cfg.pai_preset)}"
            )
        else:
            cfg.experiment_name = (
                f"FSD50K-{cfg.model.model_name}-pai-{cfg.pai.switch_mode}"
            )

    slug = cfg.experiment_name
    if not cfg.wandb.name:
        cfg.wandb.name = slug
    if not cfg.wandb.group:
        cfg.wandb.group = slug
    if not cfg.wandb.notes:
        cfg.wandb.notes = (
            f"Fine-tune {cfg.model.model_name} on FSD50K with Perforated AI Dendrites."
        )


def apply_runtime_overrides(
    cfg: "Config",
    model_name: Optional[str] = None,
    pai_preset: Optional[str] = None,
    save_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    resume_pai: bool = False,
    pai_resume_tag: Optional[str] = None,
    exit_on_pai_restructure: Optional[bool] = None,
    max_wall_minutes: Optional[float] = None,
    sync_pai_saves: Optional[bool] = None,
    phase_index: Optional[int] = None,
    allow_overwrite: Optional[bool] = None,
) -> "Config":
    """Apply CLI overrides on top of a config object and refresh run names."""
    if model_name:
        cfg.model = replace(cfg.model, model_name=model_name)

    if pai_preset:
        defaults = PAIConfig()
        cfg.pai = replace(
            cfg.pai,
            perforate_names_override=defaults.perforate_names_override,
            perforate_module_ids=defaults.perforate_module_ids,
            track_module_names=defaults.track_module_names,
            track_module_ids=defaults.track_module_ids,
        )
        cfg.pai_preset = _normalize_pai_preset(pai_preset)
        _apply_pai_preset(cfg)

    if model_name or pai_preset:
        _derive_run_identity(cfg, force=True)

    if save_name:
        cfg.runtime.save_name = save_name
        cfg.experiment_name = save_name
        cfg.wandb.name = save_name
        cfg.wandb.group = save_name
    if output_dir is not None:
        cfg.runtime.output_dir = output_dir
    if resume_pai:
        cfg.runtime.resume_pai = True
    if pai_resume_tag is not None:
        cfg.runtime.pai_resume_tag = pai_resume_tag
    if exit_on_pai_restructure is not None:
        cfg.runtime.exit_on_pai_restructure = exit_on_pai_restructure
    if max_wall_minutes is not None:
        cfg.runtime.max_wall_minutes = max_wall_minutes
    if sync_pai_saves is not None:
        cfg.runtime.sync_pai_saves = sync_pai_saves
    if phase_index is not None:
        cfg.runtime.phase_index = phase_index
    if allow_overwrite is not None:
        cfg.runtime.allow_overwrite = allow_overwrite

    return cfg


# ============================================================================
# Perforated AI license credentials
# Must be set BEFORE `perforatedai` is imported anywhere in the process.
# ============================================================================
PAI_EMAIL = "PAIUser3.11.2026@perforatedai.com"
PAI_TOKEN = (
    "g3ZDrJNAmBlh/tAdFUkkedY+mUGqaueCPQXAtkVyNq405Fc9+20MhmEIDttx285EhfFqDHFtMRA20BWKpQaqai3wNxaOeCNvsNsF7Nn2nTmpFUmkHGvmVWGD5JI1uxdPdW6jeUmaQBKNUNcNKOumr1iDaQpFnCvFDDcSi3yUmZ5JoCd0c/lAyGmRXWQ+dInOXX6MdE4XVmv7DI8jW626pNLemX7ZMo4dEGikNuhyuiAD1IJYNYaJxUK0zaizx/Avq6QbmwQviPjXDNyBsTdVzJQzhAG6Zf2DlU9j29RnEaj0XGd4j3MJeqsrq0FeXKqEMShnBKO0oL69CP4icVTOpQ=="
)


@dataclass
class PreprocessConfig:
    resample_rate: int = 32000
    window_size: int = 800
    hop_size: int = 320
    n_fft: int = 1024
    n_mels: int = 128
    freqm: int = 0
    timem: int = 0
    fmin: int = 0
    fmax: Optional[int] = None
    fmin_aug_range: int = 10
    fmax_aug_range: int = 2000


@dataclass
class ModelConfig:
    pretrained: bool = True
    model_name: str = "mn04_as"
    pretrain_final_temp: float = 1.0  # DyMN only
    model_width: float = 1.0
    head_type: str = "mlp"
    se_dims: str = "c"
    num_classes: int = 200


@dataclass
class DataConfig:
    # Augmentation
    roll: bool = True
    wavmix: bool = True
    gain_augment: int = 12
    variable_eval_length: bool = False  # True => val/eval batch_size forced to 1
    # DataLoader
    batch_size: int = 64
    num_workers: int = 5
    prefetch_factor: int = None


@dataclass
class OptimConfig:
    lr: float = 7e-5
    weight_decay: float = 0.0
    mixup_alpha: float = 0.3

    # ------------------------------------------------------------------ LR schedule
    # Which schedule to run on top of Adam:
    #   "plateau"       -> PAI-managed ReduceLROnPlateau (uses
    #                      scheduler_patience / scheduler_mode below). The
    #                      historical PAI baseline.
    #   "lambda_warmup" -> mirror ex_fsd50k.py: exp warmup + linear rampdown
    #                      + floor, via exp_warmup_linear_down(...). Driven
    #                      manually against optimizer.param_groups so PAI's
    #                      validation hook cannot clobber it. A no-op
    #                      LambdaLR is still registered with the PAI tracker
    #                      so optimizer-rebuild-on-restructure keeps working.
    #                      See `lambda_warmup_per_cycle` below -- you almost
    #                      certainly want True (default).
    scheduler_name: str = "plateau"

    # --- "plateau" params (ReduceLROnPlateau).
    # `scheduler_patience` must be < pai.n_epochs_to_switch so a plateau is
    # detected before PAI switches modes when pai.switch_mode == "fixed". In
    # "history" mode PAI does its own plateau detection so patience mostly
    # just controls how fast LR decays within a given PAI phase.
    #
    # Tuning notes (motivated by the empirical LR log on mn04_as):
    # * factor=0.1 (PyTorch default) drops LR 10x per trigger. With
    #   patience=5 and no cooldown, a long neuron phase hit three consecutive
    #   triggers and dragged LR from 7e-5 -> 7e-8 in 18 epochs, effectively
    #   freezing training until the next PAI restructure reset it. Prefer
    #   factor=0.5 (half per trigger) and a non-zero cooldown.
    # * cooldown forces a gap between consecutive triggers so RLoP doesn't
    #   fire 3x back-to-back while the metric is legitimately flat during
    #   dendrite training (neuron weights frozen -> val mAP won't improve
    #   regardless of LR; dropping further is pointless).
    # * min_lr caps the bleeding. Below ~1e-7 Adam is numerically stationary
    #   on fp32 weights; going lower just wastes the rest of the PAI phase.
    scheduler_patience: int = 5
    scheduler_mode: str = "max"
    scheduler_factor: float = 0.5        # per-trigger multiplier (was 0.1)
    scheduler_cooldown: int = 3          # epochs of silence after a trigger
    scheduler_min_lr: float = 1e-7       # hard floor for RLoP

    # --- "lambda_warmup" params (mirror ex_fsd50k.py defaults).
    # Schedule shape (applied to whatever epoch counter we feed in):
    #   e = 0 .. warm_up_len - 1           exponential rampup 0 -> 1
    #   e = warm_up_len .. ramp_down_start - 1   constant at 1.0
    #   e = ramp_down_start .. +ramp_down_len    linear 1 -> last_lr_value
    #   after that                         constant at last_lr_value
    #
    # If lambda_warmup_per_cycle is True (recommended), the counter RESETS
    # to 0 on every PAI `restructured=True` event, i.e. every time dendrites
    # are added and the optimizer is rebuilt. Each PAI cycle then gets its
    # own warmup+rampdown, which matches what ex_fsd50k.py does for a single
    # fine-tune. Tune (warm_up_len, ramp_down_start, ramp_down_len) so the
    # full curve fits inside a typical cycle length; otherwise the schedule
    # will keep restarting mid-warmup and never reach full lr.
    #
    # If False, the schedule runs over the whole PAI training run using the
    # absolute epoch counter. Only use this if you expect your total PAI
    # epoch count to be close to ramp_down_start + ramp_down_len; otherwise
    # the curve gets stretched/truncated and most of training sits at the
    # last_lr_value floor.
    # Tuned for long PAI history segments (~30-40 epochs): long hold at full
    # LR, then a gradual linear decay. Short gaps (~11 epochs) may never reach
    # ramp_down_start and stay near full LR until the next switch — intentional
    # if you prioritize long segments over short-cycle curve completion.
    lambda_warmup_per_cycle: bool = True
    warm_up_len: int = 3
    ramp_down_start: int = 10
    ramp_down_len: int = 25
    last_lr_value: float = 0.01


@dataclass
class PAIConfig:
    """PAI knobs. See https://www.perforatedai.com/docs and perforatedai source."""

    # ------------------------------------------------------------------- debug
    print_model_layout: bool = False
    # ------------------------------------------------------------------ master
    perforated_bp: bool = True
    testing_dendrite_capacity: bool = False
    verbose: bool = False

    # ------------------------------------------------------- tensor dimensions
    # NCHW features: channel/neuron dim is index 1. Other dims variable (-1).
    output_dimensions: Tuple[int, int, int, int] = (-1, 0, -1, -1)

    # ------------------------------------------------------------ module selection
    # Extra class names appended to `module_names_to_perforate` IN ADDITION
    # to whatever the per-model branch in `_select_pai_blocks_for_model`
    # picks. PAI's library default for CNNs is ["Conv2d", "Linear",
    # "PAISequential", ...]; the training script may override that wholesale
    # via `perforate_names_override` below, in which case this extras list
    # is appended AFTER the override.
    extra_module_names_to_perforate: Tuple[str, ...] = ()

    # REPLACE PAI's default class-name perforate list with exactly this list
    # (instead of appending). Use sparingly -- the main use-case is to strip
    # "Conv2d" out of the default so the walker passes through Conv2d leaves
    # without wrapping them.
    #
    # MobileNet strategy (probe-validated; see probes/mn04_as_plateau.csv):
    #   set_module_names_to_perforate(["Linear"]) so PAI descends into every
    #   InvertedResidual / Conv2dNormActivation WITHOUT wrapping them, and
    #   wraps only the Linears it finds on the way down. Given MN's module
    #   layout, those Linears are exactly:
    #     * features[N].block[2].conc_se_layers[i].fc1   (SE squeeze)
    #     * features[N].block[2].conc_se_layers[i].fc2   (SE excite)
    #     * classifier.2                                 (640 -> 512)
    #     * classifier.5                                 (512 -> 200)
    #   Gradient probe showed these 15 SE Linears + 2 classifier Linears
    #   carry >=26x the gradient magnitude of any IR/C2NA wrapper.
    #
    # DyMN uses the default list (None here) + append of DY_Block in the
    # training script, so block-level perforation keeps working there.
    #
    # None  -> keep PAI's library default.
    # Tuple -> overwrite with these class names only.
    perforate_names_override: Optional[Tuple[str, ...]] = None
    
    # Full-model layout reference (MobileNet mn04_as, width_mult=0.4):
    #
    #   features[0]            Conv2dNormActivation   (stem, 1 -> 8)   [walker descends -- no Linear inside]
    #   features[1..15]        InvertedResidual                        [walker descends into .block[2] SE]
    #     .block[0]            Conv2dNormActivation   (1x1 expand)     [skipped -- Conv2d only]
    #     .block[1]            Conv2dNormActivation   (dw conv)        [skipped -- Conv2d only]
    #     .block[2]            ConcurrentSEBlock                       [walker descends]
    #       .conc_se_layers[i] SqueezeExcitation
    #         .fc1             Linear                                  [PERFORATED]
    #         .fc2             Linear                                  [PERFORATED]
    #   Presets ``late_se_blocks_only`` / ``prelogit_plus_late_se_blocks``: override
    #   ``("Linear",)`` only; list late ``.features.{11..15}.block.2`` in
    #   ``perforate_module_ids`` so PAI wraps those whole SE modules by id.
    #   Earlier SE stages (``.features.{4,5,6,10}.block.2`` on default MN conf)
    #   are listed in ``track_module_ids`` so they stay tracked, not perforated.
    #   Never use ``("ConcurrentSEBlock",)`` in the override with a short id
    #   list only — PAI matches *all* ``ConcurrentSEBlock`` instances by class name.
    #     .block[3]            Conv2dNormActivation   (1x1 project)    [skipped -- Conv2d only]
    #   features[16]           Conv2dNormActivation   (head 80 -> 480) [walker descends -- no Linear inside]
    #   classifier.2           Linear                                  [PERFORATED]
    #   classifier.5           Linear                                  [PERFORATED]
    #
    # DyMN (dymn04_as):
    #   model.in_c             Conv2dNormActivation   (stem)           [tracked via track_module_ids]
    #   model.layers[0..14]    DY_Block               (body)           [perforated w/ DYBlockProcessor]
    #   model.out_c            Conv2dNormActivation   (head)           [tracked -- see below]
    #   model.classifier.2     Linear                                  [perforated]
    #   model.classifier.5     Linear                                  [perforated]
    #
    # DY_Block is a multi-input module (`forward(x, g=None)`, where `g` is
    # the upstream global-context vector). Without intervention PAI's default
    # dendrite clone receives only `x` and `g` defaults to None -- every
    # DynamicConv inside the clone then produces near-constant kernels and
    # PBScore sits at ~0. The training script fixes this by registering a
    # ``DYBlockProcessor`` via ``modules_with_processing`` (customization.md
    # section 2.2) so the dendrite sees the same ``(x, g)`` pair as the
    # neuron. ``out_c`` stays track-only: prior-run PBScore on it was ~0
    # (wide projection right before global pooling).
    #
    # Whitelist of exact dotted ids to perforate. Set to None for the usual
    # class-based wiring above. When a list is given, PAI restricts dendrite
    # addition to these exact ids (still subject to the class-name filter).
    perforate_module_ids: Optional[List[str]] = None

    # Skip these dotted module ids even if a class/id filter would otherwise
    # pick them up. Kept for DyMN's `.in_c` stem. For MobileNet the probe-
    # validated Linear-only class filter already ignores the stem (there are
    # no Linears inside `features.0`), so entries like `.features.0` are
    # no-ops but are retained for compatibility with the DyMN path. PAI
    # silently ignores ids that don't exist in the current model.
    track_module_ids: Tuple[str, ...] = (".features.0", ".in_c")

    # Class *names* to wrap as track-only (NOT perforated, gradients pass
    # through cleanly). Use this for modules that PAI can't dendrite safely
    # but also shouldn't walk into -- e.g. side-branch modules with their own
    # norm layers. PAI's own warning recommends this route for any module
    # flagged as "potentially found a norm Layer that wont be converted".
    # Model-specific defaults (e.g. DyMN's `ContextGen`) are added in the
    # training script via `_configure_pai`; this tuple appends extras.
    track_module_names: Tuple[str, ...] = ()

    # When True, silences PAI's "unwrapped modules found" warning + pdb break
    # during `perforate_model`. Flip to True only after you've manually
    # verified your `track_module_names`/`modules_to_perforate` cover every
    # module you care about -- otherwise you lose the only safety net.
    unwrapped_modules_confirmed: bool = False

    # ----------------------------------------------------------- correlation
    initial_correlation_batches: int = 16
    using_safe_tensors: bool = True

    # ---------------------------------------------------------- switch mode
    switch_mode: str = "fixed"           # "fixed" | "history"
    fixed_switch_num: int = 25            # epochs between dendrite switches
    first_fixed_switch_num: int = 20     # epochs before the first switch
    n_epochs_to_switch: int = 10         # history: neuron plateau window
    p_epochs_to_switch: int = 5         # history: dendrite plateau window
    history_lookback: int = 1

    # --------------------------------------------------------- dendrite caps
    cap_at_n: bool = True
    max_dendrites: int = 2
    # Hard stop: max consecutive epochs while tracker is in mode "p".
    max_consecutive_epochs_in_p_mode: Optional[int] = None

    # --------------------------------------------------------- val smoothing
    running_average_pb: bool = True

    # ----------------------- mode-P "last improved" thresholds (per-node) ---
    pai_improvement_threshold: float = 0.1
    pai_improvement_threshold_raw: float = 0.0001

    # --------------------------------- validation improvement thresholds ----
    improvement_threshold: Tuple[float, ...] = (0.01, 0.0001, 0.0)
    # mAP is reported on the native 0-1 scale (see `add_validation_score(mAP,..)`).
    improvement_threshold_raw: float = 0.0001

    # --------------------------------- candidate dendrite init + stability --
    candidate_weight_initialization_multiplier: float = 0.005
    candidate_grad_clipping: float = 1.0
    drawing_extra_graphs: bool = True

    # ------------------------------------------- resume from prior PAI save --
    # Resume training from a checkpoint written by a *previous* PAI run
    # (``UPA.save_system`` format = ``{name}.pt`` with tracker_string buffer).
    # Canonical target: ``beforeSwitch_0.pt`` -- the snapshot PAI writes at the
    # end of the initial neuron cycle, right before the first dendrite switch.
    # Loading that lets you take a fully-trained neuron baseline and then
    # drive the dendrite phase with *different* PAI hyper-parameters (e.g.
    # a different ``switch_mode`` or ``p_epochs_to_switch``) without paying
    # the cost of the neuron-only training again.
    #
    # Requirements
    # ------------
    # * ``ModelConfig`` (model_name, width, num_classes, head_type, se_dims)
    #   must match the run that wrote the checkpoint -- the un-perforated
    #   backbone has to produce identical Conv2d/Linear shapes.
    # * The PAI wrapper layout (``perforate_names_override``,
    #   ``perforate_module_ids``, ``track_module_names``, ``track_module_ids``)
    #   must also match -- otherwise the state_dict keys won't align and
    #   ``load_net_from_dict`` will error on missing/unexpected modules.
    # * ``using_safe_tensors`` must match how the checkpoint was written
    #   (both runs default to True, so usually a no-op).
    #
    # What IS restored
    #   - neuron weights + BN stats on every wrapped module
    #   - PAI tracker ``member_vars``: score history, switch_epochs,
    #     num_epochs_run, mode, num_dendrites_added, ...
    # What is NOT restored
    #   - ``GPA.pc`` fields (switch_mode / p_epochs_to_switch / thresholds).
    #     These come from the current Config and stay in effect after load,
    #     which is exactly the "different P training mode setting" knob.
    #   - optimizer / scheduler state (rebuilt post-load via
    #     ``pai_tracker.setup_optimizer``).
    #
    # ``resume_from_folder`` is the directory containing ``{resume_checkpoint}.pt``
    # (usually the prior run's ``pai/<experiment_name>`` save dir). Set to
    # ``None`` (default) to disable. Paths are relative to the training
    # script's CWD unless absolute.
    resume_from_folder: Optional[str] = None
    resume_checkpoint: str = "beforeSwitch_0"

    # After ``load_system``, optionally slam the tracker back into neuron
    # mode ("n") with ``epoch_last_improved = num_epochs_run``. Use this
    # when the saved tracker was mid-switch ("n" that had just plateaued,
    # about to flip to "p") and you want the *new* ``p_epochs_to_switch``
    # / plateau thresholds to drive the next cycle from a clean starting
    # point instead of immediately firing on the old plateau detection.
    # ``load_system`` already clears ``current_best_validation_score`` and
    # ``epoch_last_improved``; this adds a mode reset on top.
    resume_force_neuron_mode: bool = False

    # Generic escape hatch: any key -> value pair applied onto
    # ``GPA.pai_tracker.member_vars`` after ``load_system`` and after the
    # ``resume_force_neuron_mode`` handling. Use sparingly; typical examples:
    #   {"switch_epochs": [], "num_epochs_run": 0}  -> start fresh counters
    #   {"num_dendrites_added": 0}                  -> pretend arch is pristine
    # Leave empty to keep PAI's own post-load defaults.
    resume_tracker_overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WandbConfig:
    entity: str = "11-785_perforated_ai"
    project: str = "FSD50K"
    name: str = ""
    group: str = ""
    notes: str = ""
    tags: Tuple[str, ...] = (
        "Audio Tagging",
        "FSD50K",
        "Fine-Tuning",
        "Dendrites",
        "PerforatedAI",
    )


@dataclass
class RuntimeConfig:
    save_name: Optional[str] = None
    output_dir: str = "runs"
    resume_pai: bool = False
    pai_resume_tag: str = "latest"
    exit_on_pai_restructure: bool = False
    max_wall_minutes: Optional[float] = None
    sync_pai_saves: bool = False
    phase_index: int = 0
    allow_overwrite: bool = False


@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    pai: PAIConfig = field(default_factory=PAIConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # PAI layout preset: "default" or a key in ``_PRESET_PAI_PATCHES``.
    pai_preset: str = PAI_PRESET_PROBE_LATE_BACKBONE_HEAD
    experiment_name: str = ""
    seed: int = 0
    cuda: bool = True

    def __post_init__(self) -> None:
        _apply_pai_preset(self)
        # Keep explicit config-file names unless a runtime override asks to refresh them.
        _derive_run_identity(self)


# Default instance consumed by the training script. Switch PAI layout here,
# e.g. ``config = Config(pai_preset=PAI_PRESET_ALL_C2NA_PLUS_LOGIT)``, or mutate
# fields in-place after import.
config = Config()
