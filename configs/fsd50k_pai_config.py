"""
Configuration for FSD50K fine-tuning with Perforated AI Dendrites.

All hyperparameters are collected here so that `ex_fsd50k_perforatedai.py`
doesn't need any argparse/CLI plumbing. Import `config` from this module
and access it as attributes (e.g. `config.optim.lr`, `config.pai.max_dendrites`).

Adjust sub-configs and call `make_fsd50k_config(...)`, or change the default
`config` instance in-place after `make_fsd50k_config()`, then pass it to
`train(config)`.

See https://www.perforatedai.com/docs for PAI semantics.

IMPORTANT PAI GOTCHAS THIS CONFIG HELPS AVOID
----------------------------------------------
* PAIConfig.__init__ auto-loads `{cwd}/{save_name}/{save_name}_config.json`
  at import time, and every `set_*` call auto-saves back. If you re-use an
  experiment name, stale JSON overrides Python defaults. The training
  script below calls `GPA.pc.set_save_name(experiment_name)` before any
  other config, which points auto-save at THIS run's folder.
* `GPA.pc.append_modules_to_convert(...)` does NOT exist. The correct API
  names are `append_modules_to_perforate` (class objects) and
  `append_module_names_to_perforate` (string class names). Unknown
  `set_*`/`append_*` calls are silently swallowed as no-ops by
  PAIConfig.__getattr__.
* `UPA.initialize_pai` is not the public entry point; use
  `UPA.perforate_model` (skill + library).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


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
    model_name: str = "dymn04_as"
    pretrain_final_temp: float = 1.0  # DyMN only
    model_width: float = 1.0
    head_type: str = "mlp"
    se_dims: str = "c"
    num_classes: int = 200


@dataclass
class DataConfig:
    roll: bool = True
    wavmix: bool = True
    gain_augment: int = 12
    variable_eval_length: bool = False  # True => val/eval batch_size forced to 1


@dataclass
class OptimConfig:
    lr: float = 7e-5
    weight_decay: float = 0.0
    mixup_alpha: float = 0.3
    # ReduceLROnPlateau is managed by the PAI tracker. `scheduler_patience`
    # must be < pai.n_epochs_to_switch so a plateau is detected before PAI
    # switches modes.
    scheduler_patience: int = 5
    scheduler_mode: str = "max"


@dataclass
class PAIConfig:
    """PAI knobs. See https://www.perforatedai.com/docs and perforatedai source."""

    # ------------------------------------------------------------------ master
    perforated_bp: bool = True          # Perforated Backpropagation (beta)
    testing_dendrite_capacity: bool = False   # keep True until 3 dendrites land
    verbose: bool = False

    # ------------------------------------------------------- tensor dimensions
    # NCHW features: channel/neuron dim is index 1. Other dims variable (-1).
    output_dimensions: Tuple[int, int, int, int] = (-1, 0, -1, -1)

    # ------------------------------------------------------------ module selection
    # Class names on `module_names_to_perforate`. "PAISequential", "Linear",
    # "Conv2d" etc. are already included in the library default; we append
    # the MobileNet block classes by type in the training script using
    # `append_modules_to_perforate`. Any class name added here is treated as
    # the short class name (e.g. "InvertedResidual").
    extra_module_names_to_perforate: Tuple[str, ...] = ()

    # Perforate the full backbone + classifier (default) or restrict to the
    # "top" of the network for parameter efficiency. Block indices map to
    # `model.features[<idx>]` on mn10_as (width_mult=1.0):
    #   0          : Stem      Conv2dNormActivation (1 -> 8)
    #   1..3       : Early     InvertedResidual      (8 -> 16 -> 24)
    #   4..10      : Middle    InvertedResidual      (24 -> 40)
    #   11..15     : Later     InvertedResidual      (40 -> 56 -> 80)
    #   16         : Final     Conv2dNormActivation  (80 -> 480)
    # Classifier linears sit at `.classifier.2` (480->640) and
    # `.classifier.5` (640->200 logits).
    #
    # None  -> perforate every InvertedResidual + final Conv2dNormActivation
    #          + classifier Linears (skill default: block-level conversion).
    # List  -> restrict dendrites to these dotted module ids only.
    perforate_module_ids: Optional[List[str]] = None

    # Skip these dotted module ids even if a class/id filter would otherwise
    # pick them up. The stem conv rarely benefits from dendrites per the
    # skill's "top-layers only" guidance. Covers both MN (`.features.0`) and
    # DyMN (`.in_c`) naming conventions; PAI silently ignores ids that don't
    # exist in the current model.
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
    initial_correlation_batches: int = 8
    using_safe_tensors: bool = True

    # ---------------------------------------------------------- switch mode
    switch_mode: str = "history"           # "fixed" | "history"
    fixed_switch_num: int = 5            # epochs between dendrite switches
    first_fixed_switch_num: int = 15     # epochs before the first switch
    n_epochs_to_switch: int = 10         # history: neuron plateau window
    p_epochs_to_switch: int = 5         # history: dendrite plateau window
    history_lookback: int = 1

    # --------------------------------------------------------- dendrite caps
    cap_at_n: bool = False
    max_dendrites: int = 3

    # --------------------------------------------------------- val smoothing
    running_average_pb: bool = False

    # ----------------------- mode-P "last improved" thresholds (per-node) ---
    pai_improvement_threshold: float = 0.1
    pai_improvement_threshold_raw: float = 0.01

    # --------------------------------- validation improvement thresholds ----
    improvement_threshold: float = 0.0
    improvement_threshold_raw: float = 0.0

    # --------------------------------- candidate dendrite init + stability --
    candidate_weight_initialization_multiplier: float = 0.005
    candidate_grad_clipping: float = 1.0
    drawing_extra_graphs: bool = True


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


# Convenience: "top-only" id list matching the "Later blocks + Final conv +
# Classifier" region of mn10_as (40 -> 56 -> 80 -> 480 -> 640 -> 200).
# Assign to `config.pai.perforate_module_ids` to restrict dendrites to these
# for parameter-efficient experiments per skill step 7.3.
TOP_ONLY_MODULE_IDS: Tuple[str, ...] = (
    ".features.11",
    ".features.12",
    ".features.13",
    ".features.14",
    ".features.15",
    ".features.16",
    ".classifier.2",
    ".classifier.5",
)


@dataclass
class Config:
    preprocess: PreprocessConfig
    model: ModelConfig
    data: DataConfig
    optim: OptimConfig
    pai: PAIConfig
    experiment_name: str
    wandb: WandbConfig
    cuda: bool = True
    batch_size: int = 64
    num_workers: int = 12


def _fsd50k_run_slug(model: ModelConfig, pai: PAIConfig) -> str:
    return f"FSD50K-{model.model_name}-pai-{pai.switch_mode}"


def make_fsd50k_config(
    *,
    preprocess: Optional[PreprocessConfig] = None,
    model: Optional[ModelConfig] = None,
    data: Optional[DataConfig] = None,
    optim: Optional[OptimConfig] = None,
    pai: Optional[PAIConfig] = None,
    cuda: bool = True,
    batch_size: int = 64,
    num_workers: int = 12,
) -> Config:
    """Build `Config` with `experiment_name` and `wandb.{name,group}` set to `FSD50K-{m}-pai-{s}`."""
    preprocess = preprocess or PreprocessConfig()
    model = model or ModelConfig()
    data = data or DataConfig()
    optim = optim or OptimConfig()
    pai = pai or PAIConfig()
    g = _fsd50k_run_slug(model, pai)
    return Config(
        preprocess=preprocess,
        model=model,
        data=data,
        optim=optim,
        pai=pai,
        experiment_name=g,
        wandb=WandbConfig(
            name=g,
            group=g,
            notes=f"Fine-tune {model.model_name} on FSD50K with Perforated AI Dendrites.",
        ),
        cuda=cuda,
        batch_size=batch_size,
        num_workers=num_workers,
    )


# Default instance consumed by the training script. Prefer editing kwargs in
# `make_fsd50k_config(...)` or mutating this object after creation.
config = make_fsd50k_config()
