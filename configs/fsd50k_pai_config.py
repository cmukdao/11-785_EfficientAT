"""
Configuration for FSD50K fine-tuning with Perforated AI Dendrites.

Usage:

    from configs.fsd50k_pai_config import config
    train(config)                     # defaults

    # or customise a sub-config:
    cfg = Config(model=ModelConfig(model_name="mn10_as"))
    cfg.data.batch_size = 32
    train(cfg)

Derived fields (`experiment_name`, `wandb.name/group/notes`) are filled
in by `Config.__post_init__` from `model.model_name` + `pai.switch_mode`
unless you pass explicit values.
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
    num_workers: int = 8
    prefetch_factor: int = None


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
    pai_improvement_threshold: float = 0.5
    pai_improvement_threshold_raw: float = 0.01

    # --------------------------------- validation improvement thresholds ----
    # PAI getter returns element `[min(len-1, num_dendrites_added)]`, so the
    # list encodes per-stage plateau tolerance: strict (0.1 % rel) at arch 0,
    # relaxed (0.01 %) once the first dendrite set lands, 0 once fully grown.
    # Values of 0.0 (the earlier default here) disabled plateau detection
    # entirely -- the first arch-switch only fired via raw epoch timeout at
    # epoch ~80 instead of ~30, wasting compute on a plateaued model.
    improvement_threshold: Tuple[float, ...] = (0.01, 0.001, 0.0)
    # mAP is reported on the 0-100 scale (see `add_validation_score(mAP*100,..)`),
    improvement_threshold_raw: float = 1e-2

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

@dataclass
class Config:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    pai: PAIConfig = field(default_factory=PAIConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    experiment_name: str = ""
    cuda: bool = True

    def __post_init__(self) -> None:
        # Derive run slug + wandb metadata from model + pai unless caller
        # supplied explicit overrides. This keeps `Config()` zero-arg usable
        # while still letting power users pin any field.
        slug = self.experiment_name or f"FSD50K-{self.model.model_name}-pai-{self.pai.switch_mode}-new"
        self.experiment_name = slug
        if not self.wandb.name:
            self.wandb.name = slug
        if not self.wandb.group:
            self.wandb.group = slug
        if not self.wandb.notes:
            self.wandb.notes = (
                f"Fine-tune {self.model.model_name} on FSD50K with Perforated AI Dendrites."
            )


# Default instance consumed by the training script. Construct a fresh
# `Config(...)` or mutate this object in-place to customise a run.
config = Config()
