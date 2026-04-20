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
    model_name: str = "dymn04_as"
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
    batch_size: int = 32
    num_workers: int = 8


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
    testing_dendrite_capacity: bool = True
    verbose: bool = False

    # ------------------------------------------------------- tensor dimensions
    # NCHW features: channel/neuron dim is index 1. Other dims variable (-1).
    output_dimensions: Tuple[int, int, int, int] = (-1, 0, -1, -1)

    # ------------------------------------------------------------ module selection
    # Class names on `module_names_to_perforate`. "PAISequential", "Linear",
    # "Conv2d" etc. are already included in the library default; we append
    # the MobileNet / DyMN block classes by type in the training script via
    # `append_modules_to_perforate`. Any class name added here is treated as
    # the short class name (e.g. "InvertedResidual").
    extra_module_names_to_perforate: Tuple[str, ...] = ()

    # Perforate the full backbone + classifier (default) or restrict to the
    # "top" of the network for parameter efficiency.
    #
    # MobileNet (mn10_as, width_mult=1.0):
    #   model.features[0]     Conv2dNormActivation (stem, 1 -> 8)   [tracked]
    #   model.features[1..3]  InvertedResidual     (8 -> 16 -> 24)  [perforated]
    #   model.features[4..10] InvertedResidual     (24 -> 40)       [perforated]
    #   model.features[11..15] InvertedResidual    (40 -> 56 -> 80) [perforated]
    #   model.features[16]    Conv2dNormActivation (80 -> 480)      [perforated]
    #   model.classifier.2    Linear               (480 -> 640)     [perforated]
    #   model.classifier.5    Linear               (640 -> 200)     [perforated]
    #
    # DyMN (dymn04_as):
    #   model.in_c            Conv2dNormActivation (stem)           [tracked via track_module_ids]
    #   model.layers[0..14]   DY_Block             (body)           [perforated w/ DYBlockProcessor]
    #   model.out_c           Conv2dNormActivation (head)           [tracked -- see below]
    #   model.classifier.2    Linear               (384 -> 512)     [perforated]
    #   model.classifier.5    Linear               (512 -> 200)     [perforated]
    #
    # DY_Block is a multi-input module (`forward(x, g=None)`, where `g` is
    # the upstream global-context vector). Without intervention PAI's default
    # dendrite clone receives only `x` and `g` defaults to None -- every
    # DynamicConv inside the clone then produces near-constant kernels and
    # PBScore sits at ~0 (exactly what the first DyMN run's PBScore plot
    # showed for all 15 blocks). The training script fixes this by
    # registering a ``DYBlockProcessor`` via ``modules_with_processing``
    # (customization.md section 2.2) so the dendrite sees the same
    # ``(x, g)`` pair as the neuron.
    #
    # ``out_c`` remains track-only: empirically its PBScore was ~0 even with
    # full wrapping (wide 64->384 projection right before global pooling --
    # no residual error left to model once the pretrained main branch runs).
    #
    # None  -> default module-class perforation (DY_Block + Linear).
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
    pai_improvement_threshold: float = 0.5
    pai_improvement_threshold_raw: float = 0.01

    # --------------------------------- validation improvement thresholds ----
    # PAI getter returns element `[min(len-1, num_dendrites_added)]`, so the
    # list encodes per-stage plateau tolerance: strict (0.1 % rel) at arch 0,
    # relaxed (0.01 %) once the first dendrite set lands, 0 once fully grown.
    # Values of 0.0 (the earlier default here) disabled plateau detection
    # entirely -- the first arch-switch only fired via raw epoch timeout at
    # epoch ~80 instead of ~30, wasting compute on a plateaued model.
    improvement_threshold: Tuple[float, ...] = (0.001, 0.0001, 0.0)
    # mAP is reported on the 0-100 scale (see `add_validation_score(mAP*100,..)`),
    # so PAI's raw-diff default of 1e-5 is far below validation noise. 1e-3
    # = 0.001 mAP points requires a genuine improvement before resetting
    # the plateau counter.
    improvement_threshold_raw: float = 1e-3

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
        slug = self.experiment_name or f"FSD50K-{self.model.model_name}-pai-{self.pai.switch_mode}"
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
