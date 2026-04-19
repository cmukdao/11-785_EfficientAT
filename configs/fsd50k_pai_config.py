"""
Configuration for FSD50K fine-tuning with Perforated AI Dendrites + Perforated BP.

All hyperparameters are collected here so that `ex_fsd50k_perforatedai.py`
doesn't need any argparse/CLI plumbing. Import `config` from this module
and access it as attributes (e.g. `config.lr`, `config.pai.max_dendrites`).

Adjust values in-place to change an experiment, or construct a new
`Config(...)` instance elsewhere and pass it to `train(config)`.

See https://www.perforatedai.com/docs for PAI-specific semantics.
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ============================================================================
# Perforated AI license credentials
# Must be set BEFORE `perforatedai` is imported anywhere in the process.
# Kept here so that the training entry-point can export them to env vars
# before touching the PAI module.
# ============================================================================
PAI_EMAIL = "PAIUser3.11.2026@perforatedai.com"
PAI_TOKEN = (
    "g3ZDrJNAmBlh/tAdFUkkedY+mUGqaueCPQXAtkVyNq405Fc9+20MhmEIDttx285EhfFqDHFtMRA20BWKpQaqai3wNxaOeCNvsNsF7Nn2nTmpFUmkHGvmVWGD5JI1uxdPdW6jeUmaQBKNUNcNKOumr1iDaQpFnCvFDDcSi3yUmZ5JoCd0c/lAyGmRXWQ+dInOXX6MdE4XVmv7DI8jW626pNLemX7ZMo4dEGikNuhyuiAD1IJYNYaJxUK0zaizx/Avq6QbmwQviPjXDNyBsTdVzJQzhAG6Zf2DlU9j29RnEaj0XGd4j3MJeqsrq0FeXKqEMShnBKO0oL69CP4icVTOpQ=="
)


@dataclass
class WandbConfig:
    entity: str = "11-785_perforated_ai"
    project: str = "FSD50K"
    group: str = "FSD50K-mn10_as-pai"
    notes: str = "Fine-tune Models on FSD50K with Perforated AI Dendrites + Perforated BP."
    tags: Tuple[str, ...] = (
        "Audio Tagging",
        "FSD50K",
        "Fine-Tuning",
        "Dendrites",
        "PerforatedBP",
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
    model_name: str = "mn10_as"
    pretrain_final_temp: float = 1.0  # DyMN only
    model_width: float = 1.0
    head_type: str = "mlp"
    se_dims: str = "c"
    num_classes: int = 200


@dataclass
class DataConfig:
    # FSD50K uses predefined train/val/eval splits (no folds).
    roll: bool = True            # set False to disable random time roll
    wavmix: bool = True          # set False to disable waveform mixup
    gain_augment: int = 12
    variable_eval_length: bool = False  # if True, val/eval batch size forced to 1


@dataclass
class OptimConfig:
    lr: float = 7e-5
    weight_decay: float = 0.0
    mixup_alpha: float = 0.3
    # ReduceLROnPlateau is managed by the PAI tracker.
    # `scheduler_patience` must be < pai.n_epochs_to_switch so that a plateau
    # is detected before PAI switches modes.
    scheduler_patience: int = 5
    scheduler_mode: str = "max"


@dataclass
class PAIConfig:
    """Perforated AI-specific knobs. See https://www.perforatedai.com/docs."""

    # Master switches
    perforated_bp: bool = True          # Perforated Backpropagation (paid beta)
    no_dendrite_test: bool = False       # skip dendrite capacity test after first run
    verbose: bool = False

    # Tensor layout for convolutional features: NCHW
    # Index 1 = channel (neuron) dimension
    output_dimensions: Tuple[int, int, int, int] = (-1, 0, -1, -1)
    input_dimensions: Tuple[int, int, int, int] = (-1, 0, -1, -1)

    # Correlation batches / debugging
    initial_correlation_batches: int = 8
    debugging_output_dimensions: int = 0
    using_safe_tensors: bool = True

    # Switch mode: "fixed" | "history"
    switch_mode: str = "fixed"
    fixed_switch_num: int = 5            # epochs between dendrite switches (after first)
    first_fixed_switch_num: int = 15     # epochs before the first dendrite switch
    n_epochs_to_switch: int = 20         # history mode: neuron-cycle plateau window
    p_epochs_to_switch: int = 10         # history mode: dendrite-cycle plateau window
    history_lookback: int = 1

    # Dendrite structure
    cap_at_n: bool = False               # cap P-cycle length to first N-cycle length
    max_dendrites: int = 3

    # Validation smoothing
    running_average_pb: bool = False     # raw validation (EMA can drift on small sets)

    # Mode-P "last improved" thresholds (per-node correlation)
    pai_improvement_threshold: float = 0.1
    pai_improvement_threshold_raw: float = 0.01

    # Validation improvement thresholds (for plateau / best-arch tracking)
    improvement_threshold: float = 0.0
    improvement_threshold_raw: float = 0.0

    # Candidate dendrite init / stability
    candidate_weight_initialization_multiplier: float = 0.005
    candidate_grad_clipping: float = 1.0
    drawing_extra_graphs: bool = True


@dataclass
class Config:
    experiment_name: str = "FSD50K-mn10_as-pai-fixed-switch"
    cuda: bool = True
    batch_size: int = 64
    num_workers: int = 12

    wandb: WandbConfig = field(default_factory=WandbConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    pai: PAIConfig = field(default_factory=PAIConfig)


# Default instance consumed by the training script. Edit fields here (or
# override from another script) to change behavior without CLI flags.
config = Config()
