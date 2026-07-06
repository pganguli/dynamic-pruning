"""
Structured config schema for every stage of the dynamic-pruning pipeline.

Each dataclass here is registered with Hydra's ConfigStore (see
`register_configs()`) so the YAML files under `configs/` are validated and
type-checked against it. Defaults encoded here are the values empirically
validated to work well for CIFAR-10/ResNet-56 dynamic-target training (see
README.md "Deviations from the paper" for the reasoning behind each one) —
treat them as the sane starting point for new scenarios, and use
`scripts/tune.py` (Optuna) to re-search them for a new architecture/dataset.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore

__all__ = [
    "DataConfig",
    "ModelConfig",
    "OptimConfig",
    "DecisionConfig",
    "DynamicRangeConfig",
    "TensorBoardConfig",
    "PretrainConfig",
    "TrainConfig",
    "FinetuneConfig",
    "CalibrateConfig",
    "ExportConfig",
    "OptunaSearchConfig",
    "register_configs",
]


@dataclass
class DataConfig:
    """Which dataset to train/evaluate on."""

    name: str = "cifar10"  # cifar10 | cifar100 | har | kws
    num_classes: int = 10
    train_batch_size: int = 512
    test_batch_size: int = 100


@dataclass
class ModelConfig:
    """Backbone architecture."""

    arch: str = "resnet56"  # resnet10 | resnet20 | resnet56 | har_cnn | kws
    dropout_prob: float = 0.0  # spatial dropout between convs; Stage 1 only


@dataclass
class OptimConfig:
    """Backbone SGD optimizer + loss shaping shared by every training stage."""

    lr: float | None = None  # None => arch-specific default (see common.py)
    momentum: float = 0.9
    weight_decay: float = 1e-4
    nesterov: bool = True
    label_smoothing: float = 0.1  # 0 = off; safe default, doesn't touch gating


@dataclass
class DecisionConfig:
    """Channel-selection decision head + its regularizers (dynamic mode)."""

    action_num: int | None = None  # None => arch-specific default
    pruning_threshold: float = 0.5
    gamma: float = 10.0  # expected-density regularizer strength (dynamic)
    gamma_under: float = 0.7  # static-mode only: gamma multiplier below target
    lambda_div: float = 20.0  # gate diversity anchor strength (dynamic)
    lambda_balance: float = 0.5  # load-balancing loss weight (dynamic)


@dataclass
class DynamicRangeConfig:
    """r_tgt sampling range for dynamic-target training/fine-tuning."""

    r_min: float = 0.1
    r_max: float = 0.9
    r_endpoint_prob: float = 0.1


@dataclass
class TensorBoardConfig:
    enabled: bool = True
    log_dir: str | None = None  # None => derived from the run's logdir


@dataclass
class PretrainConfig:
    """Stage 1 — pretrain the backbone with no pruning."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(weight_decay=1e-4))
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    epochs: int = 160
    log_interval: int = 100
    seed: int = 0


@dataclass
class TrainConfig:
    """Stage 2 (static) / Stage 2D (dynamic) — joint decision-head + backbone training."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    optim: OptimConfig = field(default_factory=lambda: OptimConfig(weight_decay=1e-9))
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    dynamic: bool = True
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    sparsity_level: float = 0.1  # static-mode target; dynamic-mode checkpoint-path key
    epochs: int = 100
    log_interval: int = 100
    seed: int = 0


@dataclass
class FinetuneConfig:
    """Stage 3 (static) / Stage 3D (dynamic) — backbone-only fine-tune, gates frozen."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    optim: OptimConfig = field(
        default_factory=lambda: OptimConfig(lr=1e-3, weight_decay=1e-4)
    )
    tensorboard: TensorBoardConfig = field(default_factory=TensorBoardConfig)
    dynamic: bool = True
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    sparsity_level: float = 0.1
    epochs: int = 160
    log_interval: int = 100
    seed: int = 0


@dataclass
class CalibrateConfig:
    """Stage 4D — sweep r_tgt and emit the calibration table."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    sparsity_level: float = 0.1
    grid_n: int = 17
    finetuned: bool = True
    out_csv: str | None = "calibration.csv"


@dataclass
class ExportConfig:
    """Stage 5 — export the (image, r_tgt) -> output model to ONNX."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    dynamic_range: DynamicRangeConfig = field(default_factory=DynamicRangeConfig)
    sparsity_level: float = 0.1
    finetuned: bool = True
    # Additional precisions to export alongside the default fp32 files.
    # Both are opt-in: fp16 is a straight cast (no accuracy-recovery step
    # needed); int8 is post-training static quantization, which *can* lose
    # accuracy on a small/tight-capacity backbone -- run.py prints an
    # onnxruntime accuracy comparison against fp32 whenever export_int8 is
    # enabled so that's a measured fact for your checkpoint, not a guess.
    export_fp16: bool = False
    export_int8: bool = False
    # Number of (image, r_tgt) calibration batches to feed onnxruntime's
    # static quantizer when export_int8 is enabled.
    int8_calibration_batches: int = 20


@dataclass
class OptunaSearchConfig:
    """Optuna study settings for `scripts/tune.py`."""

    n_trials: int = 30
    epochs_per_trial: int = 8  # short budget per trial; full recipe uses `train.epochs`
    study_name: str = "dynamic_pruning_search"
    storage: str | None = None  # e.g. "sqlite:///optuna.db" to persist/resume
    train: TrainConfig = field(default_factory=TrainConfig)


def register_configs() -> None:
    """Register every dataclass above as a Hydra structured-config schema.

    Call once, before @hydra.main runs, from each script in scripts/. YAML
    files under configs/ are validated against these schemas at compose time.
    """
    from ._compat import _safe_check_help, argparse

    assert getattr(argparse.ArgumentParser, "_check_help") is _safe_check_help, (
        "Patching argparse failed"
    )

    cs = ConfigStore.instance()
    cs.store(name="pretrain_schema", node=PretrainConfig)
    cs.store(name="train_schema", node=TrainConfig)
    cs.store(name="finetune_schema", node=FinetuneConfig)
    cs.store(name="calibrate_schema", node=CalibrateConfig)
    cs.store(name="export_schema", node=ExportConfig)
    cs.store(name="optuna_schema", node=OptunaSearchConfig)
