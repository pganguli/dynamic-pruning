"""Architecture-specific defaults, model construction, and the decision-head
transform shared by every training/eval stage."""

import torch.nn as nn

from .. import models
from ..decision import (
    apply_func,
    collect_params,
    decision_basicblock_forward,
    decision_conv_block_forward,
    init_decision_basicblock,
    init_decision_conv_block,
    normalize_head_weights,
    replace_func,
)

__all__ = [
    "NUM_CLASSES",
    "num_classes",
    "default_action_num",
    "default_pretrain_learning_rate",
    "default_learning_rate",
    "initialize_model",
    "transform_model",
]

NUM_CLASSES = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}

_ACTION_NUM_DEFAULTS = {"har_cnn": 5, "kws": 5}  # resnet* handled separately below
_LR_DEFAULTS = {"kws": 0.01, "har_cnn": 0.01}  # resnet* handled separately below
# Stage 1 (from-scratch backbone) needs a much larger step size than Stage 2's
# joint decision-head recipe below — the original train_baseline.py used 0.1
# for ResNet/CIFAR (batch_size=128); reusing Stage 2's 0.01 here silently
# starves Stage 1 of signal and the backbone plateaus well short of converged.
_PRETRAIN_LR_DEFAULTS = {"kws": 0.01, "har_cnn": 0.01}  # resnet* handled separately below


def num_classes(dataset: str) -> int:
    if dataset not in NUM_CLASSES:
        raise ValueError(f"Unknown dataset {dataset!r}")
    return NUM_CLASSES[dataset]


def default_action_num(arch: str) -> int:
    """Architecture-specific action_num when not explicitly configured.

    40 is the paper's static-mode ResNet default; dynamic-target training
    should explicitly set `decision.action_num: 16` (see configs/scenario/).
    """
    if arch.startswith("resnet"):
        return 40
    if arch in _ACTION_NUM_DEFAULTS:
        return _ACTION_NUM_DEFAULTS[arch]
    raise ValueError(f"Unknown model architecture {arch!r}")


def default_pretrain_learning_rate(arch: str) -> float:
    """Stage 1 (train.py's `pretrain.py`) from-scratch backbone LR."""
    if arch.startswith("resnet"):
        return 0.1
    if arch in _PRETRAIN_LR_DEFAULTS:
        return _PRETRAIN_LR_DEFAULTS[arch]
    raise ValueError(f"Unknown model architecture {arch!r}")


def default_learning_rate(arch: str) -> float:
    """Stage 2/3 joint decision-head recipe LR (0.01 per paper, ResNet)."""
    if arch.startswith("resnet"):
        return 0.01
    if arch in _LR_DEFAULTS:
        return _LR_DEFAULTS[arch]
    raise ValueError(f"Unknown model architecture {arch!r}")


def initialize_model(
    dataset: str, arch: str, n_classes: int, dropout_prob: float = 0.0
) -> nn.Module:
    print("==> Initializing model...")
    if dataset in ("cifar10", "cifar100"):
        return models.__dict__["cifar_" + arch](n_classes, dropout_prob=dropout_prob)
    if dataset == "har":
        return models.har_cnn(dropout_prob=dropout_prob)
    if dataset == "kws":
        return models.KWS_CNN_S(dropout_prob=dropout_prob)
    raise ValueError(f"Unknown dataset {dataset!r}")


def transform_model(model: nn.Module, arch: str, action_num: int) -> None:
    """Inject target-conditioned DecisionHeads into every gateable block."""
    if arch.startswith("resnet"):
        init_func, new_forward, module_type = (
            init_decision_basicblock,
            decision_basicblock_forward,
            "BasicBlock",
        )
    elif arch in ("har_cnn", "kws"):
        init_func, new_forward, module_type = (
            init_decision_conv_block,
            decision_conv_block_forward,
            "ConvBlock",
        )
    else:
        raise ValueError(f"Unknown model architecture {arch!r}")

    print("==> Transforming model...")
    apply_func(model, module_type, init_func, action_num=action_num)
    apply_func(model, "DecisionHead", collect_params)
    replace_func(model, module_type, new_forward)
    apply_func(model, "DecisionHead", normalize_head_weights)
