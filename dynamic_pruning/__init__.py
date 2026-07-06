"""Runtime-adjustable dynamic channel pruning via target-conditioned decision heads.

Extends Wang et al. (2020) "Dynamic Network Pruning with Interpretable
Layerwise Channel Selection" so a single trained model trades accuracy for
MACs at inference time by varying a scalar target keep-fraction r_tgt,
instead of needing one model per operating point.

Submodules:
  config          Hydra/dataclass config schema for every pipeline stage
  checkpoints     logs/... path conventions shared by every stage
  data            dataset loading (CIFAR-10/100, HAR, KWS)
  decision        TorchGraph registry + DecisionHead gating logic
  logging_utils   console/file/TensorBoard run logging
  models          backbone architectures (ResNet, HAR-CNN, KWS-CNN)
  training        Stage 1/2/2D/3/3D training loops
  calibrate       Stage 4D r_tgt calibration sweep
  export          Stage 5 ONNX export
  optuna_search   hyperparameter search harness (scripts/tune.py)
"""

from . import (
    calibrate,
    checkpoints,
    config,
    data,
    decision,
    export,
    logging_utils,
    models,
    optuna_search,
    training,
)

__version__ = "0.1.0"

__all__ = [
    "calibrate",
    "checkpoints",
    "config",
    "data",
    "decision",
    "export",
    "logging_utils",
    "models",
    "optuna_search",
    "training",
]
