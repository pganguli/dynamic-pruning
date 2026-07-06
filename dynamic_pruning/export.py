"""Stage 5 — export a trained dynamic-pruning model to ONNX.

Produces two ONNX files, each taking TWO graph inputs — the image and r_tgt —
so a single exported model covers the entire operating range; no per-r_tgt
re-export needed:

  <dataset>_<arch>-dynamic-single.onnx   batch=1, pruning_threshold=0 (soft gates)
  <dataset>_<arch>-dynamic-batched.onnx  dynamic batch, hard gates at pruning_threshold

The single-input file is intended for on-device inference where the NodPA
runtime re-implements the decision-map generation in C and applies its own
thresholding. The batched file uses hard 0/1 gates and is suitable for
accuracy evaluation via ONNX runtime.

r_tgt is threaded through training via a Python-side TorchGraph registry (see
decision.py), not a forward() argument, so it would normally be baked into
the ONNX graph as a constant at trace time. ExportWrapper routes r_tgt
through the traced forward() call itself — setting the registry from inside
the traced function, not before it — so the exporter captures it as a
genuine second graph input instead of a frozen constant.
"""

import os
import tempfile
from typing import Any

import onnx
import onnxoptimizer
import onnxsim
import torch
import torch.nn as nn
import torch.onnx

from . import checkpoints
from .config import ExportConfig
from .decision import (
    apply_func,
    default_graph,
    set_deterministic_value,
    set_pruning_threshold,
)
from .training.common import (
    default_action_num,
    initialize_model,
    num_classes,
    transform_model,
)

__all__ = ["ExportWrapper", "run"]

_ONNX_OPSET = 17
_DUMMY_SHAPES = {
    "cifar10": (1, 3, 32, 32),
    "cifar100": (1, 3, 32, 32),
    "har": (1, 9, 128),
    "kws": (1, 1, 25, 10),
}


class ExportWrapper(nn.Module):
    """Routes r_tgt through the traced forward() call so ONNX export captures
    it as a genuine graph input rather than a constant frozen at trace time."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, r_tgt: torch.Tensor):
        default_graph.clear_tensor_list("r_tgt")
        default_graph.append_tensor("r_tgt", r_tgt)
        return self.model(x)


def _optimize_and_save(input_path: str, output_path: str) -> None:
    onnx_model = onnx.load_model(input_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx_model = onnxoptimizer.optimize(onnx_model)
    result = onnxsim.simplify(onnx_model)
    onnx_model = result[0] if isinstance(result, tuple) else result
    onnx.save_model(onnx_model, output_path)


def _export_one(
    wrapped: nn.Module, dummy_input, dummy_r_tgt, output_path: str, dynamic_batch: bool
) -> None:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        kwargs: dict[str, Any] = dict(
            opset_version=_ONNX_OPSET, dynamo=False, input_names=["input.1", "r_tgt"]
        )
        if dynamic_batch:
            kwargs["dynamic_axes"] = {"input.1": {0: "N"}, "r_tgt": {0: "N"}}
        torch.onnx.export(wrapped, (dummy_input, dummy_r_tgt), tmp_path, **kwargs)
        _optimize_and_save(tmp_path, output_path)
    finally:
        os.unlink(tmp_path)


def run(cfg: ExportConfig) -> tuple[str, str]:
    action_num = cfg.decision.action_num or default_action_num(cfg.model.arch)

    model = initialize_model(cfg.data.name, cfg.model.arch, num_classes(cfg.data.name))
    transform_model(model, cfg.model.arch, action_num)
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)

    if cfg.finetuned:
        ckpt_path = checkpoints.finetune_checkpoint(
            action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
        )
        print(f"==> Loading Stage 3 fine-tuned checkpoint from {ckpt_path} ...")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
        )
    else:
        ckpt_path = checkpoints.decision_checkpoint(
            action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
        )
        print(f"==> Loading Stage 2 checkpoint from {ckpt_path} ...")
        checkpoint = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(checkpoint["state_dict"])

    wrapped = ExportWrapper(model)

    if cfg.data.name not in _DUMMY_SHAPES:
        raise ValueError(f"Unknown dataset {cfg.data.name!r}")
    dummy_input = torch.zeros(_DUMMY_SHAPES[cfg.data.name])
    dummy_r_tgt = torch.full((1, 1), 0.5)

    stem = f"{cfg.data.name}_{cfg.model.arch}-dynamic"
    single_path = f"{stem}-single.onnx"
    batched_path = f"{stem}-batched.onnx"

    # Single-input export: pruning_threshold left at 0 (soft fractional gates).
    _export_one(wrapped, dummy_input, dummy_r_tgt, single_path, dynamic_batch=False)

    # Switch to hard 0/1 gates before the batched export so ONNX-runtime
    # evaluation uses the same threshold as the training-time sparsity metric.
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=cfg.decision.pruning_threshold,
    )
    _export_one(wrapped, dummy_input, dummy_r_tgt, batched_path, dynamic_batch=True)

    return single_path, batched_path
