"""Stage 5 — export a trained dynamic-pruning model to ONNX.

Produces two fp32 ONNX files, each taking TWO graph inputs — the image and
r_tgt — so a single exported model covers the entire operating range; no
per-r_tgt re-export needed:

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

Optionally (`export_fp16` / `export_int8`), also emits reduced-precision
variants of the batched file:

  <dataset>_<arch>-dynamic-batched-fp16.onnx
  <dataset>_<arch>-dynamic-batched-int8.onnx

fp16 is a lossless-enough straight cast (via onnxconverter_common) -- no
accuracy-recovery step is needed. int8 is post-training static quantization
(via onnxruntime.quantization, calibrated on real (image, r_tgt) batches from
the test set), which only quantizes Conv nodes -- the decision heads (Gemm/
MatMul) and final classifier are left in fp32 so channel-selection logits
keep their original precision. PTQ int8 *can* still lose accuracy on a
small/tight-capacity backbone (see the "Deviations from the paper" note on
ResNet10 having little redundancy to prune), so whenever `export_int8` is
enabled, `run()` evaluates fp32 vs int8 accuracy on the real test set via
onnxruntime and prints the comparison -- a measured fact for your checkpoint,
not a guess about whether you need quantization-aware fine-tuning.
"""

import os
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxoptimizer
import onnxsim
import torch
import torch.nn as nn
import torch.onnx
from onnxconverter_common.float16 import convert_float_to_float16
from onnxruntime.quantization import CalibrationDataReader, quantize_static

from . import checkpoints
from .config import ExportConfig
from .data import prepare_data
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
_INPUT_NAMES = ("input.1", "r_tgt")
_ACCURACY_R_VALUES = (0.1, 0.5, 0.9)  # spot-check points for the fp32-vs-int8 report


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
            opset_version=_ONNX_OPSET, dynamo=False, input_names=list(_INPUT_NAMES)
        )
        if dynamic_batch:
            kwargs["dynamic_axes"] = {"input.1": {0: "N"}, "r_tgt": {0: "N"}}
        torch.onnx.export(wrapped, (dummy_input, dummy_r_tgt), tmp_path, **kwargs)
        _optimize_and_save(tmp_path, output_path)
    finally:
        os.unlink(tmp_path)


def _export_fp16(input_path: str, output_path: str) -> None:
    """Cast an existing ONNX file's weights/activations to fp16.

    A straight cast (no calibration data, no retraining) -- fp16 has enough
    mantissa precision that this is standard practice with negligible
    accuracy impact. `keep_io_types=True` keeps the graph's inputs/outputs
    fp32 so callers don't need to change how they feed the model; only the
    internal compute/weights become fp16.
    """
    onnx_model = onnx.load_model(input_path)
    onnx_model = convert_float_to_float16(onnx_model, keep_io_types=True)
    onnx.save_model(onnx_model, output_path)


class _CalibrationReader(CalibrationDataReader):
    """Feeds real (image, r_tgt) batches from the test set to onnxruntime's
    static quantizer, so int8 activation ranges reflect actual deployment
    inputs across the full r_tgt operating range -- not just one point."""

    def __init__(self, testloader, r_values: tuple[float, ...], max_batches: int):
        self._batches = self._make_batches(testloader, r_values, max_batches)

    @staticmethod
    def _make_batches(testloader, r_values, max_batches):
        batches = []
        for data, _target in testloader:
            for r_val in r_values:
                if len(batches) >= max_batches:
                    return batches
                r_t = np.full((data.shape[0], 1), r_val, dtype=np.float32)
                batches.append(
                    {
                        _INPUT_NAMES[0]: data.numpy().astype(np.float32),
                        _INPUT_NAMES[1]: r_t,
                    }
                )
            if len(batches) >= max_batches:
                break
        return batches

    def get_next(self):
        if not self._batches:
            return None
        return self._batches.pop(0)


def _export_int8(input_path: str, output_path: str, testloader, cfg: ExportConfig) -> None:
    """Post-training static int8 quantization via onnxruntime.

    Only Conv nodes are quantized (`op_types_to_quantize=["Conv"]`) -- these
    are the backbone convs that dominate both size and MACs (see macs.py).
    The decision heads (Gemm/MatMul: fc1, r_proj) and final classifier are
    left in fp32 so channel-selection logits keep their original precision;
    quantization noise there could flip which action/channels get selected,
    which is a correctness risk out of proportion to the negligible size
    those tiny layers would save (see macs.py's overhead_frac finding that
    decision heads are already a tiny fraction of total MACs).
    """
    reader = _CalibrationReader(
        testloader, _ACCURACY_R_VALUES, cfg.int8_calibration_batches
    )
    quantize_static(
        input_path,
        output_path,
        calibration_data_reader=reader,
        per_channel=True,  # closes most of the PTQ accuracy gap vs per-tensor
        op_types_to_quantize=["Conv"],
    )


def _evaluate_onnx_accuracy(
    onnx_path: str, testloader, r_values: tuple[float, ...]
) -> dict[float, float]:
    """Top-1 accuracy of an exported ONNX model on the real test set, at a
    few r_tgt spot-check points. Used to give a measured (not guessed)
    answer to "did quantization hurt accuracy enough to need recovering?".
    """
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    results = {}
    for r_val in r_values:
        correct, total = 0, 0
        for data, target in testloader:
            r_t = np.full((data.shape[0], 1), r_val, dtype=np.float32)
            outputs = session.run(
                None,
                {
                    _INPUT_NAMES[0]: data.numpy().astype(np.float32),
                    _INPUT_NAMES[1]: r_t,
                },
            )
            logits = np.asarray(outputs[0])
            correct += (logits.argmax(axis=1) == target.numpy()).sum()
            total += target.shape[0]
        results[r_val] = correct / total
    return results


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

    if cfg.export_fp16:
        fp16_path = f"{stem}-batched-fp16.onnx"
        print(f"==> Exporting fp16 variant to {fp16_path} ...")
        _export_fp16(batched_path, fp16_path)
        print(f"Exported: {fp16_path}")

    if cfg.export_int8:
        _, testloader = prepare_data(
            cfg.data.name, cfg.data.train_batch_size, cfg.data.test_batch_size
        )
        int8_path = f"{stem}-batched-int8.onnx"
        print(f"==> Calibrating + exporting int8 variant to {int8_path} ...")
        _export_int8(batched_path, int8_path, testloader, cfg)
        print(f"Exported: {int8_path}")

        print("==> Evaluating fp32 vs int8 accuracy on the test set ...")
        fp32_acc = _evaluate_onnx_accuracy(batched_path, testloader, _ACCURACY_R_VALUES)
        int8_acc = _evaluate_onnx_accuracy(int8_path, testloader, _ACCURACY_R_VALUES)
        print(f"{'r_tgt':>8} | {'fp32 acc':>10} | {'int8 acc':>10} | {'drop':>8}")
        max_drop = 0.0
        for r_val in _ACCURACY_R_VALUES:
            drop = fp32_acc[r_val] - int8_acc[r_val]
            max_drop = max(max_drop, drop)
            print(
                f"{r_val:>8.2f} | {fp32_acc[r_val]:>10.4f} | {int8_acc[r_val]:>10.4f} | "
                f"{drop:>+8.4f}"
            )
        if max_drop > 0.02:
            print(
                f"⚠  int8 PTQ costs up to {max_drop * 100:.2f} accuracy points at some "
                "r_tgt. If that's too much for your use case, this needs "
                "quantization-aware fine-tuning (retraining the backbone with "
                "fake-quantization in the loop) rather than post-training "
                "quantization alone -- ask and I'll add that stage."
            )
        else:
            print(
                f"✓  int8 PTQ costs at most {max_drop * 100:.2f} accuracy points -- "
                "no fine-tuning needed for this checkpoint."
            )

    return single_path, batched_path
