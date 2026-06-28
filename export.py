"""
Export a trained dynamic-pruning model to ONNX.

Loads a checkpoint from the decision model log directory and exports two ONNX files:
  <dataset>_<arch>-single.onnx   (batch size = 1, for on-device inference)
  <dataset>_<arch>-batched.onnx  (dynamic batch, for accuracy evaluation)

Usage: python train/export.py --arch resnet10 --dataset cifar10
"""

import io
import os.path
from typing import IO

import torch
import torch.onnx
import onnx
import onnxoptimizer
import onnxsim

import misc
from decision import (
    apply_func,
    set_deterministic_value,
    set_pruning_threshold,
)


def optimize_model(pytorch_exported_model: IO[bytes], model_name: str):
    onnx_model = onnx.load_model(pytorch_exported_model)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx_model = onnxsim.simplify(onnx_model)  # 0.5+ returns model directly, raises on failure
    onnx.save_model(onnx_model, model_name)


def main():
    # No training involved - use dummy values to wd
    parser = misc.get_basic_argument_parser(default_wd=0)
    parser.add_argument("--sparsity_level", default=0.1, type=float)
    parser.add_argument("--pruning_threshold", default=0.5, type=float)
    parser.add_argument("--action_num", default=None, type=int,
                        help="Must match the value used in Stage 2. "
                             "Defaults to architecture-specific value.")
    args = parser.parse_args()

    args.num_classes = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}.get(args.dataset, 10)
    if args.action_num is None:
        args.action_num = misc.action_num(args.arch)
    if args.lr is None:
        args.lr = 0.0

    args.logdir = "decision-%d/%s-%s/sparsity-%.2f" % (
        args.action_num,
        args.dataset,
        args.arch,
        args.sparsity_level,
    )
    misc.prepare_logging(args)

    model = misc.initialize_model(args.dataset, args.arch, args.num_classes)

    print("==> Loading pretrained model...")
    checkpoint = torch.load(
        os.path.join(args.logdir, "checkpoint.pth.tar"),
        map_location=torch.device("cpu"),
        weights_only=True,
    )

    misc.transform_model(model, args.arch, args.action_num)

    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)

    model.load_state_dict(checkpoint["state_dict"])

    pytorch_exported_model_single = io.BytesIO()
    pytorch_exported_model_batched = io.BytesIO()

    if args.dataset.startswith("cifar"):
        dummy_input = torch.zeros((1, 3, 32, 32))
    elif args.dataset == "har":
        dummy_input = torch.zeros((1, 9, 128))
    elif args.dataset == "kws":
        dummy_input = torch.zeros((1, 1, 25, 10))

    onnx_opset = 17

    torch.onnx.export(
        model,
        dummy_input,
        pytorch_exported_model_single,
        opset_version=onnx_opset,
        dynamo=False,
    )

    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=args.pruning_threshold,
    )

    torch.onnx.export(
        model,
        dummy_input,
        pytorch_exported_model_batched,
        opset_version=onnx_opset,
        dynamo=False,
        input_names=["input.1"],
        dynamic_axes={
            "input.1": {0: "N"},
        },
    )

    pytorch_exported_model_single.seek(0)
    pytorch_exported_model_batched.seek(0)

    optimize_model(
        pytorch_exported_model_single, f"{args.dataset}_{args.arch}-single.onnx"
    )
    optimize_model(
        pytorch_exported_model_batched, f"{args.dataset}_{args.arch}-batched.onnx"
    )


if __name__ == "__main__":
    main()
