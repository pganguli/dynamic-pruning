"""
Export a trained dynamic-pruning model to ONNX.

Loads a Stage 2 (or Stage 3 fine-tuned) checkpoint, latches a target keep-fraction
r_tgt in the TorchGraph registry, then exports two ONNX files per operating point:

  <dataset>_<arch>-r<r_tgt>-single.onnx   batch=1, pruning_threshold=0 (soft gates)
  <dataset>_<arch>-r<r_tgt>-batched.onnx  dynamic batch, hard gates at pruning_threshold

The single-input file is intended for on-device inference where the NodPA runtime
re-implements the decision-map generation in C and applies its own thresholding.
The batched file uses hard 0/1 gates (values below pruning_threshold zeroed) and is
suitable for accuracy evaluation via ONNX runtime.

r_tgt is baked into the graph at tracing time (it is a Python-side constant read from
TorchGraph, not a model input tensor). Export once per desired operating point.

Usage:
  # Static model (one operating point)
  python export.py --arch resnet56 --dataset cifar10 \
      --sparsity_level 0.4 --action_num 16 --r_tgt 0.4

  # Dynamic model — export several operating points
  python export.py --arch resnet56 --dataset cifar10 \
      --action_num 16 --r_tgt 0.3
  python export.py --arch resnet56 --dataset cifar10 \
      --action_num 16 --r_tgt 0.5
  python export.py --arch resnet56 --dataset cifar10 \
      --action_num 16 --r_tgt 0.7
"""

import os
import os.path
import tempfile

import torch
import torch.onnx
import onnx
import onnxoptimizer
import onnxsim

import misc
from decision import (
    default_graph,
    apply_func,
    set_deterministic_value,
    set_pruning_threshold,
)


def optimize_model(input_path: str, model_name: str) -> None:
    onnx_model = onnx.load_model(input_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    onnx_model = onnxoptimizer.optimize(onnx_model)
    result = onnxsim.simplify(onnx_model)
    onnx_model = result[0] if isinstance(result, tuple) else result
    onnx.save_model(onnx_model, model_name)


def main():
    parser = misc.get_basic_argument_parser(default_wd=0)
    parser.add_argument("--sparsity_level", default=0.1, type=float)
    parser.add_argument("--pruning_threshold", default=0.5, type=float)
    parser.add_argument(
        "--action_num",
        default=None,
        type=int,
        help="Must match the value used in Stage 2. "
        "Defaults to architecture-specific value.",
    )
    parser.add_argument(
        "--r_tgt",
        default=0.5,
        type=float,
        help="Target keep-fraction to latch for this export. The "
        "conditioned action head bakes this value into the ONNX "
        "graph at tracing time. Export once per desired operating "
        "point. The on-device decision-map generator must mirror "
        "this value (see spec Step 7).",
    )
    parser.add_argument(
        "--finetuned",
        action="store_true",
        default=False,
        help="Load Stage 3D fine-tuned checkpoint instead of Stage 2. "
        "Recommended: the fine-tuned model has significantly higher "
        "accuracy at the same sparsity level.",
    )
    args = parser.parse_args()

    args.num_classes = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}.get(
        args.dataset, 10
    )
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
    misc.transform_model(model, args.arch, args.action_num)
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)

    if args.finetuned:
        ckpt_path = "logs/finetune-decision-%d/%s-%s/sparsity-%.2f/checkpoint.pth" % (
            args.action_num,
            args.dataset,
            args.arch,
            args.sparsity_level,
        )
        print("==> Loading Stage 3 fine-tuned checkpoint from %s ..." % ckpt_path)
        state_dict = torch.load(
            ckpt_path, map_location=torch.device("cpu"), weights_only=True
        )
        model.load_state_dict(state_dict)
    else:
        print("==> Loading Stage 2 checkpoint...")
        checkpoint = torch.load(
            os.path.join(args.logdir, "checkpoint.pth.tar"),
            map_location=torch.device("cpu"),
            weights_only=True,
        )
        model.load_state_dict(checkpoint["state_dict"])

    # Latch r_tgt in the registry so tracing bakes this operating point into
    # the ONNX graph. Run export.py once per r_tgt to cover multiple points.
    # The on-device decision-map generator must be updated to mirror this value
    # (see spec Step 7 — NodPA on-device integration).
    default_graph.clear_tensor_list("r_tgt")
    default_graph.append_tensor(
        "r_tgt",
        torch.full((1, 1), args.r_tgt),  # batch size 1 matches the single-input export
    )

    if args.dataset.startswith("cifar"):
        dummy_input = torch.zeros((1, 3, 32, 32))
    elif args.dataset == "har":
        dummy_input = torch.zeros((1, 9, 128))
    elif args.dataset == "kws":
        dummy_input = torch.zeros((1, 1, 25, 10))
    else:
        raise ValueError("Unknown dataset: %s" % args.dataset)

    onnx_opset = 17
    stem = f"{args.dataset}_{args.arch}-r{args.r_tgt:.2f}"

    # Single-input export: pruning_threshold left at 0 (soft fractional gates).
    # On-device (NodPA) the decision-map generator applies its own thresholding in C.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_single = tmp.name
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            tmp_single,
            opset_version=onnx_opset,
            dynamo=False,
        )
        optimize_model(tmp_single, f"{stem}-single.onnx")
    finally:
        os.unlink(tmp_single)

    # Switch to hard 0/1 gates before the batched export so ONNX-runtime evaluation
    # uses the same threshold as the training-time sparsity metric.
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=args.pruning_threshold,
    )

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        tmp_batched = tmp.name
    try:
        torch.onnx.export(
            model,
            (dummy_input,),
            tmp_batched,
            opset_version=onnx_opset,
            dynamo=False,
            input_names=["input.1"],
            dynamic_axes={
                "input.1": {0: "N"},
            },
        )
        optimize_model(tmp_batched, f"{stem}-batched.onnx")
    finally:
        os.unlink(tmp_batched)


if __name__ == "__main__":
    main()
