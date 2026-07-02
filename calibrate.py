"""
Calibration sweep for a dynamic-target pruning model.

Loads a trained checkpoint (Stage 2 or Stage 3 fine-tune) and sweeps r_tgt
over a grid, recording realized keep-fraction and accuracy at each point.
Emits a calibration table mapping r_tgt → (keep-fraction, MACs-reduction,
accuracy) and checks monotonicity of realized density vs r_tgt.

The calibration table is the offline artefact needed to build the runtime
power→r_tgt policy (spec Step 6): desired latency/power → r_tgt via
piecewise-linear interpolation of the inverse table.

Usage:
  python calibrate.py --arch resnet56 --dataset cifar10 \
      --action_num 16 --r_min 0.3 --r_max 0.7 --grid_n 11

  # Use a fine-tuned Stage 3 checkpoint
  python calibrate.py --arch resnet56 --dataset cifar10 \
      --action_num 16 --r_min 0.3 --r_max 0.7 --finetuned

Note: MACs-reduction here is 1 - keep_fraction, a *channel-count-weighted*
proxy. A true per-layer spatial-size-weighted MACs profile is deferred.
"""

import csv
from collections.abc import Sized

import numpy as np
import torch

from decision import (
    default_graph,
    apply_func,
    set_deterministic_value,
    set_pruning_threshold,
)
import misc

print = misc.logger.info

parser = misc.get_basic_argument_parser(default_wd=0)
parser.add_argument(
    "--sparsity_level",
    default=0.4,
    type=float,
    help="Sparsity level used in Stage 2 (determines checkpoint path).",
)
parser.add_argument("--pruning_threshold", default=0.5, type=float)
parser.add_argument(
    "--action_num", default=None, type=int, help="Must match Stage 2 training."
)
parser.add_argument("--r_min", default=0.3, type=float, help="Sweep lower bound.")
parser.add_argument("--r_max", default=0.7, type=float, help="Sweep upper bound.")
parser.add_argument(
    "--grid_n",
    default=11,
    type=int,
    help="Number of r_tgt grid points (inclusive endpoints).",
)
parser.add_argument(
    "--finetuned",
    action="store_true",
    default=False,
    help="Load Stage 3 fine-tuned checkpoint instead of Stage 2.",
)
parser.add_argument(
    "--out_csv",
    default=None,
    type=str,
    help="Optional path to write calibration table as CSV.",
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

_, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)
misc.transform_model(model, args.arch, args.action_num)

if args.finetuned:
    ckpt_path = "logs/finetune-decision-%d/%s-%s/sparsity-%.2f/checkpoint.pth" % (
        args.action_num,
        args.dataset,
        args.arch,
        args.sparsity_level,
    )
    print("==> Loading Stage 3 fine-tuned checkpoint from %s ..." % ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
else:
    ckpt_path = "logs/decision-%d/%s-%s/sparsity-%.2f/checkpoint.pth.tar" % (
        args.action_num,
        args.dataset,
        args.arch,
        args.sparsity_level,
    )
    print("==> Loading Stage 2 checkpoint from %s ..." % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
apply_func(
    model,
    "DecisionHead",
    set_pruning_threshold,
    pruning_threshold=args.pruning_threshold,
)

r_tgt_grid = np.linspace(args.r_min, args.r_max, args.grid_n).tolist()

print(
    "\nCalibration sweep: %d points in [%.2f, %.2f]"
    % (args.grid_n, args.r_min, args.r_max)
)
print("-" * 70)
print("  r_tgt  | keep-frac | MACs-redux | accuracy | tracking-err")
print("-" * 70)

assert isinstance(testloader.dataset, Sized), "test dataset must implement __len__"
n_test = len(testloader.dataset)

rows = []
for r_val in r_tgt_grid:
    correct = 0
    densities = []
    with torch.no_grad():
        for data, target in testloader:
            default_graph.clear_all_tensors()
            B = data.shape[0]
            r_t = torch.full((B, 1), r_val, device=device)
            default_graph.append_tensor("r_tgt", r_t)
            data, target = data.to(device), target.to(device)
            output = model(data)
            sel = default_graph.get_tensor_list("selected_channels")
            cc = torch.cat(sel, dim=1)
            densities.append((cc > args.pruning_threshold).float().mean().item())
            correct += (output.max(1)[1] == target).float().sum().item()

    acc = correct / n_test
    keep_frac = float(np.mean(densities))
    macs_redux = 1.0 - keep_frac
    tracking_err = abs(keep_frac - r_val)
    rows.append((r_val, keep_frac, macs_redux, acc, tracking_err))
    print(
        "  %.4f  |  %.4f   |   %.4f   | %.4f   |   %.4f"
        % (r_val, keep_frac, macs_redux, acc, tracking_err)
    )

print("-" * 70)

# Monotonicity check: realized keep-fraction should be non-decreasing with r_tgt
keep_fracs = [r[1] for r in rows]
non_monotone = [
    i for i in range(1, len(keep_fracs)) if keep_fracs[i] < keep_fracs[i - 1] - 1e-4
]
if non_monotone:
    print(
        "\n⚠  Non-monotone realized density at r_tgt grid points: %s "
        "(indices where density decreased). Consider increasing --gamma or "
        "widening the training range." % non_monotone
    )
else:
    print("\n✓  Realized density is monotonically non-decreasing with r_tgt.")

mean_tracking_err = float(np.mean([r[4] for r in rows]))
print("Mean tracking error across grid: %.4f" % mean_tracking_err)

# Markdown table to stdout
print("\n### Calibration table (r_tgt → operating point)\n")
print("| r_tgt | keep-frac | MACs-redux | accuracy | tracking-err |")
print("|---|---|---|---|---|")
for r_val, keep_frac, macs_redux, acc, tracking_err in rows:
    print(
        "| %.2f | %.4f | %.2f%% | %.4f | %.4f |"
        % (r_val, keep_frac, macs_redux * 100, acc, tracking_err)
    )
print("")
print(
    "**Note:** MACs-redux = 1 − keep-frac is a channel-count-weighted proxy. "
    "True per-layer spatial-size-weighted MACs profiling is deferred."
)

# Optional CSV output
if args.out_csv:
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["r_tgt", "keep_frac", "macs_redux", "accuracy", "tracking_err"]
        )
        for row in rows:
            writer.writerow(["%.4f" % v for v in row])
    print("Calibration table written to %s" % args.out_csv)
