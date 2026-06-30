"""
Fine-tune a dynamic model with decision units held in deterministic mode.

Loads a decision-model checkpoint and fine-tunes the backbone weights with the
pruning gates frozen at their deterministic outputs. Requires a CUDA-capable GPU.

In dynamic mode (--dynamic), r_tgt is sampled per batch across [r_min, r_max]
so the backbone weights are fine-tuned to perform well across the full operating
range rather than at a single operating point (spec Phase 2).

Usage:
  # Static (matches Stage 2 operating point)
  python finetune.py --arch resnet56 --dataset cifar10 \
      --sparsity_level 0.4 --action_num 16

  # Dynamic (fine-tune across range)
  python finetune.py --arch resnet56 --dataset cifar10 --dynamic \
      --r_min 0.3 --r_max 0.7 --action_num 16
"""

import torch.nn.functional as F
import numpy as np
import torch
import os

from decision import (
    default_graph,
    apply_func,
    set_deterministic_value,
    set_pruning_threshold,
)
import misc

print = misc.logger.info

parser = misc.get_basic_argument_parser(default_wd=1e-4)
parser.add_argument("--sparsity_level", default=0.4, type=float)
parser.add_argument("--pruning_threshold", default=0.5, type=float)
parser.add_argument(
    "--action_num",
    default=None,
    type=int,
    help="Must match the value used in Stage 2. "
    "Defaults to architecture-specific value.",
)
parser.add_argument(
    "--dynamic",
    action="store_true",
    default=False,
    help="Fine-tune across r_tgt range instead of at a single "
    "sparsity_level. Must match whether Stage 2 used --dynamic.",
)
parser.add_argument(
    "--r_min",
    default=0.3,
    type=float,
    help="Lower bound of r_tgt training range (dynamic mode).",
)
parser.add_argument(
    "--r_max",
    default=0.7,
    type=float,
    help="Upper bound of r_tgt training range (dynamic mode).",
)

args = parser.parse_args()

args.num_classes = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}.get(
    args.dataset, 10
)
if args.action_num is None:
    args.action_num = misc.action_num(args.arch)
if args.lr is None:
    args.lr = 1e-3

args.device = "cuda"
torch.backends.cudnn.benchmark = True

args.logdir = "finetune-decision-%d/%s-%s/sparsity-%.2f" % (
    args.action_num,
    args.dataset,
    args.arch,
    args.sparsity_level,
)
misc.prepare_logging(args)

trainloader, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)
model_params = list(model.parameters())

misc.transform_model(model, args.arch, args.action_num)

print("==> Loading pretrained decision model...")
ckpt = torch.load(
    "logs/decision-%d/%s-%s/sparsity-%.2f/checkpoint.pth.tar"
    % (args.action_num, args.dataset, args.arch, args.sparsity_level),
    weights_only=True,
)
model.load_state_dict(ckpt["state_dict"])

model = model.to(args.device)

apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
apply_func(
    model,
    "DecisionHead",
    set_pruning_threshold,
    pruning_threshold=args.pruning_threshold,
)

optimizer = torch.optim.SGD(
    model_params,
    lr=args.lr,
    momentum=args.mm,
    weight_decay=args.wd,
)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[80, 120], gamma=0.1
)


def train(epoch):
    model.train()
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)
        B = data.shape[0]

        # Feed r_tgt to the frozen (deterministic) action head so its conditioning
        # still operates across the full range during backbone fine-tuning.
        if args.dynamic:
            r_tgt = torch.empty(B, 1, device=args.device).uniform_(
                args.r_min, args.r_max
            )
        else:
            r_tgt = torch.full((B, 1), args.sparsity_level, device=args.device)
        default_graph.append_tensor("r_tgt", r_tgt)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if i % args.log_interval == 0:
            selected_channels = default_graph.get_tensor_list("selected_channels")
            concat_channels = torch.cat(selected_channels, dim=1)
            sparsity = (concat_channels > args.pruning_threshold).float().mean()
            acc = (output.max(1)[1] == target).float().mean()
            if args.dynamic:
                print(
                    "Train Epoch: %d [%d/%d]\tLoss: %.4f, r_tgt_mean: %.4f, "
                    "Sparsity: %.4f, Accuracy: %.4f"
                    % (
                        epoch,
                        i,
                        len(trainloader),
                        loss.item(),
                        r_tgt.mean().item(),
                        sparsity.item(),
                        acc.item(),
                    )
                )
            else:
                print(
                    "Train Epoch: %d [%d/%d]\tLoss: %.4f, Sparsity: %.4f, Accuracy: %.4f"
                    % (
                        epoch,
                        i,
                        len(trainloader),
                        loss.item(),
                        sparsity.item(),
                        acc.item(),
                    )
                )


_EVAL_GRID_N = 5


def test():
    model.eval()

    if args.dynamic:
        r_tgt_grid = np.linspace(args.r_min, args.r_max, _EVAL_GRID_N).tolist()
        point_results = {}
        for r_val in r_tgt_grid:
            correct = 0
            densities = []
            with torch.no_grad():
                for data, target in testloader:
                    default_graph.clear_all_tensors()
                    B = data.shape[0]
                    r_t = torch.full((B, 1), r_val, device=args.device)
                    default_graph.append_tensor("r_tgt", r_t)
                    data, target = data.to(args.device), target.to(args.device)
                    output = model(data)
                    sel = default_graph.get_tensor_list("selected_channels")
                    cc = torch.cat(sel, dim=1)
                    densities.append(
                        (cc > args.pruning_threshold).float().mean().item()
                    )
                    correct += (output.max(1)[1] == target).float().sum().item()
            acc = correct / len(testloader.dataset)
            realized = float(np.mean(densities))
            point_results[r_val] = (realized, acc)
            print(
                "  [r_tgt=%.2f] keep-frac=%.4f, MACs-redux=%.2f%%, Acc=%.4f"
                % (r_val, realized, (1.0 - realized) * 100, acc)
            )
        mean_acc = float(np.mean([v[1] for v in point_results.values()]))
        mean_tracking_err = float(
            np.mean([abs(v[0] - k) for k, v in point_results.items()])
        )
        print(
            "Test sweep: mean_acc=%.4f, mean_tracking_err=%.4f\n"
            % (mean_acc, mean_tracking_err)
        )
        return mean_acc, mean_tracking_err

    else:
        test_loss = []
        test_sparsity = []
        correct = 0
        with torch.no_grad():
            for data, target in testloader:
                default_graph.clear_all_tensors()
                B = data.shape[0]
                r_t = torch.full((B, 1), args.sparsity_level, device=args.device)
                default_graph.append_tensor("r_tgt", r_t)
                data, target = data.to(args.device), target.to(args.device)
                output = model(data)

                selected_channels = default_graph.get_tensor_list("selected_channels")
                concat_channels = torch.cat(selected_channels, dim=1)

                test_loss.append(F.cross_entropy(output, target).item())
                test_sparsity.append(
                    (concat_channels > args.pruning_threshold).float().mean().item()
                )
                correct += (output.max(1)[1] == target).float().sum().item()

        acc = correct / len(testloader.dataset)
        print(
            "Test set: Loss: %.4f, Sparsity: %.4f, Accuracy: %.4f\n"
            % (np.mean(test_loss), np.mean(test_sparsity), acc)
        )
        return acc, np.mean(test_sparsity)


best_acc = 0.0
for epoch in range(args.epochs):
    train(epoch)
    result_a, result_b = test()
    scheduler.step()

    if args.dynamic:
        acc = result_a
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.logdir, "checkpoint.pth"))
            print(
                "New best @ Epoch %d, mean_acc=%.4f — checkpoint saved\n" % (epoch, acc)
            )
        else:
            print("Epoch %d, mean_acc=%.4f (best=%.4f)\n" % (epoch, acc, best_acc))
    else:
        acc, sparsity = result_a, result_b
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.logdir, "checkpoint.pth"))
            print(
                "New best @ Epoch %d, Accuracy=%.4f, Sparsity=%.4f — checkpoint saved\n"
                % (epoch, acc, sparsity)
            )
        else:
            print(
                "Epoch %d, Accuracy=%.4f, Sparsity=%.4f (best=%.4f)\n"
                % (epoch, acc, sparsity, best_acc)
            )
