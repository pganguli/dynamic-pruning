"""
Train a model with dynamic channel-pruning decision units.

Loads a pre-trained checkpoint from logs/pretrained/ and jointly trains the
pruning decision heads and the backbone network with a sparsity-regularisation term.
Requires a CUDA-capable GPU.

Static mode (default):
  A single target keep-fraction --sparsity_level is used throughout training.
  The model learns to select channels such that mean realized density ≈ r.

Dynamic mode (--dynamic):
  A per-batch target keep-fraction r_tgt is sampled uniformly in [r_min, r_max]
  and fed to the conditioned action head, producing a model that follows r_tgt
  at inference time. This trains a single shared-weight model covering a range
  of operating points.

Usage:
  # Static (single operating point)
  python main.py --arch resnet56 --dataset cifar10 --sparsity_level 0.4 \
      --gamma 2.2 --action_num 16 --epochs 400

  # Dynamic (knob model)
  python main.py --arch resnet56 --dataset cifar10 --dynamic \
      --r_min 0.3 --r_max 0.7 --gamma 2.2 --action_num 16 --epochs 400
"""

import torch.nn.functional as F
import numpy as np
import torch
import os

from decision import (
    default_graph,
    apply_func,
    set_deterministic_value,
    normalize_head_weights,
    set_pruning_threshold,
)
import misc

np.set_printoptions(precision=2, linewidth=160)
print = misc.logger.info

parser = misc.get_basic_argument_parser(default_wd=1e-9)
parser.add_argument(
    "--sparsity_level",
    default=0.1,
    type=float,
    help="Target fraction of channels to keep active (r in paper). "
    "Used as the single operating point in static mode, and as "
    "the r_tgt constant fed to the conditioned head when not in "
    "dynamic mode.",
)
parser.add_argument(
    "--pruning_threshold",
    default=0.5,
    type=float,
    help="Hard gate threshold at evaluation time.",
)
parser.add_argument(
    "--gamma",
    default=1.0,
    type=float,
    help="Regularization balance factor γ (Eq. 1 in Wang et al. 2020).",
)
parser.add_argument(
    "--gamma_under",
    default=0.7,
    type=float,
    help="Multiplier on gamma when sparsity is below target (static "
    "mode only; prevents gate collapse). Ignored in --dynamic mode "
    "where symmetric tracking is used.",
)
parser.add_argument(
    "--action_num",
    default=None,
    type=int,
    help="Number of channel-selection actions m per decision unit. "
    "Defaults to architecture-specific value (5 for HAR/KWS, 40 "
    "for ResNet). Use 16 for dynamic-target training.",
)
# Dynamic target arguments
parser.add_argument(
    "--dynamic",
    action="store_true",
    default=False,
    help="Enable dynamic-target mode: sample r_tgt per batch in "
    "[r_min, r_max] and feed it to the conditioned action head. "
    "Produces a single model that can trade accuracy for MACs at "
    "inference time by varying r_tgt.",
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
parser.add_argument(
    "--r_endpoint_prob",
    default=0.1,
    type=float,
    help="Per-sample probability of oversampling an endpoint (r_min or "
    "r_max) instead of uniform-sampling r_tgt. Ensures extreme "
    "operating points are well trained.",
)

args = parser.parse_args()

args.num_classes = {"cifar10": 10, "cifar100": 100, "har": 6, "kws": 12}.get(
    args.dataset, 10
)
if args.action_num is None:
    args.action_num = misc.action_num(args.arch)
if args.lr is None:
    args.lr = misc.learning_rate(args.arch)

args.device = "cuda"
torch.backends.cudnn.benchmark = True

args.logdir = "decision-%d/%s-%s/sparsity-%.2f" % (
    args.action_num,
    args.dataset,
    args.arch,
    args.sparsity_level,
)
misc.prepare_logging(args)

trainloader, testloader = misc.prepare_data(args.dataset, args.train_batch_size)

model = misc.initialize_model(args.dataset, args.arch, args.num_classes)

# Collect backbone params *before* transform_model injects decision heads so
# that head/gate params stay out of optimizer_model (SGD).
model_params = list(model.parameters())

print("==> Loading pretrained model...")
model.load_state_dict(
    torch.load(
        "logs/pretrained/%s/%s/checkpoint.pth" % (args.dataset, args.arch),
        weights_only=True,
    )
)

misc.transform_model(model, args.arch, args.action_num)

model = model.to(args.device)

head_params = default_graph.get_tensor_list("head_params")
gate_params = default_graph.get_tensor_list("gate_params")

optimizer_gate = torch.optim.Adam(head_params + gate_params, lr=args.lr)
optimizer_model = torch.optim.SGD(
    model_params,
    lr=args.lr,
    momentum=args.mm,
    weight_decay=args.wd,
)

scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_gate, T_max=args.epochs, eta_min=args.lr * 1e-2
)
scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_model, T_max=args.epochs, eta_min=args.lr * 1e-2
)


def _sample_r_tgt(batch_size):
    """Sample per-sample r_tgt for a batch (dynamic mode).

    Draws uniformly in [r_min, r_max], then with probability r_endpoint_prob
    replaces some samples with exactly r_min or r_max so extreme operating
    points are well covered.
    """
    r_tgt = torch.empty(batch_size, 1, device=args.device).uniform_(
        args.r_min, args.r_max
    )
    if args.r_endpoint_prob > 0:
        is_endpoint = torch.rand(batch_size, device=args.device) < args.r_endpoint_prob
        which_end = torch.rand(batch_size, device=args.device) < 0.5
        endpoints = torch.where(
            which_end,
            torch.full((batch_size,), args.r_min, device=args.device),
            torch.full((batch_size,), args.r_max, device=args.device),
        )
        r_tgt[:, 0] = torch.where(is_endpoint, endpoints, r_tgt[:, 0])
    return r_tgt


def train(epoch):
    model.train()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=False)
    for i, (data, target) in enumerate(trainloader):
        default_graph.clear_all_tensors()

        data = data.to(args.device)
        target = target.to(args.device)
        B = data.shape[0]

        # --- set r_tgt for this batch (both forward passes share it) ---
        if args.dynamic:
            r_tgt = _sample_r_tgt(B)
        else:
            r_tgt = torch.full((B, 1), args.sparsity_level, device=args.device)
        default_graph.append_tensor("r_tgt", r_tgt)

        # --- gate / head optimizer step ---
        optimizer_gate.zero_grad()
        output = model(data)
        loss_ce = F.cross_entropy(output, target)

        selected_channels = default_graph.get_tensor_list("selected_channels")
        concat_channels = torch.cat(selected_channels, dim=1)  # [B, sum_C]

        # Sigmoid soft-sparsity: differentiable proxy for fraction-above-threshold.
        # In dynamic mode: per-sample density [B,1] vs per-sample r_tgt [B,1].
        # In static mode: scalar density vs scalar sparsity_level (asymmetric).
        soft = torch.sigmoid(10.0 * (concat_channels - args.pruning_threshold))

        if args.dynamic:
            # Expected-density regularizer: for each decision head independently,
            # push the *routing-probability-weighted* density toward r_tgt,
            # rather than only the aggregate realized density across all heads.
            # The aggregate-only form let any individual head's routing stay
            # r_tgt-independent as long as other heads compensated on average
            # (population-matching instead of per-sample conditional routing,
            # confirmed via CPU repro: loss converged low while eval-time
            # routing never differentiated by r_tgt). Each head's expected
            # density is differentiable end-to-end through action_probs
            # (hence through fc1 + r_proj) without going through the noisy
            # Gumbel-sampled selection, giving a much cleaner gradient into
            # the routing decision itself.
            action_routing = default_graph.get_tensor_list("action_routing")
            per_head_losses = []
            for action_probs, channel_gates in action_routing:
                gate_density = torch.sigmoid(
                    10.0 * (channel_gates - args.pruning_threshold)
                ).mean(dim=1)  # [action_num]
                expected_density = action_probs @ gate_density  # [B]
                per_head_losses.append(
                    (expected_density.unsqueeze(1) - r_tgt) ** 2
                )
            loss_reg = args.gamma * torch.cat(per_head_losses, dim=1).mean()
        else:
            soft_sparsity = soft.mean()  # scalar
            diff = soft_sparsity - args.sparsity_level
            sparsity_frac = (concat_channels > args.pruning_threshold).float().mean()
            gamma_eff = (
                args.gamma
                if sparsity_frac > args.sparsity_level
                else args.gamma * args.gamma_under
            )
            loss_reg = gamma_eff * diff**2

        loss = loss_ce + loss_reg

        loss.backward()

        if args.dynamic and i % args.log_interval == 0:
            # Diagnostic: compare gradient magnitude reaching r_proj (the
            # r_tgt conditioning pathway) against fc1 (the feature pathway)
            # across all decision heads, to check whether r_tgt's gradient
            # signal is actually present before debugging further upstream.
            r_proj_norms = []
            fc1_norms = []
            for m in model.modules():
                if m.__class__.__name__ == "DecisionHead":
                    if m.r_proj.weight.grad is not None:
                        r_proj_norms.append(m.r_proj.weight.grad.norm().item())
                    if m.fc1.weight.grad is not None:
                        fc1_norms.append(m.fc1.weight.grad.norm().item())
            if r_proj_norms and fc1_norms:
                print(
                    "  [grad-diag] mean |grad r_proj|: %.6e, mean |grad fc1|: %.6e, "
                    "ratio: %.6f"
                    % (
                        np.mean(r_proj_norms),
                        np.mean(fc1_norms),
                        np.mean(r_proj_norms) / (np.mean(fc1_norms) + 1e-12),
                    )
                )

            # Diagnostic: compare forward-pass logit magnitude contributed by
            # r_proj (the r_tgt term) against fc1 (the feature term). If fc1's
            # output is much larger in scale, r_tgt's contribution gets
            # swamped after softmax regardless of how healthy its gradient is.
            logit_diag = default_graph.get_tensor_list("head_logit_diag")
            if logit_diag:
                fc1_mags = [fo.abs().mean().item() for fo, _ in logit_diag]
                rproj_mags = [ro.abs().mean().item() for _, ro in logit_diag]
                print(
                    "  [scale-diag] mean |fc1_out|: %.6f, mean |r_proj_out|: %.6f, "
                    "ratio (fc1/r_proj): %.4f"
                    % (
                        np.mean(fc1_mags),
                        np.mean(rproj_mags),
                        np.mean(fc1_mags) / (np.mean(rproj_mags) + 1e-12),
                    )
                )

        optimizer_gate.step()

        for p in gate_params:
            p.data.clamp_(0, 1)

        apply_func(model, "DecisionHead", normalize_head_weights)

        # --- backbone optimizer step (CE only) ---
        optimizer_model.zero_grad()
        output = model(data)
        loss_model = F.cross_entropy(output, target)
        loss_model.backward()
        optimizer_model.step()

        if i % args.log_interval == 0:
            # concat_channels was computed after the first forward (L layers).
            # The list now has 2L entries after the second forward, but the
            # sparsity/density numbers from the first-forward snapshot are stable.
            sparsity = (concat_channels > args.pruning_threshold).float().mean().item()
            acc = (output.max(1)[1] == target).float().mean().item()
            if args.dynamic:
                mean_rtgt = r_tgt.mean().item()
                print(
                    "Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, "
                    "r_tgt_mean: %.4f, Sparsity: %.4f, Accuracy: %.4f"
                    % (
                        epoch,
                        i,
                        len(trainloader),
                        loss.item(),
                        loss_ce.item(),
                        loss_reg.item(),
                        mean_rtgt,
                        sparsity,
                        acc,
                    )
                )
            else:
                mean_gate = concat_channels.mean().item()
                print(
                    "Train Epoch: %d [%d/%d]\tLoss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, "
                    "Sparsity: %.4f, Mean gate: %.4f, Accuracy: %.4f"
                    % (
                        epoch,
                        i,
                        len(trainloader),
                        loss.item(),
                        loss_ce.item(),
                        loss_reg.item(),
                        sparsity,
                        mean_gate,
                        acc,
                    )
                )


# r_tgt evaluation grid for dynamic mode: 5 evenly spaced points
_EVAL_GRID_N = 5


def test():
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=args.pruning_threshold,
    )

    if args.dynamic:
        # Sweep r_tgt over the evaluation grid; one full test-set pass per point.
        r_tgt_grid = np.linspace(args.r_min, args.r_max, _EVAL_GRID_N).tolist()
        point_results = {}  # r_tgt_val -> (realized_density, accuracy)

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
        # Static mode: existing single-pass test
        test_loss_ce = []
        test_loss_reg = []
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

                test_loss_ce.append(F.cross_entropy(output, target).item())
                test_sparsity_frac = (
                    (concat_channels > args.pruning_threshold).float().mean()
                )
                test_soft = torch.sigmoid(
                    10.0 * (concat_channels - args.pruning_threshold)
                ).mean()
                test_diff = test_soft - args.sparsity_level
                test_gamma_eff = (
                    args.gamma
                    if test_sparsity_frac > args.sparsity_level
                    else args.gamma * args.gamma_under
                )
                test_loss_reg.append((test_gamma_eff * test_diff**2).item())
                test_sparsity.append(test_sparsity_frac.item())

                pred = output.max(1)[1]
                correct += (pred == target).float().sum().item()

        actions = torch.stack(default_graph.get_tensor_list("sampled_actions")).permute(
            1, 0
        )
        acc = correct / len(testloader.dataset)
        print(
            "Test set: Loss: %.4f, Loss_CE: %.4f, Loss_REG: %.4f, "
            "Sparsity: %.4f, Accuracy: %.4f"
            % (
                np.mean(test_loss_ce) + np.mean(test_loss_reg),
                np.mean(test_loss_ce),
                np.mean(test_loss_reg),
                np.mean(test_sparsity),
                acc,
            )
        )
        print("   First 10 sampled actions: \n" + str(actions[:10].cpu().numpy()))
        print("   First 10 targets: " + str(target[:10].cpu().numpy()) + "\n")
        return acc, np.mean(test_sparsity)


def save_checkpoint(state, filepath):
    torch.save(state, os.path.join(filepath, "checkpoint.pth.tar"))


_T_START = 5.0
_T_END = 0.5
_SPARSITY_TOL = 0.05  # static mode: sparsity must be in [target - tol, target]
_TRACKING_TOL = 0.05  # dynamic mode: mean |realized - r_tgt| must be < this

best_acc = 0.0
best_metric = float("inf")  # static: sparsity_dist; dynamic: mean_tracking_err
ever_on_target = False

for epoch in range(args.epochs):
    # Linear temperature annealing (Wang et al. 2020, Implementation Details)
    temperature = _T_START + (_T_END - _T_START) * epoch / max(args.epochs - 1, 1)
    default_graph.clear_tensor_list("temperature")
    default_graph.append_tensor("temperature", temperature)

    train(epoch)
    metric_a, metric_b = test()  # (acc, sparsity) or (mean_acc, mean_tracking_err)
    scheduler_gate.step()
    scheduler_model.step()

    if args.dynamic:
        acc = metric_a
        mean_tracking_err = metric_b
        on_target = mean_tracking_err < _TRACKING_TOL
        dist = mean_tracking_err
    else:
        acc = metric_a
        sparsity = metric_b
        on_target = (
            (args.sparsity_level - _SPARSITY_TOL) <= sparsity <= args.sparsity_level
        )
        dist = abs(sparsity - args.sparsity_level)

    if on_target:
        ever_on_target = True

    should_save = (on_target and acc > best_acc) or (
        not ever_on_target and dist < best_metric
    )

    if should_save:
        if on_target:
            best_acc = acc
        best_metric = dist
        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": model.state_dict(),
            },
            filepath=args.logdir,
        )
        label = (
            "New best"
            if on_target
            else "Closest to target so far (no on-target epoch yet)"
        )
        if args.dynamic:
            print(
                "%s @ Epoch %d, mean_acc=%.4f, mean_tracking_err=%.4f — checkpoint saved\n"
                % (label, epoch, acc, mean_tracking_err)
            )
        else:
            print(
                "%s @ Epoch %d, Accuracy=%.4f, Sparsity=%.4f — checkpoint saved\n"
                % (label, epoch, acc, sparsity)
            )
    else:
        if args.dynamic:
            reason = "" if on_target else " (tracking off-target)"
            print(
                "Epoch %d, mean_acc=%.4f, mean_tracking_err=%.4f (best_acc=%.4f)%s\n"
                % (epoch, acc, mean_tracking_err, best_acc, reason)
            )
        else:
            reason = "" if on_target else " (sparsity off-target)"
            print(
                "Epoch %d, Accuracy=%.4f, Sparsity=%.4f (best=%.4f)%s\n"
                % (epoch, acc, sparsity, best_acc, reason)
            )
