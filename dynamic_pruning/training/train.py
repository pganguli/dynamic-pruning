"""Stage 2 (static) / Stage 2D (dynamic) — joint decision-head + backbone training.

Static mode: a single target keep-fraction `sparsity_level` is used throughout
  training. The model learns to select channels so mean realized density ~= r.

Dynamic mode: a per-batch target keep-fraction r_tgt is sampled uniformly in
  [r_min, r_max] and fed to the conditioned action head, producing a model
  that follows r_tgt at inference time — a single shared-weight model
  covering a range of operating points.

See README.md "Deviations from the paper" for the full rationale behind the
expected-density regularizer, gate diversity anchor, and load-balancing loss
below — each one fixes a specific collapse mode found by earlier iterations.
"""

from collections.abc import Callable

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .. import checkpoints
from ..config import TrainConfig
from ..data import prepare_data
from ..decision import (
    apply_func,
    default_graph,
    normalize_head_weights,
    set_deterministic_value,
    set_pruning_threshold,
)
from ..logging_utils import RunLogger
from .common import (
    default_action_num,
    default_learning_rate,
    initialize_model,
    num_classes,
    transform_model,
)

__all__ = ["run"]

_T_START = (
    5.0  # Gumbel-softmax temperature anneal, Wang et al. 2020 Implementation Details
)
_T_END = 0.5
_SPARSITY_TOL = 0.05  # static mode: sparsity must be within [target - tol, target]
_TRACKING_TOL = 0.05  # dynamic mode: mean |realized - r_tgt| must be below this
_EVAL_GRID_N = 5  # r_tgt points swept each epoch in dynamic mode


def _sample_r_tgt(batch_size: int, cfg: TrainConfig, device: str) -> torch.Tensor:
    """Per-sample r_tgt: uniform in [r_min, r_max], with endpoints oversampled
    at `r_endpoint_prob` so extreme operating points are well trained."""
    r_tgt = torch.empty(batch_size, 1, device=device).uniform_(
        cfg.dynamic_range.r_min, cfg.dynamic_range.r_max
    )
    if cfg.dynamic_range.r_endpoint_prob > 0:
        is_endpoint = (
            torch.rand(batch_size, device=device) < cfg.dynamic_range.r_endpoint_prob
        )
        which_end = torch.rand(batch_size, device=device) < 0.5
        endpoints = torch.where(
            which_end,
            torch.full((batch_size,), cfg.dynamic_range.r_min, device=device),
            torch.full((batch_size,), cfg.dynamic_range.r_max, device=device),
        )
        r_tgt[:, 0] = torch.where(is_endpoint, endpoints, r_tgt[:, 0])
    return r_tgt


def run(
    cfg: TrainConfig, on_epoch_end: Callable[[int, float, float], None] | None = None
) -> float:
    """Run Stage 2/2D training. Returns the best (on-target) accuracy.

    `on_epoch_end(epoch, metric_a, metric_b)` is called after each epoch's
    test sweep — (mean_acc, mean_tracking_err) in dynamic mode, (acc,
    sparsity) in static mode — letting callers (e.g. optuna_search.py) do
    early-stopping/pruning without train.py depending on Optuna directly.
    """
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    action_num = cfg.decision.action_num or default_action_num(cfg.model.arch)
    logdir = checkpoints.decision_dir(
        action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
    )
    log = RunLogger(
        logdir, tensorboard=cfg.tensorboard.enabled, tb_log_dir=cfg.tensorboard.log_dir
    )
    assert isinstance(cfg, DictConfig), "TrainConfig must be converted to DictConfig"
    log.log_config(cfg)

    trainloader, testloader = prepare_data(
        cfg.data.name, cfg.data.train_batch_size, cfg.data.test_batch_size
    )
    n_test = len(testloader.dataset)  # ty: ignore[invalid-argument-type]

    model = initialize_model(cfg.data.name, cfg.model.arch, num_classes(cfg.data.name))
    model_params = list(
        model.parameters()
    )  # backbone only, before decision heads are injected

    log.info("==> Loading pretrained model...")
    model.load_state_dict(
        torch.load(
            checkpoints.pretrain_checkpoint(cfg.data.name, cfg.model.arch),
            weights_only=True,
        )
    )

    transform_model(model, cfg.model.arch, action_num)
    model = model.to(device)

    head_params = default_graph.get_tensor_list("head_params")
    gate_params = default_graph.get_tensor_list("gate_params")

    lr = (
        cfg.optim.lr
        if cfg.optim.lr is not None
        else default_learning_rate(cfg.model.arch)
    )
    optimizer_gate = torch.optim.Adam(head_params + gate_params, lr=lr)
    optimizer_model = torch.optim.SGD(
        model_params,
        lr=lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        nesterov=cfg.optim.nesterov,
    )
    scheduler_gate = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gate, T_max=cfg.epochs, eta_min=lr * 1e-2
    )
    scheduler_model = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_model, T_max=cfg.epochs, eta_min=lr * 1e-2
    )

    def train(epoch: int) -> None:
        model.train()
        apply_func(model, "DecisionHead", set_deterministic_value, deterministic=False)
        for i, (data, target) in enumerate(trainloader):
            default_graph.clear_all_tensors()
            data, target = data.to(device), target.to(device)
            B = data.shape[0]

            if cfg.dynamic:
                r_tgt = _sample_r_tgt(B, cfg, device)
            else:
                r_tgt = torch.full((B, 1), cfg.sparsity_level, device=device)
            default_graph.append_tensor("r_tgt", r_tgt)

            # --- gate / head optimizer step ---
            optimizer_gate.zero_grad()
            output = model(data)
            loss_ce = F.cross_entropy(
                output, target, label_smoothing=cfg.optim.label_smoothing
            )

            selected_channels = default_graph.get_tensor_list("selected_channels")
            concat_channels = torch.cat(selected_channels, dim=1)  # [B, sum_C]

            if cfg.dynamic:
                loss_reg, loss_div, loss_balance = _dynamic_losses(
                    cfg, r_tgt, action_num, device
                )
            else:
                soft = torch.sigmoid(
                    10.0 * (concat_channels - cfg.decision.pruning_threshold)
                )
                soft_sparsity = soft.mean()
                diff = soft_sparsity - cfg.sparsity_level
                sparsity_frac = (
                    (concat_channels > cfg.decision.pruning_threshold).float().mean()
                )
                gamma_eff = (
                    cfg.decision.gamma
                    if sparsity_frac > cfg.sparsity_level
                    else cfg.decision.gamma * cfg.decision.gamma_under
                )
                loss_reg = gamma_eff * diff**2
                loss_div = torch.tensor(0.0, device=device)
                loss_balance = torch.tensor(0.0, device=device)

            loss = loss_ce + loss_reg + loss_div + loss_balance
            loss.backward()
            optimizer_gate.step()

            for p in gate_params:
                p.data.clamp_(0, 1)
            apply_func(model, "DecisionHead", normalize_head_weights)

            # --- backbone optimizer step (CE only) ---
            optimizer_model.zero_grad()
            output = model(data)
            loss_model = F.cross_entropy(
                output, target, label_smoothing=cfg.optim.label_smoothing
            )
            loss_model.backward()
            optimizer_model.step()

            if i % cfg.log_interval == 0:
                sparsity = (
                    (concat_channels > cfg.decision.pruning_threshold)
                    .float()
                    .mean()
                    .item()
                )
                acc = (output.max(1)[1] == target).float().mean().item()
                step = epoch * len(trainloader) + i
                scalars = {
                    "train/loss": loss.item(),
                    "train/loss_ce": loss_ce.item(),
                    "train/loss_reg": loss_reg.item(),
                    "train/sparsity": sparsity,
                    "train/accuracy": acc,
                }
                if cfg.dynamic:
                    scalars["train/loss_div"] = loss_div.item()
                    scalars["train/loss_balance"] = loss_balance.item()
                    scalars["train/r_tgt_mean"] = r_tgt.mean().item()
                    log.info(
                        f"Train Epoch: {epoch} [{i}/{len(trainloader)}]\tLoss: {loss.item():.4f}, "
                        f"Loss_CE: {loss_ce.item():.4f}, Loss_REG: {loss_reg.item():.4f}, "
                        f"Loss_DIV: {loss_div.item():.4f}, Loss_BAL: {loss_balance.item():.4f}, "
                        f"r_tgt_mean: {r_tgt.mean().item():.4f}, Sparsity: {sparsity:.4f}, Accuracy: {acc:.4f}"
                    )
                else:
                    scalars["train/mean_gate"] = concat_channels.mean().item()
                    log.info(
                        f"Train Epoch: {epoch} [{i}/{len(trainloader)}]\tLoss: {loss.item():.4f}, "
                        f"Loss_CE: {loss_ce.item():.4f}, Loss_REG: {loss_reg.item():.4f}, "
                        f"Sparsity: {sparsity:.4f}, Mean gate: {concat_channels.mean().item():.4f}, Accuracy: {acc:.4f}"
                    )
                log.scalars(scalars, step)

    def test(epoch: int) -> tuple[float, float]:
        model.eval()
        apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
        apply_func(
            model,
            "DecisionHead",
            set_pruning_threshold,
            pruning_threshold=cfg.decision.pruning_threshold,
        )

        if cfg.dynamic:
            r_tgt_grid = np.linspace(
                cfg.dynamic_range.r_min, cfg.dynamic_range.r_max, _EVAL_GRID_N
            ).tolist()
            point_results = {}
            for r_val in r_tgt_grid:
                correct, densities = 0, []
                with torch.no_grad():
                    for data, target in testloader:
                        default_graph.clear_all_tensors()
                        r_t = torch.full((data.shape[0], 1), r_val, device=device)
                        default_graph.append_tensor("r_tgt", r_t)
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        cc = torch.cat(
                            default_graph.get_tensor_list("selected_channels"), dim=1
                        )
                        densities.append(
                            (cc > cfg.decision.pruning_threshold).float().mean().item()
                        )
                        correct += (output.max(1)[1] == target).float().sum().item()
                acc = correct / n_test
                realized = float(np.mean(densities))
                point_results[r_val] = (realized, acc)
                log.info(
                    f"  [r_tgt={r_val:.2f}] keep-frac={realized:.4f}, MACs-redux={(1 - realized) * 100:.2f}%, Acc={acc:.4f}"
                )

            mean_acc = float(np.mean([v[1] for v in point_results.values()]))
            mean_tracking_err = float(
                np.mean([abs(v[0] - k) for k, v in point_results.items()])
            )
            log.info(
                f"Test sweep: mean_acc={mean_acc:.4f}, mean_tracking_err={mean_tracking_err:.4f}\n"
            )
            log.scalars(
                {
                    "test/mean_accuracy": mean_acc,
                    "test/mean_tracking_err": mean_tracking_err,
                },
                epoch,
            )
            log.scalar_group(
                "test/keep_frac_by_r_tgt",
                {
                    f"r={r_val:.2f}": realized
                    for r_val, (realized, _acc) in point_results.items()
                },
                epoch,
            )
            log.scalar_group(
                "test/accuracy_by_r_tgt",
                {
                    f"r={r_val:.2f}": acc
                    for r_val, (_realized, acc) in point_results.items()
                },
                epoch,
            )
            return mean_acc, mean_tracking_err

        test_loss_ce, test_sparsity, correct = [], [], 0
        with torch.no_grad():
            for data, target in testloader:
                default_graph.clear_all_tensors()
                r_t = torch.full((data.shape[0], 1), cfg.sparsity_level, device=device)
                default_graph.append_tensor("r_tgt", r_t)
                data, target = data.to(device), target.to(device)
                output = model(data)
                concat_channels = torch.cat(
                    default_graph.get_tensor_list("selected_channels"), dim=1
                )
                test_loss_ce.append(F.cross_entropy(output, target).item())
                test_sparsity.append(
                    (concat_channels > cfg.decision.pruning_threshold)
                    .float()
                    .mean()
                    .item()
                )
                correct += (output.max(1)[1] == target).float().sum().item()

        acc = correct / n_test
        mean_sparsity = float(np.mean(test_sparsity))
        log.info(
            f"Test set: Loss_CE: {np.mean(test_loss_ce):.4f}, Sparsity: {mean_sparsity:.4f}, Accuracy: {acc:.4f}\n"
        )
        log.scalars({"test/accuracy": acc, "test/sparsity": mean_sparsity}, epoch)
        return acc, mean_sparsity

    best_acc, best_metric, ever_on_target = 0.0, float("inf"), False
    for epoch in range(cfg.epochs):
        temperature = _T_START + (_T_END - _T_START) * epoch / max(cfg.epochs - 1, 1)
        default_graph.clear_tensor_list("temperature")
        default_graph.append_tensor("temperature", temperature)

        train(epoch)
        metric_a, metric_b = test(epoch)
        scheduler_gate.step()
        scheduler_model.step()

        if on_epoch_end is not None:
            on_epoch_end(epoch, metric_a, metric_b)

        if cfg.dynamic:
            acc, mean_tracking_err = metric_a, metric_b
            on_target = mean_tracking_err < _TRACKING_TOL
            dist = mean_tracking_err
        else:
            acc, sparsity = metric_a, metric_b
            on_target = (
                (cfg.sparsity_level - _SPARSITY_TOL) <= sparsity <= cfg.sparsity_level
            )
            dist = abs(sparsity - cfg.sparsity_level)

        ever_on_target = ever_on_target or on_target
        should_save = (on_target and acc > best_acc) or (
            not ever_on_target and dist < best_metric
        )

        if should_save:
            if on_target:
                best_acc = acc
            best_metric = dist
            torch.save(
                {"epoch": epoch, "state_dict": model.state_dict()},
                checkpoints.decision_checkpoint(
                    action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
                ),
            )
            label = (
                "New best"
                if on_target
                else "Closest to target so far (no on-target epoch yet)"
            )
            log.info(
                f"{label} @ Epoch {epoch}, acc={acc:.4f}, dist={dist:.4f} — checkpoint saved\n"
            )
        else:
            reason = "" if on_target else " (off-target)"
            log.info(
                f"Epoch {epoch}, acc={acc:.4f}, dist={dist:.4f} (best_acc={best_acc:.4f}){reason}\n"
            )

    log.close()
    return best_acc


def _dynamic_losses(
    cfg: TrainConfig, r_tgt: torch.Tensor, action_num: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expected-density regularizer + gate diversity anchor + load-balancing
    loss (dynamic mode only). See module docstring / README for rationale."""
    action_routing = default_graph.get_tensor_list("action_routing")
    target_densities = torch.linspace(
        cfg.dynamic_range.r_min, cfg.dynamic_range.r_max, action_num, device=device
    )
    per_head_losses, div_losses, balance_losses = [], [], []

    for action_probs, channel_gates in action_routing:
        # Straight-through thresholded density: forward pass is the EXACT
        # hard fraction-above-threshold used by the test-time metric, so the
        # loss optimizes precisely what gets measured/deployed. Backward
        # pass substitutes a soft sigmoid gradient (gentle slope 4) so
        # individual gate values near the boundary still get a pull. A raw
        # mean of gate values doesn't distinguish "uniformly low" from the
        # correct bimodal split and let rows collapse silently — see README.
        hard_gate = (channel_gates > cfg.decision.pruning_threshold).float()
        soft_gate = torch.sigmoid(
            4.0 * (channel_gates - cfg.decision.pruning_threshold)
        )
        gate_indicator = (hard_gate - soft_gate).detach() + soft_gate
        gate_density = gate_indicator.mean(dim=1)  # [action_num]
        expected_density = action_probs @ gate_density  # [B]
        per_head_losses.append((expected_density.unsqueeze(1) - r_tgt) ** 2)

        # Sum (not mean) over actions: averaging dilutes any single action's
        # correction signal by 1/action_num before the head-average dilutes
        # it again by 1/num_heads.
        div_losses.append(((gate_density - target_densities) ** 2).sum())

        hard_choice = action_probs.argmax(dim=1)
        f = F.one_hot(hard_choice, num_classes=action_num).float().mean(dim=0).detach()
        P = action_probs.mean(dim=0)
        balance_losses.append(action_num * (f * P).sum())

    loss_reg = cfg.decision.gamma * torch.cat(per_head_losses, dim=1).mean()
    loss_div = cfg.decision.lambda_div * torch.stack(div_losses).mean()
    loss_balance = cfg.decision.lambda_balance * torch.stack(balance_losses).mean()
    return loss_reg, loss_div, loss_balance
