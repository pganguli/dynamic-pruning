"""Stage 3 (static) / Stage 3D (dynamic) — backbone-only fine-tune, gates frozen.

Loads a Stage 2/2D checkpoint and fine-tunes backbone weights with the
pruning gates held in deterministic (argmax) mode. In dynamic mode, r_tgt is
still sampled per batch across [r_min, r_max] so the backbone is fine-tuned
to perform well across the full operating range, not just one point.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from .. import checkpoints
from ..config import FinetuneConfig
from ..data import prepare_data
from ..decision import (
    apply_func,
    default_graph,
    set_deterministic_value,
    set_pruning_threshold,
)
from ..logging_utils import RunLogger
from .common import default_action_num, initialize_model, num_classes, transform_model

__all__ = ["run"]

_EVAL_GRID_N = 5


def run(cfg: FinetuneConfig) -> float:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    action_num = cfg.decision.action_num or default_action_num(cfg.model.arch)
    logdir = checkpoints.finetune_dir(
        action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
    )
    log = RunLogger(
        logdir, tensorboard=cfg.tensorboard.enabled, tb_log_dir=cfg.tensorboard.log_dir
    )
    assert isinstance(cfg, DictConfig), "FineTuneConfig must be converted to DictConfig"
    log.log_config(cfg)

    trainloader, testloader = prepare_data(
        cfg.data.name, cfg.data.train_batch_size, cfg.data.test_batch_size
    )
    n_test = len(testloader.dataset)  # type: ignore[arg-type]

    model = initialize_model(cfg.data.name, cfg.model.arch, num_classes(cfg.data.name))
    model_params = list(model.parameters())
    transform_model(model, cfg.model.arch, action_num)

    log.info("==> Loading pretrained decision model...")
    ckpt = torch.load(
        checkpoints.decision_checkpoint(
            action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
        ),
        weights_only=True,
    )
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)

    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=cfg.decision.pruning_threshold,
    )

    lr = cfg.optim.lr if cfg.optim.lr is not None else 1e-3
    optimizer = torch.optim.SGD(
        model_params,
        lr=lr,
        momentum=cfg.optim.momentum,
        weight_decay=cfg.optim.weight_decay,
        nesterov=cfg.optim.nesterov,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[80, 120], gamma=0.1
    )

    def train(epoch: int) -> None:
        model.train()
        for i, (data, target) in enumerate(trainloader):
            default_graph.clear_all_tensors()
            data, target = data.to(device), target.to(device)
            B = data.shape[0]

            if cfg.dynamic:
                r_tgt = torch.empty(B, 1, device=device).uniform_(
                    cfg.dynamic_range.r_min, cfg.dynamic_range.r_max
                )
            else:
                r_tgt = torch.full((B, 1), cfg.sparsity_level, device=device)
            default_graph.append_tensor("r_tgt", r_tgt)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, label_smoothing=cfg.optim.label_smoothing
            )
            loss.backward()
            optimizer.step()

            if i % cfg.log_interval == 0:
                concat_channels = torch.cat(
                    default_graph.get_tensor_list("selected_channels"), dim=1
                )
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
                    "train/sparsity": sparsity,
                    "train/accuracy": acc,
                }
                if cfg.dynamic:
                    scalars["train/r_tgt_mean"] = r_tgt.mean().item()
                    log.info(
                        f"Train Epoch: {epoch} [{i}/{len(trainloader)}]\tLoss: {loss.item():.4f}, "
                        f"r_tgt_mean: {r_tgt.mean().item():.4f}, Sparsity: {sparsity:.4f}, Accuracy: {acc:.4f}"
                    )
                else:
                    log.info(
                        f"Train Epoch: {epoch} [{i}/{len(trainloader)}]\tLoss: {loss.item():.4f}, "
                        f"Sparsity: {sparsity:.4f}, Accuracy: {acc:.4f}"
                    )
                log.scalars(scalars, step)

    def test(epoch: int):
        model.eval()

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
            return mean_acc, mean_tracking_err

        test_loss, test_sparsity, correct = [], [], 0
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
                test_loss.append(F.cross_entropy(output, target).item())
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
            f"Test set: Loss: {np.mean(test_loss):.4f}, Sparsity: {mean_sparsity:.4f}, Accuracy: {acc:.4f}\n"
        )
        log.scalars({"test/accuracy": acc, "test/sparsity": mean_sparsity}, epoch)
        return acc, mean_sparsity

    best_acc = 0.0
    finetune_ckpt = checkpoints.finetune_checkpoint(
        action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
    )
    for epoch in range(cfg.epochs):
        train(epoch)
        result_a, result_b = test(epoch)
        scheduler.step()

        acc = result_a
        if acc > best_acc:
            best_acc = acc
            os.makedirs(logdir, exist_ok=True)
            torch.save(model.state_dict(), finetune_ckpt)
            if cfg.dynamic:
                log.info(
                    f"New best @ Epoch {epoch}, mean_acc={acc:.4f} — checkpoint saved\n"
                )
            else:
                log.info(
                    f"New best @ Epoch {epoch}, Accuracy={acc:.4f}, Sparsity={result_b:.4f} — checkpoint saved\n"
                )
        else:
            log.info(f"Epoch {epoch}, acc={acc:.4f} (best={best_acc:.4f})\n")

    log.close()
    return best_acc
