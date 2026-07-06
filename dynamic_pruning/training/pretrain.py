"""Stage 1 — pretrain the backbone with no pruning.

Saves a checkpoint to logs/pretrained/<dataset>/<arch>/checkpoint.pth, the
starting point for Stage 2/2D (dynamic_pruning/training/train.py).
"""

import os

import torch
import torch.nn.functional as F

from .. import checkpoints
from ..config import PretrainConfig
from ..data import prepare_data
from ..logging_utils import RunLogger
from .common import default_learning_rate, initialize_model, num_classes

__all__ = ["run"]


def run(cfg: PretrainConfig) -> float:
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    logdir = checkpoints.pretrain_dir(cfg.data.name, cfg.model.arch)
    log = RunLogger(
        logdir, tensorboard=cfg.tensorboard.enabled, tb_log_dir=cfg.tensorboard.log_dir
    )

    trainloader, testloader = prepare_data(
        cfg.data.name, cfg.data.train_batch_size, cfg.data.test_batch_size
    )
    n_test = len(testloader.dataset)  # type: ignore[arg-type]

    model = initialize_model(
        cfg.data.name,
        cfg.model.arch,
        num_classes(cfg.data.name),
        dropout_prob=cfg.model.dropout_prob,
    ).to(device)

    lr = (
        cfg.optim.lr
        if cfg.optim.lr is not None
        else default_learning_rate(cfg.model.arch)
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
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
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(
                output, target, label_smoothing=cfg.optim.label_smoothing
            )
            loss.backward()
            optimizer.step()

            if i % cfg.log_interval == 0:
                acc = (output.max(1)[1] == target).float().mean().item()
                step = epoch * len(trainloader) + i
                log.info(
                    f"Train Epoch: {epoch} [{i}/{len(trainloader)}]\tLoss: {loss.item():.6f}, Accuracy: {acc:.4f}"
                )
                log.scalars({"train/loss": loss.item(), "train/accuracy": acc}, step)

    def test() -> float:
        model.eval()
        test_loss, correct = 0.0, 0
        with torch.no_grad():
            for data, target in testloader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                correct += (output.max(1)[1] == target).float().sum().item()
        test_loss /= n_test
        acc = correct / n_test
        log.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {acc:.4f}\n")
        return acc

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        train(epoch)
        acc = test()
        log.scalars({"test/accuracy": acc}, epoch)
        scheduler.step()

        if acc > best_acc:
            best_acc = acc
            os.makedirs(logdir, exist_ok=True)
            torch.save(
                model.state_dict(),
                checkpoints.pretrain_checkpoint(cfg.data.name, cfg.model.arch),
            )
            log.info(f"  -> New best accuracy {best_acc:.4f}, checkpoint saved.")

    log.info(f"Best saved model test accuracy = {best_acc:.4f}")
    log.close()
    return best_acc
