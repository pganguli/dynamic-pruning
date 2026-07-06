"""Checkpoint path conventions shared by every training/eval stage.

Centralizing these avoids each script re-deriving (and risking drifting)
the same "logs/<stage>-<action_num>/<dataset>-<arch>/sparsity-<r>/..." layout.
"""

import os

__all__ = [
    "LOGS_ROOT",
    "pretrain_dir",
    "pretrain_checkpoint",
    "decision_dir",
    "decision_checkpoint",
    "finetune_dir",
    "finetune_checkpoint",
]

LOGS_ROOT = "logs"


def pretrain_dir(dataset: str, arch: str) -> str:
    return os.path.join(LOGS_ROOT, "pretrained", dataset, arch)


def pretrain_checkpoint(dataset: str, arch: str) -> str:
    return os.path.join(pretrain_dir(dataset, arch), "checkpoint.pth")


def decision_dir(
    action_num: int, dataset: str, arch: str, sparsity_level: float
) -> str:
    return os.path.join(
        LOGS_ROOT,
        f"decision-{action_num}",
        f"{dataset}-{arch}",
        f"sparsity-{sparsity_level:.2f}",
    )


def decision_checkpoint(
    action_num: int, dataset: str, arch: str, sparsity_level: float
) -> str:
    return os.path.join(
        decision_dir(action_num, dataset, arch, sparsity_level), "checkpoint.pth.tar"
    )


def finetune_dir(
    action_num: int, dataset: str, arch: str, sparsity_level: float
) -> str:
    return os.path.join(
        LOGS_ROOT,
        f"finetune-decision-{action_num}",
        f"{dataset}-{arch}",
        f"sparsity-{sparsity_level:.2f}",
    )


def finetune_checkpoint(
    action_num: int, dataset: str, arch: str, sparsity_level: float
) -> str:
    return os.path.join(
        finetune_dir(action_num, dataset, arch, sparsity_level), "checkpoint.pth"
    )
