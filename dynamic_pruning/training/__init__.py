"""Training loops for every stage of the dynamic-pruning pipeline.

common    architecture-specific defaults, model construction, decision-head transform
pretrain  Stage 1 — pretrain the backbone with no pruning
train     Stage 2 (static) / Stage 2D (dynamic) — joint decision-head + backbone training
finetune  Stage 3 (static) / Stage 3D (dynamic) — backbone-only fine-tune, gates frozen
"""

from . import common, finetune, pretrain, train

__all__ = ["common", "finetune", "pretrain", "train"]
