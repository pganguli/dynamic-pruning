"""Thin Hydra CLI entrypoints, one per pipeline stage.

Each submodule is a standalone script, run via `python scripts/<name>.py`
from the repo root — importing this package does not eagerly import them,
since each defines an `@hydra.main` entrypoint with config-loading side
effects better triggered explicitly than as an import side effect.
"""

from . import calibrate, export, finetune, pretrain, train, tune

__all__ = ["pretrain", "train", "finetune", "calibrate", "export", "tune"]
