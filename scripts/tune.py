#!/usr/bin/env python
"""Optuna search over Stage 2D hyperparameters. See configs/scenario/tune.yaml.

Runs on a GPU machine — a full trial budget is still substantial compute
(n_trials x epochs_per_trial short training runs). Narrows the search space
around configs/scenario/train_dynamic.yaml's defaults; always validate the
winning config with a full-length `scripts/train.py` run afterward.

Usage:
  python scripts/tune.py
  python scripts/tune.py n_trials=50 epochs_per_trial=12
"""

import hydra

from dynamic_pruning import optuna_search
from dynamic_pruning.config import OptunaSearchConfig, register_configs

register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="scenario/tune")
def main(cfg: OptunaSearchConfig) -> None:
    optuna_search.run(cfg)


if __name__ == "__main__":
    main()
