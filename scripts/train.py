#!/usr/bin/env python
"""Stage 2D (dynamic, default) / Stage 2 (static) — joint decision-head +
backbone training. See configs/scenario/train_dynamic.yaml / train_static.yaml.

Usage:
  python scripts/train.py
  python scripts/train.py --config-name scenario/train_static
  python scripts/train.py decision.gamma=15 dynamic_range.r_max=0.8
"""

import hydra

from dynamic_pruning.config import TrainConfig, register_configs
from dynamic_pruning.training import train

register_configs()


@hydra.main(
    version_base=None, config_path="../configs", config_name="scenario/train_dynamic"
)
def main(cfg: TrainConfig) -> None:
    train.run(cfg)


if __name__ == "__main__":
    main()
