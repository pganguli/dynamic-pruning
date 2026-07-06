#!/usr/bin/env python
"""Stage 1 — pretrain the backbone. See configs/scenario/pretrain*.yaml.

Usage:
  python scripts/pretrain.py
  python scripts/pretrain.py --config-name scenario/pretrain_har
  python scripts/pretrain.py model.arch=resnet20 epochs=200
"""

import hydra

from dynamic_pruning.config import PretrainConfig, register_configs
from dynamic_pruning.training import pretrain

register_configs()


@hydra.main(
    version_base=None, config_path="../configs", config_name="scenario/pretrain"
)
def main(cfg: PretrainConfig) -> None:
    pretrain.run(cfg)


if __name__ == "__main__":
    main()
