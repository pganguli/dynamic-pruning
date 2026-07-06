#!/usr/bin/env python
"""Stage 3D (dynamic, default) / Stage 3 (static) — backbone-only fine-tune,
gates frozen. See configs/scenario/finetune_dynamic.yaml / finetune_static.yaml.

Usage:
  python scripts/finetune.py
  python scripts/finetune.py --config-name scenario/finetune_static
"""

import hydra

from dynamic_pruning.config import FinetuneConfig, register_configs
from dynamic_pruning.training import finetune

register_configs()


@hydra.main(
    version_base=None, config_path="../configs", config_name="scenario/finetune_dynamic"
)
def main(cfg: FinetuneConfig) -> None:
    finetune.run(cfg)


if __name__ == "__main__":
    main()
