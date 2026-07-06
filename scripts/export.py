#!/usr/bin/env python
"""Stage 5 — export the (image, r_tgt) -> output model to ONNX.
See configs/scenario/export.yaml.

Usage:
  python scripts/export.py
"""

import hydra

from dynamic_pruning import export
from dynamic_pruning.config import ExportConfig, register_configs

register_configs()


@hydra.main(version_base=None, config_path="../configs", config_name="scenario/export")
def main(cfg: ExportConfig) -> None:
    single_path, batched_path = export.run(cfg)
    print(f"Exported: {single_path}, {batched_path}")


if __name__ == "__main__":
    main()
