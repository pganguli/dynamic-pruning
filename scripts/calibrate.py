#!/usr/bin/env python
"""Stage 4D — calibration sweep over r_tgt. See configs/scenario/calibrate.yaml.

Usage:
  python scripts/calibrate.py
  python scripts/calibrate.py grid_n=33 out_csv=calibration_fine.csv
"""

import hydra

from dynamic_pruning import calibrate
from dynamic_pruning.config import CalibrateConfig, register_configs

register_configs()


@hydra.main(
    version_base=None, config_path="../configs", config_name="scenario/calibrate"
)
def main(cfg: CalibrateConfig) -> None:
    calibrate.run(cfg)


if __name__ == "__main__":
    main()
