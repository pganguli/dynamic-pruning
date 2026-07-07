"""Run logging: console + file text log, plus optional TensorBoard scalars.

RunLogger wraps both so every training script writes to the same three
places (stdout, logs/<run>/log, and a TensorBoard event file) through a
single small API, instead of each stage re-deriving its own print/log setup.
"""

import logging
import os
from datetime import datetime

from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter

__all__ = ["RunLogger"]


class RunLogger:
    """Console + file logger, with an optional attached TensorBoard writer."""

    def __init__(
        self, logdir: str, tensorboard: bool = True, tb_log_dir: str | None = None
    ):
        os.makedirs(logdir, exist_ok=True)
        log_file = os.path.join(logdir, "log")
        if os.path.exists(log_file):
            os.remove(log_file)

        self._logger = logging.getLogger(logdir)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        for h in list(self._logger.handlers):
            self._logger.removeHandler(h)
        self._logger.addHandler(logging.FileHandler(log_file))
        self._logger.addHandler(logging.StreamHandler())

        self.writer: SummaryWriter | None = None
        if tensorboard:
            if SummaryWriter is None:
                self.info(
                    "tensorboard package not installed; skipping TensorBoard "
                    "logging (pip install tensorboard to enable it)."
                )
            else:
                self.writer = SummaryWriter(
                    log_dir=tb_log_dir or os.path.join(logdir, "tb")
                )

    def info(self, msg: str) -> None:
        now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._logger.info("[%s] %s", now, msg)

    def log_config(self, cfg: DictConfig) -> None:
        self.info("=================CONFIG==================")
        for line in OmegaConf.to_yaml(cfg).splitlines():
            self.info(line)
        self.info("==========================================")

    def scalar(self, tag: str, value: float, step: int) -> None:
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def scalars(self, tag_value: dict, step: int) -> None:
        for tag, value in tag_value.items():
            self.scalar(tag, value, step)

    def scalar_group(self, main_tag: str, tag_value: dict, step: int) -> None:
        """Log several related scalars as ONE overlaid multi-line chart.

        Unlike `scalar`/`scalars` (each tag gets its own separate chart),
        this uses `SummaryWriter.add_scalars`, which draws every value in
        `tag_value` as a distinct colored line on a single set of axes under
        `main_tag` -- e.g. one chart with 5 lines for keep-frac at each
        r_tgt grid point, instead of 5 separate charts you'd have to eyeball
        side by side.
        """
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
