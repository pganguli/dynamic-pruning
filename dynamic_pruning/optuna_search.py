"""Optuna hyperparameter search for Stage 2D dynamic training.

Runs short (`epochs_per_trial`-epoch) abbreviated dynamic-training trials,
searching gamma / lambda_div / lambda_balance / lr around the validated
defaults in config.py (TrainConfig), and reports a combined tracking-error +
accuracy objective, pruning bad trials early via Optuna's MedianPruner.

Intended to be run on a GPU machine — a full trial budget (e.g. 30 trials x
8 epochs) is still a meaningful chunk of compute; this is a search harness,
not something to run casually on every change. Results narrow the search
space around configs/scenario/train_dynamic.yaml's defaults — they don't
replace validating the winning config with a full-length training run.
"""

from copy import deepcopy

import optuna

from .config import OptunaSearchConfig, TrainConfig
from .training import train as train_stage

__all__ = ["run"]


def _objective(
    trial: optuna.Trial, base_cfg: TrainConfig, epochs_per_trial: int
) -> float:
    cfg: TrainConfig = deepcopy(base_cfg)
    cfg.epochs = epochs_per_trial
    cfg.tensorboard.enabled = False  # avoid one event file per trial
    cfg.decision.gamma = trial.suggest_float("gamma", 1.0, 30.0, log=True)
    cfg.decision.lambda_div = trial.suggest_float("lambda_div", 1.0, 100.0, log=True)
    cfg.decision.lambda_balance = trial.suggest_float(
        "lambda_balance", 0.05, 5.0, log=True
    )
    cfg.optim.lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)

    best = {"mean_acc": 0.0, "mean_tracking_err": 1.0}

    def on_epoch_end(epoch: int, mean_acc: float, mean_tracking_err: float) -> None:
        best["mean_acc"] = mean_acc
        best["mean_tracking_err"] = mean_tracking_err
        # Equal-weight combined score: reward low tracking error AND high
        # accuracy: a model that tracks r_tgt perfectly but can't classify,
        # or classifies well but ignores r_tgt, both score poorly.
        score = mean_tracking_err + (1.0 - mean_acc)
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    train_stage.run(cfg, on_epoch_end=on_epoch_end)
    return best["mean_tracking_err"] + (1.0 - best["mean_acc"])


def run(cfg: OptunaSearchConfig) -> optuna.Study:
    study = optuna.create_study(
        study_name=cfg.study_name,
        storage=cfg.storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=2),
    )
    study.optimize(
        lambda trial: _objective(trial, cfg.train, cfg.epochs_per_trial),
        n_trials=cfg.n_trials,
    )
    print("Best trial:")
    print(f"  value:  {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")
    return study
