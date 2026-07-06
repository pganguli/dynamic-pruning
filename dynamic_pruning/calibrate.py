"""Stage 4D — calibration sweep for a dynamic-target pruning model.

Loads a trained checkpoint (Stage 2/2D or Stage 3/3D fine-tune) and sweeps
r_tgt over a grid, recording realized keep-fraction, real MACs reduction,
decision-head overhead, and accuracy at each point. Emits a calibration
table and checks monotonicity of realized density vs r_tgt — the offline
artefact needed to build a runtime power->r_tgt policy.

MACs figures come from dynamic_pruning/macs.py: `reduction` is the estimated
net MACs saved relative to a no-pruning baseline if channel pruning were
structurally realized (as the on-device deployment is expected to do), and
`overhead` is the decision heads' own compute as a fraction of that baseline
-- since they always run in full regardless of r_tgt, high overhead can
erode or exceed the savings.

keep-frac and MACs-redux are reported as mean +/- std *over the test set*
(one value per input image, not per batch) -- the mean alone can hide a
model that tracks r_tgt well on average but swings wildly image to image;
the std tells you how much to trust a single-image MACs estimate at
deployment time.
"""

import csv

import numpy as np
import torch

from . import checkpoints
from .config import CalibrateConfig
from .data import prepare_data
from .decision import (
    DecisionHead,
    apply_func,
    default_graph,
    set_deterministic_value,
    set_pruning_threshold,
)
from .logging_utils import RunLogger
from .macs import macs_report, profile_dense_macs
from .training.common import (
    default_action_num,
    initialize_model,
    num_classes,
    transform_model,
)

__all__ = ["run"]


def run(cfg: CalibrateConfig) -> list[tuple[float, float, float, float, float, float, float, float]]:
    action_num = cfg.decision.action_num or default_action_num(cfg.model.arch)
    logdir = checkpoints.decision_dir(
        action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
    )
    log = RunLogger(logdir, tensorboard=False)

    _, testloader = prepare_data(
        cfg.data.name, cfg.data.train_batch_size, cfg.data.test_batch_size
    )
    n_test = len(testloader.dataset)  # type: ignore[arg-type]

    model = initialize_model(cfg.data.name, cfg.model.arch, num_classes(cfg.data.name))
    transform_model(model, cfg.model.arch, action_num)

    if cfg.finetuned:
        ckpt_path = checkpoints.finetune_checkpoint(
            action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
        )
        log.info(f"==> Loading Stage 3 fine-tuned checkpoint from {ckpt_path} ...")
        model.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
    else:
        ckpt_path = checkpoints.decision_checkpoint(
            action_num, cfg.data.name, cfg.model.arch, cfg.sparsity_level
        )
        log.info(f"==> Loading Stage 2 checkpoint from {ckpt_path} ...")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    apply_func(model, "DecisionHead", set_deterministic_value, deterministic=True)
    apply_func(
        model,
        "DecisionHead",
        set_pruning_threshold,
        pruning_threshold=cfg.decision.pruning_threshold,
    )

    sample_input, _ = next(iter(testloader))
    dense_macs = profile_dense_macs(model, sample_input[:1].to(device))
    log.info(
        f"==> Profiled {len(dense_macs)} Conv2d/Linear modules "
        f"({sum(dense_macs.values()):,} dense MACs/sample, incl. decision heads)"
    )

    r_tgt_grid = np.linspace(
        cfg.dynamic_range.r_min, cfg.dynamic_range.r_max, cfg.grid_n
    ).tolist()

    log.info(
        f"\nCalibration sweep: {cfg.grid_n} points in [{cfg.dynamic_range.r_min:.2f}, {cfg.dynamic_range.r_max:.2f}]"
    )
    log.info("-" * 110)
    log.info(
        "  r_tgt  |   keep-frac (mean+-std)  |  MACs-redux (mean+-std)  | head-overhead | accuracy | tracking-err"
    )
    log.info("-" * 110)

    # Per-DecisionHead realized density, accumulated *per sample* (not just a
    # running mean) alongside the existing accuracy/keep-frac loop below (no
    # extra dataset pass needed) -- needed to report input-to-input std, not
    # just a test-set-wide average.
    density_chunks: dict[str, list[np.ndarray]] = {}

    def make_density_hook(name: str):
        def hook(_module, _inp, output):
            _sampled_actions, selected_channels = output
            d = (
                (selected_channels > cfg.decision.pruning_threshold)
                .float()
                .mean(dim=1)  # per-sample density, not collapsed over the batch
                .cpu()
                .numpy()
            )
            density_chunks.setdefault(name, []).append(d)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, DecisionHead):
            module.register_forward_hook(make_density_hook(name))

    rows = []
    for r_val in r_tgt_grid:
        correct, densities = 0, []
        density_chunks.clear()
        with torch.no_grad():
            for data, target in testloader:
                default_graph.clear_all_tensors()
                r_t = torch.full((data.shape[0], 1), r_val, device=device)
                default_graph.append_tensor("r_tgt", r_t)
                data, target = data.to(device), target.to(device)
                output = model(data)
                cc = torch.cat(
                    default_graph.get_tensor_list("selected_channels"), dim=1
                )
                densities.append(
                    (cc > cfg.decision.pruning_threshold)
                    .float()
                    .mean(dim=1)  # per-sample, not collapsed over the batch
                    .cpu()
                    .numpy()
                )
                correct += (output.max(1)[1] == target).float().sum().item()

        acc = correct / n_test
        per_sample_keep_frac = np.concatenate(densities)
        keep_frac = float(per_sample_keep_frac.mean())
        keep_frac_std = float(per_sample_keep_frac.std())
        tracking_err = abs(keep_frac - r_val)

        block_densities = {
            name: np.concatenate(chunks) for name, chunks in density_chunks.items()
        }
        report = macs_report(dense_macs, block_densities)
        macs_redux = float(np.mean(report["reduction_frac"]))
        macs_redux_std = float(np.std(report["reduction_frac"]))
        head_overhead = float(np.mean(report["overhead_frac"]))

        rows.append(
            (
                r_val,
                keep_frac,
                keep_frac_std,
                macs_redux,
                macs_redux_std,
                head_overhead,
                acc,
                tracking_err,
            )
        )
        log.info(
            f"  {r_val:.4f}  |  {keep_frac:.4f} +- {keep_frac_std:.4f}    "
            f"|  {macs_redux:.4f} +- {macs_redux_std:.4f}    |    {head_overhead:.4f}     "
            f"| {acc:.4f}   |   {tracking_err:.4f}"
        )

    log.info("-" * 110)

    keep_fracs = [r[1] for r in rows]
    non_monotone = [
        i for i in range(1, len(keep_fracs)) if keep_fracs[i] < keep_fracs[i - 1] - 1e-4
    ]
    if non_monotone:
        log.info(
            f"\n⚠  Non-monotone realized density at r_tgt grid points: {non_monotone} "
            "(indices where density decreased). Consider increasing gamma or "
            "widening the training range."
        )
    else:
        log.info("\n✓  Realized density is monotonically non-decreasing with r_tgt.")

    mean_tracking_err = float(np.mean([r[7] for r in rows]))
    log.info(f"Mean tracking error across grid: {mean_tracking_err:.4f}")

    head_overhead_frac = rows[0][5] if rows else 0.0
    log.info(
        f"Decision-head overhead: {head_overhead_frac * 100:.2f}% of no-pruning baseline MACs "
        "(paid in full regardless of r_tgt)"
    )
    worst = min(rows, key=lambda r: r[3]) if rows else None
    if worst is not None and worst[3] < 0:
        log.info(
            f"⚠  At r_tgt={worst[0]:.2f}, decision-head overhead exceeds backbone savings "
            f"(net MACs INCREASE of {-worst[3] * 100:.2f}%)."
        )

    log.info("\n### Calibration table (r_tgt -> operating point)\n")
    log.info(
        "| r_tgt | keep-frac (mean+-std) | MACs-redux (mean+-std) | head-overhead | "
        "accuracy | tracking-err |"
    )
    log.info("|---|---|---|---|---|---|")
    for (
        r_val,
        keep_frac,
        keep_frac_std,
        macs_redux,
        macs_redux_std,
        head_overhead,
        acc,
        tracking_err,
    ) in rows:
        log.info(
            f"| {r_val:.2f} | {keep_frac:.4f} +/- {keep_frac_std:.4f} | "
            f"{macs_redux * 100:.2f}% +/- {macs_redux_std * 100:.2f}% | "
            f"{head_overhead * 100:.2f}% | {acc:.4f} | {tracking_err:.4f} |"
        )

    if cfg.out_csv:
        with open(cfg.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "r_tgt",
                    "keep_frac",
                    "keep_frac_std",
                    "macs_redux",
                    "macs_redux_std",
                    "head_overhead",
                    "accuracy",
                    "tracking_err",
                ]
            )
            for row in rows:
                writer.writerow([f"{v:.4f}" for v in row])
        log.info(f"Calibration table written to {cfg.out_csv}")

    log.close()
    return rows
