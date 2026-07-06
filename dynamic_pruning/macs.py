"""End-to-end MACs accounting for the dynamic-pruning backbone + decision heads.

The channel gating in decision.py masks conv1's *output* activations by
elementwise multiplication -- it never changes tensor shapes, so a naive
PyTorch eager forward pass runs every conv at full dense cost regardless of
r_tgt. That means naively profiling the executed graph (e.g. with a generic
hook-based FLOPs counter) always reports the *same* MACs whether r_tgt is
0.1 or 0.9, plus a bit extra for the decision heads themselves -- making it
look like dynamic pruning can only ever add overhead, never save anything.

This module instead estimates the MACs an on-device deployment *would*
achieve if channel pruning were structurally realized (skipping the pruned
channels' compute in conv1's output / conv2's input, exactly what the
NodPA C port is expected to do), and separately reports the decision heads'
own compute, which is real overhead paid in full regardless of r_tgt:

  profile_dense_macs()      -- per-module MACs from one dummy forward pass
  measure_block_densities() -- per-DecisionHead realized keep-fraction over
                                a dataloader, at a fixed r_tgt
  macs_report()             -- combines the two into reduction/overhead figures
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decision import DecisionHead, default_graph

__all__ = ["profile_dense_macs", "measure_block_densities", "macs_report"]


def _conv2d_macs(module: nn.Conv2d, out_shape: torch.Size) -> int:
    _, out_c, h, w = out_shape
    kh, kw = module.kernel_size
    in_c_per_group = module.in_channels // module.groups
    return int(out_c * in_c_per_group * kh * kw * h * w)


def _linear_macs(module: nn.Linear) -> int:
    return int(module.in_features * module.out_features)


def profile_dense_macs(model: nn.Module, dummy_input: torch.Tensor) -> dict[str, int]:
    """Per-module dense (full, un-pruned) MACs via a single dummy forward pass.

    Keyed by each Conv2d/Linear submodule's dotted qualified name. `dummy_input`
    must have batch size 1, so the returned values are MACs *per sample*.
    """
    assert dummy_input.shape[0] == 1, "dummy_input must have batch size 1"
    macs: dict[str, int] = {}
    handles = []

    def make_hook(name: str, module: nn.Module):
        def hook(_module, _inp, out):
            if isinstance(module, nn.Conv2d):
                macs[name] = _conv2d_macs(module, out.shape)
            elif isinstance(module, nn.Linear):
                macs[name] = _linear_macs(module)

        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(module.register_forward_hook(make_hook(name, module)))

    was_training = model.training
    model.eval()
    device = dummy_input.device
    default_graph.clear_all_tensors()
    default_graph.append_tensor(
        "r_tgt", torch.full((1, 1), 0.5, device=device)
    )
    with torch.no_grad():
        model(dummy_input)
    default_graph.clear_all_tensors()

    for h in handles:
        h.remove()
    model.train(was_training)
    return macs


def measure_block_densities(
    model: nn.Module,
    dataloader,
    r_tgt: float,
    device: str,
    pruning_threshold: float,
) -> dict[str, float]:
    """Mean realized keep-fraction per DecisionHead, over `dataloader`, at a fixed r_tgt.

    Keyed by the DecisionHead's dotted qualified name (e.g.
    "layers.0.0.decision_head"), matching the naming `macs_report` expects.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inp, output):
            _sampled_actions, selected_channels = output
            density = (selected_channels > pruning_threshold).float().mean().item()
            sums[name] = sums.get(name, 0.0) + density
            counts[name] = counts.get(name, 0) + 1

        return hook

    for name, module in model.named_modules():
        if isinstance(module, DecisionHead):
            handles.append(module.register_forward_hook(make_hook(name)))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        for data, _target in dataloader:
            default_graph.clear_all_tensors()
            data = data.to(device)
            r_t = torch.full((data.shape[0], 1), r_tgt, device=device)
            default_graph.append_tensor("r_tgt", r_t)
            model(data)

    for h in handles:
        h.remove()
    model.train(was_training)
    default_graph.clear_all_tensors()

    return {name: sums[name] / counts[name] for name in sums}


def macs_report(dense_macs: dict[str, int], densities: dict[str, float]) -> dict[str, float]:
    """Combine dense per-module MACs with realized per-block densities.

    Returns:
      dense_total       -- MACs of an equivalent model with no decision heads
                            at all (the "nothing pruned, no gating apparatus"
                            baseline -- e.g. the Stage 1 pretrained model)
      head_overhead     -- MACs spent purely on decision heads (paid in full
                            regardless of r_tgt, whether or not pruning is
                            structurally realized)
      effective_total   -- estimated MACs if channel pruning *were*
                            structurally realized at the measured densities
      reduction_frac    -- 1 - effective_total / dense_total: net MACs saved
                            relative to the no-pruning baseline
      overhead_frac     -- head_overhead / dense_total: cost of the decision
                            heads alone, as a fraction of the baseline
    """
    head_overhead = 0
    gated_conv_dense = 0
    gated_conv_effective = 0.0
    other_dense = 0

    gated_prefixes = {name[: -len(".decision_head")] for name in densities}

    for name, m in dense_macs.items():
        if ".decision_head." in name:
            head_overhead += m
            continue
        prefix = next(
            (
                p
                for p in gated_prefixes
                if name.startswith(p + ".conv1") or name.startswith(p + ".conv2")
            ),
            None,
        )
        if prefix is not None:
            gated_conv_dense += m
            gated_conv_effective += m * densities[prefix + ".decision_head"]
        else:
            other_dense += m

    dense_total = other_dense + gated_conv_dense
    effective_total = other_dense + gated_conv_effective + head_overhead

    return {
        "dense_total": dense_total,
        "head_overhead": head_overhead,
        "effective_total": effective_total,
        "reduction_frac": 1.0 - effective_total / dense_total,
        "overhead_frac": head_overhead / dense_total,
    }
