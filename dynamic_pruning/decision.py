"""
Dynamic channel-pruning decision heads and gating logic.

TorchGraph: a lightweight named-list registry used to wire intermediate
  activations between backbone layers and decision heads during the forward
  pass.  The `persistence` flag keeps a tensor list across calls (e.g., for
  multi-scale feature aggregation) while non-persistent lists are cleared
  before each forward pass.

DecisionHead: a per-layer soft gate that learns whether to skip (prune) an
  output channel group.  During training it uses the Gumbel-softmax
  (RelaxedOneHotCategorical) to keep the gate differentiable; at export/
  inference time it is replaced by a hard argmax threshold.

  The action head is *target-conditioned*: a per-batch scalar r_tgt (target
  keep-fraction, same units as sparsity_level) is projected through a small
  linear layer (`r_proj`) and added directly onto the feature-driven fc1
  logits, rather than being embedded and concatenated with the pooled
  features. r_tgt is threaded through the global TorchGraph registry (same
  mechanism as temperature), so no model-signature changes are needed. The
  additive form keeps the r_tgt gradient pathway architecturally separate
  from the feature pathway (no shared normalization, no concat dead zones),
  which is needed for the head to learn a real (image, r_tgt) -> mask
  mapping instead of collapsing to an r_tgt-independent average mask.

At export time (see dynamic_pruning/export.py) the model is wrapped so r_tgt
flows through the traced forward() call as a genuine second graph input.

Only two backbone block types are wired up: BasicBlock (ResNet) and ConvBlock
(HAR/KWS) — the only architectures this project actively trains and validates.
"""

import types
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import RelaxedOneHotCategorical

if TYPE_CHECKING:
    from .models.har_cnn import ConvBlock
    from .models.resnet import BasicBlock

__all__ = [
    "TorchGraph",
    "default_graph",
    "DecisionHead",
    "apply_func",
    "replace_func",
    "collect_params",
    "set_deterministic_value",
    "normalize_head_weights",
    "set_pruning_threshold",
    "init_decision_basicblock",
    "decision_basicblock_forward",
    "init_decision_conv_block",
    "decision_conv_block_forward",
]


class TorchGraph:
    def __init__(self) -> None:
        self._graph: dict[str, list] = {}
        self.persistence: dict[str, bool] = {}

    def add_tensor_list(self, name: str, persist: bool = False) -> None:
        self._graph[name] = []
        self.persistence[name] = persist

    def append_tensor(self, name: str, val: object) -> None:
        self._graph[name].append(val)

    def clear_tensor_list(self, name: str) -> None:
        self._graph[name].clear()

    def get_tensor_list(self, name: str) -> list:
        return self._graph[name]

    def clear_all_tensors(self) -> None:
        for k in self._graph:
            if not self.persistence[k]:
                self.clear_tensor_list(k)


default_graph = TorchGraph()
default_graph.add_tensor_list("head_params", persist=True)
default_graph.add_tensor_list("gate_params", persist=True)
default_graph.add_tensor_list("sampled_actions")
default_graph.add_tensor_list("selected_channels")
default_graph.add_tensor_list("temperature", persist=True)
default_graph.add_tensor_list("r_tgt")  # non-persistent: set once per batch
default_graph.add_tensor_list(
    "head_logit_diag"
)  # non-persistent: (fc1_out, r_proj_out) per DecisionHead, for scale diagnostics
default_graph.add_tensor_list(
    "action_routing"
)  # non-persistent: (action_probs, channel_gates) per DecisionHead, for the
# expected-density regularizer (dynamic mode) — see training/train.py


class DecisionHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        action_num: int,
        deterministic: bool = False,
        pruning_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.action_num = action_num
        self.deterministic = deterministic
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, action_num, bias=False)
        # Additive target-conditioning term: projects the scalar r_tgt
        # straight onto the action logits, architecturally separate from
        # the feature pathway. Concatenating an embedded r_tgt into fc1's
        # input (and normalizing the combined row) starved the r_tgt
        # gradient — it never escaped a near-zero-gradient initialization,
        # so the head learned to ignore r_tgt and output an r_tgt-
        # independent average density. This additive path gives r_tgt a
        # full-strength, undiluted gradient into every action logit.
        self.r_proj = nn.Linear(1, action_num, bias=True)
        self.relu = nn.ReLU()
        # Initialize each action's gate row as a genuinely BIMODAL split
        # (not a uniform value) so its hard-threshold-above-0.5 fraction
        # already matches its target density at init. A uniform-value row
        # has the right *mean* for a low target density but the WRONG
        # hard-threshold fraction (0%, since every channel sits below 0.5)
        # — this mismatch between "mean value" and "fraction above
        # threshold" is what let raw-mean-based losses collapse the whole
        # row toward 0 or 1 while reporting a deceptively small loss.
        # Assigning round(d_k * out_channels) channels to ~0.8 and the rest
        # to ~0.2 makes the hard-threshold fraction match d_k from the
        # start, so gradient descent only needs to nudge individual
        # channels across the 0.5 boundary rather than discover
        # bimodality from scratch.
        target_densities = torch.linspace(0.1, 0.9, action_num)
        gate_init = torch.full((action_num, out_channels), 0.2)
        for k in range(action_num):
            n_high = int(round(target_densities[k].item() * out_channels))
            high_idx = torch.randperm(out_channels)[:n_high]
            gate_init[k, high_idx] = 0.8
        gate_init = gate_init + 0.02 * torch.randn_like(gate_init)
        gate_init = gate_init.clamp(0.0, 1.0)
        self.channel_gates = nn.Parameter(gate_init)
        self.pruning_threshold = pruning_threshold

    def head_params(self) -> list[nn.Parameter | torch.Tensor]:
        return [self.fc1.weight] + list(self.r_proj.parameters())

    def gate_params(self) -> list[nn.Parameter]:
        return [self.channel_gates]

    def normalize_weights(self) -> None:
        # fc1's input (`out`) is L2-normalized in forward(), so fc1's output
        # is already bounded to roughly [-1, 1] per action. We still
        # normalize fc1's weight rows for parity with the paper's
        # weight-normalized decision head.
        self.fc1.weight.data = F.normalize(self.fc1.weight.data, dim=1)
        # r_proj is deliberately left unnormalized: r_tgt's input range is
        # narrow ([r_min, r_max]), so the weight needs freedom to scale up
        # enough that r_tgt produces a logit spread comparable to fc1's —
        # pinning it to unit norm caps that spread regardless of how much
        # the regularizer wants more differentiation.

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.avgpool(self.relu(x))
        out = out.view(x.shape[0], x.shape[1])  # [B, C_in]
        # `out`'s magnitude grows with the backbone's feature scale over
        # training; normalizing it bounds fc1's output to the same scale as
        # r_proj's (whose weight is itself unbounded), so both terms
        # compete on comparable footing in the softmax rather than fc1
        # drowning out the r_tgt signal.
        out = F.normalize(out, dim=1)

        # Read per-batch r_tgt from the registry. Falls back to 0.5 if the
        # caller hasn't set one (e.g. a raw forward pass with no training
        # loop or export wrapper managing the registry).
        r_tgt_list = default_graph.get_tensor_list("r_tgt")
        if r_tgt_list:
            r_tgt = r_tgt_list[0]  # [B, 1], already on the right device
        else:
            r_tgt = torch.full((x.shape[0], 1), 0.5, device=x.device)

        fc1_out = self.fc1(out)
        r_proj_out = self.r_proj(r_tgt)
        default_graph.append_tensor(
            "head_logit_diag", (fc1_out.detach(), r_proj_out.detach())
        )
        out = fc1_out + r_proj_out  # [B, action_num]

        action_probs = F.softmax(out, dim=1)
        if self.training:
            # Expose the routing distribution and gate densities (un-detached)
            # so the training loop can build a per-head expected-density loss:
            # E_a~action_probs[density(a)] vs r_tgt. This is a clean,
            # low-variance gradient path directly into action_probs (hence
            # into fc1 + r_proj), unlike the realized/sampled density which
            # only constrains the aggregate outcome across all heads and
            # lets the routing decision itself stay r_tgt-independent.
            default_graph.append_tensor(
                "action_routing", (action_probs, self.channel_gates)
            )

        if self.deterministic or not self.training:
            sampled_actions = action_probs.max(1)[1]
            selected_channels = self.channel_gates[sampled_actions]
        else:
            temp_list = default_graph.get_tensor_list("temperature")
            temp_val = float(temp_list[0]) if temp_list else 1.0
            temperature = torch.tensor(temp_val, device=x.device)
            m = RelaxedOneHotCategorical(temperature, action_probs)
            actions = m.rsample()
            onehot_actions = torch.zeros(actions.size(), device=x.device)
            sampled_actions = actions.max(1)[1]
            onehot_actions.scatter_(1, sampled_actions.unsqueeze(1), 1)
            substitute_actions = (onehot_actions - actions).detach() + actions
            selected_channels = torch.mm(substitute_actions, self.channel_gates)

        if self.pruning_threshold:
            selected_channels[selected_channels < self.pruning_threshold] = 0

        return sampled_actions, selected_channels


def apply_func(
    model: nn.Module, module_type: str, func: Callable[..., None], **kwargs: object
) -> None:
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            func(m, **kwargs)


def replace_func(
    model: nn.Module, module_type: str, func: Callable[..., torch.Tensor]
) -> None:
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            m.forward = types.MethodType(func, m)


def collect_params(m: DecisionHead) -> None:
    for p in m.head_params():
        default_graph.append_tensor("head_params", p)
    for p in m.gate_params():
        default_graph.append_tensor("gate_params", p)


def set_deterministic_value(m: DecisionHead, deterministic: bool) -> None:
    m.deterministic = deterministic


def normalize_head_weights(m: DecisionHead) -> None:
    m.normalize_weights()


def set_pruning_threshold(m: DecisionHead, pruning_threshold: float) -> None:
    m.pruning_threshold = pruning_threshold


def init_decision_basicblock(m: "BasicBlock", action_num: int) -> None:
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_basicblock_forward(self: "BasicBlock", x: torch.Tensor) -> torch.Tensor:
    sampled_actions, selected_channels = self.decision_head(x)

    default_graph.append_tensor("sampled_actions", sampled_actions)
    default_graph.append_tensor("selected_channels", selected_channels)

    out = self.conv1(x)
    out = self.bn1(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = F.relu(out)

    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


def init_decision_conv_block(m: "ConvBlock", action_num: int) -> None:
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_conv_block_forward(self: "ConvBlock", x: torch.Tensor) -> torch.Tensor:
    out = self.conv1(x)

    sampled_actions, selected_channels = self.decision_head(x)

    default_graph.append_tensor("sampled_actions", sampled_actions)
    default_graph.append_tensor("selected_channels", selected_channels)

    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.relu2(out)

    return out
