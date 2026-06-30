"""
Dynamic channel-pruning decision heads and gating logic.

TorchGraph: a lightweight named-list registry used to wire intermediate
  activations between backbone layers and decision heads during the forward
  pass.  The `persistence` flag keeps a tensor list across calls (e.g., for
  multi-scale feature aggregation) while non-persistent lists are cleared
  before each forward pass.

DecisionUnit: a per-layer soft gate that learns whether to skip (prune) an
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
  which is needed for the head to learn a real (image, r_tgt) → mask
  mapping instead of collapsing to an r_tgt-independent average mask.

At export time (export.py) the model is traced with a latched r_tgt value and the
hard-argmax gate selection is baked into the ONNX graph per operating point.
"""

import torch.nn.functional as F
import torch.nn as nn
import torch
import types
from torch.distributions import RelaxedOneHotCategorical


class TorchGraph(object):
    def __init__(self):
        self._graph = {}
        self.persistence = {}

    def add_tensor_list(self, name, persist=False):
        self._graph[name] = []
        self.persistence[name] = persist

    def append_tensor(self, name, val):
        self._graph[name].append(val)

    def clear_tensor_list(self, name):
        self._graph[name].clear()

    def get_tensor_list(self, name):
        return self._graph[name]

    def clear_all_tensors(self):
        for k in self._graph.keys():
            if not self.persistence[k]:
                self.clear_tensor_list(k)


default_graph = TorchGraph()
default_graph.add_tensor_list("head_params", True)
default_graph.add_tensor_list("gate_params", True)
default_graph.add_tensor_list("sampled_actions")
default_graph.add_tensor_list("selected_channels")
default_graph.add_tensor_list("temperature", True)
default_graph.add_tensor_list(
    "r_tgt"
)  # non-persistent: set once per batch by training loop


class DecisionHead(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        action_num,
        deterministic=False,
        pruning_threshold=0,
    ):
        super(DecisionHead, self).__init__()
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
        self.channel_gates = nn.Parameter(torch.ones(action_num, out_channels))
        self.pruning_threshold = pruning_threshold

    def head_params(self):
        return [self.fc1.weight] + list(self.r_proj.parameters())

    def gate_params(self):
        return [self.channel_gates]

    def normalize_weights(self):
        # r_proj is additive and architecturally separate from fc1, so
        # normalizing fc1's feature weights no longer dilutes the r_tgt
        # conditioning pathway. But fc1's rows are pinned to unit norm
        # every step while r_proj's are not, so without an equivalent
        # floor, gradient pressure from CE (which benefits from ignoring
        # r_tgt) was free to shrink r_proj's contribution toward
        # irrelevance over training. Normalizing both pathways to unit
        # norm guarantees r_tgt a fixed-scale, non-decaying voice in the
        # action logits.
        self.fc1.weight.data = F.normalize(self.fc1.weight.data, dim=1)
        self.r_proj.weight.data = F.normalize(self.r_proj.weight.data, dim=1)

    def forward(self, x):
        out = self.avgpool(self.relu(x))
        out = out.view(x.shape[0], x.shape[1])  # [B, C_in]

        # Read per-batch r_tgt from the registry.  Falls back to 0.5 if the
        # training loop has not set one (e.g. export-time tracing with a specific
        # r_tgt latched before the export call).
        r_tgt_list = default_graph._graph.get("r_tgt", [])
        if r_tgt_list:
            r_tgt = r_tgt_list[0]  # [B, 1], already on the right device
        else:
            r_tgt = torch.full((x.shape[0], 1), 0.5, device=x.device)

        out = self.fc1(out) + self.r_proj(r_tgt)  # [B, action_num]

        action_probs = F.softmax(out, dim=1)
        if self.deterministic or not self.training:
            sampled_actions = action_probs.max(1)[1]
            selected_channels = self.channel_gates[sampled_actions]
        else:
            temp_list = default_graph._graph["temperature"]
            temperature = temp_list[0] if temp_list else 1.0
            m = RelaxedOneHotCategorical(temperature, action_probs)
            actions = m.rsample()
            onehot_actions = torch.zeros(actions.size()).to(x.device)
            sampled_actions = actions.max(1)[1]
            onehot_actions.scatter_(1, sampled_actions.unsqueeze(1), 1)
            substitute_actions = (onehot_actions - actions).detach() + actions
            selected_channels = torch.mm(substitute_actions, self.channel_gates)

        if self.pruning_threshold:
            selected_channels[selected_channels < self.pruning_threshold] = 0

        return sampled_actions, selected_channels


def apply_func(model, module_type, func, **kwargs):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            func(m, **kwargs)


def replace_func(model, module_type, func):
    for m in model.modules():
        if m.__class__.__name__ == module_type:
            m.forward = types.MethodType(func, m)


def collect_params(m):
    for p in m.head_params():
        default_graph.append_tensor("head_params", p)

    for p in m.gate_params():
        default_graph.append_tensor("gate_params", p)


def set_deterministic_value(m, deterministic):
    m.deterministic = deterministic


def normalize_head_weights(m):
    m.normalize_weights()


def set_pruning_threshold(m, pruning_threshold):
    m.pruning_threshold = pruning_threshold


def init_decision_convbn(m, action_num):
    m.decision_head = DecisionHead(m.conv.in_channels, m.conv.out_channels, action_num)


def decision_convbn_forward(self, x):
    out = self.conv(x)
    out = self.bn(out)
    if self.conv.in_channels > 3:
        sampled_actions, selected_channels = self.decision_head(x)

        default_graph.append_tensor("sampled_actions", sampled_actions)
        default_graph.append_tensor("selected_channels", selected_channels)

        out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu(out)
    return out


def init_decision_conv_block(m, action_num):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_conv_block_forward(self, x):
    out = self.conv1(x)

    sampled_actions, selected_channels = self.decision_head(x)

    default_graph.append_tensor("sampled_actions", sampled_actions)
    default_graph.append_tensor("selected_channels", selected_channels)

    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.relu2(out)

    return out


def init_decision_basicblock(m, action_num):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_basicblock_forward(self, x):
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


def init_decision_bottleneck(m, action_num):
    m.decision_head = DecisionHead(
        m.conv1.in_channels, m.conv1.out_channels, action_num
    )


def decision_bottleneck_forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)

    sampled_actions, selected_channels = self.decision_head(out)

    default_graph.append_tensor("sampled_actions", sampled_actions)
    default_graph.append_tensor("selected_channels", selected_channels)

    out = self.conv1(out)

    out = self.bn2(out)
    out = selected_channels.unsqueeze(2).unsqueeze(3) * out
    out = self.relu(out)
    out = self.conv2(out)

    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)

    if self.downsample is not None:
        residual = self.downsample(x)

    out += residual

    return out
