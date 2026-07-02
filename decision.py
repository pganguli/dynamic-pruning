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
default_graph.add_tensor_list(
    "head_logit_diag"
)  # non-persistent: (fc1_out, r_proj_out) per DecisionHead, for scale diagnostics
default_graph.add_tensor_list(
    "action_routing"
)  # non-persistent: (action_probs, channel_gates) per DecisionHead, for the
# expected-density regularizer (dynamic mode) — see main.py train()


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
        # Initialize gate densities spread uniformly across [0.1, 0.9] using the
        # logit-inverse mapping: v_k = 0.5 + logit(d_k)/10 gives exactly density
        # d_k under sigmoid(10*(v - 0.5)), and all values land in ~[0.28, 0.72]
        # — well inside the active gradient region. Binary {0, 1} init saturates
        # the sigmoid (gradient ≈ 0), so CE re-collapses gate diversity as soon
        # as Loss_REG → 0; soft init keeps gradients live throughout training.
        target_densities = torch.linspace(0.1, 0.9, action_num)
        gate_values = 0.5 + torch.logit(target_densities) / 10.0  # [action_num]
        gate_init = gate_values.unsqueeze(1).expand(-1, out_channels).clone()
        gate_init = gate_init + 0.01 * torch.randn_like(gate_init)
        self.channel_gates = nn.Parameter(gate_init)
        self.pruning_threshold = pruning_threshold

    def head_params(self):
        return [self.fc1.weight] + list(self.r_proj.parameters())

    def gate_params(self):
        return [self.channel_gates]

    def normalize_weights(self):
        # fc1's input (`out`) is now L2-normalized in forward(), so fc1's
        # output is already bounded to roughly [-1, 1] per action without
        # needing the input-magnitude protection unit-row-normalization was
        # originally for. We keep it anyway for parity with the paper's
        # weight-normalized decision head.
        self.fc1.weight.data = F.normalize(self.fc1.weight.data, dim=1)
        # r_proj is deliberately left unnormalized. r_tgt's input range is
        # narrow ([r_min, r_max], e.g. [0.3, 0.7]), so the weight needs
        # freedom to scale up enough that r_tgt produces a logit spread
        # comparable to fc1's — pinning it to unit norm (whether per-row,
        # which degenerates to sign(), or as a whole block) caps that spread
        # at a few tenths regardless of how much the regularizer wants more
        # differentiation, which was silently capping r_tgt's influence on
        # action selection no matter how high --gamma was pushed. Now that
        # fc1's input is bounded too (see above), CE no longer has an
        # unbounded-magnitude shortcut to outcompete r_proj, so the original
        # concern motivating this floor (CE shrinking r_proj toward
        # irrelevance) is far less likely to dominate.
        pass

    def forward(self, x):
        out = self.avgpool(self.relu(x))
        out = out.view(x.shape[0], x.shape[1])  # [B, C_in]
        # fc1's weight rows are pinned to unit norm, but `out` itself is not
        # bounded — its magnitude grows with the backbone's feature scale
        # over training (observed empirically: ~0.8 -> ~3.7 over 3 epochs).
        # r_proj's output is implicitly capped by its own unit-norm weight
        # constraint, so an unbounded `out` lets fc1's contribution to the
        # logits grow to dominate r_proj's by 5x+ within a few epochs,
        # drowning out the r_tgt signal in softmax regardless of gradient
        # health. Normalizing `out` bounds fc1's output to the same scale
        # as r_proj's, so both terms compete on comparable footing.
        out = F.normalize(out, dim=1)

        # Read per-batch r_tgt from the registry.  Falls back to 0.5 if the
        # training loop has not set one (e.g. export-time tracing with a specific
        # r_tgt latched before the export call).
        r_tgt_list = default_graph._graph.get("r_tgt", [])
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
            temp_list = default_graph._graph["temperature"]
            temp_val: float = float(temp_list[0]) if temp_list else 1.0
            temperature = torch.tensor(temp_val, device=x.device)
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
