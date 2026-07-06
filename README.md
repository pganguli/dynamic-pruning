# dynamic-pruning

Implementation of **Dynamic Network Pruning with Interpretable Layerwise Channel Selection**
(Wang et al., AAAI 2020 — [paper](https://aaai.org/ojs/index.php/AAAI/article/view/6098)).

Based on the original repository at <https://github.com/frankwang345/dynamic-pruning>,
extended to support HAR and KWS datasets alongside CIFAR-10/100, and to support a
**runtime-adjustable target keep-fraction** (`r_tgt`) so a single trained model
trades accuracy for MACs at inference time, instead of needing one model per
operating point.

## How it works

Each convolutional layer gets a lightweight **decision unit** that looks at the incoming
feature map and picks one of *m* learned channel-selection masks **G[i]**. At training
time the selection is differentiable via Gumbel-softmax; at inference time the mask with
the highest probability is chosen deterministically, zeroing out channels before the
convolution so the multiply-accumulate operations can be skipped entirely.

The training objective is (Eq. 1 in the paper):

```text
L = L_cross_entropy + γ · (mean(|selected_channels|) − r)²
```

where **γ** controls how hard to push toward sparsity ratio **r**.

### Dynamic target extension

In **dynamic-target mode** (the default training recipe — see `configs/scenario/train_dynamic.yaml`),
the action head also receives a scalar `r_tgt` (target keep-fraction) alongside the pooled
feature vector, enabling a single offline-trained model to cover a range of operating points
at inference time by varying `r_tgt`. Specifically:

- The action head is **target-conditioned**: a small linear layer (`r_proj`) projects the
  scalar `r_tgt` and adds its output **directly onto** the feature-driven action logits,
  keeping the r_tgt gradient path architecturally separate from the feature path.
- During training, a different `r_tgt` is sampled **per sample** (uniformly in
  `[r_min, r_max]`), forcing the shared mask menu to differentiate into masks of
  differing densities so the head can map `(image, r_tgt) → appropriate mask`.
- The regularizer is a **per-head expected-density loss**: for each decision head
  independently, `γ · mean((E_a[density(a)] − r_tgt)²)`, where the expectation is taken
  over the softmax routing distribution — differentiable end-to-end into the routing
  decision without going through the noisy Gumbel sample.
- A **gate diversity anchor** and a **load-balancing loss** (see "Deviations from the
  paper" below) keep the *m* masks spread across the full density range and prevent
  routing from collapsing onto a handful of them.

At inference, `r_tgt` is a genuine second input to the exported ONNX graph (see Stage 5
below) — sweeping it traces an accuracy-vs-MACs Pareto curve with no retraining and no
re-export.

## Project layout

```text
dynamic_pruning/          importable package
  config.py                 Hydra/dataclass config schema for every stage
  checkpoints.py             logs/... path conventions
  data.py                    dataset loading (CIFAR-10/100, HAR, KWS)
  decision.py                TorchGraph registry + DecisionHead gating logic
  logging_utils.py           console/file/TensorBoard run logging
  calibrate.py               Stage 4D r_tgt calibration sweep
  export.py                  Stage 5 ONNX export (ExportWrapper)
  optuna_search.py           hyperparameter search harness
  models/                    backbone architectures (ResNet, HAR-CNN, KWS-CNN)
  training/
    common.py                 architecture defaults, model construction
    pretrain.py                Stage 1
    train.py                   Stage 2 (static) / Stage 2D (dynamic)
    finetune.py                Stage 3 (static) / Stage 3D (dynamic)

configs/                   Hydra YAML configs
  data/                      per-dataset (cifar10, cifar100, har, kws)
  model/                     per-architecture (resnet56, resnet20, resnet10, har_cnn, kws_cnn)
  scenario/                  per-stage, standalone configs (pretrain, train_dynamic, ...)

scripts/                   thin @hydra.main CLI entrypoints, one per stage
```

Every training run writes text logs to `logs/.../log` and TensorBoard scalars
to `logs/.../tb/` (view with `tensorboard --logdir logs`).

## Prerequisites

- CUDA-capable GPU (required for all training scripts)
- Python 3.11 or newer

Install the project (editable, so `dynamic_pruning` is importable from anywhere
and `scripts/*.py` can be run directly from the repo root):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### HAR dataset setup

The Human Activity Recognition dataset requires two additional setup steps.

**1. Copy the HAR utilities** from the upstream project's `dnn-models/deep-learning-HAR/`
directory into a `dnn-models/` sibling of this project root:

```text
<parent-dir>/
  dynamic-pruning/          ← this repo
  dnn-models/
    deep-learning-HAR/
      utils/
        utilities.py        ← must exist
```

**2. Download the UCI HAR Dataset** and extract it to `~/.cache/UCI HAR Dataset/`
so that the following paths exist:

```text
~/.cache/UCI HAR Dataset/
  train/
    Inertial Signals/
    y_train.txt
  test/
    Inertial Signals/
    y_test.txt
```

Dataset download: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>

### KWS dataset setup

For KWS (keyword spotting) dataset support, install the extra dependencies
in a Python ≤ 3.12 environment (TensorFlow does not yet support Python 3.13+):

```bash
pip install -e ".[kws]"
git clone https://github.com/ARM-software/ML-KWS-for-MCU dnn-models/ML-KWS-for-MCU
```

## Configuration

Every stage is driven by a [Hydra](https://hydra.cc) config composed of three
pieces: a `scenario/*.yaml` (the full recipe for that stage), a `data/*.yaml`
(dataset), and a `model/*.yaml` (architecture). Override anything from the
command line with dotted paths:

```bash
python scripts/train.py decision.gamma=15 dynamic_range.r_max=0.8 epochs=50
python scripts/train.py --config-name scenario/train_static model=resnet20
python scripts/pretrain.py --config-name scenario/pretrain_har
```

Run `python scripts/<name>.py --cfg job` to print the fully-resolved config
without running anything — useful for checking overrides before a long run.

## Training workflow

All commands run from the project root. There are two tracks:

- **Static track** (`scenario/train_static.yaml`) — a model fixed at one keep-fraction
  (reproduces the paper).
- **Dynamic track** (`scenario/train_dynamic.yaml`, the default) — a single model that
  covers a range of keep-fractions at inference time, steered by runtime scalar `r_tgt`.

Both tracks start from the same Stage 1 pretrained backbone.

---

### Stage 1 — Pretrain the backbone (both tracks)

Trains the backbone with no pruning. Takes ~20 min on a modern GPU.

```bash
# CIFAR-10 / ResNet-56 (default)
python scripts/pretrain.py

# HAR (Human Activity Recognition)
python scripts/pretrain.py --config-name scenario/pretrain_har

# KWS (Keyword Spotting)
python scripts/pretrain.py --config-name scenario/pretrain_kws
```

Checkpoint → `logs/pretrained/<dataset>/<arch>/checkpoint.pth`

`optim.label_smoothing=0.1` is on by default (safe, doesn't touch checkpoint
shapes). `model.dropout_prob` defaults to `0.0`; set it (e.g. `0.1`–`0.3`) for a
modest generalization boost — safe here since Stage 1 has no channel-gating
to interact with, and `Dropout2d` has no learnable parameters so it can't
affect checkpoint compatibility with Stage 2/2D onward:

```bash
python scripts/pretrain.py model.dropout_prob=0.1
```

---

### Static track — single operating point

Use this to reproduce the paper results or produce a model fixed at one
keep-fraction (e.g. keep 40% of channels).

#### Stage 2S — Train with pruning

Jointly trains the decision heads (Adam) and backbone (SGD). The realized
keep-fraction on the test set should converge toward `sparsity_level`.

```bash
python scripts/train.py --config-name scenario/train_static
```

Checkpoint → `logs/decision-40/cifar10-resnet56/sparsity-0.40/checkpoint.pth.tar`

**What to watch:** the per-epoch test line. `Sparsity` is the fraction of
channels kept active — aim for it to settle near `sparsity_level`:

```text
Test set: Loss_CE: 0.30, Sparsity: 0.3988, Accuracy: 0.9241
```

If sparsity does not converge within ~50 epochs, raise `decision.gamma`. If
accuracy collapses, reduce `decision.gamma` or raise `decision.gamma_under`.

#### Stage 3S (optional) — Fine-tune backbone

Freezes the decision heads and fine-tunes the backbone weights to recover
accuracy. Typically gains 0.5–2% top-1.

```bash
python scripts/finetune.py --config-name scenario/finetune_static
```

Fine-tuned checkpoint → `logs/finetune-decision-40/cifar10-resnet56/sparsity-0.40/checkpoint.pth`

#### Stage 4S — Export to ONNX

```bash
python scripts/export.py --config-name scenario/export sparsity_level=0.4 decision.action_num=40
```

Outputs: `cifar10_resnet56-dynamic-single.onnx`, `cifar10_resnet56-dynamic-batched.onnx`.
Both take `r_tgt` as a genuine second graph input (see Stage 5D below) even for a
statically-trained checkpoint — varying it won't do anything useful since the head
never saw a range of r_tgt during static training, but the exported graph shape is
the same either way.

---

### Dynamic track — runtime-adjustable keep-fraction

Use this to train a single model whose compute can be tuned at inference time
by varying `r_tgt ∈ [r_min, r_max]`. Requires Stage 1 to already be done.

#### Stage 2D — Train with dynamic-target pruning

Each training sample receives a different `r_tgt` drawn from `[r_min, r_max]`,
forcing the mask menu to differentiate into masks of varying densities.

```bash
python scripts/train.py
```

Checkpoint → `logs/decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth.tar`
*(the path uses `sparsity_level` as an identifier key, not a training target,
in dynamic mode)*

**What to watch:** the per-epoch test sweep across `r_tgt` values (5 evenly-spaced
points in `[r_min, r_max]`), and the same scalars logged to TensorBoard:

```text
  [r_tgt=0.10] keep-frac=0.0963, MACs-redux=90.37%, Acc=0.7218
  [r_tgt=0.30] keep-frac=0.3205, MACs-redux=67.95%, Acc=0.8631
  [r_tgt=0.50] keep-frac=0.5137, MACs-redux=48.63%, Acc=0.9001
  [r_tgt=0.70] keep-frac=0.6484, MACs-redux=35.16%, Acc=0.9017
  [r_tgt=0.90] keep-frac=0.8650, MACs-redux=13.50%, Acc=0.9251
Test sweep: mean_acc=0.8824, mean_tracking_err=0.0756
```

The key signal is **monotonicity**: realized keep-fraction should rise as `r_tgt` rises.
If it is flat or non-monotone, raise `decision.gamma`, `decision.lambda_div`, or
`decision.lambda_balance` (see "Deviations from the paper" for what each one fixes).
The best-tracking checkpoint is saved automatically (often around epoch 2–5);
Stage 3D fine-tune then recovers accuracy from it.

Note: the `MACs-redux` shown during training is the cheap `1 - keep_frac` proxy
(channel-count-weighted, ignores per-layer spatial size and decision-head cost) —
good enough for a live progress readout. For real MACs accounting (accounting for
the decision heads' own compute), use Stage 4D's calibration sweep below.

#### Stage 3D (optional) — Fine-tune backbone across the range

Fine-tunes the backbone with `r_tgt` still sampled per batch, so accuracy
is recovered across the full operating range (not just one point).

```bash
python scripts/finetune.py
```

Fine-tuned checkpoint → `logs/finetune-decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth`

#### Stage 4D — Calibrate

Sweeps `r_tgt` over a fine grid on the test set and emits the calibration table
(`r_tgt → keep-fraction, MACs-reduction, decision-head overhead, accuracy`). This
table is the offline artefact needed to build the runtime power→`r_tgt` policy.

```bash
python scripts/calibrate.py
```

The script reports monotonicity. If non-monotone points appear, increase
`decision.gamma` (or `lambda_div`/`lambda_balance`) in Stage 2D and retrain.

**MACs accounting** (`dynamic_pruning/macs.py`): the channel gating in
`decision.py` masks conv activations by multiplication, so a naive PyTorch
forward pass runs every conv at full dense cost regardless of `r_tgt` — profiling
the executed graph directly would show *zero* savings and only the decision
heads' extra compute. Instead, `calibrate.py` profiles the dense (un-pruned)
per-module MACs once via `profile_dense_macs`, measures each `DecisionHead`'s
realized keep-fraction per r_tgt via forward hooks, and combines them in
`macs_report` to estimate the MACs an on-device deployment would achieve *if*
channel pruning were structurally realized (skipping pruned channels' compute,
as the NodPA C port does) — separately reporting:

- `MACs-redux`: net MACs saved vs. a no-pruning baseline at that operating point
- `head-overhead`: the decision heads' own compute, as a fraction of that
  baseline, paid in full regardless of `r_tgt`

If `head-overhead` ever exceeds `MACs-redux` at some `r_tgt`, the decision heads
are costing more than the pruning saves at that operating point — `calibrate.py`
flags this explicitly.

#### Stage 5D — Export to ONNX

A **single export** produces a model that takes `r_tgt` as a genuine second
graph input alongside the image — no per-operating-point re-export needed.
`ExportWrapper` (in `dynamic_pruning/export.py`) routes `r_tgt` through the
traced `forward()` call itself (rather than pre-latching it into the
`TorchGraph` registry before tracing), so the ONNX exporter captures it as a
real input tensor instead of baking in a frozen constant. Always export from
the fine-tuned checkpoint (`finetuned: true`, the default) for production quality.

```bash
python scripts/export.py
```

Outputs: `cifar10_resnet56-dynamic-single.onnx`, `cifar10_resnet56-dynamic-batched.onnx`.
At inference time, feed `(image, r_tgt)` to either file — varying `r_tgt` per call
trades accuracy for MACs on the same loaded model, using the calibration table
to pick the operating point for a given power/latency budget.

---

## Evaluation

### During training

Test-set metrics are printed at the end of every epoch and logged to
TensorBoard (`logs/.../tb/`). The best checkpoint is saved whenever a new best
accuracy is reached while the realized keep-fraction (static) or mean tracking
error (dynamic) is within tolerance.

### After training — static

Re-evaluate the saved checkpoint at any time with a single-point calibration sweep:

```bash
python scripts/calibrate.py sparsity_level=0.4 decision.action_num=40 \
    dynamic_range.r_min=0.4 dynamic_range.r_max=0.4 grid_n=1
```

### After training — dynamic (Pareto curve)

Run the full calibration sweep (the default `scenario/calibrate.yaml`), then
plot the accuracy-vs-MACs-reduction curve:

```bash
python scripts/calibrate.py
```

The CSV columns are `r_tgt, keep_frac, macs_redux, head_overhead, accuracy, tracking_err`.
Plot `macs_redux` (x-axis) vs `accuracy` (y-axis) for the Pareto curve.
A well-trained dynamic model should trace a smooth, monotone curve. Compare
against a set of separately-trained static models (one per operating point)
to gauge the accuracy cost of sharing weights across the range.

## Hyperparameter search (Optuna)

`scripts/tune.py` runs a short-epoch Optuna study over `decision.gamma`,
`decision.lambda_div`, `decision.lambda_balance`, and `optim.lr`, pruning bad
trials early via `MedianPruner`. It's a search harness, not a casual
per-change tool — a full budget (30 trials × 8 epochs) is still a meaningful
chunk of GPU time.

```bash
python scripts/tune.py
python scripts/tune.py n_trials=50 epochs_per_trial=12
```

Progress persists to `sqlite:///optuna.db` (resumable across runs — rerun the
same command to continue an interrupted study). The objective equally weights
tracking error and `(1 - accuracy)`; always validate the winning trial's
parameters with a full-length `scripts/train.py` run afterward — a short
trial budget narrows the search space around
`configs/scenario/train_dynamic.yaml`'s defaults, it doesn't replace
validating the final recipe.

## Hyperparameter reference

The values below are what `configs/scenario/train_dynamic.yaml` and friends
already encode — this table explains what each one does and why its default
was chosen, for when you need to override it.

| Config field | Paper symbol | Meaning | Default (dynamic / static) |
|---|---|---|---|
| `sparsity_level` | *r* | Target keep-fraction (static mode) or checkpoint-path key (dynamic mode) | 0.1 / 0.4 |
| `decision.gamma` | *γ* | Expected-density regularization strength | 10.0 / 2.2 |
| `decision.lambda_div` | — | Gate diversity anchor strength (dynamic only). Penalises each action's realized density (straight-through hard-threshold fraction) deviating from its target density (`linspace(r_min, r_max, action_num)`) | 20.0 |
| `decision.lambda_balance` | — | Load-balancing loss weight (dynamic only, Switch-Transformer-style). Penalises routing collapse onto a handful of actions | 0.5 |
| `decision.gamma_under` | — | Fraction of γ applied when sparsity is below target (static mode only) | 0.7 |
| `decision.action_num` | *m* | Channel-selection masks per decision unit | 16 (dynamic) / 40 (static, paper CIFAR default) |
| `epochs` | — | Training epochs | 100 (Stage 2D) / 400 (Stage 2S), 160 (Stages 1, 3) |
| `optim.momentum` | — | SGD momentum for backbone optimizer (Nesterov enabled) | 0.9 |
| `optim.weight_decay` | — | Weight decay for backbone optimizer | 1e-4 (Stage 1) / 1e-9 (Stage 2) |
| `optim.label_smoothing` | — | Label smoothing on the training cross-entropy loss | 0.1 (only reshapes CE's target distribution; doesn't touch the reg/div/balance loss terms or checkpoint shapes) |
| `data.train_batch_size` | — | Batch size | 512 (Stage 2/2D/3/3D) / 128 (Stage 1) |
| `decision.pruning_threshold` | — | Hard gate threshold at evaluation time | 0.5 |
| `dynamic` | — | Enable dynamic-target mode | true (default scenario) |
| `dynamic_range.r_min` / `r_max` | — | r_tgt training range | 0.1 / 0.9 |
| `dynamic_range.r_endpoint_prob` | — | Per-sample probability of oversampling r_min or r_max | 0.1 |

Temperature τ is not a config field — it is annealed automatically from 5.0 to
0.5 linearly over `epochs`, matching the paper's Implementation Details section.

### Asymmetric sparsity regularizer (static mode)

The static-mode regularization loss is asymmetric around the target sparsity `r`:

```text
soft_sparsity = sigmoid(10 · (gate − pruning_threshold)) averaged over all gates
sparsity_frac = fraction of gates strictly above pruning_threshold  (non-differentiable)

γ_eff = γ            if sparsity_frac > r   (above target: push down hard)
γ_eff = γ · γ_under  if sparsity_frac ≤ r  (below target: push up gently)

L_reg = γ_eff · (soft_sparsity − r)²
```

`soft_sparsity` is a differentiable sigmoid approximation of `sparsity_frac`.
The sigmoid gradient is largest for gates near `pruning_threshold`, so the
regularizer nudges borderline channels rather than applying uniform pressure
across all gates. The asymmetry decision still uses the hard `sparsity_frac`
to ensure the correct penalty direction even when the soft approximation lags.
When below target, the weaker pressure lets the cross-entropy loss dominate,
so the model uses as many channels as accuracy justifies up to the budget `r`.
`gamma_under: 0.0` disables upward pressure entirely (risks gate collapse);
`gamma_under: 1.0` restores a fully symmetric penalty.

## Deviations from the paper

The following are known differences between this implementation and the
paper's described settings. They are minor and unlikely to prevent reproducing
the reported results, but are worth being aware of.

| # | What | Paper | This code | Impact |
|---|---|---|---|---|
| 1 | Default `action_num` for ResNet (static mode) | 5 (CIFAR-10), 40 (ImageNet) | 40 | Dynamic-target training uses 16 (`configs/scenario/train_dynamic.yaml`) |
| 2 | Gate clamping | Not mentioned | Gates clamped to [0, 1] after each Adam step | Forces gate mean to represent fraction of active channels |
| 3 | Regularization strength `gamma` (static) | 1.0 (paper) | 2.2 (empirically tuned) | Paper value causes slow sparsity convergence with batch 512; increase if sparsity takes many epochs to reach target |
| 4 | Supported architectures | VGG16-BN, ResNet-56/50 | ResNet variants, HAR-CNN, KWS-CNN | VGG-family models were never wired into the dynamic-target head and were removed as dead code |
| 5 | Target-conditioned action head | Not in paper | `r_proj` (`Linear(1, action_num)`) added directly onto `fc1`'s logits | Enables runtime `r_tgt` knob; old static checkpoints with `action_num=5` are incompatible — retrain from Stage 2 |
| 6 | Per-head expected-density regularizer (dynamic mode) | Grand-mean Ω over whole batch | Per-head `γ · mean((E_{a~probs}[density(a)] − r_tgt)²)` where `E = action_probs @ gate_density` and `gate_density` is a straight-through hard-threshold fraction (forward = exact `(channel_gates > pruning_threshold).mean(dim=1)`, backward = sigmoid slope 4 gradient) | Differentiable path directly into action probabilities (no Gumbel sample needed); forces each head's routing distribution to track r_tgt independently. Using the *exact* test-time threshold formula (not a raw mean or looser sigmoid proxy) in the forward pass eliminates a train/test mismatch that let gate rows collapse to a uniform low value while reporting a deceptively small loss |
| 7 | `r_tgt` threading (training/eval) | N/A | Via global `TorchGraph` registry (same mechanism as temperature) | Avoids changing model `forward()` signatures during training/eval scripts |
| 8 | `r_tgt` threading (export) | N/A | `ExportWrapper` (`dynamic_pruning/export.py`) accepts `(x, r_tgt)` and sets the `TorchGraph` registry from *inside* the traced `forward()` call | Setting the registry before tracing (the original approach) bakes whatever value was latched in as an ONNX constant, requiring one export per operating point. Setting it inside the traced call makes the exporter capture `r_tgt` as a genuine second graph input — a single exported model covers the full `[r_min, r_max]` range |
| 9 | Channel-gate initialization | Not specified | Bimodal per-action split: `round(d_k * out_channels)` channels init to ~0.8, rest to ~0.2, with `d_k = linspace(0.1, 0.9, action_num)` | A uniform-value row (all channels at the same value) has the right *mean* density but the wrong *hard-threshold* density (0% or 100%, since every channel is on the same side of 0.5). The bimodal split matches the target density under hard thresholding from the start, so training only needs to nudge individual channels across the boundary rather than discover bimodality from scratch |
| 10 | Gate diversity anchor (dynamic mode) | Not in paper | `λ_div · sum_k((gate_density_k − target_density_k)²)`, summed (not averaged) over actions, then averaged over heads, independent of routing | Without this, actions that routing rarely selects get ~zero gradient from the expected-density loss (∝ action_probs), so CE drifts their gate values toward whatever maximizes accuracy, collapsing the density menu. Summing over actions (rather than averaging) avoids diluting any single action's correction signal by 1/action_num before the head-average dilutes it again |
| 11 | Load-balancing loss (dynamic mode) | Not in paper | Switch-Transformer-style: `λ_balance · action_num · sum_k f_k · P_k`, `f_k` detached hard-routed fraction, `P_k` mean softmax probability | Prevents routing from collapsing onto a handful of the `action_num` actions (rich-get-richer gradient concentration) even when gate densities are correctly spread by the diversity anchor |
| 12 | Backbone optimizer | SGD with momentum | SGD with Nesterov momentum enabled | Free, well-established improvement — same parameters, no checkpoint impact |
| 13 | Spatial dropout (Stage 1 only) | Not in paper | Optional `model.dropout_prob` between the two convs in each `BasicBlock`/`ConvBlock`, default off | Scoped to Stage 1 (no channel-gating yet) since `Dropout2d` has no parameters and the gated forward path (Stage 2D onward) never calls it — safe regardless of what Stage 1 used |
