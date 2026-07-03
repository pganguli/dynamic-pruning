# dynamic-pruning

Implementation of **Dynamic Network Pruning with Interpretable Layerwise Channel Selection**
(Wang et al., AAAI 2020 — [paper](https://aaai.org/ojs/index.php/AAAI/article/view/6098)).

Based on the original repository at <https://github.com/frankwang345/dynamic-pruning>,
extended to support HAR and KWS datasets alongside CIFAR-10/100.

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

In **dynamic-target mode** (`--dynamic`), the action head also receives a scalar
`r_tgt` (target keep-fraction) alongside the pooled feature vector, enabling a single
offline-trained model to cover a range of operating points at inference time by
varying `r_tgt`. Specifically:

- The action head is **target-conditioned**: a small linear layer (`r_proj`) projects the
  scalar `r_tgt` and adds its output **directly onto** the feature-driven action logits,
  keeping the r_tgt gradient path architecturally separate from the feature path.
- During training, a different `r_tgt` is sampled **per sample** (uniformly in
  `[r_min, r_max]`), forcing the shared mask menu to differentiate into masks of
  differing densities so the head can map `(image, r_tgt) → appropriate mask`.
- The regularizer is a **per-head expected-density loss**: for each decision head
  independently, `γ · mean((E_a[density(a)] − r_tgt)²)`, where the expectation is taken
  over the softmax routing distribution. This is differentiable end-to-end into the
  routing decision without going through the noisy Gumbel sample.

At inference, `r_tgt` is **latched** for the whole inference pass (per-inference
consistency). Sweeping `r_tgt` traces an accuracy-vs-MACs Pareto curve with no retraining.

## Prerequisites

- CUDA-capable GPU (required for all training scripts)
- Python 3.11 or newer

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
pip install -r requirements-kws.txt
git clone https://github.com/ARM-software/ML-KWS-for-MCU dnn-models/ML-KWS-for-MCU
```

## Training workflow

All commands run from the project root. There are two tracks:

- **Static track** — train a model fixed at one keep-fraction (reproduces the paper).
- **Dynamic track** — train a single model that covers a range of keep-fractions at
  inference time, steered by a runtime scalar `r_tgt`.

Both tracks start from the same Stage 1 pretrained backbone.

---

### Stage 1 — Pretrain the backbone (both tracks)

Trains the backbone with no pruning. Takes ~20 min on a modern GPU.

```bash
# CIFAR-10 / ResNet-56
python train_baseline.py --arch resnet56 --dataset cifar10 \
    --epochs 160 --mm 0.9 --wd 1e-4

# HAR (Human Activity Recognition)
python train_baseline.py --arch har_cnn --dataset har \
    --epochs 160 --mm 0.9 --wd 1e-4

# KWS (Keyword Spotting)
python train_baseline.py --arch kws --dataset kws \
    --epochs 160 --mm 0.9 --wd 1e-4
```

Checkpoint → `logs/pretrained/<dataset>/<arch>/checkpoint.pth`

---

### Static track — single operating point

Use this to reproduce the paper results or produce a model fixed at one
keep-fraction (e.g. keep 40 % of channels).

#### Stage 2S — Train with pruning

Jointly trains the decision heads (Adam) and backbone (SGD). The realized
keep-fraction on the test set should converge toward `--sparsity_level`.

```bash
python main.py --arch resnet56 --dataset cifar10 \
    --sparsity_level 0.4 --gamma 2.2 --action_num 16 --epochs 400
```

Checkpoint → `logs/decision-16/cifar10-resnet56/sparsity-0.40/checkpoint.pth.tar`

**What to watch:** the per-epoch test line. `Sparsity` is the fraction of
channels kept active — aim for it to settle near `--sparsity_level`:

```text
Test set: Loss: 0.31, Loss_CE: 0.30, Loss_REG: 0.01,
          Sparsity: 0.3988, Accuracy: 0.9241
```

If sparsity does not converge within ~50 epochs, raise `--gamma`. If
accuracy collapses, reduce `--gamma` or raise `--gamma_under`.

#### Stage 3S (optional) — Fine-tune backbone

Freezes the decision heads and fine-tunes the backbone weights to recover
accuracy. Typically gains 0.5–2 % top-1.

```bash
python finetune.py --arch resnet56 --dataset cifar10 \
    --sparsity_level 0.4 --action_num 16 --epochs 160
```

Fine-tuned checkpoint → `logs/finetune-decision-16/cifar10-resnet56/sparsity-0.40/checkpoint.pth`

#### Stage 4S — Export to ONNX

```bash
python export.py --arch resnet56 --dataset cifar10 \
    --sparsity_level 0.4 --action_num 16
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
python main.py --arch resnet56 --dataset cifar10 \
    --dynamic --r_min 0.1 --r_max 0.9 \
    --gamma 10 --lambda_div 20 --lambda_balance 0.5 --action_num 16 --epochs 100 \
    --train_batch_size 2048
```

Checkpoint → `logs/decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth.tar`
*(path uses the `--sparsity_level` default of 0.10 as an identifier; override with
`--sparsity_level` if you want a different path key)*

**What to watch:** the per-epoch test sweep across `r_tgt` values (5 evenly-spaced
points in `[r_min, r_max]`):

```text
  [r_tgt=0.10] keep-frac=0.0963, MACs-redux=90.37%, Acc=0.7218
  [r_tgt=0.30] keep-frac=0.3205, MACs-redux=67.95%, Acc=0.8631
  [r_tgt=0.50] keep-frac=0.5137, MACs-redux=48.63%, Acc=0.9001
  [r_tgt=0.70] keep-frac=0.6484, MACs-redux=35.16%, Acc=0.9017
  [r_tgt=0.90] keep-frac=0.8650, MACs-redux=13.50%, Acc=0.9251
Test sweep: mean_acc=0.8824, mean_tracking_err=0.0756
```

The key signal is **monotonicity**: realized keep-fraction should rise as `r_tgt` rises.
If it is flat or non-monotone, raise `--gamma`. The best-tracking checkpoint is saved
automatically (often around epoch 2–5); Stage 3D fine-tune then recovers accuracy from it.

#### Stage 3D (optional) — Fine-tune backbone across the range

Fine-tunes the backbone with `r_tgt` still sampled per batch, so accuracy
is recovered across the full operating range (not just one point).

```bash
python finetune.py --arch resnet56 --dataset cifar10 --dynamic \
    --r_min 0.1 --r_max 0.9 --action_num 16 --epochs 160 --sparsity_level 0.1
```

Fine-tuned checkpoint → `logs/finetune-decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth`

#### Stage 4D — Calibrate

Sweeps `r_tgt` over a fine grid on the test set and emits the calibration table
(`r_tgt → keep-fraction, MACs-reduction, accuracy`). This table is the offline
artefact needed to build the runtime power→`r_tgt` policy.

```bash
python calibrate.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --r_min 0.1 --r_max 0.9 --grid_n 17 \
    --finetuned --sparsity_level 0.1 --out_csv calibration.csv
```

The script reports monotonicity. If non-monotone points appear, increase
`--gamma` in Stage 2D and retrain.

#### Stage 5D — Export to ONNX

A **single export** produces a model that takes `r_tgt` as a genuine second
graph input alongside the image — no per-operating-point re-export needed.
`ExportWrapper` (in `export.py`) routes `r_tgt` through the traced `forward()`
call itself (rather than pre-latching it into the `TorchGraph` registry before
tracing), so the ONNX exporter captures it as a real input tensor instead of
baking in a frozen constant. Always export from the fine-tuned checkpoint
(`--finetuned`) for production quality.

```bash
python export.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.1 --finetuned
```

Outputs: `cifar10_resnet56-dynamic-single.onnx`, `cifar10_resnet56-dynamic-batched.onnx`.
At inference time, feed `(image, r_tgt)` to either file — varying `r_tgt` per call
trades accuracy for MACs on the same loaded model, using the calibration table
to pick the operating point for a given power/latency budget.

---

## Evaluation

### During training

Test-set metrics are printed at the end of every epoch automatically. The best
checkpoint is saved whenever a new best accuracy is reached while the realized
keep-fraction (static) or mean tracking error (dynamic) is within tolerance.

### After training — static

Re-evaluate the saved checkpoint at any time by running `calibrate.py` at a
single point:

```bash
python calibrate.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.4 \
    --r_min 0.4 --r_max 0.4 --grid_n 1 --finetuned
```

### After training — dynamic (Pareto curve)

Run the full calibration sweep, then plot the accuracy-vs-MACs-reduction curve:

```bash
python calibrate.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.1 \
    --r_min 0.1 --r_max 0.9 --grid_n 17 \
    --finetuned --out_csv calibration.csv
```

The CSV columns are `r_tgt, keep_frac, macs_redux, accuracy, tracking_err`.
Plot `macs_redux` (x-axis) vs `accuracy` (y-axis) for the Pareto curve.
A well-trained dynamic model should trace a smooth, monotone curve. Compare
against a set of separately-trained static models (one per operating point)
to gauge the accuracy cost of sharing weights across the range.

## Hyperparameter reference

| CLI flag | Paper symbol | Meaning | Default / recommended value |
|---|---|---|---|
| `--sparsity_level` | *r* | Target keep-fraction (static mode) or checkpoint-path key (dynamic mode) | 0.4 |
| `--gamma` | *γ* | Expected-density regularization strength | 10 (dynamic mode), 2.2 (static mode) |
| `--lambda_div` | — | Gate diversity anchor strength (dynamic mode only). Penalises each action's mean gate value deviating from its target density (linspace(r_min, r_max, action_num)), using raw mean (not sigmoid proxy) so gradient is constant and doesn't vanish when CE drives gate values toward 0 or 1 | 20.0 |
| `--lambda_balance` | — | Load-balancing loss weight (dynamic mode only, Switch-Transformer-style). Penalises routing collapse onto a handful of actions: `action_num * sum_k f_k * P_k` where `f_k` is the detached hard-routed fraction and `P_k` is the mean softmax probability per action. Prevents rich-get-richer routing collapse even when gate densities are correctly spread | 0.5 |
| `--gamma_under` | — | Fraction of γ applied when sparsity is below target (static mode only) | 0.7 |
| `--action_num` | *m* | Channel-selection masks per decision unit | 16 (dynamic), 5 (paper CIFAR) |
| `--epochs` | — | Training epochs | 100 (Stage 2D @ batch 2048), 160 (Stages 1, 3) |
| `--mm` | — | SGD momentum for backbone optimizer | 0.9 |
| `--wd` | — | Weight decay for backbone optimizer | 1e-4 (Stage 1), 1e-9 (Stage 2) |
| `--train_batch_size` | — | Batch size | 2048 (Stage 2D dynamic), 512 (others) |
| `--pruning_threshold` | — | Hard gate threshold at evaluation time | 0.5 |
| `--log_interval` | — | Log every N batches | 100 |
| `--dynamic` | — | Enable dynamic-target mode | off (static by default) |
| `--r_min` | — | Lower bound of r_tgt training range | 0.1 |
| `--r_max` | — | Upper bound of r_tgt training range | 0.9 |
| `--r_endpoint_prob` | — | Per-sample probability of oversampling r_min or r_max | 0.1 |

Temperature τ is not a CLI flag — it is annealed automatically from 5.0 to 0.5
linearly over `--epochs`, matching the paper's Implementation Details section.

### Asymmetric sparsity regularizer

The regularization loss is asymmetric around the target sparsity `r`:

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
`--gamma_under 0.0` disables upward pressure entirely (risks gate collapse);
`--gamma_under 1.0` restores a fully symmetric penalty.

## Deviations from the paper

The following are known differences between this implementation and the
paper's described settings. They are minor and unlikely to prevent reproducing
the reported results, but are worth being aware of.

| # | What | Paper | This code | Impact |
|---|---|---|---|---|
| 1 | Default `--action_num` for ResNet | 5 (CIFAR-10), 40 (ImageNet) | 40 if not specified | Always pass `--action_num` explicitly; use 16 for dynamic-target training |
| 2 | Default `--epochs` | 100 | 10 | Always pass `--epochs 100` (static) or `--epochs 400` (dynamic @ batch 512) explicitly |
| 3 | Gate clamping | Not mentioned | Gates clamped to [0, 1] after each Adam step | Forces gate mean to represent fraction of active channels |
| 4 | Regularization strength `--gamma` | 1.0 (paper) | 2.2 (empirically tuned) | Paper value causes slow sparsity convergence with batch 512; increase if sparsity takes many epochs to reach target |
| 5 | Supported architectures | VGG16-BN, ResNet-56/50 | ResNet variants, HAR-CNN, KWS-CNN | VGG-family models not available |
| 6 | Target-conditioned action head | Not in paper | `r_proj` (`Linear(1, action_num)`) added directly onto `fc1`'s logits | Enables runtime `r_tgt` knob; old static checkpoints with `action_num=5` are incompatible — retrain from Stage 2 |
| 7 | Per-head expected-density regularizer (dynamic mode) | Grand-mean Ω over whole batch | Per-head `γ · mean((E_{a~probs}[density(a)] − r_tgt)²)` where `E = action_probs @ gate_density` and `gate_density` is a straight-through hard-threshold fraction (forward = exact `(channel_gates > pruning_threshold).mean(dim=1)`, backward = sigmoid slope 4 gradient) | Differentiable path directly into action probabilities (no Gumbel sample needed); forces each head's routing distribution to track r_tgt independently. Using the *exact* test-time threshold formula (not a raw mean or looser sigmoid proxy) in the forward pass eliminates a train/test mismatch that let gate rows collapse to a uniform low value while reporting a deceptively small loss |
| 8 | `r_tgt` threading (training/eval) | N/A | Via global `TorchGraph` registry (same mechanism as temperature) | Avoids changing model `forward()` signatures during training/eval scripts |
| 12 | `r_tgt` threading (export) | N/A | `ExportWrapper` (export.py) accepts `(x, r_tgt)` and sets the `TorchGraph` registry from *inside* the traced `forward()` call | Setting the registry before tracing (the original approach) bakes whatever value was latched in as an ONNX constant, requiring one export per operating point. Setting it inside the traced call makes the exporter capture `r_tgt` as a genuine second graph input — a single exported model covers the full `[r_min, r_max]` range |
| 9 | Channel-gate initialization | Not specified | Bimodal per-action split: `round(d_k * out_channels)` channels init to ~0.8, rest to ~0.2, with `d_k = linspace(0.1, 0.9, action_num)` | A uniform-value row (all channels at the same value) has the right *mean* density but the wrong *hard-threshold* density (0% or 100%, since every channel is on the same side of 0.5). The bimodal split matches the target density under hard thresholding from the start, so training only needs to nudge individual channels across the boundary rather than discover bimodality from scratch |
| 10 | Gate diversity anchor (dynamic mode) | Not in paper | `λ_div · sum_k((gate_density_k − target_density_k)²)`, summed (not averaged) over actions, then averaged over heads, independent of routing | Without this, actions that routing rarely selects get ~zero gradient from the expected-density loss (∝ action_probs), so CE drifts their gate values toward whatever maximizes accuracy, collapsing the density menu. Summing over actions (rather than averaging) avoids diluting any single action's correction signal by 1/action_num before the head-average dilutes it again |
| 11 | Load-balancing loss (dynamic mode) | Not in paper | Switch-Transformer-style: `λ_balance · action_num · sum_k f_k · P_k`, `f_k` detached hard-routed fraction, `P_k` mean softmax probability | Prevents routing from collapsing onto a handful of the `action_num` actions (rich-get-richer gradient concentration) even when gate densities are correctly spread by the diversity anchor |
