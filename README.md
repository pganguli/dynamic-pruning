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

- The action head is **target-conditioned**: it concatenates a small learned embedding of
  `r_tgt` with the pooled features before the linear logit layer.
- During training, a different `r_tgt` is sampled **per sample** (uniformly in
  `[r_min, r_max]`), forcing the shared mask menu to differentiate into masks of
  differing densities so the head can map `(image, r_tgt) → appropriate mask`.
- The regularizer becomes a **per-sample symmetric** density loss: `γ · mean((d_k − r_tgt_k)²)`,
  where `d_k` is the realized keep-fraction for sample `k`.

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
    --sparsity_level 0.4 --action_num 16 --r_tgt 0.4
```

Outputs: `cifar10_resnet56-r0.40-single.onnx`, `cifar10_resnet56-r0.40-batched.onnx`

---

### Dynamic track — runtime-adjustable keep-fraction

Use this to train a single model whose compute can be tuned at inference time
by varying `r_tgt ∈ [r_min, r_max]`. Requires Stage 1 to already be done.

#### Stage 2D — Train with dynamic-target pruning

Each training sample receives a different `r_tgt` drawn from `[r_min, r_max]`,
forcing the mask menu to differentiate into masks of varying densities.

```bash
python main.py --arch resnet56 --dataset cifar10 \
    --dynamic --r_min 0.3 --r_max 0.7 \
    --gamma 2.2 --action_num 16 --epochs 400
```

Checkpoint → `logs/decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth.tar`
*(path uses the `--sparsity_level` default of 0.10 as an identifier; override with
`--sparsity_level` if you want a different path key)*

**What to watch:** the per-epoch test sweep across `r_tgt` values:

```text
  [r_tgt=0.30] keep-frac=0.3012, MACs-redux=69.88%, Acc=0.8831
  [r_tgt=0.50] keep-frac=0.4998, MACs-redux=50.02%, Acc=0.9105
  [r_tgt=0.70] keep-frac=0.6991, MACs-redux=30.09%, Acc=0.9217
Test sweep: mean_acc=0.9051, mean_tracking_err=0.0015
```

The key signal is **monotonicity**: realized keep-fraction should rise as
`r_tgt` rises. If it is flat or non-monotone, raise `--gamma`.

#### Stage 3D (optional) — Fine-tune backbone across the range

Fine-tunes the backbone with `r_tgt` still sampled per batch, so accuracy
is recovered across the full operating range (not just one point).

```bash
python finetune.py --arch resnet56 --dataset cifar10 --dynamic \
    --r_min 0.3 --r_max 0.7 --action_num 16 --epochs 160
```

Fine-tuned checkpoint → `logs/finetune-decision-16/cifar10-resnet56/sparsity-0.10/checkpoint.pth`

#### Stage 4D — Calibrate

Sweeps `r_tgt` over a fine grid on the test set and emits the calibration table
(`r_tgt → keep-fraction, MACs-reduction, accuracy`). This table is the offline
artefact needed to build the runtime power→`r_tgt` policy.

```bash
python calibrate.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --r_min 0.3 --r_max 0.7 --grid_n 11 \
    --finetuned --out_csv calibration.csv
```

The script reports monotonicity. If non-monotone points appear, increase
`--gamma` in Stage 2D and retrain.

#### Stage 5D — Export to ONNX

`r_tgt` is **latched** at export time — each ONNX file corresponds to one
operating point. Export once per desired point and use the calibration table
to choose the right file at runtime.

```bash
python export.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.1 --r_tgt 0.30
python export.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.1 --r_tgt 0.50
python export.py --arch resnet56 --dataset cifar10 \
    --action_num 16 --sparsity_level 0.1 --r_tgt 0.70
```

Outputs: `cifar10_resnet56-r0.30-single.onnx`, `cifar10_resnet56-r0.50-single.onnx`, …

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
    --action_num 16 --r_min 0.3 --r_max 0.7 --grid_n 11 \
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
| `--gamma` | *γ* | Regularization strength | 2.2 |
| `--gamma_under` | — | Fraction of γ applied when sparsity is below target (static mode only) | 0.7 |
| `--action_num` | *m* | Channel-selection masks per decision unit | 16 (dynamic), 5 (paper CIFAR) |
| `--epochs` | — | Training epochs | 400 (Stage 2 @ batch 512), 160 (Stages 1, 3) |
| `--mm` | — | SGD momentum for backbone optimizer | 0.9 |
| `--wd` | — | Weight decay for backbone optimizer | 1e-4 (Stage 1), 1e-9 (Stage 2) |
| `--train_batch_size` | — | Batch size | 512 |
| `--pruning_threshold` | — | Hard gate threshold at evaluation time | 0.5 |
| `--log_interval` | — | Log every N batches | 100 |
| `--d_embed` | — | Embedding dimension for r_tgt conditioning | 8 |
| `--dynamic` | — | Enable dynamic-target mode | off (static by default) |
| `--r_min` | — | Lower bound of r_tgt training range | 0.3 |
| `--r_max` | — | Upper bound of r_tgt training range | 0.7 |
| `--r_endpoint_prob` | — | Per-sample probability of oversampling r_min or r_max | 0.1 |
| `--r_tgt` | — | Latched r_tgt value for export (export.py only) | 0.5 |

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
| 6 | Target-conditioned action head | Not in paper | `tgt_embed` (d_embed=8) concatenated to pooled features; `fc1` widened accordingly | Enables runtime `r_tgt` knob; old static checkpoints with `action_num=5` are incompatible — retrain from Stage 2 |
| 7 | Per-sample regularizer (dynamic mode) | Grand-mean Ω over whole batch | Per-sample `γ · mean((d_k − r_tgt_k)²)` using sigmoid proxy | Forces mask menu to span a range of densities; required for the knob to work |
| 8 | `r_tgt` threading | N/A | Via global `TorchGraph` registry (same mechanism as temperature) | Avoids changing model `forward()` signatures; `r_tgt` is latched at export time per operating point |
