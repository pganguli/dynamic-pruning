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

There are three stages. Run them in order from the project root.

### Stage 1 — Pre-train the baseline model

Trains the backbone without any pruning. The checkpoint from this stage is
the starting point for Stage 2.

```bash
# CIFAR-10 with ResNet-56  (paper settings)
python train_baseline.py --arch resnet56 --dataset cifar10 \
    --epochs 160 --mm 0.9 --wd 1e-4

# HAR (Human Activity Recognition)
python train_baseline.py --arch har_cnn --dataset har \
    --epochs 160 --mm 0.9 --wd 1e-4

# KWS (Keyword Spotting)
python train_baseline.py --arch kws --dataset kws \
    --epochs 160 --mm 0.9 --wd 1e-4
```

Checkpoint saved to `logs/pretrained/<dataset>/<arch>/checkpoint.pth`.

### Stage 2 — Train with dynamic pruning

Loads the Stage 1 checkpoint and jointly trains the decision heads and the
backbone. Decision head weights are updated with Adam; backbone weights with SGD.
The Gumbel-softmax temperature is annealed linearly from τ=5.0 to τ=0.5 over
training, following the paper.

```bash
# CIFAR-10 / ResNet-56 (batch 512, ~400 epochs to match paper's gradient step count)
python main.py --arch resnet56 --dataset cifar10 \
    --sparsity_level 0.4 \
    --gamma 2.2 \
    --action_num 5 \
    --epochs 400

# HAR
python main.py --arch har_cnn --dataset har \
    --sparsity_level 0.5 --gamma 2.2 --epochs 100

# KWS
python main.py --arch kws --dataset kws \
    --sparsity_level 0.5 --gamma 2.2 --epochs 100
```

Checkpoints saved to `logs/decision-<m>/<dataset>-<arch>/sparsity-<r>/checkpoint.pth.tar`.

The log prints after every `--log_interval` batches:

```text
Train Epoch: 5 [200/391]  Loss: 0.4231, Loss_CE: 0.3890, Loss_REG: 0.0341,
                           Sparsity: 0.3921, Mean gate: 0.4102, Accuracy: 0.8750
```

And after each full epoch:

```text
Test set: Loss: 0.3102, Loss_CE: 0.2980, Loss_REG: 0.0122,
          Sparsity: 0.3988, Accuracy: 0.9241
```

**Sparsity** is the fraction of channels active (non-zero) averaged over the test set
— you want this to converge toward your `--sparsity_level`. **Accuracy** is top-1.
The paper reports 92.57 % accuracy at ~52 % MACs reduction on ResNet-56/CIFAR-10
with the settings above.

### Stage 3 (optional) — Fine-tune backbone with gates frozen

Holds the decision heads in deterministic mode and fine-tunes only the backbone
weights to recover any accuracy lost during Stage 2.

```bash
python finetune.py --arch resnet56 --dataset cifar10 \
    --sparsity_level 0.4 --action_num 5 --epochs 160
```

### Stage 4 — Export to ONNX

Produces two ONNX files: a single-input model for on-device inference and a
dynamic-batch model for accuracy evaluation.

```bash
python export.py --arch resnet56 --dataset cifar10 --sparsity_level 0.4
```

Outputs: `<dataset>_<arch>-single.onnx` and `<dataset>_<arch>-batched.onnx`.

## Hyperparameter reference

| CLI flag | Paper symbol | Meaning | Paper value (CIFAR-10/ResNet-56) |
|---|---|---|---|
| `--sparsity_level` | *r* | Target fraction of channels to keep active | 0.4 |
| `--gamma` | *γ* | Regularization balance factor (Eq. 1) | 2.2 |
| `--gamma_under` | — | Fraction of γ applied when sparsity is *below* target (see below) | 0.7 |
| `--action_num` | *m* | Channel-selection masks per decision unit | 5 |
| `--epochs` | — | Training epochs | 400 (Stage 2 @ batch 512), 160 (Stages 1 & 3) |
| `--mm` | — | SGD momentum for backbone optimizer | 0.9 |
| `--wd` | — | Weight decay for backbone optimizer | 1e-4 (Stage 1), 1e-9 (Stage 2) |
| `--train_batch_size` | — | Batch size | 512 (default) |
| `--pruning_threshold` | — | Hard gate threshold at evaluation time | 0.5 |
| `--log_interval` | — | Log every N batches | 100 |

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
| 1 | Default `--action_num` for ResNet | 5 (CIFAR-10), 40 (ImageNet) | 40 if not specified | Always pass `--action_num 5` for CIFAR-10 experiments |
| 2 | Default `--epochs` | 100 | 10 | Always pass `--epochs 100` explicitly |
| 3 | Gate clamping | Not mentioned | Gates clamped to [0, 1] after each Adam step | Forces gate mean to represent fraction of active channels |
| 4 | Supported architectures | VGG16-BN, ResNet-56/50 | ResNet variants, HAR-CNN, KWS-CNN | VGG-family models not available |
