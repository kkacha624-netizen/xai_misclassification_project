# Source Code Specification

This document summarizes the source-code roles for Experiment 1: a CIFAR-10 ResNet18 baseline classifier.

## Common Settings

- Language: Python
- Framework: PyTorch
- Dataset: CIFAR-10
- Model: ResNet18 adapted for CIFAR-10
- Device: `cuda` when available, otherwise `cpu`
- Random seed: `42` by default
- Checkpoint: `checkpoints/baseline_resnet18.pth`
- Evaluation outputs: `results/`

## `src/train.py`

Trains the CIFAR-10 ResNet18 baseline and saves the checkpoint with the best validation accuracy.

Default arguments:

| Argument | Default | Description |
|---|---:|---|
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `128` | Batch size |
| `--lr` | `0.001` | Adam learning rate |
| `--seed` | `42` | Random seed |
| `--data-dir` | `data` | CIFAR-10 data directory |
| `--checkpoint-dir` | `checkpoints` | Checkpoint directory |
| `--num-workers` | `2` | DataLoader workers |

## `src/evaluate.py`

Loads the trained checkpoint, evaluates it on the CIFAR-10 test split, prints summary metrics, and saves evaluation artifacts.

Outputs:

- Accuracy, macro Precision, macro Recall, macro F1-score
- `results/confusion_matrix.png`
- `results/classification_report.csv`
- misclassified image samples in `results/misclassified/`

## `src/dataset.py`

Defines CIFAR-10 classes, normalization constants, transforms, and DataLoaders.

Training transform:

- random crop with padding
- random horizontal flip
- tensor conversion
- normalization

Test transform:

- tensor conversion
- normalization

## `src/model.py`

Builds a ResNet18 model modified for CIFAR-10:

- first convolution: `3x3`, stride `1`, padding `1`
- initial max-pooling replaced with `nn.Identity()`
- final fully connected layer changed to 10 classes
- no ImageNet pretrained weights

## `src/utils.py`

Contains shared utility functions:

- `set_seed`
- `get_device`
- `ensure_dir`
