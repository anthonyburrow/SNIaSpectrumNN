# SNIaSpectrumNN

Neural network models for predicting observational properties of Type Ia
supernovae (SNe Ia) from their optical spectra.

## Overview

This project provides a collection of models that predict various properties of
SNe Ia (e.g., Si II velocity, pseudo-equivalent widths, line strengths) given
only their optical spectrum. Each model shares a common transformer-based
autoencoder backbone that learns a compact representation of the spectrum,
which is then used by task-specific output heads.

### Architecture

- **Base Encoder**: A transformer autoencoder with gated residual network
layers that learns spectral features.
- **Pre-training**: The autoencoder is pre-trained on spectrum reconstruction
to learn meaningful representations.
- **Task-Specific Heads**: After pre-training, the encoder backbone is
fine-tuned with different output heads for specific prediction tasks.

## Installation

Before installing, check the `PYTORCH_INDEX` variable in the `Makefile` and
update it to match your CUDA version. The default is `cu118` (CUDA 11.8).
Change this to the appropriate version for your GPU (e.g., `cu130` for the
most recent version of CUDA that is compatible with more modern GPUs).

Install the project and its dependencies using the Makefile:

```bash
# Create venv if desired
make venv

# Install dependencies
make install
```

This will create a virtual environment (if needed) and install the package with
GPU-enabled PyTorch.

## Usage

### Training Workflow

1. **Pre-train the autoencoder**:
   ```bash
   python scripts/pretrain.py
   ```

2. **Train a specific model** (e.g., Si II velocity prediction):
   ```bash
   python scripts/model_VelocitySiII.py
   ```

Additional model-specific training scripts will be added as the project
develops.

### Testing

Run the test suite:

```bash
make test
```

## Development

The project structure:

- `SNIaSpectrumNN/`: Main package
  - `models/`: Model architectures (base encoder and task-specific heads)
  - `layers/`: Custom neural network layers
  - `data/`: Dataset classes and data loading utilities
  - `util/`: Loss functions and other utilities
- `scripts/`: Training scripts for pre-training and specific models
- `tests/`: Unit tests
