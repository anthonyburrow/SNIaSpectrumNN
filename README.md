# SNIaSpectrumNN

SN Ia spectrum utilities and neural network experimentation.

## PyTorch Rewrite (Experimental)

The new PyTorch implementation lives in `snia_torch/` and introduces a base class `SpectrumModelBase` with:

- Transformer-like encoder backbone (`TransformerEncoderBlock`).
- Pluggable heads via `set_head(head_module)`.
- Minimal training loop (`fit`), evaluation, prediction, and checkpointing.

Concrete reconstruction example: `ReconstructionSpectrumModel` which attaches a `ReconstructionHead` predicting per-token flux.

### Quick Start

```python
from snia_torch import ReconstructionSpectrumModel
import torch

# Fake batch: (batch, seq_len, 2) -> (wave, flux)
batch = torch.rand(4, 512, 2)
model = ReconstructionSpectrumModel(embed_dim=64, num_layers=2)
model.load_checkpoint("recon.pt")  # safe if missing
pred = model.predict(batch)
print(pred.shape)  # (4, 512, 1)
```

### Training Loop (Skeleton)

```python
from torch.utils.data import DataLoader, TensorDataset
import torch

waves_fluxes = torch.rand(64, 512, 2)
target_flux = waves_fluxes[..., 1:].mean(-1, keepdim=True).expand(-1, 512, 1)  # dummy target
ds = TensorDataset(waves_fluxes, target_flux)
loader = DataLoader([{"x": x, "y": y} for x, y in ds], batch_size=8)

model = ReconstructionSpectrumModel()
model.fit(loader, epochs=3)
```

### Adding a New Head

Create a new `nn.Module` that takes encoder features and returns desired output shape, then either:

1. Subclass `SpectrumModelBase` and set the head in `__init__`, or
2. Instantiate `SpectrumModelBase` directly and call `set_head(custom_head)`.

This keeps weight sharing in the backbone while specializing outputs for different tasks (e.g., line feature regression, classification, uncertainty quantification).

### Roadmap

- Data adapter connecting `SNIaSpectrumGen` outputs to PyTorch datasets.
- Additional heads (feature regression, global classification, sequence-to-sequence).
- Mixed precision and distributed training utilities.
- Rich logging (TensorBoard / WandB) integration.

> The existing TensorFlow code will be deprecated once parity is achieved.