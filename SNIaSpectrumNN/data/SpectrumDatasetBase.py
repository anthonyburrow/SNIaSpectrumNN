import numpy as np
import torch
from torch import Tensor


class SpectrumDatasetBase(torch.utils.data.Dataset):
    def __init__(
        self,
        steps: int,
        batch_size: int,
        max_len: int,
    ):
        self.steps = steps
        self.batch_size = batch_size
        self.max_len = max_len

    def _pad_spectra(self, spectra: list[np.ndarray]) -> Tensor:
        shape: tuple = (len(spectra), self.max_len, 2)
        padded = torch.zeros(*shape, dtype=torch.float32)

        for i, spec in enumerate(spectra):
            arr = torch.as_tensor(spec, dtype=torch.float32)
            seq_len = min(arr.size(0), self.max_len)
            padded[i, :seq_len] = arr[:seq_len]

        return padded

    def _get_targets(self, x: Tensor, idx: int) -> Tensor:
        raise NotImplementedError('Subclasses must implement _get_targets()')

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        raise NotImplementedError('Subclasses must implement __getitem__()')
