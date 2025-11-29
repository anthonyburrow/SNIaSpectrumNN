import numpy as np
from SNIaSpectrumGen.SpectrumGenerator import SpectrumGenerator
from torch import Tensor

from .SpectrumDatasetBase import SpectrumDatasetBase


class ReconstructionDataset(SpectrumDatasetBase):
    def __init__(
        self,
        steps: int,
        batch_size: int,
        max_len: int,
    ):
        super().__init__(steps, batch_size, max_len)

        self.generator = SpectrumGenerator(
            steps=self.steps,
            length_range=(800, self.max_len),
        )

    def _get_targets(self, x: Tensor, idx: int) -> Tensor:
        return x[:, :, 1:2].clone()

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        batch: list[np.ndarray] = self.generator.generate_batch(
            self.batch_size
        )

        x = self._pad_spectra(batch)
        y = self._get_targets(x, idx)

        return {'x': x, 'y': y}
