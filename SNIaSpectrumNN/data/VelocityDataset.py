import h5py
import torch
from torch import Tensor
from typing import Dict
from pathlib import Path

from .SpectrumDatasetBase import SpectrumDatasetBase


class VelocityDataset(SpectrumDatasetBase):

    def __init__(
        self,
        data_file: Path | str,
        max_len: int,
        batch_size: int = 32,
    ):
        self.data_file = Path(data_file)
        if not self.data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load data dimensions
        with h5py.File(self.data_file, 'r') as f:
            self.n_samples = f['spectra'].shape[0]
            # Expected structure:
            # f['spectra']: (n_samples, seq_len, 2) - wavelength, flux pairs
            # f['velocities']: (n_samples,) or (n_samples, 1) - velocity values
        
        steps = (self.n_samples + batch_size - 1) // batch_size
        super().__init__(steps=steps, batch_size=batch_size, max_len=max_len)

    def _get_targets(self, x: Tensor, idx: int) -> Tensor:
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        with h5py.File(self.data_file, 'r') as f:
            velocities = f['velocities'][start_idx:end_idx]
            velocities = torch.as_tensor(velocities, dtype=torch.float32)
            
            # Ensure shape is (batch_size, 1)
            if velocities.ndim == 1:
                velocities = velocities.unsqueeze(-1)
        
        return velocities

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        with h5py.File(self.data_file, 'r') as f:
            # Load spectra
            spectra = f['spectra'][start_idx:end_idx]
            spectra_list = [spec for spec in spectra]
        
        x = self._pad_spectra(spectra_list)
        y = self._get_targets(x, idx)
        
        return {"x": x, "y": y}
